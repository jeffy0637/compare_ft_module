# -*- coding: utf-8 -*-
"""
PrecisionSync 離線嵌入評估 — rev-X8
=================================
目標
----
1. **挑選最適特徵模型**：比較多模態嵌入 (CLIP / SigLIP / Fashion-CLIP …)，
   為後續建置混合協同個人化時尚推薦系統，找出最能表徵豐富服飾上下文的表徵模型。
2. **完整上下文呈現**：同時評估多種影像前處理
   （crop224 / 白底 pad_white / resize256）以及
   200 筆隨機樣本 (單品 + 套裝)，輸出 hit@1 / hit@5 / MRR / mean_rank …。
3. 商品資料僅含 **中文品名** 與 **商品平拍圖 URL**，故實驗應於**翻譯後**有效運用此兩項資料。

注意事項說明
----
| #  | 關鍵事項                   | 重點說明                                                                                                                                                                     |
| -- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **資料假設**               | *CSV* **必含** `sku`、`pic_url`、`pname` 欄位；<br>任何前置 ETL 都不得移除或更名此三欄。                                                                                                                |
| 2  | **Single / Outfit 分類** | 以 **SKU 是否含「+」** 判斷：<br>`+` → *outfit*；無 `+` → *single*。<br>此規則直接影響分組指標，**嚴禁改動**。                                                                                                |
| 3  | **中文品名清理**             | 處理流程：<br>`原字串 → 去括號/特殊符號 → 取第一段(遇 '-') → strip → fallback 'unknown'`。<br>若要增修 Regex，**確保不會誤刪關鍵描述**。                                                                              |
| 4  | **翻譯切換**               | 透過 `Cfg.use_translation` 控制；預設啟用 Baidu engine，並在結果中把 **wear → outfit**。                                                                   |
| 5  | **模型包裝 ─ 文字長度**        | OpenCLIP / SigLIP 等家族多採固定 `context_length`（77/256…）。**須確保填補至固定長度**，否則會觸發 positional embedding 維度錯誤。|
| 6  | **GPU 批次自適應**          | `batch = (VRAM_GiB // 8) * base`，保護 8 GB VRAM 卡免於 OOM。                                                                                                     |
"""

# ── 內建 ───────────────────────────────────
from __future__ import annotations
import argparse, asyncio, dataclasses, gc, io, logging, math, os, random, re, sys, warnings
from pathlib import Path
from statistics import median
from typing import List, Sequence
import functools

# ── 第三方 ─────────────────────────────────
import aiohttp, numpy as np, pandas as pd, torch
from PIL import Image, ImageFile
from tqdm.asyncio import tqdm
from torchvision import transforms as T
from transformers import AutoModel, AutoProcessor

# Fashion-CLIP 官方封裝（若未安裝則使用通用路徑）
try:
    from fashion_clip.fashion_clip import FashionCLIP
except ImportError:  # pragma: no cover
    FashionCLIP = None

try:
    import translators as ts
except ImportError:
    ts = None

# DataFrame 美觀化（選用）
try:
    from tabulate import tabulate
    _fmt = lambda df: tabulate(df, headers="keys", tablefmt="github", showindex=False)
except ImportError:  # pragma: no cover
    _fmt = lambda df: df.to_string(index=False)

# Pillow 容忍截斷檔
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="`resume_download`")

# ── 全域組態 ───────────────────────────────
@dataclasses.dataclass(slots=True)
class Cfg:
    """集中管理所有可調參數；利於 CLI / Notebook 共用。"""
    csv: Path = Path("data/all_products.csv")      # 資料來源
    num_single: int = 200                          # 單品抽樣數
    num_outfit: int = 200                          # 套裝抽樣數
    seed: int = 42                                 # 隨機種子
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp16"                        # cuda→fp16；cpu→fp32
    dl_workers: int = 32                           # 非同步下載並行度
    preproc_workers: int = os.cpu_count() or 8     # CPU 前處理並行度
    batch_per_8gb: int = 64                        # 每 8 GiB 可處理張數
    use_translation: bool = True                   # 是否啟用 Baidu 翻譯
    preproc_modes: tuple[str, ...] = ("crop224", "pad_white", "resize256")
    models: tuple[str, ...] = (
        "Marqo/marqo-fashionCLIP",
        "patrickjohncyh/fashion-clip",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch16",
        "Marqo/marqo-fashionSigLIP",
        "google/siglip-base-patch16-256",
    )
    verify_n: int = 3                              # 顯示翻譯對照筆數

# ── 共用輔助 ───────────────────────────────
def set_seed(seed: int) -> None:
    """鎖定所有隨機種子，確保結果可重現。"""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# 影像前處理 ----------------------------------------------------------
_CROP224   = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224)])
_RESIZE256 = T.Resize((256, 256), antialias=True)
def _pad_white(img: Image.Image) -> Image.Image:
    """以白底置中，縮放至 256×256。"""
    w, h = img.size
    side   = max(w, h)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((256, 256), Image.BICUBIC)

_PREPROC = {"crop224": _CROP224, "resize256": _RESIZE256, "pad_white": _pad_white}
def preprocess(img: Image.Image, mode: str) -> Image.Image:
    """依指定模式前處理影像。"""
    return _PREPROC[mode](img)

# 非同步下載 ----------------------------------------------------------
async def _download(sess: aiohttp.ClientSession, url: str, retry: int = 3) -> Image.Image | None:
    """下載單張圖片；失敗重試至多 `retry` 次，失敗回傳 None。"""
    backoff = .5
    for _ in range(retry):
        try:
            async with sess.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status == 200:
                    return Image.open(io.BytesIO(await r.read())).convert("RGB")
        except Exception:
            await asyncio.sleep(backoff); backoff *= 2
    return None

async def download_all(urls: list[str], cfg: Cfg) -> list[Image.Image | None]:
    """批次非同步下載圖片。"""
    imgs: list[Image.Image | None] = [None] * len(urls)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=cfg.dl_workers)
    ) as sess:
        tasks = [_download(sess, u) for u in urls]
        for idx, coro in tqdm(enumerate(asyncio.as_completed(tasks)),
                              total=len(tasks), desc="Download"):
            imgs[idx] = await coro
    return imgs

# 翻譯模組 ------------------------------------------------------------
class Translator:
    """簡易封裝百度翻譯並加上後處理。"""
    def __init__(self, cfg: Cfg): self.cfg = cfg

    @functools.lru_cache(maxsize=None)
    def _tr(self, zh: str) -> str:
        if not ts or not self.cfg.use_translation:
            return zh
        try:
            return ts.translate_text(
                zh, translator="baidu",
                from_language="zh", to_language="en"
            ).replace("wear", "outfit")  # 見注意事項 #4
        except Exception:
            return zh

    def zh2en(self, lst: list[str], tag: str) -> list[str]:
        out = [self._tr(t) for t in tqdm(lst, desc=f"Trans[{tag}]", mininterval=.5)]
        for zh, en in list(zip(lst, out))[:self.cfg.verify_n]:
            logging.info("「%s」  \"%s\"", zh, en)
        return out

# ── 模型包裝 ---------------------------------------------------------
class BaseClipWrapper:
    """
    通用 CLIP / SigLIP / open_clip 包裝器  
    *自動偵測* `context_length`；保證 token 長度符合模型期望，修復
    「The size of tensor a (x) must match ... (77)」錯誤。
    """
    def __init__(self, name: str, cfg: Cfg):
        self.name, self.cfg = name, cfg
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self.model     = AutoModel.from_pretrained(name, trust_remote_code=True)

        # 1️⃣ 偵測 context_length（優先讀 model→processor→fallback 77）
        self.context_length: int = (
            getattr(self.model, "context_length", 0)
            or getattr(getattr(self.processor, "tokenizer", None), "model_max_length", 0)
            or 77
        )

        # 2️⃣ 裝置 / 精度設定
        self.dtype   = torch.float16 if cfg.device.startswith("cuda") else torch.float32
        self.device  = cfg.device
        self.model.to(self.device, dtype=self.dtype).eval()

        # 3️⃣ 依 VRAM 計算 batch（最少 1）
        if self.device.startswith("cuda"):
            vram   = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
            self.batch = max(int(vram // 8) * cfg.batch_per_8gb, 1)
        else:
            self.batch = 16  # CPU 預設

    # ---------- 特徵擷取 ----------
    @torch.inference_mode()
    def encode_text(self, txts: Sequence[str]) -> torch.Tensor:
        feats = []
        for i in range(0, len(txts), self.batch):
            tok = self.processor(
                text=txts[i:i + self.batch],
                padding="max_length",                # 必須 pad 至固定長度
                max_length=self.context_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            # open-clip/clip 模型 API 有差異 → 動態選擇
            sub = (
                self.model.get_text_features(**tok)
                if hasattr(self.model, "get_text_features")
                else self.model.encode_text(tok["input_ids"])
            )
            feats.append(sub)
        out = torch.cat(feats).float()
        return out / out.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def encode_image(self, imgs: List[Image.Image]) -> torch.Tensor:
        feats = []
        for i in range(0, len(imgs), self.batch):
            pix = self.processor(images=imgs[i:i + self.batch],
                                 return_tensors="pt")["pixel_values"].to(
                                     self.device, dtype=self.dtype)
            sub = (
                self.model.get_image_features(pix)
                if hasattr(self.model, "get_image_features")
                else self.model.encode_image(pix)
            )
            feats.append(sub)
        out = torch.cat(feats).float()
        return out / out.norm(dim=-1, keepdim=True)

    # ---------- 相似度 ----------
    @torch.inference_mode()
    def similarity(self, txts: Sequence[str], imgs: List[Image.Image]) -> torch.Tensor:
        t = self.encode_text(txts)
        v = self.encode_image(imgs)
        return (t @ v.T).cpu() * 100  # cosine * 100 一致化

    # ---------- with 語法糖 ----------
    def __enter__(self): return self
    def __exit__(self, *_):
        del self.model; torch.cuda.empty_cache(); gc.collect()

class FashionClipWrapper(BaseClipWrapper):
    """官方 Fashion-CLIP 封裝，優先使用原生推論 API；失敗降級至 BaseClipWrapper。"""
    def __init__(self, name: str, cfg: Cfg):
        if FashionCLIP is None:                     # 未安裝 → 使用通用包裝
            super().__init__(name, cfg); self._fclip = None
        else:
            self.cfg   = cfg
            self._fclip = FashionCLIP("fashion-clip", device=cfg.device)
            # 直接取得 batch 公用設定
            if cfg.device.startswith("cuda"):
                vram        = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
                self.batch  = max(int(vram // 8) * cfg.batch_per_8gb, 1)
            else:
                self.batch  = 16

    @torch.inference_mode()
    def encode_text(self, txts: Sequence[str]) -> torch.Tensor:
        if self._fclip:
            arr = self._fclip.encode_text(list(txts), batch_size=self.batch)
            arr /= np.linalg.norm(arr, axis=-1, keepdims=True)
            return torch.from_numpy(arr)
        return super().encode_text(txts)

    @torch.inference_mode()
    def encode_image(self, imgs: List[Image.Image]) -> torch.Tensor:
        if self._fclip:
            arr = self._fclip.encode_images(imgs, batch_size=self.batch)
            arr /= np.linalg.norm(arr, axis=-1, keepdims=True)
            return torch.from_numpy(arr)
        return super().encode_image(imgs)

# ── 指標計算 ---------------------------------------------------------
def _stats(r: torch.Tensor) -> dict[str, float]:
    """計算一組排名向量的常用指標。"""
    return {
        "hit@1":        (r == 0).float().mean().item(),
        "hit@5":        (r < 5).float().mean().item(),
        "r@10":         (r < 10).float().mean().item(),
        "mean_rank":    r.mean().item() + 1,
        "median_rank":  median(r.tolist()) + 1,
        "mrr":          (1 / (r + 1)).mean().item(),
    }

def calc_metrics(mat: torch.Tensor, single_mask: torch.Tensor) -> dict[str, float]:
    """
    由完整相似度矩陣計算 *all / single / outfit* 三組指標  
    `single_mask=True` 表示 single；False → outfit。
    """
    idx  = torch.arange(mat.size(0))
    # ↑ 每列排序降冪後，找出對應真實圖片之排名
    rank = (mat.argsort(dim=1, descending=True) == idx[:, None]).nonzero(as_tuple=False)[:, 1].float()

    out = {}
    for tag, mask in {
        "all":    torch.ones_like(rank, dtype=torch.bool),
        "single": single_mask,
        "outfit": ~single_mask,
    }.items():
        r_sub = rank[mask]
        out.update({f"{k}_{tag}": v for k, v in _stats(r_sub).items()})
    return out

# ── 主流程 -----------------------------------------------------------
async def run(cfg: Cfg):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    set_seed(cfg.seed)

    log = logging.getLogger("fashion_eval_opt")
    log.info("=== Evaluation start ===")

    # 讀取資料 -------------------------------------------------------
    df = pd.read_csv(cfg.csv).query("sku.notna() and pic_url.notna()").reset_index(drop=True)

    # 分群抽樣：single 與 outfit 各 cfg.num_single / num_outfit
    is_outfit  = df.sku.astype(str).str.contains(r"\+")
    df_single  = df[~is_outfit].sample(min(cfg.num_single,  len(df[~is_outfit])),  random_state=cfg.seed)
    df_outfit  = df[ is_outfit].sample(min(cfg.num_outfit, len(df[ is_outfit])), random_state=cfg.seed)
    df         = pd.concat([df_single, df_outfit]).reset_index(drop=True)
    single_mask = (~df.sku.astype(str).str.contains(r"\+")).to_numpy()

    # 下載圖片 -------------------------------------------------------
    imgs = await download_all(df.pic_url.tolist(), cfg)
    ok   = [i for i, im in enumerate(imgs) if im]  # 過濾失敗
    if not ok:
        log.error("❌ 所有圖片下載失敗"); return
    imgs         = [imgs[i] for i in ok]
    single_mask  = torch.tensor(single_mask[ok])

    # 中文品名預處理 → 翻譯 ------------------------------------------
    zh_names = [
        re.sub(r"\(.*?\)|【.*?】", "", t.split("-")[0]).strip() or "unknown"
        for t in df.pname.tolist()
    ]
    captions = Translator(cfg).zh2en(zh_names, tag="pname")

    # 影像前處理 (CPU bound) -----------------------------------------
    async def prep(idx: int, im: Image.Image, mode: str):
        """將影像派至 thread pool 前處理；失敗回傳 None。"""
        try:
            res = await asyncio.to_thread(preprocess, im.copy(), mode)
            return idx, mode, res
        except Exception as e:
            logging.warning("[PREPROC FAIL] idx=%d mode=%s err=%s", idx, mode, e)
            return idx, mode, None

    tasks = [prep(i, im, m) for i, im in enumerate(imgs) for m in cfg.preproc_modes]
    proc: dict[str, list[Image.Image | None]] = {m: [None] * len(imgs) for m in cfg.preproc_modes}
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Preprocess"):
        i, m, im = await fut; proc[m][i] = im

    # 逐模型 × 前處理評估 ---------------------------------------------
    records: list[dict[str, float | str]] = []
    for name in cfg.models:
        # ── 自動判斷是否為 Fashion-CLIP ──
        if re.search(r"fashion[-_]?clip", name, re.I):
            Wrapper = FashionClipWrapper
        else:
            Wrapper = BaseClipWrapper

        with Wrapper(name, cfg) as clip:
            clip.encode_text(captions[:1])  # *warm-up*，優化首批延遲
            for mode in cfg.preproc_modes:
                try:
                    sim = clip.similarity(captions, proc[mode])
                    rec = calc_metrics(sim, single_mask)
                except Exception as e:
                    log.error("[ERR ] %s | %s — %s", name, mode, e)
                    rec = {f"{k}_{g}": math.nan
                           for g in ("all", "single", "outfit")
                           for k in ("hit@1", "hit@5", "r@10",
                                     "mean_rank", "median_rank", "mrr")}
                rec.update(model=name, preproc=mode)
                records.append(rec)
                log.info("[DONE] %-35s | %-9s hit@1_all=%.3f",
                         name, mode, rec["hit@1_all"])

    df_out = (pd.DataFrame(records)
              .sort_values(["preproc", "hit@1_all"], ascending=[True, False])
              .reset_index(drop=True))
    print("\n" + _fmt(df_out))

# ── CLI --------------------------------------------------------------
def main(argv: list[str] | None = None):
    pa = argparse.ArgumentParser(
        prog="PrecisionSync Evaluator — fast & clean (rev-X8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    pa.add_argument("--csv",      default="data/all_products.csv", help="CSV 路徑")
    pa.add_argument("--single",   type=int, default=200,           help="單品抽樣數")
    pa.add_argument("--outfit",   type=int, default=200,           help="套裝抽樣數")
    pa.add_argument("--device",   choices=("cuda", "cpu"),         help="指定裝置")
    pa.add_argument("--no-translate", action="store_true",         help="停用翻譯")
    args = pa.parse_args(argv)

    cfg = Cfg(
        csv=Path(args.csv).expanduser(),
        num_single=args.single,
        num_outfit=args.outfit,
    )
    if args.device:
        cfg.device = args.device
        cfg.precision = "fp32" if cfg.device == "cpu" else "fp16"
    if args.no_translate:
        cfg.use_translation = False

    logging.basicConfig(
        format="[%(asctime)s] %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    asyncio.run(run(cfg))

if __name__ == "__main__":
    main()
