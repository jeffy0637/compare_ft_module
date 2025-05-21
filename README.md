# Fashion Embedding Evaluation

This repository contains a script (`main.py`) for evaluating various image-text embedding models on a set of fashion products. The script downloads product images, optionally translates Chinese product names to English, preprocesses the images using different strategies, and then computes retrieval metrics for several embedding models.

## Environment Setup

1. **Python**: Use Python 3.11 or later.
2. **Install dependencies**:

   ```bash
   python -m pip install -r requirements.txt
   ```

   If you do not need automatic translation, run the script with `--no-translate` to avoid the optional `translators` dependency.

## Running the Script

```bash
python main.py --csv all_products.csv
```

Additional arguments:

- `--single`: number of single-item products to sample (default: 200)
- `--outfit`: number of outfit products to sample (default: 200)
- `--device`: `cuda` or `cpu` (auto-detected by default)
- `--no-translate`: disable Baidu translation

## Data

The repository includes a sample dataset `all_products.csv`. The CSV file must contain the columns `sku`, `pic_url`, and `pname` as expected by the script.

