# ME3297 Project: Flagging DataCo orders at risk of late delivery

The code runs top to bottom and reproduces every table and figure in the report.
Tested with Python 3.14 on Linux.

## Steps

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Get the data. It is NOT included in the submission (licence terms).
#    Download the zip "8gx2fvg2k6-5.zip" (~26 MB) from
#    https://data.mendeley.com/datasets/8gx2fvg2k6/5  ("Download All")
#    and put it in this folder, or pass its path as an argument:
python get_data.py                     # or: python get_data.py path/to/8gx2fvg2k6-5.zip

# 2. Run the whole analysis (~4 minutes on 4 CPU cores)
python run_all.py
```

Outputs:

- `outputs/tables/*.csv`: data audit, cross-validation folds and summary, test results with bootstrap intervals, policy and cost-sensitivity tables, promise scenarios.
- `outputs/figures/*.png`: all figures.
- `outputs/summary.json`: headline numbers.

All random seeds are fixed (`SEED = 42`).

## Dataset

Constante, F., Silva, F., & Pereira, A. (2019). *DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS* (Version 5) [Data set]. Mendeley Data. https://doi.org/10.17632/8gx2fvg2k6.5. Licence: CC BY 4.0.

## Code layout

| File | Purpose |
|---|---|
| `get_data.py` | Extracts the two CSV files from the Mendeley zip into `data/raw/` |
| `src/data_prep.py` | Loading and the data-quality audit. Also builds the order-level table and the leakage-safe history features (a prior outcome is used only once it was known) |
| `src/modeling.py` | Preprocessing pipeline, baselines, the seven models, metrics and the cost model |
| `run_all.py` | Time split, rolling-origin CV, test evaluation, interpretation, recommendation tables |

## Key design decisions

- **Unit and moment:** one row per order, scored at order release (after payment, before dispatch).
- **Excluded as post-outcome:** `Days for shipping (real)`, `Delivery Status`, `Order Status` and `shipping date` (which equals order date + real days in 97% of rows). Cancelled and suspected-fraud orders (7,754 lines) are removed, because they are never delivered.
- **Split:** by time. Training covers orders before 2017-07-01; the test set covers 2017-07-01 to 2018-01-31. Validation uses `TimeSeriesSplit(5)` on the training period.
- **Preprocessing:** all of it sits inside a scikit-learn `Pipeline`/`ColumnTransformer`, so it is refit in every fold.
