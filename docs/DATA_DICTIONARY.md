# Dataset Documentation

## Merged Dataset

**File:** `data/inflation_dataset.csv`
**Rows:** 317 (monthly frequency, 2000-01-31 to 2026-05-31)
**Columns:** 6

### Column Reference

| # | Column Name | Human Name | Frequency | Source | Unit |
|---|-------------|-----------|-----------|--------|------|
| 1 | `Date` | Month-end date | Monthly | — | YYYY-MM-DD |
| 2 | `INDCPIALLMINMEI` | Consumer Price Index | Monthly | [FRED](https://fred.stlouisfed.org/series/INDCPIALLMINMEI) | Index 2015=100 |
| 3 | `WPIATT01INM661N` | Wholesale Price Index | Monthly | [FRED](https://fred.stlouisfed.org/series/WPIATT01INM661N) | Index |
| 4 | `INTDSRINM193N` | RBI Repo Rate | Monthly | [FRED](https://fred.stlouisfed.org/series/INTDSRINM193N) | % |
| 5 | `DEXINUS` | USD/INR Exchange Rate | Monthly (ME resampled) | [FRED](https://fred.stlouisfed.org/series/DEXINUS) | INR per USD |
| 6 | `Average of DCOILBRENTEU` | Brent Crude Oil (ICE) | Monthly (ME resampled) | [FRED](https://fred.stlouisfed.org/series/DCOILBRENTEU) | USD/barrel |

### Important Notes

1. **Monthly truth (no daily forward-fill):** `src/data_refresh.py` resamples all series to month-end (`ME`) and takes the last observation per month. No flat forward-fill rows are stored. Earlier versions with 7,023 daily rows are deprecated.

2. **CPI coverage:** CPI is OECD-sourced monthly. The dataset is current to 2026-05-31. `app.py:_build_data_status()` reports `fresh / delayed / stale` per indicator by checking `days since last observation`.

3. **Target variable:** YoY inflation is not a column — computed at training time:
   ```
   inflation_yoy = ((CPI_t - CPI_{t-12}) / CPI_{t-12}) × 100
   ```
   The training target is `target_future_inflation = current_inflation_yoy.shift(-1)` (1-month-ahead).

4. **Validation:** `src/schemas.py::InflationDatasetSchema` (pandera) enforces `Date` non-null, `INDCPIALLMINMEI >= 0`, optional indicators `>=0`, alias for Brent column. Checked in `tests/contracts/`.

5. **Refresh log:** `data/refresh_log.json` appends `{timestamp, sources: {cpi,wpi,rate,fx,brent,merged}}` on every `POST /api/refresh` or scheduler run.

---

## Raw Source Files

Located in `data/`, these are the original CSVs before merging (FRED daily series are resampled to ME on merge):

| File | Indicator | Frequency in FRED | Merge handling |
|------|-----------|-------------------|----------------|
| `INDCPIALLMINMEI (Consumer Price Index).csv` | CPI | Monthly | Direct month-end |
| `WPIATT01INM661N (Wholesale Prices).csv` | WPI | Monthly | Direct month-end |
| `INTDSRINM193N (Interest Rate).csv` | RBI Rate | Monthly | Direct month-end |
| `DEXINUS (USDINR).csv` | USD/INR | Daily | `resample(ME).last()` |
| `DCOILBRENTEU(crude oil).csv` | Brent Crude | Daily | `resample(ME).last()` |

### Merging Process

`src/data_refresh.py::fetch_all_sources(cadence)` + `merge_and_save()`:
1. `refresh_*()` fetches each FRED series (or RBI DBIE fallback for WPI — `fetch_rbi_dbie_series()`)
2. `merge_and_save()` outer-joins on `Date`, `sort + resample(ME).last()`, writes `inflation_dataset.csv` (317×6)
3. `log_refresh()` appends to `data/refresh_log.json`
4. Scheduler (`src/scheduler.py`) calls `fetch_all_sources` daily (FX/Brent) and monthly-5th (CPI/WPI/Rate)

---

## Engineered Features

The pipeline `src/feature_engineering.py::create_features()` creates **30 engineered features** from the 5 indicators (plus `Date` + `CPI` target basis):

### Feature Types

| Type | Count | Example | Purpose |
|------|-------|---------|---------|
| Calendar | 2 | `month_sin`, `month_cos` | Seasonality |
| AR inflation lags | 7 | `inflation_lag_1m` … `inflation_lag_12m`, `inflation_acceleration` | Autocorrelation (strongest signal) |
| MoM / YoY pct + 6m rolling | 12 | `WPIATT01INM661N_pct_1m`, `_pct_12m`, `_rolling_6m` (×4 indicators) | Momentum |
| Ratios | 4 | `WPI_CPI_ratio`, `oil_inr_ratio`, `oil_inr_pct_1m`, `brent_vol_12m` | Cross-indicator |
| Spreads / momentum | 4 | `wpi_cpi_spread`, `inr_momentum_3m`, `rate_direction_3m`, `current_inflation_yoy` | Divergence |
| Volatility | 1 | `brent_vol_12m` (12m std of Brent MoM) | Energy shock |

All pct/lag/rolling cols are clipped to `[-50, +50]` to preserve crisis signal while capping outliers. Raw indicator columns listed in `RAW_COLS` are dropped; output is `(n_months - dropna) × 32` (30 features + Date + CPI). Pinned in `FEATURE_COLUMNS` (30 names) and validated in `tests/unit/test_feature_engineering.py`.

### Why these features

* Autoregressive lags dominate (lag-1 autocorr >0.99) — 7 lags cover 1m/2m/3m/6m/9m/12m + acceleration.
* Percentage changes are scale-independent (raw GDP-scale would not generalize across decades).
* Ratios capture WPI→CPI pass-through and oil/FX co-movement without leaking level.

---

## Model Artifacts

### `models/best_model.pkl`

Joblib-serialized dict:

```python
{
    "pipeline": Pipeline(
        [VarianceThreshold, SelectKBest(f_regression), RobustScaler, Ridge]
    ),
    "feature_columns": [
        ...
    ],  # 30 names (FEATURE_COLUMNS at train time; legacy artifact has 34)
    "model_name": "Ridge",  # Display name (winner among 7 candidates)
    "best_model_choice": "Ridge",
    "environment": {"scikit-learn", "xgboost", "pandas", "numpy", "python", "platform"},
}
```
`environment` is validated soft (warn, not crash) on `app.py` startup via `_validate_environment()`.

### `models/metrics.json`

Cross-validation (5-fold `TimeSeriesSplit`, scoring `r2`) + holdout:

```json
{
  "best_model": "Ridge",
  "metrics": [
    {"name":"Ridge","r2_mean":0.588,"mae":0.883,"rmse":1.109},
    {"name":"Ridge(MI)","r2_mean":0.482,...},
    {"name":"ElasticNet","r2_mean":0.495,...},
    {"name":"Ridge(Poly)","r2_mean":0.236,...},
    {"name":"Random Forest","r2_mean":-0.295,...},
    {"name":"XGBoost",...},
    {"name":"LightGBM",...}
  ]
}
```
Holdout is `models/holdout.csv` (22 rows, 2024-07 → 2026-05, `target_future_inflation` vs `Predicted_Inflation`). Benchmark `benchmarks/bench_holdout.py` asserts `R2>0.3 and MAE < naive MAE`.

### `models/holdout.csv` vs `metrics.json`

* `metrics.json:r2_mean` is CV R² (in-sample splits).
* `holdout.csv` gives true out-of-sample R², compared to naive persistence baseline `naive_baseline_mae()` in `src/train_model.py`.

## Data Contracts

* `src/config.py::Settings` (frozen dataclass) — single source for `COLUMN_MAP`, `DATA_PATH`, `MODEL_PATH`, `METRICS_PATH`, `HOLDOUT_PATH`, `REFRESH_LOG`, `MODEL_DIR`.
* `src/schemas.py::InflationDatasetSchema` — pandera DataFrameModel used in preprocessing and tests.
* `app/services/schemas.py::SandboxRequest` — Pydantic validation for `POST /api/predictive-sandbox` (all pct fields `ge=-50 le=50`, alias for Brent).
