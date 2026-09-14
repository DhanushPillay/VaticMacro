# Benchmarks

Reproduction and results for VaticMacro — Indian CPI YoY 1-month-ahead forecasting. All scores are reproducible from `data/inflation_dataset.csv` (317 × 6) with the frozen pipeline in `src/train_model.py` and `src/feature_engineering.py`.

Last run: **2026-09-14** — Python 3.12, scikit-learn 1.5+, xgboost 2.1+, lightgbm 4.5+, pandas 2.2+

---

## Dataset and Split

- Source: `data/inflation_dataset.csv`, monthly `ME` resampled, 2000-01-31 to 2026-05-31, 317 rows, 6 columns. No daily forward-fill.
- Target: `inflation_yoy = (CPI_t − CPI_{t-12}) / CPI_{t-12} × 100`, then `target_future = current_inflation_yoy.shift(-1)`.
- Features: 30 columns from `src/feature_engineering.py:FEATURE_COLUMNS` (see `docs/DATA_DICTIONARY.md#engineered-features`), clipped `±50`, `dropna` → ~293 × 32 (30 features + `Date` + `CPI`).
- Split: train `Date < 2024-07-01` (216 rows after `dropna`), holdout `Date ≥ 2024-07-01` (22 rows, 2024-07 → 2026-05). Holdout saved as `models/holdout.csv` with `target_future_inflation` and `Predicted_Inflation`.
- Validation: `TimeSeriesSplit(n_splits=5)` without shuffle, scoring `r2`. No leakage: scaling and selection are inside the pipeline.

## Pipeline

Single pipeline per candidate (same splits, same preprocessing):

```
VarianceThreshold(0.01) → SelectKBest(f_regression, k ∈ {10, 15, 20, 30}) → RobustScaler → estimator
```

Candidates (7):

| Key | Estimator | Grid |
|-----|-----------|------|
| `Ridge` | `Ridge` | `alpha {0.1, 1, 10, 100}` |
| `Ridge(MI)` | `Ridge` with `mutual_info_regression` | same `alpha` |
| `ElasticNet` | `ElasticNet` | `alpha {0.1,1,10}, l1_ratio {0.2,0.5,0.8}` |
| `Ridge(Poly)` | `PolynomialFeatures(2)` + `Ridge` | `alpha {1,10}` |
| `Random Forest` | `RandomForestRegressor` | `n_estimators {100,200}, max_depth {5,10}` |
| `XGBoost` | `XGBRegressor` | `n_estimators {100}, max_depth {3,5}` |
| `LightGBM` | `LGBMRegressor` | `n_estimators {100}, max_depth {3,5}` |

Winner: `max(CV r2_mean)` → `models/best_model.pkl` `{pipeline, feature_columns, model_name, environment}`. Per-model artifacts also saved as `models/<key>.pkl`.

Helpers `compute_metrics(y_true, y_pred)` and `naive_baseline_mae(y_true)` in `src/train_model.py` are the single source for MAE/RMSE/R² and the persistence baseline (`shift(1)`).

---

## How to Reproduce

```powershell
# 1. Train (overwrites models/*.pkl, metrics.json, holdout.csv)
python run_train.py
# or legacy
python main.py

# 2. Benchmarks
python benchmarks/bench_holdout.py
python benchmarks/bench_cv.py
python benchmarks/bench_latency.py

# 3. Unit gate (also run in CI)
pytest -q
```

`bench_holdout.py` loads `models/metrics.json` and `models/holdout.csv`, computes `compute_metrics` and `naive_baseline_mae`, prints and asserts `R² > 0.30 and MAE < naive`. `bench_cv.py` times `create_features` on the full 317-row CSV. `bench_latency.py` hits `app.factory:create_app` test client for p50/p95.

---

## Results

### Per-model CV (5-fold TimeSeriesSplit, `r2_mean`)

From `models/metrics.json` (2026-09-14):

| Model | `r2_mean` | MAE | RMSE |
|-------|-----------|-----|------|
| **Ridge** | **0.588** | 0.883 | 1.109 |
| Ridge(MI) | 0.482 | 0.984 | 1.275 |
| ElasticNet | 0.495 | 0.973 | 1.203 |
| Ridge(Poly) | 0.236 | 1.222 | 1.502 |
| Random Forest | −0.295 | 1.551 | 1.934 |
| XGBoost | −0.213 | 1.519 | 1.873 |
| LightGBM | −0.401 | 1.573 | 1.956 |

Best model: **Ridge** (regularized linear generalizes at n=216; trees overfit — same splits, negative R²).

### Holdout (out-of-sample, 22 rows)

From `benchmarks/bench_holdout.py` on `models/holdout.csv`:

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| CV R² (Ridge, 5-fold) | 0.588 | — | — |
| **Holdout R²** | **0.543** | > 0.30 | Pass |
| **Holdout MAE** | **0.556** | < naive 0.654 | Pass |
| Holdout RMSE | 0.738 | — | — |
| Naive persistence MAE (`y_{t-1}`) | 0.654 | — | baseline |

Holdout predictions stored in `models/holdout.csv:Predicted_Inflation` vs `target_future_inflation`. R² computed via `sklearn.metrics.r2_score`; MAE/RMSE via `src/train_model.py:compute_metrics`.

### Latency

| Benchmark | Measured | Threshold | Status |
|-----------|----------|-----------|--------|
| `bench_cv.py` — `create_features` (293 × 32) | **20.5 ms** | < 250 ms | Pass |
| `bench_latency.py` — sandbox `POST /api/predictive-sandbox` p50 | **0.2 ms** | — | — |
| `bench_latency.py` — sandbox p95 (20 calls, test client) | **49.0 ms** | < 100 ms | Pass |

Feature engineering is vectorized pandas; endpoint latency is pipeline `predict` on a single 30-feature row.

---

## Interpretation

- **Why Ridge wins:** 216 training months is small-n. The signal is autoregressive (YoY inflation lag-1 autocorr > 0.99) plus smooth macro pct changes; L2 regularization captures it without fitting noise. Trees memorize splits and degrade out of fold.
- **Why holdout MAE matters:** Naive persistence (repeat last YoY) already tracks inflation well (MAE 0.654). Beating it by 0.10 points at R² 0.543 shows learned macro signal beyond autocorrelation.
- **Why 0.30 gate:** With 22 holdout points, R² is volatile. 0.30 is a conservative floor for “better than seasonal mean” while keeping headroom for the observed 0.54.

---

## Environment Provenance

Every artifact `models/best_model.pkl` stores:

```python
environment = {
  "scikit-learn": "1.5.x",
  "xgboost": "2.1.x",
  "pandas": "2.2.x",
  "numpy": "1.26.x",
  "python": "3.12.x",
  "platform": "...",
}
```

Captured by `src/train_model.py:_capture_environment()` at train time, checked soft on startup in `app.py:_validate_environment()` (banner warning, not crash) and hard in CI by comparing `metrics.json:environment` to runtime. Retrain on upgrade to clear mismatch.

---

## Files

- `models/metrics.json` — CV table (source for per-model block above)
- `models/holdout.csv` — 22-row holdout with predictions (source for holdout block)
- `models/best_model.pkl` — winner pipeline + `feature_columns` + `environment`
- `benchmarks/bench_holdout.py`, `bench_cv.py`, `bench_latency.py` — runners that print `PASS`/`FAIL`
- `src/train_model.py:compute_metrics`, `naive_baseline_mae` — metric single source

## Notes

- R² values are YoY percent R² (target is YoY %), not CPI level R².
- All 317 rows are monthly truth; earlier daily 7023-row builds are deprecated and not comparable.
- For API benchmarks against the running server (instead of test client), hit `GET /health` and `POST /api/predictive-sandbox` under load and measure p95 externally.
