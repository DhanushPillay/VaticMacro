# System Architecture

## Overview

VaticMacro is a Flask SPA (cockpit) that forecasts Indian YoY CPI inflation 1-month ahead from 5 monthly macro indicators. Monthly-resampled truth (317×6) avoids the daily forward-fill bias of earlier versions.

```
┌───────────┐    ┌────────────────┐    ┌───────────────┐    ┌──────────────────┐
│ FRED / RBI│───>│ data_refresh.py│───>│ Merged CSV    │───>│ feature_engineering│
│  (APIs)   │    │ merge_and_save │    │ 317×6 monthly │    │ 30 features      │
└───────────┘    └────────────────┘    └──────┬────────┘    └────────┬─────────┘
                                              │                      │
                          ┌───────────────────┴──────────────────────┴──┐
                          │            train_model.py (7 candidates)    │
                          │  Ridge / Ridge(MI) / ElasticNet / Poly /    │
                          │  RF / XGB / LGBM → best_model.pkl +         │
                          │  metrics.json + holdout.csv (22 rows)       │
                          └───────────────────┬─────────────────────────┘
                                              │
                          ┌───────────────────┴──────────────────────────┐
                          │  app.py (legacy Flask) + app/factory.py      │
                          │  SPA cockpit.html ← /api/* JSON              │
                          └───────────────────┬──────────────────────────┘
                                              │
                          ┌───────────────────┴──────────────────────────┐
                          │  Browser — Command Center / Sandbox / Registry│
                          └──────────────────────────────────────────────┘
Scheduler: src/scheduler.py (daily FX/Brent, monthly CPI/WPI/Rate) → fetch_all_sources
Contracts: src/config.py (Settings) + src/schemas.py (pandera) + app/services/schemas.py (Pydantic)
```

---

## Data Flow

### 1. Data Ingestion (`src/data_refresh.py`)

- `fetch_fred_series(series_id)` — generic FRED fetch (30s timeout, `RuntimeError` on failure)
- `fetch_rbi_dbie_series()` — best-effort RBI DBIE fallback for WPI (never raises; returns `None` stub)
- `refresh_cpi()` — CPI from local CSV + TradingEconomics scrape extrapolation (MoM append) + `validate_cpi()`
- `refresh_wpi/rate/fx/brent` — FRED + per-indicator validators (`validate_*`)
- `merge_and_save()` — outer join on `Date` → `sort → resample(ME).last()` → `inflation_dataset.csv` (317×6). `ffill` deliberately removed to avoid fake flatlines.
- `fetch_all_sources(cadence)` — respects `cadence=daily|monthly|all`; missing cadence loads existing merged slice via `_load_existing_series()`. Logs to `data/refresh_log.json`.
- `log_refresh()` — appends `{timestamp, sources:{cpi,wpi,rate,fx,brent,merged}}` (list JSON)

FRED series IDs (src/data_refresh.py::SERIES): `INDCPIALLMINMEI`, `WPIATT01INM661N`, `INTDSRINM193N`, `DEXINUS`, `DCOILBRENTEU`.

### 2. Preprocessing (`src/preprocessing.py`)

Handles heterogeneous raw CSV formats (observation_date, Row Labels, Year/Month pivot, year-columns). Standardizes to `observation_date` + value, filters to date range, drops Grand Total, `ffill().bfill()` numeric cols. Tested in `tests/unit/test_preprocessing.py`.

### 3. Feature Engineering (`src/feature_engineering.py`)

Creates **30 features** (see `FEATURE_COLUMNS` pinned list):

- Calendar: `month_sin/cos`
- AR lags: `inflation_lag_1m..12m` + `inflation_acceleration`
- Pct 1m/12m + 6m rolling for 4 indicators (WPI, Rate, FX, Brent)
- Ratios: `WPI_CPI_ratio`, `oil_inr_ratio`, `brent_vol_12m`, etc. + spreads
- Clips pct/lag/rolling to `[-50, +50]`; adds `current_inflation_yoy`; `dropna()`; drops `RAW_COLS` (11 raw cols). Tested in `tests/unit/test_feature_engineering.py` (pinned 30, no raws, no NaNs).
- `RAW_COLS` / `KEEP_COLUMNS` centralize column governance.

**Output:** `300± × 32` (30 features + Date + CPI) — `len(feat)==293` on current 317-row source.

### 4. Model Training (`src/train_model.py`)

1. **Target:** `target_future_inflation = current_inflation_yoy.shift(-1)` (1-month-ahead YoY%)
2. **Split:** train `Date < 2024-07-01`, holdout `>= 2024-07-01` (22 rows, 2024-07→2026-05)
3. **CV:** `TimeSeriesSplit(n_splits=5)` (no shuffle, no leakage)
4. **Candidates (7):** Ridge, Ridge(MI), ElasticNet, Ridge(Poly), RandomForest, XGBoost, LightGBM — each `VarianceThreshold(0.01) + SelectKBest + RobustScaler + estimator` + `GridSearchCV(scoring=r2)`
5. **Winner:** `max(CV r2)` → `models/best_model.pkl` (dict: pipeline, feature_columns, model_name, environment). Also saves per-model `.pkl` artifacts.
6. **Holdout:** `r2_score(y_test, y_pred)` on holdout; saved as `models/holdout.csv` with `Predicted_Inflation`
7. **Metrics:** `models/metrics.json` (`best_model` + list of `{name,r2_mean,mae,rmse}`)
8. **Helpers:** `compute_metrics()` + `naive_baseline_mae()` single source for metrics vs benchmark; `_capture_environment()` snapshots sklearn/xgb/pandas/numpy/python/platform for `_validate_environment()` in `app.py`.

Ridge wins on small-n (216 train months) vs tree overfit — validated by `benchmarks/bench_holdout.py` (`R2>0.3 and MAE < naive`).

### 5. Web Application

**Legacy entry:** `app.py` (1065 lines, `app = Flask(...)`) — serves SPA + JSON APIs. Kept for `gunicorn app:app` compatibility (Dockerfile/Render).

**Hardened factory:** `app/factory.py::create_app(Settings)` — factory pattern with `/health` (soft env check) and `POST /api/predictive-sandbox` (Pydantic `SandboxRequest` 422 on `|val|>50`). Blueprints `app/blueprints/` registered best-effort.

**Routes (SPA cockpit):**

| Route | Method | Handler | Response |
|-------|--------|---------|----------|
| `/` | GET | `cockpit()` | `cockpit.html` (SPA shell) |
| `/health` | GET | `health()` / factory `health()` | `{status, model_loaded, data_status}` |
| `/api/command-center` | GET | `_build_dashboard_data()` + `_build_analysis_data()` | `{dashboard:{inflation_rate,cpi,wpi,rate,fx,brent,peak/low,trend,history:24m}, analysis:{corr_matrix,stationarity}, data_status}` |
| `/api/model-registry` | GET | `_build_models_data()` | `{best_model_name, models_metrics, feature_importances, prediction_chart_data}` |
| `/api/predictive-sandbox` | GET | `_build_forecast_data()` + `_get_dynamic_defaults()` | `{forecast:{12m trend}, defaults:{wpi,rate,fx,brent}, model_name, model_r2}` |
| `/api/predictive-sandbox` | POST | `_run_prediction()` | `{display_prediction, interpretation_text, model_r2}` |
| `/api/refresh` | POST | `fetch_all_sources(api_key,cadence=all)` | `{status,details}` (401 if `REFRESH_TOKEN` mismatch) |
| `/api/env-status` | GET | `ENV_MISMATCH_WARNING` | `{warning}` |

Helpers: `_build_data_status()` (fresh ≤45d, delayed ≤90d, stale >90d, future), `_compute_yoy_r2_from_holdout()`, `_extract_feature_importances()` (coef_/feature_importances_).

**Frontend:** `app/templates/cockpit.html` — hash-router SPA (command-center / predictive-sandbox / model-registry / about), Chart.js, Tailwind via CDN, a11y: `skip-link`, `prefers-reduced-motion`, `role=main tabindex=-1`, `data-testid` on wpi input + prediction.

---

## Key Design Decisions

### Why monthly resampling?

CPI released monthly; daily forward-fill creates 96% duplicate rows that inflate training confidence then collapse on TimeSeriesSplit. Monthly `ME` resampling (317 true observations) restored CV truth (Ridge CV 0.588 vs RF -0.29).

### Why autoregressive features?

YoY inflation autocorr >0.99 at lag-1; 7 AR lags are the dominant signal. Without them CV R² drops below 0.3.

### Why Ridge over XGBoost?

n=216 training months — regularized linear generalizes; trees overfit (empirical: Ridge 0.588 vs XGB ~0.1 on same splits). Kept XGB/LGBM in grid to validate, not to win.

### Why percentage-change features?

Raw levels (e.g., CPI 35→161) drift across decades; pct changes are scale-stationary and stationarity-tested via ADF in analysis view.

### Why soft env check?

Artifact trained on sklearn 1.8.0 vs runtime 1.9.1 would crash Render on `joblib.load` hard mismatch. `_validate_environment()` downgraded to banner warning (degrade, don't crash); CI hard-fails on mismatch via `metrics.json::environment` pin.

### Why dual entry (app.py + factory)?

`app.py` preserves `gunicorn app:app` working deploy; `app/factory.py` enables `pytest` factory isolation and `create_app()` testability without circular import (`app/__init__.py` lazy re-export).

---

## Module Dependencies

```
app.py
├── src/config.py           (Settings, COLUMN_MAP, paths)
├── src/feature_engineering (create_features, FEATURE_COLUMNS)
├── src/data_refresh       (fetch_all_sources)
└── models/best_model.pkl + metrics.json + holdout.csv

app/factory.py
├── src/config.py (SETTINGS)
└── app/services/schemas.py (SandboxRequest)

src/data_refresh.py
└── src/config.py (indirect via SERIES)

src/train_model.py
├── src/feature_engineering (consumed via run_train.py)
└── src/config.py (via _capture_environment)

src/schemas.py (pandera InflationDatasetSchema)
└── src/config.py (COLUMN_MAP reference in docs)
```

---

## Configuration (`src/config.py`)

```python
@dataclass(frozen=True)
class Settings:
    COLUMN_MAP: dict = {
        "cpi": "INDCPIALLMINMEI",
        "wpi": "WPIATT01INM661N",
        "interest_rate": "INTDSRINM193N",
        "usd_inr": "DEXINUS",
        "brent_crude": "Average of DCOILBRENTEU",
        "industrial_prod": "INDPRINTO01GYSAM",
        "trade_balance": "XTNTVA01INM667N",
    }
    DATA_PATH: Path = Path("data/inflation_dataset.csv")
    MODEL_PATH: Path = Path("models/best_model.pkl")
    METRICS_PATH: Path = Path("models/metrics.json")
    HOLDOUT_PATH: Path = Path("models/holdout.csv")
    REFRESH_LOG: Path = Path("data/refresh_log.json")
    MODEL_DIR: Path = Path("models")


SETTINGS = Settings()
```

Backwards-compat re-exports `DATA_PATH/MODEL_PATH/METRICS_PATH` as strings for `app.py`.

## Observability & Security

- **Health:** `GET /health` returns `{status:ok, model_loaded:bool, data_status}`; Dockerfile `HEALTHCHECK` curls it on `:10000`.
- **Provenance:** every artifact carries `environment` (sklearn/xgb/pandas/numpy/python/platform) checked soft on startup and hard in `benchmarks/`.
- **Refresh auth:** `POST /api/refresh` checks `REFRESH_TOKEN` env var via `X-Refresh-Token` header / `?token=` / JSON `token` (401 on mismatch); Render `render.yaml` sets `REFRESH_TOKEN: sync:false`.
- **Non-root container:** Dockerfile creates `appuser` and `USER appuser`; `EXPOSE 10000`, `gunicorn app:app --bind 0.0.0.0:10000 --workers 2 --preload`.
- **Benchmarks:** `bench_holdout.py` (R2>0.3 + MAE<naive), `bench_cv.py` (create_features <100ms), `bench_latency.py` (sandbox p95 <200ms).
- **CI:** `.github/workflows/ci.yml` (py3.12, pip install -e .[dev], ruff check+format, mypy, pytest --cov, docker build).
```

