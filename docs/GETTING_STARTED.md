# Getting Started

This guide covers local setup, verification, running the app, retraining, and troubleshooting for VaticMacro.

## Prerequisites

- Python 3.12 (required, see `pyproject.toml:requires-python`)
- pip 24+
- Git
- Optional: Docker Desktop (for container run), FRED API key (for live refresh via `POST /api/refresh`)

Check your version:

```powershell
python --version  # expect 3.12.x
pip --version
```

## Quick Start

```powershell
git clone https://github.com/DhanushPillay/VaticMacro.git
cd VaticMacro
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
python app.py
# open http://127.0.0.1:5000  (local Flask)
# Docker instead: docker compose up --build  → http://127.0.0.1:10000
```

## Installation (detailed)

### 1. Clone

```powershell
git clone https://github.com/DhanushPillay/VaticMacro.git
cd VaticMacro
```

### 2. Virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS
python -m pip install --upgrade pip
```

### 3. Install dependencies

Single source is `pyproject.toml`. Install editable with dev tools:

```powershell
pip install -e ".[dev]"
```

This provides: `flask`, `gunicorn`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `pydantic`, `pandera`, `requests`, `beautifulsoup4`, `matplotlib`, `seaborn`, `statsmodels` + dev `pytest`, `pytest-cov`, `pytest-benchmark`, `ruff`, `mypy`, `playwright`, `mlflow`.

Strict pins are in `requirements-strict.txt` (generated via `generate_strict_env.py`). `requirements.txt` mirrors it.

### 4. Pre-commit hooks (optional but recommended)

```powershell
pre-commit install
# runs ruff check + ruff format on commit
```

## Verify Installation

```powershell
ruff check .
ruff format --check .
mypy src app
pytest -q                          # 25 tests, cov ≥50%
python benchmarks/bench_holdout.py  # Holdout R2>0.30 and MAE < naive
python benchmarks/bench_cv.py       # create_features <250ms
python benchmarks/bench_latency.py  # sandbox p95 <100ms
```

Expected (2026-09-14):

```
Ridge CV R2=0.588 | Holdout R2=0.543 MAE=0.556 RMSE=0.738 | Naive MAE=0.654  PASS
create_features 293 rows 32 cols in 20.5ms — PASS
sandbox p50=0.2ms p95=49.0ms — PASS
```

CI runs the same gate on Python 3.12: `ruff check`, `ruff format --check`, `mypy`, `pytest --cov --cov-fail-under=50`.

## Run the Application

### Local (Flask)

`app.py` is the legacy entry kept for `gunicorn app:app` compatibility. `app/factory.py:create_app()` is the testable factory used by tests and benchmarks.

```powershell
python app.py
# or
python -m flask --app app run --port 5000
# health
curl http://127.0.0.1:5000/health
# sandbox — valid
curl -X POST http://127.0.0.1:5000/api/predictive-sandbox -H "Content-Type: application/json" -d "{\"WPIATT01INM661N_pct_1m\":10}"
# sandbox — clipped (422 if |val|>50 via factory; app.py clamps to median fallback)
```

Port: local defaults to `5000` (`PORT` or `FLASK_PORT` env). Docker and Render expose `10000` (see `Dockerfile`, `render.yaml`, `docker-compose.yml`).

### Factory (for tests / benchmarks)

```powershell
python -c "from app.factory import create_app; app=create_app(); print(list(app.url_map.iter_rules()))"
```

### Docker

```powershell
docker build -t vaticmacro:test .
docker run -p 10000:10000 -e FRED_API_KEY=xxx -e REFRESH_TOKEN=yyy vaticmacro:test
# compose (recommended)
docker compose up --build
curl http://127.0.0.1:10000/health
```

`Dockerfile`: `python:3.12-slim`, non-root `appuser`, `gunicorn app:app --bind 0.0.0.0:10000 --workers 2 --preload`, `HEALTHCHECK curl --fail http://localhost:10000/health`.

### Windows helpers

```powershell
.\start.ps1        # creates .venv, installs, runs app.py
.\start.bat        # cmd equivalent
.\run_train.py     # retrain helper
```

## API Reference

SPA is served at `/` (`app/templates/cockpit.html`, hash-router: `#command-center`, `#predictive-sandbox`, `#model-registry`, `#about`).

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | — | SPA shell |
| GET | `/health` | — | `{status, model_loaded, data_status}` |
| GET | `/api/command-center` | — | Dashboard KPIs + analysis (corr, ADF) + `data_status` |
| GET | `/api/model-registry` | — | `{best_model_name, models_metrics, feature_importances, prediction_chart_data}` |
| GET | `/api/predictive-sandbox` | — | `{forecast: 12m, defaults, model_name, model_r2}` |
| POST | `/api/predictive-sandbox` | — | `{wpi_index, interest_rate, usd_inr, brent_crude, scenario_date?}` → `{display_prediction, interpretation_text}`. Factory validates `|pct|>50 → 422`. |
| POST | `/api/refresh` | `REFRESH_TOKEN` | Triggers `fetch_all_sources(cadence=all)`. Header `X-Refresh-Token` or `?token=` or JSON `token`. 401 on mismatch, 500 if `FRED_API_KEY` missing, 207 if partial errors. |
| GET | `/api/env-status` | — | `{warning}` environment mismatch banner |

`data_status` per indicator: `fresh ≤45d`, `delayed ≤90d`, `stale >90d`, `future` if date > today.

## Data and Refresh

- Truth file: `data/inflation_dataset.csv` (317 × 6, 2000-01-31 to 2026-05-31). Also mirrored in `Data/` for legacy.
- Raw sources: `data/*.csv` (FRED daily → `ME` resampled). See `docs/DATA_DICTIONARY.md`.
- Refresh: `src/data_refresh.py` fetches FRED series `INDCPIALLMINMEI`, `WPIATT01INM661N`, `INTDSRINM193N`, `DEXINUS`, `DCOILBRENTEU` + RBI DBIE fallback for WPI (never raises). Scheduler `src/scheduler.py` calls `fetch_all_sources(cadence=daily|monthly|all)`; logs append to `data/refresh_log.json`.
- Manual trigger:

```powershell
$env:FRED_API_KEY="xxx"; $env:REFRESH_TOKEN="yyy"
curl -X POST http://127.0.0.1:5000/api/refresh -H "X-Refresh-Token: yyy"
# or directly
python -c "from src.data_refresh import fetch_all_sources; print(fetch_all_sources(api_key='xxx', cadence='all'))"
```

## Retrain the Model

A pre-trained `models/best_model.pkl` is included. To retrain:

```powershell
python main.py          # legacy pipeline (resample + 3 models)
# or canonical
python run_train.py     # src/train_model via src/feature_engineering
```

Pipeline:
1. Load `data/inflation_dataset.csv`
2. `create_features` → 30 features (see `docs/DATA_DICTIONARY.md#engineered-features`)
3. Target `target_future_inflation = current_inflation_yoy.shift(-1)`
4. Split train `<2024-07-01` (216 rows), holdout `≥2024-07-01` (22 rows)
5. `TimeSeriesSplit(5)` + `GridSearchCV(scoring=r2)` over 7 candidates
6. Save `models/best_model.pkl` `{pipeline, feature_columns, model_name, environment}`, `models/metrics.json`, `models/holdout.csv`, per-model `.pkl`

Helpers `compute_metrics` and `naive_baseline_mae` in `src/train_model.py` are single source for scores. See `docs/BENCHMARKS.md` for results and reproduction.

## Merge Helper

If you update raw CSVs in `data/`:

```powershell
python merge_data.py   # outer join → ME resample → inflation_dataset.csv
python main.py         # retrain
```

`merge_data.py` mirrors `src/data_refresh.py:merge_and_save` for offline use.

## Project Structure (key paths)

```
VaticMacro/
├── app/
│   ├── factory.py              # create_app — /health + POST /api/predictive-sandbox (Pydantic)
│   ├── blueprints/             # health/api/views
│   ├── services/schemas.py     # SandboxRequest
│   └── templates/cockpit.html  # SPA
├── src/
│   ├── config.py               # frozen SETTINGS
│   ├── schemas.py              # pandera InflationDatasetSchema
│   ├── preprocessing.py        # heterogeneous CSV → ME
│   ├── feature_engineering.py  # 30 FEATURE_COLUMNS
│   ├── train_model.py          # 7 candidates, compute_metrics
│   ├── data_refresh.py         # FRED + RBI fallback
│   └── scheduler.py
├── benchmarks/                 # bench_holdout / bench_cv / bench_latency
├── tests/                      # unit + contracts + integration
├── data/inflation_dataset.csv  # 317×6 truth
├── models/                     # best_model.pkl + metrics.json + holdout.csv
└── docs/                       # ARCHITECTURE, DATA_DICTIONARY, BENCHMARKS, GETTING_STARTED
```

Full map in `docs/ARCHITECTURE.md`.

## Troubleshooting

### Model file not found

```
Warning: Could not load model at models/best_model.pkl
```

Fix: `python main.py` or `python run_train.py` to regenerate. Check `models/` exists.

### Import errors

```
ModuleNotFoundError: No module named 'src'
```

Fix: run from project root. `pyproject.toml` sets `pythonpath = ["."]` for pytest; for scripts `sys.path` is adjusted in `app.py`.

### sklearn version warning

```
InconsistentVersionWarning: Trying to unpickle estimator from version 1.8.0 ...
```

Artifact stores `environment` versions. App downgrades mismatch to banner warning (soft check) and continues. Retrain on current env to clear: `python main.py`. CI hard-fails if `metrics.json:environment` diverges.

### Refresh returns 401 or 500

- 401: `REFRESH_TOKEN` mismatch. Set same value in env and request header `X-Refresh-Token`.
- 500 `FRED_API_KEY missing`: set `FRED_API_KEY` env before `POST /api/refresh`.

### Port already in use

Local `app.py` defaults to 5000; Docker to 10000. Set `PORT=5001 python app.py` or change `docker-compose.yml` mapping.

### Tests fail on cov gate

`--cov-fail-under=50` requires 50% coverage over `src`+`app`. Run `pytest --cov=src --cov=app --cov-report=term-missing` to find gaps.

### ruff / mypy failures

```powershell
ruff check . --fix
ruff format .
mypy src app
```

Excluded paths: `app.py`, `merge_data.py`, `main.py`, `run_train.py`, `notebooks/`, `paper/`, `docs/` (see `pyproject.toml:tool.ruff.exclude`).

## Next Steps

- Read `docs/ARCHITECTURE.md` for data flow and design decisions.
- Read `docs/DATA_DICTIONARY.md` for column and feature definitions.
- Read `docs/BENCHMARKS.md` for reproduction and per-model results.
