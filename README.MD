# VaticMacro

## Indian Macroeconomic Inflation Prediction and Trend Forecasting

VaticMacro is an advanced machine learning framework explicitly architected for the Indian economy. It is designed to forecast inflation trends through the rigorous analysis of domestic macroeconomic indicators, global market dynamics, and regional policy metrics. The system employs complex data analytics, statistical modeling, and predictive forecasting techniques to yield precise inflation estimates and actionable economic insights tailored to the Indian subcontinent.

This platform empowers domestic policy-makers, researchers, institutional investors, and financial analysts to comprehensively evaluate regional inflation paradigms, assess macroeconomic risks within emerging markets, and execute data-driven strategic decisions based on empirical economic projections.

## Status — FAANG-hardened (2026-09-14)

- **Stack:** Python 3.12 (`requires-python >=3.12,<3.15`), Flask 3.1 + Gunicorn 22, Pandas 2.2, scikit-learn 1.5, XGBoost 2.1, LightGBM 4.5, Pydantic 2.8 + Pandera 0.20 — single source `pyproject.toml`, `requirements-strict.txt` (15 libs, hatch build)
- **Data truth:** 317 monthly rows (2000-01-31 → 2026-05-31), 6 core indicators (`INDCPIALLMINMEI`, `WPIATT01INM661N`, `INTDSRINM193N`, `DEXINUS`, `DCOILBRENTEU`, `Date`), 30 engineered features (lags/rolling/momentum, clipped ±50%, `src/feature_engineering.py:12` `FEATURE_COLUMNS`), target YoY CPI `INDCPIALLMINMEI`
- **Benchmarks (live 2026-09-14, `benchmarks/`):** `bench_holdout.py` Ridge CV R² 0.588 | Holdout R² 0.543 MAE 0.556 RMSE 0.738 vs Naive MAE 0.654 — **PASS**; `bench_cv.py` `create_features` 293 rows × 32 cols in 20.5 ms — **PASS** (<250 ms); `bench_latency.py` sandbox p50 0.2 ms p95 49.0 ms — **PASS** (<100 ms)
- **Quality gates:** `pytest -q` 25 tests 56% coverage (`--cov-fail-under=50`), `ruff check` 0, `ruff format --check` 28 files, `mypy src app` 0 errors (16 files), Pydantic validation 422 on `±50` overflow, `GET /health` 200 `model_loaded:true`
- **Prod:** `Dockerfile` `python:3.12-slim` non-root `appuser` + `HEALTHCHECK` `http://localhost:10000/health`, `render.yaml` `healthCheckPath: /health` + `REFRESH_TOKEN` protects `POST /api/refresh`, cron `daily` FX/Brent + `monthly` CPI/WPI/Rate, RBI fallback `src/data_refresh.py:fetch_rbi_dbie_series`
- **Docs sync:** `docs/ARCHITECTURE.md` + `docs/DATA_DICTIONARY.md` updated to 317×6 monthly truth

---

# Problem and Solution

## Problem Statement

Inflation serves as a critical barometer of macroeconomic health, exerting profound effects on purchasing power, capital allocation, municipal policy formulation, and corporate financial strategy.

The accurate prediction of intra- and inter-period inflation is a multifaceted challenge, largely due to its dependency on a vast array of interconnected systemic factors, which include:

- Consumer Price Index (CPI)
- Wholesale Price Index (WPI)
- Benchmark Interest Rates
- Gross Domestic Product (GDP) Growth Trajectories
- Unemployment Metrics
- Foreign Exchange Rate Volatility
- Commodity and Energy Valuations
- Aggregate Money Supply

Conventional forecasting paradigms often fail to discern non-linear or emergent patterns within voluminous economic datasets and frequently lack the agility required to model dynamic macroeconomic shifts.

As global economies grow increasingly interconnected, the necessity for robust, data-driven analytical solutions capable of parsing multivariate economic indicators to synthesize dependable inflation forecasts has become paramount.

---

## The VaticMacro Methodology

VaticMacro centralizes the complex process of economic trend analysis and inflation projection.

The framework systematically:

- Aggregates and normalizes historical macroeconomic data.
- Quantifies correlations and causal relationships among primary economic indicators.
- Implements advanced machine learning topologies to forecast inflation velocity.
- Renders multi-dimensional economic patterns through comprehensive visual dashboards.
- Validates predictive efficacy utilizing industry-standard statistical evaluation metrics.
- Synthesizes complex outputs into actionable strategic intelligence.

By translating dense economic datasets into coherent predictive models, VaticMacro provides stakeholders with the requisite intelligence to anticipate future macroeconomic conditions and formulate rigorous strategic responses.

---

# Core Capabilities

## Economic Data Analysis

- Historical inflation trend aggregation
- Economic indicator benchmarking and comparison
- Multivariate correlation analysis
- Rigorous data preprocessing and normalization
- Advanced feature engineering and selection

---

## Machine Learning Forecasting

- Inflation trajectory prediction utilizing historical datasets
- Predictive modeling via ensemble and linear methodologies
- Time-series statistical forecasting
- Macroeconomic trend extrapolation
- Forward-looking inflation estimation

Supported Methodologies:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- ARIMA (AutoRegressive Integrated Moving Average)

---

## Data Visualization

- Programmatic rendering of inflation trend trajectories
- Forecast divergence and comparison graphing
- Empirical correlation heatmaps
- Macroscopic economic indicator dashboards
- Diagnostic visual insight generation

---

## Performance Evaluation

Algorithmic performance is rigorously validated utilizing:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- Coefficient of Determination (R² Score)

This methodology ensures maximum fidelity and statistical reliability in all generated forecasts.

---

# Technical Architecture

## Backend and Development Environment

- Python 3.12 (`requires-python >=3.12,<3.15`, `hatchling` build) — `Dockerfile` `python:3.12-slim` + Gunicorn 22, Flask 3.1
- `pyproject.toml` single source (15 deps) + `requirements-strict.txt` pinned via `generate_strict_env.py`, `.pre-commit-config.yaml` (ruff/mypy)

## Data Processing Libraries

- Pandas 2.2 + NumPy 1.26 + Pandera 0.20 (`src/schemas.py` `InflationDatasetSchema`)

## Data Visualization Modules

- Matplotlib 3.8 + Seaborn 0.13 + Statsmodels 0.14

## Machine Learning and Forecasting Implementations

- Scikit-learn 1.5 + XGBoost 2.1 + LightGBM 4.5 + Ridge ensemble (`src/train_model.py:compute_metrics`, `naive_baseline_mae`, `_capture_environment`)
- 7 candidates: Ridge / MI / Poly / ElasticNet / RandomForest / LightGBM / XGBoost — best saved `models/ridge.pkl` + `metrics.json` + `holdout.csv`

## Analytical Methodologies

- 30 FEATURE_COLUMNS clipped ±50% (`src/feature_engineering.py:12`), monthly resample `ME`, Pydantic `SandboxRequest` validation 422

---

# Dataset Characteristics

Monthly truth — 317 rows (2000-01-31 → 2026-05-31), 6 columns, resampled `ME` (no daily duplication). Validated by `src/schemas.py:InflationDatasetSchema` + `src/config.py:SETTINGS` frozen dataclass.

### Primary Core Indicators (raw `data/inflation_dataset.csv`)
- Indian CPI `INDCPIALLMINMEI` [Monthly, FRED, target YoY `pct_change(12)*100`]
- WPI Industry `WPIATT01INM661N` [Monthly, FRED + RBI DBIE fallback `src/data_refresh.py:fetch_rbi_dbie_series`]
- RBI Rate `INTDSRINM193N` [Monthly, FRED]
- FX `DEXINUS` USD/INR [Daily → monthly `ME`]
- Brent `DCOILBRENTEU` / `Average of DCOILBRENTEU` [Daily → monthly]

### Derived Target & Features
- **Target:** `INDCPIALLMINMEI` YoY % (monthly, shift 12)
- **30 engineered features** (`src/feature_engineering.py:FEATURE_COLUMNS`) — lags 1/3/6/12, rolling mean/std, momentum, pct_change, clipped `±50%`, `RobustScaler` + `VarianceThreshold` + `SelectKBest` in `train_model.py`
- **Refresh:** `src/data_refresh.py:merge_and_save` + `log_refresh` + `fetch_all_sources` cadence `daily` (FX/Brent) / `monthly` (CPI/WPI/Rate), `src/scheduler.py` checks `FRED_API_KEY`

### Phase II Extensibility
Future: Unemployment annual, India Basket Crude, M1/M3 — plaintiffs kept out of core 30 to avoid leakage; add when monthly proxy available.

---

# Project Workflow

## 1. Data Aggregation
Systematic collection of historical macroeconomic and inflation-related datasets from authoritative public and institutional financial sources.

## 2. Data Preprocessing Pipeline
- **Time-Series Frequency Alignment:** Algorithmic downsampling of Daily data (Exchange Rates) and interpolation of Annual data (GDP) to synthesize a unified Monthly frequency matrix.
- Mitigation and imputation of null values.
- Comprehensive data sanitization.
- Algorithmic outlier detection and handling.
- Statistical data normalization.
- Predictive feature selection strategies.

## 3. Exploratory Data Analysis (EDA)

- Empirical trend identification and smoothing
- Statistical distribution variance analysis
- Bivariate and multivariate correlation mapping
- Discovery of latent macroeconomic patterns

## 4. Feature Engineering

- Synthesis of derived analytical variables
- Optimization of critical economic indicators
- Algorithmic dataset dimensionality reduction

## 5. Model Architecture and Training

Training of diverse forecasting algorithms, including:

- Linear Regression
- Random Forest Ensembles
- XGBoost Architectures
- ARIMA Time-Series Models

## 6. Model Evaluation and Validation

Rigorous evaluation of predictive fidelity utilizing:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- Coefficient of Determination (R² Score)

## 7. Predictive Generation

Generate future inflation forecasts and economic trend predictions.

---

# Project Structure

```text
VaticMacro/
├── app/
│   ├── factory.py (FAANG factory `create_app` — /health + /api/predictive-sandbox Pydantic 422)
│   ├── blueprints/{health,api,views}.py
│   ├── services/schemas.py (SandboxRequest clipped ±50)
│   └── templates/cockpit.html (1243-line SPA — skip-link, prefers-reduced-motion, data-testid)
├── src/
│   ├── config.py (frozen SETTINGS dataclass)
│   ├── schemas.py (Pandera InflationDatasetSchema)
│   ├── preprocessing.py (Daily→ME resample, pivot/year variants)
│   ├── feature_engineering.py (30 FEATURE_COLUMNS, clipping)
│   ├── train_model.py (compute_metrics/naive_baseline_mae/_capture_environment, 7 candidates, mlflow hook)
│   ├── data_refresh.py (RBI fallback, merge_and_save, log_refresh, fetch_all_sources)
│   └── scheduler.py (FRED_API_KEY check, daily/monthly cadence)
├── benchmarks/{bench_holdout,bench_cv,bench_latency}.py (holdout R² 0.543 MAE 0.556 vs naive 0.654)
├── tests/{unit,contracts,integration}/ (25 tests, 56% cov --cov-fail-under=50)
├── data/inflation_dataset.csv (317×6 monthly truth — also Data/ legacy copy)
├── models/{ridge,lightgbm,xgboost,best_model}.pkl + metrics.json + holdout.csv
├── docs/{ARCHITECTURE,DATA_DICTIONARY,GETTING_STARTED}.md (monthly truth)
├── pyproject.toml (single source, requires-python >=3.12,<3.15) + requirements-strict.txt
├── Dockerfile (3.12-slim, non-root, HEALTHCHECK 10000/health) + render.yaml + docker-compose.yml
└── app.py / merge_data.py / run_train.py (legacy entrypoints — prefer factory + src/)
```

---

# Expected Operational Outcomes

The execution of the framework yields:

- Quantifiable future inflation projections
- Formulated economic trend predictions
- Empirical indicator importance metrics
- Robust forecast accuracy analytics
- Synthesized, graphical economic intelligence

---

# Applied Use Cases

VaticMacro is architected to facilitate:

### Government and Policy Planning
Supports macroeconomic stabilization strategies and empirical policy formulation.

### Financial Institutions
Assists banking and regulatory organizations in extensive systemic risk assessment and operational forecasting.

### Investment Analysis
Offers institutional investors quantifiable data regarding future macroeconomic climates.

### Strategic Corporate Planning
Supports rigorous budgeting, pricing models, and structural strategic allocations.

### Academic Research
Serves as an empirical foundation for advanced economic and temporal forecasting research initiatives.

---

# Future Development Horizons

- **Continuous Integration:** Integration of live, real-time macroeconomic datasets via API pipelines.
- **Hybrid Architectures:** Implementation of dual-model ensembles, matching statistical baseline algorithms like ARIMA with advanced non-linear deep learning (e.g., ARIMA-LSTM).
- **Time-Series Transformers:** Application of state-of-the-art Temporal Fusion Transformers (TFT) equipped with attention mechanics for superior multi-horizon extrapolation.
- **Alternative Behavioral Data:** Integration of Natural Language Processing (NLP) models to ingest and execute sentiment analysis on Central Bank minutes and macroeconomic news indicators.
- **Interactive Dashboards:** Implementation of dynamic, interactive forecasting suites.
- **Granularity:** Region-specific and localized inflation metrics.
- **Scale:** Expansion into cross-border, multi-national macroeconomic modeling.

---

# Research Contributors

- Dhanush Pillay & Shubhangini Dixit

---

# Licensing

This repository and framework are published explicitly for academic, research, and non-commercial educational purposes.

---

## Running locally (Windows 3.12)

**Install (single source):**
```powershell
python -m pip install -e ".[dev]"        # or: pip install -r requirements-strict.txt
pre-commit install
```

**Quality gates (must pass):**
```powershell
ruff check .; ruff format --check .; mypy src app   # 0 errors, 28 files
pytest -q                                            # 25 passed, 56% cov (gate 50)
python benchmarks/bench_holdout.py   # Ridge CV 0.588 | Holdout R2 0.543 MAE 0.556 vs naive 0.654 PASS
python benchmarks/bench_cv.py        # 20.5 ms PASS
python benchmarks/bench_latency.py   # p95 49 ms PASS
```

**App (factory, 10000):**
```powershell
python -m pip install -r requirements.txt   # legacy compat
python app.py                      # legacy SPA (still works)
# or factory:
$gunicorn app:app --bind 0.0.0.0:10000 --workers 2 --preload   # Dockerfile CMD
# health + sandbox smoke:
curl http://127.0.0.1:10000/health
curl -X POST http://127.0.0.1:10000/api/predictive-sandbox -H "Content-Type: application/json" -d "{\"WPIATT01INM661N_pct_1m\":10}"
# 422 on bad: {"WPIATT01INM661N_pct_1m":60}
```

**Docker / Render:**
```powershell
docker build -t vaticmacro:test .
docker run -p 10000:10000 -e FRED_API_KEY=xxx -e REFRESH_TOKEN=yyy vaticmacro:test
docker compose up --build
```

**Windows quick start (legacy scripts):**
```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
# or: start.bat
```
