# Breathe Project - Context Document

## Project Overview

**Goal:** End-to-end MLOps portfolio project for PM2.5 air quality prediction with automated drift detection and retraining.

**Scope:**
- 5 cities (Paris, Delhi, Beijing, NYC, LA)
- 2 years data (May 2023 - April 2025)
- Monthly automated pipeline: ingestion → drift check → conditional retraining → deployment

**Target Audience:** Data Science hiring managers (demonstrate MLOps competencies)

---

## Tech Stack

**Core:**
- Python 3.13.11 with `uv` package manager
- Data: OpenAQ API (PM2.5) + OpenWeather API (weather features)
- ML: LightGBM/XGBoost with MLflow (tracking, registry)
- Orchestration: Prefect (flows + Cloud Scheduler)
- Cloud: GCP (BigQuery for data, GCS for artifacts, Cloud Run for deployment)
- API: FastAPI for serving

**Development:**
- Jupyter notebooks for EDA/experimentation
- Git with feature branches
- Black/Ruff for code quality

---

## Architecture (High-Level)
```
Monthly Pipeline (Autonomous):
Cloud Scheduler → Prefect Flow →
  1. Ingest data (OpenAQ + OpenWeather)
  2. Store in BigQuery
  3. Evaluate current model
  4. Drift detection (performance comparison)
  5. IF drift: Retrain on rolling window
  6. IF new model better: Deploy to MLflow

API Serving (Independent):
User → FastAPI (Cloud Run) → Load model from MLflow → Prediction
```

**Key Point:** Pipeline and API are separate. Pipeline runs monthly (autonomous), API serves on-demand.

---

## Current Status

**✅ Completed (Steps 1-3):**
- Project planning & architecture
- Environment setup (uv, pyproject.toml, GCP project)
- Data collection classes implemented:
  - `OpenAQClient`: Cache + retry, handles 500 errors, sensor filtering
  - `OpenWeatherClient`: Cache + retry, progress bar (tqdm)
- Configuration: `src/config.py` with absolute paths (PROJECT_ROOT, CITIES, dates)
- Data ingestion executed: OpenAQ done, OpenWeather in progress

**🔄 In Progress (Step 4):**
- EDA on OpenAQ data (6 phases planned)

**❌ Not Started (Steps 5-12):**
- Preprocessing pipeline
- Model training with MLflow
- Drift detection implementation
- Prefect orchestration
- FastAPI deployment
- Cloud Run setup

---

## Code Structure
```
breathe_project/
├── src/
│   ├── config.py              # Paths, cities, dates, API keys
│   ├── ingestion/
│   │   ├── openaq.py          # OpenAQClient (done)
│   │   ├── openweather.py     # OpenWeatherClient (done)
│   │   └── utils.py           # save_data_local
│   ├── processing/            # TODO: cleaning, features
│   ├── models/                # TODO: train, evaluate, predict
│   ├── flows/                 # TODO: Prefect flows
│   └── api/                   # TODO: FastAPI
├── data/
│   ├── cache/                 # Temp JSONs (gitignored)
│   ├── raw/                   # CSVs from ingestion
│   └── processed/             # Clean data (TODO)
├── notebooks/
│   └── 01_data_collection.ipynb (done)
├── .env                       # API keys (gitignored)
└── pyproject.toml
```

**Import pattern:**
```python
# After: uv add -e .
from src.ingestion.openaq import OpenAQClient
from src.config import CITIES, START_TRAIN_DATE_STR
```

---

## Key Technical Decisions Made

**1. Data Quality Check Location: Preprocessing (not ingestion)**
- Rationale: Thresholds are data-driven; raw data stays immutable; sensors may degrade over time
- Ingestion: Fetch everything (monitor-grade only)
- Preprocessing: Decide coverage thresholds, handle gaps, aggregate sensors

**2. Paths: Absolute from PROJECT_ROOT**
- `src/config.py` defines all paths (CACHE_DIR, RAW_DIR, etc.)
- Works from notebooks, scripts, Prefect, Cloud Run

**3. Cache Strategy:**
- OpenAQ: 1 JSON per sensor (all dates)
- OpenWeather: 1 JSON per day per city
- Persistent cache (re-runs are instant)

**4. Error Handling:**
- API errors (500, 404) → skip sensor/day, continue
- Retry logic: 3 attempts with wait
- Return `{"results": []}` on failure (no crash)

**5. Retraining Frequency: TBD after EDA**
- Will use volatility analysis + simulation on historical data
- Likely monthly (balance performance vs operational cost)

**6. Model Architecture: TBD after EDA**
- Decision: 1 global model vs 5 city-specific models
- Depends on pattern similarity across cities

---

## Data Collection Details

**OpenAQ Data:**
- Multi-sensor per city (3-20 sensors depending on city)
- Daily aggregated PM2.5 (avg, min, max, quartiles)
- Coverage % per sensor per day
- Format: Long format (sensor_id, date, pm25_avg, coverage, city)

**OpenWeather Data:**
- 1 entry per city per day
- Features: temp_min, temp_max, temp_avg, humidity, precipitation, pressure, wind_speed, wind_direction
- Format: Long format (city, date, features)

**Status:**
- OpenAQ: ✅ Fetched and saved to `data/raw/openaq_all.csv`
- OpenWeather: 🔄 In progress (fetching takes ~1 hour with rate limits)

---

## EDA Plan (Next Step)

**6 Phases (3-4 hours total):**

**Phase 1: Data Quality (30 min)**
- Coverage per city, gap analysis, sensor quality
- Decision: Drop cities/sensors below threshold

**Phase 2: Sensor Aggregation (20 min)**
- Correlation between sensors, variance analysis
- Decision: Aggregation method (mean, median, weighted)

**Phase 3: Distribution (20 min)**
- Histogram, skewness, outliers, transformation candidates
- Decision: log(PM2.5) or not? RMSE vs RMSLE?

**Phase 4: Temporal Patterns (45 min)**
- Trend, seasonality (monthly), volatility
- Retraining frequency simulation
- Decision: Retraining frequency (weekly/monthly/quarterly)

**Phase 5: Autocorrelation (30 min)**
- ACF/PACF plots, identify significant lags
- Decision: Which lags to include (e.g., 1, 7, 30 days)

**Phase 6: Multi-City Comparison (30 min)**
- Magnitude differences, pattern similarity
- Decision: Global model or city-specific models

**Outputs:** `docs/EDA_FINDINGS.md` with all decisions documented

---

## Open Questions (To Resolve in EDA)

**Critical Decisions:**
1. **Target transformation?** Log(PM2.5 + 1) vs raw values
2. **Metric?** RMSE (if normal) vs RMSLE (if log-transformed)
3. **Lag features?** Which lags are predictive (1, 7, 30 days?)
4. **Retraining frequency?** Weekly, monthly, quarterly
5. **Model architecture?** 1 global model + city feature OR 5 separate models
6. **Sensor aggregation?** Simple mean vs weighted by coverage
7. **Quality thresholds?** Min coverage % per sensor (75%? 85%?)
8. **Max gap duration?** Acceptable gap for interpolation (3 days? 7 days?)

**Feature Engineering (After EDA):**
- Temporal features: month_sin/cos, weekday, is_weekend
- Lag features: pm25_lag_1, pm25_lag_7
- Rolling features: pm25_rolling_7d_mean (if ACF suggests)
- Weather features: Current day weather (no lags initially)

---

## Roadmap (12 Steps)

**✅ Done:**
1. Project setup & planning
2. Development environment (uv, GCP)
3. Data collection implementation

**🔄 Current:**
4. EDA (OpenAQ analysis)

**⏭️ Next:**
5. Preprocessing pipeline (clean, aggregate, feature engineering)
6. Baseline model (lag-1 naive predictor)
7. Model training with MLflow (LightGBM, track experiments)
8. Prefect flow: Ingestion + preprocessing
9. Prefect flow: Drift detection + conditional retraining
10. FastAPI deployment (load model, /predict endpoint)
11. Cloud Run deployment (API + Prefect jobs)
12. Documentation & polish

**Estimated Total Time:** 2-3 weeks part-time

---

## GCP Infrastructure

**Project:** `breathe-air-quality`

**BigQuery:**
- Datasets: `pm25_raw`, `pm25_processed` (EU region)

**GCS:**
- Bucket: `gs://breathe-mlflow-gcs-1080515972983/`
- Usage: MLflow artifacts (models, metrics)

**Service Accounts:**
- `breathe-training`: BQ dataEditor + GCS objectAdmin (for training pipeline)
- `breathe-serving`: GCS objectViewer (read-only, for API)

**Cloud Run:**
- Service: FastAPI (always-on, serves predictions)
- Job: Prefect flows (triggered monthly by Cloud Scheduler)

---

## Important Notes

**Scope Philosophy:**
- **Must-have:** Complete MLOps pipeline (ingestion → training → monitoring → retraining → deployment)
- **Nice-to-have:** Simple Streamlit UI (optional), basic monitoring (Prefect logs sufficient)
- **Out of scope:** Complex dashboards (Grafana), advanced drift detection (simple performance comparison OK), hyperparameter optimization at scale

**Portfolio Focus:**
- Demonstrate end-to-end ML lifecycle
- Production patterns (versioning, orchestration, monitoring)
- Realistic trade-offs (zero-cost GCP, manageable scope)
- Clear documentation and decision rationale

**Timeline Management:**
- EDA: 1-2 days (critical for good decisions)
- Modeling: 2-3 days (with MLflow)
- MLOps: 3-4 days (Prefect + deployment)
- Polish: 1-2 days (docs, testing)

---

## Next Immediate Steps

1. **Complete OpenWeather data fetch** (waiting for completion)
2. **Start EDA Phase 1** (data quality assessment)
3. **Progress through EDA Phases 2-6** (make decisions)
4. **Document findings** in `docs/EDA_FINDINGS.md`
5. **Move to preprocessing** (implement cleaning pipeline based on EDA)

---

**This document provides complete context for continuing the project in a fresh conversation.**
