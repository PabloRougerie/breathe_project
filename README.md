# Breathe — PM2.5 forecasting & MLOps on GCP

End-to-end MLOps pipeline for next-day PM2.5 forecasting across six cities (Paris, Lyon, London, Berlin, New York, Rome). The primary goal is architectural: demonstrate a production-grade monitoring loop with automated drift detection, challenger retraining, and model promotion. 

**Live dashboard:** [Streamlit UI](https://breatheproject-n9uhfau2pdjhxrymh4nq7x.streamlit.app) — drift monitoring log + city-level prediction curves.

---

## What this project demonstrates

- **Full data pipeline** — OpenAQ + OpenWeather APIs → idempotent data ingestion pipeline → BigQuery (raw, processed, monitoring)
- **MLflow model registry** — champion / challenger aliases, reference RMSE tags, artifact storage on GCS
- **Drift-aware periodic loop** — evaluate champion on each new batch, retrain challenger if drift detected, promote if challenger improves
- **Containerised execution** — two Cloud Run images (MLflow server + Prefect flows), triggered manually or via Cloud Scheduler
- **Monitoring UI** — Streamlit reads BigQuery monitoring tables; no API layer needed

---

## Results (6-batch simulation, Jun 2025 → Feb 2026)

| Metric | Value |
|---|---|
| Initial champion RMSE (log scale) | 0.265 |
| Persistence baseline RMSE | 0.650 |
| Drift events detected | 4 / 6 batches |
| Promotions triggered | 2 |
| Avg improvement at promotion | ~+1.5% RMSE reduction |

Performance degrades in winter batches (RMSE ~0.43 vs ~0.30 in summer) — consistent with harder PM2.5 dynamics (thermal inversions, heating season) that daily weather features can't fully capture. The drift detection and retraining loop works as designed; the ceiling is a data/feature limit, not a code issue.

---

## Stack

Python 3.13 · LightGBM · scikit-learn · MLflow 2.x · Prefect 3 · Google Cloud (BigQuery, GCS, Artifact Registry, Cloud Run) · Streamlit · OpenAQ / OpenWeather APIs

---

## Repository layout

```
├── docker/
│   ├── mlflow/Dockerfile       # MLflow tracking server (deployed as Cloud Run Service)
│   └── package/Dockerfile      # Prefect flows + entrypoints (deployed as Cloud Run Job)
├── src/
│   ├── entrypoint_bootstrap.py # One-time: train initial champion + eval on test set
│   ├── entrypoint_periodic.py  # Periodic: ingest batch → eval → optional retrain/promote
│   ├── flows/
│   │   ├── bootstrap.py
│   │   └── periodic.py
│   ├── ingestion/              # OpenAQ + OpenWeather clients with GCS cache
│   ├── preprocess/             # Cleaning, feature engineering, pipeline
│   ├── models/                 # LightGBM pipeline, MLflow logging, registry helpers
│   ├── utils.py                # BQ storage clients, MonitoringClient
│   └── params.py               # All constants: cities, features, thresholds, BQ datasets
├── ui/
│   └── streamlit.py            # Monitoring dashboard (deployed on Streamlit Cloud)
├── notebooks/                  # EDA, model experiments
├── job.env.yaml                # Cloud Run env vars — not committed (secrets)
├── PROJECT_SUMMARY.md          # Full write-up: data science + MLOps design decisions
└── DEPLOYMENT_GUIDE.md         # Step-by-step commands to redeploy from scratch
```

---

## Quick start (local)

```bash
# Install
uv sync   # or: pip install -e .

# Set environment variables
cp .env.example .env   # fill in GCP_PROJECT, BUCKET_NAME, API keys, MLFLOW_TRACKING_URI
source .env

# Run bootstrap (initial training + eval)
PYTHONPATH=. python src/entrypoint_bootstrap.py

# Run a periodic batch
PYTHONPATH=. python src/entrypoint_periodic.py --batch_num 1

# Streamlit (needs GCP credentials and BQ access)
PYTHONPATH=. streamlit run ui/streamlit.py
```

For full deployment on GCP, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).  
For data science and architectural decisions, see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).
