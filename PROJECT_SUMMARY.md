# Project Summary — Breathe

---

## 1. Objectives

### MLOps focus — model deployment and monitoring

The core of this project is an autonomous post-deployment monitoring architecture: evaluate a production model on each new data batch, detect performance degradation, retrain a challenger model when drift is confirmed, and promote it if it improves on the current champion. The system runs end-to-end on GCP with no manual intervention required once deployed.

### Forecasting task

**Predict next-day PM2.5 (µg/m³) per city** from historical air quality and same-day weather data, for six cities: Paris, Lyon, London, Berlin, New York, Rome.

PM2.5 was chosen as the use case because it involves real APIs, messy real-world data (sensor gaps, coverage issues), meaningful seasonality, and a realistic reason to monitor for drift over time.

---

## 2. Data

### Sources

- **OpenAQ** — sensor-level PM2.5 measurements, aggregated to city/day level during preprocessing.
- **OpenWeather** — daily weather aggregates (temperature, wind, precipitation, pressure, humidity).

### Date range

May 2023 → February 2026 (training: May 2023 → April 2025 / test set: May 2025 → June 2025 / 6 monitoring batches: June 2025 → February 2026).

### Ingestion and caching

A custom client handles each data source: API calls, response parsing, city-level aggregation, and basic quality checks before storage. Each result is cached as a JSON file on GCS (`{city}/{source}/{date}.json`). On re-run, only missing dates trigger new API calls — the pipeline is fully idempotent. Uploads to BigQuery use a DELETE + WRITE_APPEND pattern so any stage can be safely re-run without duplicating data.

### City selection and sensor filtering

Twelve candidate cities were evaluated. Six were retained based on data quality criteria: sensor coverage rate over the full period, gap length and distribution, and months with abnormally low or noisy readings. Cities with persistent coverage gaps, unstable sensors, or insufficient history were dropped. A coverage dip on some European sensors in spring 2024 was treated as an upstream OpenAQ reporting issue and not used as grounds for rejection.

---

## 3. Preprocessing

The preprocessing pipeline runs in two modes:

- **`train` mode** — applies sensor-level filtering before aggregating. Used for bootstrap training.
- **`eval` mode** — skips sensor filtering. Used for periodic batch evaluation (short windows would be emptied by monthly coverage rules if filtering were applied).

### Target

`pm25_avg` at horizon J+1, log-transformed with `log1p`. **The log transform is applied before generating lag features**, so all lags live in the same space as the target. This is important for RMSE interpretation: all metrics reported (in MLflow and on the dashboard) are in log space, making them equivalent to RMSLE.

### Why LightGBM with hand-crafted features rather than a sequence model

A direct approach for time series would be to feed the full PM2.5 history into a recurrent or transformer model. We chose LightGBM with explicit lag features instead for three reasons: simpler deployment and monitoring (no GPU, no sequence padding), faster iteration during feature selection, and better interpretability of drift causes. The tradeoff is that we must engineer the temporal structure manually — which is where the feature selection work below matters.

### Feature engineering

Lag features were first identified from ACF analysis of the PM2.5 series, which showed significant autocorrelation at lags 1, 3, and 7 days, and a weaker but persistent signal up to ~14 days. From that starting point, 15 features were selected via permutation importance (n_repeats=400) from a larger candidate set:

| Group | Features |
|---|---|
| PM2.5 lags | `lag_1`, `lag_3`, `lag_7`, `lag_avg_14`, `week_std` |
| Weather | `wind_speed`, `wind_direction`, `precipitation`, `pressure`, `temp_max`, `temp_min`, `humidity` |
| Derived | `temp_gradient` (temp_max − temp_min) |
| Calendar | `month_cos` |
| Categorical | `city` (OHE in the sklearn pipeline) |

---

## 4. Model

**Algorithm:** LightGBM inside a sklearn `Pipeline` with `OneHotEncoder` for `city`.

**Hyperparameters:** tuned with Optuna (~1000 trials, `TPESampler`, `TimeSeriesSplit` with gap=42 and test_size=180 to respect temporal order and avoid leakage).

**Performance:**

| Model | RMSE (log scale) |
|---|---|
| Persistence baseline | 0.650 |
| Extrapolation baseline | ~0.55 |
| Tuned LightGBM | **0.265** (initial champion) |

The gap between LightGBM and the baselines is substantial. The model captures the main PM2.5 signal well in summer months (RMSE ~0.20–0.30) but degrades in winter (RMSE ~0.40–0.43). This is consistent with harder winter dynamics — thermal inversions, heating season — that are not captured by daily weather features. Further improvement would require higher-frequency meteorological data or additional features. This is a data ceiling, not a modeling issue.

---

## 5. MLOps architecture

### Overview

```
OpenAQ / OpenWeather APIs
        ↓
  GCS cache (JSON)
        ↓
  BigQuery raw (airqual, weather)
        ↓
  Preprocessing pipeline
        ↓
  BigQuery processed
        ↓
  MLflow (registry + experiment tracking)
        ↓
  BigQuery monitoring (batches, models, predictions)
        ↓
  Streamlit dashboard
```

### Components

| Component | Technology | Notes |
|---|---|---|
| Orchestration | Prefect 3 (tasks + flows) | Provides task-level observability and logging; execution is triggered externally |
| Flow execution | Cloud Run Job | Ephemeral container, triggered on demand or by Cloud Scheduler |
| Model registry | MLflow 2.x | Aliases: `@champion`, `@challenger`; archived versions tagged |
| Artifact store | GCS | MLflow model artifacts stored at `gs://breathe-mlflow-gcs-*/artifacts` |
| Metadata store | BigQuery | Monitoring tables mirror MLflow state (see below) |
| MLflow server | Cloud Run Service | Always-on service; SQLite backend in `/tmp` — see limitations |
| Monitoring UI | Streamlit Cloud | Reads BigQuery monitoring tables directly; no API layer |
| Flow logging | Prefect Cloud | UI only — Cloud Run Job sends logs to Prefect Cloud via API key |

### Why two Docker images

The project uses two separate images:

- `docker/mlflow/Dockerfile` — minimal image running the MLflow tracking server. Deployed as a **Cloud Run Service** (always-on). Handles artifact storage to GCS.
- `docker/package/Dockerfile` — full project image (installs `pyproject.toml`, copies `src/`). Deployed as a **Cloud Run Job** (ephemeral, triggered per batch). Connects to the MLflow service via `MLFLOW_TRACKING_URI`.

The separation is intentional: MLflow needs to be reachable at all times, while the training/eval job runs periodically and should not carry MLflow server dependencies.

### Prefect as observability layer only

Prefect Cloud is used solely for its UI — flow run history, task states, logs. It does not orchestrate execution; Cloud Scheduler does. This avoids the complexity of Prefect workers and work pools while keeping structured logging visible in a proper UI.

### Workflows

**Bootstrap** (one-time, `entrypoint_bootstrap.py`):
1. Ingest training data (May 2023 → April 2025) → BQ raw
2. Preprocess → BQ processed
3. Train LightGBM → register as `@champion` in MLflow
4. Ingest + preprocess test set (May → June 2025)
5. Evaluate champion on test set → set `reference_rmse` tag on model version
6. Write initial model metadata to BQ monitoring (`models` table)

**Periodic** (`entrypoint_periodic.py`, driven by `batch_num`):
1. Ingest a window starting 14 days before `batch_start` to ensure lag features are fully defined at the start of the evaluation window. After preprocessing, only the strict `batch_start → batch_end` interval is used for evaluation — no data leakage, since using lag values from the training period to compute features at the boundary is standard practice
2. Evaluate `@champion` on the batch → `self_compare` vs `reference_rmse` tag
3. **If drift detected:** preprocess training window → train `@challenger` → evaluate on batch → `cross_compare` → promote if improved
4. Log batch row to `batches` table, predict with current `@champion`, log to `predictions` table

### Drift and promotion logic

Implemented in `src/models/evaluate.py`, configured in `src/params.py`:

- **Drift threshold (10%):** drift flagged if `score_batch >= reference_rmse × 1.10`
- **Promotion threshold (1%):** challenger promoted if `score_challenger <= score_champion × 0.99`

### BigQuery monitoring schema

Three tables in `breathe_bq_monitoring`:

**`batches`** — one row per periodic run:
```
batch_start | batch_end | champion_version | champion_rmse
drift_detected | challenger_version | rmse_challenger | promotion_applied | run_date
```

**`models`** — one row per registered model version (upserted on alias change):
```
model_version | alias | reference_rmse | train_start | train_end | eval_start | eval_end
```

**`predictions`** — city-level daily predictions (stored in original µg/m³ scale after `expm1`):
```
date | city | y_true | y_pred | model_version
```

---

## 6. Architectural decisions and workarounds

### MLflow persistence — the SQLite/GCSFuse problem

The natural approach for MLflow persistence on Cloud Run would be to mount the GCS bucket as a filesystem volume and store `mlflow.db` there. This fails: GCSFuse simulates a POSIX filesystem but does not support SQLite's random-write access patterns, causing the MLflow container to crash at startup.

**Workaround:** SQLite lives in `/tmp` — local to the container, lost on restart. Model artifacts are unaffected (they go directly to GCS via the artifact store URI). However, losing the SQLite database means losing the MLflow model registry, including the `@champion` alias. Once the container restarts, the periodic flow can no longer fetch the production model by alias — making a true monthly automation impossible in this setup.

For this demo, all 6 batches were run in sequence while the MLflow container was still live, so the registry stayed intact long enough to populate all monitoring tables. The BigQuery monitoring tables — which mirror all critical metadata — mean the dashboard remains intact regardless of MLflow state, and would allow reconstituting the registry from scratch if needed.

**In a real deployment:** the standard fix is Cloud SQL (PostgreSQL) as the MLflow backend store, connected from Cloud Run via Cloud SQL Auth Proxy. Alternatively, managed ML platforms (Databricks, Vertex AI Experiments) solve this natively without extra infrastructure.

### MLflow version pinning

MLflow 3.x launches background Huey workers for GenAI/LLM features at startup, causing an out-of-memory (OOM) crash on Cloud Run even at 1Gi. Pinned to `mlflow>=2.18.0,<3.0.0` in both `docker/mlflow/Dockerfile` and `pyproject.toml`.

---

## 7. Results (6-batch simulation, Jun 2025 → Feb 2026)

| Batch | Period | Champion RMSE | Drift | Promoted | Challenger RMSE |
|---|---|---|---|---|---|
| 1 | Jun–Jul 2025 | 0.265 | NO | — | — |
| 2 | Jul–Sep 2025 | 0.265 | NO | — | — |
| 3 | Sep–Oct 2025 | 0.300 | YES | NO | 0.299 (+0.5%) |
| 4 | Oct–Nov 2025 | 0.300 | YES | YES | 0.397 (+1.7%) → promoted |
| 5 | Nov 2025–Jan 2026 | 0.397 | NO | — | — |
| 6 | Jan–Feb 2026 | 0.432 | NO | — | — |

The winter degradation is structural and consistent with what the feature set can capture. The drift detection loop correctly identifies it; retraining partially compensates but cannot overcome the signal-to-noise ceiling of daily weather features in cold months.
