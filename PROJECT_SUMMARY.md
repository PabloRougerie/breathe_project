# Project summary — Breathe

This document expands on the [README](README.md): what problem we solve, what the data and modeling work established, and how the system is operated in production.

---

## 1. Business / learning objective

The **primary goal** of the project is an **MLOps demonstration on GCP**: ingestion → warehouse → training → registry → scheduled evaluation → drift-aware retrain → monitoring UI. The PM2.5 use case is a **realistic vehicle** for that stack (APIs, time series, batch scoring).

**Prediction task:** daily **PM2.5 at J+1** (one step ahead), **per city**, from historical PM2.5 and same-day weather.

**Cities in scope (6):** Paris, Lyon, New York, London, Berlin, Rome — defined in `src/params.py` (`CITIES`).

---

## 2. Data science

### 2.1 Sources and EDA

- **Air quality:** OpenAQ (sensor-level PM2.5, aggregated to city level in preprocessing).
- **Weather:** OpenWeather (daily aggregates aligned to the air-quality dates).

**Quality and city selection** (see also `eda.md`):

- Twelve candidate cities were screened over roughly **May 2023 – April 2025**.
- Sensors were filtered using rules encoded in preprocessing (`MAX_GAP`, `MAX_Q`, `MIN_COVERAGE_PCT`, `MIN_BAD_MONTH_PCT` in `params.py`): e.g. long gaps, gap distribution, months with low coverage.
- **Six cities retained**; others dropped for unstable or insufficient coverage.
- **Negative PM2.5** values are removed (not imputed) to keep gap logic trustworthy.
- A **coverage dip in spring 2024** on some European sensors was treated as a likely **upstream OpenAQ reporting issue**, not automatic sensor rejection.

### 2.2 Preprocessing approach (`src/preprocess/`)

- **Merge** air quality (averaged per city/date after optional sensor filtering) with weather.
- **Small gaps:** imputation with a limited linear interpolation (`LIMIT` in config).
- **Target:** `pm25` at **horizon 1** (`HORIZON = 1`), after **`log1p`** transform (`target_transform` in the pipeline). **`pm25_avg` is log-transformed before lag features** so lags live in the same space as the target — important for consistent RMSE interpretation.
- **Features (current set, k=15):** listed in `SELECTED_FEATURES` — mixes short lags (`lag_1`, `lag_3`, `lag_7`), rolling-style signals (`lag_avg_14`, `week_std`), weather fields, calendar (`month_cos`), and `temp_gradient` (`temp_max - temp_min`). `city` is categorical for the model pipeline.
- **Train vs eval mode:** `mode="train"` runs **sensor filtering**; `mode="eval"` skips it so short evaluation windows are not emptied by monthly coverage rules (`preproc_pipeline.py`).

**Periodic batch warmup:** the periodic flow preprocesses from **`batch_start − 14 days`** through `batch_end` (constant `PREPROCESS_LAG_WARMUP_DAYS` in `periodic.py`) so rolling/lag features are defined at the first day of the batch, while ingestion still targets the strict batch window. Raw history is assumed available in BigQuery from bootstrap or prior loads.

### 2.3 Model choice and tuning

- **Algorithm:** **LightGBM** inside a **sklearn pipeline** with **OneHotEncoder** for `city` (`src/models/train.py` / pipeline setup).
- **Hyperparameters:** `BEST_PARAMS` in `params.py` from **Optuna** search (README legacy note: ~1000 trials, time-series CV with gap — details live in notebooks / experiment logs).
- **Reported CV performance (reference):** RMSE ≈ **0.236** on the **log1p target** (same metric family as production eval).

### 2.4 Baselines

Implemented in `src/models/baseline.py` (RMSE in **log space**, consistent with transformed target):

- **Persistence** — predict from `lag_1`.
- **Extrapolation** — linear step from `lag_1`, `lag_2`.
- **Average** — mean of `lag_1`–`lag_3`.

Cross-validation helper aggregates scores across folds for comparison with the GBM.

### 2.5 Metrics and drift observations

- **Training / eval RMSE** uses `sklearn.metrics.root_mean_squared_error` on the **log1p target** (`evaluate.py`).
- **Operational drift** compares the champion’s score on a **reference** window vs a **new batch** (`self_compare` in `evaluate.py`).
- **Seasonality:** empirical analysis noted **better scores in summer than winter** for similar features — consistent with harder winter PM2.5 dynamics not fully captured by daily weather alone (e.g. thermal inversions). The **MLOps loop** still behaves as designed (drift flag, retrain, promotion rules); the **ceiling** is partly a **data / feature** limit, not only a code bug.

---

## 3. MLOps

### 3.1 Objectives

- **Reproducible** training and evaluation with **MLflow** (params, metrics, artifacts, **aliases** `champion` / `challenger`).
- **Central data** in **BigQuery** (raw + processed + monitoring).
- **Resumable ingestion** via **GCS** JSON cache.
- **Automatable** execution via **Cloud Run Jobs** (container image built from `docker/package/Dockerfile`).
- **Human-readable monitoring** via **Streamlit** reading monitoring tables.

### 3.2 Architecture (logical)

```
OpenAQ / OpenWeather
        ↓
  GCS cache (JSON)  ←→  optional local cache for dev
        ↓
 BigQuery raw (airqual, weather)
        ↓
 preprocessing_pipeline
        ↓
 BigQuery processed
        ↓
 MLflow (registry) ← training / eval runs
        ↓
 BigQuery monitoring (batches, models, predictions)
        ↓
 Streamlit dashboard
```

**MLflow server** (separate image under `docker/mlflow/`) exposes the tracking UI; **artifact root** is configured to **GCS** in that Dockerfile. **Backend store** in the checked-in MLflow Dockerfile is **SQLite under `/tmp`** (suitable for demos; for durable production you typically mount a volume or use a managed DB — see deployment notes).

### 3.3 Workflows

**Bootstrap** (`entrypoint_bootstrap.py`):

1. `bootstrap_train_masterflow` — ingest train range → preprocess → train → register (**first** model → `champion`, later → `challenger` logic in `model_pipeline.py`).
2. `bootstrap_eval_masterflow` — ingest test window → preprocess → evaluate champion → write **monitoring** model metadata.

**Periodic** (`entrypoint_periodic.py`):

1. Ingest **current batch** (strict `batch_start`–`batch_end`).
2. Preprocess with **warmup** start date (batch_start − 14 days) for eval path.
3. Evaluate **champion** on the batch; **self_compare** vs `reference_rmse` tag.
4. If drift: preprocess **train** window → train **challenger** → evaluate vs champion on batch → **cross_compare** → optional **promotion** (`registry.py`).
5. Log **batch** row, load processed batch, **predict** with champion, **log predictions** to BigQuery.

**Batch calendar:** `BATCH_SCHEDULE` in `periodic.py` (explicit windows; cron is expected to advance `batch_num` or equivalent — automation details outside this doc).

### 3.4 Drift and promotion thresholds

Configured in `params.py` and used as **percentages** in `evaluate.py`:

- **`DRIFT_THRESHOLD` (default 10):** drift if `score_new >= score_ref * (1 + 10/100)` — i.e. **10% relative degradation**.
- **`IMPROVEMENT_THRESHOLD` (default 1):** promote challenger if `score_new <= score_old * (1 - 1/100)` — i.e. **at least ~1% relative improvement** on the batch.

### 3.5 Monitoring schema (BigQuery)

Dataset name from `BQ_DATASET_MONITORING` (default `breathe_bq_monitoring`):

- **`batches`** — one row per periodic run (dates, RMSEs, drift flag, promotion flag, versions).
- **`models`** — registered model metadata and aliases.
- **`predictions`** — city-level `y_true` / `y_pred` (stored in **original scale**, µg/m³, after `expm1` in the flow) for dashboard curves; **RMSLE** in Streamlit is recomputed on `log1p` for alignment with MLflow.

`MonitoringClient` in `src/utils.py` implements upserts / deletes for those tables.

---

## 4. Results (summary)

- **Model:** strong gain vs simple baselines; tuned LightGBM with fixed 15-feature set.
- **Ops:** full path from APIs to BQ to MLflow to monitoring tables; periodic path exercises retrain and promotion logic.
- **Known limitation:** winter / complex pollution episodes remain hard; the pipeline surfaces drift and retrains, but **cannot invent** missing physical signal.

For **exact numbers** and charts, rely on MLflow UI, Streamlit, and notebooks under `notebooks/`.
