import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


# =============================================================================
# ENVIRONMENT-SPECIFIC (loaded from .env)
# =============================================================================

GCP_PROJECT          = os.environ.get("GCP_PROJECT")
BUCKET_NAME          = os.environ.get("BUCKET_NAME")
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI")
API_OW               = os.environ.get("API_OW")
API_AQ               = os.environ.get("API_AQ")
GCP_REGION           = os.environ.get("GCP_REGION", "europe-west1")
BQ_REGION            = os.environ.get("BQ_REGION", "EU")
SA_TRAINING          = os.environ.get("SA_TRAINING", "breathe-sa-training")
SA_SERVING           = os.environ.get("SA_SERVING", "breathe-sa-serving")


# =============================================================================
# GCP INFRASTRUCTURE
# =============================================================================

# These need to be available in shell/CI too (bq/gcloud), so they live in .env.
# Keep defaults for convenience when env vars are missing.
BQ_DATASET_RAW        = os.environ.get("BQ_DATASET_RAW", "breathe_bq_raw")
BQ_DATASET_PROCESSED  = os.environ.get("BQ_DATASET_PROCESSED", "breathe_bq_processed")
BQ_DATASET_MONITORING = os.environ.get("BQ_DATASET_MONITORING", "breathe_bq_monitoring")
GAR_REPO              = os.environ.get("GAR_REPO", "breathe-gar-repo")


# =============================================================================
# MLFLOW
# =============================================================================

MLFLOW_MODEL_NAME      = os.environ.get("MLFLOW_MODEL_NAME", "breathe-mlflow-model")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "breathe-mlflow-experiment")


# =============================================================================
# ORCHESTRATION (Prefect + Cloud Run + Scheduler)
# =============================================================================

PREFECT_FLOW_NAME  = "breathe-prefect-flow"

# Cloud Run Job
MEMORY      = ""
CPU         = ""
TIMEOUT     = ""
MAX_RETRIES = ""
JOB_NAME    = ""

# Cloud Scheduler
SCHEDULER_NAME     = "prefect-cron"
SCHEDULER_LOCATION = "europe-west1"
CRON_SCHEDULE      = ""
TIMEZONE           = "Europe/Paris"

# Serving API (Cloud Run)
API_MEMORY        = ""
API_CPU           = ""
API_TIMEOUT       = ""
API_MIN_INSTANCES = "1"
API_MAX_INSTANCES = ""


# =============================================================================
# PROJECT SCOPE
# =============================================================================

START_PROJECT_DATE_STR = "2023-05-01"
END_PROJECT_DATE_STR   = "2025-12-31"
START_TRAIN_DATE_STR   = "2023-05-01"
END_TRAIN_DATE_STR     = "2025-04-30"
START_TEST_DATE_STR    = "2025-05-01"
END_TEST_DATE_STR      = "2025-06-11"


CITIES = {
    "Paris":    {"lat": 48.8622, "lon":  2.3470},
    "Lyon":     {"lat": 45.7267, "lon":  4.8275},
    "New York": {"lat": 40.8259, "lon": -73.9508},
    "London":   {"lat": 51.5045, "lon": -0.1363},
    "Berlin":   {"lat": 52.4990, "lon": 13.4437},
    "Rome":     {"lat": 41.9028, "lon": 12.4964},
}


# =============================================================================
# LOCAL PATHS
# =============================================================================

CACHE_DIR         = Path(__file__).parent.parent / "data" / "cache"
LOCAL_STORAGE_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# PREPROCESSING
# =============================================================================

MAX_GAP           = 30
MAX_Q             = 10.0
MIN_COVERAGE_PCT  = 70
MIN_BAD_MONTH_PCT = 0.20
HORIZON           = 1
LIMIT             = 1
DEFAULT_APPROACH  = "custom"

CUSTOM_SHIFTS = {
    "lag_1": ("pm25_avg", 0),
    "lag_2": ("pm25_avg", 1),
    "lag_3": ("pm25_avg", 2),
    "lag_7": ("pm25_avg", 6),
}

ALL_LAGS_14_SHIFTS = {f"lag_{k + 1}": ("pm25_avg", k) for k in range(14)}
ALL_LAGS_21_SHIFTS = {f"lag_{k + 1}": ("pm25_avg", k) for k in range(21)}

# Top 15 from feature importance (saisonnalité: month_cos ; gradient thermique: temp_gradient)
SELECTED_FEATURES = [
    "lag_1",
    "lag_3",
    "lag_7",
    "city",
    "wind_speed",
    "precipitation",
    "lag_avg_14",
    "week_std",
    "wind_direction",
    "pressure",
    "temp_max",
    "humidity",
    "temp_min",
    "month_cos",
    "temp_gradient",
]


# =============================================================================
# MODEL
# =============================================================================

# Optuna best (k=15 features), score 0.2358
BEST_PARAMS = {
    "learning_rate":     0.029430170491189643,
    "n_estimators":      670,
    "num_leaves":        44,
    "max_depth":         12,
    "min_child_samples": 38,
    "reg_alpha":         1.0998029678527025e-05,
    "reg_lambda":        1.282458658923553e-07,
    "subsample":         0.9143332144889533,
    "colsample_bytree":  0.9767977854994194,
    "subsample_freq":    1,
    "random_state":      273,
    "verbose":           -1,
}

DRIFT_THRESHOLD       = 10
IMPROVEMENT_THRESHOLD = 1
