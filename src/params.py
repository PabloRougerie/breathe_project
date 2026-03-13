import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


# =============================================================================
# ENVIRONMENT-SPECIFIC (loaded from .env)
# =============================================================================

GCP_PROJECT          = os.environ.get("GCP_PROJECT")       # may differ between dev/prod
BUCKET_NAME          = os.environ.get("BUCKET_NAME")       # deployment-specific suffix
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI")  # sqlite local vs cloud server
API_OW               = os.environ.get("API_OW")            # secret
API_AQ               = os.environ.get("API_AQ")            # secret


# =============================================================================
# GCP INFRASTRUCTURE
# =============================================================================

GCP_REGION           = "europe-west1"
BQ_REGION            = "EU"
BQ_DATASET_RAW       = "breathe_bq_raw"
BQ_DATASET_PROCESSED = "breathe_bq_processed"
SA_TRAINING          = "breathe-sa-training"
SA_SERVING           = "breathe-sa-serving"
GAR_REPO             = "breathe-gar-repo"


# =============================================================================
# MLFLOW
# =============================================================================

MLFLOW_MODEL_NAME      = "breathe-mlflow-model"
MLFLOW_EXPERIMENT_NAME = "breathe-mlflow-experiment"


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
END_TEST_DATE_STR      = "2025-12-31"
START_TEST_DATE_STR   = "2025-05-01"
END_TEST_DATE_STR     = "2025-05-31"

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

MAX_GAP          = 30
MAX_Q            = 10.0
MIN_COVERAGE_PCT = 70
MIN_BAD_MONTH_PCT = 0.20
HORIZON          = 1
LIMIT            = 1
DEFAULT_APPROACH = "custom"

CUSTOM_SHIFTS = {
    "lag_1": ("pm25_avg", 0),
    "lag_2": ("pm25_avg", 1),
    "lag_3": ("pm25_avg", 2),
    "lag_7": ("pm25_avg", 6),
}

ALL_LAGS_14_SHIFTS = {f"lag_{k + 1}": ("pm25_avg", k) for k in range(14)}
ALL_LAGS_21_SHIFTS = {f"lag_{k + 1}": ("pm25_avg", k) for k in range(21)}

SELECTED_FEATURES = [
    "lag_1",
    "city",
    "precipitation",
    "temp_max",
    "lag_avg_14",
    "wind_speed",
    "pressure",
    "wind_direction",
    "lag_3",
    "humidity"
]


# =============================================================================
# MODEL
# =============================================================================

BEST_PARAMS = {
    "learning_rate":    0.039215086373645916,
    "n_estimators":     365,
    "num_leaves":       192,
    "max_depth":        3,
    "min_child_samples": 14,
    "reg_alpha":        1.0104102789955898,
    "reg_lambda":       0.00016444594773119733,
    "subsample":        0.5287082281151364,
    "colsample_bytree": 0.9863936278891989,
    "subsample_freq":   1,
    "random_state":     273,
    "verbose":          -1,
}

DRIFT_THRESHOLD       = 0.25
IMPROVEMENT_THRESHOLD = 0.05
