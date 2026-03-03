import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


# ========== GCP Core Configuration ==========
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_REGION = os.environ.get("BQ_REGION")

# ========== BigQuery Configuration ==========
BQ_DATASET_RAW = os.environ.get("BQ_DATASET_RAW")
BQ_DATASET_PROCESSED = os.environ.get("BQ_DATASET_PROCESSED")

# ========== GCS Configuration ==========
BUCKET_NAME = os.environ.get("BUCKET_NAME")

# ========== ML Logic Configuration ==========
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
MODEL_SAVE = os.environ.get("MODEL_SAVE")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
USE_CACHE = os.environ.get("USE_CACHE")

# ========== Service Accounts ==========
SA_TRAINING = os.environ.get("SA_TRAINING")
SA_SERVING = os.environ.get("SA_SERVING")

# ========== Prefect Configuration ==========
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")

# ========== Cloud Run Job Configuration ==========
MEMORY = os.environ.get("MEMORY")
CPU = os.environ.get("CPU")
TIMEOUT = os.environ.get("TIMEOUT")
MAX_RETRIES = os.environ.get("MAX_RETRIES")
JOB_NAME = os.environ.get("JOB_NAME")

# ========== Cloud Scheduler Configuration ==========
SCHEDULER_NAME = os.environ.get("SCHEDULER_NAME")
SCHEDULER_LOCATION = os.environ.get("SCHEDULER_LOCATION")
CRON_SCHEDULE = os.environ.get("CRON_SCHEDULE")
TIMEZONE = os.environ.get("TIMEZONE")

# ========== API Configuration ==========
API_MEMORY = os.environ.get("API_MEMORY")
API_CPU = os.environ.get("API_CPU")
API_TIMEOUT = os.environ.get("API_TIMEOUT")
API_MIN_INSTANCES = os.environ.get("API_MIN_INSTANCES")
API_MAX_INSTANCES = os.environ.get("API_MAX_INSTANCES")
API_OW = os.environ.get("API_OW")
API_AQ = os.environ.get("API_AQ")

# ========== Artifact Registry ==========
GAR_REPO = os.environ.get("GAR_REPO")

# ========== PROJECT DATES ==========
START_TRAIN_DATE_STR = os.environ.get("START_TRAIN_DATE_STR")
START_PROJECT_DATE_STR = os.environ.get("START_PROJECT_DATE_STR")
END_TRAIN_DATE_STR = os.environ.get("END_TRAIN_DATE_STR")
END_PROJECT_DATE_STR = os.environ.get("END_PROJECT_DATE_STR")


# ========== Cities ==========
CITIES = {
    "Paris": {"lat": 48.8622, "lon": 2.3470},        # Les Halles
    "Lyon": {"lat": 45.7267, "lon": 4.8275},          # Gerland
    "New York": {"lat": 40.8259, "lon": -73.9508},    # St Nicholas Terrace, Manhattanville
    "London": {"lat": 51.5045, "lon": -0.1363},       # St James, Central London
    "Berlin": {"lat": 52.4990, "lon": 13.4437},       # Near Görlitzer Park
    "Rome": {"lat": 41.9028, "lon": 12.4964},
}


# ========== Preprocessing constants ==========
MAX_GAP = 30
MAX_Q = 10.0
MIN_COVERAGE_PCT = 70
MIN_BAD_MONTH_PCT = 0.20
HORIZON = 1
LIMIT = 1
DEFAULT_APPROACH = "custom"

CUSTOM_SHIFTS = {"lag_1": ("pm25_avg", 0),
                 "lag_2": ("pm25_avg", 1),
                 "lag_3": ("pm25_avg", 2),
                 "lag_7": ("pm25_avg", 6)


}

ALL_LAGS_14_SHIFTS = {f"lag_{k + 1}" : ("pm25_avg", k) for k in range(14)}
ALL_LAGS_21_SHIFTS = {f"lag_{k + 1}" : ("pm25_avg", k) for k in range(21)}


SELECTED_FEATURES = ['lag_1',
 'city',
 'precipitation',
 'temp_max',
 'lag_avg_14',
 'wind_speed',
 'pressure',
 'wind_direction',
 'lag_3',
 'humidity']

BEST_PARAMS = {
  "learning_rate": 0.039215086373645916,
  "n_estimators": 365,
  "num_leaves": 192,
  "max_depth": 3,
  "min_child_samples": 14,
  "reg_alpha": 1.0104102789955898,
  "reg_lambda": 0.00016444594773119733,
  "subsample": 0.5287082281151364,
  "colsample_bytree": 0.9863936278891989,
  "subsample_freq": 1,
  "random_state": 273,
  "verbose": -1
}

DRIFT_THRESHOLD = 0.25
IMPROVEMENT_THRESHOLD = 0.05
