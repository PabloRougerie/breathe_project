import os

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

# ========== Artifact Registry ==========
GAR_REPO = os.environ.get("GAR_REPO")
