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
START_TRAIN_DATE_STR= os.environ.get("START_TRAIN_DATE_STR")
START_PROJECT_DATE_STR= os.environ.get("START_PROJECT_DATE_STR")
END_TRAIN_DATE_STR= os.environ.get("END_TRAIN_DATE_STR")
END_PROJECT_DATE_STR= os.environ.get("END_PROJECT_DATE_STR")


CITIES = {
    "Paris": {'lat': 48.8622, 'lon': 2.3470},           # Les Halles
    "Lyon": {'lat': 45.7267, 'lon': 4.8275},            # Gerland
    "New York": {"lat": 40.8259, "lon": -73.9508},      # St Nicholas Terrace, Manhattanville
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437},  # Original coordinates
    "Delhi": {"lat": 28.6070, "lon": 77.2456},          # Near Sher Shah Road
    "London": {"lat": 51.5045, "lon": -0.1363},         # St James, Central London # Near Largo da Carioca
    "Berlin": {"lat": 52.4990, "lon": 13.4437},         # Near Görlitzer Park
    "Rome": {"lat": 41.9028, "lon": 12.4964},
    "Barcelona": {"lat": 41.3954, "lon": 2.1161},       # Near Real Club de Polo         # Near Kalina Kurla Road   # Avenida Municipio Libre, Colonia Letrán Valle
    "Santiago": {"lat": -33.4192, "lon": -70.7417}      # Cerro Navia
}
