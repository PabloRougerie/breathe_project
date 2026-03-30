# Deployment Guide — Breathe

This guide covers everything needed to deploy the full system from scratch on GCP. The overall sequence is:

1. **Infrastructure** — GCP APIs, GCS bucket, Artifact Registry, service accounts, BigQuery datasets
2. **MLflow server** — build and deploy the tracking server as a Cloud Run Service
3. **Cloud Run Jobs** — build and deploy the Prefect flows (bootstrap + periodic)
4. **Streamlit UI** — local run and Streamlit Cloud deployment
5. **Cloud Scheduler** — optional, for true automation

**Prerequisites:** Docker Desktop running, `gcloud` CLI installed and authenticated (`gcloud auth login`), billing-enabled GCP project.

---

## Phase 0 — Shell variables

Set these once at the start of your session. All commands below reference them:

```bash
export GCP_PROJECT="breathe-air-quality"
export GCP_REGION="europe-west1"
export BQ_REGION="EU"
export BUCKET_NAME="breathe-mlflow-gcs-1080515972983"
export SA_TRAINING="breathe-sa-training"
export SA_SERVING="breathe-sa-serving"
export GAR_REPO="breathe"
```

Also load your local environment file for Python scripts:
```bash
source .env
```

---

## Phase 1 — GCP infrastructure (one-time)

### 1.1 Enable required GCP APIs

Before creating any resource, the relevant GCP services need to be activated on the project:

```bash
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    bigquery.googleapis.com \
    cloudscheduler.googleapis.com \
    --project $GCP_PROJECT
```

### 1.2 Create GCS bucket

This bucket serves two purposes: ingestion cache (one JSON file per city/date/source) and MLflow artifact storage (serialised models):

```bash
gcloud storage buckets create gs://$BUCKET_NAME \
    --location=$BQ_REGION \
    --project=$GCP_PROJECT
```

### 1.3 Create Artifact Registry repository

Docker images cannot be pushed directly to Cloud Run — they need to be stored in a registry first. We use Google Artifact Registry:

```bash
gcloud artifacts repositories create $GAR_REPO \
    --repository-format=docker \
    --location=$GCP_REGION \
    --project=$GCP_PROJECT
```

Then configure Docker to authenticate against this registry using your gcloud credentials:

```bash
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
```

### 1.4 Create service accounts

Two service accounts are used, following least-privilege principles:

**Training SA** — attached to Cloud Run Jobs. Needs read/write access to BigQuery (storing raw, processed, and monitoring data) and GCS (ingestion cache and MLflow artifacts):

```bash
gcloud iam service-accounts create $SA_TRAINING \
    --display-name="Breathe training SA" \
    --project=$GCP_PROJECT

gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:$SA_TRAINING@$GCP_PROJECT.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:$SA_TRAINING@$GCP_PROJECT.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

**Serving SA** — used by Streamlit to query BigQuery. Read-only access is sufficient. We also need to download a JSON key file for this SA, because Streamlit Cloud does not have a GCP metadata server to generate tokens automatically — the key needs to be passed explicitly (see Phase 4):

```bash
gcloud iam service-accounts create $SA_SERVING \
    --display-name="Breathe serving SA" \
    --project=$GCP_PROJECT

gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:$SA_SERVING@$GCP_PROJECT.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:$SA_SERVING@$GCP_PROJECT.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser"

# Download the key — this file will be used in Streamlit secrets
gcloud iam service-accounts keys create .gcp-serving-key.json \
    --iam-account="$SA_SERVING@$GCP_PROJECT.iam.gserviceaccount.com"
```

### 1.5 Create BigQuery datasets

Three datasets: raw ingested data, preprocessed features, and monitoring logs:

```bash
bq --location=$BQ_REGION mk --dataset $GCP_PROJECT:breathe_bq_raw
bq --location=$BQ_REGION mk --dataset $GCP_PROJECT:breathe_bq_processed
bq --location=$BQ_REGION mk --dataset $GCP_PROJECT:breathe_bq_monitoring
```

---

## Phase 2 — MLflow server

The MLflow tracking server runs as a persistent Cloud Run Service. It exposes the experiment and model registry UI, stores run metadata in a local SQLite file, and writes model artifacts directly to GCS.

**Note on persistence:** SQLite lives in `/tmp` inside the container — it is lost when Cloud Run scales the instance down. This is a known limitation; see PROJECT_SUMMARY for the full discussion and the production-grade alternative (Cloud SQL).

### 2.1 Build and push the MLflow image

```bash
docker build --platform linux/amd64 \
    -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/mlflow:latest \
    -f docker/mlflow/Dockerfile .

docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/mlflow:latest
```

`--platform linux/amd64` is required when building on Apple Silicon — Cloud Run runs on x86_64.

### 2.2 Deploy as Cloud Run Service

```bash
gcloud run deploy breathe-mlflow \
    --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/mlflow:latest \
    --region $GCP_REGION \
    --port 8080 \
    --memory 1Gi \
    --allow-unauthenticated \
    --service-account $SA_TRAINING@$GCP_PROJECT.iam.gserviceaccount.com \
    --project $GCP_PROJECT
```

`--allow-unauthenticated` is required because the MLflow Python client sends plain HTTP requests without authentication headers. A fully authenticated setup would require a proxy or a managed MLflow service — see PROJECT_SUMMARY.

The `--service-account` flag attaches the training SA to the container via GCP's workload identity mechanism. The container can then access GCS for artifact storage without any explicit credentials in the code.

### 2.3 Get the service URL

You will need this URL in the next phase:

```bash
gcloud run services describe breathe-mlflow \
    --region $GCP_REGION \
    --format "value(status.url)"
```

You can also verify the server is up:
```bash
curl $(gcloud run services describe breathe-mlflow --region $GCP_REGION --format "value(status.url)")/health
# → OK
```

---

## Phase 3 — Cloud Run Jobs (Prefect flows)

The Prefect flows run inside a Cloud Run Job — an ephemeral container that executes once and exits. Two jobs are deployed: one for the one-time bootstrap, one for periodic monitoring.

### 3.1 Prepare the environment file

Create `job.env.yaml` at the repo root. This file injects all environment variables into the container at deploy time. **Do not commit this file** — it contains secrets.

```yaml
GCP_PROJECT: "breathe-air-quality"
GCP_REGION: "europe-west1"
BQ_REGION: "EU"
BUCKET_NAME: "breathe-mlflow-gcs-1080515972983"
BQ_DATASET_RAW: "breathe_bq_raw"
BQ_DATASET_PROCESSED: "breathe_bq_processed"
BQ_DATASET_MONITORING: "breathe_bq_monitoring"
SA_TRAINING: "breathe-sa-training"
SA_SERVING: "breathe-sa-serving"
MLFLOW_TRACKING_URI: "https://breathe-mlflow-xxxx-ew.a.run.app"   # URL from Phase 2.3
API_AQ: "your-openaq-key"
API_OW: "your-openweather-key"
PREFECT_API_KEY: "your-prefect-api-key"
PREFECT_API_URL: "https://api.prefect.cloud/api/accounts/xxx/workspaces/xxx"
```

To get Prefect credentials (login once, then read the config):
```bash
prefect cloud login
prefect config view   # → shows PREFECT_API_KEY and PREFECT_API_URL
```

### 3.2 Build and push the package image

This image contains the full project (all of `src/`, with dependencies installed from `pyproject.toml`):

```bash
docker build --platform linux/amd64 \
    -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/package:prod \
    -f docker/package/Dockerfile .

docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/package:prod
```

### 3.3 Deploy the bootstrap job

The default `CMD` in the Dockerfile runs `src/entrypoint_bootstrap.py`, which trains the initial champion model and evaluates it on the test set:

```bash
gcloud run jobs deploy breathe-bootstrap-job \
    --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/package:prod \
    --region $GCP_REGION \
    --memory 2Gi \
    --cpu 2 \
    --task-timeout 3600 \
    --max-retries 0 \
    --env-vars-file job.env.yaml \
    --service-account $SA_TRAINING@$GCP_PROJECT.iam.gserviceaccount.com \
    --project $GCP_PROJECT
```

Execute it once to run the bootstrap. This must complete before running any periodic batch:

```bash
gcloud run jobs execute breathe-bootstrap-job \
    --region $GCP_REGION \
    --project $GCP_PROJECT \
    --wait
```

### 3.4 Deploy the periodic job

Same image, but the entrypoint is overridden to run `src/entrypoint_periodic.py`:

```bash
gcloud run jobs deploy breathe-periodic-job \
    --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/package:prod \
    --region $GCP_REGION \
    --memory 2Gi \
    --cpu 2 \
    --task-timeout 3600 \
    --max-retries 0 \
    --env-vars-file job.env.yaml \
    --service-account $SA_TRAINING@$GCP_PROJECT.iam.gserviceaccount.com \
    --project $GCP_PROJECT \
    --command "python" \
    --args "src/entrypoint_periodic.py"
```

Execute manually with a specific batch number (repeat for batches 1 through 6 to simulate the full monitoring history):

```bash
gcloud run jobs execute breathe-periodic-job \
    --region $GCP_REGION \
    --project $GCP_PROJECT \
    --wait \
    --args "src/entrypoint_periodic.py,--batch_num,1"
```

### 3.5 Update after code changes

When you modify the code, rebuild and push the image, then update the job to pick up the new image:

```bash
docker build --platform linux/amd64 \
    -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/package:prod \
    -f docker/package/Dockerfile .
docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/package:prod

gcloud run jobs update breathe-periodic-job \
    --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_REPO/package:prod \
    --region $GCP_REGION \
    --project $GCP_PROJECT
```

---

## Phase 4 — Streamlit UI

The dashboard reads directly from BigQuery — no API layer in between.

### 4.1 Local

```bash
uv sync   # or: pip install -e ".[ui]"
source .env

# BigQuery access via Application Default Credentials
gcloud auth application-default login

PYTHONPATH=. streamlit run ui/streamlit.py
```

### 4.2 Streamlit Cloud deployment

Streamlit Cloud does not have a GCP metadata server, so the BigQuery client cannot auto-discover credentials the way Cloud Run does. The serving SA key must be passed explicitly. The app handles this in `_bq_client()`: if a `[gcp_credentials]` table is present in Streamlit secrets, it builds a `service_account.Credentials` object from it and passes it to the BigQuery client. If not (local dev), it falls back to Application Default Credentials.

**Steps:**

1. Push the repo to GitHub.
2. Connect it on [share.streamlit.io](https://share.streamlit.io) and set the main file to `ui/streamlit.py`.
3. In **Settings → Secrets**, paste the following (fill in values from `.gcp-serving-key.json` and your project):

```toml
GCP_PROJECT = "breathe-air-quality"
BQ_DATASET_MONITORING = "breathe_bq_monitoring"
BQ_DATASET_RAW = "breathe_bq_raw"
BQ_DATASET_PROCESSED = "breathe_bq_processed"
GCP_REGION = "europe-west1"
BQ_REGION = "EU"
BUCKET_NAME = "breathe-mlflow-gcs-1080515972983"

[gcp_credentials]
type = "service_account"
project_id = "breathe-air-quality"
private_key_id = "..."
private_key = "-----BEGIN RSA PRIVATE KEY-----\n..."
client_email = "breathe-sa-serving@breathe-air-quality.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

The `[gcp_credentials]` block should match the structure of `.gcp-serving-key.json` exactly.

---

## Phase 5 — Cloud Scheduler (optional)

Once the periodic job is validated, Cloud Scheduler can trigger it automatically on a fixed cadence. The example below triggers weekly on Monday mornings — adjust to match your batch frequency (every 6 weeks in this project):

```bash
gcloud scheduler jobs create http breathe-cron \
    --location=$GCP_REGION \
    --schedule="0 8 * * 1" \
    --time-zone="Europe/Paris" \
    --uri="https://$GCP_REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$GCP_PROJECT/jobs/breathe-periodic-job:run" \
    --http-method=POST \
    --oauth-service-account-email=$SA_TRAINING@$GCP_PROJECT.iam.gserviceaccount.com \
    --project $GCP_PROJECT
```

When triggered without `--batch_num`, `entrypoint_periodic.py` raises a `ValueError` (automatic date calculation from BQ not yet implemented). Automatic batch scheduling — computing the next `batch_start` / `batch_end` from the last row in the `batches` table — is the logical next step.

---

## Useful diagnostics

```bash
# Check that MLflow is responding
curl $(gcloud run services describe breathe-mlflow \
    --region $GCP_REGION --format "value(status.url)")/health

# View the last 50 log lines from the periodic job
gcloud logging read \
    "resource.type=cloud_run_job AND resource.labels.job_name=breathe-periodic-job" \
    --project=$GCP_PROJECT \
    --limit=50 \
    --format="value(textPayload)"

# Check what roles the training SA has
gcloud projects get-iam-policy $GCP_PROJECT \
    --flatten="bindings[].members" \
    --filter="bindings.members:$SA_TRAINING@$GCP_PROJECT.iam.gserviceaccount.com" \
    --format="table(bindings.role)"
```
