# Deployment guide — Breathe

Step-by-step commands to **rebuild container images** and **redeploy** the main components: **MLflow tracking server**, **Cloud Run Jobs** (bootstrap + periodic flows), and **Streamlit UI**.
**Cloud Scheduler (cron)** is not fully specified here — wire it to trigger the periodic job with the same image/args you use manually.

Prerequisites: **Docker**, **`gcloud` CLI**, **billing-enabled GCP project**, **Artifact Registry** repository, IAM for deployer and runtime service accounts.

---

## 1. One-time / occasional setup

### 1.1 Variables (shell)

Adapt to your project:

```bash
export GCP_PROJECT="your-gcp-project-id"
export GCP_REGION="europe-west1"
export SA_TRAINING="breathe-sa-training"   # must match IAM service account name
```

### 1.2 Artifact Registry & Docker auth

If the Docker repository does not exist yet:

```bash
gcloud services enable artifactregistry.googleapis.com --project "$GCP_PROJECT"

gcloud artifacts repositories create breathe \
  --repository-format=docker \
  --location="$GCP_REGION" \
  --project="$GCP_PROJECT"
```

Authenticate Docker to push:

```bash
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev"
```

### 1.3 Application environment file for Cloud Run Jobs

Cloud Run Jobs expect env vars (GCP, MLflow, API keys). Create a file **`job.env.yaml`** at the repo root (or path of your choice) — **do not commit secrets**. Example shape:

```yaml
GCP_PROJECT: "your-gcp-project-id"
GCP_REGION: "europe-west1"
BUCKET_NAME: "your-gcs-bucket"
MLFLOW_TRACKING_URI: "https://your-mlflow-run-url"
API_OW: "your-openweather-key"
API_AQ: "your-openaq-key"
BQ_DATASET_RAW: "breathe_bq_raw"
BQ_DATASET_PROCESSED: "breathe_bq_processed"
BQ_DATASET_MONITORING: "breathe_bq_monitoring"
```

Add any other keys your code reads from `os.environ` / `params.py`.

---

## 2. Application image (flows / Prefect entrypoints)

Built from **`docker/package/Dockerfile`** (installs `pyproject.toml`, copies `src/`). Default `CMD` runs bootstrap; the **periodic job** overrides command/args.

From **repository root**:

```bash
docker build --platform linux/amd64 \
  -t "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/breathe/package:prod" \
  -f docker/package/Dockerfile .

docker push "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/breathe/package:prod"
```

### 2.1 Cloud Run Job — bootstrap (train + eval)

```bash
gcloud run jobs deploy breathe-bootstrap-job \
  --image "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/breathe/package:prod" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT" \
  --memory 1Gi \
  --cpu 1 \
  --env-vars-file job.env.yaml \
  --service-account "${SA_TRAINING}@${GCP_PROJECT}.iam.gserviceaccount.com" \
  --tasks 1 \
  --max-retries 1 \
  --task-timeout 600
```

Uses image default: `python src/entrypoint_bootstrap.py`.

### 2.2 Cloud Run Job — periodic monitoring

```bash
gcloud run jobs deploy breathe-periodic-job \
  --image "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/breathe/package:prod" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT" \
  --memory 1Gi \
  --cpu 1 \
  --env-vars-file job.env.yaml \
  --service-account "${SA_TRAINING}@${GCP_PROJECT}.iam.gserviceaccount.com" \
  --tasks 1 \
  --max-retries 1 \
  --task-timeout 600 \
  --command "python" \
  --args "src/entrypoint_periodic.py"
```

**Execute** (examples):

```bash
# Default batch in code (see periodic flow fallback)
gcloud run jobs execute breathe-periodic-job --region "$GCP_REGION" --project "$GCP_PROJECT" --wait

# Specific batch index
gcloud run jobs execute breathe-periodic-job \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT" \
  --wait \
  --args "src/entrypoint_periodic.py,--batch_num,1"
```

---

## 3. MLflow tracking server image

Source: **`docker/mlflow/Dockerfile`**. It starts MLflow with:

- **Artifacts:** GCS URI baked in the Dockerfile (`gs://breathe-mlflow-gcs-…/artifacts` — **change** to your bucket if different).
- **Backend store:** `sqlite:////tmp/mlflow.db` inside the container (**ephemeral** unless you add a volume mount).

Build and push:

```bash
docker build --platform linux/amd64 \
  -t "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/breathe/mlflow:latest" \
  -f docker/mlflow/Dockerfile .

docker push "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/breathe/mlflow:latest"
```

Deploy as **Cloud Run service** (adjust memory, auth, volume for durable SQLite if needed):

```bash
gcloud run deploy breathe-mlflow \
  --image "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/breathe/mlflow:latest" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT" \
  --port 8080 \
  --memory 512Mi \
  --no-allow-unauthenticated \
  --service-account "${SA_TRAINING}@${GCP_PROJECT}.iam.gserviceaccount.com"
```

For **persistent** MLflow backend store, mount Cloud Storage or use a managed database; update `--backend-store-uri` accordingly in the Dockerfile or container command.

Point **`MLFLOW_TRACKING_URI`** in `job.env.yaml` and local `.env` to this service URL.

---

## 4. Streamlit UI

### 4.1 Local

```bash
cd /path/to/breathe_project
uv sync --group ui
set -a && source .env && set +a
PYTHONPATH=. streamlit run ui/streamlit.py
```

`.env` must include at least `GCP_PROJECT`, BigQuery dataset names, and Application Default Credentials (`gcloud auth application-default login`) for `google.cloud.bigquery`.

### 4.2 Cloud Run (optional pattern)

Build a small image that installs the project **with** `ui` extras, sets `WORKDIR`, runs `streamlit run ui/streamlit.py --server.port=8080 --server.address=0.0.0.0` — or reuse the package Dockerfile with overridden `CMD`. Deploy as a **service** (not a job) with the same env vars as needed for BigQuery.

---

## 5. Scheduler (cron)

Not detailed in this guide. Typical pattern: **Cloud Scheduler** → HTTP or **Cloud Run Jobs execute** API — pass the same **region**, **job name** `breathe-periodic-job`, and **args** as your manual runs. Increment `batch_num` according to your operational rules (`BATCH_SCHEDULE` in `src/flows/periodic.py`).

---

## 6. Post-deploy checks

1. **MLflow UI** loads; experiment and model registry visible after a bootstrap run.
2. **BigQuery:** rows appear in raw / processed / monitoring datasets after jobs succeed.
3. **Streamlit:** tabs show batches and prediction curves without query errors.

---

## 7. IAM reminders (high level)

- **Training / batch job SA** (`breathe-sa-training`): BigQuery data editor (or narrower scoped), GCS read/write for cache and MLflow artifacts as needed.
- **Serving SA** (if you add an API later): minimal read-only access.

Exact roles depend on your org’s security baseline.
