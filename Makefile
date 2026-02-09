.PHONY: gcp-setup check-env setup-storage setup-sa-training setup-sa-serving clean


#assign variables from .env file
GCP_PROJECT := $(shell echo $$GCP_PROJECT)
GCP_REGION := $(shell echo $$GCP_REGION)
BUCKET_NAME := $(shell echo $$BUCKET_NAME)
BQ_DATASET_RAW := $(shell echo $$BQ_DATASET_RAW)
BQ_DATASET_PROCESSED := $(shell echo $$BQ_DATASET_PROCESSED)
SA_TRAINING := $(shell echo $$SA_TRAINING)
SA_SERVING := $(shell echo $$SA_SERVING)
BQ_REGION := $(shell echo $$BQ_REGION)

# setup gcp resource
gcp-setup: check-env setup-storage setup-sa-training setup-sa-serving
	@echo "GCP resources created successfully"
	@echo "service accounts created successfully"
	@echo "IAM roles created successfully"
	@echo " - .gcp-training-key.json"
	@echo " - .gcp-serving-key.json"

#check if environment variables are set
check-env:
	@echo "Checking environment variables..."
	@test -n "$(GCP_PROJECT)" || (echo "❌ GCP_PROJECT is not set" && exit 1)
	@test -n "$(GCP_REGION)" || (echo "❌ GCP_REGION is not set" && exit 1)
	@test -n "$(BQ_REGION)" || (echo "❌ BQ_REGION is not set" && exit 1)
	@test -n "$(BQ_DATASET_RAW)" || (echo "❌ BQ_DATASET_RAW is not set" && exit 1)
	@test -n "$(BQ_DATASET_PROCESSED)" || (echo "❌ BQ_DATASET_PROCESSED is not set" && exit 1)
	@test -n "$(BUCKET_NAME)" || (echo "❌ BUCKET_NAME is not set" && exit 1)
	@test -n "$(SA_TRAINING)" || (echo "❌ SA_TRAINING is not set" && exit 1)
	@test -n "$(SA_SERVING)" || (echo "❌ SA_SERVING is not set" && exit 1)
	@echo "✅ Environment variables are set correctly"


#setup storage: create the bucket for MLflow
setup-storage:
	@echo "Setting up storage..."
	-gcloud storage buckets create gs://$(BUCKET_NAME) --location=$(GCP_REGION)
	@echo "✅ Bucket created successfully"

	-bq --location=$(BQ_REGION) mk -d $(GCP_PROJECT):$(BQ_DATASET_RAW)
	-bq --location=$(BQ_REGION) mk -d $(GCP_PROJECT):$(BQ_DATASET_PROCESSED)
	@echo "✅ Big Query datasets created successfully"

#SETUP SERVICE ACCOUNT FOR TRAINING
setup-sa-training:
	@echo "Setting up service account for training..."
	-gcloud iam service-accounts create $(SA_TRAINING) --display-name="Service Account for Training"

#add IAM role to the service account to access the bucket
	-gcloud projects add-iam-policy-binding $(GCP_PROJECT) \
  --member=serviceAccount:$(SA_TRAINING)@$(GCP_PROJECT).iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin

#add IAM role to the service account to access the big query dataset
	-gcloud projects add-iam-policy-binding $(GCP_PROJECT) \
  --member=serviceAccount:$(SA_TRAINING)@$(GCP_PROJECT).iam.gserviceaccount.com \
  --role=roles/bigquery.dataEditor

#create the json key for the service account
	-gcloud iam service-accounts keys create .gcp-training-key.json --iam-account="$(SA_TRAINING)@$(GCP_PROJECT).iam.gserviceaccount.com"
	@echo "✅ Service account for training created successfully"

#SETUP SERVICE ACCOUNT FOR SERVING
setup-sa-serving:
	@echo "Setting up service account for serving..."

#create the service account for serving
	-gcloud iam service-accounts create $(SA_SERVING) --display-name="Service Account for Serving"

#add IAM role to the service account to access the bucket
	-gcloud storage buckets add-iam-policy-binding gs://$(BUCKET_NAME) \
  --member=serviceAccount:$(SA_SERVING)@$(GCP_PROJECT).iam.gserviceaccount.com \
  --role=roles/storage.objectViewer

#create the json key for the service account
	-gcloud iam service-accounts keys create .gcp-serving-key.json --iam-account="$(SA_SERVING)@$(GCP_PROJECT).iam.gserviceaccount.com"
	@echo "✅ Service account for serving created successfully"
