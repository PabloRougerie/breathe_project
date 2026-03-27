FROM python:3.13-slim
RUN pip install "mlflow>=2.18.0,<3.0.0" google-cloud-storage
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////tmp/mlflow.db", \
"--artifacts-destination", "gs://breathe-mlflow-gcs-1080515972983/artifacts", \
"--host", "0.0.0.0", "--port", "8080"]
