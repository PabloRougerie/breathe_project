# Breathe Project — Context Document
*À utiliser comme contexte de départ pour une nouvelle conversation Claude*

---

## 1. Objectifs du projet

### Objectif principal — MLOps / Mise en production
Démontrer la capacité à mettre un modèle ML en production sur GCP dans une architecture full cloud :
- Containerisation et déploiement Cloud Run
- Pipeline d'ingestion automatisé (OpenAQ + OpenWeather → BigQuery)
- Monitoring du drift et réentraînement automatique
- Orchestration Prefect + Cloud Scheduler
- Dashboard de monitoring Streamlit

Le modèle PM2.5 et le dataset sont un **support** pour démontrer l'architecture MLOps — pas l'objectif principal. D'autres projets du portfolio couvrent le développement de modèles.

### Objectif carrière
Portfolio "full stack data scientist". Poste visé principal : Data Scientist. Poste secondaire : ML Engineer.

---

## 2. Modèle et données

### Problème
Prédire PM2.5 à J+1 pour 6 villes : Berlin, Londres, Lyon, New York, Paris, Rome.
Sources : OpenAQ (qualité de l'air) + OpenWeather (météo).

### Preprocessing
- Target : `pm25_avg` J+1, log-transformée (`np.log1p`)
- **`pm25_avg` est log-transformé AVANT la génération des lags** → tous les lags sont en espace log
- Métrique : RMSE sur valeurs log (= RMSLE implicite)
- Imputation gaps 1 jour : `interpolate(method='linear', limit=1, limit_area='inside')` groupby city

### Features retenues (v2, permutation importance n_repeats=400, k=15)
```
lag_1, lag_3, lag_7, city, wind_speed, precipitation, lag_avg_14,
week_std, wind_direction, pressure, temp_max, humidity, temp_min,
month_cos, temp_gradient
```
`temp_gradient = temp_max - temp_min`

### Modèle
- **Algorithme :** LightGBM avec OHE sklearn pipeline pour `city`
- **Baseline :** RMSE log = 0.70 (corrigée — lags en espace log)
- **Score CV :** RMSE log = **0.2358**
- **Tuning :** Optuna, 1000 trials, TPESampler(seed=273), TimeSeriesSplit(n_splits=5, gap=42, test_size=180)
```python
BEST_PARAMS = {
    'learning_rate': 0.029430170491189643,
    'n_estimators': 670,
    'num_leaves': 44,
    'max_depth': 12,
    'min_child_samples': 38,
    'reg_alpha': 1.0998029678527025e-05,
    'reg_lambda': 1.282458658923553e-07,
    'subsample': 0.9143332144889533,
    'colsample_bytree': 0.9767977854994194,
    'subsample_freq': 1,
    'random_state': 273,
    'verbose': -1
}
```

**Datasets :**
- Train v1 : mai 2023 → avril 2025
- Test set v1 : batch 1 (6 semaines suivant le train)

### Analyse drift
- Drift détecté et promotions déclenchées correctement
- Dégradation hivernale structurelle : RMSE ~0.20 été → ~0.43 hiver
- Réentraînement ne corrige pas significativement — limite des données météo journalières (inversions thermiques non capturées)
- **Conclusion portfolio** : l'architecture fonctionne correctement ; la limite identifiée est la prédictibilité intrinsèque de PM2.5 hivernal avec les features disponibles

---

## 3. Plan de fonctionnement en production

### Cycle tous les 6 semaines (déclenché par Cloud Scheduler)
```
Nouveau batch de 6 semaines disponible :

1. INGESTION
   → API OpenAQ + OpenWeather pour le batch
   → Cache JSON sur GCS (idempotent, reprend si crash)
   → Save raw tables sur BigQuery (DELETE + WRITE_APPEND, idempotent)

2. PREPROCESSING
   → Load raw depuis BQ → preprocessing_pipeline() → Save processed sur BQ

3. ÉVALUATION DU CHAMPION
   → score_batch = evaluate(@champion, X_batch)
   → self_compare(score_ref_champion, score_batch)
   → score_ref = score du champion sur SON test set initial (batch suivant son train)
   → Si score_batch >= score_ref * (1 + DRIFT_THRESHOLD) : DRIFT DÉTECTÉ

4. SI DRIFT :
   → Train challenger sur fenêtre étendue :
     train_start fixe (mai 2023), train_end = fin du batch précédent
     (la end date est shiftée de 6 semaines à chaque nouveau batch)
   → Le batch qui a déclenché le drift devient le test set du challenger
   → score_ref_challenger = evaluate(@challenger, X_batch)
   → cross_compare(score_batch_champion, score_ref_challenger)
   → Si challenger améliore de IMPROVEMENT_THRESHOLD : PROMOTION
      → promote_challenger() → ancien champion archivé
   → Sinon : challenger archivé, champion conservé

5. PRÉDICTION J+1
   → Charge @champion depuis MLflow
   → Prédictions pour les 6 villes
```

**Seuils :**
- `DRIFT_THRESHOLD = 0.10`
- `IMPROVEMENT_THRESHOLD = 0.05`

**Gestion des dates de batch :**
- Pas de calcul automatique pour l'instant
- Dictionnaire `BATCH_SCHEDULE` dans le code, le cron incrémente l'index de 1 à chaque run
- À implémenter plus tard : calcul automatique basé sur la date d'exécution
```python
BATCH_SCHEDULE = {
    1: {"batch_start": "2025-06-12", "batch_end": "2025-07-23", "train_start": "2023-05-01", "train_end": "2025-06-11"},
    2: {"batch_start": "2025-07-24", "batch_end": "2025-09-03", "train_start": "2023-05-01", "train_end": "2025-07-23"},
    3: {"batch_start": "2025-09-04", "batch_end": "2025-10-15", "train_start": "2023-05-01", "train_end": "2025-09-03"},
    4: {"batch_start": "2025-10-16", "batch_end": "2025-11-26", "train_start": "2023-05-01", "train_end": "2025-10-15"},
    5: {"batch_start": "2025-11-27", "batch_end": "2026-01-07", "train_start": "2023-05-01", "train_end": "2025-11-26"},
    6: {"batch_start": "2026-01-08", "batch_end": "2026-02-18", "train_start": "2023-05-01", "train_end": "2026-01-07"},
}
```

---

## 4. Architecture cloud

### Composants
```
Cloud Scheduler (cron paramétrable)
      ↓
Cloud Run Job (éphémère)
  → flows Prefect : ingestion → preprocessing → eval → train si drift

Cloud Run Service — MLflow Server (permanent)
  → Backend store : SQLite (mlflow.db) sur GCS via volume mount
  → Artifact store : GCS bucket (gs://breathe-mlflow-gcs-1080515972983/artifacts)

Cloud Run Service — Streamlit (permanent, à créer)
  → Dashboard monitoring : courbes réelles vs prédites, logs drift/promotion
  → Peut inclure les prédictions J+1 directement (FastAPI optionnelle)

GCS Bucket : breathe-mlflow-gcs-1080515972983
  → Cache JSON ingestion ({city}/{api_source}/{filename}.json)
  → MLflow artifacts (modèles)
  → mlflow.db (backend store via volume mount)

BigQuery
  → Dataset raw : breathe_bq_raw (tables: weather, airqual)
  → Dataset processed : breathe_bq_processed (table: processed)
```

### MLflow — configuration
- Serveur MLflow sur Cloud Run
- Backend store : SQLite via GCS volume mount (`/mlflow-data/mlflow.db`)
- Artifact store : `gs://breathe-mlflow-gcs-1080515972983/artifacts`
- En local : `MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"`
- En prod : `MLFLOW_TRACKING_URI = URL du Cloud Run MLflow` (à configurer)
- `setup_mlflow()` appelé en entrypoint du script exécuté
- Aliases : `@champion`, `@challenger`, tag `status=archived`

### Dockerfile MLflow
```dockerfile
FROM python:3.13-slim
RUN pip install mlflow google-cloud-storage
CMD ["mlflow", "server",
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db",
     "--artifacts-destination", "gs://breathe-mlflow-gcs-1080515972983/artifacts",
     "--host", "0.0.0.0",
     "--port", "8080"]
```

### Service accounts
- `breathe-sa-training` : GCS (objectAdmin) + BigQuery (dataEditor)
- `breathe-sa-serving` : GCS (objectViewer) uniquement

---

## 5. Structure du projet
```
breathe_project/
├── .env                   # GCP_PROJECT, GCP_REGION, BUCKET_NAME, MLFLOW_TRACKING_URI, API_OW, API_AQ
├── Makefile
├── Dockerfile.mlflow
├── pyproject.toml
└── src/
    ├── params.py           # BEST_PARAMS, SELECTED_FEATURES, seuils, dates, cities, paths locaux
    ├── utils.py            # StorageClient, CacheClient, filter_columns, merge_source_df
    ├── ingestion/
    │   ├── openaq.py       # OpenAQClient
    │   └── openweather.py  # OpenWeatherClient
    ├── preprocess/
    │   ├── cleaning.py
    │   ├── features.py
    │   └── preproc_pipeline.py
    ├── models/
    │   ├── baseline.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── registry.py
    │   └── model_pipeline.py
    ├── flows/
    │   └── flow.py         # tasks + subflows Prefect
    └── api/
        └── main.py         # FastAPI optionnelle
```

### Classes clés dans utils.py
```python
# Cache JSON ingestion
CacheClient(ABC) → LocalCacheClient(cache_dir), GCSCacheClient(bucket_name)
  methods: read(file_name), write(data, file_name), exists(file_name), list(prefix)
  convention: {city}/{api_source}/{filename}.json

# DataFrames raw et processed
StorageClient(ABC) → LocalStorageClient(base_storage_dir), GCSStorageClient()
  methods: save_data(df, data_type, start_date, end_date), get_data(data_type, start_date, end_date)
  GCSStorageClient: BigQuery, DELETE+WRITE_APPEND (idempotent)
```

---

## 6. État actuel

| Composant | Statut |
|-----------|--------|
| EDA | ✅ |
| Preprocessing pipeline | ✅ |
| Feature selection v2 (15 features) | ✅ |
| Model tuning v2 (RMSE=0.2358) | ✅ |
| Pipeline MLflow (local) | ✅ fonctionnel |
| Infrastructure GCP (GCS, BQ, SA, GAR) | ✅ |
| Ingestion pipeline (local + GCS) | ✅ |
| BigQuery save/load | ✅ |
| Flows Prefect (bootstrap + périodique) | ✅ testés en local |
| Simulation drift 6 batches | ✅ drift détecté, promotions déclenchées |
| MLflow sur Cloud Run | ⏳ Dockerfile prêt, déploiement en cours |
| Cloud Run Job + Cloud Scheduler | ❌ |
| FastAPI | ❌ optionnelle |
| Dashboard Streamlit | ❌ |

---

## 7. Prochaines étapes

**En cours — Déploiement MLflow sur Cloud Run**
```bash
gcloud services enable artifactregistry.googleapis.com

gcloud artifacts repositories create breathe \
    --repository-format=docker \
    --location=europe-west1

gcloud auth configure-docker europe-west1-docker.pkg.dev

docker build --platform linux/amd64 \
    -t europe-west1-docker.pkg.dev/$GCP_PROJECT/breathe/mlflow:latest \
    -f Dockerfile.mlflow .

docker push europe-west1-docker.pkg.dev/$GCP_PROJECT/breathe/mlflow:latest

gcloud run deploy breathe-mlflow \
    --image europe-west1-docker.pkg.dev/$GCP_PROJECT/breathe/mlflow:latest \
    --region europe-west1 \
    --port 8080 \
    --memory 512Mi \
    --no-allow-unauthenticated \
    --add-volume name=mlflow-data,type=cloud-storage,bucket=breathe-mlflow-gcs-1080515972983 \
    --add-volume-mount volume=mlflow-data,mount-path=/mlflow-data \
    --service-account breathe-sa-training@$GCP_PROJECT.iam.gserviceaccount.com
```

**Ensuite**
1. Tester MLFLOW_TRACKING_URI → URL Cloud Run dans .env
2. Cloud Run Job + Cloud Scheduler pour flow périodique
3. Dashboard Streamlit (monitoring + prédictions)
