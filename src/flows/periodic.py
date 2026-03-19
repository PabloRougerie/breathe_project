import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from prefect import task, flow, get_run_logger
from google.api_core.exceptions import NotFound

from src.params import *
from src.utils import *
from src.models.registry import promote_challenger, load_model
from src.ingestion.openaq import OpenAQClient
from src.ingestion.openweather import OpenWeatherClient
from src.preprocess.preproc_pipeline import preprocessing_pipeline
from src.models.model_pipeline import run_training, run_evaluating, setup_mlflow
from src.models.evaluate import self_compare, cross_compare


# =============================================================================
# BATCH SCHEDULE
# One entry per periodic cycle. train_start is fixed (maximum historical data).
# train_end = batch_start - 1 day (no leakage from batch into training).
# Batches are 42-day windows (6 weeks).
# =============================================================================

BATCH_SCHEDULE = {
    1: {"batch_start": "2025-06-12", "batch_end": "2025-07-23", "train_start": "2023-05-01", "train_end": "2025-06-11"},
    2: {"batch_start": "2025-07-24", "batch_end": "2025-09-03", "train_start": "2023-05-01", "train_end": "2025-07-23"},
    3: {"batch_start": "2025-09-04", "batch_end": "2025-10-15", "train_start": "2023-05-01", "train_end": "2025-09-03"},
    4: {"batch_start": "2025-10-16", "batch_end": "2025-11-26", "train_start": "2023-05-01", "train_end": "2025-10-15"},
    5: {"batch_start": "2025-11-27", "batch_end": "2026-01-07", "train_start": "2023-05-01", "train_end": "2025-11-26"},
    6: {"batch_start": "2026-01-08", "batch_end": "2026-02-18", "train_start": "2023-05-01", "train_end": "2026-01-07"},
}



# =============================================================================
# TASKS
# Important: GCSStorageClient is instantiated *inside* tasks.
# Passing a client object would cause Prefect errors.
# =============================================================================

@task
def ingestion(start_date, end_date):
    """Fetch raw air quality and weather data from APIs.

    GCS-cached per day: only days missing from the cache trigger an API call.
    Returns (airqual_df, weather_df) with a 'date' column.
    """
    aq_client = OpenAQClient(api_key=API_AQ, storage="gcp")
    airqual_df = aq_client.get_data(
        cities=CITIES,
        start_date=start_date,
        end_date=end_date,
        start_project_date=START_PROJECT_DATE_STR,
        end_project_date=END_PROJECT_DATE_STR
    )
    weather_client = OpenWeatherClient(api_key=API_OW, storage="gcp")
    weather_df = weather_client.get_all_data(
        cities=CITIES,
        start_date=start_date,
        end_date=end_date
    )
    logger = get_run_logger()
    logger.info(f"ingested {len(airqual_df)} rows for air_quality")
    logger.info(f"ingested {len(weather_df)} rows for weather")
    return airqual_df, weather_df

@task
def delete_cache(data_type):
    """ data_type: "airqual", "weather", "processed" """
    cache_client = GCSCacheClient(bucket_name=BUCKET_NAME)
    total_deleted = 0

    if data_type == "airqual":
        for city in CITIES.keys():
            cache_files = cache_client.list(prefix=f"{city}/airqual/")
            if cache_files:
                total_deleted += cache_client.delete(cache_files)

    logger = get_run_logger()
    logger.info(f"Cache cleared — {total_deleted} blobs deleted ({data_type})")


@task
def check_data_exist(data_type, start_date, end_date):
    """Return True if BQ already holds data of data_type for that date range."""
    try:
        df, _ = GCSStorageClient().get_data(
            data_type=data_type,
            start_date=start_date,
            end_date=end_date
        )
        #TODO log table found
        return not df.empty
    except NotFound:
        return False
    #TODO log table NOT found


@task
def upload_data(df, data_type, start_date, end_date):
    """Save a DataFrame to BigQuery (idempotent: DELETE existing rows then WRITE_APPEND)."""
    full_table_name = GCSStorageClient().save_data(
        data=df,
        data_type=data_type,
        start_date=start_date,
        end_date=end_date
    )
    logger = get_run_logger()
    logger.info(f"✅ Saved {len(df)} rows to {full_table_name} ({start_date} → {end_date})")


@task
def download_data(data_type, start_date, end_date):
    """Load a DataFrame from BigQuery for the given date range."""

    df, full_table_name= GCSStorageClient().get_data(
        data_type=data_type,
        start_date=start_date,
        end_date=end_date
        )
    logger = get_run_logger()
    logger.info(f"✅ Loaded {len(df)} rows from {full_table_name} ({start_date} → {end_date})")


    return df


@task
def preprocess_raw_data(airqual_df, weather_df, mode="train"):
    """Run the full preprocessing pipeline on raw DataFrames.

    Args:
        mode (str): 'train' applies sensor filtering; 'eval' skips it.

    Returns:
        tuple: (dataset_metadata dict, preprocessed DataFrame with date + target + features)
    """
    return preprocessing_pipeline(airqual_df=airqual_df, weather_df=weather_df, mode=mode)


@task
def train_model(data, dataset_metadata):
    """Train and register a model. Returns (trained_model, model_version).

    X/y split done here. Champion vs challenger registration handled by run_training.
    """
    X = data.drop(columns=["target", "date"])
    y = data["target"]

    trained_model, model_version, assigned_alias = run_training(X, y, dataset_metadata=dataset_metadata)
    logger = get_run_logger()
    logger.info(f"train model version {model_version} under alias {assigned_alias}")
    return trained_model, model_version


@task
def evaluate_model(data, dataset_metadata, alias, eval_mode, model=None, model_version=None):
    """Evaluate a model and log metrics to MLflow. Returns RMSE.

    If model is passed in-memory, it is used directly (no registry lookup).
    """
    X = data.drop(columns=["target", "date"])
    y_true = data["target"]

    score, model_version= run_evaluating(
        X_val=X,
        y_true=y_true,
        dataset_metadata=dataset_metadata,
        alias=alias,
        eval_mode=eval_mode,
        model = model,
        model_version= model_version
    )
    logger = get_run_logger()
    logger.info(f"{alias} — RMSE: {round(score, 4)} ({eval_mode})")
    return score, model_version

@task
def get_prediction(data):
    mlflow_client = MlflowClient()

    X = data.drop(columns = ["target", "date"])
    y_true = data["target"]

    model, predict_model_version = load_model(mlflow_client, alias= "champion")
    y_pred = model.predict(X)

    return y_true, y_pred, data["city"], data["date"], predict_model_version


@task
def self_compare_champion(new_batch_rmse):
    """Compare current score against champion's reference_rmse tag. Returns True if drift."""
    client = MlflowClient()
    champion = client.get_model_version_by_alias(name=MLFLOW_MODEL_NAME, alias="champion")
    ref_rmse = float(champion.tags["reference_rmse"])

    logger = get_run_logger()
    logger.info(f"ref RMSE: {ref_rmse:.4f} | batch RMSE: {new_batch_rmse:.4f} | perf change: {(1 - new_batch_rmse / ref_rmse)*100:.2f}%")

    is_drift = self_compare(score_ref=ref_rmse, score_new=new_batch_rmse)
    return is_drift

@task
def cross_compare_challenger(challenger_score, champion_score):
    """Return True if challenger RMSE is better (lower) than champion RMSE."""
    logger = get_run_logger()
    logger.info(f"champion RMSE: {champion_score:.4f} | challenger RMSE: {challenger_score:.4f}")
    return cross_compare(score_old=champion_score, score_new=challenger_score)

@task
def promote_better_model():
    """Promote current challenger to champion in MLflow registry."""
    return promote_challenger()




# =============================================================================
# SUBFLOWS
# Each subflow handles one logical stage of the periodic pipeline.
# ingestion_subflow   : BQ existence check + force override
# preprocess_subflow  : fallback reload from BQ if data not passed in-memory
# train_subflow       : triggered on drift; uses HISTORICAL dates, not fresh batch
# eval_and_drift_check_subflow : evaluates champion + self-compare; returns (score, drift_detected)
# eval and promote: evalutes challenger of last month data and promote if better
# =============================================================================

@flow
def ingestion_subflow(start_date, end_date, force=False):
    """Fetch raw data from APIs and upload to BQ. Returns (airqual_df, weather_df).

    Skips API calls if both tables already exist in BQ (unless force=True).
    BQ existence checks for airqual and weather run in parallel.
    """
    if force:
        # Bypass BQ check: always fetch from APIs and overwrite BQ
        airqual_df, weather_df = ingestion.submit(
            start_date=start_date, end_date=end_date
        ).result()

        upload_data.submit(airqual_df, data_type="airqual",
                           start_date=start_date, end_date=end_date).result()
        upload_data.submit(weather_df, data_type="weather",
                           start_date=start_date, end_date=end_date).result()

        return airqual_df, weather_df

    # Check BQ in parallel for both data types
    airqual_exist_f = check_data_exist.submit(
        data_type="airqual", start_date=start_date, end_date=end_date
    )
    weather_exist_f = check_data_exist.submit(
        data_type="weather", start_date=start_date, end_date=end_date
    )

    if airqual_exist_f.result() and weather_exist_f.result():
        # Data already in BQ: load directly, skip API calls and upload
        airqual_df = download_data.submit(
            data_type="airqual", start_date=start_date, end_date=end_date
        ).result()
        weather_df = download_data.submit(
            data_type="weather", start_date=start_date, end_date=end_date
        ).result()
    else:
        # Data missing in BQ: fetch from APIs and upload
        airqual_df, weather_df = ingestion.submit(
            start_date=start_date, end_date=end_date
        ).result()

        upload_data.submit(airqual_df, data_type="airqual",
                           start_date=start_date, end_date=end_date).result()
        upload_data.submit(weather_df, data_type="weather",
                           start_date=start_date, end_date=end_date).result()

    delete_cache.submit(data_type= "airqual").result()
    return airqual_df, weather_df


@flow
def preprocess_subflow(start_date, end_date, airqual_df=None, weather_df=None, mode="train"):
    """Preprocess raw data and upload processed dataset to BQ. Returns (metadata, data).

    Fallback: reloads raw data from BQ if DataFrames are not passed in-memory.

    Args:
        mode (str): 'train' applies sensor filtering; 'eval' skips it.
    """
    if airqual_df is None or weather_df is None:
        # Fallback: reload raw data from BQ in parallel
        airqual_f = download_data.submit(
            data_type="airqual", start_date=start_date, end_date=end_date
        )
        weather_f = download_data.submit(
            data_type="weather", start_date=start_date, end_date=end_date
        )
        airqual_df, weather_df = airqual_f.result(), weather_f.result()

    dataset_metadata, data = preprocess_raw_data.submit(airqual_df, weather_df, mode=mode).result()

    upload_data.submit(
        df=data, data_type="processed", start_date=start_date, end_date=end_date
    ).result()

    return dataset_metadata, data


@flow
def train_subflow(start_date, end_date, data=None, dataset_metadata=None):
    """Train and register a challenger model. Returns (trained_model, model_version).

    Dates should cover the HISTORICAL training window (not the fresh batch).
    Since a champion already exists, run_training auto-registers as challenger.
    Fallback: reloads processed data from BQ if not passed in-memory.
    """
    if data is None:
        # Fallback: reload processed data from BQ
        data = download_data.submit(
            data_type="processed", start_date=start_date, end_date=end_date
        ).result()

    if dataset_metadata is None:
        # Reconstruct metadata from the DataFrame if not passed
        dataset_metadata = {
            "date_start":    str(data["date"].min()),
            "date_end":      str(data["date"].max()),
            "n_rows":        len(data),
            "n_features":    len(data.columns) - 2,
            "list_features": [feat for feat in data.columns if feat not in ["date", "target"]],
        }

    trained_model, model_version = train_model.submit(data=data, dataset_metadata=dataset_metadata).result()
    return trained_model, model_version

@flow
def eval_and_drift_check_subflow(start_date, end_date, alias, eval_mode,
                           data=None, dataset_metadata=None, model=None, model_version=None):
    """Evaluate champion on the fresh batch and check for drift against its reference score.

    In the periodic flow, always called with alias='champion', eval_mode='fresh_batch'.
    Fallback: reloads processed data from BQ if not passed in-memory.

    Returns:
        tuple: (score: float, is_drift: bool)
    """
    if data is None:
        # Fallback: reload processed data from BQ
        data = download_data.submit(
            data_type="processed", start_date=start_date, end_date=end_date
        ).result()

    if dataset_metadata is None:
        # Reconstruct metadata from the DataFrame if not passed
        dataset_metadata = {
            "date_start":    str(data["date"].min()),
            "date_end":      str(data["date"].max()),
            "n_rows":        len(data),
            "n_features":    len(data.columns) - 2,
            "list_features": [feat for feat in data.columns if feat not in ["date", "target"]],
        }
    score_champion, version_champion = evaluate_model.submit(
        data, dataset_metadata, alias=alias, eval_mode=eval_mode,
        model=model, model_version=model_version
    ).result()

    is_drift = self_compare_champion.submit(new_batch_rmse=score_champion).result()




    return score_champion, version_champion, is_drift



@flow
def eval_and_promote_subflow(start_date, end_date, alias, eval_mode, champion_score,
                           data=None, dataset_metadata=None, model=None, model_version=None):
    """Evaluate challenger on the fresh batch, cross-compare with champion, promote if better.

    In the periodic flow, always called with alias='challenger', eval_mode='test_set'.
    Challenger's reference_rmse tag is set here (used for future self_compare).
    Fallback: reloads processed data from BQ if not passed in-memory.
    """
    if data is None:
        # Fallback: reload processed data from BQ
        data = download_data.submit(
            data_type="processed", start_date=start_date, end_date=end_date
        ).result()

    if dataset_metadata is None:
        # Reconstruct metadata from the DataFrame if not passed
        dataset_metadata = {
            "date_start":    str(data["date"].min()),
            "date_end":      str(data["date"].max()),
            "n_rows":        len(data),
            "n_features":    len(data.columns) - 2,
            "list_features": [feat for feat in data.columns if feat not in ["date", "target"]],
        }

    score_challenger, version_challenger = evaluate_model.submit(
        data, dataset_metadata, alias=alias, eval_mode=eval_mode,
        model=model, model_version=model_version
    ).result()

    is_better = cross_compare_challenger.submit(
        challenger_score=score_challenger, champion_score=champion_score
    )



    if is_better.result():
        promote_better_model.submit().result()



    return score_challenger, version_challenger, is_better.result()


# =============================================================================
# MASTERFLOWS
# Entry points for the periodic pipeline. Orchestrate subflows sequentially.
#
# Data is passed in-memory between subflows to avoid redundant BQ round-trips:
#   ingestion → preprocess → train/eval
# =============================================================================

@flow
def periodic_monitoring_masterflow(batch_num: int = 1):
    """Periodic monitoring pipeline: ingest new batch, detect drift, retrain if needed.

    Steps:
      1. Ingest + preprocess the batch window (fresh data)
      2. Evaluate champion on batch → drift check
      3. If drift: retrain challenger on full historical window (train_start → train_end),
         evaluate on batch, promote if better

    Args:
        batch_num (int): Batch number from BATCH_SCHEDULE (1-indexed). Controls all dates.
    """
    if batch_num not in BATCH_SCHEDULE:
        raise ValueError(f"batch_num {batch_num} not found in BATCH_SCHEDULE (valid: {list(BATCH_SCHEDULE.keys())})")

    setup_mlflow()
    logger = get_run_logger()
    monitoring_client= MonitoringClient()

    batch_start = BATCH_SCHEDULE[batch_num]["batch_start"]
    batch_end   = BATCH_SCHEDULE[batch_num]["batch_end"]

    # --- Ingest and preprocess the fresh batch ---
    logger.info("Ingestion")
    airqual_df, weather_df = ingestion_subflow(batch_start, batch_end)
    dataset_metadata, batch_data = preprocess_subflow(batch_start, batch_end,
                                                    airqual_df, weather_df, mode="eval")

    # --- Evaluate champion on fresh batch; champion loaded from registry ---
    logger.info(f"Evaluating champion on batch {batch_start} to {batch_end}")
    score_champion, version_champion, is_drift = eval_and_drift_check_subflow(
        start_date=batch_start,
        end_date=batch_end,
        alias="champion",
        eval_mode="fresh_batch",
        data=batch_data,
        dataset_metadata=dataset_metadata
    )

    logger.info(f"Champion RMSE on fresh batch: {score_champion:.4f}")


    score_challenger = None
    version_challenger = None
    promoted = None

    # --- No drift: nothing to do ---
    if not is_drift:
        logger.info("No drift detected: champion remains the serving model")

    # --- Drift detected: retrain challenger on shifted historical window ---
    else:
        logger.warning(f"Drift detected")

        train_start = BATCH_SCHEDULE[batch_num]["train_start"]
        train_end   = BATCH_SCHEDULE[batch_num]["train_end"]


        train_metadata, train_data = preprocess_subflow(train_start, train_end, mode="train")
        logger.info(f"Training new challenger on {train_start} → {train_end}")

        challenger_model, challenger_version = train_subflow(
            start_date=train_start,
            end_date=train_end,
            data=train_data,
            dataset_metadata=train_metadata
        )

        # Evaluate challenger on fresh batch (becomes its reference test set),
        # cross-compare with champion score, promote if better
        score_challenger, version_challenger, promoted = eval_and_promote_subflow(
            start_date=batch_start,
            end_date=batch_end,
            alias="challenger",
            eval_mode="test_set",
            champion_score=score_champion,
            data=batch_data,
            dataset_metadata=dataset_metadata,
            model=challenger_model,
            model_version=challenger_version
        )

        challenger_metadata = {"model_version": version_challenger,
                          "train_start": BATCH_SCHEDULE[batch_num]["train_start"] ,
                          "train_end": BATCH_SCHEDULE[batch_num]["train_end"],
                          "eval_start": BATCH_SCHEDULE[batch_num]["batch_start"],
                          "eval_end": BATCH_SCHEDULE[batch_num]["batch_end"],
                          "ref_rmse": score_challenger
                          }



        logger.info(f"Challenger RMSE: {score_challenger:.4f} | Champion RMSE: {score_champion:.4f}")
        if promoted:
            logger.info("Challenger promoted to champion")
            challenger_metadata["alias"] = "champion"
            monitoring_client.update_model_alias(version_champion, "archived")



        else:
            logger.info("Challenger not better — champion retained")
            challenger_metadata["alias"] = "challenger"

        monitoring_client.upsert_model(challenger_metadata) #challenger created will e appended



    batch_log = {"batch_start": batch_start,
                        "batch_end": batch_end,
                        "champion_version": version_champion,
                        "champion_rmse": score_champion,
                        "drift_detected": is_drift,
                        "challenger_version": version_challenger,
                        "rmse_challenger": score_challenger,
                        "run_date":pd.Timestamp.now(),
                        "promotion_applied": promoted
                        }

    monitoring_client.log_batch(batch_log)
    y_true, y_pred, city, date, predict_model_version=  get_prediction(data= batch_data)
    monitoring_client.log_predict(y_true= np.expm1(y_true), y_pred= np.expm1(y_pred),
                                  city= city, date= date,
                                  predict_model_version = predict_model_version)
