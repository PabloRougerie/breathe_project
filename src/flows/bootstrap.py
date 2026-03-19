import pandas as pd
from prefect import task, flow, get_run_logger
from google.api_core.exceptions import NotFound

from src.params import *
from src.utils import *
from src.ingestion.openaq import OpenAQClient
from src.ingestion.openweather import OpenWeatherClient
from src.preprocess.preproc_pipeline import preprocessing_pipeline
from src.models.model_pipeline import run_training, run_evaluating, setup_mlflow


# =============================================================================
# TASKS
# Important: GCSStorageClient is always instantiated *inside* tasks, never
# passed as an argument. Passing a client object would cause Prefect errors.
# =============================================================================

@task
def ingestion(start_date, end_date):
    """Fetch air quality and weather data from APIs for the given date range.

    Both clients use GCS as a JSON cache: if a given day's data is already
    cached, the API call is skipped. Only missing days trigger new API calls.

    Args:
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)

    Returns:
        tuple: (airqual_df, weather_df) as raw DataFrames with a 'date' column
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
    #TODO log shape df
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
    """Check whether a BigQuery table already contains data for the given date range.

    Args:
        data_type (str): 'airqual', 'weather', or 'processed'
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)

    Returns:
        bool: True if data exists in BQ for that range, False otherwise
    """
    try:
        df, _ = GCSStorageClient().get_data(
            data_type=data_type,
            start_date=start_date,
            end_date=end_date
        )
        return not df.empty
    except NotFound:
        return False


@task
def upload_data(df, data_type, start_date, end_date):
    """Save a DataFrame to BigQuery (idempotent: DELETE then WRITE_APPEND).

    Args:
        df (pd.DataFrame): Data to upload
        data_type (str): 'airqual', 'weather', or 'processed'
        start_date (str): Start date (YYYY-MM-DD) — used in DELETE filter
        end_date (str): End date (YYYY-MM-DD) — used in DELETE filter
    """
    GCSStorageClient().save_data(
        data=df,
        data_type=data_type,
        start_date=start_date,
        end_date=end_date
    )


@task
def download_data(data_type, start_date, end_date):
    """Load a DataFrame from BigQuery for the given date range.

    Args:
        data_type (str): 'airqual', 'weather', or 'processed'
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)

    Returns:
        pd.DataFrame
    """
    df, _ = GCSStorageClient().get_data(
        data_type=data_type,
        start_date=start_date,
        end_date=end_date
    )
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
    """Train a model on the given dataset and register it in MLflow.

    X/y split is done here: 'date' and 'target' columns are excluded from features.
    Registration logic (champion vs challenger) is handled inside run_training.

    Args:
        data (pd.DataFrame): Processed dataset including 'date' and 'target' columns
        dataset_metadata (dict): Metadata logged to MLflow alongside the model

    Returns:
        tuple: (trained_model, model_version str)
    """
    X = data.drop(columns=["target", "date"])
    y = data["target"]
    #TODO log model version + fit time
    return run_training(X, y, dataset_metadata=dataset_metadata)


@task
def evaluate_model(data, dataset_metadata, alias, eval_mode):
    """Evaluate a registered model and log metrics to MLflow.

    Args:
        data (pd.DataFrame): Dataset with features, 'date' and 'target' columns
        dataset_metadata (dict): Metadata about the evaluation dataset
        alias (str): MLflow model alias to load ('champion' or 'challenger')
        eval_mode (str): Evaluation context label logged to MLflow (e.g. 'test_set')

    Returns:
        tuple: (score: float, model_version: str)
    """
    X = data.drop(columns=["target", "date"])
    y_true = data["target"]
    #TODO log alias and eval mode
    #TODO log rmse
    score, model_version = run_evaluating(
        X_val=X,
        y_true=y_true,
        dataset_metadata=dataset_metadata,
        alias=alias,
        eval_mode=eval_mode
    )
    return score, model_version


# =============================================================================
# SUBFLOWS
# Each subflow handles one logical stage of the pipeline.
#
# Ingestion subflow: has a BQ existence check + force override.
# Preprocess / train / eval subflows: have a fallback (data reloaded from BQ
# if not passed in-memory, e.g. when running a subflow standalone after a crash).
# =============================================================================

@flow
def bootstrap_ingestion_subflow(start_date, end_date, force=False):
    """Fetch raw data from APIs and upload to BigQuery.

    If data already exists in BQ for the requested date range and force=False,
    the API ingestion is skipped and data is loaded directly from BQ.
    Use force=True to re-ingest and overwrite existing data (e.g. after a
    data quality issue or a change in ingestion logic).

    airqual and weather checks / downloads run in parallel.

    Args:
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        force (bool): If True, always re-ingest from APIs regardless of BQ state

    Returns:
        tuple: (airqual_df, weather_df)
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
def bootstrap_preprocess_subflow(start_date, end_date, airqual_df=None, weather_df=None, mode="train"):
    """Preprocess raw data and upload the processed dataset to BigQuery.

    Fallback: if airqual_df or weather_df are None (e.g. subflow run standalone
    after a crash), raw data is reloaded from BigQuery before preprocessing.
    The preprocessed dataset (features + date + target) is always uploaded to BQ.

    Args:
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        airqual_df (pd.DataFrame, optional): Raw air quality data passed in-memory
        weather_df (pd.DataFrame, optional): Raw weather data passed in-memory
        mode (str): 'train' applies sensor filtering; 'eval' skips it.

    Returns:
        tuple: (dataset_metadata dict, processed DataFrame)
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
def bootstrap_train_subflow(start_date, end_date, data=None, dataset_metadata=None):
    """Train a model on the processed dataset and register it in MLflow.

    Fallback: if data is None, processed dataset is reloaded from BigQuery.
    If dataset_metadata is None, it is reconstructed from the loaded DataFrame.

    Args:
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        data (pd.DataFrame, optional): Processed dataset passed in-memory
        dataset_metadata (dict, optional): Dataset metadata passed in-memory
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

    train_model.submit(data=data, dataset_metadata=dataset_metadata).result()


@flow
def bootstrap_eval_subflow(start_date, end_date, alias, eval_mode,
                           data=None, dataset_metadata=None):
    """Evaluate a registered model on the given dataset and log metrics to MLflow.

    Fallback: if data is None, processed dataset is reloaded from BigQuery.
    If dataset_metadata is None, it is reconstructed from the loaded DataFrame.

    Args:
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        alias (str): MLflow model alias to evaluate ('champion' or 'challenger')
        eval_mode (str): Evaluation context label logged to MLflow (e.g. 'test_set')
        data (pd.DataFrame, optional): Processed dataset passed in-memory
        dataset_metadata (dict, optional): Dataset metadata passed in-memory
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

    score, model_version = evaluate_model.submit(
        data, dataset_metadata, alias=alias, eval_mode=eval_mode
    ).result()

    return score, model_version


# =============================================================================
# MASTERFLOWS
# Entry points for the bootstrap pipeline. Orchestrate subflows sequentially.
#
# Data is passed in-memory between subflows to avoid redundant BQ round-trips:
#   ingestion → preprocess → train/eval
#
# The `force` parameter is propagated to the ingestion subflow only, since
# preprocess and train/eval always re-run on whatever data is passed to them.
# =============================================================================

@flow
def bootstrap_train_masterflow(force=False):
    """Bootstrap training pipeline: ingest → preprocess → train.

    Intended for the first run of the project. Ingests historical data
    (START_TRAIN_DATE_STR to END_TRAIN_DATE_STR), preprocesses it, trains
    a model, and registers it as champion in MLflow.

    Args:
        force (bool): If True, re-ingest data from APIs even if it already
                      exists in BigQuery. Default: False.
    """
    setup_mlflow()

    airqual_df, weather_df = bootstrap_ingestion_subflow(
        start_date=START_TRAIN_DATE_STR,
        end_date=END_TRAIN_DATE_STR,
        force=force
    )

    dataset_metadata, data = bootstrap_preprocess_subflow(
        start_date=START_TRAIN_DATE_STR,
        end_date=END_TRAIN_DATE_STR,
        airqual_df=airqual_df,
        weather_df=weather_df
    )

    bootstrap_train_subflow(
        start_date=START_TRAIN_DATE_STR,
        end_date=END_TRAIN_DATE_STR,
        data=data,
        dataset_metadata=dataset_metadata
    )


@flow
def bootstrap_eval_masterflow(force=False):
    """Bootstrap evaluation pipeline: ingest → preprocess → evaluate champion.

    Ingests test set data (START_TEST_DATE_STR to END_TEST_DATE_STR),
    preprocesses it, and evaluates the current champion model.
    Metrics are logged to MLflow.

    Args:
        force (bool): If True, re-ingest data from APIs even if it already
                      exists in BigQuery. Default: False.
    """
    setup_mlflow()

    airqual_df, weather_df = bootstrap_ingestion_subflow(
        start_date=START_TEST_DATE_STR,
        end_date=END_TEST_DATE_STR,
        force=force
    )

    dataset_metadata, data = bootstrap_preprocess_subflow(
        start_date=START_TEST_DATE_STR,
        end_date=END_TEST_DATE_STR,
        airqual_df=airqual_df,
        weather_df=weather_df,
        mode="eval"
    )

    score, model_version= bootstrap_eval_subflow(
        start_date=START_TEST_DATE_STR,
        end_date=END_TEST_DATE_STR,
        alias="champion",
        eval_mode="test_set",
        data=data,
        dataset_metadata=dataset_metadata
    )

    model_metadata = {"model_version": model_version,
                      "train_start": START_TRAIN_DATE_STR,
                      "train_end": END_TRAIN_DATE_STR,
                      "eval_start": START_TEST_DATE_STR,
                      "eval_end": END_TEST_DATE_STR,
                      "ref_rmse": score,
                      "alias": "champion"

                      }
    MonitoringClient().upsert_model(model_metadata)
