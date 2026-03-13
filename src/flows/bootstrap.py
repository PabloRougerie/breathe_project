import pandas as pd
from prefect import task, flow

from src.params import *
from src.utils import *
from src.ingestion.openaq import OpenAQClient
from src.ingestion.openweather import OpenWeatherClient
from src.preprocess.preproc_pipeline import preprocessing_pipeline
from src.models.model_pipeline import run_training, run_evaluating


# =============================================================================
# TASKS
# GCSStorageClient is instantiated inside tasks (not passed as arg) to avoid
# serialization issues with Prefect task runners.
# =============================================================================

@task
def train_set_ingestion(start_date, end_date):
    """Fetch air quality and weather data from APIs for the given date range.

    Uses cloud cache (GCS) to avoid redundant API calls.

    Returns:
        tuple: (airqual_df, weather_df)
    """
    aq_client = OpenAQClient(api_key=API_AQ, storage="cloud")
    airqual_df = aq_client.get_data(
        cities=CITIES,
        start_date=start_date,
        end_date=end_date,
        start_project_date=START_PROJECT_DATE_STR,
        end_project_date=END_PROJECT_DATE_STR
    )
    weather_client = OpenWeatherClient(api_key=API_OW, storage="cloud")
    weather_df = weather_client.get_all_data(
        cities=CITIES,
        start_date=start_date,
        end_date=end_date
    )
    return airqual_df, weather_df


@task
def upload_data(df, data_type, start_date, end_date):
    """Save a DataFrame to BigQuery (raw or processed dataset).

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
    return GCSStorageClient().get_data(
        data_type=data_type,
        start_date=start_date,
        end_date=end_date
    )


@task
def preprocess_raw_data(airqual_df, weather_df):
    """Run the full preprocessing pipeline on raw DataFrames.

    Returns:
        tuple: (dataset_metadata dict, preprocessed DataFrame with date + target + features)
    """
    return preprocessing_pipeline(airqual_df=airqual_df, weather_df=weather_df)


@task
def train_model(data, dataset_metadata):
    """Train model on the given dataset and register it in MLflow.

    Splits data into X/y internally. In the bootstrap flow, the model
    is registered as champion directly.

    Returns:
        tuple: (trained_model, model_version)
    """
    X = data.drop(columns=["target", "date"])
    y = data["target"]
    return run_training(X, y, dataset_metadata=dataset_metadata)


@task
def evaluate_model(data, dataset_metadata, alias, eval_mode):
    """Evaluate a registered model and log metrics to MLflow.

    Args:
        data (pd.DataFrame): Dataset with features, date and target columns
        dataset_metadata (dict): Metadata about the evaluation dataset
        alias (str): MLflow model alias to evaluate ('champion' or 'challenger')
        eval_mode (str): Evaluation context label (e.g. 'test_set', 'train_set')

    Returns:
        float: RMSE score
    """
    X = data.drop(columns=["target", "date"])
    y_true = data["target"]
    return run_evaluating(
        X=X,
        y_true=y_true,
        dataset_metadata=dataset_metadata,
        alias=alias,
        eval_mode=eval_mode
    )


# =============================================================================
# SUBFLOWS
# Each subflow handles one stage of the pipeline with a failsafe:
# if data is not passed (e.g. after a crash), it is reloaded from BigQuery.
# =============================================================================

@flow
def bootstrap_ingestion_subflow(start_date, end_date):
    """Fetch raw data from APIs and upload to BigQuery.

    airqual and weather uploads run in parallel.

    Returns:
        tuple: (airqual_df, weather_df)
    """
    ingestion_f = train_set_ingestion.submit(start_date=start_date, end_date=end_date)
    airqual_df, weather_df = ingestion_f.result()

    # upload airqual and weather in parallel
    upload_airqual_f = upload_data.submit(airqual_df, data_type="airqual",
                                          start_date=start_date, end_date=end_date)
    upload_weather_f = upload_data.submit(weather_df, data_type="weather",
                                          start_date=start_date, end_date=end_date)
    upload_airqual_f.result()
    upload_weather_f.result()

    return airqual_df, weather_df


@flow
def bootstrap_preprocess_subflow(start_date, end_date, airqual_df=None, weather_df=None):
    """Preprocess raw data and upload processed dataset to BigQuery.

    Fallback: if airqual_df or weather_df are None (e.g. subflow run standalone
    after a crash), raw data is reloaded from BigQuery before preprocessing.

    Returns:
        tuple: (dataset_metadata dict, processed DataFrame)
    """
    if airqual_df is None or weather_df is None:
        airqual_f = download_data.submit(data_type="airqual",
                                         start_date=start_date, end_date=end_date)
        weather_f = download_data.submit(data_type="weather",
                                         start_date=start_date, end_date=end_date)
        airqual_df, weather_df = airqual_f.result(), weather_f.result()

    preproc_f = preprocess_raw_data.submit(airqual_df, weather_df)
    dataset_metadata, data = preproc_f.result()

    upload_data.submit(df=data, data_type="processed",
                       start_date=start_date, end_date=end_date).result()

    return dataset_metadata, data


@flow
def bootstrap_train_subflow(start_date, end_date, data=None, dataset_metadata=None):
    """Train model on the processed dataset and register it in MLflow.

    Fallback: if data is None, processed dataset is reloaded from BigQuery.
    If dataset_metadata is None, it is reconstructed from the loaded DataFrame.
    """
    if data is None:
        data = download_data.submit(data_type="processed",
                                    start_date=start_date, end_date=end_date).result()

    if dataset_metadata is None:
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
    """Evaluate the registered model on the given dataset and log metrics to MLflow.

    Fallback: if data is None, processed dataset is reloaded from BigQuery.
    If dataset_metadata is None, it is reconstructed from the loaded DataFrame.

    Args:
        alias (str): MLflow model alias to evaluate ('champion' or 'challenger')
        eval_mode (str): Evaluation context label (e.g. 'test_set')
    """
    if data is None:
        data = download_data.submit(data_type="processed",
                                    start_date=start_date, end_date=end_date).result()

    if dataset_metadata is None:
        dataset_metadata = {
            "date_start":    str(data["date"].min()),
            "date_end":      str(data["date"].max()),
            "n_rows":        len(data),
            "n_features":    len(data.columns) - 2,
            "list_features": [f for f in data.columns if f not in ["date", "target"]],
        }

    evaluate_model.submit(data, dataset_metadata,
                          alias=alias, eval_mode=eval_mode).result()


# =============================================================================
# MASTERFLOWS
# Top-level flows that orchestrate subflows sequentially.
# Data is passed in-memory between subflows to avoid redundant BQ round-trips.
# =============================================================================

@flow
def bootstrap_train_masterflow():
    """Bootstrap training flow: ingest historical data, preprocess, and train first model.

    Runs sequentially: ingestion, preprocess, train.
    Data is passed in-memory between stages (no BQ round-trips between subflows).
    The trained model is registered as champion in MLflow.
    """
    airqual_df, weather_df = bootstrap_ingestion_subflow.submit(
        start_date=START_TRAIN_DATE_STR,
        end_date=END_TRAIN_DATE_STR
    ).result()

    dataset_metadata, data = bootstrap_preprocess_subflow.submit(
        start_date=START_TRAIN_DATE_STR,
        end_date=END_TRAIN_DATE_STR,
        airqual_df=airqual_df,
        weather_df=weather_df
    ).result()

    bootstrap_train_subflow.submit(
        start_date=START_TRAIN_DATE_STR,
        end_date=END_TRAIN_DATE_STR,
        data=data,
        dataset_metadata=dataset_metadata
    ).result()


@flow
def bootstrap_eval_masterflow():
    """Bootstrap evaluation flow: ingest test data, preprocess, and evaluate champion model.

    Runs sequentially: ingestion, preprocess,eval.
    Evaluates the champion model on the test set and logs metrics to MLflow.
    """
    airqual_df, weather_df = bootstrap_ingestion_subflow.submit(
        start_date=START_TEST_DATE_STR,
        end_date=END_TEST_DATE_STR
    ).result()

    dataset_metadata, data = bootstrap_preprocess_subflow.submit(
        start_date=START_TEST_DATE_STR,
        end_date=END_TEST_DATE_STR,
        airqual_df=airqual_df,
        weather_df=weather_df
    ).result()

    bootstrap_eval_subflow.submit(
        start_date=START_TEST_DATE_STR,
        end_date=END_TEST_DATE_STR,
        alias="champion",
        eval_mode="test_set",
        data=data,
        dataset_metadata=dataset_metadata
    ).result()
