from pathlib import Path
import pandas as pd
import json
import os
from src.params import *
from google.cloud import bigquery

from abc import ABC, abstractmethod
from google.cloud import storage


class StorageClient(ABC):
    """Abstract interface for structured DataFrame persistence (raw and processed data).

    Concrete implementations handle local CSV files or BigQuery tables.
    All methods are keyed by:
        - data_type: 'weather', 'airqual', or 'processed'
        - start_date / end_date: date range strings (YYYY-MM-DD)

    File/table naming convention:
        Local : {base_storage_dir}/raw/{data_type}_{start_date}_{end_date}.csv
                {base_storage_dir}/processed/{data_type}_{start_date}_{end_date}.csv
        BQ    : {GCP_PROJECT}.{BQ_DATASET_RAW}.{data_type}
                {GCP_PROJECT}.{BQ_DATASET_PROCESSED}.{data_type}
    """

    def __init__(self):
        pass

    @abstractmethod
    def save_data(self, data: pd.DataFrame, data_type: str, start_date: str, end_date: str):
        """Persist a DataFrame for the given data type and date range."""
        pass

    @abstractmethod
    def get_data(self, data_type: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load and return a DataFrame for the given data type and date range."""
        pass


class LocalStorageClient(StorageClient):
    """StorageClient backed by the local filesystem.

    Saves and loads CSV files under:
        {base_storage_dir}/raw/      for 'weather' and 'airqual'
        {base_storage_dir}/processed/ for 'processed'

    Args:
        base_storage_dir (Path | str): Root data directory, e.g. PROJECT_ROOT / 'data'
    """

    def __init__(self, base_storage_dir):
        super().__init__()
        self.base_storage_dir = base_storage_dir

    def get_data(self, data_type, start_date, end_date):
        """Load CSV from local filesystem and return as DataFrame.

        The 'date' column is parsed as datetime on load (csv stores dates as strings).

        Args:
            data_type (str): 'weather', 'airqual', or 'processed'
            start_date (str): Start date (YYYY-MM-DD) — part of the filename
            end_date (str): End date (YYYY-MM-DD) — part of the filename

        Returns:
            pd.DataFrame
        """
        if data_type not in ["weather", "airqual", "processed"]:
            raise ValueError(f"data_type must be 'weather', 'airqual', or 'processed', got '{data_type}'")

        subfolder = "raw" if data_type in ["weather", "airqual"] else "processed"
        path = Path(self.base_storage_dir) / subfolder / f"{data_type}_{start_date}_{end_date}.csv"

        if not path.exists():
            raise FileNotFoundError(f"❌ File not found: {path}")

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])  # csv stores dates as strings, reparse to datetime

        print(f"✅ Loaded {len(df)} rows from {path}")
        return df

    def save_data(self, data, data_type, start_date, end_date):
        """Save DataFrame as CSV to local filesystem.

        Args:
            data (pd.DataFrame): DataFrame to save
            data_type (str): 'weather', 'airqual', or 'processed'
            start_date (str): Start date (YYYY-MM-DD) — included in filename
            end_date (str): End date (YYYY-MM-DD) — included in filename
        """
        if data_type not in ["weather", "airqual", "processed"]:
            raise ValueError(f"data_type must be 'weather', 'airqual', or 'processed', got '{data_type}'")

        subfolder = "raw" if data_type in ["weather", "airqual"] else "processed"
        path = Path(self.base_storage_dir) / subfolder / f"{data_type}_{start_date}_{end_date}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        print(f"✅ Saved {len(data)} rows to {path}")


class GCSStorageClient(StorageClient):
    """StorageClient backed by Google BigQuery.

    Raw data ('weather', 'airqual') is stored in BQ_DATASET_RAW.
    Processed data ('processed') is stored in BQ_DATASET_PROCESSED.
    Table name == data_type (e.g. project.raw_dataset.weather).

    save_data uses DELETE + WRITE_APPEND to avoid duplicates:
    existing rows for the date range are deleted before inserting new ones.
    This makes the operation idempotent (safe to re-run on the same period).
    """

    def __init__(self):
        super().__init__()
        self.bq_client = bigquery.Client(project=GCP_PROJECT)

    def get_data(self, data_type, start_date, end_date):
        """Query BigQuery and return DataFrame for the given date range.

        BQ returns date columns as datetime natively: no extra parsing needed.

        Args:
            data_type (str): 'weather', 'airqual', or 'processed'
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)

        Returns:
            pd.DataFrame
        """
        if data_type not in ["weather", "airqual", "processed"]:
            raise ValueError(f"data_type must be 'weather', 'airqual', or 'processed', got '{data_type}'")

        dataset = BQ_DATASET_RAW if data_type in ["weather", "airqual"] else BQ_DATASET_PROCESSED
        full_table_name = f"{GCP_PROJECT}.{dataset}.{data_type}"

        query = f"""
            SELECT *
            FROM `{full_table_name}`
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """

        df = self.bq_client.query(query).result().to_dataframe()
        print(f"✅ Loaded {len(df)} rows from {full_table_name} ({start_date} → {end_date})")
        return df

    def save_data(self, data, data_type, start_date, end_date):
        """Save DataFrame to BigQuery using DELETE + WRITE_APPEND.

        Deletes existing rows for the date range first to prevent duplicates,
        then appends all rows. The DELETE is wrapped in try/except to handle
        the first-run case where the table does not exist yet.

        Args:
            data (pd.DataFrame): DataFrame to save
            data_type (str): 'weather', 'airqual', or 'processed'
            start_date (str): Start date (YYYY-MM-DD) — used in DELETE filter
            end_date (str): End date (YYYY-MM-DD) — used in DELETE filter
        """
        if data_type not in ["weather", "airqual", "processed"]:
            raise ValueError(f"data_type must be 'weather', 'airqual', or 'processed', got '{data_type}'")

        dataset = BQ_DATASET_RAW if data_type in ["weather", "airqual"] else BQ_DATASET_PROCESSED
        full_table_name = f"{GCP_PROJECT}.{dataset}.{data_type}"

        # Delete rows in the date range before inserting to avoid duplicates on re-run
        try:
            delete_job = self.bq_client.query(f"""
                DELETE FROM `{full_table_name}`
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
            """)
            delete_job.result()  # wait for completion; dml_stats lives on the job, not the result
            deleted = delete_job.dml_stats.deleted_row_count
            if deleted > 0:
                print(f"Deleted {deleted} rows from {full_table_name} ({start_date} → {end_date})")
        except Exception:
            pass  # table doesn't exist yet on first run: skip delete

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            autodetect=True  # infer schema from DataFrame; creates table if it doesn't exist
        )
        self.bq_client.load_table_from_dataframe(data, full_table_name, job_config=job_config).result()
        print(f"✅ Saved {len(data)} rows to {full_table_name} ({start_date} → {end_date})")




class CacheClient(ABC):
    """Abstract interface for JSON cache storage.

    All methods use a logical file_name / prefix of the form:
        {city}/{api_source}/{filename}.json
    e.g. "Paris/weather/weather_2023-01-01.json"
         "Paris/sensor_12345.json"

    Concrete implementations resolve this logical name to a physical
    location (local filesystem or GCS bucket).
    """

    def __init__(self):
        pass

    @abstractmethod
    def read(self, file_name: str) -> dict:
        """Load and return a JSON file as a dict. file_name: logical path relative to root."""
        pass

    @abstractmethod
    def write(self, data: dict, file_name: str):
        """Serialize data as JSON and write to file_name."""
        pass

    @abstractmethod
    def exists(self, file_name: str) -> bool:
        """Return True if the file at file_name exists."""
        pass

    @abstractmethod
    def list(self, prefix: str) -> list:
        """Return list of file_names (logical paths) matching the given prefix.
        e.g. prefix='Paris/weather' returns ['Paris/weather/weather_2023-01-01.json', ...]
        Returned names are directly passable to read().
        """
        pass


class LocalCacheClient(CacheClient):
    """CacheClient backed by the local filesystem.

    Args:
        cache_dir (Path): Absolute base directory for all cache files.
                          e.g. PROJECT_ROOT / 'data' / 'cache'

    Full path resolution: cache_dir / file_name
        e.g. /project/data/cache/Paris/weather/weather_2023-01-01.json
    """

    def __init__(self, cache_dir):
        super().__init__()
        self.cache_dir = cache_dir

    def write(self, data, file_name):
        # full path = cache_dir / {city}/{api_source}/{filename}.json
        path = Path(self.cache_dir) / file_name
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def read(self, file_name):
        path = Path(self.cache_dir) / file_name
        with open(path, "r") as f:
            return json.load(f)

    def exists(self, file_name):
        path = Path(self.cache_dir) / file_name
        if os.path.exists(path):
            return True
        else:
            return False

    def list(self, prefix):
        # prefix resolves to a directory e.g. cache_dir/Paris/weather/
        # glob("*.json") lists files in that dir only (non-recursive)
        # relative_to(cache_dir) gives back the logical file_name passable to read()
        path = Path(self.cache_dir) / prefix
        file_list = [str(file.relative_to(self.cache_dir)) for file in path.glob("*.json")]
        return file_list


class GCSCacheClient(CacheClient):
    """CacheClient backed by Google Cloud Storage.

    Args:
        bucket_name (str): GCS bucket name (from BUCKET_NAME in params).

    Blob name == file_name (logical path), e.g.:
        'Paris/weather/weather_2023-01-01.json'
        'Paris/sensor_12345.json'
    """

    def __init__(self, bucket_name):
        super().__init__()
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def read(self, file_name):
        blob = self.bucket.blob(file_name)
        return json.loads(blob.download_as_text())

    def write(self, data, file_name):
        blob = self.bucket.blob(file_name)
        blob.upload_from_string(data=json.dumps(data), content_type="application/json")

    def exists(self, file_name):
        blob = self.bucket.blob(file_name)
        return blob.exists()

    def list(self, prefix):
        # list_blobs returns all blobs whose name starts with prefix
        # blob.name is the full blob path = logical file_name passable to read()
        blobs = self.client.list_blobs(BUCKET_NAME, prefix=prefix)
        blob_list = [blob.name for blob in blobs]
        return blob_list








def filter_columns(df, col_to_keep=None, col_to_remove=None):
    """
    Select or drop columns from a DataFrame.

    At least one of col_to_keep or col_to_remove must be provided.
    Both can be passed simultaneously: col_to_keep is applied first,
    then col_to_remove — but no column can appear in both.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_to_keep (list[str], optional): Columns to keep.
        col_to_remove (list[str], optional): Columns to drop.

    Returns:
        pd.DataFrame: DataFrame with selected/dropped columns.
    """
    if not (col_to_keep or col_to_remove):
        raise ValueError("At least one of col_to_keep or col_to_remove must be provided")

    overlap= set(col_to_keep or []) & set(col_to_remove or [])
    df_filtered = df.copy()

    if overlap:
        raise ValueError(f"Columns cannot be in both col_to_keep and col_to_remove: {overlap}")

    df_filtered = df.copy()

    if col_to_keep:
        df_filtered = df_filtered[col_to_keep]
    if col_to_remove:
        df_filtered = df_filtered.drop(columns=col_to_remove)

    return df_filtered


def merge_source_df(df_airqual, df_weather, col_order = ["city", "date", "pm25_avg",
                                                         "temp_min", "temp_max",
                                                        "cloud_cover", "humidity", "precipitation",
                                                        "pressure", "wind_speed", "wind_direction"]):
    """
    Merge air quality and weather DataFrames on date and city.

    Performs a left join on weather (complete date range), so missing airqual
    days appear as NaN. Raises if weather has gaps or if the merge produces an empty result.

    Args:
        df_airqual (pd.DataFrame): Air quality DataFrame with 'date' and 'city' columns.
        df_weather (pd.DataFrame): Weather DataFrame with 'date' and 'city' columns.

    Returns:
        pd.DataFrame: Merged DataFrame with standardized column order.
    """
    df_weather = df_weather.sort_values(["city", "date"])
    max_gap = df_weather.groupby("city")["date"].diff().dt.days.max()

    if max_gap > 1:
        raise ValueError("⚠️ Missing days detected in weather dataframe. Please check")

    df_merged = pd.merge(left=df_weather, right=df_airqual, on=["date", "city"], how="left")


    if set(col_order) != set(df_merged.columns):
        missing = set(col_order) - set(df_merged.columns)
        remaining = set(df_merged.columns) - set(col_order)
        raise Exception(f"❌ col mismatch: absent in merged df: {missing}, remaining and not reordered: {remaining}")

    df_merged = df_merged[col_order]

    if df_merged.empty:
        raise ValueError("❌ Merged DataFrame is empty")

    matched = df_merged["pm25_avg"].notna().sum()
    total = len(df_merged)
    print(f"✅ DataFrames merged successfully — {matched}/{total} days have airqual data ({matched / total * 100:.1f}%)")
    return df_merged
