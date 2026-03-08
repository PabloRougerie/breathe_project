from pathlib import Path
import pandas as pd
import json
import os
from src.params import *

from abc import ABC, abstractmethod
from google.cloud import storage

def save_data_local(df, output_path):
    """
    Save DataFrame to local CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Output file path

    Returns:
        None
    """
    # Create parent directories if they don't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} rows to {output_path}")


def load_data_local(filepath, source: str):
    """
    Load a CSV file and parse the date column.

    Args:
        filepath (str): Path to the CSV file.
        source (str): Either 'weather' or 'airqual'.

    Returns:
        pd.DataFrame: Loaded DataFrame with a parsed 'date' column.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"❌ File not found: {filepath}")

    df = pd.read_csv(filepath)

    if source == "weather":
        df["date"] = pd.to_datetime(df["date"])

    elif source == "airqual":
        if "date_from_local" not in df.columns:
            raise KeyError("❌ Column 'date_from_local' not found in airqual dataframe")
        df["date"] = pd.to_datetime(df["date_from_local"].str[:10])

    else:
        raise ValueError(f"❌ source must be 'weather' or 'airqual', got '{source}'")

    print(f"✅ Loaded {len(df)} rows from {filepath}")
    return df




class StorageClient(ABC):
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


class LocalStorageClient(StorageClient):
    """StorageClient backed by the local filesystem.

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


class GCSStorageClient(StorageClient):
    """StorageClient backed by Google Cloud Storage.

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

    def read(self, blob_name):
        blob = self.bucket.blob(blob_name)
        return json.loads(blob.download_as_text()) #download

    def write(self, data, blob_name):
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(data=json.dumps(data), content_type="application/json")

    def exists(self, blob_name):
        blob = self.bucket.blob(blob_name)
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
