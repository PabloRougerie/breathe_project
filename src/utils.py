from pathlib import Path
import pandas as pd
import json
import os

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

    def __init__(self):
        pass

    @abstractmethod
    def read(self, path):
        pass

    @abstractmethod
    def write(self,data, path):
        pass

    @abstractmethod
    def exists(self,path):
        pass

class LocalStorageClient(StorageClient):
        def __init__(self, cache_dir):
            super().__init__()
            self.cache_dir = cache_dir

        def write(self, data, file_name):

            path = Path(self.cache_dir) / file_name #create all path until json file
            os.makedirs(path.parent, exist_ok= True) #create all dir from root to json file
            with open(path, "w") as f: #write within the path including a json, meaning that it will crate a json file
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



class GCSStorageClient(StorageClient):

        def __init__(self, bucket_name):
            super().__init__()

            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)

        def read(self,blob_name):
            blob = self.bucket.blob(blob_name)
            return json.loads(blob.download_as_text())

        def write(self, data, blob_name):
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(data= json.dumps(data), content_type="application/json")

        def exists(self, blob_name):

            blob = self.bucket.blob(blob_name)

            return blob.exists()








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
