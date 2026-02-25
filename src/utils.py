from pathlib import Path
import pandas as pd

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


def clean_neg_values(df, col_to_clean="pm25_avg"):
    """
    Remove rows with negative or zero values in the target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_to_clean (str): Column to check for negative/zero values. Defaults to 'pm25_avg'.

    Returns:
        pd.DataFrame: Cleaned DataFrame with non-positive values removed.
    """
    if col_to_clean not in df.columns:
        raise KeyError(f"❌ Column '{col_to_clean}' not found in DataFrame")

    total_values = len(df)
    neg_values_before = (df[col_to_clean] <= 0).sum()
    print(f"⚠️  {neg_values_before} aberrant (negative or zero) values found ({neg_values_before / total_values * 100:.2f}%)")

    df_clean = df[df[col_to_clean] > 0].copy()

    neg_values_after = (df_clean[col_to_clean] <= 0).sum()
    if neg_values_after == 0:
        print(f"✅ All negative values removed — {len(df_clean)} rows remaining")
        return df_clean
    else:
        raise Exception(f"❌ Not all negative values removed. {neg_values_after} remaining")


def merge_source_df(df_airqual, df_weather):
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
        raise Exception("⚠️ Missing days detected in weather dataframe. Please check")

    df_merged = pd.merge(left=df_weather, right=df_airqual, on=["date", "city"], how="left")

    new_order = ["city", "sensor_id", "date", "pm25_avg", "pm25_min", "pm25_q25", "pm25_median",
                 "pm25_q75", "pm25_max", "coverage", "temp_min", "temp_max", "temp_avg",
                 "cloud_cover", "humidity", "precipitation", "pressure", "wind_speed", "wind_direction"]


    df_merged = df_merged[new_order]

    if df_merged.empty:
        raise Exception("❌ Merged DataFrame is empty")

    matched = df_merged["pm25_avg"].notna().sum()
    total = len(df_merged)
    print(f"✅ DataFrames merged successfully — {matched}/{total} days have airqual data ({matched / total * 100:.1f}%)")
    return df_merged
