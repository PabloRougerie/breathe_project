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


    df = pd.read_csv(filepath)

    #convert to datetime
    if source == "weather":
        df["date"] = pd.to_datetime(df["date"])

    elif source == "airqual":
        df["date"] = pd.to_datetime(df["date_from_local"].str[:10])

    else:

        raise Exception(f"❌ source must be either 'weather' or 'airqual' {source} was passed instead")

    return df

def merge_source_df(df_airqual, df_weather):

    #check no missing days in weather
    df_weather = df_weather.sort_values(["city", "date"])
    has_gaps = df_weather.groupby("city")["date"].diff().dt.days.max() != 1

    if has_gaps:
        raise Exception("⚠️ Missing days detected in weather dataframe. Please check")


    else:
        df_merged = pd.merge(left= df_weather, right= df_airqual, on=["date", "city"], how= "left")
        new_order = ["city", "sensor_id","date", 'pm25_avg', 'pm25_min', 'pm25_q25', 'pm25_median',
       'pm25_q75', 'pm25_max', 'coverage','temp_min',
       'temp_max', 'temp_avg', 'cloud_cover', 'humidity', 'precipitation',
       'pressure', 'wind_speed', 'wind_direction' ]
        df_merged = df_merged[new_order]

        if df_merged.empty:
            raise Exception("❌ No DataFrame merged")

        else:
            print("✅ DataFrames merged successfully. Missing days in airqual will appear as NaN")
            return df_merged
