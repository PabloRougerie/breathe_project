import pandas as pd

def get_bad_sensors_gap(df, max_gap, max_q):
    """
    Flag sensors with excessive gaps in their time series.

    A sensor is flagged if its longest gap >= max_gap OR
    the 75th percentile of its gap distribution >= max_q.

    Args:
        df (pd.DataFrame): Airqual DataFrame with 'sensor_id' and 'date' columns.
        max_gap (int): Max acceptable gap duration in days.
        max_q (float): Max acceptable 75th percentile of gap distribution.

    Returns:
        set: sensor_ids to exclude.
    """
    df = df.sort_values(by=["sensor_id", "date"]).copy()
    df["delta"] = df.groupby("sensor_id")["date"].diff().dt.days

    gaps = df[df["delta"] > 1]
    gap_stats = gaps.groupby(["city", "sensor_id"])["delta"].agg(
        ["max", ("q75", lambda x: x.quantile(0.75))]
    )

    bad = set(
        gap_stats[(gap_stats["max"] >= max_gap) | (gap_stats["q75"] >= max_q)]
        .index.get_level_values("sensor_id")
        .tolist()
    )
    print(f"  [gap filter] {len(bad)} sensor(s) flagged — max_gap={max_gap}d, max_q75={max_q}d")
    return bad


def get_bad_sensors_coverage(df, min_coverage_pct, min_bad_month_pct):
    """
    Flag sensors with too many months of low daily coverage.

    A month is 'bad' if its daily coverage < min_coverage_pct%.
    A sensor is flagged if its ratio of bad months > min_bad_month_pct.

    Args:
        df (pd.DataFrame): Airqual DataFrame with 'sensor_id', 'city', and 'date' columns.
        min_coverage_pct (int): Min % of days in a month to consider it a good month.
        min_bad_month_pct (float): Max ratio of bad months before sensor is excluded.

    Returns:
        set: sensor_ids to exclude.
    """
    df = df.sort_values(by=["sensor_id", "date"]).copy()

    coverage_monthly = (
        df.groupby(["city", "sensor_id", pd.Grouper(key="date", freq="ME")])
        .size()
        .reset_index(name="nb_readings")
    )
    coverage_monthly["days_in_month"] = coverage_monthly["date"].dt.days_in_month
    coverage_monthly["coverage_pct"] = coverage_monthly["nb_readings"] / coverage_monthly["days_in_month"] * 100
    coverage_monthly["bad_month"] = (coverage_monthly["coverage_pct"] < min_coverage_pct).astype(int)

    bad_month_ratio = (
        coverage_monthly.groupby(["city", "sensor_id"])["bad_month"]
        .mean()
        .reset_index(name="bad_month_pct")
    )

    bad = set(
        bad_month_ratio[bad_month_ratio["bad_month_pct"] > min_bad_month_pct]["sensor_id"].tolist()
    )
    print(f"  [coverage filter] {len(bad)} sensor(s) flagged — min_cov={min_coverage_pct}%, max_bad_months={min_bad_month_pct:.0%}")
    return bad

def filter_sensors(df, max_gap, max_q, min_coverage_pct, min_bad_month_pct):
    """
    Filter out bad-quality sensors from the airqual DataFrame.

    Combines gap-based and coverage-based criteria: a sensor is removed
    if flagged by either filter (union).

    Args:
        df (pd.DataFrame): Airqual DataFrame with 'sensor_id', 'city', and 'date' columns.
        max_gap (int): Max acceptable gap duration in days.
        max_q (float): Max acceptable 75th percentile of gap distribution.
        min_coverage_pct (int): Min % of days in a month to consider it a good month.
        min_bad_month_pct (float): Max ratio of bad months before sensor is excluded.

    Returns:
        pd.DataFrame: Filtered DataFrame with bad sensors removed.
    """
    bad = get_bad_sensors_gap(df, max_gap, max_q) | get_bad_sensors_coverage(df, min_coverage_pct, min_bad_month_pct)

    df_filtered = df[~df["sensor_id"].isin(bad)].copy()

    n_removed = df["sensor_id"].nunique() - df_filtered["sensor_id"].nunique()
    print(f"✅ filter_sensors: {n_removed} sensor(s) removed, {df_filtered['sensor_id'].nunique()} remaining")
    return df_filtered


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



def average_sensors(df, col_to_average= "pm25_avg"):
    """
    Average PM2.5 values across sensors for each city/date.

    After sensor filtering, multiple sensors may still exist per city.
    This produces one row per city/date by taking the mean of all pm25 columns.

    Args:
        df (pd.DataFrame): Filtered airqual DataFrame.

    Returns:
        pd.DataFrame: One row per (city, date) with averaged pm25 columns.
    """


    df_avg = (
        df.groupby(["city", "date"])[col_to_average]
        .mean()
        .reset_index()
    )

    print(f"✅ Sensors averaged — {df_avg.shape[0]} rows (city × date)")
    return df_avg


def single_gaps_imputer():
    """ interpolate one day gap"""
    pass

def target_transform():
    """ log transform"""
    pass

def month_encoding():
    """ encode month as sin cos"""
    pass

def day_encoding():

    """ encode days of week and add 'is weekend' feature """
    pass

def generate_lag_features():

    """ general fonction that shift a given column by a given number of steps.
    necessary for target generation, lag feature generation, month and day feature tomorrow"""

def average_feature_generation():
    """generate avg at lag 13,14,15 counting from target day """
    pass

def std_feature_generation():
    """ genrated std value for the week counting from target day"""
    pass

def feature_engineering():
    """ wrapper function calling all the feature engineering functions"""
