
import pandas as pd
import numpy as np
from src.params import CUSTOM_SHIFTS, ALL_LAGS_14_SHIFTS, ALL_LAGS_21_SHIFTS

def shift_by_city(df,col, shift):
    return df.groupby("city")[col].shift(shift)

def generate_target(df, horizon = 1):
    df["target"] = df.groupby("city")["pm25_avg"].shift(- horizon)
    return df

def target_transform(df):
    """ log transform of target column"""
    df["target"] = np.log1p(df["target"])
    return df


def month_encoding(df):
    """ encode month as sin cos"""
    df["month_cos"] = np.cos(df["date"].dt.month * 2*np.pi / 12)
    df["month_sin"] = np.sin(df["date"].dt.month * 2*np.pi / 12)
    return df

def day_encoding(df):

    """ encode days of week and add 'is weekend' feature """
    df["dow"] = df["date"].dt.day_of_week
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)

    return df




def average_feature_generation(df, shifts = [12,13,14]):
    """generate avg at lag 13,14,15 counting from target day """

    df["lag_avg_14"]= pd.concat(
                    [df.groupby("city")["pm25_avg"].shift(shift) for shift in shifts],
                    axis= 1).mean(axis= 1, skipna= False)

    return df


def std_feature_generation(df, time_window= 7):
    """ genrated std value for the week counting from target day"""

    df["week_std"] = df.groupby("city")["pm25_avg"].transform(lambda x : x.rolling(time_window).std())

    return df


def generate_lag_features(df, shifts: dict):

    """ general fonction that shift a given column by a given number of steps.
    dictionary of shape {output_col : (input_col,shift)}
    """

    for output_col, input in shifts.items():
        df[output_col] = shift_by_city(df, col= input[0], shift= input[1])

    return df



def feature_engineering(df, approach= "custom"):
    """ wrapper function calling all the feature engineering functions"""

    # generate time features

    df = month_encoding(df)
    df = day_encoding(df)

    #generate lag features

    if approach == "custom":

        df = average_feature_generation(df) # average pm25 value in the past, on lag 13-15 by default
        df = std_feature_generation(df) #add pm25 variability over most recent week by default

        df = generate_lag_features(df, shifts= CUSTOM_SHIFTS)

    elif approach == "all_lags_14":
        df = generate_lag_features(df, shifts= ALL_LAGS_14_SHIFTS)

    elif approach == "all_lags_21":
        df = generate_lag_features(df, shifts= ALL_LAGS_21_SHIFTS)

    else:
        raise ValueError(f"approach parameter must be either 'custom', 'all_lags_14' or 'all_lags_21', got {approach} instead")

    return df
