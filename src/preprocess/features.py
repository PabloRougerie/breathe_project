
import pandas as pd
import numpy as np

def shift_by_city(df,col, shift):
    return df.groupby("city")["col"].shift(shift)

def target_transform(df, horizon= 1):
    """ log transform of target column"""
    df["target"] = np.log1p(df["target"])

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




def generate_lag_features():

    """ general fonction that shift a given column by a given number of steps.
    necessary for target generation, lag feature generation, month and day feature tomorrow"""

def average_feature_generation(df, shifts = [12,13,14]):
    """generate avg at lag 13,14,15 counting from target day """

    df["lag_avg_14"]= pd.concat(
                    [df.groupby("city")["pm25_avg"].shift(shift) for shift in shifts],
                    axis= 1).mean(axis= 1, skipna= False)

    return df


def std_feature_generation(df, time_window= 7):
    """ genrated std value for the week counting from target day"""

    df["week_std"] = df.groupby("city").transform(lambda x : x.rolling(time_window).std())

    return df

def feature_engineering():
    """ wrapper function calling all the feature engineering functions"""
