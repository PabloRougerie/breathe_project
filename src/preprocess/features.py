
import pandas as pd

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
