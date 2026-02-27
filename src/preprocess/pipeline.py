
import pandas as pd
import numpy as np

from src.params import *
from src.utils import *
from src.preprocess.cleaning import *
from src.preprocess.features import *

from dataclasses import dataclass

@dataclass
class PreprocessConfig:
    max_gap: int = MAX_GAP
    max_q: float = MAX_Q
    min_coverage_pct: int = MIN_COVERAGE_PCT
    min_bad_month_pct: float = MIN_BAD_MONTH_PCT
    approach: str = DEFAULT_APPROACH
    limit: int = LIMIT
    horizon: int = HORIZON


def preprocessing_pipeline(airqual_df, weather_df, config: PreprocessConfig = PreprocessConfig()):

    #------------------
    # INITIAL CLEANING
    #------------------

    airqual_no_neg = clean_neg_values(airqual_df)
    airqual_sensor_filtered = filter_sensors(df= airqual_no_neg,
                                             max_gap= config.max_gap,
                                             max_q= config.max_q,
                                             min_bad_month_pct= config.min_bad_month_pct,
                                             min_coverage_pct= config.min_coverage_pct)
    airqual_col_selected = filter_columns(df= airqual_sensor_filtered,
                                          col_to_keep=["date", "city", "sensor_id", "pm25_avg"])

    airqual_ready = average_sensors(df= airqual_col_selected)

    weather_ready = filter_columns(df= weather_df, col_to_remove= ["temp_avg"])

    #-------
    # MERGE
    #-------

    data = merge_source_df(df_airqual= airqual_ready, df_weather= weather_ready)
    data = single_gaps_imputer(df= data, limit= config.limit)

    #------------------
    # TARGET ENGINEERING
    #------------------

    data = generate_target(data, horizon= config.horizon)
    data = target_transform(data)

    #------------------
    # FEATURE ENGINEERING
    #------------------

    data = feature_engineering(data, approach= config.approach)


        #------------------
    # FINAL CLEANING
    #------------------

    data = drop_na(data)
    data = drop_preprocess_cols(data)


    print("✅ raw data processed susscessfully!")

    return data
