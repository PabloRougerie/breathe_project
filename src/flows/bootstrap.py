import pandas as pd
from pathlib import Path
import pandas as pd
import json
import os
import prefect

from src.params import *
from src.utils import *
from src.ingestion.openaq import *
from src.ingestion.openweather import *
from src.preprocess.preproc_pipeline import *
from src.models.model_pipeline import *

def features_target_split():
    """ split X y NOT A FLOW TASK"""
    pass

gcs_storage_client = GCSStorageClient()


#-------
# TASKS
#-------



def train_set_ingestion(start_date, end_date):
    """ api call for train set and upload raw data to bq"""
    aq_client = OpenAQClient(api_key= API_AQ, storage= "cloud")
    airqual_df = aq_client.get_data(cities= CITIES,
                   start_date= start_date,
                   end_date= end_date,
                   start_project_date= START_PROJECT_DATE_STR,
                   end_project_date= END_PROJECT_DATE_STR)
    weather_client = OpenWeatherClient(api_key= API_OW, storage= "cloud")
    weather_df = weather_client.get_all_data(cities= CITIES,
                            start_date= START_TRAIN_DATE_STR,
                            end_date= END_TRAIN_DATE_STR)

    return airqual_df, weather_df


def upload_data(gcs_storage_client, df, data_type, start_date, end_date):
    """ upload raw or processed data to bq"""
    #TODO add a safety chek that data_type is airqual, weather or processed

    gcs_storage_client.save_data(data= df,
                               data_type= data_type,
                               start_date= start_date,
                               end_date= end_date)



def download_data(gcs_storage_client, data_type, start_date, end_date):
    """ dl raw or processed data from bq"""
    df = gcs_storage_client.get_data(data_type= data_type,
                                                   start_date= START_TRAIN_DATE_STR,
                                                   end_date= END_TRAIN_DATE_STR)

    return df

def preprocess_raw_data(airqual_df, weather_df):
    dataset_metadata, data = preprocessing_pipeline(airqual_df= airqual_df,
                                                weather_df= weather_df,
                                                )
    return dataset_metadata, data

def train_model():
    """ train model and set to challenger or champion, in boostrap it's champion"""

def evaluate_model():
    """evaluate model on set, here it's the champion model on test set"""
