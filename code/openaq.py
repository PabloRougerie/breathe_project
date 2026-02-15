import requests
import pandas as pd
import numpy as np
import json
import time
import os
from tqdm.auto import tqdm
from pathlib import Path

def fetch_location(coords, radius= 5000, api_key= None):
    """ get locations with sensors based on coordinates

    Inputs:
    coords: Lat and Lon coordinates in string "lat,lon"
    radius: radius around coordinates, in METERS (max 50000 meters I think)
    api_key: openAQ API key

    Outputs:
    a json file with location data

    """

    params_loc = {"parameters_id" : 2, #PM2.5 sensors
               "coordinates": coords, # Lat and Lon
               "radius": radius, # 5km radius
               "limit": 1000 #limit of results
              }
    header_loc = {"X-API-key": API_AQ}
    url_loc = "https://api.openaq.org/v3/locations"
    data_loc = requests.get(url= url_loc, params= params_loc, headers= header_loc).json()

    return data_loc

def filter_sensors(data_loc, start_project_date, end_project_date):
    """filter sensors to keep only high grade sensor
    that are alive throughout the project duration
    """
    #filter for monitor grade sensors
    monitor_loc = [loc for loc in data_loc["results"] if loc["isMonitor"]]

    #filter for sensor still alive at end of project (train + val + tests)
    end_project_date_dt = pd.to_datetime(end_project_date, utc= True)
    alived_loc = [loc for loc in monitor_loc if pd.to_datetime(loc["datetimeLast"]["utc"]) >= end_project_date_dt]

    #filter for availability since beginning of training set
    start_project_date_dt = pd.to_datetime(start_project_date, utc= True)
    full_time_loc = [loc for loc in alived_loc if pd.to_datetime(loc["datetimeFirst"]["utc"]) <= start_project_date_dt]

    print("============")
    print(f"\n Initial number of points of measurement: {len(monitor_loc)}")
    print(f"Number of points of measurement still alive at end of dataset: {len(alived_loc)}")
    print(f"Number of points of measurement available throughout dataset: {len(full_time_loc)}")

    #get pm2.5 sensors id
    sensors_id = []
    for loc in full_time_loc:
        for sensor in loc["sensors"]:
            if sensor["parameter"]["id"] == 2:
                sensors_id.append(sensor["id"])

    return sensors_id


def fetch_one_sensor_data(sensor_id, start_date: str, end_date: str, api_key):
    """fetch PM2.5 measurements for a given sensor between specified dates"""

    url_sensors = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements/daily"
    params_sensors = {
                    "datetime_from": start_train_date,
                    "datetime_to": end_train_date,
                    "limit": 1000
    }

    results_sensor = requests.get(url_sensors, params= params_sensors, headers = header_loc).json()
    return results_sensor

def extract_all_sensor_data(sensor_id, start_date: str, end_date: str, api_key):
    """fetch PM2.5 measurements for a list of sensors between specified dates
    and extract relevant features"""
    sensor_data = {}

    #get raw data for each sensor
    for sensor in sensor_id:
        sensor_data[sensor] = fetch_one_sensor_data(sensor, start_date= start_date, end_date= end_date, api_key= api_key)

    all_dataframes= []

    #iterates through all sensors and data
    for sensor_id, data in sensor_data.items():
        rows = [] #empty list to add required fields for a given sensor_id

    #iterates over daily measurements for that sensor
        for results in data["results"]:
            rows.append({
                "sensor_id": sensor_id,
                "date_from_utc": results["period"]['datetimeFrom']['utc'],
                "date_from_local": results["period"]['datetimeFrom']['local'],
                "date_to_utc": results["period"]['datetimeTo']['utc'],
                "date_to_local": results["period"]['datetimeTo']['local'],
                "pm25_avg": results["value"],
                "pm25_min": results["summary"]["min"],
                "pm25_q25": results["summary"]["q25"],
                "pm25_median": results["summary"]["median"],
                "pm25_q75": results["summary"]["q75"],
                "pm25_max": results["summary"]["max"],
                "coverage": results["coverage"]["percentComplete"]
            })
        df_sensor = pd.DataFrame(rows)
        all_dataframes.append(df_sensor)

    all_aq_measurements_by_city = pd.concat(all_dataframes, ignore_index= True)
