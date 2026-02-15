import requests
import pandas as pd
import numpy as np
import json
import time
import os
from tqdm.auto import tqdm
from pathlib import Path
from src.ingestion.utils import save_data_local


class OpenAQClient:
    """
    Client for fetching PM2.5 air quality data from OpenAQ API.
    
    Args:
        api_key (str): OpenAQ API key (if None, reads from API_AQ env var)
        radius (int): Default search radius in meters (default: 5000)
    
    Example:
        >>> client = OpenAQClient(api_key="your_key")
        >>> data = client.get_data(
        ...     cities={"Paris": "48.8566,2.3522"},
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31"
        ... )
    """
    
    def __init__(self, api_key=None, radius=5000):
        self.api_key = api_key or os.getenv("API_AQ")
        self.radius = radius
        self.base_url = "https://api.openaq.org/v3"
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in API_AQ environment variable")
    
    def _get_headers(self):
        """Get API request headers with authentication."""
        return {"X-API-key": self.api_key}
    
    def fetch_location(self, coords):
        """
        Get locations with PM2.5 sensors based on coordinates.
        
        Args:
            coords (str): Lat and Lon coordinates as "lat,lon"
        
        Returns:
            dict: JSON response with location data
        """
        params = {
            "parameters_id": 2,  # PM2.5 sensors
            "coordinates": coords,
            "radius": self.radius,
            "limit": 1000
        }
        url = f"{self.base_url}/locations"
        response = requests.get(url=url, params=params, headers=self._get_headers())
        return response.json()
    
    def filter_sensors(self, data_loc, start_project_date, end_project_date):
        """
        Filter sensors to keep only monitor-grade sensors active throughout project duration.
        
        Args:
            data_loc (dict): Location data from fetch_location()
            start_project_date (str): Project start date (YYYY-MM-DD)
            end_project_date (str): Project end date (YYYY-MM-DD)
        
        Returns:
            list: PM2.5 sensor IDs that meet filtering criteria
        """
        # Filter for monitor-grade sensors only
        monitor_loc = [loc for loc in data_loc["results"] if loc["isMonitor"]]
        
        # Filter for sensors still active at end of project
        end_project_date_dt = pd.to_datetime(end_project_date, utc=True)
        alive_loc = [loc for loc in monitor_loc 
                     if pd.to_datetime(loc["datetimeLast"]["utc"]) >= end_project_date_dt]
        
        # Filter for sensors available since beginning of project
        start_project_date_dt = pd.to_datetime(start_project_date, utc=True)
        full_time_loc = [loc for loc in alive_loc 
                         if pd.to_datetime(loc["datetimeFirst"]["utc"]) <= start_project_date_dt]
        
        print("=" * 50)
        print(f"Initial monitor-grade sensors: {len(monitor_loc)}")
        print(f"Sensors active at end date: {len(alive_loc)}")
        print(f"Sensors with full coverage: {len(full_time_loc)}")
        print("=" * 50)
        
        # Extract PM2.5 sensor IDs
        sensor_ids = []
        for loc in full_time_loc:
            for sensor in loc["sensors"]:
                if sensor["parameter"]["id"] == 2:  # PM2.5 parameter
                    sensor_ids.append(sensor["id"])
        
        return sensor_ids
    
    def fetch_one_sensor_data(self, sensor_id, start_date, end_date):
        """
        Fetch daily PM2.5 measurements for a single sensor.
        
        Args:
            sensor_id (int): Sensor ID
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
        
        Returns:
            dict: JSON response with daily measurements
        """
        url = f"{self.base_url}/sensors/{sensor_id}/measurements/daily"
        params = {
            "datetime_from": start_date,
            "datetime_to": end_date,
            "limit": 1000
        }
        response = requests.get(url, params=params, headers=self._get_headers())
        return response.json()
    
    def extract_all_sensor_data(self, sensor_ids, start_date, end_date):
        """
        Fetch and extract PM2.5 measurements for multiple sensors.
        
        Args:
            sensor_ids (list): List of sensor IDs to fetch
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
        
        Returns:
            pd.DataFrame: Aggregated measurements from all sensors
        """
        sensor_data = {}
        
        # Fetch raw data for each sensor
        for sensor_id in sensor_ids:
            sensor_data[sensor_id] = self.fetch_one_sensor_data(sensor_id, start_date, end_date)
        
        all_dataframes = []
        
        # Extract and structure data for each sensor
        for sensor_id, data in sensor_data.items():
            rows = []
            
            # Extract daily measurements
            for result in data["results"]:
                rows.append({
                    "sensor_id": sensor_id,
                    "date_from_utc": result["period"]['datetimeFrom']['utc'],
                    "date_from_local": result["period"]['datetimeFrom']['local'],
                    "date_to_utc": result["period"]['datetimeTo']['utc'],
                    "date_to_local": result["period"]['datetimeTo']['local'],
                    "pm25_avg": result["value"],
                    "pm25_min": result["summary"]["min"],
                    "pm25_q25": result["summary"]["q25"],
                    "pm25_median": result["summary"]["median"],
                    "pm25_q75": result["summary"]["q75"],
                    "pm25_max": result["summary"]["max"],
                    "coverage": result["coverage"]["percentComplete"]
                })
            
            df_sensor = pd.DataFrame(rows)
            all_dataframes.append(df_sensor)
        
        # Concatenate all sensor DataFrames
        all_measurements = pd.concat(all_dataframes, ignore_index=True)
        return all_measurements
    
    def get_data(self, cities, start_date, end_date, start_project_date, 
                 end_project_date, output_path="../../data/raw/aq_data.csv"):
        """
        Get PM2.5 measurements from OpenAQ API for multiple cities.
        
        Args:
            cities (dict): Cities with coordinates {city_name: "lat,lon"}
            start_date (str): Data fetch start date (YYYY-MM-DD)
            end_date (str): Data fetch end date (YYYY-MM-DD)
            start_project_date (str): Project start date for sensor filtering (YYYY-MM-DD)
            end_project_date (str): Project end date for sensor filtering (YYYY-MM-DD)
            output_path (str): Output CSV path (default: "../../data/raw/aq_data.csv")
        
        Returns:
            pd.DataFrame: Combined PM2.5 measurements for all cities
        """
        all_cities = []
        
        # Process each city
        for city, coords in cities.items():
            print(f"\nProcessing {city}...")
            
            # Get sensor locations in the city
            data_loc = self.fetch_location(coords)
            
            # Filter sensors based on project timeline
            sensor_list = self.filter_sensors(data_loc, start_project_date, end_project_date)
            
            if not sensor_list:
                print(f"⚠️  No sensors found for {city}")
                continue
            
            # Extract measurements for filtered sensors
            aq_by_city = self.extract_all_sensor_data(sensor_list, start_date, end_date)
            aq_by_city['city'] = city  # Add city column
            all_cities.append(aq_by_city)
        
        # Combine all cities data
        all_aq_measurements = pd.concat(all_cities, ignore_index=True)
        
        # Save to disk
        save_data_local(df=all_aq_measurements, output_path=output_path)
        
        return all_aq_measurements
