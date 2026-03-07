import requests
import pandas as pd
import os
import time
from datetime import datetime
import json
from src.utils import *
from src.params import *


class OpenAQClient:
    """
    Client for fetching PM2.5 air quality data from OpenAQ API.

    Args:
        api_key (str): OpenAQ API key (if None, reads from API_AQ env var)
        radius (int): Default search radius in meters (default: 5000)

    """

    def __init__(self, api_key=None, radius=7000, max_retry=3, storage= "local", min_coverage=0.75):
        self.api_key = api_key or os.getenv("API_AQ")
        self.radius = radius
        self.base_url = "https://api.openaq.org/v3"
        self.max_retry = max_retry
        self.storage = storage
        self.min_coverage = min_coverage

        if not self.api_key:
            raise ValueError("API key must be provided or set in API_AQ environment variable")

        if self.storage not in ["local", "gcp"]:
            raise ValueError(f"storage is either 'local' or on 'gcp', got {storage} instead")

        if storage == "local":
            self.storage_client = LocalStorageClient(cache_dir= CACHE_DIR)
        else:
            self.storage_client = GCSStorageClient(bucket_name= BUCKET_NAME)


    def _get_headers(self):
        """Get API request headers with authentication."""
        return {"X-API-key": self.api_key}

    def fetch_location(self, lat, lon):
        """
        Get locations with PM2.5 sensors based on coordinates.

        Args:
            lat (float): Latitude
            lon (float): Longitude

        Returns:
            dict: JSON response with location data
        """
        coords_str = f"{lat},{lon}"

        params = {
            "parameters_id": 2,  # PM2.5 only
            "coordinates": coords_str,
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
        # Step 1: Keep only monitor-grade locations
        monitor_loc = [loc for loc in data_loc["results"] if loc["isMonitor"]]

        # Step 2: Keep locations still active at end of project
        end_project_date_dt = pd.to_datetime(end_project_date, utc=True)
        alive_loc = [loc for loc in monitor_loc
                     if pd.to_datetime(loc["datetimeLast"]["utc"]) >= end_project_date_dt]

        # Step 3: Keep locations already active at start of project
        start_project_date_dt = pd.to_datetime(start_project_date, utc=True)
        full_time_loc = [loc for loc in alive_loc
                         if pd.to_datetime(loc["datetimeFirst"]["utc"]) <= start_project_date_dt]

        print("  " + "=" * 46)
        print(f"  Monitor-grade locations: {len(monitor_loc)}")
        print(f"  Active at end date: {len(alive_loc)}")
        print(f"  Full coverage: {len(full_time_loc)}")
        print("  " + "=" * 46)

        # Extract PM2.5 sensor IDs from filtered locations
        sensor_ids = []
        for loc in full_time_loc:
            for sensor in loc["sensors"]:
                if sensor["parameter"]["id"] == 2:  # PM2.5
                    sensor_ids.append(sensor["id"])

        print(f"  PM2.5 sensors found: {len(sensor_ids)}")
        return sensor_ids

    def fetch_one_sensor_data(self, sensor_id, start_date, end_date, file_name):
        """
        Fetch daily PM2.5 measurements for a single sensor.

        Args:
            sensor_id (int): Sensor ID
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            cache_dir (str): Directory to cache sensor data

        Returns:
            dict: JSON response with daily measurements
        """
        # Check cache first


        if self.storage_client.exists(file_name= file_name):
            print(f"  └─ Sensor {sensor_id}: using cached data")

            return self.storage_client.read(file_name)


        # Fetch from API
        url = f"{self.base_url}/sensors/{sensor_id}/days"
        params = {
            "date_from": start_date,
            "date_to": end_date,
            "limit": 1000
        }

        for attempt in range(self.max_retry):

            response = requests.get(url, params=params, headers=self._get_headers())

            if response.status_code == 200:
                results = response.json()
                print(f"  └─ Sensor {sensor_id}: fetched from API")
                print(results)

                # Check 1: empty JSON?
                if not results.get("results") or len(results.get("results")) == 0:
                    print(f"      ⚠️  No measurements returned, skipping")
                    return {"results": []}

                # Check 2: low coverage?
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                expected_measurements = (end_dt - start_dt).days + 1

                nb_measurements = results.get("meta", {}).get("found", 0)
                coverage_ratio = nb_measurements / expected_measurements

                if coverage_ratio < self.min_coverage:
                    print(f"      ⚠️  Coverage too low ({coverage_ratio:.0%}), skipping")
                    return {"results": []}

                # All good, save to cache
                self.storage_client.write(results, file_name)

                return results

            elif attempt < self.max_retry - 1:
                print(f"      ⚠️  API error {response.status_code}, retrying ({attempt + 2}/{self.max_retry})...")
                time.sleep(5)
                continue

            else:
                print(f"      ✗ API call failed after {self.max_retry} attempts")
                return {"results": []}

    def extract_all_sensor_data(self, sensor_ids, start_date, end_date, city):
        """
        Fetch and extract PM2.5 measurements for multiple sensors.

        Args:
            sensor_ids (list): List of sensor IDs to fetch
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            cache_dir (str): Directory to cache sensor data

        Returns:
            pd.DataFrame: Aggregated measurements from all sensors
        """
        sensor_data = {}
        print(f"  Fetching data for {len(sensor_ids)} sensors...")

        # Fetch raw data for each sensor
        for i, sensor_id in enumerate(sensor_ids, 1):
            print(f"  [{i}/{len(sensor_ids)}] Sensor {sensor_id}")
            file_name = f"{city}/sensor_{sensor_id}.json"
            sensor_data[sensor_id] = self.fetch_one_sensor_data(sensor_id, start_date, end_date, file_name = file_name)

        all_dataframes = []

        # Extract and structure data for each sensor
        for sensor_id, data in sensor_data.items():
            # Skip sensors with no data
            if not data.get("results") or len(data.get("results")) == 0:
                continue

            rows = []
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

        # Check if we got any data
        if not all_dataframes:
            print(f"  ⚠️  No valid sensor data found")
            return pd.DataFrame()

        # Concatenate all sensor DataFrames
        all_measurements = pd.concat(all_dataframes, ignore_index=True)
        print(f"  ✓ {len(all_measurements)} measurements extracted")
        return all_measurements

    def get_data(self, cities, start_date, end_date, start_project_date,
                 end_project_date):
        """
        Get PM2.5 measurements from OpenAQ API for multiple cities.

        Args:
            cities (dict): Cities with coordinates {city_name: {"lat": float, "lon": float}}
            start_date (str): Data fetch start date (YYYY-MM-DD)
            end_date (str): Data fetch end date (YYYY-MM-DD)
            start_project_date (str): Project start date for sensor filtering (YYYY-MM-DD)
            end_project_date (str): Project end date for sensor filtering (YYYY-MM-DD)
            output_path (str): Output CSV path (default: "../data/raw/airqual.csv")

        Returns:
            pd.DataFrame: Combined PM2.5 measurements for all cities
        """
        all_cities = []

        # Process each city
        for city, coords in cities.items():
            print(f"\nProcessing {city}...")

            # Get sensor locations in the city
            data_loc = self.fetch_location(lat=coords["lat"], lon=coords["lon"])

            # Filter sensors based on project timeline
            sensor_list = self.filter_sensors(data_loc, start_project_date, end_project_date)

            if not sensor_list:
                print(f"⚠️  No sensors found for {city}")
                continue

            print(f"✓ {len(sensor_list)} sensor(s) selected for {city}")

            # Extract measurements for filtered sensors

            aq_by_city = self.extract_all_sensor_data(sensor_list, start_date, end_date, city= city)

            # Only add if we got data
            if not aq_by_city.empty:
                aq_by_city['city'] = city
                all_cities.append(aq_by_city)
            else:
                print(f"  ⚠️  No valid data extracted for {city}")

        # Check if we got any data
        if not all_cities:
            raise ValueError("No valid data found for any city")

        # Combine all cities data
        all_aq_measurements = pd.concat(all_cities, ignore_index=True)


        print(f"\n{'=' * 50}")
        print(f"✓ Ingestion complete!")
        print(f"  {len(all_cities)} cities processed")
        print(f"  {len(all_aq_measurements)} total measurements")

        # Save to disk
        if self.storage == "local":
            save_data_local(df=all_aq_measurements, output_path= LOCAL_RAW_DIR)
            print(f"  Saved to: {output_path}")
        else:
            # TODO add function to load to bq
            pass



        return all_aq_measurements
