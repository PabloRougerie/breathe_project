import requests
import pandas as pd
import numpy as np
import json
import time
import os
from tqdm.auto import tqdm
from pathlib import Path
from src.utils import *
from src.params import *


class OpenWeatherClient:
    """
    Client for fetching daily weather data from OpenWeather API.

    Args:
        api_key (str): OpenWeather API key (if None, reads from API_OW env var)
        max_retry (int): Maximum retry attempts for failed requests (default: 3)

    Example:
        >>> client = OpenWeatherClient(api_key="your_key")
        >>> cities = {"Paris": {"lat": 48.8566, "lon": 2.3522}}
        >>> df = client.get_all_data(
        ...     cities=cities,
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ... )
    """

    def __init__(self, api_key=None, max_retry=3, storage= "local"):
        self.api_key = api_key or os.getenv("API_OW")
        self.max_retry = max_retry
        self.base_url = "https://api.openweathermap.org/data/3.0/onecall/day_summary"
        self.storage = storage

        if not self.api_key:
            raise ValueError("API key must be provided or set in API_OW environment variable")

        if self.storage not in ["local", "gcp"]:
            raise ValueError(f"storage is either 'local' or on 'gcp', got {storage} instead")

        if storage == "local":
            self.storage_client = LocalCacheClient(cache_dir= CACHE_DIR)
        else:
            self.storage_client = GCSCacheClient(bucket_name= BUCKET_NAME)

    def fetch_city_data(self, city_name, lat, lon, start_date, end_date):
        """
        Fetch daily weather data for a single city with caching and retry logic.

        Args:
            city_name (str): Name of the city (for logging)
            lat (float): Latitude
            lon (float): Longitude
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)

        Returns:
            None (data saved to cache via storage_client)
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        successful_call = 0
        cached_count = 0

        # Iterate through date range
        for day in tqdm(date_range, desc=f"Fetching {city_name} weather data"):
            day_str = day.strftime('%Y-%m-%d')
            file_name = f"{city_name}/weather/weather_{day_str}.json"

            # Skip if already cached
            if self.storage_client.exists(file_name):
                cached_count += 1
                continue

            # Prepare API parameters
            params = {
                "lat": lat,
                "lon": lon,
                "date": day_str,
                "appid": self.api_key,
                "units": "metric"
            }

            # Retry loop
            for attempt in range(self.max_retry):
                response = requests.get(self.base_url, params)

                # Handle rate limit (HTTP 429)
                if response.status_code == 429:
                    if attempt < self.max_retry - 1:
                        tqdm.write(f"⚠️ [{city_name}] Rate limit on {day_str}. Retrying in 60s...")
                        time.sleep(60)
                        continue
                    else:
                        tqdm.write(f"❌ [{city_name}] Failed after {self.max_retry} attempts on {day_str}")
                        break

                # Handle success (HTTP 200)
                elif response.status_code == 200:
                    try:
                        results = response.json()
                        results_dict = {
                            "date": results["date"],
                            "temp_min": results["temperature"]["min"],
                            "temp_max": results["temperature"]["max"],
                            "temp_avg": (results["temperature"]["min"] + results["temperature"]["max"]) / 2,
                            "cloud_cover": results["cloud_cover"]["afternoon"],
                            "humidity": results["humidity"]["afternoon"],
                            "precipitation": results["precipitation"]["total"],
                            "pressure": results["pressure"]["afternoon"],
                            "wind_speed": results["wind"]["max"]["speed"],
                            "wind_direction": results["wind"]["max"]["direction"]
                        }

                        # Save to cache
                        self.storage_client.write(results_dict, file_name)

                        successful_call += 1
                        break

                    except (KeyError, json.JSONDecodeError) as e:
                        tqdm.write(f"❌ [{city_name}] Error parsing {day_str}: {e}")
                        break

                # Handle other HTTP errors
                else:
                    tqdm.write(f"❌ [{city_name}] HTTP {response.status_code} on {day_str}")
                    break

            # Rate limiting
            time.sleep(1)

        # Summary
        if successful_call + cached_count == len(date_range):
            print(f"✅ [{city_name}] All days fetched! Cached: {cached_count}, New: {successful_call}")
        else:
            print(f"⚠️ [{city_name}] Incomplete: {successful_call + cached_count}/{len(date_range)} days")

    def merge_cached_data(self, city):
        """
        Load all cached JSON files from a directory and merge into DataFrame.

        Args:
            city (str): City name, used to scope the list prefix

        Returns:
            pd.DataFrame: Merged and sorted weather data for that city
        """
        prefix = f"{city}/weather"
        data = []
        file_list = self.storage_client.list(prefix= prefix)

        for file in file_list:
            data.append(self.storage_client.read(str(file)))

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date").reset_index(drop=True)

        return df


    def get_all_data(self, cities, start_date, end_date):
        """
        Fetch weather data for multiple cities.

        Args:
            cities (dict): Cities with coordinates {"city_name": {"lat": float, "lon": float}}
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)


        Returns:
            pd.DataFrame: Combined weather data for all cities
        """
        all_dataframes = []

        # Fetch data for each city
        for city, coords in cities.items():
            print(f"Processing {city}...")



            # Fetch and cache data
            self.fetch_city_data(
                city_name=city,
                lat=coords["lat"],
                lon=coords["lon"],
                start_date=start_date,
                end_date=end_date
            )

            # Load all cached data then filter to requested window
            df = self.merge_cached_data(city)
            df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

            if not df.empty:
                df["city"] = city
                all_dataframes.append(df)

        # Combine all cities
        if not all_dataframes:
            print("⚠️ No data fetched for any city")
            return pd.DataFrame()

        all_cities_df = pd.concat(all_dataframes, ignore_index=True)
        all_cities_df["date"] = pd.to_datetime(all_cities_df["date"])
        print(f"✅ OpenWeather ingestion complete — {len(all_dataframes)} cities, {len(all_cities_df)} measurements")

        return all_cities_df
