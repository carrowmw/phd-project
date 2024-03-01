import os
from datetime import date
import pickle
import pandas as pd

from data_processing.execute_requests import (
    execute_raw_sensor_data_request,
    execute_sensors_request,
    print_sensor_request_metrics,
)
from src.utils.pipeline_utils import process_data
from src.utils.general_utils import load_config
from src.utils.polygon_utils import create_wkb_polygon


def create_file_path_from_config() -> str:
    """
    Creates a file path for storing or retrieving sensor data based on the API configuration.

    Returns:
        str: The file path for storing or retrieving the sensor data.
    """
    api_config_path = "configs/api_config.json"
    api_config = load_config(api_config_path)
    today = date.today()
    last_n_days = api_config["api"]["endpoints"]["raw_sensor_data"]["params"][
        "last_n_days"
    ]
    coords = api_config["api"]["coords"]
    bbox = create_wkb_polygon(coords[0], coords[1], coords[2], coords[3])
    file_path = f"data/raw/{today}_Last_{last_n_days}_Days_{bbox}.pkl"
    return file_path


def download_and_save_raw_data(file_path: str) -> pd.DataFrame:
    """
    Downloads raw sensor data from the API and saves it to local storage.

    Args:
        file_path (str): Path where the downloaded data will be saved.

    Returns:
        pd.DataFrame: The raw sensor data.
    """
    # Assuming execute_sensors_request() and execute_raw_sensor_data_request() are defined elsewhere
    sensors_df = execute_sensors_request()
    series_of_sensor_names = sensors_df["Sensor Name"]
    raw_dfs = execute_raw_sensor_data_request(sensors_df)
    print_sensor_request_metrics(raw_dfs, series_of_sensor_names)

    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(raw_dfs, f)

    return raw_dfs


def fetch_raw_data() -> pd.DataFrame:
    """
    Fetches raw data from local storage if available, or downloads it if not.

    Returns:
        pd.DataFrame: The raw sensor data.
    """
    file_path = create_file_path_from_config()

    if os.path.exists(file_path):
        print("\nReading in raw data from local storage...\n")
        with open(file_path, "rb") as f:
            raw_dfs = pickle.load(f)
    else:
        print("\nDownloading raw data...\n")
        raw_dfs = download_and_save_raw_data(file_path)
    return raw_dfs
