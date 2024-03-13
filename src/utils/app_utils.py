import os
import json
from datetime import date
import pickle

from src.utils.polygon_utils import create_wkb_polygon


def create_and_load_file_path(api_config_file_path, app_data_directory):
    """
    Loads the API configuration, calculates the bounding box, and constructs a file path
    for storing or retrieving processed data. This path incorporates today's date, the number
    of last days from the configuration, and the calculated bounding box.

    Args:
        api_config_file_path (str): Path to the API configuration file.
        app_data_directory (str): The directory within 'data/processed' to store or retrieve the file.

    Returns:
        str: The constructed file path for the data file.
    """
    # Load API configuration
    with open(api_config_file_path, "r", encoding="utf-8") as config_file:
        api_config = json.load(config_file)

    last_n_days = api_config["api"]["endpoints"]["raw_sensor_data"]["params"][
        "last_n_days"
    ]
    coords = api_config["api"]["coords"]

    # Assume create_wkb_polygon is a predefined function
    bbox = create_wkb_polygon(coords[0], coords[1], coords[2], coords[3])

    today = date.today()
    file_path = f"data/processed/{app_data_directory}/{today}_Last_{last_n_days}_Days_{bbox}.pkl"

    return file_path


def save_data_to_file(file_path, data):
    """
    Saves the given data to a file at the specified path, creating any necessary directories.

    Args:
        file_path (str): The path where the data should be saved.
        data (Any): The data to be saved.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_data_from_file(file_path):
    """
    Loads data from the specified file path if it exists.

    Args:
        file_path (str): The path of the file to load data from.

    Returns:
        The data loaded from the file if it exists, otherwise None.
    """
    if os.path.exists(file_path):
        print("\nReading in app data from local storage\n")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None
