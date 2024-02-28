"""
This module provides functions to extract dataframes from the API requests.
"""

import json
import concurrent.futures
import pandas as pd

from src.api import (
    raw_sensor_data_api,
    sensors_api,
    sensor_types_api,
    themes_api,
    variables_api,
)

from src.utils.polygon_utils import create_wkb_polygon
from src.utils.general_utils import load_config


def json_to_dataframe(json_data):
    """
    Convert a JSON object to a pandas DataFrame.

    Parameters:
    - json_data (dict): The JSON data to convert.

    Returns:
    - DataFrame: A pandas DataFrame constructed from the input JSON data.
    """
    return pd.DataFrame(json_data)


def print_api_response_information(sensor_name, index, total_sensors):
    """
    Print API response information, overwriting the previous line.

    Parameters:
    - sensor_name (str): Name of the sensor.
    - index (int): Index of the sensor.
    - total_sensors (int): Total number of sensors to process.
    """
    # Use carriage return (`\r`) to return the cursor to the beginning of the line.
    # Use `end=''` to prevent advancing to a new line.
    # Use `flush=True` to force the output to be written to the terminal.
    print(
        f"\rProcessing sensor {index + 1} of {total_sensors}: {sensor_name}",
        end="",
        flush=True,
    )

    # If this is the last sensor, print 'finished' message on the next line.
    if index + 1 == total_sensors:
        print("\nFinished processing all sensors.")


def process_sensor_data(params, sensor_name, index, total_sensors):
    """
    Process sensor data for a given sensor, modified to include total_sensors parameter.

    Parameters:
    - params (dict): Dictionary of parameters for API request.
    - sensor_name (str): Name of the sensor.
    - index (int): Index of the sensor.
    - total_sensors (int): Total number of sensors to process.

    Returns:
    - tuple: A tuple containing the sensor name and its corresponding DataFrame.
    """

    raw_data_dict = raw_sensor_data_api.request(sensor_name, params)

    if (
        raw_data_dict is not None
        and "sensors" in raw_data_dict
        and len(raw_data_dict["sensors"]) > 0
    ):
        sensor_data = raw_data_dict["sensors"][0]["data"]

        if sensor_data and "Walking" in sensor_data:
            raw_data_dict = sensor_data["Walking"]
            print_api_response_information(sensor_name, index, total_sensors)
            print(f"        Length of Raw Data Dict: {len(raw_data_dict)}")

            df = json_to_dataframe(raw_data_dict)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")

            return sensor_name, df

        print_api_response_information(sensor_name, index, total_sensors)
        print("        Empty Sensor...")
        return None

    print_api_response_information(sensor_name, index, total_sensors)
    print("        Error in API request or no sensor data available.")
    return None


def get_all_sensor_data_parallel(params, series_of_sensor_names):
    """
    Get data for all sensors in parallel, modified to pass total_sensors parameter.

    Parameters:
    - series_of_sensor_names (pd.Series): Series of sensor names.
    - params (dict): Parameters for the raw data API request.

    Returns:
    - list: List of tuples containing sensor names and their corresponding DataFrames.
    """
    list_of_dataframes = []
    total_sensors = len(series_of_sensor_names)

    with concurrent.futures.ThreadPoolExecutor() as executor:  # or ProcessPoolExecutor
        # Modify lambda function to include total_sensors parameter.
        results = list(
            executor.map(
                lambda x: process_sensor_data(params, x[1], x[0], total_sensors),
                series_of_sensor_names.items(),
            )
        )

    for result in results:
        if result:
            list_of_dataframes.append(result)

    return list_of_dataframes


def print_sensor_request_metrics(list_of_dataframes, series_of_sensor_names):
    """
    Print metrics related to sensor data.

    Parameters:
    - list_of_dataframes (list): List of the returned data from successful sensor requests.
    - series_of_sensor_names (pd.Series): Series of all the sensors.
    """
    active_sensor_count = len(list_of_dataframes)
    empty_sensor_count = len(series_of_sensor_names) - active_sensor_count
    empty_sensor_perc = empty_sensor_count / len(series_of_sensor_names)

    print(f"\n Percentage Empty Sensors:   \n     {100*round(empty_sensor_perc, 2)}%")
    print(f"\n Count of Empty Sensors:     \n     {empty_sensor_count}")
    print(f"\n Count of Active Sensors:    \n     {active_sensor_count}")


def get_daily_counts_dataframes(list_of_dataframes):
    """
    Get daily counts dataframes.

    Parameters:
    - list_of_dataframes (list): A list of tuples containing DataFrame name and DataFrame.

    Returns:
    list: A list of daily counts DataFrames.

    For each DataFrame in the list, this function calculates daily counts based on the 'Timestamp'
    column, creates a new DataFrame with 'Timestamp' and 'Count' columns, and appends it to a list.

    Example:
    get_daily_counts_dataframes([('Sensor1', df1), ('Sensor2', df2)])
    """
    daily_counts_list = []
    for df in list_of_dataframes:
        name, df = df[0], df[1]
        # Group by the date part of the timestamp and count the rows
        daily_counts = (
            df.groupby(df["Timestamp"].dt.date).size().reset_index(name="Count")
        )
        daily_counts_list.append((name, daily_counts))

    return daily_counts_list


def save_daily_counts_dataframes(list_of_dataframes):
    """
    Save daily counts dataframes to CSV files.

    Parameters:
    - list_of_dataframes (list): A list of tuples containing DataFrame name and DataFrame.

    Returns:
    None

    For each DataFrame in the list, this function calculates daily counts based on the 'Timestamp'
    column, creates a new DataFrame with 'Timestamp' and 'Count' columns, and saves it to a CSV file
    in the 'daily_counts' directory.

    Example:
    save_daily_counts_dataframes([('Sensor1', df1), ('Sensor2', df2)])
    """
    for df in list_of_dataframes:
        name, df = df[0], df[1]
        # Group by the date part of the timestamp and count the rows
        daily_counts = (
            df.groupby(df["Timestamp"].dt.date).size().reset_index(name="Count")
        )
        daily_counts.to_csv(f"./data/processed/daily_counts/{name}.csv")
    print("All daily counts saved in ./data/processed/daily_counts/{name}.csv")


def execute_sensor_type_request():
    """
    Fetch sensor type data via API request and convert it to a pandas DataFrame.

    Uses the global `params` configuration for the request. Prints the result summary.
    """
    sensor_types_dict = sensor_types_api.request()
    print(sensor_types_dict)
    sensor_types_df = json_to_dataframe(sensor_types_dict["Variables"])
    print("Sensor Types Request Successful...")
    print(f"    Length of Sensor Types: {len(sensor_types_df)}")


def execute_variables_request():
    """
    Fetch variables data via API request and convert it to a pandas DataFrame.

    Uses the global `params` configuration for the request. Prints the result summary.
    """
    variables_json = variables_api.request()
    variables_df = json_to_dataframe(variables_json["Variables"])
    print("Variables Request Successful...")
    print(f"    Length of Variables DataFrame: {len(variables_df)}")


def execute_themes_request():
    """
    Fetch themes data via API request and convert it to a pandas DataFrame.

    This function does not use the `params` configuration as themes may not require parameters.
    Prints the result summary and returns the themes DataFrame.

    Returns:
    - DataFrame: A pandas DataFrame containing themes data.
    """
    themes_dict = themes_api.request()
    themes_df = json_to_dataframe(themes_dict["Themes"])
    print("Themes Request Successful...")
    print(f"    Length of Themes DataFrame: {len(themes_df)}")
    return themes_df


def execute_sensors_request() -> pd.DataFrame:
    """
    Executes a request to retrieve sensor data based on the coordinates specified in the
    configuration file.

    This function reads the application configuration from 'configs/api_config.json', extracts
    the coordinates for the area of interest, converts these coordinates into a well-known
    binary (WKB) representation of the bounding box, and uses this bounding box to make an API
    request for sensors data. The response from the API is then converted into a pandas
    DataFrame before being returned.

    The configuration file must contain a section 'api' with a nested 'coords' list that specifies
    the bounding box coordinates, and a 'endpoints' section with a 'sensors' subsection that
    includes the necessary parameters for the sensor data request.

    Returns:
        pandas.DataFrame: A DataFrame containing the sensors data retrieved from the API,
        structured according to the response's format. Each row represents a sensor, with
        columns for each attribute provided by the API.

    Raises:
        FileNotFoundError: If the 'configs/api_config.json' file does not exist or cannot be opened.
        json.JSONDecodeError: If there's an error parsing the configuration file.
        KeyError: If essential keys ('api', 'coords', 'endpoints', 'sensors', 'params') are
        missing in the configuration file. Exception: If the API request fails or the response
        cannot be processed into a DataFrame.
    """
    # Needed to turn the coords found in the config section into a bbox for the request
    api_config_path = "configs/api_config.json"
    api_config = load_config(api_config_path)
    coords = api_config["api"]["coords"]
    bbox = create_wkb_polygon(coords[0], coords[1], coords[2], coords[3])
    sensor_params = api_config["api"]["endpoints"]["sensors"]["params"]
    sensor_params["polygon_wkb"] = bbox
    sensors_json = sensors_api.request(sensor_params)
    sensors_df = json_to_dataframe(sensors_json["sensors"])

    return sensors_df


def execute_raw_sensor_data_request(sensors_df: pd.DataFrame) -> list:
    """
    Executes requests for raw sensor data in parallel for a given set of sensors and
    returns the results.

    This function reads the API configuration from a JSON file, extracts parameters for
    the raw sensor data endpoint, and initiates parallel requests for sensor data based
    on the sensor names provided in the input DataFrame. It prints metrics about the
    sensor requests and returns a list of DataFrames, each containing the raw data for
    a sensor.

    Parameters:
    - sensors_df (pd.DataFrame): A DataFrame containing sensor information. Must include a
    column 'Sensor Name' that lists the names of the sensors for which raw data is to be
    requested.

    Returns:
    - list of pd.DataFrame: A list containing pandas DataFrames, each with the raw sensor data
    for the sensors listed in the input DataFrame. Each DataFrame in the list corresponds to the
    raw data of a sensor identified by its name in the 'Sensor Name' column of the input DataFrame.

    Raises:
    - ValueError: If there is an issue with reading the API configuration or if the input
    DataFrame does not contain the required 'Sensor Name' column.

    Example:
    ```
    import pandas as pd

    # Assuming sensors_df is a DataFrame with at least a 'Sensor Name' column.
    sensors_df = pd.DataFrame({
        'Sensor Name': ['Sensor1', 'Sensor2', ...]
    })

    list_of_raw_sensor_dfs = execute_raw_sensor_data_request(sensors_df)
    for df in sensor_data_dfs:
        print(df.head())  # Print the first few rows of each sensor's raw data DataFrame
    ```
    """
    with open("configs/api_config.json", "r", encoding="utf-8") as config_file:
        api_config = json.load(config_file)
    series_of_sensor_names = sensors_df["Sensor Name"]
    raw_sensor_data_params = api_config["api"]["endpoints"]["raw_sensor_data"]["params"]

    dfs = get_all_sensor_data_parallel(raw_sensor_data_params, series_of_sensor_names)
    print_sensor_request_metrics(dfs, series_of_sensor_names)
    return dfs
