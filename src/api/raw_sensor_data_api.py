"""
This module is designed to interact with the Urban Observatory's raw data API, allowing for
the retrieval of JSON data based on sensor names and parameters. It utilizes the API configuration
specified in `api_config.json` todynamically construct request URLs.

Further information on the API and its capabilities can be found at the Urban Observatory API
documentation:

https://newcastle.urbanobservatory.ac.uk/api_docs/doc/sensors-dash-%7Bsensor_name%7D-dash-data-json/

Functions:
- request(params, sensor_name): Sends a request to the API and returns a JSON dictionary with the
response data.

Dependencies:
- api_utils: Contains the `handle_api_response` and `make_api_request` functions used for making API
calls.
"""

import json
from src.api.api_utils import handle_api_response, make_api_request

# Load the configuration file
with open("configs/api_config.json", "r", encoding="utf-8") as config_file:
    api_config = json.load(config_file)

# Access API configuration
api_base_url = api_config["api"]["base_url"]
api_endpoint_template = api_config["api"]["endpoints"]["raw_sensor_data"]["url"]
api_params = api_config["api"]["endpoints"]["raw_sensor_data"]["params"]


# Function to replace placeholder and construct the URL
def request(sensor_name, params=api_params):
    """
    Sends a request to the urban observatory raw data API. This returns a json dictionary.
    """
    # Replace {sensor_name} placeholder with the actual sensor name
    api_endpoint = api_endpoint_template.format(sensor_name=sensor_name)
    url = api_base_url + api_endpoint
    try:
        response = make_api_request(url, params)
        return handle_api_response(response)
    except ValueError as ve:
        print(f"Error in API request: {ve}")
        return None
