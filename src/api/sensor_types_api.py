"""
Further information can be found at:
https://newcastle.urbanobservatory.ac.uk/api_docs/doc/sensors-dash-types-json/
"""

import json
from src.api.api_utils import handle_api_response, make_api_request

# Load the configuration file
with open("configs/api_config.json", "r", encoding="utf-8") as config_file:
    api_config = json.load(config_file)

# Access API configuration
api_base_url = api_config["api"]["base_url"]
api_endpoint = api_config["api"]["endpoints"]["sensor_types"]["url"]
api_params = api_config["api"]["endpoints"]["sensor_types"]["params"]


def request(params=api_params):
    """
    Sends a request to the urban observatory sensors types API. This returns a json dictionary.
    """
    url = api_base_url + api_endpoint
    try:
        response = make_api_request(url, params)
        return handle_api_response(response)
    except ValueError as ve:
        print(f"Error in API request: {ve}")
        # Handle the error as needed, e.g., log it, return a default value, etc.
        return None
