import json
import unittest
from unittest.mock import patch
from src.api import sensors_api


class TestAPIRequests(unittest.TestCase):
    @patch("src.api.sensors_api.make_api_request")
    def test_sensor_api_request(self, mock_make_api_request):
        # Load configuration within the test to ensure isolation
        with open("configs/api_config.json", "r", encoding="utf-8") as config_file:
            api_config = json.load(config_file)

        # Construct the expected URL dynamically
        base_url = api_config["api"]["base_url"]
        sensors_url = api_config["api"]["endpoints"]["sensors"]["url"]
        expected_url = base_url + sensors_url.format(sensor_name="test_sensor")

        # Expected params (adjust as needed)
        params = api_config["api"]["endpoints"]["sensors"]["params"]

        # Mock response setup
        mock_response = {"sensors": []}
        mock_make_api_request.return_value = mock_response

        # Execute the request function
        response = sensors_api.request(params)

        # Assertions
        self.assertEqual(response, mock_response)
        mock_make_api_request.assert_called_once_with(expected_url, params)


if __name__ == "__main__":
    unittest.main()
