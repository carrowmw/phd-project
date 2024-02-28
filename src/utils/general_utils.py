import json
import re


# Load configuration from JSON
def load_config(config_path):
    """
    Loads the configuration from a JSON file.

    Args:
        config_path (str): The file path to the configuration JSON file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
        return config


def get_step_config(step_name, config):
    """
    Retrieves the configuration for a specific preprocessing step.

    Args:
        step_name (str): The name of the preprocessing step.
        config (dict): The full configuration dictionary.

    Returns:
        dict: The configuration dictionary for the specified step. Returns an empty dictionary if the step is not found.
    """
    for step in config.get("preprocessing_steps", []):
        if step["name"] == step_name:
            return step.get("kwargs", {})
    return {}


def extract_values_from_filename(filename):
    """
    Extracts the Completeness, Sequence Length, Horizon, and Window Size values
    from the given filename using regular expressions.

    Args:
    - filename (str): The filename to extract the values from.

    Returns:
    - dict: A dictionary containing the extracted values. Returns None for values not found.
    """
    # Regular expressions for each value
    regex_patterns = {
        "Completeness": r"Completeness([\d.]+)",
        "SequenceLength": r"SequenceLength(\d+)",
        "Horizon": r"Horizon(\d+)",
        "WindowSize": r"WindowSize(\d+)",
        "TestNumber": r"TestNumber(\d+)",
    }

    extracted_values = {}
    for key, pattern in regex_patterns.items():
        match = re.search(pattern, filename)
        if match:
            # Convert to float if it has a decimal point, else convert to int
            value = (
                float(match.group(1)) if "." in match.group(1) else int(match.group(1))
            )
            extracted_values[key] = value
        else:
            extracted_values[key] = None

    return extracted_values
