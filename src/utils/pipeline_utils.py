import importlib
from functools import reduce
from src.utils.general_utils import load_config


def apply_steps(df, steps_config):
    """
    Applies a series of steps to a DataFrame based on a configuration.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        steps_config (list): A list of dictionaries, each containing the 'name' of the function to apply
                             and 'kwargs' for any arguments to pass to the function.

    Returns:
        pd.DataFrame: The processed DataFrame after all steps have been applied.
    """

    def apply_step(df, step):
        module_name, function_name = step["name"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        kwargs = step.get("kwargs", {})
        return func(df, **kwargs)

    return reduce(apply_step, steps_config, df)


def process_data(df, config_path):
    """
    Processes data by dynamically applying defined steps in a configuration file. The steps
    can be part of preprocessing, feature engineering, or modeling stages.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        config_path (str): Path to the configuration file that defines steps for processing.

    Returns:
        pd.DataFrame: The DataFrame after processing.
    """
    config = load_config(config_path)
    # Dynamically identify the steps key based on what's available in the config
    for key in ["preprocessing_steps", "feature_engineering_steps", "model_steps"]:
        if key in config:
            steps_config = config[key]
            break
    else:
        raise ValueError("Config file does not contain valid steps key.")

    processed_df = apply_steps(df, steps_config)
    return processed_df
