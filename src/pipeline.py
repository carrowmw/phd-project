import numpy as np

from src.data_processing.data_loaders import fetch_raw_data
from src.utils.general_utils import load_config
from src.utils.pipeline_utils import process_data


def preprocess_raw_data() -> list:
    """ """
    raw_dfs = fetch_raw_data()

    preprocessing_config_path = "configs/preprocessing_config.json"
    print(f"\n\n {len(raw_dfs)} DataFrames found\n")

    preprocessed_dfs = []
    preprocessing_config = load_config("configs/preprocessing_config.json")
    completeness_threshold = preprocessing_config["kwargs"]["completeness_threshold"]
    print(f"Completeness: {completeness_threshold*100}% per day")

    for i, df in enumerate(raw_dfs):
        print(f"\n\nProcessing {i+1}/{len(raw_dfs)}\n")
        processed_df = process_data(df[1], preprocessing_config_path)
        sensor_name = df[0]
        preprocessed_dfs.append((sensor_name, processed_df))

    print("\n\nFinished preprocessing\n")
    return preprocessed_dfs


def apply_feature_engineering(preprocessed_dfs):
    """ """

    empty_df_count = 0

    # Filter out empty DataFrames and count them
    non_empty_preprocessed_dfs = []
    for sensor_name, df in preprocessed_dfs[:]:
        if df.empty:
            empty_df_count += 1
        else:
            non_empty_preprocessed_dfs.append((sensor_name, df))

    feature_engineering_config_path = "configs/feature_engineering_config.json"
    print(f"\n\n Dropped {empty_df_count} empty DataFrames\n")
    print(
        f"\n\n Engineering features for {len(non_empty_preprocessed_dfs)} DataFrames\n"
    )

    engineered_dfs = []
    for i, df in enumerate(non_empty_preprocessed_dfs):
        print(f"\n\nProcessing {i+1}/{len(non_empty_preprocessed_dfs)}\n")
        engineered_df = process_data(df[1], feature_engineering_config_path)
        sensor_name = df[0]
        engineered_dfs.append((sensor_name, engineered_df))

    print("\n\nFinished engineering\n")
    return engineered_dfs


def load_training_data(engineered_dfs):

    training_loader_config_path = "configs/training_loader_config.json"
    print(f"\n\nLoading {len(engineered_dfs)} engineered DataFrames\n")

    training_data_list = []
    for i, df in enumerate(engineered_dfs):
        print(f"\n\nProcessing {i+1}/{len(engineered_dfs)}\n")
        array = df[1].to_numpy().astype(np.float64)
        training_data = process_data(array, training_loader_config_path)
        sensor_name = df[0]
        training_data_list.append((sensor_name, training_data))
    print("\n\nFinished loading training data\n")
    return training_data_list


preprocessed_dfs = preprocess_raw_data()
engineered_dfs = apply_feature_engineering(preprocessed_dfs)
training_data_list = load_training_data(engineered_dfs)
