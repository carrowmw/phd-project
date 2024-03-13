import numpy as np
import torch
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
    # TEST
    print(preprocessed_dfs[-1][1])
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
    # TEST
    print(engineered_dfs[-1][1])
    print("\n\nFinished engineering\n")
    return engineered_dfs


def load_training_data(engineered_dfs):

    training_loader_config_path = "configs/training_loader_config.json"
    print(f"\n\nLoading {len(engineered_dfs)} engineered DataFrames\n")

    empty_df_count = 0

    # Filter out empty DataFrames with short sequences.
    non_empty_engineered_dfs = []
    training_loader_config = load_config(training_loader_config_path)
    window_size = training_loader_config["feature_engineering_steps"][0]["kwargs"][
        "window_size"
    ]
    print(f"Window size: {window_size}")
    for sensor_name, df in engineered_dfs[:]:
        if len(df) <= window_size * 10:
            empty_df_count += 1
        else:
            non_empty_engineered_dfs.append((sensor_name, df))
    print(f"\n\n Dropped {empty_df_count} empty DataFrames\n")
    print(f"\n\n Loading in {len(non_empty_engineered_dfs)} tensors\n")

    list_of_models_and_dataloaders = []
    for i, df in enumerate(non_empty_engineered_dfs):
        print(f"\n\nProcessing {i+1}/{len(non_empty_engineered_dfs)}\n")
        print(f"Length of NaN values: {df[1].isna().sum().sum()}")
        array = df[1].to_numpy().astype(np.float64)
        tensors = process_data(array, training_loader_config_path)
        sensor_name = df[0]
        list_of_models_and_dataloaders.append((sensor_name, tensors))
    print("\n\nFinished loading tensors\n")
    return list_of_models_and_dataloaders


def train_model(list_of_models_and_dataloaders):
    """ """
    training_config_path = "configs/training_config.json"
    print(f"\n\nTraining {len(list_of_models_and_dataloaders)} models...")

    list_of_metrics = []
    for i, tensor in enumerate(list_of_models_and_dataloaders):
        print(f"\n\nTraining {i+1}/{len(list_of_models_and_dataloaders)}\n")
        metrics = process_data(tensor[1], training_config_path)
        sensor_name = tensor[0]
        list_of_metrics.append((sensor_name, metrics))
    print("\n\nFinished training\n")
    return list_of_metrics
