import torch
from src.pipeline import (
    preprocess_raw_data,
    apply_feature_engineering,
    load_training_data,
    train_model,
)


torch.backends.mps.is_available()
# Execute pipeline steps
preprocessed_dfs = preprocess_raw_data()
engineered_dfs = apply_feature_engineering(preprocessed_dfs)
training_data_list = load_training_data(engineered_dfs)
training_metrics_list = train_model(training_data_list)
