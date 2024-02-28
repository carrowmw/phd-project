"""
All the functions to run the analysis loops for both univariate and multivariate models.
"""

import os

import torch

from data_processing.preprocess_data import preprocess_data
from data_processing.windowing_functions import sliding_windows
from visualisation.custom_plots import get_custom_palette, get_custom_colormap

# Constants
COMPLETENESS_THRESHOLD = 1
WINDOW_SIZE = 12
HORIZON = 3
STRIDE = 1
INPUT_INDICES = 0
TARGET_INDEX = 0
BATCH_SIZE = 8
EPOCHS = 5
ERROR_STD = 4
DATA_PATH = r"C:\#code\#python\#current\mres-project\data\saville_row_east_west"
OUTPUT_TABLES_PATH = r"../output/tables/4/"
OUTPUT_FIGURES_PATH = r"../output/figures/4/"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get font settings
os.system(r"C:\#code\#python\#current\mres-project\analysis_files\mpl_config.py")

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_TABLES_PATH, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_PATH, exist_ok=True)

custom_palette = get_custom_palette()
custom_colormap = get_custom_colormap()


def train_and_evaluate_model(
    model, train_dataloader, test_dataloader, model_type="Linear", epochs=1
):
    """
    Train and evaluate a given model.
    :param model: PyTorch model (Linear or LSTM)
    :param train_dataloader: training dataloader
    :param test_dataloader: test dataloader
    :param model_type: type of model ("Linear" or "LSTM")
    :return: predictions, metrics
    """
    if model_type == "Linear":
        return train_and_evaluate_linear_model(
            model, train_dataloader, test_dataloader, epochs=epochs
        )
    elif model_type == "LSTM":
        return train_and_evaluate_lstm_model(
            model, train_dataloader, test_dataloader, epochs=epochs
        )
