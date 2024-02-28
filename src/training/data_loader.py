from typing import Tuple
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from src.training.datasets import TimeSeriesDataset


def sliding_windows(
    data: np.ndarray,
    window_size: int,
    input_feature_indices: list,
    target_feature_index: int,
    horizon: int,
    stride=1,
    shapes=False,
):
    """
    Generate sliding windows from the provided time-series data for sequence learning.

    Parameters:
    - data (np.ndarray): The time-series data from which windows will be generated.
    - window_size (int): Specifies the size of each sliding window.
    - input_feature_indices (list of ints): The indices of features to be considered as input.
    - target_feature_index (int): Index of the feature that needs to be predicted.
    - horizon (int): How many steps ahead the prediction should be.
    - stride (int, optional): Steps between the start of each window. Defaults to 1.
    - shapes (bool, optional): If set to True, it prints shapes of input and target for the first window. Defaults to False.

    Returns:
    - tuple: Contains inputs and targets as torch tensors.
    """

    inputs = []
    targets = []
    for i in range(0, len(data) - window_size - horizon + 1, stride):
        input_data = data[
            i : i + window_size, input_feature_indices
        ]  # selects only the features indicated by input_feature_indices
        target_data = data[
            i + window_size + horizon - 1, target_feature_index
        ]  # selects the feature indicated by target_feature_index, horizon steps ahead
        if i == 0 and shapes:
            print(
                f"Input shape: {input_data.shape} | Target shape: {target_data.shape}"
            )
        inputs.append(input_data)
        targets.append(target_data)

    # Convert lists of numpy arrays to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)

    return torch.tensor(inputs), torch.tensor(targets)


def prepare_dataloaders(
    data: np.ndarray,
    window_size: int,
    input_feature_indices: list,
    target_feature_index: int,
    horizon: int,
    stride: int,
    batch_size: int,
    shuffle=False,
    num_workers=0,
) -> Tuple[Dataset, Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares training and test dataloaders using sliding windows on the given time-series data.

    Parameters:
    - data (np.ndarray): Time-series data.
    - window_size (int): Size of each sliding window.
    - input_feature_indices (list of ints): Indices of features to be considered as input.
    - target_feature_index (int): The index of the feature that needs to be predicted.
    - horizon (int): Steps ahead for the prediction.
    - stride (int): Steps between the start of each window.
    - batch_size (int): Number of samples per batch to load.
    - shuffle (bool, optional): Whether to shuffle the data samples. Defaults to False.
    - num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.

    Returns:
    - tuple: Contains train DataLoader, test DataLoader, test inputs, test targets, train inputs, and train targets.
    """
    inputs, targets = sliding_windows(
        data=data,
        window_size=window_size,
        input_feature_indices=input_feature_indices,
        target_feature_index=target_feature_index,
        horizon=horizon,
        stride=stride,
    )

    # Split data into train and test sets
    train_size = int(0.8 * len(inputs))
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]

    # Create custom PyTorch Dataset and DataLoader objects
    train_dataset = TimeSeriesDataset(train_inputs, train_targets)
    test_dataset = TimeSeriesDataset(test_inputs, test_targets)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return (
        train_dataloader,
        test_dataloader,
        test_inputs,
        test_targets,
        train_inputs,
        train_targets,
    )
