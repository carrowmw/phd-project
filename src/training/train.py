from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from src.utils.training_utils import (
    create_criterion,
    create_optimiser,
    map_model_to_mps,
    map_tensor_to_mps,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_tensor_to_numpy(tensor):
    """
    Safely converts a tensor to a numpy array.

    Parameters:
    - tensor: A PyTorch tensor.

    Returns:
    - A numpy array converted from the input tensor.
    """
    try:
        return tensor.detach().cpu().numpy()
    except AttributeError as e:
        raise ValueError("Input is not a tensor.") from e


def validate_dataloader(dataloader):
    """Ensures the dataloader is initialized and not empty."""
    if not isinstance(dataloader, DataLoader):
        raise TypeError(
            "The dataloader must be an instance of torch.utils.data.DataLoader."
        )
    if len(dataloader) == 0:
        raise ValueError(
            "The dataloader is empty. Please provide a dataloader with data."
        )


def get_config(kwargs, key, default_value):
    """
    Safely retrieves a configuration value from kwargs with a default.
    Validates type or value range if necessary.
    """
    value = kwargs.get(key, default_value)
    # Add specific validations if necessary, e.g., check type or value range
    return value


def validate_model_and_criterion(model, criterion):
    """Validates that model and criterion are initialized."""
    if model is None:
        raise ValueError("Model is not initialized.")
    if criterion is None:
        raise ValueError("Criterion (loss function) is not initialized.")


def evaluate_model(model, dataloader, criterion, **kwargs):
    """
    Evaluate the model performance on a given dataloader.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): The DataLoader for evaluation data.
        criterion: The loss function used for evaluation.

    Returns:
        tuple: Returns a tuple containing average loss, MAE, RMSE, and R2 score.
    """
    model.eval()  # Set model to evaluation mode
    total_loss, total_mae, total_rmse, total_r2_score, total_count = 0, 0, 0, 0, 0

    with torch.no_grad():  # No need to track gradients
        for X, y in dataloader:
            X, y = X.float(), y.unsqueeze(-1).float()
            X, y = map_tensor_to_mps(X, **kwargs), map_tensor_to_mps(
                y, **kwargs
            )  # Map tensors to Metal Performance Shaders (MPS) if available

            predictions = model(X)
            loss = criterion(predictions, y)

            # Detach predictions and labels from the graph and move to CPU for metric calculation
            predictions_np = safe_tensor_to_numpy(predictions)
            labels_np = safe_tensor_to_numpy(y)

            # Accumulate metrics
            total_loss += loss.item() * X.size(0)
            total_mae += mean_absolute_error(labels_np, predictions_np) * X.size(0)
            total_rmse += np.sqrt(
                mean_squared_error(labels_np, predictions_np)
            ) * X.size(0)
            total_r2_score += r2_score(labels_np, predictions_np) * X.size(0)
            total_count += X.size(0)

    # Calculate average metrics
    avg_loss = total_loss / total_count
    avg_mae = total_mae / total_count
    avg_rmse = total_rmse / total_count
    avg_r2_score = total_r2_score / total_count

    return avg_loss, avg_mae, avg_rmse, avg_r2_score


def train(
    pipeline: Tuple[nn.Module, DataLoader, DataLoader],
    **kwargs,
):
    """
    Train and evaluate the neural network model.

    Args:
        model (nn.Module): The neural network model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        test_dataloader (DataLoader): DataLoader for the test data.
        **kwargs: Keyword arguments for configurations like epochs, criterion_config, and
        optimiser_config.

    Prints:
        Loss and metric information for each epoch.
    """
    model, train_dataloader, test_dataloader = pipeline[0], pipeline[1], pipeline[2]
    # Config setup and error handling
    validate_dataloader(train_dataloader)
    validate_dataloader(test_dataloader)
    criterion = create_criterion(**kwargs.get("criterion_config", {}))
    validate_model_and_criterion(model, criterion)
    optimiser = create_optimiser(
        model.parameters(), **kwargs.get("optimiser_config", {})
    )
    scheduler = lr_scheduler.StepLR(
        optimiser,
        step_size=kwargs.get("scheduler_step_size", 1),
        gamma=kwargs.get("scheduler_gamma", 0.1),
    )
    epochs = kwargs.get("epochs")
    map_model_to_mps(model, **kwargs)  # Map model to MPS if available

    # Model training
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        for X, y in train_dataloader:
            X, y = X.float(), y.unsqueeze(-1).float()
            X, y = map_tensor_to_mps(X, **kwargs), map_tensor_to_mps(y, **kwargs)
            optimiser.zero_grad()

            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimiser.step()

            total_train_loss += loss.item() * X.size(0)

        scheduler.step()

        # Evaluate model performance on both training and test datasets
        train_loss, train_mae, train_rmse, _ = evaluate_model(
            model, train_dataloader, criterion
        )
        test_loss, test_mae, test_rmse, _ = evaluate_model(
            model, test_dataloader, criterion
        )

        # Print training and evaluation results for the current epoch
        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}"
        )
        print(
            f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}"
        )
