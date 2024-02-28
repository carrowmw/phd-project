import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_performance_metrics(predictions, targets):
    """
    Compute common performance metrics for regression.

    Parameters:
        - predictions (torch.Tensor): The predicted values.
        - targets (torch.Tensor): The true target values.

    Returns:
        dict: A dictionary containing the following metrics:
            - MAE (float): Mean Absolute Error.
            - MSE (float): Mean Squared Error.
            - rmse (float): Root Mean Squared Error.
            - R^2 (float): R squared or Coefficient of Determination.

    Note:
        The function assumes the predictions and targets are torch tensors.
        They are then flattened and detached before computation.
    """
    predictions_flat_np = predictions.flatten()
    targets_flat_np = targets.flatten()

    return {
        "MAE": mean_absolute_error(targets_flat_np, predictions_flat_np),
        "MSE": mean_squared_error(targets_flat_np, predictions_flat_np),
        "rmse": np.sqrt(mean_absolute_error(targets_flat_np, predictions_flat_np)),
        "R^2": r2_score(targets_flat_np, predictions_flat_np),
    }
