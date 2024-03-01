import torch
import torch.nn as nn
from src.models.model_definitions import LinearModel


def check_mps_availability(**kwargs):
    """
    Checks if MPS (Metal Performance Shaders) is available and returns the device.

    Parameters:
    - **kwargs: Keyword arguments, expecting 'device' to specify the desired device type.

    Returns:
    - torch.device: The device to use, either 'mps' if available or 'cpu' as fallback.
    """
    device_type = kwargs.get("device", "cpu")  # Default to CPU if not specified
    if device_type == "mps" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
            )
        device_type = "cpu"  # Fallback to CPU if MPS is not available
    return torch.device(device_type)


def map_model_to_mps(model: nn.Module, **kwargs):
    """
    Moves the given model to MPS if available.

    Parameters:
    - model: The PyTorch model to be moved.
    - **kwargs: Keyword arguments for device selection and additional configurations.
    """
    mps_device = check_mps_availability(**kwargs)
    model.to(mps_device)
    print(f"{type(model).__name__} successfully mapped to {mps_device}.")


def map_tensor_to_mps(tensor: torch.Tensor, **kwargs):
    """
    Moves the given tensor to MPS if available.

    Parameters:
    - tensor: The PyTorch tensor to be moved.
    - **kwargs: Keyword arguments for device selection and additional configurations.

    Returns:
    - torch.Tensor: The tensor moved to the specified device.
    """
    mps_device = check_mps_availability(**kwargs)
    return tensor.to(mps_device)


import torch


def create_optimiser(model_params, **kwargs):
    """
    Creates and returns a PyTorch optimiser based on the provided arguments.

    This function allows for dynamic creation of an optimiser for a model,
    using the specified learning rate, momentum, and other relevant parameters.

    Parameters:
    - model_params (iterable): The parameters of the model to optimise.
    - **kwargs: Arbitrary keyword arguments including:
        - name (str): The name of the optimiser to create (default: 'adam').
        - lr (float): The learning rate (default: 0.01).
        - momentum (float): The momentum used with some optimisers (default: 0.9).

    Returns:
    - optimiser (torch.optim.Optimizer): The created PyTorch optimiser.

    Raises:
    - ValueError: If an unsupported optimiser name is provided.

    Example Usage:
    ```python
    model_params = model.parameters()
    optimiser_kwargs = {'name': 'adam', 'lr': 0.001}
    optimiser = create_optimiser(model_params, **optimiser_kwargs)
    ```
    """
    optimiser_name = kwargs.get("name", "adam").lower()
    lr = kwargs.get("lr", 0.01)  # Default learning rate
    momentum = kwargs.get("momentum", 0.9)  # Default momentum

    if optimiser_name == "adam":
        optimiser = torch.optim.Adam(model_params, lr=lr)
    elif optimiser_name == "sgd":
        optimiser = torch.optim.SGD(model_params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimiser: {optimiser_name}")

    return optimiser


def create_criterion(**kwargs):
    """
    Creates and returns a PyTorch loss function (criterion) based on the provided arguments.

    This function facilitates the dynamic selection of a loss function for model training,
    according to the specified criterion name.

    Parameters:
    - **kwargs: Arbitrary keyword arguments including:
        - name (str): The name of the criterion to create (default: 'mse').

    Returns:
    - criterion (torch.nn.modules.loss._Loss): The PyTorch loss function.

    Raises:
    - ValueError: If an unsupported criterion name is provided.

    Example Usage:
    ```python
    criterion_kwargs = {'name': 'crossentropy'}
    criterion = create_criterion(**criterion_kwargs)
    ```
    """
    criterion_name = kwargs.get("name", "mse").lower()

    if criterion_name == "mse":
        criterion = torch.nn.MSELoss()
    elif criterion_name == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    return criterion
