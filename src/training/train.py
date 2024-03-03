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


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    **kwargs,
):  # This probably needs to be a tuple instead for the pipeline to work
    # Get loss function and optimiser from training_config
    criterion = create_criterion()
    optimiser = create_optimiser(model.parameters())
    scheduler = lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.9)

    epochs = kwargs.get("epochs")

    map_model_to_mps(model)

    linear_weights = []
    train_metrics = []
    test_metrics = []

    for epoch in range(epochs):
        train_loss, train_mae, train_rmse, train_r2 = 0, 0, 0, 0
        model.train()

        for X, y in train_dataloader:
            # Prepare data and move to mps if available
            X = X.float()
            y = y.unsqueeze(-1).unsqueeze(-1).float()
            map_tensor_to_mps(X)
            map_tensor_to_mps(y)

            # Model prediction and loss calculation
            train_preds = model(X)
            loss = criterion(train_preds.unsqueeze(dim=1), y)

            # Backprop and optimisation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Accumulate loss and other metrics
            train_loss += loss.item()


def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the model on a given dataloader."""
    model.eval()
    total_loss, total_mae, total_mse, total_count = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)

            mae = torch.nn.functional.l1_loss(preds, y, reduction="sum")
            mse = torch.nn.functional.mse_loss(preds, y, reduction="sum")

            total_loss += loss.item() * X.size(0)
            total_mae += mae.item()
            total_mse += mse.item()
            total_count += X.size(0)

    avg_loss = total_loss / total_count
    avg_mae = total_mae / total_count
    avg_mse = total_mse / total_count
    return avg_loss, avg_mae, avg_mse


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    **kwargs,
):  # This probably needs to be a tuple instead for the pipeline to work
    # Get loss function and optimiser from training_config
    criterion = create_criterion(**kwargs.get("criterion_config", {}))
    optimiser = create_optimiser(
        model.parameters(), **kwargs.get("optimiser_config", {})
    )
    scheduler = lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.9)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    epochs = kwargs.get("epochs", 10)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item() * X.size(0)

        scheduler.step()

        # Evaluation on the training set
        train_loss, train_mae, train_mse = evaluate_model(
            model, train_dataloader, criterion, device
        )
        # Evaluation on the test set
        test_loss, test_mae, test_mse = evaluate_model(
            model, test_dataloader, criterion, device
        )

        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train MSE: {train_mse:.4f}"
        )
        print(
            f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}"
        )
