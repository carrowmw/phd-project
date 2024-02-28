def compute_metrics(model, dataloader, criterion):
    """
    Compute MSE and rmse metrics for a given model and dataloader.

    Args:
    - model (torch.nn.Module): PyTorch model.
    - dataloader (torch.utils.data.DataLoader): DataLoader object.
    - criterion (torch.nn.Module): Loss function.

    Returns:
    - tuple: (Mean Squared Error, Root Mean Squared Error).
    """
    model.eval()
    test_mse, test_rmse = 0, 0

    for X, y in dataloader:
        with torch.inference_mode():
            X = X.float().to(device)
            y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)
            predictions = model(X)
            loss_mse = criterion(predictions.unsqueeze(dim=1), y)

            test_mse += loss_mse.item()
            test_rmse += np.sqrt(loss_mse.item())

    test_mse /= len(dataloader)
    test_rmse /= len(dataloader)

    return test_mse, test_rmse


def evaluate_anomalies(predictions, targets):
    errors = np.abs(predictions - targets)

    # Set anomaly threshold
    error_deviations = 6
    anomaly_threshold = np.round(errors.mean() + error_deviations * errors.std(), 1)
    anomalies = errors > anomaly_threshold

    # Count total anomalies
    total_anomalies = np.sum(anomalies)

    anomaly_percentage = (total_anomalies / len(targets)) * 100
    mean_error = errors.mean()

    return anomaly_percentage, anomaly_threshold, mean_error


def evaluate_linear_model(
    data,
    completeness,
    input_feature_indices,
    target_feature_index,
    sequence_length,
    horizon,
    epochs,
    stride,
    total_iterations,
):
    """
    Evaluate the performance of the Linear Model.

    Args:
    - east_timeseries (pandas.DataFrame): Processed timeseries data.
    - completeness (float): The completeness threshold.
    - horizon (int): The forecasting horizon.

    Returns:
    - tuple: (Linear Model Weights, Performance Metrics).
    """
    test_metrics_df = pd.DataFrame()
    train_metrics_df = pd.DataFrame()

    print(
        f"Test Number: {total_iterations} |  Completeness: {completeness} | Sequence Length {sequence_length} | Horizon: {horizon} | Window Size: {1}"
    )

    (
        train_dataloader,
        test_dataloader,
        test_inputs,
        test_targets,
        train_inputs,
        train_targets,
    ) = prepare_dataloaders(
        data=data[:sequence_length],
        window_size=1,
        input_feature_indices=input_feature_indices,
        target_feature_index=target_feature_index,
        horizon=horizon,
        stride=stride,
        batch_size=1,
        shuffle=False,
    )

    # Train and evaluate linear model
    linear_model = LinearModel(input_size=len(input_feature_indices)).to(device)
    print("\nTraining Linear Model...")
    start_time = timeit.default_timer()
    linear_weights, train_metrics, test_metrics = train_and_evaluate_linear_model(
        linear_model, train_dataloader, test_dataloader, epochs
    )

    end_time = timeit.default_timer()

    train_metrics_entry = {
        "Completeness": completeness,
        "Sequence Length": sequence_length,
        "Horizon": horizon,
        "Window Size": 1,
        "Train Metrics": train_metrics,
        "Test Metrics": test_metrics,
        "Linear Weights": linear_weights,
        "Test Number": total_iterations,
    }

    train_metrics_df = train_metrics_df._append(train_metrics_entry, ignore_index=True)

    # Compute performance metrics
    test_mse, test_rmse = compute_metrics(linear_model, test_dataloader, nn.MSELoss())
    print()
    print("Evaluating Linear Model...")
    print(f"Test MSE: {test_mse:.4f} | Test rmse: {test_rmse:.4f}")
    print()

    test_metrics_entry = {
        "Completeness": completeness,
        "Sequence Length": sequence_length,
        "Horizon": horizon,
        "Window Size": 1,
        "Test MSE": test_mse,
        "Test rmse": test_rmse,
        "Test Number": total_iterations,
    }

    test_metrics_df = test_metrics_df._append(test_metrics_entry, ignore_index=True)

    all_predictions = []
    for X, _ in test_dataloader:
        batch_predictions = linear_model(X.squeeze(0).float().to(device))
        all_predictions.append(batch_predictions.detach().cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    test_targets_numpy = test_targets.detach().cpu().numpy().flatten()

    # Construct metrics dictionary
    metrics = compute_performance_metrics(all_predictions, test_targets_numpy)
    metrics.update(
        {
            "Model": "Linear",
            "Training Time (s)": end_time - start_time,
            "Test Number": total_iterations,
            "Completeness": completeness,
            "Sequence Length": sequence_length,
            "Horizon": horizon,
            "WindowSize": 1,
        }
    )

    folder_name = (
        "multivariate_model_states"
        if len(input_feature_indices) > 1
        else "univariate_model_states"
    )
    # Save model state
    save_model_state(
        linear_model,
        f"Linear_TestNumber{total_iterations}_Completeness{completeness}_SequenceLength{sequence_length}_Horizon{horizon}_WindowSize1",
        folder_name=folder_name,
    )

    test_metrics_df.to_csv(
        "linear_test_metrics.csv",
        index=False,
        mode="a",
        header=not os.path.exists("linear_test_metrics.csv"),
    )
    train_metrics_df.to_csv(
        "linear_train_metrics.csv",
        index=False,
        mode="a",
        header=not os.path.exists("linear_train_metrics.csv"),
    )

    return linear_model.linear.weight.detach().numpy(), metrics


def evaluate_lstm_model(
    data,
    completeness,
    input_feature_indices,
    target_feature_index,
    sequence_length,
    horizon,
    window_size,
    epochs,
    total_iterations,
):
    """
    Evaluate the performance of the LSTM Model.

    Args:
    - east_timeseries (pandas.DataFrame): Processed timeseries data.
    - completeness (float): The completeness threshold.
    - horizon (int): The forecasting horizon.
    - window_size (int): The window size for LSTM.

    Returns:
    - dict: Performance Metrics.
    """

    test_metrics_df = pd.DataFrame()
    train_metrics_df = pd.DataFrame()

    print(
        f"Test Number: {total_iterations} | Completeness: {completeness} | Sequence Length {sequence_length} | Horizon: {horizon} | Window Size: {window_size} "
    )

    (
        train_dataloader,
        test_dataloader,
        test_inputs,
        test_targets,
        train_inputs,
        train_targets,
    ) = prepare_dataloaders(
        data[:sequence_length],
        window_size,
        input_feature_indices=input_feature_indices,
        target_feature_index=target_feature_index,
        horizon=horizon,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Train and evaluate LSTM model
    lstm_model = LSTMModel(feature_dim=len(input_feature_indices)).to(device)
    print("\nTraining LSTM Model...")
    start_time = timeit.default_timer()
    train_metrics, test_metrics = train_and_evaluate_lstm_model(
        lstm_model, train_dataloader, test_dataloader, epochs=epochs
    )
    end_time = timeit.default_timer()

    train_metrics_entry = {
        "Completeness": completeness,
        "Sequence Length": sequence_length,
        "Horizon": horizon,
        "Window Size": 1,
        "Train Metrics": train_metrics,
        "Test Metrics": test_metrics,
        "Test Number": total_iterations,
    }

    train_metrics_df = train_metrics_df._append(train_metrics_entry, ignore_index=True)

    # Compute performance metrics
    test_mse, test_rmse = compute_metrics(lstm_model, test_dataloader, nn.MSELoss())
    print()
    print("Evaluating LSTM Model...")
    print(f"Test MSE: {test_mse:.4f} | Test rmse: {test_rmse:.4f}")
    print()

    test_metrics_entry = {
        "Completeness": completeness,
        "Sequence Length": sequence_length,
        "Horizon": horizon,
        "Window Size": 1,
        "Test MSE": test_mse,
        "Test rmse": test_rmse,
        "Test Number": total_iterations,
    }

    test_metrics_df = test_metrics_df._append(test_metrics_entry, ignore_index=True)

    all_lstm_predictions = []

    for X, _ in test_dataloader:
        batch_predictions = lstm_model(X.float().to(device))
        all_lstm_predictions.append(batch_predictions.detach().cpu().numpy())
    all_lstm_predictions_flat = np.concatenate(all_lstm_predictions).ravel()
    test_targets_numpy = test_targets.detach().cpu().numpy().flatten()

    # Construct metrics dictionary
    metrics = compute_performance_metrics(all_lstm_predictions_flat, test_targets_numpy)
    metrics.update(
        {
            "Model": "LSTM",
            "Training Time (s)": end_time - start_time,
            "Test Number": total_iterations,
            "Completeness": completeness,
            "Sequence Length": sequence_length,
            "Horizon": horizon,
            "WindowSize": window_size,
        }
    )
    folder_name = (
        "multivariate_model_states"
        if len(input_feature_indices) > 1
        else "univariate_model_states"
    )
    # Save model state
    save_model_state(
        lstm_model,
        f"LSTM_TestNumber{total_iterations}_Completeness{completeness}_SequenceLength{sequence_length}_Horizon{horizon}_WindowSize{window_size}",
        folder_name=folder_name,
    )

    test_metrics_df.to_csv(
        "lstm_test_metrics.csv",
        index=False,
        mode="a",
        header=not os.path.exists("lstm_test_metrics.csv"),
    )
    train_metrics_df.to_csv(
        "lstm_train_metrics.csv",
        index=False,
        mode="a",
        header=not os.path.exists("lstm_train_metrics.csv"),
    )

    return metrics
