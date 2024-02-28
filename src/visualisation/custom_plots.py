def get_custom_palette():
    """
    Generate and return a predefined seaborn color palette.

    This function creates a color palette with predefined colors, sets it as the current
    seaborn palette and then returns the created palette. The returned palette is a list
    of RGB tuples, where each tuple represents a color.

    Returns:
        list: A list of RGB tuples defining the color palette.
    """
    colors = [
        "#4060AF",
        "#FF5416",
        "#FDC82F",
        "#00B2A9",
        "#E7E6E6",
        "#93509E",
        "#00A9E0",
        "#CF0071",
    ]
    sns.set_palette(colors)
    return sns.color_palette(colors)


def get_custom_colormap():
    """
    Return a custom matplotlib colormap.

    This function creates a custom colormap from a predefined set of colors.

    Returns:
        LinearSegmentedColormap: A colormap object that can be used in plotting functions.
    """
    colors = ["#22335C", "#00B2A9", "#FDC82F"]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)


def get_custom_heatmap():
    """
    Return a custom matplotlib colormap.

    This function creates a custom colormap from a predefined set of colors.

    Returns:
        LinearSegmentedColormap: A colormap object that can be used in plotting functions.
    """
    colors = ["#00A9E0", "#CF0071"]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)


def plot_windows(inputs, predictions, targets, horizon, num_plots=5, step=1, title=""):
    """
    Plot a random selection of the given inputs, predictions, and targets.

    Parameters:
    inputs (array-like): The input data.
    predictions (array-like): The predicted values.
    targets (array-like): The actual target values.
    horizon (int): The prediction horizon.
    num_plots (int, optional): The number of plots to make. Defaults to 5.
    step (int, optional): The step to take between plots. Defaults to 1.

    Returns:
    None
    """
    custom_palette = get_custom_palette()
    num_plots = min(num_plots, len(inputs))
    start_idx = np.random.choice(len(inputs) - num_plots)

    # Get the global minimum and maximum y-values for the entire inputs dataset
    min_y = -1
    max_y = 4

    fig, axs = plt.subplots(num_plots, 1, figsize=(6, 2 * num_plots))
    for i in range(num_plots):
        idx = start_idx + step * i
        axs[i].plot(
            range(len(inputs[idx])),
            inputs[idx],
            label="Inputs",
            color=custom_palette[0],
        )
        axs[i].scatter(
            range(
                len(inputs[idx]) + horizon,
                len(inputs[idx]) + horizon + len(predictions[idx]),
            ),
            predictions[idx],
            label="Predictions",
            color=custom_palette[1],
            marker="x",
            s=60,
        )
        axs[i].scatter(
            range(
                len(inputs[idx]) + horizon,
                len(inputs[idx]) + horizon + len(targets[idx]),
            ),
            targets[idx],
            label="Targets",
            color=custom_palette[0],
        )
        axs[i].set_ylim(min_y, max_y)
        if i == 0:
            axs[i].legend(loc="upper left")
            axs[i].set_xlabel("Step (15 minutes)")
            axs[i].set_ylabel("Scaled value")

        # Add grid to the plots
        axs[i].grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.suptitle(title, fontsize=10, y=0.93)


def plot_mv_windows(
    inputs,
    predictions,
    targets,
    horizon,
    num_plots=5,
    step=1,
    title="",
    input_feature_names=None,
):
    """
    Plot a random selection of the given inputs, predictions, and targets.

    Parameters:
    inputs (array-like): The input data.
    predictions (array-like): The predicted values.
    targets (array-like): The actual target values.
    horizon (int): The prediction horizon.
    num_plots (int, optional): The number of plots to make. Defaults to 5.
    step (int, optional): The step to take between plots. Defaults to 1.

    Returns:
    None
    """
    print(inputs.shape, predictions.shape, targets.shape)

    custom_palette = get_custom_palette()
    num_plots = min(num_plots, len(inputs))

    if len(inputs) <= num_plots:
        start_idx = 0
    else:
        start_idx = np.random.choice(len(inputs) - num_plots)

    # Get the global minimum and maximum y-values for the entire inputs dataset
    min_y = -1.5
    max_y = 3.5

    # Check if multivariate
    is_multivariate = len(inputs.shape) > 1 and inputs.shape[1] > 1

    fig, axs = plt.subplots(
        num_plots, 1, figsize=(10, 4 * num_plots)
    )  # Adjusted from 6 to 8

    for i in range(num_plots):
        idx = start_idx + step * i

        current_input = inputs[idx]

        if is_multivariate and len(current_input.shape) > 1:
            for j in range(current_input.shape[1]):
                axs[i].plot(
                    range(len(current_input)),
                    current_input[:, j],
                    label=input_feature_names[j],
                    color=custom_palette[j % len(custom_palette)],
                )
        else:
            axs[i].plot(
                range(len(current_input)),
                current_input,
                label="Inputs",
                color=custom_palette[0],
            )

        # Handle predictions and targets as scalar values
        prediction_value = predictions[i]  # Assuming predictions are 1D
        target_value = targets[i]  # Assuming targets are 1D

        axs[i].scatter(
            len(inputs[idx]) + horizon,
            prediction_value,
            label="Predictions",
            color=custom_palette[-1],
            marker="x",
            s=60,
        )
        axs[i].scatter(
            len(inputs[idx]) + horizon,
            target_value,
            label="Targets",
            color=custom_palette[0],
        )

        axs[i].set_ylim(min_y, max_y)
        if i == 0:
            axs[i].legend(loc="upper left")
            axs[i].set_xlabel("Timetep (15 minutes)")
            axs[i].set_ylabel("Scaled value")

        # Add grid to the plots
        # axs[i].grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.suptitle(title, y=0.90)
