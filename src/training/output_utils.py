def save_outputs_to_csv(performance_df, weights_df, runtime_dict):
    """
    Save the metrics and weights dataframes to CSV.
    :param performance_df: dataframe containing performance metrics
    :param weights_df: dataframe containing model weights
    :param runtime_dict: dictionary containing runtimes
    """
    performance_df.to_csv(
        "performance_metrics_mm.csv",
        mode="a",
        index=False,
        header=not os.path.exists("performance_metrics_mm.csv"),
    )
    weights_df.to_csv(
        "weights_mm.csv",
        mode="a",
        index=False,
        header=not os.path.exists("weights_mm.csv"),
    )
    runtime_df = pd.DataFrame(
        list(runtime_dict.items()), columns=["Parameters", "Runtime (s)"]
    )
    runtime_df.to_csv(
        "runtime_mm.csv",
        mode="w",
        index=False,
        header=not os.path.exists("runtime_mm.csv"),
    )
