def save_model_state(
    model,
    model_name,
    folder_name,
):
    """
    Save the state of a model to disk.

    Args:
        model (torch.nn.Module): The model to save.
        model_name (str): Name for the saved model file (without extension).
        path (str): Directory where the model should be saved.
    """
    path = "C:\\#code\\#python\\#current\\mres-project\\analysis_files\\"
    path = os.path.join(path, folder_name)
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists
    model_path = os.path.join(path, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)
