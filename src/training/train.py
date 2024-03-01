from src.utils.training_utils import create_criterion, create_optimiser


def train(
    model, train_dataloader, test_dataloader, **kwargs
):  # this probably needs to be a tuple instead
    # Get loss function and optimiser from training_config
    criterion = create_criterion()
    optimiser = create_optimiser()
