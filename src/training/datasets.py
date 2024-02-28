from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    A custom Dataset for time series data.

    Args:
        sequences (torch.Tensor): Input sequences for the dataset.
                                  Shape: [num_samples, sequence_length, feature_dim].
        targets (torch.Tensor): Corresponding targets for the input sequences.
                                Shape: [num_samples, feature_dim].
    """

    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, index):
        """
        Fetches the sequence and target at a particular index.

        Args:
            index (int): The index to retrieve the data from.

        Returns:
            tuple: A tuple containing:
                - sequence (torch.Tensor): Input sequence of shape [sequence_length, feature_dim].
                - target (torch.Tensor): Corresponding target of shape [feature_dim].
        """
        return self.sequences[index], self.targets[index]
