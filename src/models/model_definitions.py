import torch.nn as nn


class LinearModel(nn.Module):
    """
    A simple linear regression model suitable for time series forecasting.

    Parameters:
    - input_size (int): Number of input features.

    Attributes:
    - linear (nn.Linear): A linear layer that transforms input features into a single output.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

    Example:
    --------
    >>> model = LinearModel(input_size=10)
    >>> input_data = torch.randn(32, 10)  # Batch of 32, each with 10 features
    >>> output = model(input_data)

    Notes:
    ------
    - The forward method can process both 2D (batch_size, num_features) and
      3D (batch_size, sequence_len, num_features) input tensors. If the input is 3D,
      it gets reshaped to 2D.
    """

    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # If x is 3D (batch_size, sequence_len, num_features), we might need to reshape it
        x = x.reshape(x.size(0), -1)
        return self.linear(x)


class LSTMModel(nn.Module):
    """
    LSTM-based model designed for time series forecasting. Suitable for both univariate and multivariate time series.

    Parameters:
    - feature_dim (int): Number of expected features in the input `x`.
    - hidden_size (int, optional): Number of features in the hidden state. Default: 50.
    - output_dim (int, optional): Number of features in the output. Default: 1.
    - num_layers (int, optional): Number of recurrent layers. Default: 1.

    Attributes:
    - lstm (nn.LSTM): LSTM layer.
    - linear (nn.Linear): Linear layer to produce the final output.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

    Example:
    --------
    >>> model = LSTMModel(feature_dim=10)
    >>> input_data = torch.randn(32, 7, 10)  # Batch of 32, sequence length of 7, each with 10 features
    >>> output = model(input_data)
    """

    def __init__(self, feature_dim, hidden_size=50, output_dim=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        Forward propagation method for the LSTM model.

        Args:
        - x (torch.Tensor): Input tensor with sequences. Expected shape: [batch_size, sequence_length, feature_dim].

        Returns:
        - torch.Tensor: Output tensor with predictions. Shape: [batch_size, output_dim].
        """
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x[:, -1, :]  # Selecting the last output of the sequence
