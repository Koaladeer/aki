import torch.nn as nn
import torch


class LSTMBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        """
        LSTM-based model for stock prediction.

        Parameters:
        - input_size: Number of input features
        - hidden_size: Number of hidden units in each LSTM layer
        - num_layers: Number of stacked LSTM layers
        - output_size: Number of output features (default is 1 for regression)
        """
        super(LSTMBase, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
        - Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Use the last time step's output for prediction
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Fully connected layer
        out = self.fc(out)  # Shape: (batch_size, output_size)
        return out
