import torch
import torch.nn as nn


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Attention Layer
        self.attention = nn.Linear(hidden_size, 1)

        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM Forward Pass
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)

        # Attention Mechanism
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Weighted Sum of LSTM Outputs
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_size)

        # Output Layer
        output = self.fc(context_vector)  # (batch_size, output_size)
        return output


# Example Usage
batch_size = 32
seq_len = 10
input_size = 8
hidden_size = 64
output_size = 1

model = AttentionLSTM(input_size, hidden_size, output_size)
x = torch.rand(batch_size, seq_len, input_size)
output = model(x)
print(output.shape)  # Output: (batch_size, output_size)
