import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the last time step
        return out
