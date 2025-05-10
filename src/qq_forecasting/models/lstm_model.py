# src/qq_forecasting/models/lstm_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Union


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # dropout only applies if num_layers > 1
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_lstm(model, dataloader, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")


def forecast_lstm_autoregressive(
    model: torch.nn.Module,
    recent_window: Union[np.ndarray, List[float]],
    steps_ahead: int,
    window_size: int,
    device: str = "cpu"
) -> np.ndarray:
    """
    Perform autoregressive forecasting with LSTM by iteratively predicting the next step
    and feeding it back as input.

    Args:
        model (torch.nn.Module): Trained LSTM model.
        recent_window (np.ndarray or list): Initial input sequence (scaled).
        steps_ahead (int): Number of future steps to predict.
        window_size (int): Size of input window.
        device (str): 'cpu' or 'cuda'.

    Returns:
        np.ndarray: Forecasted values (still scaled).
    """
    model.eval()
    window = list(recent_window[-window_size:])
    predictions = []

    for _ in range(steps_ahead):
        x = torch.tensor(window[-window_size:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq_len, 1)
        with torch.no_grad():
            pred = model(x).item()
        predictions.append(pred)
        window.append(pred)

    return np.array(predictions)
