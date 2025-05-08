import torch
import torch.nn as nn
import math
import numpy as np
from typing import List, Union


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]  # assuming x is (batch, seq_len, d_model)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=1, feature_size=250, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, feature_size)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=10, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.embedding_layer(x)  # (batch, seq_len, feature_size)
        x = x + self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)  # (batch, seq_len, d_model)
        return self.decoder(output[:, -1, :])  # last time step per batch

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)


def train_transformer(model, dataloader, num_epochs=10, lr=0.001, scheduler=None):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            # X_batch = X_batch.transpose(0, 1)  # (seq_len, batch_size, 1)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


def forecast_transformer(
    model: torch.nn.Module,
    recent_window: Union[np.ndarray, List[float]],
    steps_ahead: int = 48,
    device: str = "cpu"
) -> np.ndarray:
    """
    Forecast future steps using a trained Transformer model, assuming inputs are already scaled.

    Args:
        model (nn.Module): Trained model.
        recent_window (np.ndarray): Most recent **scaled** values.
        steps_ahead (int): Number of future steps to forecast.
        device (str): 'cpu' or 'cuda'.

    Returns:
        np.ndarray: Forecasted values (still scaled).
    """
    model.eval()
    window = list(recent_window[-len(recent_window):])  # make mutable
    predictions = []

    for _ in range(steps_ahead):
        x = torch.tensor(window[-len(recent_window):], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # shape = (1, seq_len, 1)
        with torch.no_grad():
            pred = model(x).item()  # scalar output
        predictions.append(pred)
        window.append(pred)

    return np.array(predictions).flatten()


def forecast_transformer_autoregressive(
    model: torch.nn.Module,
    recent_window: Union[np.ndarray, List[float]],
    steps_ahead: int = 48,
    device: str = "cpu"
) -> np.ndarray:
    """
    Forecast future steps using a trained Transformer model in an autoregressive way.

    Args:
        model (nn.Module): Trained model.
        recent_window (np.ndarray): Most recent **scaled** values.
        steps_ahead (int): Number of future steps to forecast.
        device (str): 'cpu' or 'cuda'.

    Returns:
        np.ndarray: Forecasted values (still scaled).
    """
    model.eval()
    window = list(recent_window)  # ensure it's mutable
    predictions = []

    for _ in range(steps_ahead):
        x = torch.tensor(window[-len(recent_window):], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # shape = (1, seq_len, 1)
        with torch.no_grad():
            pred = model(x).item()
        predictions.append(pred)
        window.append(pred)

    return np.array(predictions)

