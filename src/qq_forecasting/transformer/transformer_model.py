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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=1, feature_size=250, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, feature_size)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x: (seq_len, batch_size, input_dim)
        x = self.embedding_layer(x)
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            self.src_mask = self._generate_square_subsequent_mask(len(x)).to(x.device)
        x = x + self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        return self.decoder(output[-1, :, :])  # only take last time step

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
            X_batch = X_batch.transpose(0, 1)  # (seq_len, batch_size, 1)
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
        x = torch.tensor(window[-len(recent_window):], dtype=torch.float32).unsqueeze(-1).unsqueeze(1).to(device)
        with torch.no_grad():
            pred = model(x).item()  # scalar output
        predictions.append(pred)
        window.append(pred)

    return np.array(predictions).flatten()


def forecast_transformer_with_truth(
    model: torch.nn.Module,
    context_window: Union[np.ndarray, List[float]],
    future_values: Union[np.ndarray, List[float]],
    device: str = "cpu"
) -> np.ndarray:
    """
    Forecast using true future values for evaluation (not autoregressive).

    Args:
        model (torch.nn.Module): Trained model.
        context_window (array-like): Initial input window (already scaled).
        future_values (array-like): True future values to feed for prediction (already scaled).
        device (str): 'cpu' or 'cuda'.

    Returns:
        np.ndarray: Model predictions using the true inputs at each step (still scaled).
    """
    model.eval()
    window = list(context_window)
    predictions = []

    for t in range(len(future_values)):
        x = torch.tensor(window[-len(context_window):], dtype=torch.float32).unsqueeze(-1).unsqueeze(1).to(device)

        with torch.no_grad():
            pred = model(x).item()

        predictions.append(pred)
        # Append the *true* next value, not the predicted one
        window.append(future_values[t])

    return np.array(predictions)


def forecast_multiple_days_autoreg(
    model: torch.nn.Module,
    full_series: np.ndarray,
    window_size: int = 48,
    horizon: int = 48,
    num_days: int = 7,
    device: str = "cpu"
) -> List[np.ndarray]:
    """
    Perform rolling day-ahead forecasts for multiple days,
    resetting the input context each day to use observed truth.

    Args:
        model (nn.Module): Trained transformer model.
        full_series (np.ndarray): Full **unscaled** series (e.g., val + test).
        scaler (sklearn Scaler): Fitted scaler.
        window_size (int): Context window length (e.g., 48).
        horizon (int): Steps to forecast each day (e.g., 48).
        num_days (int): Number of daily forecasts to make.
        device (str): 'cpu' or 'cuda'.

    Returns:
        List[np.ndarray]: Each element is a day-ahead forecast of length `horizon` (unscaled).
    """
    model.eval()
    forecasts = []

    for day in range(num_days):
        # Get real context window
        start_idx = day * horizon
        context = full_series[start_idx : start_idx + window_size]

        preds = []
        window = list(context)

        for _ in range(horizon):
            x = torch.tensor(window[-window_size:], dtype=torch.float32).unsqueeze(-1).unsqueeze(1).to(device)
            with torch.no_grad():
                pred = model(x).item()
            preds.append(pred)
            window.append(pred)  # autoregressive

        # Inverse scale this day's forecast
        forecasts.append(preds)

    return forecasts


def forecast_with_daily_resets(
    model: torch.nn.Module,
    test_series: Union[np.ndarray, List[float]],
    context_window: Union[np.ndarray, List[float]],
    horizon: int = 48,
    device: str = "cpu"
) -> np.ndarray:
    """
    Forecast full test series in day-ahead blocks:
    - Each block uses autoregressive prediction for 48 steps.
    - After each block, reset context window to the true values.
    - No scaling is applied anywhere.

    Args:
        model (torch.nn.Module): Trained model (expects raw values).
        test_series (np.ndarray): True test series (raw).
        context_window (np.ndarray): Initial input values before test set (raw).
        horizon (int): Number of steps to predict per day (default: 48).
        device (str): "cpu" or "cuda".

    Returns:
        np.ndarray: Full predicted series (same length as test_series).
    """
    model.eval()
    test_series = np.array(test_series)
    context_window = np.array(context_window)
    predictions = []

    window_size = len(context_window)
    num_rounds = int(np.ceil(len(test_series) / horizon))

    for i in range(num_rounds):
        window = list(context_window.copy())
        block_preds = []

        for _ in range(horizon):
            x = torch.tensor(window[-window_size:], dtype=torch.float32).unsqueeze(-1).unsqueeze(1).to(device)
            with torch.no_grad():
                pred = model(x).item()
            block_preds.append(pred)
            window.append(pred)

        # Clip to remaining test points
        block_preds = block_preds[:len(test_series) - len(predictions)]
        predictions.extend(block_preds)

        # Reset context window with true values for next round
        start = i * horizon
        end = start + horizon
        context_window = test_series[start:end]
        if len(context_window) < window_size:
            context_window = np.concatenate([test_series[:start], context_window])[-window_size:]

    return np.array(predictions)


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
        x = torch.tensor(window[-len(recent_window):], dtype=torch.float32).unsqueeze(-1).unsqueeze(1).to(device)
        with torch.no_grad():
            pred = model(x).item()
        predictions.append(pred)
        window.append(pred)

    return np.array(predictions)

