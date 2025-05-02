import os
import torch
import joblib
import numpy as np
import pandas as pd

from qq_forecasting.lstm.lstm_model import LSTM, forecast_lstm_autoregressive
from qq_forecasting.utils import (
    inverse_scale,
    forecast_metrics,
    plot_forecast_vs_actual
)

# ========== CONFIG ==========
DATA_PATH = "data/processed/electricity_demand"
MODEL_SAVE_PATH = "outputs/models/lstm_model.pt"
SCALER_PATH = os.path.join(DATA_PATH, "scaler.pkl")
METRICS_SAVE_PATH = "outputs/results/lstm_metrics.txt"
PLOT_SAVE_PATH = "outputs/results/lstm_forecast.png"

WINDOW_SIZE = 48
HIDDEN_SIZE = 64
NUM_LAYERS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD ==========
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).squeeze()
val = pd.read_csv(os.path.join(DATA_PATH, "val.csv")).squeeze()
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv")).squeeze()
scaler = joblib.load(SCALER_PATH)

train_val = pd.concat([train, val], ignore_index=True)
actual_values = inverse_scale(test.values, scaler)

# ========== LOAD MODEL ==========
model = LSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
print(f"✅ Loaded LSTM model from {MODEL_SAVE_PATH}")

# ========== FORECAST ==========
context_window = train_val[-WINDOW_SIZE:].values  # scaled
predictions = forecast_lstm_autoregressive(
    model=model,
    recent_window=context_window,
    steps_ahead=len(test),
    window_size=WINDOW_SIZE
)

# ========== EVALUATE ==========
predicted_values = inverse_scale(predictions, scaler)
metrics = forecast_metrics(actual_values, predicted_values, save_path=METRICS_SAVE_PATH, print_scores=True)
plot_forecast_vs_actual(actual_values, predicted_values, save_path=PLOT_SAVE_PATH)
