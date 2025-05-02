import os
import torch
import joblib
import numpy as np
import pandas as pd

from qq_forecasting.transformer.transformer_model import (
    TransformerEncoder,
    forecast_with_daily_resets,
    forecast_transformer_autoregressive,
    forecast_transformer_with_truth
)

from qq_forecasting.utils import (
    inverse_scale,
    forecast_metrics,
    plot_forecast_vs_actual,
)

# ========== CONFIG ==========
DATA_PATH = "data/processed/electricity_demand"
MODEL_SAVE_PATH = "outputs/models/transformer_model.pt"
SCALER_PATH = os.path.join(DATA_PATH, "scaler.pkl")
METRICS_SAVE_PATH = "outputs/results/transformer_metrics.txt"
PLOT_SAVE_PATH = "outputs/results/transformer_forecast.png"
WINDOW_SIZE = 48
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD ==========
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).squeeze()
val = pd.read_csv(os.path.join(DATA_PATH, "val.csv")).squeeze()
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv")).squeeze()
scaler = joblib.load(SCALER_PATH)
train_val = pd.concat([train, val], ignore_index=True)
actual_values = inverse_scale(test.values, scaler)

# ========== LOAD MODEL ==========
loaded_model = TransformerEncoder().to(device)
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
print(f"Loaded model from {MODEL_SAVE_PATH}")

# ========== FORECAST ==========
context_window = train_val[-WINDOW_SIZE:]
predictions = forecast_transformer_autoregressive(
    model=loaded_model,
    recent_window=context_window,
    steps_ahead=len(test)
)

predictions = forecast_with_daily_resets(
    model=loaded_model,
    test_series=test.values,
    context_window=context_window,
    horizon=48
)

# ========== EVALUATE ==========
predicted_values = inverse_scale(predictions, scaler)
metrics = forecast_metrics(actual_values, predicted_values, save_path=METRICS_SAVE_PATH, print_scores=True)
plot_forecast_vs_actual(actual_values, predicted_values, save_path=PLOT_SAVE_PATH)
