import os
import yaml
import torch
import joblib
import pandas as pd

from qq_forecasting.models.lstm_model import LSTM, forecast_lstm_autoregressive
from qq_forecasting.utils import inverse_scale, forecast_metrics, plot_forecast_vs_actual

# ========== LOAD CONFIG ==========
with open("config/lstm_demand.yaml") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["paths"]["data_path"]
MODEL_SAVE_PATH = config["paths"]["model_path"]
SCALER_PATH = config["paths"]["scaler_path"]
METRICS_SAVE_PATH = config["paths"]["metrics_path"]
PLOT_SAVE_PATH = config["paths"]["plot_path"]
PREDICTION_PATH = config["paths"]["prediction_path"]

hp = config["training"]
model_cfg = config["model"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD ==========
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).squeeze()
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv")).squeeze()
scaler = joblib.load(SCALER_PATH)

# ========== LOAD MODEL ==========
model = LSTM(**model_cfg).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()
print(f"Loaded LSTM model from {MODEL_SAVE_PATH}")

# ========== FORECAST ==========
context_window = train[-hp["window_size"]:].values
predictions = forecast_lstm_autoregressive(
    model=model,
    recent_window=context_window,
    steps_ahead=len(test),
    window_size=hp["window_size"]
)

# ========== EVALUATE ==========
predicted_values = inverse_scale(predictions, scaler)
metrics = forecast_metrics(test.values, predicted_values, save_path=METRICS_SAVE_PATH, print_scores=True)
plot_forecast_vs_actual(test.values, predicted_values, save_path=PLOT_SAVE_PATH)
pd.Series(predicted_values).to_csv(PREDICTION_PATH, index=False)
