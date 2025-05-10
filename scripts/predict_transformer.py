import os
import yaml
import torch
import joblib
import pandas as pd

from qq_forecasting.models.transformer_model import TransformerEncoder, forecast_transformer_autoregressive
from qq_forecasting.utils import inverse_scale, forecast_metrics, plot_forecast_vs_actual

# ========== LOAD CONFIG ==========
with open("config/transformer_demand.yaml") as f:
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
model = TransformerEncoder(**model_cfg).to(device)
model.load_state_dict(torch.load(config["paths"]["model_path"], map_location=device))
model.eval()
print(f"Loaded Transformer model from {config['paths']['model_path']}")

# ========== FORECAST ==========
context_window = train[-hp["window_size"]:].values
predictions = forecast_transformer_autoregressive(
    model=model,
    recent_window=context_window,
    steps_ahead=len(test)
)

# ========== EVALUATE ==========
predicted_values = inverse_scale(predictions, scaler)
forecast_metrics(test.values, predicted_values, save_path=config["paths"]["metrics_path"], print_scores=True)
plot_forecast_vs_actual(test.values, predicted_values, save_path=config["paths"]["plot_path"])
pd.Series(predicted_values).to_csv(PREDICTION_PATH, index=False)
