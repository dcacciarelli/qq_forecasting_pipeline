import os
import yaml
import torch
import joblib
import pandas as pd
from qq_forecasting.models.arima_model import fit_arima

# ===== LOAD CONFIG =====
with open("config/arima_demand.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ===== LOAD DATA =====
train = pd.read_csv(os.path.join(cfg["paths"]["data_path"], "train.csv")).squeeze()

# ===== TRAIN AND SAVE MODEL =====
model = fit_arima(
    series=train,
    order=tuple(cfg["model"]["order"]),
    seasonal_order=tuple(cfg["model"]["seasonal_order"]),
    trend=cfg["model"].get("trend", None),
    enforce_stationarity=cfg["model"].get("enforce_stationarity", False),
    enforce_invertibility=cfg["model"].get("enforce_invertibility", False),
    initialization=cfg["model"].get("initialization", "approximate_diffuse"),
    measurement_error=cfg["model"].get("measurement_error", False),
    time_varying_regression=cfg["model"].get("time_varying_regression", False),
    mle_regression=cfg["model"].get("mle_regression", True),
    save_path=cfg["paths"]["model_path"]
)

