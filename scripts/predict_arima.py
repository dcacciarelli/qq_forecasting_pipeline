import os
import yaml
import joblib
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

from qq_forecasting.models.arima_model import forecast_arima
from qq_forecasting.utils import inverse_scale, forecast_metrics, plot_forecast_vs_actual

# ===== WARNINGS =====
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ===== LOAD CONFIG =====
with open("config/arima_demand.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ===== LOAD DATA =====
test = pd.read_csv(os.path.join(cfg["paths"]["data_path"], "test.csv")).squeeze()[:48*7]
scaler = joblib.load(cfg["paths"]["scaler_path"])

# ===== LOAD MODEL =====
model = joblib.load(cfg["paths"]["model_path"])
model_path = cfg["paths"]["model_path"]
print(f"Loaded ARIMA model from {model_path}")

# ===== FORECAST =====
forecast = forecast_arima(model, steps=len(test))
forecast_inv = inverse_scale(np.array(forecast), scaler)

# ===== EVALUATE & PLOT =====
metrics = forecast_metrics(test.values, forecast_inv, save_path=cfg["paths"]["metrics_path"], print_scores=True)
plot_forecast_vs_actual(test.values, forecast_inv, save_path=cfg["paths"]["plot_path"])
