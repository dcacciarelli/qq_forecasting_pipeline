import os
import joblib
import numpy as np
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from qq_forecasting.models.arima_model import forecast_arima
from qq_forecasting.utils import inverse_scale, forecast_metrics, plot_forecast_vs_actual


# Suppress annoying convergence warnings from statsmodels
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ===== CONFIG =====
DATA_PATH = "data/processed/electricity_demand"
MODEL_SAVE_PATH = "outputs/models/arima_model.pkl"
METRICS_SAVE_PATH = "outputs/results/arima_metrics.txt"
PLOT_SAVE_PATH = "outputs/results/arima_forecast.png"

SEASONALITY = 48  # daily seasonality for half-hourly data

# Hyperparameter grid
p_values = [1, 2]
d_values = [0]
q_values = [1]
P_values = [1]
D_values = [0]
Q_values = [1]

# ===== LOAD DATA =====
# train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).iloc[:200, :]
# val = pd.read_csv(os.path.join(DATA_PATH, "val.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))[:48*7]
scaler = joblib.load(os.path.join(DATA_PATH, "scaler.pkl"))

# ===== TUNE =====
# best_order, best_seasonal_order, _ = tune_arima(
#     train=train,
#     val=val,
#     p_values=p_values,
#     d_values=d_values,
#     q_values=q_values,
#     P_values=P_values,
#     D_values=D_values,
#     Q_values=Q_values,
#     s=SEASONALITY
# )
#
# # ===== TRAIN AND SAVER FINAL MODEL (train + val) =====
# train_val = pd.concat([train, val], ignore_index=True)
# fit_arima(series=train_val, order=best_order, seasonal_order=best_seasonal_order, save_path=MODEL_SAVE_PATH)

# ===== LOAD MODEL =====
model = joblib.load(MODEL_SAVE_PATH)

# ===== FORECAST =====
forecast = forecast_arima(model, steps=len(test))

# ===== INVERSE SCALE =====
forecast_inv = inverse_scale(np.array(forecast), scaler)
test_inv = inverse_scale(test.values, scaler)

# ===== EVALUATE & PLOT =====
metrics = forecast_metrics(test_inv, forecast_inv, save_path=METRICS_SAVE_PATH, print_scores=True)
plot_forecast_vs_actual(test_inv, forecast_inv, save_path=PLOT_SAVE_PATH)
