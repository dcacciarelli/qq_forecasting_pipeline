import os
import yaml
import logging
import joblib
import pandas as pd

from qq_forecasting.data.load_demand import load_demand_data
from qq_forecasting.models.arima_model import fit_arima_model
from qq_forecasting.tuning.tuning_arima import tune_arima

# Setup logging
os.makedirs('outputs', exist_ok=True)
logging.basicConfig(filename='outputs/train_arima.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load config
with open("config/arima_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load and prepare data
df = load_demand_data(config["data"]["folder_path"], config["data"]["years"])
series = df["national_demand"]

# Hyperparameter tuning
p_values = config["model"]["p_range"]
d_values = config["model"]["d_range"]
q_values = config["model"]["q_range"]
initial_train_size = config["data"]["initial_train_size"]
horizon = config["data"]["forecast_horizon"]
step = config["data"]["step_size"]

best_order, best_rmse = tune_arima(series, p_values, d_values, q_values, initial_train_size, horizon, step)
logging.info(f"Best ARIMA order: {best_order} with RMSE={best_rmse:.2f}")

# Train final model
final_model = fit_arima_model(series, best_order)

# Save model
joblib.dump(final_model, "outputs/models/arima_model.pkl")
logging.info("Final ARIMA model saved successfully.")
