import yaml
import logging
from src.qq_forecasting.data.load_demand_data import load_and_merge_demand
from src.qq_forecasting.models.arima_model import fit_arima_model
from src.qq_forecasting.tuning.tuning_arima import tune_arima
import joblib
import os

# Ensure output folder exists
os.makedirs('../../../outputs', exist_ok=True)

# Load config
with open("../../../config/arima_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(filename='../../../outputs/train.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load data
folder_path = config["data"]["folder_path"]
years = range(config["data"]["years"][0], config["data"]["years"][-1]+1)
df = load_and_merge_demand(folder_path=folder_path, years=years)
df = df.asfreq('30min')
series = df["national_demand"]

# Grid search tuning
p_values = config["model"]["p_range"]
d_values = config["model"]["d_range"]
q_values = config["model"]["q_range"]

initial_train_size = eval(str(config["data"]["initial_train_size"]))
horizon = config["data"]["forecast_horizon"]
step = config["data"]["step_size"]

best_order, best_rmse = tune_arima(series, p_values, d_values, q_values, initial_train_size, horizon, step)

logging.info(f"Best ARIMA order: {best_order} with RMSE={best_rmse:.2f}")

# Fit final model
final_model = fit_arima_model(series, best_order)

# Save model
os.makedirs('../../../outputs', exist_ok=True)
joblib.dump(final_model, "outputs/arima_model.pkl")
logging.info("Final model saved to outputs/arima_model.pkl")
