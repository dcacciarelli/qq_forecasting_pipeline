import yaml
import pandas as pd
import logging
from pipelines.load_demand_data import load_and_merge_demand
from models.arima_model import fit_arima_model, forecast_arima
from utils.plotting import plot_forecast_vs_actual
from utils.metrics import evaluate_forecast

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load config
with open("config/arima_config.yaml", "r") as f:
    config = yaml.safe_load(f)

order = tuple(config["model"]["order"])
horizon = config["data"]["forecast_horizon"]
input_path = config["data"]["input_path"]
resample_freq = config["data"]["resample_freq"]

# Load data
df = pd.read_csv(input_path, index_col="DATETIME", parse_dates=True)
series = df["national_demand"].resample(resample_freq).mean().dropna()

# Train/test split
train = series[:-horizon]
test = series[-horizon:]

# Fit and forecast
model_fit = fit_arima_model(train, order)
forecast = forecast_arima(model_fit, steps=horizon)

# Evaluate
metrics = evaluate_forecast(test, forecast)
logging.info(f"Forecast MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")

# Plot
plot_forecast_vs_actual(test, forecast)

# (Optional) Save model, metrics, plots here
