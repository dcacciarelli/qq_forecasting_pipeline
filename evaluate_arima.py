import yaml
import logging
import pandas as pd
import joblib
from pipelines.load_demand_data import load_and_merge_demand
from utils.metrics import evaluate_forecast
from utils.plotting import plot_forecast_vs_actual

# Load config
with open("config/arima_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(filename='outputs/evaluate.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load data
folder_path = config["data"]["folder_path"]
years = range(config["data"]["years"][0], config["data"]["years"][-1]+1)
df = load_and_merge_demand(folder_path=folder_path, years=years)
series = df["national_demand"]

# Load model
model = joblib.load("outputs/arima_model.pkl")

# Forecast
forecast = model.forecast(steps=config["data"]["forecast_horizon"])
test = series[-config["data"]["forecast_horizon"]:]

# Evaluate
metrics = evaluate_forecast(test, forecast)
logging.info(f"Evaluation metrics: {metrics}")

# Save metrics
if config["evaluation"]["save_metrics_csv"]:
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("outputs/evaluation_metrics.csv", index=False)

# Plot
if config["evaluation"]["save_forecast_plot"]:
    plot_forecast_vs_actual(test, forecast)
