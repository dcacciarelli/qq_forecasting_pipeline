# src/qq_forecasting/evaluation/evaluate_sarima.py

import yaml
import os
import joblib
from qq_forecasting.common.load_demand import load_demand_data
from qq_forecasting.common.splits import train_val_test_split
from qq_forecasting.common.metrics import evaluate_forecast
from qq_forecasting.common.plotting import plot_forecast_vs_actual

def evaluate_arima(config_path="config/arima_config.yaml", model_path="outputs/models/sarima_model.pkl"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    df = load_demand_data(config["data"]["folder_path"], years=config["data"]["years"])
    series = df["national_demand"]
    if config["data"]["max_samples"]:
        series = series[:config["data"]["max_samples"]]

    # Split
    _, _, test = train_val_test_split(series, config["split"]["val_size"], config["split"]["test_size"])

    # Load model
    model = joblib.load(model_path)

    # Forecast
    forecast = model.forecast(steps=len(test))

    # Evaluate
    metrics = evaluate_forecast(test, forecast)

    # Save metrics
    os.makedirs("outputs/metrics", exist_ok=True)
    metrics_path = "outputs/metrics/test_metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)

    print(f"✅ Test metrics saved to {metrics_path}")

    # Plot
    plot_forecast_vs_actual(test, forecast)

    return metrics

if __name__ == "__main__":
    evaluate_arima()
