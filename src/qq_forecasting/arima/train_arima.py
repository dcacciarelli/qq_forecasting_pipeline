# src/qq_forecasting/training/train_sarima.py

import yaml
import os
import pandas as pd
import joblib
from qq_forecasting.utils.load_demand import load_demand_data
from qq_forecasting.utils.splits import train_val_test_split
from qq_forecasting.arima.arima_model import fit_arima_model


def train_final_arima(config_path="config/arima_config.yaml", params_path="outputs/params/best_sarima_params.yaml"):
    # Load configs
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(params_path, "r") as f:
        best_params = yaml.safe_load(f)

    # Load data
    df = load_demand_data(config["data"]["folder_path"], years=config["data"]["years"])
    series = df["ND"]
    if config["data"]["max_samples"]:
        series = series[:config["data"]["max_samples"]]

    # Split
    train, val, _ = train_val_test_split(series, config["split"]["val_size"], config["split"]["test_size"])
    train_val = pd.concat([train, val])

    # Fit
    model = fit_arima_model(
        train_val,
        order=tuple(best_params["order"]),
        seasonal_order=tuple(best_params["seasonal_order"]),
        disp=True
    )

    # Save
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/sarima_model.pkl")
    print("✅ SARIMA model trained and saved to outputs/models/sarima_model.pkl")

if __name__ == "__main__":
    train_final_arima()
