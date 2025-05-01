# src/qq_forecasting/tuning/tune_arima.py

import yaml
import os
from qq_forecasting.utils.load_demand import load_demand_data
from qq_forecasting.utils.splits import train_val_test_split
from qq_forecasting.arima.tuning_arima import tune_arima

def run_arima_tuning(config_path="config/arima_config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load and preprocess data
    df = load_demand_data(config["data"]["folder_path"], years=config["data"]["years"])
    series = df["national_demand"]
    if config["data"]["max_samples"]:
        series = series[:config["data"]["max_samples"]]

    # Split
    train, val, _ = train_val_test_split(series, config["split"]["val_size"], config["split"]["test_size"])

    # Tune
    best_order, best_seasonal_order, best_rmse = tune_arima(
        train, val,
        p_values=config["model"]["p_values"],
        d_values=config["model"]["d_values"],
        q_values=config["model"]["q_values"],
        P_values=config["model"]["P_values"],
        D_values=config["model"]["D_values"],
        Q_values=config["model"]["Q_values"],
        s=config["model"]["seasonality_period"],
    )

    # Save best params
    best_params = {
        "order": list(best_order),
        "seasonal_order": list(best_seasonal_order),
        "seasonality_period": config["model"]["seasonality_period"],
        "validation_rmse": float(best_rmse)
    }

    os.makedirs("outputs/params", exist_ok=True)
    with open("outputs/params/best_sarima_params.yaml", "w") as f:
        yaml.dump(best_params, f)

    print("✅ Best SARIMA parameters saved.")

if __name__ == "__main__":
    run_arima_tuning()
