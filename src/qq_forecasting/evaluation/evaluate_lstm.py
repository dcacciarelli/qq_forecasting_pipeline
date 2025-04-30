import yaml
import torch
import joblib
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from qq_forecasting.data.load_demand import load_demand_data
from qq_forecasting.utils.lstm_preprocessing import create_sliding_windows
from qq_forecasting.models.lstm_model import LSTM
from qq_forecasting.utils.plotting import plot_forecast_vs_actual

def predict_lstm(config_path="config/lstm_config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model and scaler
    model = LSTM(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"]
    )
    model.load_state_dict(torch.load(config["paths"]["model_path"]))
    model.eval()

    scaler = joblib.load(config["paths"]["scaler_path"])

    # Load and preprocess test data
    df = load_demand_data(config["data"]["folder_path"], years=config["data"]["years"])
    series = df[config["data"]["column_name"]]
    if config["data"]["max_samples"]:
        series = series[:config["data"]["max_samples"]]

    split_idx = int(len(series) * config["split"]["train_size"])
    test_series = series[split_idx:]

    test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).flatten()
    X_test, y_test = create_sliding_windows(test_scaled, config["split"]["window_size"])

    # Predict
    with torch.no_grad():
        predictions = model(X_test).squeeze()

    # Inverse scale
    y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    predictions_inv = scaler.inverse_transform(predictions.numpy().reshape(-1, 1)).flatten()

    # Plot
    plot_forecast_vs_actual(y_test_inv, predictions_inv)


if __name__ == "__main__":
    predict_lstm()
