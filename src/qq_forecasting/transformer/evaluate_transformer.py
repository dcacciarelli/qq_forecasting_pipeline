import yaml
import torch
import joblib
import numpy as np

from qq_forecasting.utils.load_demand import load_demand_data
from qq_forecasting.utils.preprocessing import create_sliding_windows
from qq_forecasting.transformer.transformer_model import TransformerEncoder
from qq_forecasting.utils.plotting import plot_forecast_vs_actual


def main(config_path="config/transformer_config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = TransformerEncoder()
    model.load_state_dict(torch.load(config["paths"]["model_path"]))
    model.eval()

    scaler = joblib.load(config["paths"]["scaler_path"])

    df = load_demand_data(config["data"]["folder_path"], years=config["data"]["years"])
    series = df[config["data"]["column_name"]]
    if config["data"]["max_samples"]:
        series = series[:config["data"]["max_samples"]]

    split_idx = int(len(series) * config["split"]["train_size"])
    test_series = series[split_idx:]

    scaled_test = scaler.transform(test_series.values.reshape(-1, 1)).flatten()
    X_test, y_test = create_sliding_windows(scaled_test, config["split"]["window_size"])

    with torch.no_grad():
        predictions = []
        for x in X_test:
            x = x.unsqueeze(1).transpose(0, 1)  # (seq_len, 1, 1)
            out = model(x)
            predictions.append(out[-1].view(-1).item())  # forces flatten to 1D, then extracts the scalar

    predictions = np.array(predictions)
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

    plot_forecast_vs_actual(y_test_inv, predictions_inv)


if __name__ == "__main__":
    main()
