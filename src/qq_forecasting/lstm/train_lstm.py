import os
import yaml
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from qq_forecasting.utils.load_demand import load_demand_data
from qq_forecasting.utils.preprocessing import create_sliding_windows
from qq_forecasting.lstm.lstm_model import LSTM, train_lstm


def main(config_path="config/lstm_config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load and preprocess data
    df = load_demand_data(config["data"]["folder_path"], years=config["data"]["years"])
    series = df[config["data"]["column_name"]]
    if config["data"]["max_samples"]:
        series = series[:config["data"]["max_samples"]]

    split_idx = int(len(series) * config["split"]["train_size"])
    train_series = series[:split_idx]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()

    X_train, y_train = create_sliding_windows(train_scaled, config["split"]["window_size"])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # Train model
    model = LSTM(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"]
    )
    train_lstm(model, train_loader, num_epochs=config["training"]["num_epochs"], lr=config["training"]["learning_rate"])

    # Save model and scaler
    os.makedirs(os.path.dirname(config["paths"]["model_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["paths"]["model_path"])
    joblib.dump(scaler, config["paths"]["scaler_path"])
    print(f"✅ LSTM model and scaler saved.")


if __name__ == "__main__":
    main()
