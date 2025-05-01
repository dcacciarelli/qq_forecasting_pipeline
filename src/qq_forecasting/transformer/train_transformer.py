import yaml
import os
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from qq_forecasting.common.load_demand import load_demand_data
from qq_forecasting.common.preprocessing import create_sliding_windows
from qq_forecasting.transformer.transformer_model import TransformerEncoder, train


def main(config_path="config/transformer_config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    df = load_demand_data(config["data"]["folder_path"], years=config["data"]["years"])
    series = df[config["data"]["column_name"]]
    if config["data"]["max_samples"]:
        series = series[:config["data"]["max_samples"]]

    split_idx = int(len(series) * config["split"]["train_size"])
    train_series = series[:split_idx]

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
    X_train, y_train = create_sliding_windows(scaled_train, config["split"]["window_size"])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    model = TransformerEncoder().to("cuda" if torch.cuda.is_available() else "cpu")
    train(
        model,
        train_loader,
        num_epochs=config["training"]["num_epochs"],
        lr=config["training"]["learning_rate"]
    )

    os.makedirs(os.path.dirname(config["paths"]["model_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["paths"]["model_path"])
    joblib.dump(scaler, config["paths"]["scaler_path"])
    print("✅ Transformer model and scaler saved.")


if __name__ == "__main__":
    main()
