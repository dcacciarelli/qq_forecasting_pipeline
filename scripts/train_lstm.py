import os
import yaml
import torch
import joblib
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from qq_forecasting.models.lstm_model import LSTM, train_lstm
from qq_forecasting.utils import create_sliding_windows

# ========== LOAD CONFIG ==========
with open("config/lstm_demand.yaml") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["paths"]["data_path"]
MODEL_SAVE_PATH = config["paths"]["model_path"]

hp = config["training"]
model_cfg = config["model"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD ==========
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).squeeze()
if hp["max_training_samples"] > 0:
    train = train.iloc[-hp["max_training_samples"]:]

# ========== SLIDING WINDOWS ==========
X_train, y_train = create_sliding_windows(train.values, window_size=hp["window_size"])
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=hp["batch_size"], shuffle=False)

# ========== TRAIN ==========
model = LSTM(**model_cfg).to(device)
train_lstm(model, train_loader, num_epochs=hp["num_epochs"], lr=hp["learning_rate"])

# ========== SAVE ==========
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Trained LSTM model saved to {MODEL_SAVE_PATH}")
