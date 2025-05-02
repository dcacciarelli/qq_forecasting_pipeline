import os
import torch
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from qq_forecasting.lstm.lstm_model import LSTM, train_lstm
from qq_forecasting.utils import create_sliding_windows

# ========== CONFIG ==========
DATA_PATH = "data/processed/electricity_demand"
MODEL_SAVE_PATH = "outputs/models/lstm_model.pt"
SCALER_PATH = os.path.join(DATA_PATH, "scaler.pkl")

NUM_LAYERS = 2
HIDDEN_SIZE = 64
WINDOW_SIZE = 48
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MAX_TRAINING_SAMPLES = 10_000  # Optional limit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD ==========
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).squeeze()
val = pd.read_csv(os.path.join(DATA_PATH, "val.csv")).squeeze()
scaler = joblib.load(SCALER_PATH)

train_val = pd.concat([train, val], ignore_index=True)[-MAX_TRAINING_SAMPLES:]

# ========== SLIDING WINDOWS ==========
X_train, y_train = create_sliding_windows(train_val.values, window_size=WINDOW_SIZE)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)

# ========== TRAIN ==========
model = LSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
train_lstm(model, train_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

# ========== SAVE ==========
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ LSTM model saved to {MODEL_SAVE_PATH}")
