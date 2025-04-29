# ===============================================================
# Sandbox: Simple LSTM Pipeline (Load, Preprocess, Train, Predict)
# ===============================================================

# ===============================
# Setup Environment
# ===============================
import sys
import os
sys.path.append(os.path.abspath("src"))

# ===============================
# Imports
# ===============================
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from qq_forecasting.data.load_demand import load_demand_data
from qq_forecasting.utils.lstm_preprocessing import create_sliding_windows
from qq_forecasting.models.lstm_model import SimpleLSTM
from qq_forecasting.training.train_lstm import train_lstm
from qq_forecasting.evaluation.evaluate_lstm import predict_lstm
from qq_forecasting.visualization.plotting import plot_forecast_vs_actual

# ===============================
# Load and Preprocess Data
# ===============================
df = load_demand_data("data/raw", years=[2019])
series = df["national_demand"][:1000]  # small slice for faster testing

# Sliding windows
window_size = 48  # one day if half-hourly data
X, y = create_sliding_windows(series, window_size)

# Train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)

# ===============================
# Instantiate and Train Model
# ===============================
model = SimpleLSTM()

train_lstm(model, train_loader, num_epochs=100, lr=0.001)

# ===============================
# Make Predictions
# ===============================
predictions = predict_lstm(model, X_test)

# ===============================
# Plot Predictions vs Actual
# ===============================
plot_forecast_vs_actual(y_test.numpy(), predictions.numpy())
