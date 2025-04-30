# ===============================================================
# Sandbox: LSTM Pipeline (Load, Scale, Train, Predict, Plot)
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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from qq_forecasting.data.load_demand import load_demand_data
from qq_forecasting.utils.lstm_preprocessing import create_sliding_windows
from qq_forecasting.models.lstm_model import LSTM
from qq_forecasting.training.train_lstm import train_lstm
from qq_forecasting.evaluation.evaluate_lstm import predict_lstm
from qq_forecasting.utils.plotting import plot_forecast_vs_actual, plot_univariate_timeseries

# ===============================
# Load and Preprocess Data
# ===============================
# Load data
df = load_demand_data("data/raw", years=[2019, 2020, 2021, 2022, 2023, 2024])
series = df["ND"][:5_000]  # Up to 10,000 samples

# Split before scaling
split_idx = int(len(series) * 0.8)
train_series = series[:split_idx]
test_series = series[split_idx:]

# Scale using only training data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).flatten()

# Create sliding windows
window_size = 48
X_train, y_train = create_sliding_windows(train_scaled, window_size)
X_test, y_test = create_sliding_windows(test_scaled, window_size)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# ===============================
# Instantiate and Train Model
# ===============================
model = LSTM(input_size=1, hidden_size=64, num_layers=1)

train_lstm(model, train_loader, num_epochs=10, lr=0.01)

# ===============================
# Make Predictions
# ===============================
predictions = predict_lstm()

# Inverse transform predictions and actuals
y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
predictions_inv = scaler.inverse_transform(predictions.numpy().reshape(-1, 1)).flatten()

# ===============================
# Plot Predictions vs Actual
# ===============================
plot_forecast_vs_actual(y_test_inv, predictions_inv)
