import os
import torch
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from qq_forecasting.transformer.transformer_model import (
    TransformerEncoder,
    train_transformer,
    forecast_transformer,
    forecast_transformer_with_truth,
    forecast_multiple_days_autoreg
)

from qq_forecasting.utils import (
    create_sliding_windows,
    inverse_scale,
    forecast_metrics,
    plot_forecast_vs_actual,
)

# ========== CONFIG ==========
DATA_PATH = "data/processed/electricity_demand"
MODEL_SAVE_PATH = "outputs/models/transformer_model.pt"
SCALER_PATH = os.path.join(DATA_PATH, "scaler.pkl")
METRICS_SAVE_PATH = "outputs/results/transformer_metrics.txt"
PLOT_SAVE_PATH = "outputs/results/transformer_forecast.png"

NUM_LAYERS = 1
WINDOW_SIZE = 10
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
MAX_TRAINING_SAMPLES = 1_000  # Optional limit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD ==========
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).squeeze()[:MAX_TRAINING_SAMPLES]
val = pd.read_csv(os.path.join(DATA_PATH, "val.csv")).squeeze()
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv")).squeeze()
scaler = joblib.load(SCALER_PATH)

# ========== SLIDING WINDOWS ==========
X_train, y_train = create_sliding_windows(train.values, window_size=WINDOW_SIZE)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)

# ========== TRAIN ==========
model = TransformerEncoder(num_layers=NUM_LAYERS).to(device)
train_transformer(model, train_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Transformer model saved to {MODEL_SAVE_PATH}")

# ========== LOAD MODEL ==========
loaded_model = TransformerEncoder().to(device)
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

print(f"Loaded model from {MODEL_SAVE_PATH}")

# ========== FORECAST ==========
# predictions = forecast_transformer(
#     model=model,
#     recent_window=np.concatenate([train, val])[-WINDOW_SIZE:],
#     steps_ahead=len(test),
#     device=device
# )

# Assume test_scaled is already scaled and you want to forecast len(test_scaled) steps
predictions = forecast_transformer_with_truth(
    model=loaded_model,
    context_window=np.concatenate([train, val])[-WINDOW_SIZE:],             # the most recent part of train+val
    future_values=test.values.flatten(),      # already scaled true test set
    device=device
)


# ========== EVALUATE ==========
actual_values = inverse_scale(test.values, scaler)
predicted_values = inverse_scale(predictions, scaler)
metrics = forecast_metrics(actual_values, predicted_values, save_path=METRICS_SAVE_PATH, print_scores=True)
plot_forecast_vs_actual(actual_values, predicted_values, save_path=PLOT_SAVE_PATH)

full_series = np.concatenate([val.values, test.values])  # unscaled
WINDOW_SIZE = 48
HORIZON = 48  # 1 day ahead
NUM_DAYS = 7  # rolling forecasts

forecasts = forecast_multiple_days_autoreg(
    model=loaded_model,
    full_series=full_series,
    window_size=WINDOW_SIZE,
    horizon=HORIZON,
    num_days=NUM_DAYS,
    device=device
)

# ========== VISUALISE OR EVALUATE ==========
import matplotlib.pyplot as plt

day = 0
truth = full_series[WINDOW_SIZE + day * HORIZON : WINDOW_SIZE + (day + 1) * HORIZON]

plt.figure(figsize=(10, 5))
plt.plot(truth, label="Actual")
plt.plot(forecasts[day], label="Forecast", linestyle="--")
plt.title(f"Day {day + 1} Forecast vs Actual")
plt.legend()
plt.show()

