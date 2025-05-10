import pandas as pd
import matplotlib.pyplot as plt
from qq_forecasting.utils import forecast_metrics
import os

# Load actual and predictions
actual = pd.read_csv("data/processed/demand/test.csv").squeeze()
lstm_preds = pd.read_csv("outputs/results/lstm_forecast_demand.csv").squeeze()
transformer_preds = pd.read_csv("outputs/results/transformer_forecast_demand.csv").squeeze()
arima_preds = pd.read_csv("outputs/results/arima_forecast_demand.csv").squeeze()

# Evaluate
metrics = {
    "LSTM": forecast_metrics(actual, lstm_preds),
    "Transformer": forecast_metrics(actual, transformer_preds),
    "ARIMA": forecast_metrics(actual, arima_preds)
}
metrics_df = pd.DataFrame(metrics).T

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual.values, label="Actual", color="black", linewidth=2)
plt.plot(lstm_preds.values, label="LSTM", linestyle="--")
plt.plot(transformer_preds.values, label="Transformer", linestyle=":")
plt.plot(arima_preds.values, label="ARIMA", linestyle="-.")

plt.title("Forecast Comparison: LSTM vs Transformer vs ARIMA")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("outputs/results", exist_ok=True)
plt.savefig("outputs/results/forecast_comparison.png")
plt.show()

print(metrics_df)
