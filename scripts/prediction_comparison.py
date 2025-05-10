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
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(actual.values, label="Actual", color="black", linewidth=2)
plt.plot(lstm_preds.values, label="LSTM", linestyle="--", lw=1.5)
plt.plot(transformer_preds.values, label="Transformer", linestyle=":", lw=1.5)
plt.plot(arima_preds.values, label="ARIMA", linestyle="-.", lw=1.5)
plt.xlabel("Settlement Period", fontsize=16)
plt.ylabel("Load [MW]", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid('y', alpha=.2)
plt.tight_layout()
os.makedirs("outputs/results", exist_ok=True)
plt.savefig("outputs/results/forecast_comparison.png")
plt.show()

print(metrics_df)
