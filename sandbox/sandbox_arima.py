# ===============================================================
# Sandbox: SARIMA Pipeline (Split, Tune, Train, Evaluate, Plot)
# ===============================================================

# ===============================
# Setup Environment
# ===============================
import sys
import os

# Add src/ to path
sys.path.append(os.path.abspath("src"))

# ===============================
# Imports
# ===============================
import joblib
import pandas as pd
from qq_forecasting.data.load_demand import load_demand_data
from qq_forecasting.data.splits import train_val_test_split
from qq_forecasting.models.arima_model import fit_arima_model, forecast_arima
from qq_forecasting.tuning.tuning_arima import tune_arima
from qq_forecasting.training.train_arima import train_final_arima
from qq_forecasting.evaluation.evaluate_arima import evaluate_arima_model
from qq_forecasting.visualization.plotting import plot_univariate_timeseries, plot_forecast_vs_actual

# ===============================
# Load Data
# ===============================
df = load_demand_data("data/raw", years=[2019, 2020, 2021, 2022, 2023, 2024])
series = df["national_demand"][:1000]  # limit for faster experimentation
plot_univariate_timeseries(series, xlabel="Date", ylabel="Demand [MW]")

# ===============================
# Train/Validation/Test Split
# ===============================
train, val, test = train_val_test_split(series, val_size=48*7, test_size=48*7)  # 1 week for val/test

# ===============================
# Tune SARIMA Hyperparameters
# ===============================
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]

s = 48

best_order, best_seasonal_order, best_rmse = tune_arima(
    train, val,
    p_values, d_values, q_values,
    P_values, D_values, Q_values,
    s
)


# ===============================
# Train Final Model (train + val)
# ===============================
print("Training final SARIMA model...")
train_val = pd.concat([train, val])
os.makedirs('outputs/models', exist_ok=True)
train_final_arima(train_val, order=best_order, seasonal_order=best_seasonal_order, save_path="outputs/models/sarima_model.pkl")

# ===============================
# Load Model and Evaluate on Test Set
# ===============================
print("Evaluating on test set...")
metrics = evaluate_arima_model(
    model_path="outputs/models/sarima_model.pkl",
    test_series=test,
    steps=len(test),
    plot=True
)
print(f"Test Metrics: {metrics}")
