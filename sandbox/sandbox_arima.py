# ===============================================================
# Sandbox: Testing SARIMA Model (Fit, Forecast, Save, Load, Plot)
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
from qq_forecasting.data.load_demand import load_demand_data
from qq_forecasting.models.arima_model import fit_arima_model, forecast_arima
from qq_forecasting.visualization.plotting import plot_univariate_timeseries, plot_forecast_vs_actual

# ===============================
# Load Data
# ===============================
df = load_demand_data("data/raw", years=[2019, 2020, 2021, 2022, 2023, 2024])
series = df["national_demand"][:1000]
plot_univariate_timeseries(series, xlabel="Date", ylabel="Demand [MW]")

# ===============================
# Fit SARIMA Model
# ===============================
model = fit_arima_model(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48), disp=True)

# ===============================
# Forecast Next Periods
# ===============================
horizon = 48  # Forecast next day (48 half-hours)
forecast = forecast_arima(model, steps=horizon)

# ===============================
# Plot Forecast vs Actual
# ===============================
actual = series.iloc[-horizon:]
plot_forecast_vs_actual(actual, forecast)

# ===============================
# Save Model
# ===============================
os.makedirs('outputs/models', exist_ok=True)
joblib.dump(model, "outputs/models/sarima_model.pkl")

# ===============================
# Load Model
# ===============================
loaded_model = joblib.load("outputs/models/sarima_model.pkl")

# ===============================
# Forecast Again with Loaded Model (Optional)
# ===============================
forecast_loaded = forecast_arima(loaded_model, steps=horizon)
plot_forecast_vs_actual(actual, forecast_loaded)
