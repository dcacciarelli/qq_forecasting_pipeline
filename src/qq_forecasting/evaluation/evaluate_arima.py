import joblib
from qq_forecasting.visualization.plotting import plot_forecast_vs_actual

def evaluate_arima(model_path, test_series, steps):
    model = joblib.load(model_path)
    forecast = model.forecast(steps=steps)
    plot_forecast_vs_actual(test_series, forecast)
    return forecast
