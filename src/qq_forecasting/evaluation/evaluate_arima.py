import joblib
from qq_forecasting.utils.metrics import evaluate_forecast
from qq_forecasting.visualization.plotting import plot_forecast_vs_actual

def evaluate_arima_model(model_path, test_series, steps, plot=True):
    model = joblib.load(model_path)
    forecast = model.forecast(steps=steps)
    metrics = evaluate_forecast(test_series, forecast)

    if plot:
        plot_forecast_vs_actual(test_series, forecast)

    return metrics
