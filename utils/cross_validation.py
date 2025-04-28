import numpy as np
from models.arima_model import fit_arima_model, forecast_arima
from utils.metrics import evaluate_forecast
import logging

def walk_forward_cv(series, order, initial_train_size, horizon=48, step=48):
    """
    Perform walk-forward cross-validation for ARIMA models.

    Args:
        series (pd.Series): Full time series at settlement period level.
        order (tuple): (p, d, q) ARIMA order.
        initial_train_size (int): Number of data points to use initially for training.
        horizon (int): Number of periods to forecast at each step.
        step (int): Number of periods to move forward each iteration.

    Returns:
        float: Mean RMSE across all steps.
    """
    n = len(series)
    errors = []

    for i in range(initial_train_size, n - horizon, step):
        train = series.iloc[:i]
        test = series.iloc[i:i+horizon]

        try:
            model_fit = fit_arima_model(train, order)
            forecast = forecast_arima(model_fit, steps=horizon)

            metrics = evaluate_forecast(test, forecast)
            rmse = metrics["rmse"]
            errors.append(rmse)

            logging.info(f"Walk {i}: RMSE={rmse:.2f}")
        except Exception as e:
            logging.warning(f"Walk {i}: Failed to fit ARIMA{order} due to {e}")
            continue

    mean_rmse = np.mean(errors)
    return mean_rmse
