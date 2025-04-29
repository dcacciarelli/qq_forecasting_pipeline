import itertools
import logging
from qq_forecasting.models.arima_model import fit_arima_model, forecast_arima
from qq_forecasting.utils.metrics import evaluate_forecast

def tune_arima(series, p_values, d_values, q_values, initial_train_size, horizon=48, step=48):
    best_score = float("inf")
    best_cfg = None

    n = len(series)

    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            errors = []
            for i in range(initial_train_size, n - horizon, step):
                train = series.iloc[:i]
                test = series.iloc[i:i+horizon]
                model = fit_arima_model(train, order)
                forecast = forecast_arima(model, steps=horizon)
                metrics = evaluate_forecast(test, forecast)
                errors.append(metrics["rmse"])
            mean_rmse = sum(errors) / len(errors)

            if mean_rmse < best_score:
                best_score = mean_rmse
                best_cfg = order

            logging.info(f"ARIMA{order} - Mean RMSE: {mean_rmse:.2f}")
        except Exception as e:
            logging.warning(f"ARIMA{order} failed: {e}")
            continue

    return best_cfg, best_score
