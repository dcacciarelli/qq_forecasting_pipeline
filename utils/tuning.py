import itertools
import logging
from models.arima_model import fit_arima_model, forecast_arima
from utils.metrics import evaluate_forecast

def tune_arima(train, test, p_values, d_values, q_values):
    best_score = float("inf")
    best_cfg = None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            model = fit_arima_model(train, order)
            forecast = forecast_arima(model, steps=len(test))
            metrics = evaluate_forecast(test, forecast)
            rmse = metrics["rmse"]
            logging.info(f"ARIMA{order} RMSE={rmse:.3f}")

            if rmse < best_score:
                best_score = rmse
                best_cfg = order
        except Exception as e:
            logging.warning(f"ARIMA{order} failed: {e}")
            continue

    return best_cfg, best_score
