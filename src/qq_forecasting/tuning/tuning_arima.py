import itertools
import logging
from tqdm import tqdm
from qq_forecasting.models.arima_model import fit_arima_model, forecast_arima
from qq_forecasting.utils.metrics import evaluate_forecast


def tune_arima(train, val, p_values, d_values, q_values, P_values, D_values, Q_values, s, verbose=True):
    """
    Tune SARIMA hyperparameters based on validation set RMSE.

    Args:
        train (pd.Series): Training series.
        val (pd.Series): Validation series.
        p_values (list): List of p values.
        d_values (list): List of d values.
        q_values (list): List of q values.
        P_values (list): List of seasonal P values.
        D_values (list): List of seasonal D values.
        Q_values (list): List of seasonal Q values.
        s (int): Seasonal period (e.g., 48 for daily if half-hourly data).
        verbose (bool): Whether to print/log progress.

    Returns:
        tuple: Best (order, seasonal_order) and best RMSE.
    """
    best_score = float("inf")
    best_order = None
    best_seasonal_order = None

    param_grid = list(itertools.product(p_values, d_values, q_values,
                                        P_values, D_values, Q_values))

    if verbose:
        print(f"Tuning {len(param_grid)} SARIMA configurations...")

    for params in tqdm(param_grid, desc="Tuning SARIMA models", ncols=100):
        p, d, q, P, D, Q = params
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)

        try:
            model = fit_arima_model(train, order=order, seasonal_order=seasonal_order)
            forecast = forecast_arima(model, steps=len(val))
            score = evaluate_forecast(val, forecast)["rmse"]

            logging.info(f"SARIMA{order}x{seasonal_order} RMSE={score:.2f}")

            if score < best_score:
                best_score = score
                best_order = order
                best_seasonal_order = seasonal_order
        except Exception as e:
            logging.warning(f"Failed SARIMA{order}x{seasonal_order}: {e}")
            continue

    if verbose:
        print(f"Best SARIMA configuration found: Order={best_order}, Seasonal Order={best_seasonal_order}, Validation RMSE={best_score:.2f}")

    return best_order, best_seasonal_order, best_score
