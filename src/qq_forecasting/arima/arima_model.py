# src/qq_forecasting/models/arima_model.py

import logging
import os
import yaml
import joblib
import pandas as pd
import itertools
import logging
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Optional, Tuple, Literal, Union

from qq_forecasting.utils import forecast_metrics


def fit_arima(
        series: pd.Series,
        order: tuple,
        seasonal_order: tuple,
        disp: bool = False,
        save_path: Optional[str] = None
) -> SARIMAX:
    """
    Fit an ARIMA model to a time series.

    Args:
        series (pd.Series): The time series data.
        order (tuple): The (p,d,q) order of the model.
        seasonal_order (tuple): The (P,D,Q,s) seasonal order.
        disp (bool, optional): Whether to display optimizer convergence output. Default is False.
        save_path (str, optional): If provided, saves the fitted model to this file path.

    Returns:
        SARIMAXResultsWrapper: The fitted model.
    """
    try:
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        trained_model = model.fit(disp=disp)  # <--- Use the function argument
        logging.info(f"Fitted ARIMA{order}x{seasonal_order} successfully.")

        if save_path:
            # Save
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(trained_model, save_path)
            print(f"ARIMA model trained and saved to {save_path}")

        return trained_model

    except Exception as e:
        logging.error(f"Error fitting ARIMA{order}x{seasonal_order}: {e}")
        raise


def forecast_arima(trained_model: SARIMAX, steps: int = 1) -> pd.Series:
    """
    Forecast future values using the fitted ARIMA model.

    Args:
        trained_model (SARIMAXResultsWrapper): The fitted model.
        steps (int): Number of steps to forecast ahead.

    Returns:
        pd.Series: The forecasted values.
    """
    return trained_model.forecast(steps=steps)


def tune_arima(
        train: pd.Series,
        val: pd.Series,
        p_values: list,
        d_values: list,
        q_values: list,
        P_values: list,
        D_values: list,
        Q_values: list,
        s: int,
        verbose: bool = True):
    """
    Tune ARIMA hyperparameters based on validation set RMSE.

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
        tuple: (best_order, best_seasonal_order, best_mse)
    """
    best_score = float("inf")
    best_order = None
    best_seasonal_order = None

    param_grid = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values))

    if verbose:
        print(f"Tuning {len(param_grid)} ARIMA configurations...")

    with tqdm(total=len(param_grid), desc="Tuning ARIMA models", ncols=130) as pbar:
        for params in param_grid:
            p, d, q, P, D, Q = params
            order = (p, d, q)
            seasonal_order = (P, D, Q, s)

            try:
                model = fit_arima(train, order=order, seasonal_order=seasonal_order)
                forecast = forecast_arima(model, steps=len(val))
                score = forecast_metrics(val, forecast)["MSE"]

                logging.info(f"ARIMA{order}x{seasonal_order} MSE={score:.2f}")

                # Progress bar
                pbar.set_postfix({
                    "O": str(order),
                    "S": str(seasonal_order),
                    "MSE": f"{score:.2f}",
                })

                if score < best_score:
                    best_score = score
                    best_order = order
                    best_seasonal_order = seasonal_order

            except Exception as e:
                logging.warning(f"Failed ARIMA{order}x{seasonal_order}: {e}")
                continue

            pbar.update(1)

    if verbose:
        print(f"Best ARIMA configuration: Order={best_order}, Seasonal={best_seasonal_order}, Validation MSE={best_score:.2f}")

    # Save best params
    best_params = {
        "order": list(best_order),
        "seasonal_order": list(best_seasonal_order),
        "seasonality_period": s,
        "validation_rmse": float(best_score)
    }

    os.makedirs("outputs/params", exist_ok=True)
    with open("outputs/params/best_arima_params.yaml", "w") as f:
        yaml.dump(best_params, f)

    print("Best ARIMA parameters saved.")

    return best_order, best_seasonal_order, best_score
