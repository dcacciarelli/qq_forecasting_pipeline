# src/qq_forecasting/models/arima_model.py

import logging
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_arima_model(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    disp: bool = False  # <--- NEW: optional argument
) -> SARIMAX:
    """
    Fit a SARIMA model to a time series.

    Args:
        series (pd.Series): The time series data.
        order (tuple): The (p,d,q) order of the model.
        seasonal_order (tuple): The (P,D,Q,s) seasonal order.
        disp (bool, optional): Whether to display optimizer convergence output. Default is False.

    Returns:
        SARIMAXResultsWrapper: The fitted model.
    """
    try:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=disp)  # <--- Use the function argument
        logging.info(f"Fitted SARIMA{order}x{seasonal_order} successfully.")
        return model_fit
    except Exception as e:
        logging.error(f"Error fitting SARIMA{order}x{seasonal_order}: {e}")
        raise


def forecast_arima(model_fit: SARIMAX, steps: int) -> pd.Series:
    """
    Forecast future values using the fitted SARIMA model.

    Args:
        model_fit (SARIMAXResultsWrapper): The fitted model.
        steps (int): Number of steps to forecast ahead.

    Returns:
        pd.Series: The forecasted values.
    """
    return model_fit.forecast(steps=steps)
