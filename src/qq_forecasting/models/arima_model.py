import logging
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def fit_arima_model(series: pd.Series, order: tuple) -> ARIMA:
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        logging.info(f"Fitted ARIMA{order} successfully.")
        return model_fit
    except Exception as e:
        logging.error(f"Error fitting ARIMA{order}: {e}")
        raise

def forecast_arima(model_fit: ARIMA, steps: int) -> pd.Series:
    return model_fit.forecast(steps=steps)
