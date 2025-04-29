import pandas as pd
from src.qq_forecasting.models.arima_model import fit_arima_model, forecast_arima

def test_arima_fit_and_forecast():
    series = pd.Series([10, 12, 13, 15, 18, 20, 23])
    model = fit_arima_model(series, order=(1,1,0))
    forecast = forecast_arima(model, steps=2)
    assert len(forecast) == 2
