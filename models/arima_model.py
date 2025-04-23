# models/arima_model.py

from statsmodels.tsa.arima.model import ARIMA

def fit_arima(train_series, order=(5, 1, 0)):
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=10):
    return model_fit.forecast(steps=steps)
