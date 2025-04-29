import joblib
from qq_forecasting.models.arima_model import fit_arima_model

def train_final_arima(train_val_series, order, seasonal_order, save_path):
    model = fit_arima_model(train_val_series, order, seasonal_order)
    joblib.dump(model, save_path)
