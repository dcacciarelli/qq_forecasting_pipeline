from src.qq_forecasting.common.metrics import evaluate_forecast
import logging

def train_validation_split_evaluation(series, model_func, predict_func, model_args={}, forecast_horizon=None, train_ratio=0.8):
    """
    Fast simple train/validation split for time series.

    Args:
        series (pd.Series): Time series data.
        model_func (callable): Function to fit the model, e.g., fit_arima_model().
        predict_func (callable): Function to forecast, e.g., forecast_arima().
        model_args (dict): Arguments to pass to the model fitting function.
        forecast_horizon (int): How many steps ahead to forecast. If None, forecast entire validation set.
        train_ratio (float): Proportion of data to use for training.

    Returns:
        dict: RMSE and MAE.
    """
    n = len(series)
    split_idx = int(train_ratio * n)

    train = series.iloc[:split_idx]
    validation = series.iloc[split_idx:]

    try:
        model = model_func(train, **model_args)

        steps = forecast_horizon if forecast_horizon else len(validation)
        forecast = predict_func(model, steps=steps)

        forecast = forecast[:len(validation)]  # match length

        metrics = evaluate_forecast(validation, forecast)

        logging.info(f"Simple Holdout - RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")

        return metrics
    except Exception as e:
        logging.error(f"Simple holdout evaluation failed: {e}")
        return None

