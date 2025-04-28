import itertools
import logging
from utils.cross_validation import walk_forward_cv

def tune_arima(series, p_values, d_values, q_values, initial_train_size, horizon=48, step=48):
    """
    Tune ARIMA hyperparameters (p,d,q) using walk-forward cross-validation.

    Args:
        series (pd.Series): Full time series
        p_values, d_values, q_values (list): Candidate values
        initial_train_size (int): Size of initial training set
        horizon (int): Forecast horizon at each step
        step (int): Step forward at each iteration

    Returns:
        best_order (tuple), best_score (float)
    """
    best_score = float("inf")
    best_order = None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            score = walk_forward_cv(series, order, initial_train_size, horizon, step)
            logging.info(f"ARIMA{order} CV RMSE={score:.2f}")

            if score < best_score:
                best_score = score
                best_order = order

        except Exception as e:
            logging.warning(f"ARIMA{order} failed during tuning: {e}")
            continue

    return best_order, best_score
