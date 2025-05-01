import pandas as pd

def train_val_test_split(series: pd.Series, val_size: int, test_size: int):
    """
    Split a time series into train, validation, and test sets.

    Args:
        series (pd.Series): Complete time series.
        val_size (int): Number of time steps for validation.
        test_size (int): Number of time steps for test.

    Returns:
        tuple: (train, val, test) splits as pandas Series.
    """
    train_end = -val_size - test_size
    val_end = -test_size

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]

    return train, val, test
