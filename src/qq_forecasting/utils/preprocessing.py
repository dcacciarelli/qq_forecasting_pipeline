import numpy as np
import pandas as pd
import torch
import os
from typing import List, Optional, Tuple, Literal, Union
from sklearn.preprocessing import MinMaxScaler


def load_csv_data(folder_path: str, years: List[int], filename_pattern: str = "demanddata_{year}.csv") -> pd.DataFrame:
    """
    Load and concatenate CSVs from a folder based on a filename pattern.

    Args:
        folder_path (str): Path to the folder with CSVs.
        years (List[int]): List of years to load.
        filename_pattern (str): Filename format string with `{year}` placeholder.

    Returns:
        pd.DataFrame: Concatenated dataframe.
    """
    dataframes = []
    for year in years:
        file_path = os.path.join(folder_path, filename_pattern.format(year=year))
        df = pd.read_csv(file_path)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def split_series(series: pd.Series, val_size: int = 0, test_size: int = 0) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
    Split a time series into train, (optional) validation, and test.

    Args:
        series (pd.Series): The full time series.
        val_size (int): Size of the validation set (from end).
        test_size (int): Size of the test set (from end).

    Returns:
        Tuple[pd.Series, pd.Series | None, pd.Series | None]: (train, val, test)
    """
    if val_size + test_size >= len(series):
        raise ValueError("Validation + test size exceeds series length.")

    train_end = -val_size - test_size if val_size > 0 else -test_size
    val_end = -test_size if test_size > 0 else None

    train = series[:train_end]
    val = series[train_end:val_end] if val_size > 0 else None
    test = series[val_end:] if test_size > 0 else None

    return train, val, test


def preprocess_series(series: pd.Series, method: Literal["interpolate", "ffill", "bfill"] = "interpolate") -> pd.Series:
    """
    Preprocess a time series by filling missing values.

    Args:
        series (pd.Series): Input series.
        method (str): Method to fill missing values.

    Returns:
        pd.Series: Preprocessed series.
    """
    if method == "interpolate":
        return series.interpolate(method='linear')
    elif method == "ffill":
        return series.ffill()
    elif method == "bfill":
        return series.bfill()
    else:
        raise ValueError(f"Unsupported fill method: {method}")


def scale_series(series: pd.Series) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scale a series to [0, 1] and return the scaled values and the scaler.

    Args:
        series (pd.Series): Series to scale.

    Returns:
        Tuple[np.ndarray, MinMaxScaler]: (scaled values, fitted scaler)
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return scaled, scaler


def inverse_scale(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Inverse transform the scaled data.

    Args:
        scaled_data (np.ndarray): Scaled array.
        scaler (MinMaxScaler): Fitted scaler.

    Returns:
        np.ndarray: Original values.
    """
    return scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()


def create_sliding_windows(series: pd.Series, window_size: int = 10) -> torch.tensor:
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X.unsqueeze(-1), y.unsqueeze(-1)  # Add feature dimension
