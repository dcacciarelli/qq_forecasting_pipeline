import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from typing import List, Optional, Tuple, Literal, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def merge_datasets(
        folder_path: str,
        years: List[int],
        column: Optional[str] = None,
        filename_pattern: str = "demanddata_{year}.csv"
) -> Union[pd.DataFrame, pd.Series]:
    """
    Load and concatenate CSVs from a folder based on a filename pattern.
    Optionally extract a single column as a Series.

    Args:
        folder_path (str): Path to the folder with CSVs.
        years (List[int]): List of years to load.
        column (str, optional): Name of the column to extract as Series.
        filename_pattern (str): Filename format string with `{year}` placeholder.

    Returns:
        Union[pd.DataFrame, pd.Series]: Concatenated dataframe or series.
    """
    dataframes = []
    for year in years:
        file_path = os.path.join(folder_path, filename_pattern.format(year=year))
        df = pd.read_csv(file_path)
        dataframes.append(df)

    full_df = pd.concat(dataframes, ignore_index=True)

    if column:
        if column not in full_df.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        return full_df[column]

    return full_df


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


def fill_series(series: pd.Series, method: Literal["interpolate", "ffill", "bfill"] = "interpolate") -> pd.Series:
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


def scale_series(series: pd.Series, method: str = "minmax") -> Tuple[np.ndarray, MinMaxScaler]:
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler: {method}")

    series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return series_scaled, scaler


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


def plot_univariate_timeseries(
    series: pd.Series,
    title: str = None,
    xlabel: str = "Date",
    ylabel: str = "Value",
    figsize: tuple = (10, 4),
    dpi: int = 300,
    color: str = 'tab:blue',
    linestyle: str = '-',
    marker: str = '',
    fontsize: int = 12,
    grid: bool = True,
    y_range: tuple = None,
    legend: bool = False
):
    """
    Plot a univariate time series.

    Args:
        series (pd.Series): Time series to plot.
        title (str, optional): Plot title.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        figsize (tuple, optional): Figure size.
        dpi (int, optional): Dots per inch for figure.
        color (str, optional): Line color.
        linestyle (str, optional): Line style ('-', '--', etc.).
        marker (str, optional): Marker style ('' for none, 'o', etc.).
        fontsize (int, optional): Font size for labels.
        grid (bool, optional): Whether to show grid.
        y_range (tuple, optional): (ymin, ymax) range.
        legend (bool, optional): Whether to show legend.

    Returns:
        None
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series.")

    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(series.index, series.values, label=series.name, color=color, linestyle=linestyle, marker=marker, alpha=0.8)

    if title:
        plt.title(title, fontsize=fontsize + 2)
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)
    if y_range:
        plt.ylim(y_range)
    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)
    if legend:
        plt.legend(fontsize=fontsize)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_forecast_vs_actual(actual, forecast):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(actual, label="Actual")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.title("Forecast vs Actual")
    plt.legend()
    plt.show()


def create_sliding_windows(series: pd.Series, window_size: int = 10) -> torch.tensor:
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X.unsqueeze(-1), y.unsqueeze(-1)  # Add feature dimension
