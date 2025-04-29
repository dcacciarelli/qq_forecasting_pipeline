import matplotlib.pyplot as plt


# Plot one variable
# src/qq_forecasting/utils/plotting.py

import matplotlib.pyplot as plt
import pandas as pd

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
    legend: bool = True
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
    plt.plot(actual.index, actual, label="Actual")
    plt.plot(actual.index, forecast, label="Forecast", linestyle="--")
    plt.title("Forecast vs Actual")
    plt.legend()
    plt.show()


