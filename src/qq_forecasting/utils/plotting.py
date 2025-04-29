import matplotlib.pyplot as plt


# Plot one variable
def plot_univariate_timeseries(df, column, y_range=None, title=None, figsize=(10, 4), dpi=150, color='tab:blue'):

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(df.index, df[column], label=column, color=color, alpha=0.8)

    if y_range:
        plt.ylim(y_range)

    plt.title(title or column)
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_forecast_vs_actual(actual, forecast, title="Forecast vs Actual"):
    # Ensure inputs are pandas Series
    if isinstance(actual, (np.ndarray, torch.Tensor)):
        actual = pd.Series(actual.flatten())
    if isinstance(forecast, (np.ndarray, torch.Tensor)):
        forecast = pd.Series(forecast.flatten())

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(actual, label="Actual")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

