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


def plot_forecast_vs_actual(actual, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, label="Actual")
    plt.plot(actual.index, forecast, label="Forecast", linestyle="--")
    plt.title("Forecast vs Actual")
    plt.legend()
    plt.show()


