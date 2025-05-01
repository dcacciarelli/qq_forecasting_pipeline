import os
import joblib
import pandas as pd
from qq_forecasting.utils import (
    merge_datasets,
    fill_series,
    split_series,
    scale_series,
    plot_univariate_timeseries
)

# ========== CONFIG ==========
FOLDER_PATH = "data/raw"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
COLUMN_NAME = "ND"
MAX_TRAINING_SAMPLES = 1000  # None if whole data
FILL_METHOD = "interpolate"
SCALE_METHOD = "minmax"
SAVE_NAME = "electricity_demand"
VAL_SIZE = 1000
TEST_SIZE = 1000

# ========== LOAD ==========
series = merge_datasets(FOLDER_PATH, YEARS, column=COLUMN_NAME)
plot_univariate_timeseries(series)

# ========== FILL MISSING VALUES ==========
series_filled = fill_series(series, method=FILL_METHOD)
plot_univariate_timeseries(series_filled)

# ========== SPLIT ==========
train, val, test = split_series(series_filled, val_size=VAL_SIZE, test_size=TEST_SIZE)
train = train[:MAX_TRAINING_SAMPLES]

# ========== SCALE USING TRAIN ==========
scaled_train, scaler = scale_series(train, method=SCALE_METHOD)
scaled_val = scaler.transform(val.values.reshape(-1, 1)).flatten()
scaled_test = scaler.transform(test.values.reshape(-1, 1)).flatten()

# ========== SAVE ==========
output_dir = f"data/processed/{SAVE_NAME}"
os.makedirs(output_dir, exist_ok=True)

pd.Series(scaled_train).to_csv(os.path.join(output_dir, "train.csv"), index=False)
pd.Series(scaled_val).to_csv(os.path.join(output_dir, "val.csv"), index=False)
pd.Series(scaled_test).to_csv(os.path.join(output_dir, "test.csv"), index=False)
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
