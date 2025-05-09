import os
import joblib
import pandas as pd
from qq_forecasting.utils import (
    fill_series,
    split_series,
    scale_series,
    plot_univariate_timeseries
)

# ========== CONFIG ==========
FOLDER_PATH = "data/raw"
COLUMN_NAME = "unconstrained_wind_production"
FILL_METHOD = "interpolate"
SCALE_METHOD = "minmax"
SAVE_NAME = "wind_production"
VAL_SIZE = 0
TEST_SIZE = TEST_SIZE = 48*7

# ========== LOAD ==========
data = pd.read_csv(os.path.join(FOLDER_PATH, "balancing_costs_preprocessed_v21.csv"))
series = data[COLUMN_NAME]
plot_univariate_timeseries(series)

# ========== SPLIT ==========
train, _, test = split_series(series, val_size=VAL_SIZE, test_size=TEST_SIZE)
plot_univariate_timeseries(test)

# ========== SCALE USING TRAIN ==========
scaled_train, scaler = scale_series(train, method=SCALE_METHOD)
# scaled_val = scaler.transform(val.values.reshape(-1, 1)).flatten()
scaled_test = scaler.transform(test.values.reshape(-1, 1)).flatten()
plot_univariate_timeseries(pd.Series(scaled_test))

# ========== SAVE ==========
output_dir = f"data/processed/{SAVE_NAME}"
os.makedirs(output_dir, exist_ok=True)

pd.Series(scaled_train).to_csv(os.path.join(output_dir, "train.csv"), index=False)
# pd.Series(scaled_val).to_csv(os.path.join(output_dir, "val.csv"), index=False)
pd.Series(scaled_test).to_csv(os.path.join(output_dir, "test.csv"), index=False)
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
