from pipelines.load_demand_data import load_and_merge_demand
from utils.cross_validation import walk_forward_cv

df = load_and_merge_demand(folder_path="data/raw", years=range(2019, 2025))
print(df.head())


series = df["national_demand"]
initial_train_size = 48 * 365  # 1 year of half-hourly periods
order = (5, 1, 0)

cv_rmse = walk_forward_cv(series, order, initial_train_size=initial_train_size, horizon=48, step=48)
print(f"Walk-forward RMSE: {cv_rmse:.2f}")
