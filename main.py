from pipelines.load_demand_data import load_and_merge_demand
from utils.plotting import plot_univariate_timeseries

folder_path = "data/raw"
years = range(2019, 2025)

df_demand = load_and_merge_demand(folder_path, years)

plot_univariate_timeseries(df_demand, 'national_demand')

df_demand.to_csv("data/processed/national_demand.csv")
