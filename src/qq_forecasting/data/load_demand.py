# src/qq_forecasting/data/load_demand.py

import pandas as pd
import os

def load_demand_data(
        folder_path: str,
        years: list,
        col: str = "ND",
        interpolate_missing: bool = True) -> pd.DataFrame:
    """
    Load and optionally interpolate national demand data.

    Args:
        folder_path (str): Path to the folder containing demand CSVs.
        years (list): List of years to load (e.g., [2019, 2020]).
        interpolate_missing (bool): Whether to interpolate missing values.

    Returns:
        pd.DataFrame: Merged demand data with datetime index.
    """
    dfs = []
    for year in years:
        file_path = os.path.join(folder_path, f"demanddata_{year}.csv")
        df_year = pd.read_csv(file_path)
        dfs.append(df_year)

    df = pd.concat(dfs)

    if interpolate_missing:
        df[col] = df[col].interpolate(method='linear')

    return df
