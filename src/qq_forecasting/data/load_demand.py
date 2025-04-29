import os
import pandas as pd
from qq_forecasting.utils.preprocessing import add_minutes_from_SP, add_datetime_index

def load_demand_data(folder_path: str, years: range) -> pd.DataFrame:
    """
    Loads, processes and merges national demand data across multiple years.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        years (range): Range of years to include (e.g., range(2019, 2025)).

    Returns:
        pd.DataFrame: Time-indexed DataFrame of national demand.
    """
    all_years = []
    for year in years:
        file_path = os.path.join(folder_path, f'demanddata_{year}.csv')
        df = pd.read_csv(file_path)
        df = add_minutes_from_SP(df, date_col='SETTLEMENT_DATE', period_col='SETTLEMENT_PERIOD', mixed=True)
        df = df[['DATETIME', 'ND']].rename(columns={'ND': 'national_demand'})
        df = add_datetime_index(df, date_col='DATETIME', remove_tz=False)
        all_years.append(df)

    df_all = pd.concat(all_years).sort_index()

    # Remove duplicate timestamps safely
    df_all = df_all[~df_all.index.duplicated(keep='first')]

    # Now enforce 30min frequency
    df_all = df_all.asfreq('30min')

    return df_all
