import pandas as pd


# Add minutes to date using info from SP columns
def add_minutes_from_SP(df, date_col='SETT_DATE', period_col='SETT_PERIOD', mixed=False):
    if mixed:
        df[date_col] = pd.to_datetime(df[date_col], format='mixed')
    else:
        df[date_col] = pd.to_datetime(df[date_col])
    df['DATETIME'] = df[date_col] + pd.to_timedelta((df[period_col] - 1) * 30, unit='m')
    return df


# Add datetime index to dataset using SETT_DATE and SETT_PERIOD
def add_datetime_index(df, date_col='SETT_DATE', remove_tz=False):
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    if remove_tz:
        df.index = df.index.tz_localize(None)
    return df

