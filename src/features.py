import pandas as pd

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['week'] = df['order_date'].dt.isocalendar().week
    return df


def create_lag_features(df: pd.DataFrame, lags=[1, 7, 30]) -> pd.DataFrame:
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, windows=[7, 14, 30]) -> pd.DataFrame:
    for w in windows:
        df[f'sales_roll_mean_{w}'] = df['sales'].rolling(w).mean()
        df[f'sales_roll_std_{w}'] = df['sales'].rolling(w).std()
    return df
