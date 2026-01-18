import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['order_date'] = pd.to_datetime(
        df['order_date'],
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )

    df = df.dropna(subset=['order_date'])
    df = df.sort_values('order_date')

    df['sales'] = df['sales'].astype(float)

    return df
