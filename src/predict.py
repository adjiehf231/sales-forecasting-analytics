import pandas as pd

def prepare_features_for_prediction(df, trained_features):
    # Drop non-numeric
    X = df.select_dtypes(include=["int64", "float64", "bool"])
    # Reorder columns sesuai X_train
    X = X[trained_features]
    return X
