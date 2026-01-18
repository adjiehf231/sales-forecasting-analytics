from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

def train_model(df):
    # Target
    y = df["sales"]

    # Drop non-numeric & ID columns
    drop_cols = [
        "order_id", "ship_date", "ship_mode",
        "customer_id", "customer_name", "segment",
        "country", "city", "state", "region",
        "product_id", "category", "sub-category",
        "product_name"
    ]
    X = df.drop(columns=drop_cols + ["sales"], errors="ignore")
    X = X.select_dtypes(include=["int64", "float64", "bool"])


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    return model, metrics
