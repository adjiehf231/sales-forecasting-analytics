# src/explain.py
import shap

def get_shap_explainer(model, X):
    # Hanya numeric
    X_numeric = X.select_dtypes(include=["float64", "int64", "bool"])
    
    explainer = shap.Explainer(model, X_numeric)
    return explainer

def compute_shap_values(explainer, X):
    X_numeric = X.select_dtypes(include=["float64", "int64", "bool"])
    shap_values = explainer(X_numeric)
    return shap_values
