import pandas as pd
from .config import N_LAGS, TARGET_COL

def create_features(df):
    df = df.copy()

    # Use LogReturn column
    RETURN_COL = "LogReturn"

    if RETURN_COL not in df.columns:
        raise ValueError(f"Column '{RETURN_COL}' not found in the dataframe.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in the dataframe.")

    # Create lagged features
    for i in range(1, N_LAGS + 1):
        df[f"ret_lag_{i}"] = df[RETURN_COL].shift(i)
        df[f"target_lag_{i}"] = df[TARGET_COL].shift(i)

    df = df.dropna().reset_index(drop=True)

    # Feature matrix and target
    X = df[[c for c in df.columns if "lag_" in c]]
    y = df[TARGET_COL]

    return df["Date"], X, y
