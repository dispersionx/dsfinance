import numpy as np
from .metrics import mse, qlike
from .config import TARGET_COL, EPS

# Baseline models
def random_walk(df, start_idx):
    """
    Forecast: y_{t+1} = RV_d,t  (naive persistence in volatility proxy)
    """
    y_true = df.loc[start_idx:, TARGET_COL].to_numpy()
    y_pred = df.loc[start_idx:, "RV_d"].to_numpy()
    return {"model": "RandomWalk", "mse": mse(y_true, y_pred), "qlike": qlike(y_true, y_pred, EPS)}

def historical_mean(df, start_idx):
    """
    Expanding mean of Target up to t-1 predicts Target at t.
    """
    y_true = df.loc[start_idx:, TARGET_COL].to_numpy()
    y_pred = np.zeros_like(y_true)

    for i, t in enumerate(range(start_idx, len(df))):
        y_pred[i] = df.loc[:t-1, TARGET_COL].mean()

    return {"model": "HistoricalMean", "mse": mse(y_true, y_pred), "qlike": qlike(y_true, y_pred, EPS)}