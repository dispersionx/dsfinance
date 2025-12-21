import numpy as np
import pandas as pd

from .config import TARGET_COL, HAR_FEATURES, EPS
from .metrics import mse, qlike

def har_ols_expanding(df: pd.DataFrame, start_idx: int):
    """
    Expanding-window HAR-RV:
      Target_t = beta0 + beta1*RV_d,t-1 + beta2*RV_w,t-1 + beta3*RV_m,t-1 + error_t

    In your dataset:
      features are at time t, Target is already RV_{t+1}.
    So for row t: X_t -> predicts Target_t (which equals RV_{t+1}).
    We estimate betas using rows [0..t-1], then predict row t (OOS).
    """

    y_true = df.loc[start_idx:, TARGET_COL].to_numpy()
    y_pred = np.zeros_like(y_true)

    # Pre-build X matrix with intercept
    X_all = df[HAR_FEATURES].to_numpy()
    X_all = np.column_stack([np.ones(len(df)), X_all])  # add intercept
    y_all = df[TARGET_COL].to_numpy()

    for i, t in enumerate(range(start_idx, len(df))):
        X_train = X_all[:t, :]
        y_train = y_all[:t]

        # OLS closed form: beta = (X'X)^(-1) X'y
        # Use lstsq for numerical stability
        beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

        y_hat = X_all[t, :].dot(beta)
        y_pred[i] = y_hat

    # clip predictions for QLIKE stability (same rule for all models)
    y_pred = np.clip(y_pred, EPS, None)

    return {
        "model": "HAR-OLS",
        "mse": mse(y_true, y_pred),
        "qlike": qlike(y_true, y_pred, EPS),
        "y_true": y_true,
        "y_pred": y_pred,
    }