import numpy as np
import pandas as pd

from .config import EPS, TRAIN_RATIO
from .metrics import mse, qlike

def garch11_expanding(df: pd.DataFrame, start_idx: int, dist: str = "normal"):
    """
    Expanding-window GARCH(1,1) on daily returns (LogReturn).
    Forecast object: next-day conditional variance (sigma^2_{t+1|t})

    We compare sigma^2 forecast to the same Target series (next-day RV proxy).
    """

    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError("Please install 'arch' first: pip install arch") from e

    # Returns series
    r = df["LogReturn"].to_numpy()

    # Target (next-day RV proxy)
    y_true = df.loc[start_idx:, "Target"].to_numpy()
    y_pred = np.zeros_like(y_true)

    for i, t in enumerate(range(start_idx, len(df))):
        r_train = r[:t]  # up to t-1
        # Scale returns to improve numerical stability (common practice)
        r_train_scaled = 100.0 * r_train

        am = arch_model(
            r_train_scaled,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist=dist,   # "normal" or "t"
            rescale=False
        )
        res = am.fit(disp="off", show_warning=False)

        # One-step-ahead forecast of variance for r_t (scaled returns)
        f = res.forecast(horizon=1, reindex=False)
        var_scaled = float(f.variance.values[-1, 0])

        # Convert back to original scale: (100*r)^2 variance -> r^2 variance
        var = var_scaled / (100.0 ** 2)

        y_pred[i] = var

    y_pred = np.clip(y_pred, EPS, None)

    return {
        "model": f"GARCH(1,1)-{dist}",
        "mse": mse(y_true, y_pred),
        "qlike": qlike(y_true, y_pred, EPS),
        "y_true": y_true,
        "y_pred": y_pred,
    }