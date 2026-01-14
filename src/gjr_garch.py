import numpy as np
import pandas as pd

from .config import EPS
from .metrics import mse, qlike


def gjr_garch_expanding(df: pd.DataFrame, start_idx: int, dist: str = "normal"):
    """
    Expanding-window GJR-GARCH(1,1) on daily returns (LogReturn).
    Forecast object: one-day-ahead conditional variance sigma^2_{t+1|t}
    Compare against Target (= next-day RV proxy).

    In arch package, GJR is implemented via GARCH(p,o,q) with o>0 (asymmetric term).
    Here: p=1, o=1, q=1.
    """
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError("Please install 'arch' first: pip install arch") from e

    r = df["LogReturn"].to_numpy()
    y_true = df.loc[start_idx:, "Target"].to_numpy()
    y_pred = np.zeros_like(y_true)

    for i, t in enumerate(range(start_idx, len(df))):
        r_train = r[:t]
        r_train_scaled = 100.0 * r_train  # numerical stability

        am = arch_model(
            r_train_scaled,
            mean="Zero",
            vol="GARCH",
            p=1,
            o=1,          # <-- this makes it GJR/TGARCH-type asymmetry
            q=1,
            dist=dist,    # "normal" or "t"
            rescale=False
        )
        res = am.fit(disp="off", show_warning=False)

        f = res.forecast(horizon=1, reindex=False)
        var_scaled = float(f.variance.values[-1, 0])
        var = var_scaled / (100.0 ** 2)  # back to original scale

        y_pred[i] = var

    y_pred = np.clip(y_pred, EPS, None)

    return {
        "model": f"GJR-GARCH(1,1)-{dist}",
        "mse": mse(y_true, y_pred),
        "qlike": qlike(y_true, y_pred, EPS),
        "y_true": y_true,
        "y_pred": y_pred,
    }