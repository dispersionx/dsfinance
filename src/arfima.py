import numpy as np
import pandas as pd

from .config import EPS
from .metrics import mse, qlike


def fracdiff_weights(d: float, K: int) -> np.ndarray:
    """
    Compute fractional differencing weights w_k for (1-L)^d up to lag K.
    w_0 = 1
    w_k = w_{k-1} * ( (k-1 - d) / k )
    """
    w = np.empty(K + 1, dtype=float)
    w[0] = 1.0
    for k in range(1, K + 1):
        w[k] = w[k - 1] * ((k - 1 - d) / k)
    return w


def fracdiff_series(y: np.ndarray, d: float, K: int) -> np.ndarray:
    """
    x_t = sum_{k=0..K} w_k * y_{t-k}
    Returns array x with NaNs for the first K points.
    """
    w = fracdiff_weights(d, K)
    n = len(y)
    x = np.full(n, np.nan, dtype=float)
    for t in range(K, n):
        x[t] = np.dot(w, y[t - K : t + 1][::-1])  # y_t, y_{t-1}, ...
    return x


def fit_ar_ols(x: np.ndarray, p: int):
    """
    Fit AR(p) by OLS on x (ignoring NaNs).
    x_t = c + sum_{i=1..p} phi_i x_{t-i} + e_t
    Returns (c, phi, sigma2, aic)
    """
    idx = np.where(~np.isnan(x))[0]
    if len(idx) < (p + 5):
        raise ValueError("Not enough non-NaN observations to fit AR model.")

    start = idx[0]
    end = idx[-1]

    # build design matrix
    rows = []
    ys = []
    for t in range(start + p, end + 1):
        if np.any(np.isnan(x[t - p : t + 1])):
            continue
        ys.append(x[t])
        rows.append([1.0] + [x[t - i] for i in range(1, p + 1)])
    Y = np.array(ys)
    X = np.array(rows)

    # OLS
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    sigma2 = float(np.mean(resid**2))

    # AIC for Gaussian AR(p) (approx): n*log(sigma2) + 2k
    n = len(Y)
    k = X.shape[1]
    aic = n * np.log(max(sigma2, EPS)) + 2 * k

    c = float(beta[0])
    phi = beta[1:].astype(float)
    return c, phi, sigma2, aic


def select_d_p(y_log: np.ndarray, K: int, d_grid=None, p_grid=None):
    """
    Select (d, p) on the initial training window via simple AIC on AR(p) fit to x=(1-L)^d y.
    This is intentionally simple/robust and avoids heavy MLE.
    """
    if d_grid is None:
        d_grid = [0.10, 0.20, 0.30, 0.40, 0.50]
    if p_grid is None:
        p_grid = [0, 1, 2, 5]

    best = None
    for d in d_grid:
        x = fracdiff_series(y_log, d=d, K=K)
        for p in p_grid:
            try:
                c, phi, sigma2, aic = fit_ar_ols(x, p=p)
                cand = (aic, d, p)
                if best is None or cand[0] < best[0]:
                    best = cand
            except Exception:
                continue

    if best is None:
        raise RuntimeError("Failed to select (d,p). Try smaller K or simpler grids.")

    _, d_star, p_star = best
    return float(d_star), int(p_star)


def arfima_on_rv_expanding(df: pd.DataFrame, start_idx: int, K: int = 200):
    """
    ARFIMA-on-RV (approx):
      1) y_t = log(RV_d_t + EPS)
      2) x_t = (1-L)^d y_t  (fractional differencing with truncation K)
      3) fit AR(p) on x_t each step (expanding)
      4) forecast x_{t+1}, then recover y_{t+1}:
            x_{t+1} = y_{t+1} + sum_{k=1..K} w_k y_{t+1-k}
        =>  yhat_{t+1} = xhat_{t+1} - sum_{k=1..K} w_k y_{t+1-k}
      5) RVhat_{t+1} = exp(yhat_{t+1})

    Compare RVhat_{t+1} with Target (= RV_{t+1} proxy).
    """
    rv = df["RV_d"].to_numpy()
    y_log = np.log(rv + EPS)

    y_true = df.loc[start_idx:, "Target"].to_numpy()
    y_pred = np.zeros_like(y_true)

    # select d,p once on the initial training window to avoid instability
    y0 = y_log[:start_idx]
    d_star, p_star = select_d_p(y0, K=min(K, max(30, min(200, start_idx // 3))))
    K_star = min(K, max(30, min(200, start_idx // 3)))
    w = fracdiff_weights(d_star, K_star)

    for i, t in enumerate(range(start_idx, len(df))):
        # training sample ends at t-1
        y_tr = y_log[:t]

        x_tr = fracdiff_series(y_tr, d=d_star, K=K_star)

        # fit AR(p) on x_tr
        c, phi, _, _ = fit_ar_ols(x_tr, p=p_star)

        # forecast x_{t} , xhat_t = c + sum phi_i x_{t-i}
        # "t" : next point relative to x_tr end (which is t-1), so we want xhat at index (t-1)+1 = t
        x_lags = [x_tr[-j] for j in range(1, p_star + 1)] if p_star > 0 else []
        xhat_next = c + float(np.dot(phi, np.array(x_lags))) if p_star > 0 else c

        # recover yhat_next using last K_star y's, yhat_{t} = xhat_{t} - sum_{k=1..K} w_k y_{t-k}
        y_past = y_tr[-K_star:]  # y_{t-K}..y_{t-1}
        # need y_{t-1},...,y_{t-K} aligned with w_1..w_K
        y_past_rev = y_past[::-1]
        adj = float(np.dot(w[1:], y_past_rev[:K_star]))
        yhat_log = xhat_next - adj

        rvhat = float(np.exp(yhat_log))
        y_pred[i] = rvhat

    y_pred = np.clip(y_pred, EPS, None)

    return {
        "model": f"ARFIMA-on-RV(d={d_star:.2f},p={p_star},K={K_star})",
        "mse": mse(y_true, y_pred),
        "qlike": qlike(y_true, y_pred, EPS),
        "y_true": y_true,
        "y_pred": y_pred,
        "d": d_star,
        "p": p_star,
        "K": K_star,
    }