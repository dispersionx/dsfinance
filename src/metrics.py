import numpy as np

# Mean Squared Error
def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

# Quasi-Likelihood Loss
def qlike(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f = np.clip(y_pred, eps, None)
    ratio = y_true / f
    return np.mean(ratio - np.log(ratio) - 1.0)



