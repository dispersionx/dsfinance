import numpy as np

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def qlike(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    y_pred = np.maximum(y_pred, eps)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)
