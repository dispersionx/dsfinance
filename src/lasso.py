import pandas as pd
from sklearn.linear_model import Lasso
from .features import create_features
from .config import FORECAST_DIR, TRAIN_RATIO

def run_lasso(df):
    dates, X, y = create_features(df)
    n_train = int(len(X) * TRAIN_RATIO)
    model = Lasso(alpha=1e-4)
    model.fit(X.iloc[:n_train], y.iloc[:n_train])
    preds = model.predict(X.iloc[n_train:])
    out = pd.DataFrame({
        "Date": dates.iloc[n_train:],
        "LASSO": preds
    })
    out.to_csv(FORECAST_DIR / "lasso.csv", index=False)
