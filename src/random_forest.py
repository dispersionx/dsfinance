import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .features import create_features
from .config import FORECAST_DIR, TRAIN_RATIO

def run_rf(df):
    dates, X, y = create_features(df)
    n_train = int(len(X) * TRAIN_RATIO)
    model = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    model.fit(X.iloc[:n_train], y.iloc[:n_train])
    preds = model.predict(X.iloc[n_train:])
    out = pd.DataFrame({
        "Date": dates.iloc[n_train:],
        "RF": preds
    })
    out.to_csv(FORECAST_DIR / "random_forest.csv", index=False)
