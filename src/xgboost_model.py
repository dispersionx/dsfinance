import pandas as pd
import xgboost as xgb
from .features import create_features
from .config import FORECAST_DIR, TRAIN_RATIO

def run_xgb(df):
    dates, X, y = create_features(df)
    n_train = int(len(X) * TRAIN_RATIO)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             objective="reg:squarederror", random_state=42)
    model.fit(X.iloc[:n_train], y.iloc[:n_train])
    preds = model.predict(X.iloc[n_train:])
    out = pd.DataFrame({
        "Date": dates.iloc[n_train:],
        "XGBoost": preds
    })
    out.to_csv(FORECAST_DIR / "xgboost.csv", index=False)
