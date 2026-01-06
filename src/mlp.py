import pandas as pd
from sklearn.neural_network import MLPRegressor
from .features import create_features
from .config import FORECAST_DIR, TRAIN_RATIO

def run_mlp(df):
    dates, X, y = create_features(df)
    n_train = int(len(X) * TRAIN_RATIO)
    model = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=42)
    model.fit(X.iloc[:n_train], y.iloc[:n_train])
    preds = model.predict(X.iloc[n_train:])
    out = pd.DataFrame({
        "Date": dates.iloc[n_train:],
        "MLP": preds
    })
    out.to_csv(FORECAST_DIR / "mlp.csv", index=False)
