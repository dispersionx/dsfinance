import pandas as pd
from .config import FORECAST_DIR, TRAIN_RATIO, TARGET_COL

def run_baseline_models(df):
    n_train = int(len(df) * TRAIN_RATIO)
    hist_mean = df[TARGET_COL].iloc[:n_train].mean()
    out = pd.DataFrame({
        "Date": df["Date"].iloc[n_train:],
        "HistMean": hist_mean
    })
    out.to_csv(FORECAST_DIR / "baselines.csv", index=False)
