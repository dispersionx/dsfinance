import pandas as pd

from .config import DATA_PROCESSED, TRAIN_RATIO
from .baselines import random_walk, historical_mean

def main():
    df = pd.read_csv(DATA_PROCESSED)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    n = len(df)
    start_idx = int(n * TRAIN_RATIO)

    results = [
        random_walk(df, start_idx),
        historical_mean(df, start_idx),
    ]

    out = pd.DataFrame(results).sort_values("mse")
    print(out)

    # save
    out_path = "results/evaluation/baselines.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()