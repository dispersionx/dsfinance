import pandas as pd

from .config import DATA_PROCESSED, TRAIN_RATIO
from .baselines import random_walk, historical_mean
from .har import har_ols_expanding

def main():
    df = pd.read_csv(DATA_PROCESSED)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Determine OOS start index
    n = len(df)
    start_idx = int(n * TRAIN_RATIO)

    results = []
    
    # Baseline models
    rw_res = random_walk(df, start_idx)
    results.append(rw_res)

    hm_res = historical_mean(df, start_idx)
    results.append(hm_res)
    
    # HAR-OLS
    har_res = har_ols_expanding(df, start_idx)
    results.append({
        "model": har_res["model"],
        "mse": har_res["mse"],
        "qlike": har_res["qlike"],
    })

    # save HAR forecasts for plots
    fcst = pd.DataFrame({
        "Date": df.loc[start_idx:, "Date"].to_numpy(),
        "y_true": har_res["y_true"],
        "y_pred": har_res["y_pred"],
    })
    fcst.to_csv("results/forecasts/har_ols.csv", index=False)
    
    out = pd.DataFrame(results).sort_values("mse")
    print(out)

    out_path = "results/evaluation/baselines_and_har.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()