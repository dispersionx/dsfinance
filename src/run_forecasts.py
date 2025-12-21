import pandas as pd

from .config import DATA_PROCESSED, TRAIN_RATIO
from .baselines import random_walk, historical_mean
from .har import har_ols_expanding
from .garch import garch11_expanding

def main():
    df = pd.read_csv(DATA_PROCESSED)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Determine out-of-sample start index (expanding window)
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
    har_fcst = pd.DataFrame({
        "Date": df.loc[start_idx:, "Date"].to_numpy(),
        "y_true": har_res["y_true"],
        "y_pred": har_res["y_pred"],
    })
    har_fcst.to_csv("results/forecasts/har_ols.csv", index=False)

    # GARCH(1,1) – Normal
    garch_res = garch11_expanding(df, start_idx, dist="normal")
    results.append({
        "model": garch_res["model"],
        "mse": garch_res["mse"],
        "qlike": garch_res["qlike"],
    })

    garch_fcst = pd.DataFrame({
        "Date": df.loc[start_idx:, "Date"].to_numpy(),
        "y_true": garch_res["y_true"],
        "y_pred": garch_res["y_pred"],
    })
    garch_fcst.to_csv("results/forecasts/garch11_normal.csv", index=False)

    # GARCH(1,1) – Student-t
    garch_t = garch11_expanding(df, start_idx, dist="t")
    results.append({
        "model": garch_t["model"],
        "mse": garch_t["mse"],
        "qlike": garch_t["qlike"],
    })

    garch_t_fcst = pd.DataFrame({
        "Date": df.loc[start_idx:, "Date"].to_numpy(),
        "y_true": garch_t["y_true"],
        "y_pred": garch_t["y_pred"],
    })
    garch_t_fcst.to_csv("results/forecasts/garch11_t.csv", index=False)

    # Collect & save evaluation
    out = pd.DataFrame(results).sort_values("mse")
    print(out)

    out_path = "results/evaluation/baselines_har_garch.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()