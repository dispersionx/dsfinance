import pandas as pd

from .config import DATA_PROCESSED, TRAIN_RATIO
from .baselines import random_walk, historical_mean
from .har import har_ols_expanding
from .garch import garch11_expanding
from .gjr_garch import gjr_garch_expanding
from .arfima import arfima_on_rv_expanding

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

    # GJR-GARCH(1,1) – Normal
    gjr_res = gjr_garch_expanding(df, start_idx, dist="normal")
    results.append({
        "model": gjr_res["model"],
        "mse": gjr_res["mse"],
        "qlike": gjr_res["qlike"],
    })

    pd.DataFrame({
        "Date": df.loc[start_idx:, "Date"].to_numpy(),
        "y_true": gjr_res["y_true"],
        "y_pred": gjr_res["y_pred"],
    }).to_csv("results/forecasts/gjr_garch_normal.csv", index=False)

    # (Optional) GJR-GARCH(1,1) – Student-t
    gjr_t = gjr_garch_expanding(df, start_idx, dist="t")
    results.append({
        "model": gjr_t["model"],
        "mse": gjr_t["mse"],
        "qlike": gjr_t["qlike"],
    })

    pd.DataFrame({
        "Date": df.loc[start_idx:, "Date"].to_numpy(),
        "y_true": gjr_t["y_true"],
        "y_pred": gjr_t["y_pred"],
    }).to_csv("results/forecasts/gjr_garch_t.csv", index=False)

    # ARFIMA-on-RV (Long memory)
    arf_res = arfima_on_rv_expanding(df, start_idx, K=200)
    results.append({
        "model": arf_res["model"],
        "mse": arf_res["mse"],
        "qlike": arf_res["qlike"],
    })

    pd.DataFrame({
        "Date": df.loc[start_idx:, "Date"].to_numpy(),
        "y_true": arf_res["y_true"],
        "y_pred": arf_res["y_pred"],
    }).to_csv("results/forecasts/arfima_on_rv.csv", index=False)
    
    # Collect & save evaluation
    out = pd.DataFrame(results).sort_values("mse")
    print(out)

    out_path = "results/evaluation/baselines_har_garch_gjr.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()