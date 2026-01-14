from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import BASE_DIR, EPS
from .metrics import mse, qlike


def load_fcst(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # numerical safety for QLIKE
    df["y_pred"] = np.clip(df["y_pred"].to_numpy(), EPS, None)
    return df


def eval_slices(df: pd.DataFrame, crisis_start: str, crisis_end: str) -> pd.DataFrame:
    cs = pd.to_datetime(crisis_start)
    ce = pd.to_datetime(crisis_end)

    mask_crisis = (df["Date"] >= cs) & (df["Date"] <= ce)
    mask_calm = ~mask_crisis

    out = []

    for label, mask in [("overall", slice(None)), ("crisis", mask_crisis), ("calm", mask_calm)]:
        d = df.loc[mask] if label != "overall" else df
        y = d["y_true"].to_numpy()
        f = d["y_pred"].to_numpy()
        out.append({
            "segment": label,
            "n": len(d),
            "mse": mse(y, f),
            "qlike": qlike(y, f, EPS),
        })

    return pd.DataFrame(out)


def main():
    # Crisis window (fixed)：COVID shock window
    CRISIS_START = "2020-02-15"
    CRISIS_END = "2020-06-30"

    forecasts_dir = BASE_DIR / "results" / "forecasts"
    plots_dir = BASE_DIR / "results" / "plots"
    eval_dir = BASE_DIR / "results" / "evaluation"
    plots_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)


    # Load model forecasts
    model_files = {
        "HAR-OLS": forecasts_dir / "har_ols.csv",
        "GARCH(1,1)-normal": forecasts_dir / "garch11_normal.csv",
        "GARCH(1,1)-t": forecasts_dir / "garch11_t.csv",
        "GJR-GARCH(1,1)-normal": forecasts_dir / "gjr_garch_normal.csv",
        "GJR-GARCH(1,1)-t": forecasts_dir / "gjr_garch_t.csv",
    }

    fcsts = {}
    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing forecast file: {path}")
        fcsts[name] = load_fcst(path)

    # Crisis vs Calm evaluation table
    rows = []
    for name, df in fcsts.items():
        seg = eval_slices(df, CRISIS_START, CRISIS_END)
        seg.insert(0, "model", name)
        seg["crisis_start"] = CRISIS_START
        seg["crisis_end"] = CRISIS_END
        rows.append(seg)

    out = pd.concat(rows, ignore_index=True)

    seg_order = pd.Categorical(out["segment"], ["overall", "crisis", "calm"], ordered=True)
    out = out.assign(segment=seg_order).sort_values(["segment", "mse"]).reset_index(drop=True)

    out_path = eval_dir / "crisis_vs_calm.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Forecast vs Realized plot (full OOS)
    master = fcsts["HAR-OLS"][["Date", "y_true"]].copy()
    master = master.rename(columns={"y_true": "Realized"})

    plot_df = master.copy()
    # add each model forecast (aligned by Date)
    for name, df in fcsts.items():
        plot_df = plot_df.merge(df[["Date", "y_pred"]].rename(columns={"y_pred": name}), on="Date", how="inner")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(plot_df["Date"], plot_df["Realized"], label="Realized")

    for name in model_files.keys():
        plt.plot(plot_df["Date"], plot_df[name], label=name, alpha=0.9)

    plt.title("Forecast vs Realized Volatility Proxy (Out-of-Sample)")
    plt.xlabel("Date")
    plt.ylabel("Volatility Proxy")
    plt.legend()
    plt.tight_layout()

    fig_path = plots_dir / "forecast_vs_realized.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()