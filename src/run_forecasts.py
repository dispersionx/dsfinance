import pandas as pd

from .config import DATA_PROCESSED
from .baselines import run_baseline_models
from .lasso import run_lasso
from .random_forest import run_rf
from .xgboost_model import run_xgb
from .mlp import run_mlp
from .diagnostics import evaluate_all_forecasts


def main():
    # Load processed data
    df = pd.read_csv(DATA_PROCESSED)
    df["Date"] = pd.to_datetime(df["Date"])

    # Run forecasting models
    run_baseline_models(df)
    run_lasso(df)
    run_rf(df)
    run_xgb(df)
    run_mlp(df)

    # Run evaluation & plots
    evaluate_all_forecasts()


# Entry point (required for python -m)
if __name__ == "__main__":
    main()
