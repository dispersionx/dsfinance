from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "SP500_processed.csv"
FORECAST_DIR = BASE_DIR / "forecasts"
EVAL_DIR = BASE_DIR / "results" / "evaluation"

# Config
TARGET_COL = "Target"
N_LAGS = 5
TRAIN_RATIO = 0.70
