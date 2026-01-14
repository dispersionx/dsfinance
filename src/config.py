from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED = BASE_DIR / "data" / "processed" / "SP500_processed.csv"

TARGET_COL = "Target"
EPS = 1e-12

# HAR regressors (t used to predict Target_t = RV_{t+1})
HAR_FEATURES = ["RV_d", "RV_w", "RV_m"]

TRAIN_RATIO = 0.70


