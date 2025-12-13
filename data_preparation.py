import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path


# 0. Configuration
# since for yfinance, `end` is often end-exclusive: using the next day avoids ambiguity.
ticker = '^GSPC'
start_date = '2010-01-01'
end_date = '2024-01-01'

# Rolling windows for volatility features
W_WEEK = 5
W_MONTH = 22

# Chronological split ratios (used only after features/target are built)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Output folder
OUT_DIR = Path('/Users/dispersion/Documents/dsfinance')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Download S&P 500 daily data
raw = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval='1d',
    auto_adjust=True,
    progress=False,
)

if raw is None or raw.empty:
    raise ValueError('Downloaded data is empty. Please check the ticker and date range.')

# 2. Drop the second column level (Ticker) and keep only Price level
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

# 3. Clean columns keeping only Close and Volume (Open/High/Low are not needed for the current scope.)
data = raw[['Close', 'Volume']].copy()
data = data.dropna().sort_index()

if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)
data.index.name = 'Date'

# 4. Returns and volatility proxy

# Log return: r_t = log(P_t) - log(P_{t-1})
data['LogReturn'] = np.log(data['Close']).diff()

# Volatility proxy (daily): RV_d = r_t^2
# You can switch to abs returns if desired: np.abs(data['LogReturn'])
data['RV_d'] = data['LogReturn'] ** 2

# Multi-scale volatility features
# Weekly/monthly are rolling means of the daily proxy
# (5 and 22 trading days are standard approximations)
data['RV_w'] = data['RV_d'].rolling(W_WEEK).mean()
data['RV_m'] = data['RV_d'].rolling(W_MONTH).mean()

# Optional: stabilize Volume scale for ML (kept as extra feature)
data['LogVolume'] = np.log1p(data['Volume'])


# 5. Forecast target (one-step-ahead)

# Target is next-day daily volatility proxy
# y_{t} = RV_{t+1}
data['Target'] = data['RV_d'].shift(-1)

# Drop rows with missing values created by diff/rolling/shift
model_data = data.dropna(subset=['LogReturn', 'RV_d', 'RV_w', 'RV_m', 'Target']).copy()


# 6. Split chronologically into Train/Val/Test
# No random shuffling; time series data requires chronological splits
n = len(model_data)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train = model_data.iloc[:train_end].copy()
val = model_data.iloc[train_end:val_end].copy()
test = model_data.iloc[val_end:].copy()


# 7. Save CSVs (Date as a column)
processed_path = OUT_DIR / 'SP500_processed.csv'
train_path = OUT_DIR / 'SP500_train.csv'
val_path = OUT_DIR / 'SP500_val.csv'
test_path = OUT_DIR / 'SP500_test.csv'

model_data.reset_index().to_csv(processed_path, index=False)
train.reset_index().to_csv(train_path, index=False)
val.reset_index().to_csv(val_path, index=False)
test.reset_index().to_csv(test_path, index=False)


# 8. Confirmation & basic diagnostics
print('All CSV files saved successfully!')
print('Saved to:', OUT_DIR.resolve())
print(
    f"Processed: {model_data.shape}, Train: {train.shape}, Val: {val.shape}, Test: {test.shape}"
)
print('Actual date range:', model_data.index.min().date(), 'to', model_data.index.max().date())

# Quick Volume sanity check (index volume can be noisy; this flags extreme issues)
zero_vol_share = (model_data['Volume'] == 0).mean()
print(f'Share of zero Volume: {zero_vol_share:.2%}')

print(model_data.reset_index().head())