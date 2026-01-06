📈 One-Day-Ahead Volatility Forecasting Using Machine Learning
Overview

This project forecasts one-day-ahead volatility in equity markets using machine learning (ML) models.

Accurate volatility forecasting is essential for:

Risk Management – e.g., Value-at-Risk (VaR), Expected Shortfall

Derivative Pricing – volatility is a key input for options

Portfolio Optimization – adjust allocations dynamically during periods of market stress

The dataset used is S&P 500 daily data, preprocessed to include realized volatilities at multiple horizons (RV_d, RV_w, RV_m) and daily log returns (LogReturn). The models predict the next-day volatility, stored in the Target column.

📂 Project Structure
dsfinance/
│
├── data/
│   └── processed/
│       └── SP500_processed.csv
│
├── src/
│   ├── baselines.py
│   ├── lasso.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── mlp.py
│   ├── diagnostics.py
│   ├── config.py
│   └── run_forecasts.py
│
├── forecasts/           # Raw model forecasts
│   ├── baselines.csv
│   ├── lasso.csv
│   ├── random_forest.csv
│   ├── xgboost.csv
│   └── mlp.csv
│
├── results/
│   ├── evaluation/      # CSV comparisons between models
│   ├── forecasts/       # Aligned forecasts with true values
│   └── plots/           # Visual comparisons
│
└── README.md

🤖 Machine Learning Models
1️⃣ Baseline Model

Description: Historical mean-based forecast

Purpose: Reference point for evaluating ML models

Output: forecasts/baselines.csv

2️⃣ Lasso Regression

Description: Linear regression with L1 regularization

Strength: Automatically selects the most relevant features

Input Features: Lagged realized volatilities (RV_d, RV_w, RV_m) + past returns

Output: forecasts/lasso.csv

3️⃣ Random Forest

Description: Ensemble tree-based model

Strength: Captures nonlinear relationships and is robust to overfitting

Output: forecasts/random_forest.csv

4️⃣ XGBoost

Description: Gradient boosting framework

Strength: High predictive accuracy and handles complex nonlinearities

Output: forecasts/xgboost.csv

5️⃣ Multi-Layer Perceptron (MLP)

Description: Feedforward neural network

Strength: Captures nonlinear dependencies in time-series features

Output: forecasts/mlp.csv

🔑 Features
Feature	Description
RV_d	Daily realized volatility
RV_w	Weekly realized volatility
RV_m	Monthly realized volatility
LogReturn	Daily log return
Lagged features	Past values to capture temporal dependencies
Target	Next-day realized volatility (prediction target)
📊 Evaluation

Metrics:

MSE – Mean Squared Error

QLIKE – Quasi-likelihood loss

Outputs:

results/forecasts/ → aligned forecasts with Date, y_true, y_pred

results/evaluation/ → pairwise metrics comparisons (MSE & QLIKE)

results/plots/ → baseline vs all ML models

Plot Example:
baseline_vs_all_models.png shows baseline, all ML model predictions, and true volatility.

Terminal Output: Metrics for each model are printed for quick reference.

⚡ How to Run

Install dependencies:

pip install -r requirements.txt


Run all models and evaluations:

python -m src.run_forecasts


Check Outputs:

Forecast CSVs → results/forecasts/

Evaluation metrics → results/evaluation/

Plots → results/plots/

📌 Notes

All paths and configurations are defined in src/config.py

Forecasts are aligned with the true volatility for accurate evaluation

A single combined plot allows visual comparison of baseline vs ML models