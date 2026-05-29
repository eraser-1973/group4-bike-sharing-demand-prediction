# AI3013 Bike-Sharing Demand Prediction - From-Scratch Code

This package is designed to match the course requirement that the selected machine-learning models must be implemented **from scratch** using only basic Python libraries.

## What this code does

- Loads the UCI Bike Sharing hourly dataset (`hour.csv`).
- Builds leakage-aware temporal features:
  - hour/week/month cyclic features,
  - weather features,
  - `cnt_lag_1`, `cnt_lag_24`, rolling mean and rolling standard deviation.
- Implements four models from scratch:
  1. Ridge Linear Regression trained by gradient descent,
  2. KNN Regression,
  3. CART Regression Tree,
  4. Gradient Boosting Regression Trees.
- Uses expanding-window time-series cross-validation.
- Evaluates regression performance by RMSE, MAE, R2 and MAPE.
- Converts regression predictions into high-demand warnings and reports accuracy, precision, recall and F1.
- Saves report-ready CSV tables and PNG figures.

## Dependencies

```bash
pip install -r requirements.txt
```

Only these libraries are used:

- NumPy
- Pandas
- Matplotlib

No scikit-learn, LightGBM, TensorFlow, Keras or PyTorch is used.

## How to run

Option A: automatic dataset download:

```bash
python run_project.py
```

Option B: manual dataset path:

```bash
python run_project.py --data-path data/hour.csv
```

Option C: quick smoke test without downloading the real dataset:

```bash
python run_project.py --synthetic-test --fast
```

## Output files

The script writes files into `outputs/`:

- `cv_results_by_fold.csv`
- `cv_summary.csv`
- `holdout_results.csv`
- `holdout_predictions.csv`
- `model_rmse_comparison.png`
- `peak_f1_comparison.png`
- `actual_vs_predicted_best_model.png`
- `residual_diagnostic_best_model.png`
- coefficient or feature-importance CSV files when available

## Recommended report wording

Use RMSE/MAE/R2 to discuss demand forecasting accuracy. Use peak-demand F1 to discuss the operational value of predicting rush-hour imbalance. Mention that the UCI dataset is hourly system-level demand rather than station-level dockless bike inventory data, so the project studies temporal demand imbalance and peak-load warning, not full spatial relocation optimization.
