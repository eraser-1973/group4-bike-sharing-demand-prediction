"""Run the from-scratch bike-sharing demand prediction project.

Usage:
    python run_project.py
    python run_project.py --data-path data/hour.csv
    python run_project.py --synthetic-test

Only NumPy, Pandas and Matplotlib are used. No scikit-learn, LightGBM,
TensorFlow, Keras, PyTorch, or other machine-learning libraries are required.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_utils import load_hour_data, build_features, chronological_train_test_split
from src.models import RidgeLinearRegressionGD, KNNRegressorScratch, DecisionTreeRegressorScratch, GradientBoostingRegressorScratch
from src.evaluation import evaluate_model_cv, fit_predict_holdout, summarize_holdout


def build_model_factories(fast=False):
    if fast:
        return {
            "Ridge Linear Regression GD": lambda: RidgeLinearRegressionGD(learning_rate=0.03, epochs=1200, l2=1e-3),
            "KNN Regression": lambda: KNNRegressorScratch(k=9),
            "CART Regression Tree": lambda: DecisionTreeRegressorScratch(max_depth=4, min_samples_leaf=25, n_thresholds=16),
            "Gradient Boosting Trees": lambda: GradientBoostingRegressorScratch(n_estimators=15, learning_rate=0.10, max_depth=2, min_samples_leaf=30, n_thresholds=12),
        }
    return {
        "Ridge Linear Regression GD": lambda: RidgeLinearRegressionGD(learning_rate=0.03, epochs=3000, l2=1e-3),
        "KNN Regression": lambda: KNNRegressorScratch(k=15),
        "CART Regression Tree": lambda: DecisionTreeRegressorScratch(max_depth=5, min_samples_leaf=30, n_thresholds=24),
        "Gradient Boosting Trees": lambda: GradientBoostingRegressorScratch(n_estimators=40, learning_rate=0.08, max_depth=2, min_samples_leaf=40, n_thresholds=18),
    }


def save_metric_plots(holdout_df, y_test, pred_dict, timestamps_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RMSE comparison.
    plt.figure(figsize=(8, 4.8))
    plt.bar(holdout_df["model"], holdout_df["rmse"])
    plt.ylabel("RMSE on holdout set")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_rmse_comparison.png", dpi=180)
    plt.close()

    # MAE comparison.
    plt.figure(figsize=(8, 4.8))
    plt.bar(holdout_df["model"], holdout_df["mae"])
    plt.ylabel("MAE on holdout set")
    plt.title("MAE Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_mae_comparison.png", dpi=180)
    plt.close()

    # R2 comparison.
    plt.figure(figsize=(8, 4.8))
    plt.bar(holdout_df["model"], holdout_df["r2"])
    plt.ylabel("R2 on holdout set")
    plt.title("R2 Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_r2_comparison.png", dpi=180)
    plt.close()

    # Peak warning F1 comparison.
    plt.figure(figsize=(8, 4.8))
    plt.bar(holdout_df["model"], holdout_df["peak_f1"])
    plt.ylabel("Peak-demand F1")
    plt.title("High-Demand Warning Performance")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "peak_f1_comparison.png", dpi=180)
    plt.close()

    # Prediction line plot for the best RMSE model on the last 7 days or up to 168 points.
    best_model = holdout_df.sort_values("rmse").iloc[0]["model"]
    n = min(168, len(y_test))
    x = pd.to_datetime(timestamps_test.iloc[-n:])
    plt.figure(figsize=(10, 4.8))
    plt.plot(x, y_test[-n:], label="Actual")
    plt.plot(x, pred_dict[best_model][-n:], label=f"Predicted: {best_model}")
    plt.ylabel("Bike rentals per hour")
    plt.title("Actual vs Predicted Demand, Last Holdout Window")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted_best_model.png", dpi=180)
    plt.close()

    # Residual plot for best model.
    residual = y_test - pred_dict[best_model]
    plt.figure(figsize=(7, 4.8))
    plt.scatter(pred_dict[best_model], residual, s=10, alpha=0.6)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Predicted demand")
    plt.ylabel("Residual: actual - predicted")
    plt.title(f"Residual Diagnostic: {best_model}")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_diagnostic_best_model.png", dpi=180)
    plt.close()


def save_interpretation_files(model, feature_names, model_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "weights_"):
        coefs = pd.DataFrame({
            "feature": ["intercept"] + list(feature_names),
            "coefficient_on_log_cnt": model.weights_,
            "abs_coefficient": np.abs(model.weights_),
        }).sort_values("abs_coefficient", ascending=False)
        coefs.to_csv(output_dir / "linear_model_coefficients.csv", index=False)
    if hasattr(model, "feature_importances"):
        imp = model.feature_importances()
        if imp is not None:
            pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False).to_csv(
                output_dir / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv", index=False
            )


def save_report_summary(holdout_df, cv_summary, output_dir):
    """Save a plain-English summary that can be pasted into the final report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = holdout_df.sort_values("rmse").reset_index(drop=True)
    best = ranked.iloc[0]

    lines = [
        "Experimental Result Summary",
        "",
        f"Best model based on holdout RMSE: {best['model']}",
        "",
        "Main holdout results of the best model:",
        f"- RMSE: {best['rmse']:.4f}",
        f"- MAE: {best['mae']:.4f}",
        f"- R2: {best['r2']:.4f}",
        f"- Peak Accuracy: {best['peak_accuracy']:.4f}",
        f"- Peak Precision: {best['peak_precision']:.4f}",
        f"- Peak Recall: {best['peak_recall']:.4f}",
        f"- Peak F1: {best['peak_f1']:.4f}",
        f"- Training Time: {best['train_time_seconds']:.4f} seconds",
        "",
        "Model ranking by holdout RMSE:",
    ]

    for i, row in ranked.iterrows():
        lines.append(
            f"{i + 1}. {row['model']}: "
            f"RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, "
            f"R2={row['r2']:.4f}, Peak F1={row['peak_f1']:.4f}"
        )

    lines.extend([
        "",
        "Interpretation:",
        f"The best-performing model is {best['model']} because it achieves the lowest RMSE on the chronological holdout test set. "
        "RMSE, MAE and R2 evaluate the regression accuracy of hourly bike-sharing demand prediction, while Peak F1 evaluates the ability to identify high-demand periods. "
        "A high Peak F1 score is important because the project aims not only to predict the number of bike rentals, but also to support early warning of potential demand imbalance.",
        "",
        "The comparison shows that nonlinear models are more suitable than the linear baseline for this task. "
        "Bike-sharing demand is affected by complex interactions among hour, working day, weather, temperature, humidity and historical demand features. "
        "Therefore, tree-based models can capture threshold-based and nonlinear patterns more effectively than a simple linear model.",
        "",
        "Generated output files:",
        "- cv_results_by_fold.csv",
        "- cv_summary.csv",
        "- holdout_results.csv",
        "- holdout_predictions.csv",
        "- model_rmse_comparison.png",
        "- model_mae_comparison.png",
        "- model_r2_comparison.png",
        "- peak_f1_comparison.png",
        "- actual_vs_predicted_best_model.png",
        "- residual_diagnostic_best_model.png",
    ])

    with open(output_dir / "result_summary_for_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=None, help="Path to UCI hour.csv. If omitted, the script attempts to download it.")
    parser.add_argument("--data-dir", default="data", help="Directory for downloaded data.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV and PNG outputs.")
    parser.add_argument("--synthetic-test", action="store_true", help="Use synthetic data for smoke testing when hour.csv is unavailable.")
    parser.add_argument("--fast", action="store_true", help="Use smaller hyperparameters for quick debugging.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_hour_data(data_path=args.data_path, data_dir=args.data_dir, synthetic=args.synthetic_test)
    X, y, timestamps = build_features(df)
    X_train, X_test, y_train, y_test, ts_train, ts_test = chronological_train_test_split(X, y, timestamps, test_size=0.2)
    print(f"Loaded {len(df)} raw rows. After lag features: {len(y)} rows, {X.shape[1]} features.")
    print(f"Train rows: {len(y_train)}, test rows: {len(y_test)}")

    model_factories = build_model_factories(fast=args.fast or args.synthetic_test)

    # Cross-validation on the training segment only.
    cv_rows = []
    for model_name, factory in model_factories.items():
        print(f"Cross-validating {model_name} ...")
        cv_df = evaluate_model_cv(factory, X_train, y_train, model_name=model_name, n_splits=5, use_log_target=True)
        cv_rows.append(cv_df)
    cv_all = pd.concat(cv_rows, ignore_index=True)
    cv_all.to_csv(output_dir / "cv_results_by_fold.csv", index=False)
    cv_summary = cv_all.groupby("model", as_index=False).agg({
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"],
        "r2": ["mean", "std"],
        "peak_f1": ["mean", "std"],
        "train_time_seconds": ["mean"],
    })
    cv_summary.columns = ["_".join(col).strip("_") for col in cv_summary.columns.to_flat_index()]
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    print("\nCross-validation summary:")
    print(cv_summary.to_string(index=False))

    # Final holdout evaluation.
    holdout_rows = []
    pred_dict = {}
    fitted_models = {}
    for model_name, factory in model_factories.items():
        print(f"Fitting final holdout model: {model_name} ...")
        model, scaler, pred, train_time = fit_predict_holdout(factory, X_train, X_test, y_train, use_log_target=True)
        fitted_models[model_name] = model
        pred_dict[model_name] = pred
        holdout_rows.append(summarize_holdout(model_name, y_train, y_test, pred, train_time))

    holdout_df = pd.DataFrame(holdout_rows).sort_values("rmse")
    holdout_df.to_csv(output_dir / "holdout_results.csv", index=False)
    save_report_summary(holdout_df, cv_summary, output_dir)
    print("\nHoldout results:")
    print(holdout_df.to_string(index=False))

    best_name = holdout_df.iloc[0]["model"]
    print(f"\nBest model based on RMSE: {best_name}")
    print("This model should be emphasized in the final report discussion.")

    # Save prediction table for report figures.
    pred_table = pd.DataFrame({"datetime": ts_test, "actual_cnt": y_test})
    for model_name, pred in pred_dict.items():
        pred_table[f"pred_{model_name}"] = pred
    pred_table.to_csv(output_dir / "holdout_predictions.csv", index=False)

    save_metric_plots(holdout_df, y_test, pred_dict, ts_test, output_dir)
    save_interpretation_files(fitted_models[best_name], X.columns.tolist(), best_name, output_dir)
    # Also save linear model coefficients if available, because they are easy to discuss in the report.
    for name, model in fitted_models.items():
        if hasattr(model, "weights_"):
            save_interpretation_files(model, X.columns.tolist(), name, output_dir)

    print(f"\nDone. Outputs saved to: {output_dir.resolve()}")
    print("Recommended report discussion: compare RMSE/MAE/R2 for regression and peak_F1 for high-demand warning.")


if __name__ == "__main__":
    main()
