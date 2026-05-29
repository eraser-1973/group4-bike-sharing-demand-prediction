"""Cross-validation and experiment helpers."""
import time
import numpy as np
import pandas as pd
from .models import StandardScalerScratch
from .metrics import rmse, mae, r2_score, mape, classification_report_from_threshold


def time_series_cv_splits(n_samples, n_splits=5):
    """Expanding-window time series CV without shuffling."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    fold_size = n_samples // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_start = train_end
        val_end = fold_size * (i + 1) if i < n_splits else n_samples
        if train_end == 0 or val_end <= val_start:
            continue
        splits.append((np.arange(train_end), np.arange(val_start, val_end)))
    return splits


def safe_expm1(x):
    return np.maximum(0.0, np.expm1(np.asarray(x, dtype=float)))


def evaluate_model_cv(model_factory, X_df, y, model_name, n_splits=5, use_log_target=True):
    X = X_df.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    rows = []
    splits = time_series_cv_splits(len(y), n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        scaler = StandardScalerScratch()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]
        y_train = np.log1p(y_train_raw) if use_log_target else y_train_raw
        model = model_factory()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        pred = model.predict(X_val)
        pred_raw = safe_expm1(pred) if use_log_target else np.maximum(0.0, pred)
        threshold = float(np.quantile(y_train_raw, 0.75))
        peak = classification_report_from_threshold(y_val_raw, pred_raw, threshold)
        rows.append({
            "model": model_name,
            "fold": fold,
            "rmse": rmse(y_val_raw, pred_raw),
            "mae": mae(y_val_raw, pred_raw),
            "r2": r2_score(y_val_raw, pred_raw),
            "mape": mape(y_val_raw, pred_raw),
            "peak_accuracy": peak["accuracy"],
            "peak_precision": peak["precision"],
            "peak_recall": peak["recall"],
            "peak_f1": peak["f1"],
            "train_time_seconds": train_time,
        })
    return pd.DataFrame(rows)


def fit_predict_holdout(model_factory, X_train_df, X_test_df, y_train, use_log_target=True):
    scaler = StandardScalerScratch()
    X_train = scaler.fit_transform(X_train_df.to_numpy(dtype=float))
    X_test = scaler.transform(X_test_df.to_numpy(dtype=float))
    target = np.log1p(y_train) if use_log_target else y_train
    model = model_factory()
    t0 = time.perf_counter()
    model.fit(X_train, target)
    train_time = time.perf_counter() - t0
    pred = model.predict(X_test)
    pred_raw = safe_expm1(pred) if use_log_target else np.maximum(0.0, pred)
    return model, scaler, pred_raw, train_time


def summarize_holdout(model_name, y_train, y_test, pred, train_time):
    threshold = float(np.quantile(y_train, 0.75))
    peak = classification_report_from_threshold(y_test, pred, threshold)
    return {
        "model": model_name,
        "rmse": rmse(y_test, pred),
        "mae": mae(y_test, pred),
        "r2": r2_score(y_test, pred),
        "mape": mape(y_test, pred),
        "peak_accuracy": peak["accuracy"],
        "peak_precision": peak["precision"],
        "peak_recall": peak["recall"],
        "peak_f1": peak["f1"],
        "train_time_seconds": train_time,
    }
