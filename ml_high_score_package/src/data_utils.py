"""Data loading and preprocessing for the UCI Bike Sharing hourly dataset."""
from pathlib import Path
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download_uci_hour_csv(data_dir="data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    hour_path = data_dir / "hour.csv"
    if hour_path.exists():
        return hour_path
    print(f"Downloading UCI Bike Sharing dataset from {UCI_ZIP_URL} ...")
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=30) as response:
            raw = response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extract("hour.csv", data_dir)
        return hour_path
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset automatically. "
            "Please download 'Bike-Sharing-Dataset/hour.csv' from the UCI repository "
            "and run: python run_project.py --data-path data/hour.csv"
        ) from exc


def make_synthetic_hour_data(n_days=120, seed=42):
    """Create a small synthetic dataset for smoke testing when the real data is absent."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for hr in range(24):
            weekday = day % 7
            workingday = 1 if weekday < 5 else 0
            season = 1 + (day // 90) % 4
            mnth = 1 + (day // 30) % 12
            temp = 0.45 + 0.25 * np.sin(2 * np.pi * day / 365) + rng.normal(0, 0.04)
            hum = 0.55 + rng.normal(0, 0.10)
            windspeed = 0.20 + rng.normal(0, 0.05)
            rush = 80 * workingday * (np.exp(-((hr - 8) ** 2) / 5) + np.exp(-((hr - 17) ** 2) / 5))
            daily = 40 + 60 * np.sin(np.pi * hr / 24) ** 2
            cnt = max(1, daily + rush + 120 * temp - 30 * hum + rng.normal(0, 20))
            rows.append({
                "instant": len(rows) + 1,
                "dteday": f"2011-01-{1 + day % 28:02d}",
                "season": season,
                "yr": 0,
                "mnth": mnth,
                "hr": hr,
                "holiday": 0,
                "weekday": weekday,
                "workingday": workingday,
                "weathersit": 1,
                "temp": float(np.clip(temp, 0, 1)),
                "atemp": float(np.clip(temp + rng.normal(0, 0.02), 0, 1)),
                "hum": float(np.clip(hum, 0, 1)),
                "windspeed": float(np.clip(windspeed, 0, 1)),
                "casual": int(cnt * 0.25),
                "registered": int(cnt * 0.75),
                "cnt": int(cnt),
            })
    return pd.DataFrame(rows)


def load_hour_data(data_path=None, data_dir="data", synthetic=False):
    if synthetic:
        return make_synthetic_hour_data()
    if data_path is not None and Path(data_path).exists():
        return pd.read_csv(data_path)
    hour_path = download_uci_hour_csv(data_dir=data_dir)
    return pd.read_csv(hour_path)


def one_hot(df, column, prefix, categories):
    out = pd.DataFrame(index=df.index)
    for value in categories:
        out[f"{prefix}_{value}"] = (df[column] == value).astype(int)
    return out


def build_features(df):
    """Build leakage-aware temporal and weather features.

    Target: cnt, the total number of rented bikes in an hour.
    Lag/rolling features use shift(1), so they only use information available before the prediction hour.
    """
    df = df.copy()
    required = {"dteday", "hr", "cnt", "season", "mnth", "weekday", "workingday", "holiday", "weathersit", "temp", "atemp", "hum", "windspeed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Cyclical encodings make time variables continuous at boundaries, e.g., 23:00 close to 00:00.
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)

    # Historical demand features.
    df["cnt_lag_1"] = df["cnt"].shift(1)
    df["cnt_lag_24"] = df["cnt"].shift(24)
    df["cnt_roll_mean_24"] = df["cnt"].shift(1).rolling(window=24, min_periods=12).mean()
    df["cnt_roll_std_24"] = df["cnt"].shift(1).rolling(window=24, min_periods=12).std()

    numeric_cols = [
        "temp", "atemp", "hum", "windspeed", "workingday", "holiday",
        "hr_sin", "hr_cos", "weekday_sin", "weekday_cos", "mnth_sin", "mnth_cos",
        "cnt_lag_1", "cnt_lag_24", "cnt_roll_mean_24", "cnt_roll_std_24",
    ]
    cat_df = pd.concat([
        one_hot(df, "season", "season", [1, 2, 3, 4]),
        one_hot(df, "weathersit", "weathersit", [1, 2, 3, 4]),
    ], axis=1)
    feature_df = pd.concat([df[numeric_cols], cat_df], axis=1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    valid = feature_df.notna().all(axis=1) & df["cnt"].notna()
    feature_df = feature_df.loc[valid].reset_index(drop=True)
    y = df.loc[valid, "cnt"].astype(float).to_numpy()
    timestamps = df.loc[valid, "datetime"].reset_index(drop=True)
    return feature_df, y, timestamps


def chronological_train_test_split(X, y, timestamps, test_size=0.2):
    n = len(y)
    split = int(n * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y[:split], y[split:], timestamps.iloc[:split], timestamps.iloc[split:]
