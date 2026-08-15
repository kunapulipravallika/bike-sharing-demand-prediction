import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bike_train_full.csv"
MODEL_DIR = ROOT / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"


MODEL_BUILDERS = {
    "linear_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "ridge": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=2.0)),
    ]),
    "lasso": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=10000)),
    ]),
    "knn": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
    ]),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    ),
}


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")

    frame["hour"] = frame["datetime"].dt.hour
    frame["month"] = frame["datetime"].dt.month
    frame["year"] = frame["datetime"].dt.year
    frame["weekday"] = frame["datetime"].dt.weekday
    frame["day"] = frame["datetime"].dt.day
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    frame["morning_rush"] = frame["hour"].isin([7, 8, 9]).astype(int)
    frame["evening_rush"] = frame["hour"].isin([16, 17, 18, 19]).astype(int)
    frame["rush_workday"] = ((frame["morning_rush"] | frame["evening_rush"]) & (frame["workingday"] == 1)).astype(int)
    frame["temp_sq"] = frame["temp"] ** 2
    frame["humidity_sq"] = frame["humidity"] ** 2
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    frame = frame.drop(columns=["datetime"], errors="ignore")
    return frame


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_PATH)
    processed = engineer_features(train)

    y = processed["count"].to_numpy()
    y_log = np.log1p(y)
    drop_cols = ["count", "casual", "registered"]
    X_df = processed.drop(columns=drop_cols, errors="ignore")

    feature_columns = X_df.columns.tolist()
    X = X_df.to_numpy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for model_name, builder in MODEL_BUILDERS.items():
        scores = []
        maes = []
        for tr_idx, val_idx in kf.split(X):
            model = builder()
            model.fit(X[tr_idx], y_log[tr_idx])
            y_pred = np.expm1(model.predict(X[val_idx]))
            scores.append(rmsle(y[val_idx], y_pred))
            maes.append(float(mean_absolute_error(y[val_idx], y_pred)))

        final_model = builder()
        final_model.fit(X, y_log)

        model_path = MODEL_DIR / f"{model_name}.pkl"
        joblib.dump(final_model, model_path)

        results[model_name] = {
            "path": str(model_path.relative_to(ROOT)),
            "cv_rmsle_mean": float(np.mean(scores)),
            "cv_rmsle_std": float(np.std(scores)),
            "cv_mae_mean": float(np.mean(maes)),
            "trained_on": len(X_df),
        }

        print(f"Saved {model_name} -> {model_path.name} | RMSLE={np.mean(scores):.5f}")

    payload = {
        "target": "count",
        "target_transform": "log1p",
        "feature_columns": feature_columns,
        "models": results,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
