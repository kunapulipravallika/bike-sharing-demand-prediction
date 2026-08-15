import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
METRICS_FILE = MODEL_DIR / "metrics.json"
DEFAULT_TEST_FILE = HERE / "test_data.csv"
TRAIN_DATA_FILE = HERE / "data" / "bike_train_full.csv"


def _model_builders() -> dict:
    # Keep cloud fallback models compact so cold-start retraining is fast.
    return {
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
            n_estimators=80,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=250,
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
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", dayfirst=True)

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

    frame = frame.drop(columns=["datetime", "casual", "registered"], errors="ignore")
    return frame


@st.cache_data
def load_metadata() -> dict:
    if not METRICS_FILE.exists():
        return {"feature_columns": [], "models": {}, "target_transform": "log1p"}
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_models() -> dict:
    models = {}
    load_errors = []
    for path in sorted(MODEL_DIR.glob("*.pkl")):
        try:
            models[path.stem] = joblib.load(path)
        except Exception as exc:
            load_errors.append((path.name, exc))

    if models:
        return models

    # If all pickles fail to unpickle (common on cloud env/version mismatch),
    # rebuild fresh models from training CSV.
    if not TRAIN_DATA_FILE.exists():
        if load_errors:
            msg = "; ".join([f"{name}: {type(err).__name__}" for name, err in load_errors])
            raise RuntimeError(f"Model files could not be loaded and no training data is available. {msg}")
        return models

    train = pd.read_csv(TRAIN_DATA_FILE)
    processed = engineer_features(train)
    y = processed["count"].to_numpy()
    y_log = np.log1p(y)
    X_df = processed.drop(columns=["count", "casual", "registered"], errors="ignore")
    X = X_df.to_numpy()

    builders = _model_builders()
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    results = {}
    for model_name, builder in builders.items():
        scores = []
        for tr_idx, val_idx in kf.split(X):
            model = builder()
            model.fit(X[tr_idx], y_log[tr_idx])
            pred = np.expm1(model.predict(X[val_idx]))
            scores.append(rmsle(y[val_idx], pred))

        final_model = builder()
        final_model.fit(X, y_log)
        models[model_name] = final_model
        results[model_name] = {
            "path": f"models/{model_name}.pkl",
            "cv_rmsle_mean": float(np.mean(scores)),
            "cv_rmsle_std": float(np.std(scores)),
            "trained_on": int(len(X_df)),
        }

    payload = {
        "target": "count",
        "target_transform": "log1p",
        "feature_columns": X_df.columns.tolist(),
        "models": results,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{model_name}.pkl")
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSLE": rmsle(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    st.set_page_config(page_title="Bike Sharing Demand Prediction", page_icon="🚲", layout="wide")
    st.title("Bike Sharing Demand Prediction - Model Explorer")
    st.caption("Upload a test CSV, pick a model, and generate predictions.")

    metadata = load_metadata()
    models = load_models()

    if not models:
        st.error("No trained models found in models/. Run models/train_models.py first.")
        st.stop()

    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    with right:
        model_name = st.selectbox("Choose model", options=list(models.keys()))

    if uploaded is not None:
        test_df = pd.read_csv(uploaded)
    elif DEFAULT_TEST_FILE.exists():
        test_df = pd.read_csv(DEFAULT_TEST_FILE)
        st.info("Using bundled test_data.csv")
    else:
        st.warning("Upload a CSV to continue.")
        st.stop()

    original = test_df.copy()
    processed = engineer_features(test_df)

    feature_columns = metadata.get("feature_columns") or processed.columns.tolist()
    missing = [c for c in feature_columns if c not in processed.columns]
    for col in missing:
        processed[col] = 0
    X = processed[feature_columns].fillna(0)

    model = models[model_name]
    y_pred = np.expm1(model.predict(X))
    y_pred = np.clip(np.round(y_pred), 0, None).astype(int)

    out = pd.DataFrame(
        {
            "datetime": original.get("datetime", pd.Series(range(len(y_pred)))).astype(str),
            "count_predicted": y_pred,
        }
    )

    st.subheader("Prediction Preview")
    st.dataframe(out.head(20), use_container_width=True)

    if "count" in original.columns:
        scores = compute_metrics(original["count"].to_numpy(), y_pred)
        cols = st.columns(3)
        for i, (name, value) in enumerate(scores.items()):
            cols[i].metric(name, f"{value:.4f}")

    st.download_button(
        "Download submission CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )

    st.subheader("Cross-validation Summary")
    cv = metadata.get("models", {})
    if cv:
        table = pd.DataFrame.from_dict(cv, orient="index").sort_values("cv_rmsle_mean")
        st.dataframe(table, use_container_width=True)


if __name__ == "__main__":
    main()
