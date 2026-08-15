import json
import runpy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
METRICS_FILE = MODEL_DIR / "metrics.json"
DEFAULT_TEST_FILE = HERE / "test_data.csv"
TRAIN_SCRIPT = MODEL_DIR / "train_models.py"


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


def _run_training_script() -> None:
    if not TRAIN_SCRIPT.exists():
        raise RuntimeError("Missing models/train_models.py needed to generate model artifacts.")
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")


def _load_pickle_models() -> tuple[dict, list[str]]:
    models = {}
    errors = []
    for path in sorted(MODEL_DIR.glob("*.pkl")):
        try:
            models[path.stem] = joblib.load(path)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}")
    return models, errors


@st.cache_resource
def load_models() -> dict:
    models, errors = _load_pickle_models()
    if models:
        return models

    # If pickles are incompatible with the cloud environment, regenerate locally.
    _run_training_script()

    models, errors_after = _load_pickle_models()
    if models:
        return models

    err_text = "; ".join(errors_after or errors)
    raise RuntimeError(f"Could not load or regenerate model artifacts. {err_text}")


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

    c1, c2, c3 = st.columns(3)
    c1.metric("Min Prediction", f"{int(y_pred.min())}")
    c2.metric("Max Prediction", f"{int(y_pred.max())}")
    c3.metric("Mean Prediction", f"{float(y_pred.mean()):.1f}")

    st.download_button(
        "Download submission CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
