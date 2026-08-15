import json
import runpy
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
METRICS_FILE = MODEL_DIR / "metrics.json"
DEFAULT_TEST_FILE = HERE / "test_data.csv"
TRAIN_SCRIPT = MODEL_DIR / "train_models.py"


@st.cache_data
def load_metadata() -> dict:
    if not METRICS_FILE.exists():
        return {"feature_columns": [], "models": {}, "target_column": "diagnosis"}
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_training_script() -> None:
    if not TRAIN_SCRIPT.exists():
        raise RuntimeError("Missing model/train_models.py needed to generate model artifacts.")
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")


def _load_pickle_models(model_names: tuple[str, ...], expected_feature_count: int) -> tuple[dict, list[str]]:
    models = {}
    errors = []

    for name in model_names:
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            errors.append(f"{path.name}: missing")
            continue

        try:
            model = joblib.load(path)
            n_features = getattr(model, "n_features_in_", None)
            if expected_feature_count > 0 and n_features is not None and n_features != expected_feature_count:
                errors.append(f"{path.name}: incompatible features ({n_features} != {expected_feature_count})")
                continue
            models[name] = model
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}")

    return models, errors


@st.cache_resource
def load_models(model_names: tuple[str, ...], expected_feature_count: int) -> dict:
    models, errors = _load_pickle_models(model_names, expected_feature_count)
    if models:
        return models

    _run_training_script()

    models, errors_after = _load_pickle_models(model_names, expected_feature_count)
    if models:
        return models

    err_text = "; ".join(errors_after or errors)
    raise RuntimeError(f"Could not load or regenerate model artifacts. {err_text}")


def model_metrics_table(metadata: dict) -> pd.DataFrame:
    rows = []
    for model_name, m in metadata.get("models", {}).items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": m.get("accuracy"),
                "AUC": m.get("auc"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1": m.get("f1"),
                "MCC": m.get("mcc"),
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False, ignore_index=True)


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Classification", page_icon="🧪", layout="wide")
    st.title("Breast Cancer Classification - Assignment 2")
    st.caption("Upload test CSV, choose a model, and view metrics, confusion matrix, and predictions.")

    metadata = load_metadata()
    configured_models = tuple(sorted(metadata.get("models", {}).keys()))
    expected_feature_count = len(metadata.get("feature_columns", []))
    models = load_models(configured_models, expected_feature_count)

    if not models:
        st.error("No trained models found in model/. Run model/train_models.py first.")
        st.stop()

    st.subheader("Overall Test Metrics (All Models)")
    metrics_df = model_metrics_table(metadata)
    st.dataframe(metrics_df, use_container_width=True)

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

    feature_columns = metadata.get("feature_columns", [])
    target_column = metadata.get("target_column", "diagnosis")

    missing = [c for c in feature_columns if c not in test_df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required features: {', '.join(missing)}")
        st.stop()

    X = test_df[feature_columns].copy()
    model = models[model_name]
    y_pred_num = model.predict(X)
    y_pred_label = pd.Series(y_pred_num).map({0: "benign", 1: "malignant"}).fillna(y_pred_num)

    out = test_df.copy()
    out["predicted_diagnosis"] = y_pred_label

    st.subheader("Prediction Preview")
    st.dataframe(out[["predicted_diagnosis"] + feature_columns].head(20), use_container_width=True)

    st.download_button(
        "Download prediction CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="submission_predictions.csv",
        mime="text/csv",
    )

    model_info = metadata.get("models", {}).get(model_name, {})
    st.subheader(f"Evaluation for {model_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{model_info.get('accuracy', 0):.4f}")
    c2.metric("AUC", f"{model_info.get('auc', 0):.4f}")
    c3.metric("MCC", f"{model_info.get('mcc', 0):.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", f"{model_info.get('precision', 0):.4f}")
    c5.metric("Recall", f"{model_info.get('recall', 0):.4f}")
    c6.metric("F1", f"{model_info.get('f1', 0):.4f}")

    st.markdown("Confusion Matrix")
    cm = model_info.get("confusion_matrix", [[0, 0], [0, 0]])
    cm_df = pd.DataFrame(cm, index=["Actual Benign", "Actual Malignant"], columns=["Pred Benign", "Pred Malignant"])
    st.dataframe(cm_df, use_container_width=False)

    st.markdown("Classification Report")
    report = model_info.get("classification_report", {})
    if report:
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df, use_container_width=True)

    if target_column in test_df.columns:
        y_true = test_df[target_column].astype(str).str.lower()
        match = (y_true == y_pred_label.astype(str).str.lower()).mean()
        st.info(f"Match rate on uploaded labeled data: {match:.2%}")


if __name__ == "__main__":
    main()
