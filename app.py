import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
METRICS_FILE = MODEL_DIR / "metrics.json"
DEFAULT_TEST_FILE = HERE / "test_data.csv"
TRAIN_SCRIPT = MODEL_DIR / "train_models.py"


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "KNN": "knn.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_data
def load_metadata() -> dict:
    if not METRICS_FILE.exists():
        return {
            "target_column": "diagnosis",
            "feature_columns": [],
            "positive_class": "benign",
            "label_order": ["malignant", "benign"],
            "models": {},
        }
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_artifacts() -> None:
    missing = [name for name in MODEL_FILES.values() if not (MODEL_DIR / name).exists()]
    if not missing and METRICS_FILE.exists() and DEFAULT_TEST_FILE.exists():
        return

    if not TRAIN_SCRIPT.exists():
        raise RuntimeError("Missing models/train_models.py needed to generate model artifacts.")

    namespace = {"__name__": "__main__"}
    code = TRAIN_SCRIPT.read_text(encoding="utf-8")
    exec(compile(code, str(TRAIN_SCRIPT), "exec"), namespace)


@st.cache_resource
def load_models() -> dict:
    _ensure_artifacts()

    models = {}
    for display_name, file_name in MODEL_FILES.items():
        model_path = MODEL_DIR / file_name
        if model_path.exists():
            models[display_name] = joblib.load(model_path)
    return models


def _compute_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "AUC": roc_auc_score(y_true_bin, y_prob),
        "Precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "Recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "F1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "MCC": matthews_corrcoef(y_true_bin, y_pred_bin),
    }


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Classification Explorer", page_icon="🔬", layout="wide")

    st.title("🔬 Breast Cancer Classification Explorer")
    st.caption(
        "Compare five machine-learning classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset. "
        "Upload your test CSV, pick a model and inspect the results."
    )

    metadata = load_metadata()
    models = load_models()
    if not models:
        st.error("No trained models found. Run models/train_models.py to generate them.")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

        if uploaded is None and DEFAULT_TEST_FILE.exists():
            st.info("No file uploaded - using bundled test_data.csv.")

        model_name = st.selectbox("Select a model", list(models.keys()))

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif DEFAULT_TEST_FILE.exists():
        df = pd.read_csv(DEFAULT_TEST_FILE)
    else:
        st.warning("Upload a CSV or add test_data.csv to continue.")
        st.stop()

    st.subheader("📄 Test Data Preview")
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    target_col = metadata.get("target_column", "diagnosis")
    feature_cols = metadata.get("feature_columns", [c for c in df.columns if c != target_col])
    label_order = metadata.get("label_order", ["malignant", "benign"])
    positive_class = metadata.get("positive_class", "benign")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        st.error(f"Uploaded file is missing required feature columns: {missing_features[:8]}...")
        st.stop()

    X = df[feature_cols].copy()
    model = models[model_name]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        raw = model.decision_function(X)
        prob = 1.0 / (1.0 + np.exp(-raw))

    pred_bin = (prob >= 0.5).astype(int)
    pred_label = np.where(pred_bin == 1, positive_class, "malignant")

    if target_col not in df.columns:
        st.warning(
            f"Target column '{target_col}' not found in uploaded file. "
            "Predictions are available, but evaluation metrics cannot be computed."
        )
        out = df.copy()
        out["prediction"] = pred_label
        st.dataframe(out.head(20), use_container_width=True)
        st.download_button(
            "Download predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
        st.stop()

    y_true_label = df[target_col].astype(str).str.lower()
    y_true_bin = (y_true_label == positive_class).astype(int).to_numpy()

    m = _compute_metrics(y_true_bin, pred_bin, prob)
    st.subheader(f"📊 Evaluation Metrics — {model_name}")
    cols = st.columns(6)
    for i, (k, v) in enumerate(m.items()):
        cols[i].metric(k, f"{v:.4f}")

    cm = confusion_matrix(y_true_label, pred_label, labels=label_order)
    report = classification_report(
        y_true_label,
        pred_label,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T

    left, right = st.columns(2)
    with left:
        st.subheader("🧮 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(label_order)
        ax.set_yticklabels(label_order, rotation=90)
        st.pyplot(fig)

    with right:
        st.subheader("📝 Classification Report")
        st.dataframe(report_df, use_container_width=True)


if __name__ == "__main__":
    main()
