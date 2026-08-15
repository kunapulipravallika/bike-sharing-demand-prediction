import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "breast_cancer_full.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.json"
TEST_DATA_PATH = ROOT / "test_data.csv"

MODEL_BUILDERS = {
    "logistic_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
    "knn": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=9, weights="distance")),
    ]),
    "naive_bayes": lambda: GaussianNB(),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}


def encode_target(series: pd.Series) -> pd.Series:
    mapping = {
        "benign": 0,
        "malignant": 1,
        0: 0,
        1: 1,
    }
    normalized = series.astype(str).str.lower().map(mapping)
    if normalized.isna().any():
        raise ValueError("Target column contains unsupported labels. Expected benign/malignant or 0/1.")
    return normalized.astype(int)


def proba_for_positive_class(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def evaluate(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_proba = proba_for_positive_class(model, X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    model_auc = float(auc(fpr, tpr))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": model_auc,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["benign", "malignant"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    target_col = "diagnosis"

    y = encode_target(df[target_col])
    X = df.drop(columns=[target_col]).copy()

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Store test data used in experiments for submission and app upload.
    test_export = X_test.copy()
    test_export[target_col] = np.where(y_test == 1, "malignant", "benign")
    test_export.to_csv(TEST_DATA_PATH, index=False)

    metrics = {}
    for model_name, builder in MODEL_BUILDERS.items():
        model = builder()
        model.fit(X_train, y_train)

        artifact_path = MODEL_DIR / f"{model_name}.pkl"
        joblib.dump(model, artifact_path)

        metrics[model_name] = {
            "path": str(artifact_path.relative_to(ROOT)),
            **evaluate(model, X_test, y_test),
        }

        print(f"Saved {artifact_path.name}")

    payload = {
        "problem_type": "binary_classification",
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "target_column": target_col,
        "positive_label": "malignant",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_columns,
        "models": metrics,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved test data -> {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
