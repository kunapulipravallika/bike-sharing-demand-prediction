import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATASET_PATH = DATA_DIR / "breast_cancer_full.csv"
TEST_PATH = ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
TARGET_COL = "diagnosis"


def _model_builders() -> dict:
    return {
        "logistic_regression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "knn": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
        "naive_bayes": lambda: GaussianNB(),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


def _build_dataset() -> tuple[pd.DataFrame, list[str]]:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df[TARGET_COL] = df["target"].map({0: "malignant", 1: "benign"})
    df = df.drop(columns=["target"])
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return df, feature_cols


def _cv_metrics(model_builder, X: np.ndarray, y: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc, auc, prec, rec, f1, mcc = [], [], [], [], [], []

    for tr_idx, val_idx in skf.split(X, y):
        model = model_builder()
        model.fit(X[tr_idx], y[tr_idx])
        prob = model.predict_proba(X[val_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)

        acc.append(accuracy_score(y[val_idx], pred))
        auc.append(roc_auc_score(y[val_idx], prob))
        prec.append(precision_score(y[val_idx], pred, zero_division=0))
        rec.append(recall_score(y[val_idx], pred, zero_division=0))
        f1.append(f1_score(y[val_idx], pred, zero_division=0))
        mcc.append(matthews_corrcoef(y[val_idx], pred))

    return {
        "accuracy": float(np.mean(acc)),
        "auc": float(np.mean(auc)),
        "precision": float(np.mean(prec)),
        "recall": float(np.mean(rec)),
        "f1": float(np.mean(f1)),
        "mcc": float(np.mean(mcc)),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    full_df, feature_cols = _build_dataset()
    full_df.to_csv(DATASET_PATH, index=False)

    train_df, test_df = train_test_split(
        full_df,
        test_size=0.25,
        random_state=42,
        stratify=full_df[TARGET_COL],
    )
    test_df.to_csv(TEST_PATH, index=False)

    # Remove stale pickle files to avoid loading wrong model types in the app.
    for old in MODEL_DIR.glob("*.pkl"):
        old.unlink()

    y_train = (train_df[TARGET_COL] == "benign").astype(int).to_numpy()
    X_train = train_df[feature_cols].to_numpy()

    metrics = {}
    for name, builder in _model_builders().items():
        model = builder()
        model.fit(X_train, y_train)
        model_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)

        cv = _cv_metrics(builder, X_train, y_train)
        metrics[name] = {
            "path": str(model_path.relative_to(ROOT)),
            **cv,
            "trained_on": int(len(train_df)),
        }
        print(f"Saved {name} -> {model_path.name}")

    payload = {
        "target_column": TARGET_COL,
        "feature_columns": feature_cols,
        "positive_class": "benign",
        "label_order": ["malignant", "benign"],
        "models": metrics,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved dataset -> {DATASET_PATH}")
    print(f"Saved test data -> {TEST_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
