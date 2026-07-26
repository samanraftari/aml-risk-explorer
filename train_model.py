"""Train and evaluate the educational AML risk model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_aml.csv"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "adverse_risk"
NUMERIC_FEATURES = [
    "age",
    "wbc_10e9_l",
    "hemoglobin_g_dl",
    "platelets_10e9_l",
    "marrow_blast_percent",
]
CATEGORICAL_FEATURES = [
    "complex_karyotype",
    "monosomal_karyotype",
    "tp53_abnormality",
    "flt3_itd",
    "npm1_mutation",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )


def evaluate(y_true: pd.Series, probabilities) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "sensitivity": round(float(recall_score(y_true, predictions)), 4),
        "specificity": round(float(tn / (tn + fp)), 4),
        "precision": round(float(precision_score(y_true, predictions)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "test_records": int(len(y_true)),
    }


def train(data: pd.DataFrame) -> tuple[Pipeline, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.25,
        stratify=data[TARGET],
        random_state=42,
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    metrics = evaluate(y_test, model.predict_proba(x_test)[:, 1])
    return model, metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python generate_data.py` first.")
    data = pd.read_csv(DATA_PATH)
    model, metrics = train(data)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

