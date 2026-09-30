"""Data loading, preprocessing, model training, and prediction utilities."""
from __future__ import annotations

import io
import time
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from urllib.request import Request, urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/npradaschnor/"
    "Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
)

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

MISSING_AS_ZERO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(timeout: int = 8) -> pd.DataFrame:
    """Download and validate the public diabetes dataset.

    Args:
        timeout: Maximum network wait in seconds.

    Returns:
        A validated pandas DataFrame.

    Raises:
        RuntimeError: If the dataset cannot be downloaded or is malformed.
    """
    request = Request(
        DATA_URL,
        headers={"User-Agent": "DiabetesRiskPredictor/2.0"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise RuntimeError("Unable to download the public dataset.") from exc

    required = set(FEATURE_NAMES + ["Outcome"])
    if not required.issubset(df.columns):
        raise RuntimeError("Dataset is missing required columns.")

    df = df[FEATURE_NAMES + ["Outcome"]].copy()
    df[MISSING_AS_ZERO] = df[MISSING_AS_ZERO].replace(0, pd.NA)
    return df


def build_pipeline(model: Any) -> Pipeline:
    """Create a preprocessing + classifier pipeline.

    Median imputation is fitted only on the training split, preventing
    preprocessing leakage from the validation set.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def train_models() -> tuple[Pipeline, Pipeline, dict[str, float], dict[str, float]]:
    """Train Random Forest and Decision Tree models and return validation metrics."""
    df = load_dataset()
    x = df[FEATURE_NAMES]
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    random_forest = build_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    )
    decision_tree = build_pipeline(
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
        )
    )

    random_forest.fit(x_train, y_train)
    decision_tree.fit(x_train, y_train)

    rf_pred = random_forest.predict(x_test)
    dt_pred = decision_tree.predict(x_test)

    metrics = {
        "random_forest_accuracy": float(accuracy_score(y_test, rf_pred)),
        "decision_tree_accuracy": float(accuracy_score(y_test, dt_pred)),
        "validation_samples": float(len(y_test)),
    }

    fitted_rf = random_forest.named_steps["model"]
    importance = dict(
        sorted(
            zip(FEATURE_NAMES, fitted_rf.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    return random_forest, decision_tree, metrics, importance


def validate_inputs(values: dict[str, float]) -> None:
    """Validate prediction inputs before inference."""
    missing = [name for name in FEATURE_NAMES if name not in values]
    if missing:
        raise ValueError(f"Missing inputs: {', '.join(missing)}")

    for name in FEATURE_NAMES:
        value = values[name]
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")


def predict_risk(model: Pipeline, values: dict[str, float]) -> dict[str, float | str]:
    """Return a risk label and positive-class probability for one patient."""
    validate_inputs(values)
    row = pd.DataFrame([values], columns=FEATURE_NAMES)
    probability = float(model.predict_proba(row)[0, 1])
    label = "Higher predicted risk" if probability >= 0.50 else "Lower predicted risk"
    return {"risk": label, "probability": probability}


def benchmark_inference(model: Pipeline, batch_size: int = 128) -> dict[str, float]:
    """Measure prediction throughput and average latency for a synthetic batch."""
    sample = pd.DataFrame(
        [{name: 1.0 for name in FEATURE_NAMES}] * batch_size,
        columns=FEATURE_NAMES,
    )
    start = time.perf_counter()
    model.predict_proba(sample)
    elapsed = time.perf_counter() - start
    return {
        "batch_size": float(batch_size),
        "elapsed_ms": elapsed * 1000,
        "rows_per_second": batch_size / elapsed if elapsed else float("inf"),
    }
