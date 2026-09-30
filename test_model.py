"""Unit tests for preprocessing, validation, and prediction."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from model import FEATURE_NAMES, build_pipeline, predict_risk, validate_inputs


def sample_values() -> dict[str, float]:
    """Return a valid sample input."""
    return {name: 1.0 for name in FEATURE_NAMES}


def test_pipeline_imputes_missing_values() -> None:
    """The pipeline should handle a missing feature without crashing."""
    frame = pd.DataFrame([sample_values(), sample_values()])
    frame.loc[0, "Glucose"] = pd.NA

    model = build_pipeline(LogisticRegression(max_iter=500))
    target = pd.Series([0, 1])
    model.fit(frame, target)

    assert model.predict(frame).shape == (2,)


def test_input_validation_rejects_negative_values() -> None:
    """Negative health measurements should be rejected."""
    values = sample_values()
    values["BMI"] = -1.0

    with pytest.raises(ValueError):
        validate_inputs(values)


def test_prediction_contract() -> None:
    """Prediction should return a stable label and probability."""
    frame = pd.DataFrame([sample_values(), {name: 2.0 for name in FEATURE_NAMES}])
    target = pd.Series([0, 1])

    model = build_pipeline(LogisticRegression(max_iter=500))
    model.fit(frame, target)

    result = predict_risk(model, sample_values())

    assert result["risk"] in {"Higher predicted risk", "Lower predicted risk"}
    assert 0.0 <= float(result["probability"]) <= 1.0


def test_missing_input_is_rejected() -> None:
    """A prediction request missing a required feature should fail clearly."""
    values = sample_values()
    values.pop("Age")

    with pytest.raises(ValueError):
        validate_inputs(values)
