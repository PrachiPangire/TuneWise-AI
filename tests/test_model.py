"""Tests for the preserved genre classifier.

These tests skip automatically when `models/genre_classifier.joblib` is absent
so CI can still validate popularity utilities without the large genre artifact.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

MODEL_PATH = Path("models/genre_classifier.joblib")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="genre_classifier.joblib is not present in this environment",
)


def test_saved_model_exists():
    """Check that the trained genre model file exists when available."""
    assert MODEL_PATH.exists()


def test_model_can_predict_genre():
    """Check that the model produces a valid genre prediction."""
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    sample = pd.DataFrame(
        [[0.5, 0.5, 0, -8.0, 1, 0.1, 0.5, 0.0, 0.2, 0.5, 120.0, 4]],
        columns=features,
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert isinstance(prediction[0], str)
    assert prediction[0] in model.classes_


def test_model_supports_multiple_genres():
    """Check that the classifier can predict across multiple genres."""
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    assert len(model.classes_) > 1
