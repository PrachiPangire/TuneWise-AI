
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/genre_classifier.joblib")


def test_saved_model_exists():
    """Check that the trained model file exists."""
    assert MODEL_PATH.exists()


def test_model_can_predict_genre():
    """Check that the model produces a valid genre prediction."""
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    sample = pd.DataFrame(
        [[0.5, 0.5, 0, -8.0, 1, 0.1, 0.5, 0.0, 0.2, 0.5, 120.0, 4]],
        columns=features
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
