"""Popularity regression tests using synthetic data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_utils import NUMERIC_AUDIO_FEATURES, clean_indian_songs
from src.prediction import (
    ModelNotFoundError,
    default_feature_values,
    load_popularity_model,
    predict_popularity,
)
from src.train_popularity_model import train_popularity_model


def _make_training_csv(path: Path, n: int = 80) -> Path:
    rng = np.random.default_rng(42)
    languages = ["Hindi", "Tamil", "English", "Unknown"]
    rows = []
    for i in range(n):
        energy = float(rng.uniform(0.1, 0.95))
        dance = float(rng.uniform(0.1, 0.95))
        # Weak synthetic signal so the model can learn something
        popularity = float(np.clip(10 + 40 * energy + 30 * dance + rng.normal(0, 5), 0, 100))
        rows.append(
            {
                "track_id": f"tid_{i}",
                "track_name": f"Track {i}",
                "artist_name": f"Artist {i % 12}",
                "year": int(rng.integers(2000, 2025)),
                "popularity": popularity,
                "artwork_url": "",
                "album_name": f"Album {i % 8}",
                "acousticness": float(rng.random()),
                "danceability": dance,
                "duration_ms": float(rng.integers(120000, 320000)),
                "energy": energy,
                "instrumentalness": float(rng.random() * 0.2),
                "key": int(rng.integers(0, 12)),
                "liveness": float(rng.random() * 0.4),
                "loudness": float(rng.uniform(-20, -3)),
                "mode": int(rng.integers(0, 2)),
                "speechiness": float(rng.random() * 0.3),
                "tempo": float(rng.uniform(70, 160)),
                "time_signature": 4,
                "valence": float(rng.random()),
                "track_url": "",
                "language": languages[i % len(languages)],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def trained_artifacts(tmp_path: Path):
    csv_path = _make_training_csv(tmp_path / "indian_songs.csv")
    model_path = tmp_path / "popularity_regressor.joblib"
    metrics_path = tmp_path / "popularity_metrics.json"
    result = train_popularity_model(
        csv_path=csv_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )
    return result


def test_training_on_synthetic_fixture(trained_artifacts):
    assert trained_artifacts["model_path"].exists()
    assert trained_artifacts["metrics_path"].exists()
    metrics = trained_artifacts["metrics"]["random_forest_random_split"]
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_prediction_range_and_numeric(trained_artifacts):
    payload = load_popularity_model(trained_artifacts["model_path"])
    features = default_feature_values()
    result = predict_popularity(
        language="Hindi",
        year=2020,
        features=features,
        model_payload=payload,
    )
    assert isinstance(result["prediction"], float)
    assert 0.0 <= result["prediction"] <= 100.0


def test_unseen_language_handled(trained_artifacts):
    payload = load_popularity_model(trained_artifacts["model_path"])
    result = predict_popularity(
        language="Klingon",  # deliberately unseen
        year=2019,
        features=default_feature_values(),
        model_payload=payload,
    )
    assert result["unseen_language"] is True
    assert 0.0 <= result["prediction"] <= 100.0


def test_missing_feature_raises(trained_artifacts):
    payload = load_popularity_model(trained_artifacts["model_path"])
    incomplete = {k: 0.5 for k in NUMERIC_AUDIO_FEATURES if k != "energy"}
    with pytest.raises(ValueError):
        predict_popularity(
            language="Tamil",
            year=2021,
            features=incomplete,
            model_payload=payload,
        )


def test_missing_model_file_behavior(tmp_path: Path):
    missing = tmp_path / "does_not_exist.joblib"
    with pytest.raises(ModelNotFoundError):
        load_popularity_model(missing)


def test_clean_drops_out_of_range_popularity_rows():
    from tests.test_data_utils import _synthetic_rows

    frame = _synthetic_rows(4)
    frame.loc[0, "popularity"] = 250
    cleaned = clean_indian_songs(frame)
    assert (cleaned["popularity"] <= 100).all()
    assert 250 not in set(cleaned["popularity"].tolist())
