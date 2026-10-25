"""Model loading and prediction helpers for TuneWise AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_utils import (
    GENRE_MODEL_PATH,
    NUMERIC_AUDIO_FEATURES,
    POPULARITY_METRICS_PATH,
    POPULARITY_MODEL_PATH,
    REGRESSION_NUMERIC_FEATURES,
    build_feature_row,
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when a required model artifact is missing."""


def popularity_model_exists(path: Path | str | None = None) -> bool:
    return Path(path or POPULARITY_MODEL_PATH).exists()


def genre_model_exists(path: Path | str | None = None) -> bool:
    return Path(path or GENRE_MODEL_PATH).exists()


def load_popularity_model(path: Path | str | None = None) -> dict[str, Any]:
    """Load the saved popularity regressor payload."""
    model_path = Path(path or POPULARITY_MODEL_PATH)
    if not model_path.exists():
        raise ModelNotFoundError(
            "Popularity model not found at "
            f"`{model_path.as_posix()}`. Train it with "
            "`python src/train_popularity_model.py`."
        )

    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError("Unexpected popularity model format.")
    return payload


def load_popularity_metrics(path: Path | str | None = None) -> dict[str, Any] | None:
    """Load evaluation metrics JSON if available."""
    metrics_path = Path(path or POPULARITY_METRICS_PATH)
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def predict_popularity(
    *,
    language: str,
    year: int | float,
    features: dict[str, float],
    model_payload: dict[str, Any] | None = None,
    clip: bool = True,
) -> dict[str, Any]:
    """
    Predict popularity for one song feature vector.

    Returns a dict with prediction and metadata. Never fabricates values when
    the model is unavailable — callers should catch ModelNotFoundError.
    """
    payload = model_payload or load_popularity_model()
    pipeline = payload["pipeline"]

    missing = [name for name in NUMERIC_AUDIO_FEATURES if name not in features]
    if missing:
        raise ValueError(f"Missing audio features: {', '.join(missing)}")

    row = build_feature_row(language=language, year=year, features=features)
    # Ensure column order matches training expectations when possible
    expected = payload.get("feature_names")
    if expected:
        row = row.reindex(columns=expected, fill_value=np.nan)

    raw_pred = float(pipeline.predict(row)[0])
    score = float(np.clip(raw_pred, 0, 100)) if clip else raw_pred

    return {
        "prediction": score,
        "raw_prediction": raw_pred,
        "language": language,
        "year": year,
        "is_estimate": True,
        "languages_seen": payload.get("languages_seen", []),
        "unseen_language": language not in set(payload.get("languages_seen", [])),
    }


def load_genre_model(path: Path | str | None = None) -> dict[str, Any]:
    """Load the preserved Spotify genre classifier payload."""
    model_path = Path(path or GENRE_MODEL_PATH)
    if not model_path.exists():
        raise ModelNotFoundError(
            "Genre classifier not found at "
            f"`{model_path.as_posix()}`. The original model file was preserved "
            "separately and was not overwritten by popularity training."
        )

    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Unexpected genre model format.")
    return payload


def predict_genre(
    feature_values: dict[str, float],
    model_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict a music genre from numeric Spotify-style features."""
    payload = model_payload or load_genre_model()
    model = payload["model"]
    feature_names = payload.get("features") or []

    if not feature_names:
        raise ValueError("Genre model does not include a feature list.")

    missing = [name for name in feature_names if name not in feature_values]
    if missing:
        raise ValueError(f"Missing genre features: {', '.join(missing)}")

    row = pd.DataFrame(
        [[feature_values[name] for name in feature_names]],
        columns=feature_names,
    )
    prediction = model.predict(row)[0]

    result: dict[str, Any] = {
        "prediction": str(prediction),
        "features_used": feature_names,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        scores = {
            str(label): float(score)
            for label, score in zip(model.classes_, probabilities)
        }
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:8]
        result["top_probabilities"] = top

    return result


def default_feature_values() -> dict[str, float]:
    """Sensible mid-range defaults for the prediction form."""
    return {
        "acousticness": 0.25,
        "danceability": 0.65,
        "duration_ms": 210000.0,
        "energy": 0.70,
        "instrumentalness": 0.0,
        "key": 5.0,
        "liveness": 0.15,
        "loudness": -7.0,
        "mode": 1.0,
        "speechiness": 0.08,
        "tempo": 110.0,
        "time_signature": 4.0,
        "valence": 0.55,
    }


def regression_feature_help() -> dict[str, str]:
    """Short beginner-friendly explanations for UI tooltips."""
    return {
        "acousticness": "How acoustic the track sounds (0 = electronic, 1 = acoustic).",
        "danceability": "How suitable the track is for dancing (0–1).",
        "duration_ms": "Track length in milliseconds.",
        "energy": "Perceived intensity and activity (0–1).",
        "instrumentalness": "Likelihood the track has little or no vocals (0–1).",
        "key": "Musical key encoded as an integer from 0 to 11.",
        "liveness": "Likelihood of a live audience presence (0–1).",
        "loudness": "Overall loudness in decibels (typically negative).",
        "mode": "Major (1) or minor (0) modality.",
        "speechiness": "Presence of spoken words (0–1).",
        "tempo": "Estimated beats per minute.",
        "time_signature": "Notational time signature (e.g. 4).",
        "valence": "Musical positivity (0 = sad/dark, 1 = happy/upbeat).",
        "year": "Release year used as a numeric context feature.",
        "language": "Language label from the dataset (Unknown kept as Unknown).",
    }


# Re-export for convenience
__all__ = [
    "ModelNotFoundError",
    "default_feature_values",
    "genre_model_exists",
    "load_genre_model",
    "load_popularity_metrics",
    "load_popularity_model",
    "popularity_model_exists",
    "predict_genre",
    "predict_popularity",
    "regression_feature_help",
    "NUMERIC_AUDIO_FEATURES",
    "REGRESSION_NUMERIC_FEATURES",
]
