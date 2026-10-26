"""Data loading and preparation utilities for TuneWise AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

_dataset_path = os.environ.get("TUNEWISE_DATASET_PATH")
if _dataset_path is None:
    INDIAN_SONGS_PATH = DATA_DIR / "indian_songs.csv"
else:
    _dataset_path = Path(_dataset_path)
    INDIAN_SONGS_PATH = (
        _dataset_path if _dataset_path.is_absolute() else PROJECT_ROOT / _dataset_path
    )
GENRE_DATASET_PATH = DATA_DIR / "dataset.csv"
POPULARITY_MODEL_PATH = MODELS_DIR / "popularity_regressor.joblib"
POPULARITY_METRICS_PATH = MODELS_DIR / "popularity_metrics.json"
GENRE_MODEL_PATH = MODELS_DIR / "genre_classifier.joblib"

REQUIRED_INDIAN_COLUMNS = [
    "track_id",
    "track_name",
    "artist_name",
    "year",
    "popularity",
    "artwork_url",
    "album_name",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "valence",
    "track_url",
    "language",
]

NUMERIC_AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "valence",
]

REGRESSION_NUMERIC_FEATURES = NUMERIC_AUDIO_FEATURES + ["year"]
TARGET_COLUMN = "popularity"
LANGUAGE_COLUMN = "language"

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "acousticness": (0.0, 1.0),
    "danceability": (0.0, 1.0),
    "duration_ms": (1_000.0, 1_800_000.0),
    "energy": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "key": (0.0, 11.0),
    "liveness": (0.0, 1.0),
    "loudness": (-60.0, 5.0),
    "mode": (0.0, 1.0),
    "speechiness": (0.0, 1.0),
    "tempo": (30.0, 250.0),
    "time_signature": (1.0, 7.0),
    "valence": (0.0, 1.0),
    "year": (1900.0, 2100.0),
    "popularity": (0.0, 100.0),
}


class DatasetError(Exception):
    """Raised when the Indian songs dataset cannot be used."""


def indian_songs_missing_message() -> str:
    return (
        "The primary dataset was not found at `data/indian_songs.csv`. "
        "Obtain the dataset separately, place the CSV at that path, then reload "
        "the app or re-run training."
    )


def validate_required_columns(df: pd.DataFrame, required: list[str] | None = None) -> list[str]:
    """Return a list of missing required column names."""
    needed = required or REQUIRED_INDIAN_COLUMNS
    return [col for col in needed if col not in df.columns]


def load_indian_songs(
    path: Path | str | None = None,
    *,
    require_exists: bool = True,
) -> pd.DataFrame:
    """Load the Indian songs CSV without mutating the original file."""
    csv_path = Path(path) if path is not None else INDIAN_SONGS_PATH

    if not csv_path.exists():
        if require_exists:
            raise DatasetError(indian_songs_missing_message())
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001 - surface useful CSV errors
        raise DatasetError(f"Could not read `{csv_path.as_posix()}`: {exc}") from exc

    missing = validate_required_columns(df)
    if missing:
        raise DatasetError(
            "The Indian songs CSV is missing required columns: "
            + ", ".join(missing)
        )
    if df.empty:
        raise DatasetError("The Indian songs CSV contains no song rows.")

    return df


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_indian_songs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Indian songs dataset for exploration and modelling.

    - Drops duplicate track IDs (keeps first occurrence)
    - Coerces numeric columns
    - Fills blank languages as Unknown
    - Drops rows with invalid/missing modelling fields
    - Clips popularity into [0, 100]
    """
    if df.empty:
        return df.copy()

    missing = validate_required_columns(df)
    if missing:
        raise DatasetError(
            "Cannot clean dataset; missing columns: " + ", ".join(missing)
        )

    out = df.copy()
    out = out.drop_duplicates(subset=["track_id"], keep="first")

    numeric_cols = REGRESSION_NUMERIC_FEATURES + [TARGET_COLUMN]
    out = coerce_numeric_columns(out, numeric_cols)

    out[LANGUAGE_COLUMN] = (
        out[LANGUAGE_COLUMN]
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )
    out.loc[out[LANGUAGE_COLUMN].isna(), LANGUAGE_COLUMN] = "Unknown"

    for text_col in ("track_name", "artist_name", "album_name"):
        out[text_col] = out[text_col].fillna("Unknown").astype(str)

    for url_col in ("artwork_url", "track_url"):
        out[url_col] = out[url_col].map(
            lambda value: "" if pd.isna(value) else str(value)
        )

    modelling_cols = REGRESSION_NUMERIC_FEATURES + [TARGET_COLUMN, LANGUAGE_COLUMN]
    out = out.dropna(subset=modelling_cols)

    for column, (minimum, maximum) in FEATURE_RANGES.items():
        if column in out.columns:
            out = out[out[column].between(minimum, maximum)]

    out["year"] = out["year"].astype(int)

    out = out.reset_index(drop=True)
    return out


def available_languages(df: pd.DataFrame) -> list[str]:
    """Return sorted unique languages present in the dataset."""
    if df.empty or LANGUAGE_COLUMN not in df.columns:
        return []
    langs = (
        df[LANGUAGE_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
        .unique()
        .tolist()
    )
    return sorted(langs, key=lambda x: (x == "Unknown", x.lower()))


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact summary dictionary for dashboards."""
    if df.empty:
        return {
            "rows": 0,
            "languages": {},
            "popularity_mean": None,
            "popularity_median": None,
            "year_min": None,
            "year_max": None,
        }

    return {
        "rows": int(len(df)),
        "languages": {
            str(k): int(v)
            for k, v in df[LANGUAGE_COLUMN].value_counts().items()
        },
        "popularity_mean": float(df[TARGET_COLUMN].mean()),
        "popularity_median": float(df[TARGET_COLUMN].median()),
        "year_min": int(df["year"].min()) if "year" in df.columns else None,
        "year_max": int(df["year"].max()) if "year" in df.columns else None,
    }


def prepare_regression_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only columns needed for popularity regression."""
    cleaned = clean_indian_songs(df)
    cols = REGRESSION_NUMERIC_FEATURES + [LANGUAGE_COLUMN, TARGET_COLUMN, "artist_name", "track_id"]
    present = [c for c in cols if c in cleaned.columns]
    return cleaned[present].copy()


def build_feature_row(
    *,
    language: str,
    year: int | float,
    features: dict[str, float],
) -> pd.DataFrame:
    """Build a single-row feature frame for prediction."""
    row: dict[str, Any] = {"language": language, "year": year}
    for name in NUMERIC_AUDIO_FEATURES:
        if name not in features:
            raise ValueError(f"Missing required feature: {name}")
        row[name] = features[name]
    return pd.DataFrame([row])


def is_valid_http_url(value: object) -> bool:
    """Return True when value looks like a usable http(s) URL."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text.startswith("http://") or text.startswith("https://")
