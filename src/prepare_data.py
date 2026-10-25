"""
Prepare modelling frames for TuneWise AI.

Popularity regression uses `data/indian_songs.csv`.
The original genre pipeline still uses `data/dataset.csv` via `src/train_model.py`.

Run from the project root:

    python src/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    GENRE_DATASET_PATH,
    INDIAN_SONGS_PATH,
    LANGUAGE_COLUMN,
    REGRESSION_NUMERIC_FEATURES,
    TARGET_COLUMN,
    DatasetError,
    clean_indian_songs,
    load_indian_songs,
    prepare_regression_frame,
)


def prepare_popularity_data():
    print("Preparing popularity regression frame from", INDIAN_SONGS_PATH)
    raw = load_indian_songs(INDIAN_SONGS_PATH)
    cleaned = clean_indian_songs(raw)
    frame = prepare_regression_frame(cleaned)

    print("Raw rows:", len(raw))
    print("Clean rows:", len(cleaned))
    print("Regression frame shape:", frame.shape)
    print("Languages:", sorted(cleaned[LANGUAGE_COLUMN].unique().tolist()))
    print("Feature columns:", REGRESSION_NUMERIC_FEATURES + [LANGUAGE_COLUMN])
    print("Target:", TARGET_COLUMN)
    print("\nSample:")
    print(frame.head())
    return frame


def preview_genre_dataset():
    if not GENRE_DATASET_PATH.exists():
        print("Original genre dataset not found at", GENRE_DATASET_PATH)
        return None

    import pandas as pd

    df = pd.read_csv(GENRE_DATASET_PATH)
    print("\nOriginal genre dataset shape:", df.shape)
    if "track_genre" in df.columns:
        print("Genre count:", df["track_genre"].nunique())
    return df


def main() -> int:
    try:
        prepare_popularity_data()
    except DatasetError as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1

    preview_genre_dataset()
    print("\nData preparation successful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
