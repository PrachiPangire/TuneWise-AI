"""Tests for Indian songs data utilities using synthetic fixtures only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_utils import (
    REQUIRED_INDIAN_COLUMNS,
    DatasetError,
    available_languages,
    clean_indian_songs,
    load_indian_songs,
    prepare_regression_frame,
    validate_required_columns,
)


def _synthetic_rows(n: int = 6) -> pd.DataFrame:
    rows = []
    languages = ["Hindi", "Tamil", "Unknown", "English"]
    for i in range(n):
        rows.append(
            {
                "track_id": f"id_{i % 5}",  # intentional duplicates
                "track_name": f"Song {i}",
                "artist_name": f"Artist {i % 3}",
                "year": 2018 + (i % 5),
                "popularity": (i * 11) % 101,
                "artwork_url": "",
                "album_name": f"Album {i}",
                "acousticness": 0.1 * (i % 10),
                "danceability": 0.2 + 0.05 * i,
                "duration_ms": 180000 + i * 1000,
                "energy": 0.3 + 0.04 * i,
                "instrumentalness": 0.0,
                "key": i % 12,
                "liveness": 0.1,
                "loudness": -8.0 - i,
                "mode": i % 2,
                "speechiness": 0.05,
                "tempo": 100 + i,
                "time_signature": 4,
                "valence": 0.4,
                "track_url": "",
                "language": languages[i % len(languages)],
            }
        )
    return pd.DataFrame(rows)


def test_validate_required_columns_detects_missing():
    df = pd.DataFrame({"track_id": [1]})
    missing = validate_required_columns(df)
    assert "track_name" in missing
    assert "popularity" in missing


def test_clean_removes_duplicate_track_ids():
    df = _synthetic_rows(6)
    cleaned = clean_indian_songs(df)
    assert cleaned["track_id"].duplicated().sum() == 0
    assert len(cleaned) < len(df)


def test_available_languages_dynamic():
    df = clean_indian_songs(_synthetic_rows())
    langs = available_languages(df)
    assert "Hindi" in langs
    assert "Unknown" in langs
    assert "Marathi" not in langs  # not fabricated


def test_prepare_regression_frame_columns():
    frame = prepare_regression_frame(_synthetic_rows())
    assert "popularity" in frame.columns
    assert "language" in frame.columns
    assert "danceability" in frame.columns
    assert "track_name" not in frame.columns


def test_load_indian_songs_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(DatasetError):
        load_indian_songs(missing, require_exists=True)


def test_load_indian_songs_from_fixture(tmp_path: Path):
    path = tmp_path / "indian_songs.csv"
    _synthetic_rows().to_csv(path, index=False)
    loaded = load_indian_songs(path)
    assert set(REQUIRED_INDIAN_COLUMNS).issubset(loaded.columns)
    assert len(loaded) == 6
