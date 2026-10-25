from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from src.data_utils import (
    NUMERIC_AUDIO_FEATURES,
    clean_indian_songs,
    load_indian_songs,
    LANGUAGE_COLUMN,
)

MOOD_LABELS = [
    "Focus / Study",
    "Gym Energy",
    "Late Night Chill",
    "Happy",
    "Sad / Reflective",
    "Romantic",
]

MOOD_ALIASES = {
    "focus": "Focus / Study",
    "study": "Focus / Study",
    "focus / study": "Focus / Study",
    "gym": "Gym Energy",
    "gym energy": "Gym Energy",
    "workout": "Gym Energy",
    "late night chill": "Late Night Chill",
    "chill": "Late Night Chill",
    "night": "Late Night Chill",
    "happy": "Happy",
    "sad": "Sad / Reflective",
    "reflective": "Sad / Reflective",
    "sad / reflective": "Sad / Reflective",
    "romantic": "Romantic",
}


def normalize_mood(mood: str | None) -> str | None:
    if not mood:
        return None
    key = str(mood).strip().lower()
    return MOOD_ALIASES.get(key, mood.strip())


@lru_cache(maxsize=1)
def get_clean_dataset() -> pd.DataFrame:
    df = load_indian_songs()
    return clean_indian_songs(df)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def row_to_feature_dict(row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for key in NUMERIC_AUDIO_FEATURES:
        features[key] = _safe_float(row.get(key), 0.0)
    if "year" in row:
        features["year"] = _safe_float(row.get("year"), 2020.0)
    return features


def _bounded(value: float, lowest: float = 0.0, highest: float = 1.0) -> float:
    if highest == lowest:
        return 0.0
    return max(0.0, min(1.0, (value - lowest) / (highest - lowest)))


def _tempo_score(tempo: float) -> float:
    return _bounded(tempo, 60.0, 180.0)


def _loudness_score(loudness: float) -> float:
    return _bounded(loudness, -25.0, 0.0)


def _mood_score(mood: str, features: dict[str, float]) -> float:
    energy = float(features.get("energy", 0.5))
    dance = float(features.get("danceability", 0.5))
    acoustic = float(features.get("acousticness", 0.5))
    valence = float(features.get("valence", 0.5))
    tempo = float(features.get("tempo", 110.0))
    speechiness = float(features.get("speechiness", 0.1))
    instrumentalness = float(features.get("instrumentalness", 0.0))
    loudness = float(features.get("loudness", -10.0))
    liveness = float(features.get("liveness", 0.1))

    focus = (
        0.35 * (1.0 - abs(energy - 0.6))
        + 0.20 * (1.0 - min(abs(valence - 0.5), 1.0))
        + 0.20 * acoustic
        + 0.15 * instrumentalness
        + 0.10 * (1.0 - speechiness)
    )
    gym = (
        0.40 * energy
        + 0.25 * dance
        + 0.20 * _tempo_score(tempo)
        + 0.10 * _loudness_score(loudness)
        + 0.05 * valence
    )
    chill = (
        0.35 * (1.0 - energy)
        + 0.25 * acoustic
        + 0.20 * (1.0 - dance)
        + 0.10 * (1.0 - _tempo_score(tempo))
        + 0.10 * (1.0 - liveness)
    )
    happy = (
        0.30 * valence
        + 0.25 * dance
        + 0.20 * energy
        + 0.15 * _tempo_score(tempo)
        + 0.10 * (1.0 - acoustic)
    )
    sad = (
        0.35 * (1.0 - valence)
        + 0.25 * acoustic
        + 0.20 * (1.0 - energy)
        + 0.10 * (1.0 - dance)
        + 0.10 * (1.0 - _tempo_score(tempo))
    )
    romantic = (
        0.30 * valence
        + 0.25 * acoustic
        + 0.20 * (1.0 - energy)
        + 0.15 * dance
        + 0.10 * instrumentalness
    )

    scores = {
        "Focus / Study": focus,
        "Gym Energy": gym,
        "Late Night Chill": chill,
        "Happy": happy,
        "Sad / Reflective": sad,
        "Romantic": romantic,
    }
    return float(scores.get(mood, 0.0))


def predict_mood_for_features(features: dict[str, float]) -> dict[str, Any]:
    scores = {
        mood: _mood_score(mood, features)
        for mood in MOOD_LABELS
    }
    predicted = max(MOOD_LABELS, key=lambda mood: scores[mood])
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_mood, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = max(0.0, top_score - second_score)
    confidence = round(min(1.0, max(0.0, margin / 1.5)), 3)
    reason = _mood_reason(predicted, features)
    explanation = _mood_explanation(predicted, features)
    return {
        "predicted_mood": predicted,
        "top_mood": top_mood,
        "scores": {mood: round(score, 4) for mood, score in scores.items()},
        "confidence": confidence,
        "rule_strength": confidence,
        "reason": reason,
        "explanation": explanation,
    }


def _mood_reason(mood: str, features: dict[str, float]) -> str:
    energy = float(features.get("energy", 0.5))
    dance = float(features.get("danceability", 0.5))
    acoustic = float(features.get("acousticness", 0.5))
    valence = float(features.get("valence", 0.5))
    tempo = float(features.get("tempo", 110.0))

    if mood == "Focus / Study":
        return f"Balanced energy ({energy:.2f}), moderate acousticness ({acoustic:.2f}), and steady tempo ({tempo:.0f} BPM) support a study-friendly track."
    if mood == "Gym Energy":
        return f"High energy ({energy:.2f}), strong danceability ({dance:.2f}), and brisk tempo ({tempo:.0f} BPM) fit an active workout vibe."
    if mood == "Late Night Chill":
        return f"Low-to-moderate energy ({energy:.2f}), softer acoustic feel ({acoustic:.2f}), and lower tempo ({tempo:.0f} BPM) suit a calm night setting."
    if mood == "Happy":
        return f"Positive valence ({valence:.2f}) and upbeat rhythm ({dance:.2f} danceability) suggest an uplifting mood."
    if mood == "Sad / Reflective":
        return f"Lower valence ({valence:.2f}) and more acoustic texture ({acoustic:.2f}) suggest a reflective or melancholy feel."
    if mood == "Romantic":
        return f"Gentle energy ({energy:.2f}), warm valence ({valence:.2f}), and acoustic softness ({acoustic:.2f}) align with romantic listening."
    return "Audio features are being compared against the available mood score rules."


def describe_mood_for_features(mood: str, features: dict[str, float]) -> tuple[str, str]:
    return _mood_reason(mood, features), _mood_explanation(mood, features)


def _mood_explanation(mood: str, features: dict[str, float]) -> str:
    energy = float(features.get("energy", 0.5))
    dance = float(features.get("danceability", 0.5))
    acoustic = float(features.get("acousticness", 0.5))
    valence = float(features.get("valence", 0.5))
    tempo = float(features.get("tempo", 110.0))
    if mood == "Focus / Study":
        return "This rule-based score prefers moderate energy, low speechiness, and a stable rhythm so the song feels productive rather than distracting."
    if mood == "Gym Energy":
        return "This rule-based score favors high energy, stronger danceability, and a faster beat to match an athletic or motivating playlist."
    if mood == "Late Night Chill":
        return "This rule-based score favors lower energy, acoustic texture, and lower motion so the song feels relaxed at night."
    if mood == "Happy":
        return "This rule-based score favors high valence and a lively rhythmic feel, which usually matches cheerful listening."
    if mood == "Sad / Reflective":
        return "This rule-based score favors lower valence and a calmer, more acoustic texture, which often suits introspective listening."
    if mood == "Romantic":
        return "This rule-based score favors warm valence, acoustic softness, and a smooth rhythm that fits intimate or affectionate listening."
    return "Mood labels are inferred from audio-feature patterns rather than a trained model."


def search_dataset(query: str, *, artist: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    q = (query or "").strip()
    artist_q = (artist or "").strip()
    df = get_clean_dataset()
    if df.empty:
        return []

    mask = pd.Series(True, index=df.index)
    if q:
        search_terms = q.lower()
        mask &= df["track_name"].fillna("").astype(str).str.lower().str.contains(search_terms, case=False, na=False) | df["artist_name"].fillna("").astype(str).str.lower().str.contains(search_terms, case=False, na=False)
    if artist_q:
        artist_terms = artist_q.lower()
        mask &= df["artist_name"].fillna("").astype(str).str.lower().str.contains(artist_terms, case=False, na=False)

    hits = df.loc[mask].head(limit).copy()
    rows: list[dict[str, Any]] = []
    for _, song in hits.iterrows():
        features = row_to_feature_dict(song.to_dict())
        mood_result = predict_mood_for_features(features)
        rows.append(
            {
                "track_id": song.get("track_id", ""),
                "track_name": song.get("track_name", "Unknown"),
                "artist_name": song.get("artist_name", "Unknown"),
                "album_name": song.get("album_name", "Unknown"),
                "year": int(song.get("year", 0) or 0),
                "language": song.get(LANGUAGE_COLUMN, "Unknown"),
                "popularity": float(song.get("popularity", 0.0) or 0.0),
                "artwork_url": song.get("artwork_url", ""),
                "track_url": song.get("track_url", ""),
                "mood": mood_result["predicted_mood"],
                "mood_reason": mood_result["reason"],
                "audio_features": {
                    key: round(float(features.get(key, 0.0)), 4)
                    for key in [
                        "acousticness",
                        "danceability",
                        "energy",
                        "valence",
                        "tempo",
                        "loudness",
                    ]
                    if key in features
                },
            }
        )
    return rows


def zip_row_to_features(row: dict[str, Any]) -> dict[str, float]:
    row_dict = {str(k).strip(): v for k, v in row.items() if k is not None}
    features: dict[str, float] = {}
    for key in NUMERIC_AUDIO_FEATURES:
        if key in row_dict:
            features[key] = _safe_float(row_dict.get(key), 0.0)
    if not features:
        for candidate in ["acousticness", "danceability", "energy", "valence", "tempo", "loudness"]:
            if candidate in row_dict:
                features[candidate] = _safe_float(row_dict.get(candidate), 0.0)
    return features


def get_recommendations_for_mood(mood: str, *, limit: int = 8) -> list[dict[str, Any]]:
    normalized = normalize_mood(mood)
    if normalized not in MOOD_LABELS:
        raise ValueError(f"Unsupported mood: {mood}")

    df = get_clean_dataset()
    rankings: list[dict[str, Any]] = []
    for _, song in df.iterrows():
        features = row_to_feature_dict(song.to_dict())
        mood_result = predict_mood_for_features(features)
        score = float(mood_result["scores"].get(normalized, 0.0))
        if score <= 0.0:
            continue
        reason, explanation = describe_mood_for_features(normalized, features)
        rankings.append(
            {
                "track_id": song.get("track_id", ""),
                "track_name": song.get("track_name", "Unknown"),
                "artist_name": song.get("artist_name", "Unknown"),
                "album_name": song.get("album_name", "Unknown"),
                "year": int(song.get("year", 0) or 0),
                "language": song.get(LANGUAGE_COLUMN, "Unknown"),
                "popularity": float(song.get("popularity", 0.0) or 0.0),
                "artwork_url": song.get("artwork_url", ""),
                "track_url": song.get("track_url", ""),
                "mood": normalized,
                "mood_score": round(score, 4),
                "mood_reason": reason,
                "mood_explanation": explanation,
                "audio_features": {
                    key: round(float(features.get(key, 0.0)), 4)
                    for key in [
                        "acousticness",
                        "danceability",
                        "energy",
                        "valence",
                        "tempo",
                        "loudness",
                    ]
                    if key in features
                },
            }
        )

    rankings.sort(key=lambda item: item["mood_score"], reverse=True)
    return rankings[:limit]


def analyze_playlist_csv(file_obj: Any) -> dict[str, Any]:
    if file_obj is None:
        raise ValueError("A CSV file is required.")

    file_name = getattr(file_obj, "filename", "uploaded.csv")
    if not str(file_name).lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    try:
        df = pd.read_csv(file_obj, low_memory=False)
    except Exception as exc:  # pragma: no cover - surfaced to API layer
        raise ValueError(f"Unable to read the CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    preview = df.head(5).to_dict(orient="records")
    columns = list(df.columns)
    title_candidates = [
        "track_name",
        "song",
        "title",
        "track",
        "name",
    ]
    artist_candidates = [
        "artist_name",
        "artist",
        "artists",
        "main_artist",
    ]

    title_col = next((col for col in title_candidates if col in df.columns), None)
    artist_col = next((col for col in artist_candidates if col in df.columns), None)

    results: list[dict[str, Any]] = []
    distribution = {mood: 0 for mood in MOOD_LABELS}
    for _, row in df.iterrows():
        features = zip_row_to_features(row.to_dict())
        if not features:
            continue
        mood_result = predict_mood_for_features(features)
        song_name = str(row[title_col]).strip() if title_col and title_col in row and not pd.isna(row.get(title_col)) else "Unknown track"
        artist_name = str(row[artist_col]).strip() if artist_col and artist_col in row and not pd.isna(row.get(artist_col)) else "Unknown artist"
        mood = mood_result["predicted_mood"]
        distribution[mood] += 1
        results.append(
            {
                "track_name": song_name,
                "artist_name": artist_name,
                "predicted_mood": mood,
                "mood_reason": mood_result["reason"],
                "audio_features": {
                    key: round(float(features.get(key, 0.0)), 4)
                    for key in [
                        "acousticness",
                        "danceability",
                        "energy",
                        "valence",
                        "tempo",
                        "loudness",
                    ]
                    if key in features
                },
            }
        )

    total_rows = int(len(df))
    eligible_rows = int(len(results))
    csv_blob = pd.DataFrame(results).to_csv(index=False)
    return {
        "file_name": file_name,
        "rows": total_rows,
        "eligible_rows": eligible_rows,
        "columns": columns,
        "preview": preview,
        "title_column": title_col,
        "artist_column": artist_col,
        "mood_distribution": distribution,
        "results": results,
        "csv_download": csv_blob,
    }


__all__ = [
    "MOOD_LABELS",
    "MOOD_ALIASES",
    "get_clean_dataset",
    "normalize_mood",
    "search_dataset",
    "predict_mood_for_features",
    "get_recommendations_for_mood",
    "analyze_playlist_csv",
]
