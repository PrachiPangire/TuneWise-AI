from __future__ import annotations

import io
from typing import Any

from flask import Flask, jsonify, render_template, request

from src.data_utils import DatasetError
from src.mood_logic import (
    MOOD_LABELS,
    analyze_playlist_csv,
    get_clean_dataset,
    get_recommendations_for_mood,
    normalize_mood,
    predict_mood_for_features,
    search_dataset,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False


def error_response(message: str, status_code: int = 400):
    return jsonify({"success": False, "error": message}), status_code


@app.errorhandler(DatasetError)
def handle_dataset_error(error: DatasetError):
    return error_response(str(error), 503)


def _as_feature_map(raw: Any) -> dict[str, float]:
    if isinstance(raw, dict):
        cleaned: dict[str, float] = {}
        for key, value in raw.items():
            try:
                cleaned[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return cleaned
    return {}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    dataset = get_clean_dataset()
    return jsonify({
        "status": "ok",
        "dataset_rows": int(len(dataset)),
        "moods": MOOD_LABELS,
        "app": "TuneWise-AI",
    })


@app.get("/api/search")
def api_search():
    query = (request.args.get("q") or "").strip()
    artist = (request.args.get("artist") or "").strip()
    if not query and not artist:
        return error_response("Please provide a song title or artist name.", 400)

    results = search_dataset(query, artist=artist, limit=12)
    if not results:
        return jsonify({
            "success": True,
            "results": [],
            "count": 0,
            "message": "No matching songs were found in the local dataset. Try another title, artist, or use the playlist analyzer.",
        })

    return jsonify({
        "success": True,
        "results": results,
        "count": len(results),
    })


@app.post("/api/predict-mood")
def api_predict_mood():
    payload = request.get_json(silent=True) or {}
    features = _as_feature_map(payload.get("features", {}))
    song = payload.get("song") or {}

    if not features and song:
        title = song.get("track_name") or song.get("song") or song.get("title")
        artist = song.get("artist_name") or song.get("artist") or ""
        match_results = search_dataset(title or "", artist=artist, limit=1)
        if match_results:
            features = _as_feature_map(match_results[0].get("audio_features", {}))

    if not features:
        return error_response("No valid audio features were supplied for mood prediction.", 400)

    prediction = predict_mood_for_features(features)
    return jsonify({
        "success": True,
        "predicted_mood": prediction["predicted_mood"],
        "reason": prediction["reason"],
        "explanation": prediction["explanation"],
        "scores": prediction["scores"],
        "feature_summary": {
            key: round(float(value), 4)
            for key, value in sorted(features.items())
            if key in {"acousticness", "danceability", "energy", "valence", "tempo", "loudness"}
        },
    })


@app.get("/api/recommendations")
def api_recommendations():
    mood = normalize_mood(request.args.get("mood") or "")
    if not mood:
        return error_response("A valid mood is required. For example: Happy, Sad / Reflective, or Gym Energy.", 400)

    try:
        results = get_recommendations_for_mood(mood, limit=8)
    except ValueError as exc:
        return error_response(str(exc), 400)

    return jsonify({
        "success": True,
        "mood": mood,
        "results": results,
        "count": len(results),
    })


@app.post("/api/analyze-playlist")
def api_analyze_playlist():
    uploaded = request.files.get("file")
    if uploaded is None:
        return error_response("Please upload a CSV file.", 400)

    if uploaded.filename and not uploaded.filename.lower().endswith(".csv"):
        return error_response("Only CSV files are allowed for playlist analysis.", 400)

    if uploaded.content_length and uploaded.content_length > 5 * 1024 * 1024:
        return error_response("CSV files must be 5 MB or smaller.", 400)

    try:
        result = analyze_playlist_csv(uploaded)
    except ValueError as exc:
        return error_response(str(exc), 400)

    return jsonify({
        "success": True,
        "file_name": result["file_name"],
        "rows": result["rows"],
        "eligible_rows": result["eligible_rows"],
        "columns": result["columns"],
        "preview": result["preview"],
        "title_column": result["title_column"],
        "artist_column": result["artist_column"],
        "mood_distribution": result["mood_distribution"],
        "results": result["results"],
        "csv_download": result["csv_download"],
    })


@app.get("/api/model-info")
def api_model_info():
    metrics = None
    try:
        import json
        from pathlib import Path

        metrics_path = Path("models/popularity_metrics.json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = None

    return jsonify({
        "success": True,
        "method": "This app uses a transparent audio-feature mood scoring system for local songs. The repository does not include a trained mood-classification model, so mood labels are inferred from feature patterns such as energy, valence, danceability, acousticness, tempo, and loudness.",
        "assumptions": [
            "The available dataset contains song audio features but no reliable song-level mood labels.",
            "Mood is estimated from patterns in energy, valence, danceability, acousticness, tempo, and loudness.",
            "This is explainable audio-feature matching, not a personalized recommendation engine.",
            "The local dataset is searched directly and not a live Spotify catalogue lookup.",
        ],
        "moods": MOOD_LABELS,
        "metrics": metrics,
        "notes": [
            "The original genre classifier and popularity regressor remain in the project for their original purposes.",
            "Mood labels are intentionally transparent and easy to audit.",
        ],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
