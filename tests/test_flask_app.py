from __future__ import annotations

import io

import pandas as pd
import pytest

import app as app_module
import src.mood_logic as mood_logic
from src.data_utils import DatasetError


@pytest.fixture()
def client(monkeypatch):
    rows = []
    for index in range(12):
        rows.append(
            {
                "track_id": f"id_{index}",
                "track_name": f"Song {index}",
                "artist_name": "Anirudh" if index < 4 else f"Artist {index}",
                "album_name": f"Album {index}",
                "year": 2020,
                "popularity": 50,
                "language": "Tamil",
                "artwork_url": "",
                "track_url": "",
                "acousticness": 0.2 + (index % 3) * 0.1,
                "danceability": 0.5 + (index % 4) * 0.1,
                "duration_ms": 180000,
                "energy": 0.4 + (index % 5) * 0.1,
                "instrumentalness": 0.0,
                "key": index % 12,
                "liveness": 0.1,
                "loudness": -8.0,
                "mode": index % 2,
                "speechiness": 0.05,
                "tempo": 90 + index * 3,
                "time_signature": 4,
                "valence": 0.3 + (index % 5) * 0.1,
            }
        )
    dataset = pd.DataFrame(rows)
    monkeypatch.setattr(app_module, "get_clean_dataset", lambda: dataset)
    monkeypatch.setattr(mood_logic, "get_clean_dataset", lambda: dataset)
    return app_module.app.test_client()


def test_homepage_serves_interface(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"TuneWise-AI" in response.data


def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert 'moods' in payload


def test_search_endpoint_returns_rows(client):
    response = client.get('/api/search?q=Song&artist=Anirudh')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['count'] > 0
    assert 'results' in payload


def test_search_requires_title_or_artist(client):
    response = client.get('/api/search')
    assert response.status_code == 400
    assert response.get_json()['success'] is False


def test_predict_mood_endpoint(client):
    payload = {
        'song': {'track_name': 'Test Song', 'artist_name': 'Demo Artist'},
        'features': {
            'energy': 0.8,
            'danceability': 0.75,
            'acousticness': 0.2,
            'valence': 0.8,
            'tempo': 128,
            'loudness': -6.5,
        },
    }
    response = client.post('/api/predict-mood', json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body['predicted_mood'] in {
        'Focus / Study',
        'Gym Energy',
        'Late Night Chill',
        'Happy',
        'Sad / Reflective',
        'Romantic',
    }


def test_predict_mood_rejects_missing_features(client):
    response = client.post('/api/predict-mood', json={'features': {}})
    assert response.status_code == 400
    assert 'audio features' in response.get_json()['error']


def test_recommendations_endpoint(client):
    response = client.get('/api/recommendations?mood=Happy')
    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert len(body['results']) > 0
    assert body['results'][0]['mood'] == 'Happy'
    assert 'Positive valence' in body['results'][0]['mood_reason']


def test_recommendations_reject_invalid_mood(client):
    response = client.get('/api/recommendations?mood=Unknown')
    assert response.status_code == 400
    assert response.get_json()['success'] is False


def test_playlist_analysis_endpoint(client):
    csv_rows = "track_name,artist_name,acousticness,danceability,energy,valence,tempo,loudness\nTune 1,Artist 1,0.2,0.78,0.82,0.8,120,-7.0\nTune 2,Artist 2,0.7,0.4,0.3,0.2,86,-12.0\n"
    data = io.BytesIO(csv_rows.encode('utf-8'))
    response = client.post(
        '/api/analyze-playlist',
        data={'file': (data, 'playlist.csv')},
        content_type='multipart/form-data',
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body['eligible_rows'] >= 1
    assert 'csv_download' in body


def test_playlist_analysis_requires_file(client):
    response = client.post('/api/analyze-playlist', data={})
    assert response.status_code == 400
    assert 'CSV file' in response.get_json()['error']


def test_playlist_analysis_rejects_non_csv(client):
    response = client.post(
        '/api/analyze-playlist',
        data={'file': (io.BytesIO(b'not a csv'), 'playlist.txt')},
        content_type='multipart/form-data',
    )
    assert response.status_code == 400
    assert 'Only CSV files' in response.get_json()['error']


def test_empty_dataset_returns_clear_service_error(monkeypatch):
    def missing_dataset():
        raise DatasetError('The Indian songs CSV contains no song rows.')

    monkeypatch.setattr(app_module, 'get_clean_dataset', missing_dataset)
    response = app_module.app.test_client().get('/api/health')
    assert response.status_code == 503
    assert response.get_json() == {
        'success': False,
        'error': 'The Indian songs CSV contains no song rows.',
    }
