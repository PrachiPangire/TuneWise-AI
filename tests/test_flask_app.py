from __future__ import annotations

import io

from app import app


client = app.test_client()


def test_health_endpoint():
    response = client.get('/api/health')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert 'moods' in payload


def test_search_endpoint_returns_rows():
    response = client.get('/api/search?q=Leo&artist=Anirudh')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['count'] > 0
    assert 'results' in payload


def test_predict_mood_endpoint():
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


def test_recommendations_endpoint():
    response = client.get('/api/recommendations?mood=Happy')
    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert len(body['results']) > 0
    assert body['results'][0]['mood'] == 'Happy'
    assert 'Positive valence' in body['results'][0]['mood_reason']


def test_playlist_analysis_endpoint(tmp_path):
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
