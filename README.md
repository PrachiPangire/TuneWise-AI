# TuneWise-AI

TuneWise-AI is a Flask web app for exploring songs in a local music dataset. Search tracks, get mood-based recommendations, or upload a playlist CSV for analysis.
Mood matching uses transparent audio-feature rules; it is not a trained mood classifier, and the app does not connect to Spotify.

## Main features

- Search the local song dataset by title or artist.
- Explore recommendations for six moods: Focus / Study, Gym Energy, Late Night Chill, Happy, Sad / Reflective, and Romantic.
- Upload a playlist CSV to view inferred moods, track details, and a mood distribution; download the analysis as CSV.

## Technologies used

- Python and Flask for the web app and API.
- HTML, CSS, and JavaScript for the browser interface.
- pandas and NumPy for dataset and playlist processing.
- scikit-learn and joblib for separate genre-classification and popularity-regression workflows; these do not power mood matching.
- pytest, Docker, and Jenkins for testing and project automation.

## Screenshots

Captured from the running Flask app using the current local dataset.

**Homepage**

![TuneWise-AI homepage](screenshots/tunewise-home.png)

**Song search results**

![Song search results](screenshots/tunewise-search-results.png)

**Mood recommendations**

![Mood recommendations](screenshots/tunewise-mood-recommendations.png)

**Playlist analysis**

![Playlist analysis](screenshots/tunewise-playlist-analysis.png)

## How to run locally

Python 3.10+ is required. The local song dataset is not included in Git; place `indian_songs.csv` in the `data` folder before starting the app.

In Windows PowerShell:

```powershell
Set-Location 'E:\TuneWise-AI'
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000). The health endpoint is [http://localhost:5000/api/health](http://localhost:5000/api/health).

## Docker setup

With Docker Desktop running and `data\indian_songs.csv` available:

```powershell
docker build -t tunewise-ai .
docker run --rm -p 5000:5000 -v "${PWD}\data:/app/data:ro" tunewise-ai
```

The container exposes port `5000` and checks `/api/health`. Docker is installed in the current environment, but its engine was unavailable, so the image and container have not been verified here.

## Jenkins pipeline

The `Jenkinsfile` expects a Windows agent with a pre-provisioned project checkout and virtual environment. Set `PROJECT_DIR` and `VENV_PYTHON` to their paths in the Jenkinsfile. Its stages check the workspace, verify Python and dependencies, compile the application modules, run pytest, and report local dataset/model assets.

The supplied screenshots show Jenkins build #3 marked successful. The console screenshot shows three passing tests; it does not show the full pipeline output or identify the source revision, so confirm the run used the current `Jenkinsfile` before treating it as current CI verification.

![Jenkins build #3 console output](screenshots/jenkins-build-3-console.png)

![Jenkins build #3 successful status](screenshots/jenkins-build-3-success.png)

## Testing

Run the test suite from the project directory:

```powershell
python -m pytest -q
```

Latest result: **20 passed**.

## Author

Prachi Pangire
