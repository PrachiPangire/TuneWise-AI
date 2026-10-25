# TuneWise-AI

### Music Discovery and Playlist Analysis Web Application

TuneWise-AI is a Flask web app for searching a local music dataset, exploring six mood-based recommendation categories, and analyzing playlist CSV files. Mood matching uses audio-feature scoring rules, not a trained mood classifier. The app does not use the Spotify API.

## Quick Links

- **GitHub repository:** [PrachiPangire/TuneWise-AI](https://github.com/PrachiPangire/TuneWise-AI)
- **Local website:** [http://localhost:5000](http://localhost:5000)
- **Local health endpoint:** [http://localhost:5000/api/health](http://localhost:5000/api/health)
- **Commit history:** [View commits on `main`](https://github.com/PrachiPangire/TuneWise-AI/commits/main)

The website and health endpoint are local development URLs. No public deployment is verified.

## Table of Contents

- [1. Project Introduction](#1-project-introduction)
- [2. Project Objectives](#2-project-objectives)
- [3. Technologies Used](#3-technologies-used)
- [4. Project Structure](#4-project-structure)
- [5. Dataset Setup](#5-dataset-setup)
- [6. How to Run Locally](#6-how-to-run-locally)
- [7. Git and GitHub Workflow](#7-git-and-github-workflow)
- [8. Docker Setup](#8-docker-setup)
- [9. DevOps Tool Integration - Jenkins](#9-devops-tool-integration---jenkins)
- [10. Testing](#10-testing)
- [11. Screenshot Evidence](#11-screenshot-evidence)
- [12. Limitations and Future Improvements](#12-limitations-and-future-improvements)
- [13. Conclusion](#13-conclusion)
- [14. Author](#14-author)

## 1. Project Introduction

Browsing a large song collection to find music for a particular mood or activity can take time. TuneWise-AI provides a local interface to search songs by title or artist, request recommendations for a chosen mood, and analyze playlist CSV files.

Mood labels are inferred from audio features such as energy, valence, danceability, acousticness, tempo, and loudness. They are transparent rule-based estimates, not predictions from a trained mood-classification model.

## 2. Project Objectives

The project provides:

- Local song search by title or artist.
- Rule-based recommendations for Focus / Study, Gym Energy, Late Night Chill, Happy, Sad / Reflective, and Romantic.
- Playlist CSV analysis with mood distribution and downloadable CSV results.
- Separate training workflows for genre classification and popularity regression.
- A Flask interface with automated tests and Docker/Jenkins configurations for learning and demonstration.

## 3. Technologies Used

### Web application

- Python and Flask for the app and JSON API.
- HTML, CSS, and vanilla JavaScript for the browser interface.

### Data and ML

- pandas and NumPy for dataset cleaning, search, and playlist analysis.
- scikit-learn and joblib for separate genre-classification and popularity-regression workflows.
- The mood recommendation feature is rule-based and is not one of the trained models.
- The application searches local CSV data and does not connect to Spotify's API.

### Testing and DevOps

- pytest for automated tests.
- Dockerfile for container setup.
- Jenkins Declarative Pipeline for dependency installation, syntax checks, tests, and local dataset reporting.
- Git and GitHub for version control.

## 4. Project Structure

```text
TuneWise-AI/
├── app.py                         # Flask routes and application startup
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container image and health check
├── .dockerignore                  # Excludes local datasets and build artifacts
├── .gitignore                     # Excludes local data, models, environments, and caches
├── Jenkinsfile                    # Linux-compatible Jenkins pipeline
├── templates/
│   └── index.html                 # Main web page
├── static/
│   ├── css/style.css              # Interface styling
│   └── js/app.js                  # Search, mood, and playlist interactions
├── src/
│   ├── data_utils.py              # Dataset loading, validation, and cleaning
│   ├── mood_logic.py              # Mood scoring, search, recommendations, playlist analysis
│   ├── prediction.py              # Separate saved-model prediction helpers
│   ├── prepare_data.py            # Data preparation utilities
│   ├── train_model.py             # Genre-classifier training script
│   ├── train_popularity_model.py  # Popularity-regression training script
│   └── visualizations.py          # Plotly visualization utilities
├── tests/                         # pytest tests
├── data/                          # Local datasets; CSV files are not tracked
├── models/                        # Local model artifacts; binaries are not tracked
└── screenshots/                   # App and Jenkins evidence
```

## 5. Dataset Setup

The datasets are intentionally excluded from Git and must be supplied separately. Their original publishers and licenses are not established by the repository, so confirm the source and usage rights before submission or redistribution.

### Song discovery and popularity data

The app and popularity-regression workflow expect `data\indian_songs.csv`. The local file was described as a copy of a file named `spotify_tracks.csv`, but its original download source and license still need confirmation.

The CSV must contain these columns:

```text
track_id, track_name, artist_name, year, popularity, artwork_url, album_name,
acousticness, danceability, duration_ms, energy, instrumentalness, key,
liveness, loudness, mode, speechiness, tempo, time_signature, valence,
track_url, language
```

Obtain the file separately and place it at `data\indian_songs.csv`. No verified download URL or license information is available in the repository.

### Genre-classification data

The separate genre workflow expects `data\dataset.csv`, with these 12 audio-feature columns and the target label `track_genre`:

```text
danceability, energy, key, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, time_signature, track_genre
```

The dataset's original source and license are unverified. Place a separately obtained copy at `data\dataset.csv` to run `python src/train_model.py`.

Both loaders report missing files, missing columns, and empty datasets as errors. Invalid numeric rows are removed during cleaning; if no usable songs remain, the app returns a clear dataset error instead of treating the file as usable.

## 6. How to Run Locally

### Requirements

- Python 3.11 or later.
- `data\indian_songs.csv` as described above.

In Windows PowerShell, clone the repository and create a virtual environment:

```powershell
git clone https://github.com/PrachiPangire/TuneWise-AI.git
Set-Location .\TuneWise-AI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the declared dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create the data folder if needed, copy the separately obtained `indian_songs.csv` into it, and start the Flask app:

```powershell
New-Item -ItemType Directory -Force data
Copy-Item 'C:\path\to\indian_songs.csv' 'data\indian_songs.csv'
python app.py
```

Replace `C:\path\to\indian_songs.csv` with the actual location of your downloaded copy. Open [http://localhost:5000](http://localhost:5000); check the dataset at [http://localhost:5000/api/health](http://localhost:5000/api/health).

To train either separate ML workflow from the project root:

```powershell
python src/train_model.py
python src/train_popularity_model.py
```

The genre script requires `data\dataset.csv`; the popularity script uses `data\indian_songs.csv`. Training writes model artifacts under `models\`, which is intentionally Git-ignored.

## 7. Git and GitHub Workflow

Git records changes and GitHub hosts the project history. Before sharing work, inspect `git status` and `git diff`, stage only the intended files, and keep local datasets, model artifacts, virtual environments, and secrets out of commits. No project changes are committed or pushed as part of this verification.

## 8. Docker Setup

The Dockerfile uses `python:3.11-slim`, installs `requirements.txt`, exposes port `5000`, and defines a health check that requests `/api/health`. The dataset is excluded from the image, so mount the local `data` folder into the container.

From PowerShell in the project root, with Docker running and `data\indian_songs.csv` present, build the image:

```powershell
docker build -t tunewise-ai:latest .
```

If a container named `tunewise-ai-app` already exists and you intend to replace it, stop and remove it first:

```powershell
docker stop tunewise-ai-app
docker rm tunewise-ai-app
```

Run the app with the local dataset mounted read-only:

```powershell
$dataPath = (Resolve-Path .\data).Path
docker run --name tunewise-ai-app -p 5000:5000 --mount "type=bind,source=$dataPath,target=/app/data,readonly" tunewise-ai:latest
```

In another PowerShell window, verify the health endpoint:

```powershell
Invoke-RestMethod http://localhost:5000/api/health
```

During this audit, the image build succeeded and a container from that image returned `status: ok` from `/api/health` with the dataset mounted. The already-running container named `tunewise-ai-app` was left unchanged; the verification container used a different name and host port.

## 9. DevOps Tool Integration - Jenkins

The Declarative Pipeline uses `sh` commands and requires an agent with `python3` and a POSIX-compatible shell. Its stages are:

1. **Python environment** — checks the Python version, creates `.venv`, upgrades pip, and installs `requirements.txt`.
2. **Syntax / import checks** — runs `py_compile` on the Flask app and selected `src` modules; this checks syntax and bytecode compilation, not runtime imports.
3. **Tests** — runs the automated suite with pytest.
4. **Local data report** — reports whether `data/indian_songs.csv` and `data/dataset.csv` are present.

A Jenkins pipeline result was reported as **SUCCESS**. No build number, test count, or source revision is asserted here. The Jenkins screenshots below show an older build and do not establish that the current `Jenkinsfile` produced that result.

## 10. Testing

Run the suite from the project root:

```powershell
python -m pytest -q
```

Latest local run: **32 passed, 0 failed, 0 skipped**. The Flask and training tests use synthetic data where applicable, so they do not require the song datasets. `tests/test_model.py` skips its three saved-genre-model checks when `models/genre_classifier.joblib` is absent; those checks ran in this environment.

## 11. Screenshot Evidence

These application screenshots show the current local Flask interface and feature results:

**Homepage**

![TuneWise-AI homepage](screenshots/tunewise-home.png)

**Song search results**

![TuneWise-AI song search results](screenshots/tunewise-search-results.png)

**Mood recommendations**

![TuneWise-AI mood recommendations](screenshots/tunewise-mood-recommendations.png)

**Playlist analysis**

![TuneWise-AI playlist analysis](screenshots/tunewise-playlist-analysis.png)

The following supplied Jenkins screenshots are from a historical build and do not verify the current Jenkinsfile:

![Jenkins build #3 console output](screenshots/jenkins-build-3-console.png)

![Jenkins build #3 status](screenshots/jenkins-build-3-success.png)

No Docker build screenshot is available. The dated screenshots in the folder are not referenced as evidence of the current Flask app.

## 12. Limitations and Future Improvements

- The app depends on local CSV datasets; their original sources and licenses still need confirmation.
- Mood recommendations use hand-written audio-feature rules rather than a trained mood classifier.
- Genre classification and popularity regression are separate training workflows, not live Spotify services; the app does not use the Spotify API.
- Future work could include documenting verified dataset provenance, clarifying model artifact setup, and expanding tests for model-training edge cases.

## 13. Conclusion

TuneWise-AI combines local music search, explainable mood recommendations, and playlist CSV analysis in a Flask web application. Its separate trained ML workflows, automated tests, and DevOps configurations provide additional material for learning and demonstration.

## 14. Author

Prachi Pangire
