# TuneWise AI 🎵

TuneWise AI is a machine learning web application that predicts the music genre of a song based on its audio features. It uses a Random Forest classification model and provides an interactive interface built with Streamlit.

## Features

- Predict music genres from audio features.
- Explore audio characteristics through interactive visualizations.
- View model performance information.
- Use a simple, interactive Streamlit web interface.
- Run the application inside a Docker container.
- Execute automated tests through Jenkins.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Random Forest Classifier
- **Web Application:** Streamlit
- **Containerization:** Docker
- **Continuous Integration:** Jenkins
- **Testing:** Pytest

## Machine Learning Model

TuneWise AI uses a Random Forest Classifier to predict one of 114 music genres from numerical audio features.

The input features include:

- Danceability
- Energy
- Key and mode
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Time signature

### Model Performance

- **Dataset size:** 114,000 tracks
- **Number of genres:** 114
- **Model:** Random Forest Classifier
- **Measured accuracy:** 21.65%

The current model is a baseline. Its performance can be improved through feature engineering, hyperparameter tuning, and further model evaluation.

## Dataset

The project uses a Spotify tracks dataset containing audio features and genre labels, with 114,000 rows and 21 columns.

The dataset is stored locally in `data/dataset.csv` and excluded from Git to avoid uploading a large data file.

Add the original dataset title, publisher, and source URL here before final submission.

## Project Structure

```text
TuneWise-AI/
├── data/
├── models/
├── src/
├── tests/
├── screenshots/
├── app.py
├── requirements.txt
├── Dockerfile
├── Jenkinsfile
├── .gitignore
└── README.md
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone