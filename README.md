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
git clone https://github.com/PrachiPangire/TuneWise-AI.git
cd TuneWise-AI
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

**Note:** The trained model file must be available at `models/genre_classifier.joblib` for predictions to work.

## Automated Testing

Run the project's automated tests with:

```bash
python -m pytest -v
```

The tests check that the saved model exists, predicts a valid genre, and supports multiple predictions.

## Docker

Build the Docker image from the project directory:

```bash
docker build -t tunewise-ai .
```

Run the container:

```bash
docker run -d --name tunewise-container -p 8501:8501 tunewise-ai
```

Open `http://localhost:8501` in your browser.

If the container already exists, remove it before recreating it:

```bash
docker rm -f tunewise-container
```

## Jenkins CI Pipeline

The project includes a `Jenkinsfile` that defines a pipeline to execute the automated ML tests.

The pipeline:

1. Checks out the project from GitHub.
2. Runs the Pytest test suite.
3. Reports whether the build succeeded or failed.

A successful Jenkins build confirms that the configured automated tests passed in that run.

## Screenshots

Project screenshots are stored in the `screenshots/` directory.

- Streamlit application running in Docker.
- Jenkins pipeline showing a successful build.
- Jenkins console output showing the automated test results.

## Project Goals

This project demonstrates a machine learning workflow combined with web application development, model testing, Docker containerization, and Jenkins-based continuous integration.

## Future Improvements

- Improve classification accuracy.
- Compare multiple classification algorithms.
- Add model evaluation visualizations.
- Improve model packaging and deployment.
- Automate model training and evaluation.

## Author

**Prachi Pangire**

GitHub: [PrachiPangire](https://github.com/PrachiPangire)

## License

Add an appropriate license if you decide to distribute the project publicly.
