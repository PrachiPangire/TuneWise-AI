pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        // Override these in Jenkins if the project lives elsewhere.
        PROJECT_DIR = 'E:\\TuneWise-AI'
        VENV_PYTHON = 'E:\\TuneWise-AI\\.venv\\Scripts\\python.exe'
    }

    stages {
        stage('Workspace check') {
            steps {
                dir("${env.PROJECT_DIR}") {
                    echo "Using pre-provisioned TuneWise AI checkout at ${env.PROJECT_DIR}"
                    bat 'git status --short --branch'
                }
            }
        }

        stage('Environment check') {
            steps {
                dir("${env.PROJECT_DIR}") {
                    bat """
                        \"${env.VENV_PYTHON}\" --version
                        \"${env.VENV_PYTHON}\" -c \"import flask, numpy, pandas, pytest, sklearn; print('imports-ok')\"
                    """
                }
            }
        }

        stage('Syntax / import checks') {
            steps {
                dir("${env.PROJECT_DIR}") {
                    bat """
                        \"${env.VENV_PYTHON}\" -m py_compile app.py src\\data_utils.py src\\mood_logic.py src\\prediction.py src\\visualizations.py src\\train_model.py src\\train_popularity_model.py
                    """
                }
            }
        }

        stage('Run pytest') {
            steps {
                dir("${env.PROJECT_DIR}") {
                    bat "\"${env.VENV_PYTHON}\" -m pytest -v"
                }
            }
        }

        stage('Local asset report') {
            steps {
                dir("${env.PROJECT_DIR}") {
                    bat """
                        echo Reporting optional local datasets and model artifacts...
                        if exist data\\indian_songs.csv (echo FOUND data\\indian_songs.csv) else (echo MISSING data\\indian_songs.csv)
                        if exist data\\dataset.csv (echo FOUND data\\dataset.csv) else (echo MISSING data\\dataset.csv)
                        if exist models\\popularity_regressor.joblib (echo FOUND models\\popularity_regressor.joblib) else (echo MISSING models\\popularity_regressor.joblib)
                        if exist models\\genre_classifier.joblib (echo FOUND models\\genre_classifier.joblib) else (echo MISSING models\\genre_classifier.joblib)
                    """
                }
            }
        }
    }

    post {
        success {
            echo 'TuneWise AI pipeline completed successfully.'
        }
        failure {
            echo 'TuneWise AI pipeline failed. Inspect the console output. No credentials are printed by this Jenkinsfile.'
        }
    }
}
