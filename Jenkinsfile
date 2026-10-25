
pipeline {
    agent any

    options {
        timestamps()
    }

    stages {
        stage('Python environment') {
            steps {
                sh 'python3 --version'
                sh 'python3 -m venv .venv'
                sh '.venv/bin/python -m pip install --upgrade pip'
                sh '.venv/bin/python -m pip install -r requirements.txt'
            }
        }

        stage('Syntax / import checks') {
            steps {
                sh '''
                    .venv/bin/python -m py_compile \
                        app.py \
                        src/data_utils.py \
                        src/mood_logic.py \
                        src/prediction.py \
                        src/visualizations.py \
                        src/train_model.py \
                        src/train_popularity_model.py
                '''
            }
        }

        stage('Tests') {
            steps {
                sh '.venv/bin/python -m pytest -v'
            }
        }

        stage('Local data report') {
            steps {
                sh '''
                    if [ -f data/indian_songs.csv ]; then
                        echo "FOUND data/indian_songs.csv"
                    else
                        echo "MISSING data/indian_songs.csv"
                    fi

                    if [ -f data/dataset.csv ]; then
                        echo "FOUND data/dataset.csv"
                    else
                        echo "MISSING data/dataset.csv"
                    fi
                '''
            }
        }
    }

    post {
        success {
            echo 'TuneWise AI pipeline completed successfully.'
        }
        failure {
            echo 'TuneWise AI pipeline failed. Inspect the console output.'
        }
    }
}
