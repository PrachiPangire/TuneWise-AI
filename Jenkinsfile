
pipeline {
    agent any

    options {
        timestamps()
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'TuneWise AI source code is available.'
            }
        }

        stage('Run ML Tests') {
            steps {
                dir('E:\\TuneWise-AI') {
                    bat '"E:\\TuneWise-AI\\.venv\\Scripts\\python.exe" -m pytest -v'
                }
            }
        }
    }

    post {
        success {
            echo 'TuneWise AI tests passed successfully!'
        }
        failure {
            echo 'TuneWise AI pipeline failed. Check the console output.'
        }
    }
}
