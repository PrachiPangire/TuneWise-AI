
pipeline {
    agent any

    options {
        timestamps()
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'TuneWise AI source code is available in the workspace.'
            }
        }

        stage('Run ML Tests') {
            steps {
                bat '"E:\\TuneWise-AI\\.venv\\Scripts\\python.exe" -m pytest -v'
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
