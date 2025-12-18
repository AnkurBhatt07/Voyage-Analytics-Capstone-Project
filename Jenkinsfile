pipeline {
    agent any

    stages {

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t flight-price-app:latest .'
            }
        }
    }
}