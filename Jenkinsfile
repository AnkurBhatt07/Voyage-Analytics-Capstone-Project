pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/AnkurBhatt07/Voyage-Analytics-Capstone-Project.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t flight-price-app:latest .'
            }
        }
    }
}