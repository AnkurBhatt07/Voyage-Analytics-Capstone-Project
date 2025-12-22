pipeline {
    agent any

    environment {
        IMAGE_NAME = 'flight-price-app'
        IMAGE_TAG = 'latest'
        K8S_DEPLOYMENT = 'flight-price-deployment'
    }

    stages {

        stage('Checkout Code') {
         steps {
             checkout scm
         }   
        }

        stage('Verify Airflow Model Exists') {
            steps {
                script {
                    if (!fileExists('artifacts/grad_boost_best_airflow.pkl')) {
                        error "Airflow-trained model not found. Run Airflow DAG to generate the model before proceeding."
                    }
                }
            }
        }



        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
            }
        }

        // stage("Load Image into Minikube") {
        //     steps {
        //         sh "minikube image load ${IMAGE_NAME}:${IMAGE_TAG}"
        //     }
        // }

        // stage('Deploy to Kubernetes') {
        //     steps {
        //         sh """
        //         kubectl apply -f flight-price-deployment.yaml
        //         kubectl apply -f flight-price-service.yaml
        //         """
        //     }
        // } 

    }

    post {
        success {
            echo "Deployment succeeded!"
        }
        failure {
            echo "Deployment failed."
        }
    }
}