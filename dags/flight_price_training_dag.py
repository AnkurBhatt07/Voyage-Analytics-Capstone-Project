from datetime import datetime , timedelta
from airflow import DAG 
from airflow.operators.python import PythonOperator
from training.train_model import train_and_save_model

default_args = {
    "owner" : "airflow",
    "depends_on_past" : False,
    "email_on_failure" : False,
    "email_on_retry" : False,
    "retries" : 1,
    "retry_delay" : timedelta(minutes=5)
}

with DAG(
    dag_id = "flight_price_model_training",
    description = "Retrain Gradient Boosting model for flight price prediction",
    default_args = default_args , 
    schedule_interval = "@weekly",
    start_date = datetime(2025 , 12 , 20),
    catchup = False ,
    tags = [ 'mlops' , 'training' , 'flight-price'] ,
) as dag :
    
    train_model_task = PythonOperator(
        task_id = "train_gradient_boost_model",
        python_callable = train_and_save_model,
    )

    train_model_task

    