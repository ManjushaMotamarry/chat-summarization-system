"""
Simple Hello World DAG to test Airflow setup.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def print_hello():
    print("Hello from Airflow!")
    return "Hello task completed"


def print_date():
    print(f"Current date and time: {datetime.now()}")
    return "Date task completed"


# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'hello_world_pipeline',
    default_args=default_args,
    description='A simple hello world DAG',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
)

# Define tasks
task_hello = PythonOperator(
    task_id='say_hello',
    python_callable=print_hello,
    dag=dag,
)

task_date = PythonOperator(
    task_id='print_date',
    python_callable=print_date,
    dag=dag,
)

# Set task order: hello runs first, then date
task_hello >> task_date