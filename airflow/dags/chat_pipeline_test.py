"""
Test DAG with smaller dataset to verify pipeline works.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_task_1():
    logger.info("✅ Task 1: Simple test - works!")
    return "task1_done"


def test_task_2():
    logger.info("✅ Task 2: Simple test - works!")
    return "task2_done"


def test_task_3():
    logger.info("✅ Task 3: Simple test - works!")
    return "task3_done"


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 10, 1),
    'retries': 0,
}

dag = DAG(
    'pipeline_test_simple',
    default_args=default_args,
    description='Simple test to verify Airflow works',
    schedule_interval=None,
    catchup=False,
)

task1 = PythonOperator(task_id='test_1', python_callable=test_task_1, dag=dag)
task2 = PythonOperator(task_id='test_2', python_callable=test_task_2, dag=dag)
task3 = PythonOperator(task_id='test_3', python_callable=test_task_3, dag=dag)

task1 >> task2 >> task3