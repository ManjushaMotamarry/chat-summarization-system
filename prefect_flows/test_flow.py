"""
Simple Prefect test flow.
"""

from prefect import flow, task
from datetime import datetime

@task
def task_1():
    print("✅ Task 1: Hello from Prefect!")
    return "task1_complete"

@task
def task_2(input_data):
    print(f"✅ Task 2: Received {input_data}")
    return "task2_complete"

@task
def task_3(input_data):
    print(f"✅ Task 3: Received {input_data}")
    print(f"⏰ Pipeline completed at {datetime.now()}")
    return "pipeline_complete"

@flow(name="test-pipeline")
def test_pipeline():
    """Simple test to verify Prefect works"""
    print("🚀 Starting test pipeline...")
    
    result1 = task_1()
    result2 = task_2(result1)
    result3 = task_3(result2)
    
    print("✅ Pipeline finished!")
    return result3

if __name__ == "__main__":
    test_pipeline()