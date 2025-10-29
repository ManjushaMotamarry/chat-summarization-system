"""
Production DAG for Chat Summarization Pipeline.
Orchestrates: Download → Load → Preprocess → Validate
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data.dataset_loader import DatasetLoader
from src.data.database import get_database_engine, get_session, Conversation, Message, Summary
from src.data.text_preprocessor import TextPreprocessor
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


def task_download_dataset(**context):
    """Task 1: Download dataset from HuggingFace"""
    logger.info("=" * 60)
    logger.info("TASK 1: Downloading Dataset")
    logger.info("=" * 60)
    
    # Use generic loader (reads from config)
    loader = DatasetLoader()
    dataset = loader.load()
    
    # Store dataset info in XCom for next tasks
    context['ti'].xcom_push(key='dataset_name', value=loader.dataset_name)
    context['ti'].xcom_push(key='train_size', value=len(dataset['train']))
    
    logger.info(f"✅ Downloaded {len(dataset['train'])} conversations")
    return "download_complete"


def task_load_to_database(**context):
    """Task 2: Load dataset into database"""
    logger.info("=" * 60)
    logger.info("TASK 2: Loading Data to Database")
    logger.info("=" * 60)
    
    # Get dataset info from previous task
    dataset_name = context['ti'].xcom_pull(key='dataset_name', task_ids='download_dataset')
    
    # Load dataset and database
    loader = DatasetLoader(dataset_name)
    dataset = loader.load()
    
    config = get_config()
    db_config = config.get_database_config()
    engine = get_database_engine(db_path=db_config['path'])
    session = get_session(engine)
    
    # Clear existing data (for re-runs)
    logger.info("Clearing existing data...")
    session.query(Summary).delete()
    session.query(Message).delete()
    session.query(Conversation).delete()
    session.commit()
    
    # Load training data
    train_data = dataset['train']
    logger.info(f"Loading {len(train_data)} conversations...")
    
    conversations_loaded = 0
    messages_loaded = 0
    
    for idx, example in enumerate(train_data):
        # Create conversation
        conversation = Conversation(
            channel="messenger",
            status="completed",
            created_at=datetime.now()
        )
        session.add(conversation)
        session.flush()
        
        # Parse dialogue
        dialogue_field = loader.get_dialogue_field()
        summary_field = loader.get_summary_field()
        
        dialogue_text = example[dialogue_field]
        parsed_messages = loader.parse_dialogue(dialogue_text)
        
        # Add messages
        for sender, message_text in parsed_messages:
            message = Message(
                conversation_id=conversation.conversation_id,
                sender=sender,
                message_text=message_text,
                timestamp=datetime.now()
            )
            session.add(message)
            messages_loaded += 1
        
        # Add summary
        summary = Summary(
            conversation_id=conversation.conversation_id,
            summary_text=example[summary_field],
            model_version="human_annotated",
            created_at=datetime.now()
        )
        session.add(summary)
        
        conversations_loaded += 1
        
        # Commit every 100 conversations
        if (idx + 1) % 100 == 0:
            session.commit()
            logger.info(f"  Loaded {conversations_loaded} conversations...")
    
    # Final commit
    session.commit()
    session.close()
    
    logger.info(f"✅ Loaded {conversations_loaded} conversations with {messages_loaded} messages")
    
    # Store stats in XCom
    context['ti'].xcom_push(key='conversations_loaded', value=conversations_loaded)
    context['ti'].xcom_push(key='messages_loaded', value=messages_loaded)
    
    return "load_complete"


def task_preprocess_data(**context):
    """Task 3: Preprocess text data"""
    logger.info("=" * 60)
    logger.info("TASK 3: Preprocessing Data")
    logger.info("=" * 60)
    
    # Initialize preprocessor (uses smart config)
    preprocessor = TextPreprocessor()
    
    # Get database session
    config = get_config()
    db_config = config.get_database_config()
    engine = get_database_engine(db_path=db_config['path'])
    session = get_session(engine)
    
    # Get all messages
    messages = session.query(Message).all()
    logger.info(f"Preprocessing {len(messages)} messages...")
    
    processed_count = 0
    for message in messages:
        # Clean the message text
        cleaned_text = preprocessor.clean_text(message.message_text)
        message.message_text = cleaned_text
        processed_count += 1
        
        if processed_count % 1000 == 0:
            logger.info(f"  Processed {processed_count} messages...")
    
    session.commit()
    session.close()
    
    logger.info(f"✅ Preprocessed {processed_count} messages")
    context['ti'].xcom_push(key='messages_preprocessed', value=processed_count)
    
    return "preprocess_complete"


def task_validate_data(**context):
    """Task 4: Validate data quality"""
    logger.info("=" * 60)
    logger.info("TASK 4: Validating Data Quality")
    logger.info("=" * 60)
    
    # Import validation function
    from scripts.validate_data import validate_database_data
    
    # Run validation
    validation_passed = validate_database_data()
    
    if validation_passed:
        logger.info("✅ All validation checks passed!")
    else:
        logger.warning("⚠️ Some validation issues found (see logs)")
    
    context['ti'].xcom_push(key='validation_passed', value=validation_passed)
    
    return "validation_complete"


def task_generate_summary_report(**context):
    """Task 5: Generate pipeline summary report"""
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    
    # Pull stats from all tasks
    dataset_name = context['ti'].xcom_pull(key='dataset_name', task_ids='download_dataset')
    train_size = context['ti'].xcom_pull(key='train_size', task_ids='download_dataset')
    conversations_loaded = context['ti'].xcom_pull(key='conversations_loaded', task_ids='load_to_database')
    messages_loaded = context['ti'].xcom_pull(key='messages_loaded', task_ids='load_to_database')
    messages_preprocessed = context['ti'].xcom_pull(key='messages_preprocessed', task_ids='preprocess_data')
    validation_passed = context['ti'].xcom_pull(key='validation_passed', task_ids='validate_data')
    
    logger.info(f"\n📊 Pipeline Results:")
    logger.info(f"   Dataset: {dataset_name}")
    logger.info(f"   Downloaded: {train_size} samples")
    logger.info(f"   Loaded: {conversations_loaded} conversations")
    logger.info(f"   Messages: {messages_loaded}")
    logger.info(f"   Preprocessed: {messages_preprocessed} messages")
    logger.info(f"   Validation: {'✅ PASSED' if validation_passed else '⚠️ ISSUES FOUND'}")
    logger.info("=" * 60)
    
    return "pipeline_complete"


# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'chat_summarization_pipeline',
    default_args=default_args,
    description='End-to-end chat summarization data pipeline',
    schedule_interval=None,  # Manual trigger only (change to @daily for automation)
    catchup=False,
    tags=['production', 'ml-pipeline', 'chat-summarization'],
)

# Define tasks
download_task = PythonOperator(
    task_id='download_dataset',
    python_callable=task_download_dataset,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_to_database',
    python_callable=task_load_to_database,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=task_preprocess_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=task_validate_data,
    dag=dag,
)

summary_task = PythonOperator(
    task_id='generate_summary',
    python_callable=task_generate_summary_report,
    dag=dag,
)

# Define pipeline flow
download_task >> load_task >> preprocess_task >> validate_task >> summary_task