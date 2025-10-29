"""
Production Chat Summarization Pipeline with Prefect.
Tasks: Download → Load → Preprocess → Validate → Report
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from prefect import flow, task
from src.data.dataset_loader import DatasetLoader
from src.data.database import get_database_engine, get_session, Conversation, Message, Summary
from src.data.text_preprocessor import TextPreprocessor
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


@task(name="download-dataset", retries=2, retry_delay_seconds=30)
def download_dataset_task():
    """Task 1: Download dataset from HuggingFace"""
    logger.info("=" * 60)
    logger.info("TASK 1: Downloading Dataset")
    logger.info("=" * 60)
    
    loader = DatasetLoader()
    dataset = loader.load()
    
    logger.info(f"✅ Downloaded {len(dataset['train'])} conversations")
    
    return {
        'dataset_name': loader.dataset_name,
        'train_size': len(dataset['train']),
        'loader': loader
    }


@task(name="load-to-database", retries=1)
def load_to_database_task(download_result):
    """Task 2: Load dataset into database"""
    logger.info("=" * 60)
    logger.info("TASK 2: Loading Data to Database")
    logger.info("=" * 60)
    
    dataset_name = download_result['dataset_name']
    
    # Reload dataset and database
    loader = DatasetLoader(dataset_name)
    dataset = loader.load()
    
    config = get_config()
    db_config = config.get_database_config()
    engine = get_database_engine(db_path=db_config['path'])
    session = get_session(engine)
    
    # Clear existing data
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
        
        # Commit every 100
        if (idx + 1) % 100 == 0:
            session.commit()
            logger.info(f"  Loaded {conversations_loaded} conversations...")
    
    session.commit()
    session.close()
    
    logger.info(f"✅ Loaded {conversations_loaded} conversations with {messages_loaded} messages")
    
    return {
        'conversations_loaded': conversations_loaded,
        'messages_loaded': messages_loaded
    }


@task(name="preprocess-data")
def preprocess_data_task(load_result):
    """Task 3: Preprocess text data with smart cleaning"""
    logger.info("=" * 60)
    logger.info("TASK 3: Preprocessing Data")
    logger.info("=" * 60)
    
    preprocessor = TextPreprocessor()
    
    config = get_config()
    db_config = config.get_database_config()
    engine = get_database_engine(db_path=db_config['path'])
    session = get_session(engine)
    
    messages = session.query(Message).all()
    logger.info(f"Preprocessing {len(messages)} messages...")
    
    processed_count = 0
    for message in messages:
        cleaned_text = preprocessor.clean_text(message.message_text)
        message.message_text = cleaned_text
        processed_count += 1
        
        if processed_count % 1000 == 0:
            logger.info(f"  Processed {processed_count} messages...")
    
    session.commit()
    session.close()
    
    logger.info(f"✅ Preprocessed {processed_count} messages")
    
    return {
        'messages_preprocessed': processed_count
    }


@task(name="validate-data")
def validate_data_task(preprocess_result):
    """Task 4: Validate data quality"""
    logger.info("=" * 60)
    logger.info("TASK 4: Validating Data Quality")
    logger.info("=" * 60)
    
    from scripts.validate_data import validate_database_data
    
    validation_passed = validate_database_data()
    
    if validation_passed:
        logger.info("✅ All validation checks passed!")
    else:
        logger.warning("⚠️ Some validation issues found")
    
    return {
        'validation_passed': validation_passed
    }


@task(name="generate-report")
def generate_report_task(download_result, load_result, preprocess_result, validate_result):
    """Task 5: Generate pipeline summary report"""
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"\n📊 Pipeline Results:")
    logger.info(f"   Dataset: {download_result['dataset_name']}")
    logger.info(f"   Downloaded: {download_result['train_size']} samples")
    logger.info(f"   Loaded: {load_result['conversations_loaded']} conversations")
    logger.info(f"   Messages: {load_result['messages_loaded']}")
    logger.info(f"   Preprocessed: {preprocess_result['messages_preprocessed']} messages")
    logger.info(f"   Validation: {'✅ PASSED' if validate_result['validation_passed'] else '⚠️ ISSUES FOUND'}")
    logger.info("=" * 60)
    
    return "pipeline_complete"


@flow(name="chat-summarization-pipeline", log_prints=True)
def chat_summarization_pipeline():
    """
    Main pipeline flow for chat summarization.
    Orchestrates: Download → Load → Preprocess → Validate → Report
    """
    print("\n🚀 Starting Chat Summarization Pipeline...\n")
    
    # Execute tasks in sequence
    download_result = download_dataset_task()
    load_result = load_to_database_task(download_result)
    preprocess_result = preprocess_data_task(load_result)
    validate_result = validate_data_task(preprocess_result)
    report = generate_report_task(download_result, load_result, preprocess_result, validate_result)
    
    print("\n✅ Pipeline Complete!\n")
    return report


if __name__ == "__main__":
    chat_summarization_pipeline()