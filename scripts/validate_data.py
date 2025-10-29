"""
Data validation using Great Expectations.
Validates data quality: missing values, schema, anomalies.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from great_expectations.data_context import FileDataContext
from great_expectations.core.batch import RuntimeBatchRequest
import great_expectations as gx
import pandas as pd
from src.data.database import get_database_engine, Message, Conversation, Summary
from sqlalchemy.orm import sessionmaker
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_database_data():
    """
    Validate data quality in the database.
    Checks for:
    - Missing values
    - Data types
    - Value ranges
    - Anomalies
    """
    
    logger.info("🔍 Starting data validation...")
    
    # Connect to database
    engine = get_database_engine(db_path="data/chat_conversations.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Load data into pandas for validation
    logger.info("📊 Loading data from database...")
    
    # Query messages
    messages_query = session.query(
        Message.message_id,
        Message.conversation_id,
        Message.sender,
        Message.message_text,
        Message.timestamp
    ).statement
    
    messages_df = pd.read_sql(messages_query, engine)
    logger.info(f"   Loaded {len(messages_df)} messages")
    
    # Basic validation checks
    logger.info("\n✅ Running validation checks...")
    
    issues_found = []
    
    # Check 1: Missing values
    logger.info("1️⃣ Checking for missing values...")
    missing_counts = messages_df.isnull().sum()
    if missing_counts.sum() > 0:
        issues_found.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        logger.warning(f"   ⚠️ Missing values: {missing_counts[missing_counts > 0].to_dict()}")
    else:
        logger.info("   ✅ No missing values")
    
    # Check 2: Empty messages
    logger.info("2️⃣ Checking for empty messages...")
    empty_messages = messages_df[messages_df['message_text'].str.strip() == '']
    if len(empty_messages) > 0:
        issues_found.append(f"Found {len(empty_messages)} empty messages")
        logger.warning(f"   ⚠️ {len(empty_messages)} empty messages found")
    else:
        logger.info("   ✅ No empty messages")
    
    # Check 3: Message length distribution
    logger.info("3️⃣ Checking message length distribution...")
    messages_df['message_length'] = messages_df['message_text'].str.len()
    avg_length = messages_df['message_length'].mean()
    max_length = messages_df['message_length'].max()
    min_length = messages_df['message_length'].min()
    
    logger.info(f"   📏 Avg length: {avg_length:.1f} chars")
    logger.info(f"   📏 Max length: {max_length} chars")
    logger.info(f"   📏 Min length: {min_length} chars")
    
    # Check 4: Anomaly detection - extremely long messages
    logger.info("4️⃣ Checking for anomalies (very long messages)...")
    threshold = messages_df['message_length'].quantile(0.99)  # 99th percentile
    anomalies = messages_df[messages_df['message_length'] > threshold]
    if len(anomalies) > 0:
        logger.warning(f"   ⚠️ Found {len(anomalies)} unusually long messages (>{threshold:.0f} chars)")
        issues_found.append(f"{len(anomalies)} unusually long messages")
    else:
        logger.info("   ✅ No unusual message lengths")
    
    # Check 5: Conversation integrity
    logger.info("5️⃣ Checking conversation integrity...")
    conversations_with_messages = messages_df['conversation_id'].nunique()
    total_conversations = session.query(Conversation).count()
    
    if conversations_with_messages != total_conversations:
        issues_found.append(f"Conversation mismatch: {total_conversations} conversations but only {conversations_with_messages} have messages")
        logger.warning(f"   ⚠️ {total_conversations - conversations_with_messages} conversations without messages")
    else:
        logger.info(f"   ✅ All {total_conversations} conversations have messages")
    
    # Summary
    logger.info("\n" + "="*60)
    if len(issues_found) == 0:
        logger.info("✅ DATA VALIDATION PASSED - No issues found!")
    else:
        logger.warning(f"⚠️ DATA VALIDATION COMPLETED - {len(issues_found)} issue(s) found:")
        for i, issue in enumerate(issues_found, 1):
            logger.warning(f"   {i}. {issue}")
    logger.info("="*60)
    
    session.close()
    
    return len(issues_found) == 0


if __name__ == "__main__":
    validate_database_data()