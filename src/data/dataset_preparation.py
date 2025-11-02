"""
Dataset preparation for model training.
Handles train/val/test splitting with stratification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetPreparator:
    """Prepares dataset for model training with stratified splits"""
    
    def __init__(self, db_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Initialize dataset preparator
        
        Args:
            db_path: Path to SQLite database
            train_ratio: Proportion for training set (default: 0.8)
            val_ratio: Proportion for validation set (default: 0.1)
            test_ratio: Proportion for test set (default: 0.1)
        """
        self.db_path = db_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.engine = create_engine(f'sqlite:///{db_path}')
        
        logger.info(f"Initialized DatasetPreparator with splits: {train_ratio}/{val_ratio}/{test_ratio}")
    
    def load_data(self):
        """Load conversations, messages, and summaries from database"""
        logger.info("Loading data from database...")
        
        # Load tables
        conversations_df = pd.read_sql_table('conversations', self.engine)
        messages_df = pd.read_sql_table('messages', self.engine)
        summaries_df = pd.read_sql_table('summaries', self.engine)
        
        logger.info(f"Loaded {len(conversations_df)} conversations, {len(messages_df)} messages, {len(summaries_df)} summaries")
        
        return conversations_df, messages_df, summaries_df
    
    def create_conversation_dataset(self, conversations_df, messages_df, summaries_df):
        """
        Create a unified dataset with conversation text and summaries
        
        Returns:
            DataFrame with columns: conversation_id, dialogue, summary, num_messages, num_words
        """
        logger.info("Creating unified conversation dataset...")
        
        # Group messages by conversation
        conversation_texts = []
        
        for conv_id in conversations_df['conversation_id']:
            # Get all messages for this conversation
            conv_messages = messages_df[messages_df['conversation_id'] == conv_id].sort_values('timestamp')
            
            # Format as dialogue: "Speaker: Message\n"
            dialogue_lines = []
            for _, msg in conv_messages.iterrows():
                dialogue_lines.append(f"{msg['sender']}: {msg['message_text']}")
            
            dialogue = "\n".join(dialogue_lines)
            
            # Get summary
            summary_row = summaries_df[summaries_df['conversation_id'] == conv_id]
            summary = summary_row['summary_text'].values[0] if len(summary_row) > 0 else ""
            
            # Calculate metrics
            num_messages = len(conv_messages)
            num_words = sum(conv_messages['message_text'].str.split().str.len())
            
            conversation_texts.append({
                'conversation_id': conv_id,
                'dialogue': dialogue,
                'summary': summary,
                'num_messages': num_messages,
                'num_words': num_words
            })
        
        dataset_df = pd.DataFrame(conversation_texts)
        logger.info(f"Created dataset with {len(dataset_df)} conversations")
        
        return dataset_df
    
    def create_stratified_bins(self, dataset_df, n_bins=5):
        """
        Create stratification bins based on conversation length
        
        Args:
            dataset_df: Dataset DataFrame
            n_bins: Number of bins for stratification
            
        Returns:
            DataFrame with added 'length_bin' column
        """
        logger.info(f"Creating {n_bins} stratification bins based on conversation length...")
        
        # Create bins based on word count
        dataset_df['length_bin'] = pd.qcut(
            dataset_df['num_words'], 
            q=n_bins, 
            labels=False, 
            duplicates='drop'
        )
        
        # Log bin distributions
        bin_counts = dataset_df['length_bin'].value_counts().sort_index()
        logger.info(f"Bin distribution:\n{bin_counts}")
        
        return dataset_df
    
    def split_dataset(self, dataset_df):
        """
        Split dataset into train/val/test with stratification
        
        Returns:
            train_df, val_df, test_df
        """
        logger.info("Splitting dataset with stratification...")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            dataset_df,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=dataset_df['length_bin'],
            random_state=42
        )
        
        # Second split: val vs test
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_df['length_bin'],
            random_state=42
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Log statistics for each split
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            logger.info(f"{name} set - Avg words: {df['num_words'].mean():.1f}, Avg messages: {df['num_messages'].mean():.1f}")
        
        return train_df, val_df, test_df
    
    def prepare_dataset(self):
        """
        Complete pipeline: load → create dataset → stratify → split
        
        Returns:
            train_df, val_df, test_df
        """
        logger.info("="*60)
        logger.info("Starting dataset preparation pipeline")
        logger.info("="*60)
        
        # Load data
        conversations_df, messages_df, summaries_df = self.load_data()
        
        # Create unified dataset
        dataset_df = self.create_conversation_dataset(conversations_df, messages_df, summaries_df)
        
        # Create stratification bins
        dataset_df = self.create_stratified_bins(dataset_df)
        
        # Split dataset
        train_df, val_df, test_df = self.split_dataset(dataset_df)
        
        logger.info("="*60)
        logger.info("Dataset preparation complete!")
        logger.info("="*60)
        
        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df, output_dir='data/processed'):
            """
            Save train/val/test splits to CSV files
            
            Args:
                train_df, val_df, test_df: DataFrames to save
                output_dir: Directory to save files
            """
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            train_path = os.path.join(output_dir, 'train.csv')
            val_path = os.path.join(output_dir, 'val.csv')
            test_path = os.path.join(output_dir, 'test.csv')
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Saved splits to {output_dir}/")
            logger.info(f"  - train.csv: {len(train_df)} rows")
            logger.info(f"  - val.csv: {len(val_df)} rows")
            logger.info(f"  - test.csv: {len(test_df)} rows")


if __name__ == "__main__":
    # Test the preparator
    preparator = DatasetPreparator(db_path='data/chat_conversations.db')
    train_df, val_df, test_df = preparator.prepare_dataset()
    
    print("\nSample from training set:")
    print(train_df[['conversation_id', 'num_messages', 'num_words']].head())
    
    # Save splits
    preparator.save_splits(train_df, val_df, test_df)

