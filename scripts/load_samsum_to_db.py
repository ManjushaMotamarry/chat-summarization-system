"""
Script to load SAMSum dataset into our database.
Parses dialogues and inserts conversations, messages, and summaries.
"""

import sys
import os
import re
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.data.database import get_database_engine, get_session, Conversation, Message, Summary


def parse_dialogue(dialogue_text):
    """
    Parse a dialogue string into individual messages.
    
    Example input:
    "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\n"
    
    Returns list of tuples: [(sender, message), (sender, message), ...]
    """
    messages = []
    
    # Split by newlines (just \n, not \r\n)
    lines = dialogue_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Split on first colon to get sender and message
        if ':' in line:
            parts = line.split(':', 1)
            sender = parts[0].strip()
            message_text = parts[1].strip()
            messages.append((sender, message_text))
    
    return messages


def load_samsum_to_database():
    print("📥 Loading SAMSum dataset...\n")
    
    # Load dataset
    dataset = load_dataset("knkarthick/samsum")
    
    # Get database session
    engine = get_database_engine(db_path="data/chat_conversations.db")
    session = get_session(engine)
    
    # We'll only load the training set for now
    # You can add validation and test later
    train_data = dataset['train']
    
    print(f"📊 Processing {len(train_data)} conversations from training set...\n")
    
    # Use tqdm for progress bar
    for idx, example in enumerate(tqdm(train_data, desc="Inserting conversations")):
        
        # Create conversation record
        conversation = Conversation(
            channel="messenger",  # SAMSum is messenger-like
            status="completed",
            created_at=datetime.now()
        )
        session.add(conversation)
        session.flush()  # Get the conversation_id without committing
        
        # Parse dialogue into individual messages
        dialogue_text = example['dialogue']
        parsed_messages = parse_dialogue(dialogue_text)
        
        # Insert messages
        for sender, message_text in parsed_messages:
            message = Message(
                conversation_id=conversation.conversation_id,
                sender=sender,
                message_text=message_text,
                timestamp=datetime.now()
            )
            session.add(message)
        
        # Insert summary
        summary = Summary(
            conversation_id=conversation.conversation_id,
            summary_text=example['summary'],
            model_version="human_annotated",  # These are human-written summaries
            created_at=datetime.now()
        )
        session.add(summary)
        
        # Commit every 100 conversations for efficiency
        if (idx + 1) % 100 == 0:
            session.commit()
    
    # Final commit for remaining records
    session.commit()
    
    print("\n✅ Data loading complete!")
    
    # Show statistics
    total_conversations = session.query(Conversation).count()
    total_messages = session.query(Message).count()
    total_summaries = session.query(Summary).count()
    
    print(f"\n📈 Database Statistics:")
    print(f"   Total conversations: {total_conversations}")
    print(f"   Total messages: {total_messages}")
    print(f"   Total summaries: {total_summaries}")
    print(f"   Average messages per conversation: {total_messages / total_conversations:.1f}")
    
    session.close()


if __name__ == "__main__":
    load_samsum_to_database()