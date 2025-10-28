"""
Test script to insert sample data and verify database works.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import get_database_engine, get_session, Conversation, Message, Summary


def test_database():
    print("🧪 Testing database with sample data...\n")
    
    # Get database connection
    engine = get_database_engine(db_path="data/chat_conversations.db")
    session = get_session(engine)
    
    # Create a sample conversation
    print("1️⃣ Creating a sample conversation...")
    conversation = Conversation(
        channel="web",
        status="completed",
        created_at=datetime.now()
    )
    session.add(conversation)
    session.commit()  # Save to database
    print(f"   ✅ Created: {conversation}")
    
    # Add messages to this conversation
    print("\n2️⃣ Adding messages to the conversation...")
    messages = [
        Message(
            conversation_id=conversation.conversation_id,
            sender="user",
            message_text="Hi, I need help with my order",
            timestamp=datetime.now()
        ),
        Message(
            conversation_id=conversation.conversation_id,
            sender="bot",
            message_text="Hello! I'd be happy to help. What's your order number?",
            timestamp=datetime.now()
        ),
        Message(
            conversation_id=conversation.conversation_id,
            sender="user",
            message_text="It's ORDER-12345",
            timestamp=datetime.now()
        ),
        Message(
            conversation_id=conversation.conversation_id,
            sender="bot",
            message_text="Let me check that for you. One moment please.",
            timestamp=datetime.now()
        )
    ]
    
    for msg in messages:
        session.add(msg)
        print(f"   ✅ Added: {msg.sender}: '{msg.message_text[:30]}...'")
    
    session.commit()
    
    # Create a summary for this conversation
    print("\n3️⃣ Creating a summary...")
    summary = Summary(
        conversation_id=conversation.conversation_id,
        summary_text="Customer inquired about order ORDER-12345. Bot is checking order status.",
        model_version="test-v1"
    )
    session.add(summary)
    session.commit()
    print(f"   ✅ Created: {summary}")
    
    # Now query the data back
    print("\n4️⃣ Querying data back from database...")
    
    # Get all conversations
    all_convos = session.query(Conversation).all()
    print(f"   📊 Total conversations: {len(all_convos)}")
    
    # Get all messages for conversation 1
    conv_messages = session.query(Message).filter_by(conversation_id=1).all()
    print(f"   💬 Messages in conversation 1: {len(conv_messages)}")
    
    # Get the summary
    conv_summary = session.query(Summary).filter_by(conversation_id=1).first()
    print(f"   📝 Summary: '{conv_summary.summary_text[:50]}...'")
    
    print("\n✅ Database test successful! Everything is working! 🎉")
    
    session.close()


if __name__ == "__main__":
    test_database()