"""
Query and display sample data from database.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import get_database_engine, get_session, Conversation, Message, Summary


def show_sample_conversation():
    print("🔍 Querying sample conversation from database...\n")
    
    engine = get_database_engine(db_path="data/chat_conversations.db")
    session = get_session(engine)
    
    # Get a random conversation (let's get conversation #100)
    conversation = session.query(Conversation).filter_by(conversation_id=100).first()
    
    if not conversation:
        print("❌ Conversation not found!")
        return
    
    print(f"📊 Conversation #{conversation.conversation_id}")
    print(f"   Channel: {conversation.channel}")
    print(f"   Status: {conversation.status}")
    print(f"   Created: {conversation.created_at}")
    
    # Get all messages for this conversation
    messages = session.query(Message).filter_by(
        conversation_id=conversation.conversation_id
    ).order_by(Message.timestamp).all()
    
    print(f"\n💬 Messages ({len(messages)} total):")
    print("="*60)
    for msg in messages:
        print(f"{msg.sender}: {msg.message_text}")
    
    # Get the summary
    summary = session.query(Summary).filter_by(
        conversation_id=conversation.conversation_id
    ).first()
    
    print("="*60)
    print(f"\n📝 Summary:")
    print(f"   {summary.summary_text}")
    print(f"   Model: {summary.model_version}")
    
    session.close()
    print("\n✅ Query complete!")


if __name__ == "__main__":
    show_sample_conversation()