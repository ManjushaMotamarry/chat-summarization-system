"""
Database models for chat summarization system.
Defines tables for conversations, messages, and summaries.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

# Base class for all models
Base = declarative_base()


class Conversation(Base):
    """
    Stores metadata about each conversation.
    One conversation can have many messages.
    """
    __tablename__ = 'conversations'
    
    conversation_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    channel = Column(String(50))  # e.g., 'web', 'mobile', 'email'
    status = Column(String(50))   # e.g., 'completed', 'escalated'
    
    # Relationships
    messages = relationship("Message", back_populates="conversation")
    summary = relationship("Summary", back_populates="conversation", uselist=False)
    
    def __repr__(self):
        return f"<Conversation(id={self.conversation_id}, status={self.status})>"


class Message(Base):
    """
    Stores individual messages within conversations.
    Each message belongs to one conversation.
    """
    __tablename__ = 'messages'
    
    message_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey('conversations.conversation_id'))
    sender = Column(String(50))      # e.g., 'user', 'bot', 'agent'
    message_text = Column(Text)      # The actual message content
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.message_id}, sender={self.sender})>"


class Summary(Base):
    """
    Stores AI-generated summaries for conversations.
    Each conversation has one summary (the most recent).
    """
    __tablename__ = 'summaries'
    
    summary_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey('conversations.conversation_id'), unique=True)
    summary_text = Column(Text)
    model_version = Column(String(100))  # e.g., 'bart-base-v1'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    conversation = relationship("Conversation", back_populates="summary")
    
    def __repr__(self):
        return f"<Summary(id={self.summary_id}, model={self.model_version})>"


# Database connection function
def get_database_engine(db_path="data/chat_conversations.db"):
    """
    Creates a database engine.
    For now, we use SQLite (file-based database).
    """
    engine = create_engine(f'sqlite:///{db_path}', echo=True)
    return engine


def create_tables(engine):
    """
    Creates all tables in the database if they don't exist.
    """
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully!")


def get_session(engine):
    """
    Creates a session for interacting with the database.
    """
    Session = sessionmaker(bind=engine)
    return Session()