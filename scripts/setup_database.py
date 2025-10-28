"""
Script to initialize the database and create tables.
Run this once to set up your database.
"""

import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import get_database_engine, create_tables


def main():
    print("🔧 Setting up database...")
    
    # Create engine (connection to database)
    engine = get_database_engine(db_path="data/chat_conversations.db")
    
    # Create all tables
    create_tables(engine)
    
    print("\n✅ Database setup complete!")
    print(f"📁 Database file created at: data/chat_conversations.db")


if __name__ == "__main__":
    main()