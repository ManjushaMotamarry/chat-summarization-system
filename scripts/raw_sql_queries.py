"""
Examples of raw SQL queries on our database.
"""

import sys
import os
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_sql_queries():
    # Connect directly to SQLite database
    conn = sqlite3.connect('data/chat_conversations.db')
    cursor = conn.cursor()
    
    print("🔍 Running Raw SQL Queries\n")
    
    # Query 1: SELECT * FROM conversations (first 5)
    print("1️⃣ SELECT * FROM conversations LIMIT 5:")
    print("="*60)
    cursor.execute("SELECT * FROM conversations LIMIT 5")
    for row in cursor.fetchall():
        print(row)
    
    # Query 2: Count messages per conversation
    print("\n2️⃣ SELECT conversation_id, COUNT(*) FROM messages GROUP BY conversation_id LIMIT 5:")
    print("="*60)
    cursor.execute("""
        SELECT conversation_id, COUNT(*) as message_count 
        FROM messages 
        GROUP BY conversation_id 
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"Conversation {row[0]}: {row[1]} messages")
    
    # Query 3: JOIN conversations and summaries
    print("\n3️⃣ SELECT * FROM conversations JOIN summaries WHERE conversation_id = 100:")
    print("="*60)
    cursor.execute("""
        SELECT c.conversation_id, c.status, s.summary_text
        FROM conversations c
        JOIN summaries s ON c.conversation_id = s.conversation_id
        WHERE c.conversation_id = 100
    """)
    row = cursor.fetchone()
    print(f"ID: {row[0]}, Status: {row[1]}")
    print(f"Summary: {row[2]}")
    
    # Query 4: Find conversations with keyword
    print("\n4️⃣ SELECT * FROM messages WHERE message_text LIKE '%order%' LIMIT 3:")
    print("="*60)
    cursor.execute("""
        SELECT sender, message_text 
        FROM messages 
        WHERE message_text LIKE '%order%' 
        LIMIT 3
    """)
    for row in cursor.fetchall():
        print(f"{row[0]}: {row[1]}")
    
    conn.close()
    print("\n✅ SQL queries complete!")


if __name__ == "__main__":
    run_sql_queries()