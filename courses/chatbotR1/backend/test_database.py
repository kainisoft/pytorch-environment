#!/usr/bin/env python3
"""
Simple test script to verify database setup and CRUD operations.
"""
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.database import (
    init_database, 
    SessionLocal, 
    ConversationCRUD, 
    MessageCRUD,
    check_connection
)


def test_database_setup():
    """Test basic database setup and operations."""
    print("Testing database setup...")
    
    # Test 1: Check connection
    print("1. Testing database connection...")
    if not check_connection():
        print("   ✗ Database connection failed")
        return False
    print("   ✓ Database connection successful")
    
    # Test 2: Initialize database
    print("2. Initializing database...")
    if not init_database():
        print("   ✗ Database initialization failed")
        return False
    print("   ✓ Database initialization successful")
    
    # Test 3: Test CRUD operations
    print("3. Testing CRUD operations...")
    db = SessionLocal()
    
    try:
        # Create a conversation
        conversation = ConversationCRUD.create(db, "Test Conversation")
        print(f"   ✓ Created conversation: {conversation.id}")
        
        # Create messages
        msg1 = MessageCRUD.create(db, conversation.id, "Hello, AI!", is_user=True)
        msg2 = MessageCRUD.create(db, conversation.id, "Hello! How can I help you?", is_user=False)
        print(f"   ✓ Created {2} messages")
        
        # Retrieve conversation with messages
        conv_with_messages = ConversationCRUD.get_with_messages(db, conversation.id)
        if not conv_with_messages or len(conv_with_messages.messages) != 2:
            print("   ✗ Failed to retrieve conversation with messages")
            return False
        print(f"   ✓ Retrieved conversation with {len(conv_with_messages.messages)} messages")
        
        # Test message retrieval
        messages = MessageCRUD.get_by_conversation(db, conversation.id)
        if len(messages) != 2:
            print("   ✗ Failed to retrieve messages")
            return False
        print(f"   ✓ Retrieved {len(messages)} messages")
        
        # Test conversation listing
        conversations = ConversationCRUD.get_all(db, limit=10)
        if len(conversations) == 0:
            print("   ✗ Failed to retrieve conversations")
            return False
        print(f"   ✓ Retrieved {len(conversations)} conversations")
        
        # Clean up test data
        ConversationCRUD.delete(db, conversation.id)
        print("   ✓ Cleaned up test data")
        
    except Exception as e:
        print(f"   ✗ CRUD operations failed: {e}")
        return False
    finally:
        db.close()
    
    print("✓ All database tests passed!")
    return True


if __name__ == "__main__":
    success = test_database_setup()
    sys.exit(0 if success else 1)