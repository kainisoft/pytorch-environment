"""
Database initialization script.
"""
import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.database.connection import SessionLocal, create_tables, check_connection
from app.database.crud import ConversationCRUD, MessageCRUD

logger = logging.getLogger(__name__)


def init_database() -> bool:
    """
    Initialize the database with tables and optional sample data.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Check database connection
        if not check_connection():
            logger.error("Database connection failed")
            return False
        
        # Create tables
        create_tables()
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def create_sample_data(db: Optional[Session] = None) -> bool:
    """
    Create sample conversations and messages for testing.
    
    Args:
        db: Optional database session. If None, creates a new session.
        
    Returns:
        bool: True if sample data created successfully, False otherwise
    """
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        # Check if sample data already exists
        existing_conversations = ConversationCRUD.get_count(db)
        if existing_conversations > 0:
            logger.info("Sample data already exists, skipping creation")
            return True
        
        # Create sample conversation 1
        conv1 = ConversationCRUD.create(db, "Getting Started with AI")
        MessageCRUD.create(db, conv1.id, "Hello! How can I help you today?", is_user=False)
        MessageCRUD.create(db, conv1.id, "I'd like to learn about artificial intelligence", is_user=True)
        MessageCRUD.create(
            db, 
            conv1.id, 
            "Great! AI is a fascinating field. What specific aspect would you like to explore?", 
            is_user=False
        )
        
        # Create sample conversation 2
        conv2 = ConversationCRUD.create(db, "Python Programming Help")
        MessageCRUD.create(db, conv2.id, "Hi there! I'm here to help with your questions.", is_user=False)
        MessageCRUD.create(db, conv2.id, "Can you help me understand Python decorators?", is_user=True)
        MessageCRUD.create(
            db, 
            conv2.id, 
            "Absolutely! Decorators are a powerful feature in Python that allow you to modify or extend the behavior of functions or classes.", 
            is_user=False
        )
        
        # Create sample conversation 3
        conv3 = ConversationCRUD.create(db, "Quick Question")
        MessageCRUD.create(db, conv3.id, "What's the weather like today?", is_user=True)
        MessageCRUD.create(
            db, 
            conv3.id, 
            "I don't have access to real-time weather data, but I'd be happy to help you find weather information or discuss other topics!", 
            is_user=False
        )
        
        logger.info("Sample data created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        db.rollback()
        return False
        
    finally:
        if close_db:
            db.close()


def reset_database() -> bool:
    """
    Reset the database by dropping and recreating all tables.
    
    WARNING: This will delete all existing data!
    
    Returns:
        bool: True if reset successful, False otherwise
    """
    try:
        from app.database.connection import drop_tables
        
        logger.warning("Resetting database - all data will be lost!")
        
        # Drop all tables
        drop_tables()
        
        # Recreate tables
        create_tables()
        
        logger.info("Database reset completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return False


def verify_database_setup() -> bool:
    """
    Verify that the database is properly set up with all required tables.
    
    Returns:
        bool: True if database setup is valid, False otherwise
    """
    try:
        db = SessionLocal()
        
        # Test basic operations
        conversations = ConversationCRUD.get_all(db, limit=1)
        logger.info(f"Database verification: Found {len(conversations)} conversations")
        
        # Test creating and deleting a test conversation
        test_conv = ConversationCRUD.create(db, "Database Test Conversation")
        test_message = MessageCRUD.create(db, test_conv.id, "Test message", is_user=True)
        
        # Verify the data was created
        retrieved_conv = ConversationCRUD.get_with_messages(db, test_conv.id)
        if not retrieved_conv or len(retrieved_conv.messages) != 1:
            raise Exception("Failed to retrieve test conversation with messages")
        
        # Clean up test data
        ConversationCRUD.delete(db, test_conv.id)
        
        logger.info("Database verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False
        
    finally:
        db.close()


if __name__ == "__main__":
    """
    Run database initialization when script is executed directly.
    """
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing database...")
    if init_database():
        print("✓ Database initialization successful")
        
        print("Creating sample data...")
        if create_sample_data():
            print("✓ Sample data created successfully")
        else:
            print("✗ Failed to create sample data")
        
        print("Verifying database setup...")
        if verify_database_setup():
            print("✓ Database verification successful")
        else:
            print("✗ Database verification failed")
    else:
        print("✗ Database initialization failed")