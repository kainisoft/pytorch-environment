#!/usr/bin/env python3
"""
Database management script for the AI Chatbot application.

This script provides commands for database initialization, migration,
and maintenance operations.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.database.init_db import (
    init_database,
    create_sample_data,
    reset_database,
    verify_database_setup
)
from app.database.connection import check_connection


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_init(args):
    """Initialize the database with tables."""
    print("Initializing database...")
    if init_database():
        print("✓ Database initialization successful")
        return True
    else:
        print("✗ Database initialization failed")
        return False


def cmd_sample_data(args):
    """Create sample data for testing."""
    print("Creating sample data...")
    if create_sample_data():
        print("✓ Sample data created successfully")
        return True
    else:
        print("✗ Failed to create sample data")
        return False


def cmd_reset(args):
    """Reset the database (WARNING: deletes all data)."""
    if not args.force:
        response = input("This will delete ALL data. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled")
            return False
    
    print("Resetting database...")
    if reset_database():
        print("✓ Database reset successful")
        return True
    else:
        print("✗ Database reset failed")
        return False


def cmd_verify(args):
    """Verify database setup and connectivity."""
    print("Verifying database setup...")
    
    # Check connection
    if not check_connection():
        print("✗ Database connection failed")
        return False
    print("✓ Database connection successful")
    
    # Verify setup
    if verify_database_setup():
        print("✓ Database verification successful")
        return True
    else:
        print("✗ Database verification failed")
        return False


def cmd_status(args):
    """Show database status and statistics."""
    try:
        from app.database import SessionLocal, ConversationCRUD
        
        print("Database Status:")
        print("=" * 40)
        
        # Check connection
        if check_connection():
            print("✓ Connection: OK")
        else:
            print("✗ Connection: FAILED")
            return False
        
        # Get statistics
        db = SessionLocal()
        try:
            conv_count = ConversationCRUD.get_count(db)
            conversations = ConversationCRUD.get_all(db, limit=5)
            
            print(f"✓ Total conversations: {conv_count}")
            
            if conversations:
                print("\nRecent conversations:")
                for conv in conversations:
                    print(f"  - {conv.title} ({conv.message_count} messages)")
            else:
                print("  No conversations found")
                
        finally:
            db.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Error getting database status: {e}")
        return False


def main():
    """Main entry point for the database management script."""
    parser = argparse.ArgumentParser(
        description="Database management for AI Chatbot application"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database tables")
    init_parser.set_defaults(func=cmd_init)
    
    # Sample data command
    sample_parser = subparsers.add_parser("sample", help="Create sample data")
    sample_parser.set_defaults(func=cmd_sample_data)
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset database (deletes all data)")
    reset_parser.add_argument(
        "--force", 
        action="store_true", 
        help="Skip confirmation prompt"
    )
    reset_parser.set_defaults(func=cmd_reset)
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify database setup")
    verify_parser.set_defaults(func=cmd_verify)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show database status")
    status_parser.set_defaults(func=cmd_status)
    
    # Setup command (init + sample data)
    setup_parser = subparsers.add_parser("setup", help="Complete setup (init + sample data)")
    setup_parser.set_defaults(func=lambda args: cmd_init(args) and cmd_sample_data(args))
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    try:
        success = args.func(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())