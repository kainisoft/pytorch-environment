# Database Documentation

This document describes the database setup, models, and usage for the AI Chatbot application.

## Overview

The application uses SQLAlchemy ORM with SQLite for development and PostgreSQL support for production. The database stores conversations and messages with proper relationships and constraints.

## Database Schema

### Tables

#### conversations
- `id` (VARCHAR(36), PRIMARY KEY) - UUID for conversation
- `title` (VARCHAR(255), NOT NULL) - Conversation title
- `created_at` (DATETIME, DEFAULT NOW()) - Creation timestamp
- `updated_at` (DATETIME, DEFAULT NOW(), ON UPDATE NOW()) - Last update timestamp

#### messages
- `id` (INTEGER, PRIMARY KEY, AUTOINCREMENT) - Message ID
- `conversation_id` (VARCHAR(36), FOREIGN KEY) - References conversations.id
- `content` (TEXT, NOT NULL) - Message content
- `is_user` (BOOLEAN, NOT NULL) - True for user messages, False for AI responses
- `timestamp` (DATETIME, DEFAULT NOW()) - Message timestamp

### Relationships
- One conversation can have many messages (1:N)
- Messages belong to one conversation
- Cascade delete: Deleting a conversation removes all its messages

## Models

### Conversation Model
```python
from app.models import Conversation

# Create a new conversation
conversation = Conversation(title="My Chat")

# Access messages
messages = conversation.messages
message_count = conversation.message_count
```

### Message Model
```python
from app.models import Message

# Create a new message
message = Message(
    conversation_id="uuid-here",
    content="Hello!",
    is_user=True
)

# Access conversation
conversation = message.conversation
```

## CRUD Operations

### Using ConversationCRUD

```python
from app.database import SessionLocal, ConversationCRUD

db = SessionLocal()

# Create conversation
conversation = ConversationCRUD.create(db, "New Chat")

# Get conversation by ID
conversation = ConversationCRUD.get_by_id(db, conversation_id)

# Get conversation with messages
conversation = ConversationCRUD.get_with_messages(db, conversation_id)

# Get all conversations
conversations = ConversationCRUD.get_all(db, limit=50)

# Update conversation title
conversation = ConversationCRUD.update_title(db, conversation_id, "New Title")

# Delete conversation
success = ConversationCRUD.delete(db, conversation_id)

db.close()
```

### Using MessageCRUD

```python
from app.database import SessionLocal, MessageCRUD

db = SessionLocal()

# Create message
message = MessageCRUD.create(db, conversation_id, "Hello!", is_user=True)

# Get messages for conversation
messages = MessageCRUD.get_by_conversation(db, conversation_id)

# Get latest messages
latest = MessageCRUD.get_latest_messages(db, conversation_id, count=10)

# Get message count
count = MessageCRUD.get_count_by_conversation(db, conversation_id)

db.close()
```

## Database Management

### Using the Management Script

The `manage_db.py` script provides convenient database operations:

```bash
# Initialize database
python manage_db.py init

# Create sample data
python manage_db.py sample

# Complete setup (init + sample)
python manage_db.py setup

# Check database status
python manage_db.py status

# Verify database setup
python manage_db.py verify

# Reset database (WARNING: deletes all data)
python manage_db.py reset --force
```

### Using Python Functions

```python
from app.database import init_database, create_sample_data

# Initialize database
init_database()

# Create sample data
create_sample_data()
```

## Database Connection

### Dependency Injection (FastAPI)

```python
from fastapi import Depends
from sqlalchemy.orm import Session
from app.database import get_db

@app.get("/conversations")
def get_conversations(db: Session = Depends(get_db)):
    return ConversationCRUD.get_all(db)
```

### Manual Session Management

```python
from app.database import SessionLocal

db = SessionLocal()
try:
    # Database operations here
    conversations = ConversationCRUD.get_all(db)
finally:
    db.close()
```

## Migrations

### Alembic Setup

The project uses Alembic for database migrations:

```bash
# Generate new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Downgrade migration
alembic downgrade -1
```

### Migration Files

- `alembic/versions/001_initial_migration.py` - Initial schema
- `alembic/env.py` - Alembic environment configuration
- `alembic.ini` - Alembic configuration file

## Configuration

### Environment Variables

```bash
# Database URL (SQLite for development)
DATABASE_URL=sqlite:///./chatbot.db

# PostgreSQL for production
DATABASE_URL=postgresql://user:password@localhost/chatbot_db

# Debug mode (enables SQL query logging)
DEBUG=true
```

### Settings

Database settings are managed in `app/core/config.py`:

```python
from app.core.config import get_settings

settings = get_settings()
database_url = settings.database_url
```

## Testing

### Running Database Tests

```bash
# Run basic database test
python test_database.py

# Run with verbose output
python test_database.py -v
```

### Test Database Setup

For testing, use a separate database:

```python
# In test configuration
DATABASE_URL=sqlite:///./test_chatbot.db
```

## Performance Considerations

### Indexes

The database includes indexes for:
- `conversations.updated_at` - For sorting conversations
- `messages.conversation_id` - For message lookups
- `messages.timestamp` - For message ordering

### Query Optimization

- Use `get_with_messages()` to load conversations with messages in one query
- Use `get_latest_messages()` for recent message retrieval
- Implement pagination for large conversation lists

### Connection Pooling

For production with PostgreSQL:
- Configure connection pool size
- Use async sessions for high concurrency
- Monitor connection usage

## Troubleshooting

### Common Issues

1. **Foreign Key Constraints**: Ensure SQLite pragma is enabled
2. **Migration Conflicts**: Check Alembic revision history
3. **Connection Errors**: Verify database URL and permissions
4. **Performance Issues**: Check indexes and query patterns

### Debug Mode

Enable debug mode to see SQL queries:

```bash
DEBUG=true
```

### Logging

Database operations are logged at INFO level:

```python
import logging
logging.getLogger('app.database').setLevel(logging.DEBUG)
```

## Security

### Best Practices

1. Use parameterized queries (SQLAlchemy handles this)
2. Validate input data with Pydantic models
3. Use proper database permissions
4. Regular backups for production
5. Monitor for SQL injection attempts

### Data Protection

- Conversation data is isolated by conversation ID
- No user authentication implemented yet (future enhancement)
- Consider encryption for sensitive data in production