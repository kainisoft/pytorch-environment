"""
CRUD (Create, Read, Update, Delete) operations for database models.
"""
import logging
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload

from app.models.conversation import Conversation
from app.models.message import Message

logger = logging.getLogger(__name__)


class ConversationCRUD:
    """CRUD operations for Conversation model."""

    @staticmethod
    def create(db: Session, title: str) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            db: Database session
            title: Conversation title
            
        Returns:
            Conversation: Created conversation object
        """
        conversation = Conversation(
            id=str(uuid4()),
            title=title
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        logger.info(f"Created conversation: {conversation.id}")
        return conversation

    @staticmethod
    def get_by_id(db: Session, conversation_id: str) -> Optional[Conversation]:
        """
        Get conversation by ID.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            
        Returns:
            Optional[Conversation]: Conversation object or None if not found
        """
        return db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

    @staticmethod
    def get_with_messages(db: Session, conversation_id: str) -> Optional[Conversation]:
        """
        Get conversation with all messages loaded.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            
        Returns:
            Optional[Conversation]: Conversation with messages or None if not found
        """
        return db.query(Conversation).options(
            joinedload(Conversation.messages)
        ).filter(
            Conversation.id == conversation_id
        ).first()

    @staticmethod
    def get_all(db: Session, limit: int = 100, offset: int = 0) -> List[Conversation]:
        """
        Get all conversations ordered by updated_at descending.
        
        Args:
            db: Database session
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            
        Returns:
            List[Conversation]: List of conversations
        """
        return db.query(Conversation).order_by(
            desc(Conversation.updated_at)
        ).offset(offset).limit(limit).all()

    @staticmethod
    def update_title(db: Session, conversation_id: str, title: str) -> Optional[Conversation]:
        """
        Update conversation title.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            title: New title
            
        Returns:
            Optional[Conversation]: Updated conversation or None if not found
        """
        conversation = ConversationCRUD.get_by_id(db, conversation_id)
        if conversation:
            conversation.title = title
            conversation.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(conversation)
            logger.info(f"Updated conversation title: {conversation_id}")
        return conversation

    @staticmethod
    def delete(db: Session, conversation_id: str) -> bool:
        """
        Delete conversation and all its messages.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        conversation = ConversationCRUD.get_by_id(db, conversation_id)
        if conversation:
            db.delete(conversation)
            db.commit()
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False

    @staticmethod
    def get_count(db: Session) -> int:
        """
        Get total number of conversations.
        
        Args:
            db: Database session
            
        Returns:
            int: Total conversation count
        """
        return db.query(func.count(Conversation.id)).scalar()


class MessageCRUD:
    """CRUD operations for Message model."""

    @staticmethod
    def create(
        db: Session, 
        conversation_id: str, 
        content: str, 
        is_user: bool
    ) -> Message:
        """
        Create a new message.
        
        Args:
            db: Database session
            conversation_id: ID of the conversation
            content: Message content
            is_user: True if message is from user, False if from AI
            
        Returns:
            Message: Created message object
        """
        message = Message(
            conversation_id=conversation_id,
            content=content,
            is_user=is_user
        )
        db.add(message)
        
        # Update conversation's updated_at timestamp
        conversation = ConversationCRUD.get_by_id(db, conversation_id)
        if conversation:
            conversation.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(message)
        
        logger.info(f"Created message in conversation {conversation_id}")
        return message

    @staticmethod
    def get_by_conversation(
        db: Session, 
        conversation_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """
        Get messages for a conversation ordered by timestamp.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List[Message]: List of messages
        """
        return db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).offset(offset).limit(limit).all()

    @staticmethod
    def get_latest_messages(
        db: Session, 
        conversation_id: str, 
        count: int = 10
    ) -> List[Message]:
        """
        Get the latest N messages from a conversation.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            count: Number of latest messages to return
            
        Returns:
            List[Message]: List of latest messages in chronological order
        """
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(desc(Message.timestamp)).limit(count).all()
        
        # Reverse to get chronological order
        return list(reversed(messages))

    @staticmethod
    def delete_by_conversation(db: Session, conversation_id: str) -> int:
        """
        Delete all messages in a conversation.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            
        Returns:
            int: Number of messages deleted
        """
        count = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).count()
        
        db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).delete()
        
        db.commit()
        logger.info(f"Deleted {count} messages from conversation {conversation_id}")
        return count

    @staticmethod
    def get_count_by_conversation(db: Session, conversation_id: str) -> int:
        """
        Get message count for a conversation.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            
        Returns:
            int: Number of messages in conversation
        """
        return db.query(func.count(Message.id)).filter(
            Message.conversation_id == conversation_id
        ).scalar()


# Convenience functions for common operations
def create_conversation_with_message(
    db: Session, 
    title: str, 
    initial_message: str
) -> tuple[Conversation, Message]:
    """
    Create a new conversation with an initial user message.
    
    Args:
        db: Database session
        title: Conversation title
        initial_message: Initial user message content
        
    Returns:
        tuple[Conversation, Message]: Created conversation and message
    """
    conversation = ConversationCRUD.create(db, title)
    message = MessageCRUD.create(db, conversation.id, initial_message, is_user=True)
    return conversation, message


def get_conversation_summary(db: Session, conversation_id: str) -> Optional[dict]:
    """
    Get conversation summary with basic stats.
    
    Args:
        db: Database session
        conversation_id: Conversation ID
        
    Returns:
        Optional[dict]: Conversation summary or None if not found
    """
    conversation = ConversationCRUD.get_by_id(db, conversation_id)
    if not conversation:
        return None
    
    message_count = MessageCRUD.get_count_by_conversation(db, conversation_id)
    latest_messages = MessageCRUD.get_latest_messages(db, conversation_id, 1)
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "message_count": message_count,
        "latest_message": latest_messages[0].content if latest_messages else None
    }