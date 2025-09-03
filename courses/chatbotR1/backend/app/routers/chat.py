"""
Chat router.

This module provides chat-related API endpoints for message handling
and conversation management.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.database.connection import get_db
from app.database.crud import ConversationCRUD, MessageCRUD

logger = get_logger(__name__)
router = APIRouter()


# Request Models
class ChatMessageRequest(BaseModel):
    """Chat message request model."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message content")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")


class ConversationCreateRequest(BaseModel):
    """Conversation creation request model."""
    title: str = Field(..., min_length=1, max_length=255, description="Conversation title")


# Response Models
class MessageResponse(BaseModel):
    """Individual message response model."""
    id: int
    content: str
    is_user: bool
    timestamp: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    conversation_id: str
    timestamp: datetime


class ConversationSummary(BaseModel):
    """Conversation summary response model."""
    id: str
    title: str
    updated_at: datetime
    message_count: int

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Conversation list response model."""
    conversations: List[ConversationSummary]


class ConversationMessagesResponse(BaseModel):
    """Conversation messages response model."""
    conversation_id: str
    title: str
    messages: List[MessageResponse]


class ConversationCreateResponse(BaseModel):
    """Conversation creation response model."""
    id: str
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class DeleteResponse(BaseModel):
    """Delete operation response model."""
    success: bool
    message: str


@router.post("/send", response_model=ChatResponse)
async def send_message(
    chat_message: ChatMessageRequest,
    db: Session = Depends(get_db)
):
    """
    Send a message to the AI chatbot.
    
    Args:
        chat_message: The chat message request
        db: Database session
        
    Returns:
        ChatResponse: AI response with conversation ID and timestamp
        
    Raises:
        HTTPException: If conversation not found or processing fails
    """
    logger.info("Chat message received", message_length=len(chat_message.message))
    
    try:
        # Handle conversation creation or validation
        if chat_message.conversation_id:
            # Validate existing conversation
            conversation = ConversationCRUD.get_by_id(db, chat_message.conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation {chat_message.conversation_id} not found"
                )
            conversation_id = chat_message.conversation_id
        else:
            # Create new conversation with auto-generated title
            title = _generate_conversation_title(chat_message.message)
            conversation = ConversationCRUD.create(db, title)
            conversation_id = conversation.id
            logger.info(f"Created new conversation: {conversation_id}")

        # Store user message
        user_message = MessageCRUD.create(
            db=db,
            conversation_id=conversation_id,
            content=chat_message.message,
            is_user=True
        )
        
        # TODO: Implement actual AI processing
        # For now, return a placeholder response
        ai_response_text = f"I received your message: '{chat_message.message}'. This is a placeholder response until the AI engine is integrated."
        
        # Store AI response
        ai_message = MessageCRUD.create(
            db=db,
            conversation_id=conversation_id,
            content=ai_response_text,
            is_user=False
        )
        
        logger.info(f"Processed message exchange in conversation {conversation_id}")
        
        return ChatResponse(
            response=ai_response_text,
            conversation_id=conversation_id,
            timestamp=ai_message.timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )


@router.get("/conversations", response_model=ConversationListResponse)
async def get_conversations(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get list of user conversations.
    
    Args:
        limit: Maximum number of conversations to return (default: 50)
        offset: Number of conversations to skip (default: 0)
        db: Database session
        
    Returns:
        ConversationListResponse: List of conversations with metadata
    """
    logger.info("Conversations list requested", limit=limit, offset=offset)
    
    try:
        conversations = ConversationCRUD.get_all(db, limit=limit, offset=offset)
        
        # Convert to response format with message counts
        conversation_summaries = []
        for conv in conversations:
            message_count = MessageCRUD.get_count_by_conversation(db, conv.id)
            conversation_summaries.append(
                ConversationSummary(
                    id=conv.id,
                    title=conv.title,
                    updated_at=conv.updated_at,
                    message_count=message_count
                )
            )
        
        logger.info(f"Retrieved {len(conversation_summaries)} conversations")
        
        return ConversationListResponse(conversations=conversation_summaries)
        
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationMessagesResponse)
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get messages for a specific conversation.
    
    Args:
        conversation_id: The conversation identifier
        limit: Maximum number of messages to return (default: 100)
        offset: Number of messages to skip (default: 0)
        db: Database session
        
    Returns:
        ConversationMessagesResponse: Conversation with messages
        
    Raises:
        HTTPException: If conversation not found
    """
    logger.info("Conversation messages requested", conversation_id=conversation_id)
    
    try:
        # Get conversation
        conversation = ConversationCRUD.get_by_id(db, conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        # Get messages
        messages = MessageCRUD.get_by_conversation(
            db, conversation_id, limit=limit, offset=offset
        )
        
        # Convert to response format
        message_responses = [
            MessageResponse(
                id=msg.id,
                content=msg.content,
                is_user=msg.is_user,
                timestamp=msg.timestamp
            )
            for msg in messages
        ]
        
        logger.info(f"Retrieved {len(message_responses)} messages for conversation {conversation_id}")
        
        return ConversationMessagesResponse(
            conversation_id=conversation_id,
            title=conversation.title,
            messages=message_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation messages"
        )


@router.post("/conversations", response_model=ConversationCreateResponse)
async def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new conversation.
    
    Args:
        request: Conversation creation request
        db: Database session
        
    Returns:
        ConversationCreateResponse: Created conversation details
    """
    logger.info("New conversation creation requested", title=request.title)
    
    try:
        conversation = ConversationCRUD.create(db, request.title)
        
        logger.info(f"Created conversation: {conversation.id}")
        
        return ConversationCreateResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at
        )
        
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )


@router.delete("/conversations/{conversation_id}", response_model=DeleteResponse)
async def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a conversation and all its messages.
    
    Args:
        conversation_id: The conversation identifier
        db: Database session
        
    Returns:
        DeleteResponse: Deletion result
        
    Raises:
        HTTPException: If conversation not found
    """
    logger.info("Conversation deletion requested", conversation_id=conversation_id)
    
    try:
        # Check if conversation exists
        conversation = ConversationCRUD.get_by_id(db, conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        # Delete conversation (messages will be deleted due to cascade)
        success = ConversationCRUD.delete(db, conversation_id)
        
        if success:
            logger.info(f"Successfully deleted conversation: {conversation_id}")
            return DeleteResponse(
                success=True,
                message=f"Conversation {conversation_id} deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete conversation"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )


def _generate_conversation_title(first_message: str) -> str:
    """
    Generate a conversation title based on the first message.
    
    Args:
        first_message: The first user message
        
    Returns:
        str: Generated conversation title
    """
    # Simple title generation - take first 50 characters and clean up
    title = first_message.strip()
    if len(title) > 50:
        title = title[:47] + "..."
    
    # Remove newlines and extra spaces
    title = " ".join(title.split())
    
    # Fallback title if message is empty or only whitespace
    if not title:
        title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    return title