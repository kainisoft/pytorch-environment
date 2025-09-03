"""
Conversation model for storing chat conversations.
"""
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Conversation(Base):
    """
    SQLAlchemy model for conversations table.
    
    Stores conversation metadata including title and timestamps.
    """
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid4())
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationship to messages
    messages: Mapped[List["Message"]] = relationship(
        "Message", 
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.timestamp"
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title={self.title})>"

    @property
    def message_count(self) -> int:
        """Get the number of messages in this conversation."""
        return len(self.messages) if self.messages else 0