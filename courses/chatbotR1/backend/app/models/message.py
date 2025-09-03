"""
Message model for storing individual chat messages.
"""
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class Message(Base):
    """
    SQLAlchemy model for messages table.
    
    Stores individual messages within conversations, including content,
    user/AI identification, and timestamps.
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_user: Mapped[bool] = mapped_column(Boolean, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )

    # Relationship to conversation
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", 
        back_populates="messages"
    )

    def __repr__(self) -> str:
        sender = "User" if self.is_user else "AI"
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(id={self.id}, sender={sender}, content='{content_preview}')>"