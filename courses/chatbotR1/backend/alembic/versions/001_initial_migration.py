"""Initial migration - create conversations and messages tables

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('conversation_id', sa.String(36), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('is_user', sa.Boolean, nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for better performance
    op.create_index('ix_conversations_updated_at', 'conversations', ['updated_at'])
    op.create_index('ix_messages_conversation_id', 'messages', ['conversation_id'])
    op.create_index('ix_messages_timestamp', 'messages', ['timestamp'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_messages_timestamp')
    op.drop_index('ix_messages_conversation_id')
    op.drop_index('ix_conversations_updated_at')
    
    # Drop tables
    op.drop_table('messages')
    op.drop_table('conversations')