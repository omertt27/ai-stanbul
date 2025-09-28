"""add_cascade_delete_to_user_sessions_fks

Revision ID: 13453b906bab
Revises: 69ac03bf9108
Create Date: 2025-09-28 23:56:32.750327

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '13453b906bab'
down_revision: Union[str, Sequence[str], None] = '69ac03bf9108'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Add CASCADE DELETE to user_sessions foreign keys."""
    
    # Drop existing foreign key constraints and recreate with CASCADE DELETE
    
    # 1. user_memory table
    op.drop_constraint('user_memory_session_id_fkey', 'user_memory', type_='foreignkey')
    op.create_foreign_key(
        'user_memory_session_id_fkey', 
        'user_memory', 'user_sessions',
        ['session_id'], ['session_id'],
        ondelete='CASCADE'
    )
    
    # 2. user_preferences table  
    op.drop_constraint('user_preferences_session_id_fkey', 'user_preferences', type_='foreignkey')
    op.create_foreign_key(
        'user_preferences_session_id_fkey',
        'user_preferences', 'user_sessions', 
        ['session_id'], ['session_id'],
        ondelete='CASCADE'
    )
    
    # 3. conversation_context table
    op.drop_constraint('conversation_context_session_id_fkey', 'conversation_context', type_='foreignkey')
    op.create_foreign_key(
        'conversation_context_session_id_fkey',
        'conversation_context', 'user_sessions',
        ['session_id'], ['session_id'],
        ondelete='CASCADE'
    )
    
    # 4. user_interactions table
    op.drop_constraint('user_interactions_session_id_fkey', 'user_interactions', type_='foreignkey')
    op.create_foreign_key(
        'user_interactions_session_id_fkey',
        'user_interactions', 'user_sessions',
        ['session_id'], ['session_id'], 
        ondelete='CASCADE'
    )


def downgrade() -> None:
    """Downgrade schema - Remove CASCADE DELETE from foreign keys."""
    
    # Revert back to foreign keys without CASCADE DELETE
    
    # 1. user_memory table
    op.drop_constraint('user_memory_session_id_fkey', 'user_memory', type_='foreignkey')
    op.create_foreign_key(
        'user_memory_session_id_fkey',
        'user_memory', 'user_sessions',
        ['session_id'], ['session_id']
    )
    
    # 2. user_preferences table
    op.drop_constraint('user_preferences_session_id_fkey', 'user_preferences', type_='foreignkey')
    op.create_foreign_key(
        'user_preferences_session_id_fkey',
        'user_preferences', 'user_sessions',
        ['session_id'], ['session_id']
    )
    
    # 3. conversation_context table
    op.drop_constraint('conversation_context_session_id_fkey', 'conversation_context', type_='foreignkey')
    op.create_foreign_key(
        'conversation_context_session_id_fkey',
        'conversation_context', 'user_sessions',
        ['session_id'], ['session_id']
    )
    
    # 4. user_interactions table
    op.drop_constraint('user_interactions_session_id_fkey', 'user_interactions', type_='foreignkey')
    op.create_foreign_key(
        'user_interactions_session_id_fkey',
        'user_interactions', 'user_sessions',
        ['session_id'], ['session_id']
    )
