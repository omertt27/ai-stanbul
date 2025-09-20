"""Add analytics tracking tables

Revision ID: 69ac03bf9108
Revises: 8017f37ca05c
Create Date: 2025-09-20 15:21:32.874905

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '69ac03bf9108'
down_revision: Union[str, Sequence[str], None] = '8017f37ca05c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Add analytics tracking tables."""
    # Create query analytics table
    op.create_table(
        'query_analytics',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('query_hash', sa.String(50), index=True),
        sa.Column('query_length', sa.Integer),
        sa.Column('response_source', sa.String(20)),  # 'ai', 'fallback', 'cache'
        sa.Column('response_time_ms', sa.Float),
        sa.Column('success', sa.Boolean),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('user_session', sa.String(100), index=True)
    )
    
    # Create cache analytics table
    op.create_table(
        'cache_analytics',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('cache_key', sa.String(100), index=True),
        sa.Column('hit', sa.Boolean),  # True for hit, False for miss
        sa.Column('created_at', sa.DateTime, default=sa.func.now())
    )
    
    # Create rate limit violations table
    op.create_table(
        'rate_limit_violations',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('client_ip', sa.String(45), index=True),
        sa.Column('endpoint', sa.String(100)),
        sa.Column('violation_count', sa.Integer),
        sa.Column('created_at', sa.DateTime, default=sa.func.now())
    )


def downgrade() -> None:
    """Downgrade schema - Remove analytics tracking tables."""
    op.drop_table('rate_limit_violations')
    op.drop_table('cache_analytics')
    op.drop_table('query_analytics')
