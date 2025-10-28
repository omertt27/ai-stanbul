"""
Alembic migration for personalization and feedback tables

Revision ID: add_personalization_feedback
Create Date: 2024-01-15 12:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'add_personalization_feedback'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create user_preferences table
    op.create_table(
        'user_preferences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('cuisines', sa.JSON(), nullable=True),
        sa.Column('price_ranges', sa.JSON(), nullable=True),
        sa.Column('districts', sa.JSON(), nullable=True),
        sa.Column('activity_types', sa.JSON(), nullable=True),
        sa.Column('attraction_types', sa.JSON(), nullable=True),
        sa.Column('transportation_modes', sa.JSON(), nullable=True),
        sa.Column('time_of_day', sa.JSON(), nullable=True),
        sa.Column('interaction_count', sa.Integer(), nullable=True, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_user_preferences_user_id', 'user_preferences', ['user_id'], unique=True)
    
    # Create user_interactions table
    op.create_table(
        'user_interactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('interaction_id', sa.String(length=255), nullable=False),
        sa.Column('interaction_type', sa.String(length=50), nullable=False),
        sa.Column('item_id', sa.String(length=255), nullable=True),
        sa.Column('item_data', sa.JSON(), nullable=True),
        sa.Column('rating', sa.Float(), nullable=True, default=0.5),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_user_interactions_user_id', 'user_interactions', ['user_id'])
    op.create_index('idx_user_interactions_interaction_id', 'user_interactions', ['interaction_id'], unique=True)
    op.create_index('idx_user_interactions_timestamp', 'user_interactions', ['timestamp'])
    op.create_index('idx_user_interactions_type', 'user_interactions', ['interaction_type'])
    
    # Create user_feedback table
    op.create_table(
        'user_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('interaction_id', sa.String(length=255), nullable=False),
        sa.Column('satisfaction_score', sa.Float(), nullable=False),
        sa.Column('was_helpful', sa.Boolean(), nullable=True, default=True),
        sa.Column('response_quality', sa.Float(), nullable=True, default=3.0),
        sa.Column('speed_rating', sa.Float(), nullable=True, default=3.0),
        sa.Column('intent', sa.String(length=100), nullable=True),
        sa.Column('feature', sa.String(length=100), nullable=True),
        sa.Column('comments', sa.Text(), nullable=True, default=''),
        sa.Column('issues', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_user_feedback_user_id', 'user_feedback', ['user_id'])
    op.create_index('idx_user_feedback_interaction_id', 'user_feedback', ['interaction_id'], unique=True)
    op.create_index('idx_user_feedback_created_at', 'user_feedback', ['created_at'])
    op.create_index('idx_user_feedback_intent', 'user_feedback', ['intent'])
    op.create_index('idx_user_feedback_feature', 'user_feedback', ['feature'])
    op.create_index('idx_user_feedback_satisfaction', 'user_feedback', ['satisfaction_score'])
    
    # Create ab_test_variants table
    op.create_table(
        'ab_test_variants',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('test_name', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('variant', sa.String(length=50), nullable=False),
        sa.Column('experiment_data', sa.JSON(), nullable=True),
        sa.Column('assigned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_ab_test_variants_test_user', 'ab_test_variants', ['test_name', 'user_id'], unique=True)
    op.create_index('idx_ab_test_variants_test_name', 'ab_test_variants', ['test_name'])
    
    # Create ab_test_results table
    op.create_table(
        'ab_test_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('test_name', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('variant', sa.String(length=50), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('interaction_data', sa.JSON(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_ab_test_results_test_name', 'ab_test_results', ['test_name'])
    op.create_index('idx_ab_test_results_variant', 'ab_test_results', ['variant'])
    op.create_index('idx_ab_test_results_metric', 'ab_test_results', ['metric_name'])
    
    # Create cf_interactions table
    op.create_table(
        'cf_interactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('item_id', sa.String(length=255), nullable=False),
        sa.Column('item_type', sa.String(length=50), nullable=False),
        sa.Column('interaction_score', sa.Float(), nullable=False),
        sa.Column('context_data', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_cf_interactions_user_item', 'cf_interactions', ['user_id', 'item_id'], unique=True)
    op.create_index('idx_cf_interactions_item_type', 'cf_interactions', ['item_type'])
    op.create_index('idx_cf_interactions_score', 'cf_interactions', ['interaction_score'])


def downgrade():
    op.drop_table('cf_interactions')
    op.drop_table('ab_test_results')
    op.drop_table('ab_test_variants')
    op.drop_table('user_feedback')
    op.drop_table('user_interactions')
    op.drop_table('user_preferences')
