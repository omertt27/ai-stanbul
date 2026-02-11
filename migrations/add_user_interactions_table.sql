-- Migration: Add user_interactions table for NCF retraining
-- Author: AI Istanbul Team
-- Date: February 10, 2026
-- Purpose: Store user interactions for model retraining

-- Create user_interactions table
CREATE TABLE IF NOT EXISTS user_interactions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    item_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,  -- 'view', 'click', 'conversion'
    implicit_rating FLOAT NOT NULL,  -- 0.5 (view), 1.0 (click), 2.0 (conversion)
    session_id VARCHAR(255),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,  -- Additional context (page, device, etc.)
    
    -- Constraints
    CONSTRAINT valid_interaction_type CHECK (interaction_type IN ('view', 'click', 'conversion')),
    CONSTRAINT valid_rating CHECK (implicit_rating >= 0 AND implicit_rating <= 5)
);

-- Create indices for fast queries
CREATE INDEX IF NOT EXISTS idx_interactions_user 
    ON user_interactions(user_id);

CREATE INDEX IF NOT EXISTS idx_interactions_item 
    ON user_interactions(item_id);

CREATE INDEX IF NOT EXISTS idx_interactions_timestamp 
    ON user_interactions(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_interactions_type 
    ON user_interactions(interaction_type);

CREATE INDEX IF NOT EXISTS idx_interactions_user_item 
    ON user_interactions(user_id, item_id);

-- Create composite index for common queries
CREATE INDEX IF NOT EXISTS idx_interactions_user_timestamp 
    ON user_interactions(user_id, timestamp DESC);

-- Add comment
COMMENT ON TABLE user_interactions IS 'User interaction history for NCF model training';
COMMENT ON COLUMN user_interactions.interaction_type IS 'Type: view (0.5), click (1.0), conversion (2.0)';
COMMENT ON COLUMN user_interactions.implicit_rating IS 'Implicit feedback signal derived from interaction type';

-- Create aggregation view for fast analytics
CREATE OR REPLACE VIEW user_interaction_stats AS
SELECT
    user_id,
    COUNT(*) as total_interactions,
    COUNT(DISTINCT item_id) as unique_items,
    COUNT(*) FILTER (WHERE interaction_type = 'view') as views,
    COUNT(*) FILTER (WHERE interaction_type = 'click') as clicks,
    COUNT(*) FILTER (WHERE interaction_type = 'conversion') as conversions,
    AVG(implicit_rating) as avg_rating,
    MAX(timestamp) as last_interaction
FROM user_interactions
GROUP BY user_id;

-- Grant permissions (adjust based on your user)
-- GRANT SELECT, INSERT ON user_interactions TO app_user;
-- GRANT SELECT ON user_interaction_stats TO app_user;

-- Sample data for testing (optional)
-- INSERT INTO user_interactions (user_id, item_id, interaction_type, implicit_rating, session_id) VALUES
-- ('user_1', 'item_10', 'view', 0.5, 'session_abc'),
-- ('user_1', 'item_10', 'click', 1.0, 'session_abc'),
-- ('user_1', 'item_10', 'conversion', 2.0, 'session_abc'),
-- ('user_2', 'item_15', 'click', 1.0, 'session_def');

COMMIT;
