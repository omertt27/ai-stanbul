-- Istanbul AI Database Schema
-- Production-ready with indexes and constraints

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- User profiles table
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    profile_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversation history table
CREATE TABLE conversation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    turn_number INTEGER NOT NULL,
    user_input TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    intents JSONB,
    entities JSONB,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analytics and metrics
CREATE TABLE interaction_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    intent_type VARCHAR(100),
    response_time_ms INTEGER,
    user_satisfaction FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System performance metrics
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_conversation_history_session ON conversation_history(session_id);
CREATE INDEX idx_conversation_history_user ON conversation_history(user_id);
CREATE INDEX idx_conversation_history_created ON conversation_history(created_at);
CREATE INDEX idx_interaction_metrics_user ON interaction_metrics(user_id);
CREATE INDEX idx_interaction_metrics_created ON interaction_metrics(created_at);
CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_created ON system_metrics(created_at);

-- GIN indexes for JSONB columns
CREATE INDEX idx_user_profiles_data ON user_profiles USING GIN(profile_data);
CREATE INDEX idx_conversation_intents ON conversation_history USING GIN(intents);
CREATE INDEX idx_conversation_entities ON conversation_history USING GIN(entities);

-- Update trigger for user_profiles
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE
    ON user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for analytics
CREATE VIEW user_interaction_summary AS
SELECT 
    user_id,
    COUNT(*) as total_interactions,
    AVG(response_time_ms) as avg_response_time,
    AVG(user_satisfaction) as avg_satisfaction,
    MIN(created_at) as first_interaction,
    MAX(created_at) as last_interaction
FROM interaction_metrics
GROUP BY user_id;

CREATE VIEW popular_intents AS
SELECT 
    intent_type,
    COUNT(*) as frequency,
    AVG(response_time_ms) as avg_response_time
FROM interaction_metrics
WHERE intent_type IS NOT NULL
GROUP BY intent_type
ORDER BY frequency DESC;
