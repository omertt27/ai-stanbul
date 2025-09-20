-- Database indexing script for Istanbul AI chatbot performance optimization
-- This script adds necessary indexes for the places table to optimize filtering queries

-- Single column indexes for category and district filtering
CREATE INDEX IF NOT EXISTS idx_places_category ON places (category);
CREATE INDEX IF NOT EXISTS idx_places_district ON places (district);

-- Composite index for combined category + district filtering (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_places_category_district ON places (category, district);

-- Additional useful indexes for common filtering patterns
CREATE INDEX IF NOT EXISTS idx_places_name ON places (name);
CREATE INDEX IF NOT EXISTS idx_places_district_category ON places (district, category);

-- For restaurants table - add location-based indexing
CREATE INDEX IF NOT EXISTS idx_restaurants_cuisine ON restaurants (cuisine);
CREATE INDEX IF NOT EXISTS idx_restaurants_location ON restaurants (location);
CREATE INDEX IF NOT EXISTS idx_restaurants_rating ON restaurants (rating);
CREATE INDEX IF NOT EXISTS idx_restaurants_price_level ON restaurants (price_level);

-- Composite indexes for restaurants for common filter combinations
CREATE INDEX IF NOT EXISTS idx_restaurants_cuisine_location ON restaurants (cuisine, location);
CREATE INDEX IF NOT EXISTS idx_restaurants_location_rating ON restaurants (location, rating);

-- For user sessions and chat history (performance optimization)
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions (session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history (session_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_session_id ON user_interactions (session_id);

-- For enhanced chat history with timestamps
CREATE INDEX IF NOT EXISTS idx_enhanced_chat_history_session_id ON enhanced_chat_history (session_id);
CREATE INDEX IF NOT EXISTS idx_enhanced_chat_history_timestamp ON enhanced_chat_history (timestamp);

-- For user preferences optimization
CREATE INDEX IF NOT EXISTS idx_user_preferences_session_id ON user_preferences (session_id);

-- For conversation context
CREATE INDEX IF NOT EXISTS idx_conversation_context_session_id ON conversation_context (session_id);
