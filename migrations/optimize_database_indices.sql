-- Database Optimization for NCF System
-- Author: AI Istanbul Team
-- Date: February 11, 2026
-- Purpose: Add indices and materialized views for optimal query performance

-- ============================================================================
-- PHASE 1: Add Missing Indices
-- ============================================================================

-- User interactions - most frequently queried table
-- These indices support the retraining pipeline and recommendations

-- 1. User-based queries (for collecting user history)
CREATE INDEX IF NOT EXISTS idx_interactions_user_timestamp 
    ON user_interactions(user_id, timestamp DESC)
    INCLUDE (item_id, implicit_rating);

COMMENT ON INDEX idx_interactions_user_timestamp IS 
    'Optimize queries fetching user interaction history ordered by recency';

-- 2. Item-based queries (for item popularity and co-occurrence)
CREATE INDEX IF NOT EXISTS idx_interactions_item_timestamp 
    ON user_interactions(item_id, timestamp DESC)
    INCLUDE (user_id, implicit_rating);

COMMENT ON INDEX idx_interactions_item_timestamp IS 
    'Optimize queries for item-based analysis and popularity trends';

-- 3. Composite index for training data collection
CREATE INDEX IF NOT EXISTS idx_interactions_user_item_time 
    ON user_interactions(user_id, item_id, timestamp DESC);

COMMENT ON INDEX idx_interactions_user_item_time IS 
    'Optimize queries joining users and items with time filtering';

-- 4. Interaction type filtering (for weighted feedback)
CREATE INDEX IF NOT EXISTS idx_interactions_type_timestamp 
    ON user_interactions(interaction_type, timestamp DESC)
    WHERE interaction_type IN ('click', 'conversion');

COMMENT ON INDEX idx_interactions_type_timestamp IS 
    'Partial index for high-value interactions (clicks and conversions)';

-- 5. Session-based analysis
CREATE INDEX IF NOT EXISTS idx_interactions_session 
    ON user_interactions(session_id, timestamp)
    WHERE session_id IS NOT NULL;

COMMENT ON INDEX idx_interactions_session IS 
    'Support session-based recommendation queries';

-- ============================================================================
-- PHASE 2: Materialized Views for Aggregations
-- ============================================================================

-- 1. User activity summary (for filtering active users)
CREATE MATERIALIZED VIEW IF NOT EXISTS user_activity_summary AS
SELECT 
    user_id,
    COUNT(*) as total_interactions,
    COUNT(DISTINCT item_id) as unique_items,
    COUNT(*) FILTER (WHERE interaction_type = 'view') as views,
    COUNT(*) FILTER (WHERE interaction_type = 'click') as clicks,
    COUNT(*) FILTER (WHERE interaction_type = 'conversion') as conversions,
    AVG(implicit_rating) as avg_rating,
    MAX(timestamp) as last_interaction,
    MIN(timestamp) as first_interaction,
    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 86400 as days_active
FROM user_interactions
GROUP BY user_id
HAVING COUNT(*) >= 3;  -- Only users with 3+ interactions

CREATE UNIQUE INDEX ON user_activity_summary(user_id);
CREATE INDEX ON user_activity_summary(total_interactions DESC);
CREATE INDEX ON user_activity_summary(last_interaction DESC);

COMMENT ON MATERIALIZED VIEW user_activity_summary IS 
    'Pre-aggregated user activity metrics for fast filtering and analysis';

-- 2. Item popularity metrics (for cold-start and trending)
CREATE MATERIALIZED VIEW IF NOT EXISTS item_popularity_summary AS
SELECT 
    item_id,
    COUNT(*) as total_interactions,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(*) FILTER (WHERE interaction_type = 'click') as clicks,
    COUNT(*) FILTER (WHERE interaction_type = 'conversion') as conversions,
    AVG(implicit_rating) as avg_rating,
    MAX(timestamp) as last_interaction,
    -- Popularity score (combines recency and volume)
    (COUNT(*) * 0.7 + COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '7 days') * 0.3) as popularity_score
FROM user_interactions
GROUP BY item_id
HAVING COUNT(*) >= 2;  -- Only items with 2+ interactions

CREATE UNIQUE INDEX ON item_popularity_summary(item_id);
CREATE INDEX ON item_popularity_summary(popularity_score DESC);
CREATE INDEX ON item_popularity_summary(total_interactions DESC);

COMMENT ON MATERIALIZED VIEW item_popularity_summary IS 
    'Pre-aggregated item popularity metrics for fallback recommendations';

-- 3. Recent interactions (for real-time features)
CREATE MATERIALIZED VIEW IF NOT EXISTS recent_interactions_summary AS
SELECT 
    user_id,
    item_id,
    MAX(timestamp) as latest_interaction,
    MAX(implicit_rating) as max_rating,
    COUNT(*) as interaction_count
FROM user_interactions
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY user_id, item_id;

CREATE INDEX ON recent_interactions_summary(user_id);
CREATE INDEX ON recent_interactions_summary(item_id);
CREATE INDEX ON recent_interactions_summary(latest_interaction DESC);

COMMENT ON MATERIALIZED VIEW recent_interactions_summary IS 
    'Recent user-item interactions for real-time recommendation features';

-- ============================================================================
-- PHASE 3: Refresh Policies for Materialized Views
-- ============================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_ncf_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_activity_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY item_popularity_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY recent_interactions_summary;
    
    RAISE NOTICE 'All NCF materialized views refreshed';
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_ncf_materialized_views IS 
    'Refresh all NCF-related materialized views concurrently';

-- ============================================================================
-- PHASE 4: Query Optimization Helpers
-- ============================================================================

-- Function to get active users for training (using materialized view)
CREATE OR REPLACE FUNCTION get_active_users_for_training(
    min_interactions INT DEFAULT 10,
    max_users INT DEFAULT 10000
)
RETURNS TABLE(user_id VARCHAR, interaction_count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        uas.user_id,
        uas.total_interactions
    FROM user_activity_summary uas
    WHERE uas.total_interactions >= min_interactions
        AND uas.last_interaction > NOW() - INTERVAL '90 days'
    ORDER BY uas.total_interactions DESC
    LIMIT max_users;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_active_users_for_training IS 
    'Efficiently retrieve active users for model training using materialized view';

-- Function to get popular items for cold-start
CREATE OR REPLACE FUNCTION get_popular_items(
    top_n INT DEFAULT 100,
    category VARCHAR DEFAULT NULL
)
RETURNS TABLE(item_id VARCHAR, popularity_score NUMERIC) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ips.item_id,
        ips.popularity_score
    FROM item_popularity_summary ips
    ORDER BY ips.popularity_score DESC
    LIMIT top_n;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_popular_items IS 
    'Get most popular items for cold-start recommendations';

-- ============================================================================
-- PHASE 5: Connection Pool Optimization
-- ============================================================================

-- Adjust PostgreSQL settings for optimal connection pooling
-- (These should be set in postgresql.conf or via ALTER SYSTEM)

-- Recommended settings for Cloud SQL (adjust based on instance size):
-- max_connections = 100
-- shared_buffers = 256MB (25% of RAM)
-- effective_cache_size = 1GB (50% of RAM)
-- work_mem = 4MB
-- maintenance_work_mem = 64MB
-- random_page_cost = 1.1 (for SSD)

-- ============================================================================
-- PHASE 6: Table Statistics Update
-- ============================================================================

-- Analyze tables to update statistics for query planner
ANALYZE user_interactions;
ANALYZE user_activity_summary;
ANALYZE item_popularity_summary;
ANALYZE recent_interactions_summary;

-- ============================================================================
-- PHASE 7: Monitoring Queries
-- ============================================================================

-- View to monitor index usage
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

COMMENT ON VIEW index_usage_stats IS 
    'Monitor index usage to identify unused or inefficient indices';

-- View to monitor table sizes
CREATE OR REPLACE VIEW table_size_stats AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as indexes_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

COMMENT ON VIEW table_size_stats IS 
    'Monitor table and index sizes for capacity planning';

-- ============================================================================
-- PHASE 8: Cleanup and Maintenance
-- ============================================================================

-- Function to clean up old interactions (data retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_interactions(
    retention_days INT DEFAULT 365
)
RETURNS BIGINT AS $$
DECLARE
    deleted_count BIGINT;
BEGIN
    DELETE FROM user_interactions
    WHERE timestamp < NOW() - make_interval(days => retention_days);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Refresh materialized views after cleanup
    PERFORM refresh_ncf_materialized_views();
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_interactions IS 
    'Clean up interactions older than retention period and refresh views';

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check index status
SELECT 
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'user_interactions'
ORDER BY indexname;

-- Check materialized view status
SELECT 
    matviewname,
    ispopulated,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size
FROM pg_matviews
WHERE schemaname = 'public';

-- Sample query performance check
EXPLAIN ANALYZE
SELECT user_id, item_id, implicit_rating
FROM user_interactions
WHERE user_id = 'user_123'
    AND timestamp > NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC
LIMIT 100;

COMMIT;

-- ============================================================================
-- POST-MIGRATION NOTES
-- ============================================================================
-- 
-- 1. Schedule materialized view refreshes:
--    - Run refresh_ncf_materialized_views() hourly via cron or pg_cron
--    - Example: SELECT cron.schedule('refresh-ncf-views', '0 * * * *', 'SELECT refresh_ncf_materialized_views()');
--
-- 2. Monitor index usage:
--    - Periodically check index_usage_stats view
--    - Drop unused indices to save space
--
-- 3. Vacuum and analyze:
--    - Run VACUUM ANALYZE weekly
--    - Enable auto-vacuum for large tables
--
-- 4. Connection pooling:
--    - Use PgBouncer or similar for connection pooling
--    - Set pool_size=20, max_overflow=10 in SQLAlchemy
--
-- ============================================================================
