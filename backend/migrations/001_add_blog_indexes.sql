-- Blog System Performance Indexes Migration
-- Created: 2025-10-27
-- Purpose: Add critical indexes for blog posts, comments, and likes

-- Speed up district filtering (used in every query)
CREATE INDEX IF NOT EXISTS idx_blog_posts_district ON blog_posts(district);

-- Speed up date sorting (default sort order)
CREATE INDEX IF NOT EXISTS idx_blog_posts_created_at ON blog_posts(created_at DESC);

-- Speed up popularity sorting  
CREATE INDEX IF NOT EXISTS idx_blog_posts_likes_count ON blog_posts(likes_count DESC);

-- Speed up comment lookups (N+1 query fix)
CREATE INDEX IF NOT EXISTS idx_blog_comments_post_id ON blog_comments(blog_post_id);
CREATE INDEX IF NOT EXISTS idx_blog_comments_approved ON blog_comments(is_approved) WHERE is_approved = TRUE;

-- Speed up like checks
CREATE INDEX IF NOT EXISTS idx_blog_likes_post_user ON blog_likes(blog_post_id, user_identifier);

-- Add composite index for common queries
CREATE INDEX IF NOT EXISTS idx_blog_posts_district_created ON blog_posts(district, created_at DESC);

-- Analyze tables to update statistics
ANALYZE blog_posts;
ANALYZE blog_comments;
ANALYZE blog_likes;
