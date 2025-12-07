-- Migration: Add missing columns to blog_posts and feedback_events tables
-- Date: 2025-12-07

-- Add columns to blog_posts table
ALTER TABLE blog_posts 
ADD COLUMN IF NOT EXISTS slug VARCHAR(250) UNIQUE,
ADD COLUMN IF NOT EXISTS excerpt TEXT,
ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'draft',
ADD COLUMN IF NOT EXISTS featured_image VARCHAR(500),
ADD COLUMN IF NOT EXISTS category VARCHAR(100),
ADD COLUMN IF NOT EXISTS tags JSON DEFAULT '[]',
ADD COLUMN IF NOT EXISTS views INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS likes INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN IF NOT EXISTS published_at TIMESTAMP;

-- Add columns to feedback_events table
ALTER TABLE feedback_events
ADD COLUMN IF NOT EXISTS rating INTEGER,
ADD COLUMN IF NOT EXISTS feedback_text TEXT,
ADD COLUMN IF NOT EXISTS context JSON;

-- Generate slugs for existing posts (if any)
UPDATE blog_posts 
SET slug = LOWER(REPLACE(REGEXP_REPLACE(title, '[^a-zA-Z0-9\s-]', '', 'g'), ' ', '-'))
WHERE slug IS NULL;

-- Set updated_at to created_at for existing posts
UPDATE blog_posts 
SET updated_at = created_at
WHERE updated_at IS NULL;
