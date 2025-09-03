-- Migration to add author fields to blog_posts table
-- Run this SQL script to update the database schema

ALTER TABLE blog_posts ADD COLUMN author_name VARCHAR(100);
ALTER TABLE blog_posts ADD COLUMN author_photo VARCHAR(500);
