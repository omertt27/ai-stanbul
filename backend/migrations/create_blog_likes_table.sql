-- Migration to create blog_likes table
-- Run this SQL script to add the likes tracking functionality

CREATE TABLE blog_likes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    blog_post_id INTEGER NOT NULL,
    user_identifier VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (blog_post_id) REFERENCES blog_posts (id),
    UNIQUE(blog_post_id, user_identifier)
);
