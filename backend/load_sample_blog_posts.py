#!/usr/bin/env python3
"""
Load sample blog posts into the database
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

def load_sample_posts():
    """Load sample blog posts"""
    # Connect to database
    database_url = os.getenv('DATABASE_URL')
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    if not database_url:
        database_url = 'postgresql://omer@localhost:5432/postgres'
    
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Sample blog posts
    sample_posts = [
        {
            'title': 'Hidden Gems of Sultanahmet',
            'content': 'Discover the lesser-known attractions in Istanbul\'s historic Sultanahmet district. Beyond the Blue Mosque and Hagia Sophia, there are countless treasures waiting to be explored...',
            'author': 'AI Istanbul Team',
            'category': 'Travel Guide',
            'district': 'Fatih',
            'status': 'published',
            'tags': '["sultanahmet", "historic", "travel guide"]',
            'featured_image': '/static/blog/sultanahmet.jpg',
            'excerpt': 'Explore hidden treasures in Sultanahmet beyond the famous landmarks.',
            'likes_count': 42,
            'views': 256
        },
        {
            'title': 'Best Rooftop Cafes in Beyoğlu',
            'content': 'Experience Istanbul from above! These rooftop cafes offer stunning views of the Bosphorus and the Golden Horn while you enjoy traditional Turkish coffee...',
            'author': 'AI Istanbul Team',
            'category': 'Food & Drink',
            'district': 'Beyoğlu',
            'status': 'published',
            'tags': '["beyoglu", "cafes", "rooftop", "views"]',
            'featured_image': '/static/blog/rooftop-cafe.jpg',
            'excerpt': 'Discover the best rooftop cafes with breathtaking Bosphorus views.',
            'likes_count': 38,
            'views': 189
        },
        {
            'title': 'A Day in Kadıköy: The Asian Side',
            'content': 'Cross the Bosphorus and explore Kadıköy, Istanbul\'s vibrant Asian district. From the bustling food markets to trendy art galleries, Kadıköy offers a different perspective on Istanbul...',
            'author': 'AI Istanbul Team',
            'category': 'District Guide',
            'district': 'Kadıköy',
            'status': 'published',
            'tags': '["kadikoy", "asian side", "food", "culture"]',
            'featured_image': '/static/blog/kadikoy.jpg',
            'excerpt': 'Experience the vibrant culture and cuisine of Istanbul\'s Asian side.',
            'likes_count': 56,
            'views': 312
        }
    ]
    
    # Clear existing posts
    session.execute(text('DELETE FROM blog_posts'))
    session.commit()
    print('🗑️  Cleared existing blog posts')
    
    # Insert sample posts
    for post_data in sample_posts:
        insert_sql = text("""
            INSERT INTO blog_posts 
            (title, content, author, category, district, status, tags, featured_image, excerpt, likes_count, views, created_at, updated_at)
            VALUES 
            (:title, :content, :author, :category, :district, :status, CAST(:tags AS json), :featured_image, :excerpt, :likes_count, :views, :created_at, :updated_at)
        """)
        
        session.execute(insert_sql, {
            **post_data,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        })
    
    session.commit()
    print(f'✅ Loaded {len(sample_posts)} sample blog posts')
    
    # Verify
    count = session.execute(text('SELECT COUNT(*) FROM blog_posts')).scalar()
    print(f'\nTotal blog posts in database: {count}')
    
    # Show posts
    posts = session.execute(text('SELECT id, title, author, district FROM blog_posts')).fetchall()
    print('\nPosts:')
    for post in posts:
        print(f'  - #{post[0]}: {post[1]} (district: {post[3]})')
    
    session.close()

if __name__ == "__main__":
    load_sample_posts()
