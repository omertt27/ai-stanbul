#!/usr/bin/env python3
"""
Create blog-related database tables
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Create a fresh Base instance
Base = declarative_base()

# Define BlogPost model directly
class BlogPost(Base):
    __tablename__ = "blog_posts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    slug = Column(String(250), nullable=True, unique=True)
    content = Column(Text, nullable=False)
    excerpt = Column(Text, nullable=True)
    author = Column(String(100), nullable=True)
    status = Column(String(20), default='draft')
    featured_image = Column(String(500), nullable=True)
    category = Column(String(100), nullable=True)
    tags = Column(JSON, default=list)
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    district = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
    likes_count = Column(Integer, default=0)

class BlogComment(Base):
    __tablename__ = "blog_comments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    blog_post_id = Column(Integer, ForeignKey('blog_posts.id', ondelete='CASCADE'), nullable=False)
    author_name = Column(String(100), nullable=False)
    author_email = Column(String(200), nullable=True)
    content = Column(Text, nullable=False)
    status = Column(String(20), default='approved')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BlogLike(Base):
    __tablename__ = "blog_likes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    blog_post_id = Column(Integer, ForeignKey('blog_posts.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def create_blog_tables():
    """Create blog_posts, blog_comments, and blog_likes tables"""
    print("Creating blog tables...")
    
    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    if not database_url:
        # Local PostgreSQL
        database_url = f"postgresql://omer@localhost:5432/postgres"
    
    print(f"Connecting to: {database_url.split('@')[1] if '@' in database_url else database_url}")
    
    # Create engine
    engine = create_engine(database_url)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    print("✅ Blog tables created successfully!")
    
    # Verify
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"\nTotal tables in database: {len(tables)}")
    print("\nBlog-related tables:")
    for table in sorted(tables):
        if 'blog' in table.lower():
            print(f"  ✅ {table}")
            # Show columns
            for col in inspector.get_columns(table):
                print(f"      - {col['name']}: {col['type']}")

if __name__ == "__main__":
    create_blog_tables()
