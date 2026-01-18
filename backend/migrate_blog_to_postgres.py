#!/usr/bin/env python3
"""
Migrate blog posts from local JSON file to PostgreSQL database
"""
import json
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))
from db.base import Base

# Database connection
# Get from Render.com dashboard or use environment variable
DATABASE_URL = os.getenv('DATABASE_URL') or input("Enter PostgreSQL Database URL: ")

# Create engine
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Blog Post Model
class BlogPost(Base):
    __tablename__ = "blog_posts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    author = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    likes_count = Column(Integer, default=0)

def migrate_blog_posts():
    """Migrate blog posts from JSON to PostgreSQL"""
    
    print("🔄 Starting blog posts migration...")
    print(f"📊 Database: {DATABASE_URL[:50]}...")
    
    # Create table if not exists
    Base.metadata.create_all(engine)
    print("✅ Table 'blog_posts' ready")
    
    # Load JSON data
    json_file = os.path.join(os.path.dirname(__file__), 'blog_posts.json')
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        posts_data = json.load(f)
    
    print(f"📁 Loaded {len(posts_data)} posts from JSON")
    
    # Create session
    session = Session()
    
    # Check existing posts
    existing_count = session.query(BlogPost).count()
    print(f"📊 Existing posts in database: {existing_count}")
    
    if existing_count > 0:
        response = input("⚠️  Database already has posts. Clear and re-import? (yes/no): ")
        if response.lower() == 'yes':
            session.query(BlogPost).delete()
            session.commit()
            print("🗑️  Cleared existing posts")
        else:
            print("ℹ️  Will skip posts that already exist")
    
    # Migrate posts
    added = 0
    skipped = 0
    
    for post_data in posts_data:
        try:
            # Parse dates
            created_at = None
            if 'created_at' in post_data:
                try:
                    created_at = datetime.fromisoformat(post_data['created_at'].replace('Z', '+00:00'))
                except:
                    created_at = datetime.utcnow()
            
            # Extract district from tags or category
            district = post_data.get('district')
            if not district and 'tags' in post_data:
                # Try to find district in tags
                districts = ['Kadıköy', 'Beyoğlu', 'Beşiktaş', 'Fatih', 'Şişli', 'Üsküdar']
                for tag in post_data.get('tags', []):
                    if tag in districts:
                        district = tag
                        break
            
            # Create post
            post = BlogPost(
                title=post_data.get('title', 'Untitled'),
                content=post_data.get('content', ''),
                author=post_data.get('author', 'Anonymous'),
                district=district,
                created_at=created_at or datetime.utcnow(),
                likes_count=post_data.get('likes_count', 0) or post_data.get('likes', 0)
            )
            
            session.add(post)
            added += 1
            
            # Commit in batches
            if added % 10 == 0:
                session.commit()
                print(f"✅ Migrated {added} posts...")
            
        except Exception as e:
            print(f"❌ Error migrating post '{post_data.get('title', 'unknown')}': {e}")
            skipped += 1
            continue
    
    # Final commit
    session.commit()
    
    # Verify
    total = session.query(BlogPost).count()
    
    print("\n" + "="*50)
    print("🎉 Migration Complete!")
    print("="*50)
    print(f"✅ Added: {added} posts")
    print(f"⚠️  Skipped: {skipped} posts")
    print(f"📊 Total in database: {total} posts")
    print("="*50)
    
    # Show sample
    print("\n📝 Sample posts:")
    sample_posts = session.query(BlogPost).limit(5).all()
    for post in sample_posts:
        print(f"  - {post.id}: {post.title[:50]}... ({post.likes_count} likes)")
    
    session.close()
    print("\n✅ Migration successful!")

if __name__ == "__main__":
    try:
        migrate_blog_posts()
    except KeyboardInterrupt:
        print("\n\n❌ Migration cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
