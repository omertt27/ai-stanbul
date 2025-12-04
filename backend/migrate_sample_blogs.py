#!/usr/bin/env python3
"""
Migrate blog posts from local JSON to Render PostgreSQL
Run this on Render.com server
"""
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get DATABASE_URL from environment (Render provides this automatically)
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in environment")
    print("This script should be run on Render.com where DATABASE_URL is set")
    exit(1)

print(f"📊 Using database from environment: {DATABASE_URL[:50]}...")

# Create engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
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

# Sample blog posts to add
SAMPLE_POSTS = [
    {
        "title": "Top 10 Hidden Gems in Istanbul",
        "content": """Discover Istanbul's best-kept secrets! From hidden rooftop gardens in Galata to secret tea houses in Üsküdar, these spots offer authentic experiences away from tourist crowds.

**Highlights:**
- Secret gardens with Bosphorus views
- Local neighborhood cafes
- Historic hammams still used by locals
- Underground cisterns few tourists know about
- Authentic local restaurants

Perfect for travelers who want to experience the real Istanbul!""",
        "author": "Istanbul Insider",
        "district": "Various",
        "likes_count": 42
    },
    {
        "title": "Best Street Food in Beyoğlu",
        "content": """A foodie's guide to Beyoğlu's incredible street food scene. From traditional simit to modern fusion bites, explore the tastiest street eats.

**Must-Try Foods:**
- Simit from century-old bakeries
- Balık ekmek (fish sandwich) at Karaköy
- Midye dolma (stuffed mussels)
- Kokoreç from historic vendors
- Turkish ice cream shows

Budget-friendly and delicious! Most items under 50 TL.""",
        "author": "Food Explorer",
        "district": "Beyoğlu",
        "likes_count": 38
    },
    {
        "title": "Weekend Guide: Asian Side Adventures",
        "content": """Explore Istanbul's Asian side! Kadıköy, Moda, and Üsküdar offer a more relaxed, local atmosphere.

**Weekend Itinerary:**
- Saturday morning at Kadıköy market
- Brunch in trendy Moda cafes
- Waterfront walk with Bosphorus views
- Sunday antique shopping
- Traditional Turkish breakfast spots

The Asian side is less touristy and more affordable than European side!""",
        "author": "Weekend Wanderer",
        "district": "Kadıköy",
        "likes_count": 27
    },
    {
        "title": "Istanbul's Coffee Culture: Traditional to Modern",
        "content": """From Ottoman-era coffee houses to third-wave specialty cafes, Istanbul's coffee scene is evolving while honoring tradition.

**Coffee Experiences:**
- Traditional Turkish coffee with fortune telling
- Specialty single-origin pour-overs
- Rooftop cafes with city views
- Historic coffee houses in Grand Bazaar
- Modern roasteries in Karaköy

Whether you prefer traditional or modern, Istanbul has incredible coffee!""",
        "author": "Coffee Culture Expert",
        "district": "Beyoğlu",
        "likes_count": 31
    },
    {
        "title": "Budget Travel: Istanbul Under $50/Day",
        "content": """Yes, you can enjoy Istanbul on a budget! Here's how to experience this amazing city affordably.

**Daily Budget Breakdown:**
- Accommodation: $15-20 (hostel/budget hotel)
- Food: $10-15 (street food + local restaurants)
- Transport: $5 (Istanbul Card)
- Attractions: $10-15 (many mosques are free!)

**Free Activities:**
- Walk across Galata Bridge
- Explore historic neighborhoods
- Visit free mosques
- People-watch at Taksim Square
- Sunset at Emirgan Park""",
        "author": "Budget Traveler",
        "district": "Various",
        "likes_count": 45
    }
]

def migrate_sample_posts():
    """Add sample blog posts to database"""
    
    print("🔄 Starting blog posts migration...")
    
    # Create table if not exists
    Base.metadata.create_all(engine)
    print("✅ Table 'blog_posts' ready")
    
    # Create session
    session = Session()
    
    # Check existing posts
    existing_count = session.query(BlogPost).count()
    print(f"📊 Existing posts in database: {existing_count}")
    
    if existing_count > 0:
        print("ℹ️  Database already has posts. Adding new ones...")
    
    # Add posts
    added = 0
    
    for post_data in SAMPLE_POSTS:
        try:
            # Check if post with same title already exists
            existing = session.query(BlogPost).filter(
                BlogPost.title == post_data['title']
            ).first()
            
            if existing:
                print(f"⏭️  Skipping '{post_data['title'][:40]}...' (already exists)")
                continue
            
            # Create post
            post = BlogPost(
                title=post_data['title'],
                content=post_data['content'],
                author=post_data['author'],
                district=post_data.get('district'),
                created_at=datetime.utcnow(),
                likes_count=post_data.get('likes_count', 0)
            )
            
            session.add(post)
            added += 1
            print(f"✅ Added: {post_data['title'][:50]}...")
            
        except Exception as e:
            print(f"❌ Error adding post '{post_data['title']}': {e}")
            continue
    
    # Commit
    session.commit()
    
    # Verify
    total = session.query(BlogPost).count()
    
    print("\n" + "="*50)
    print("🎉 Migration Complete!")
    print("="*50)
    print(f"✅ Added: {added} new posts")
    print(f"📊 Total in database: {total} posts")
    print("="*50)
    
    # Show all posts
    print("\n📝 All posts in database:")
    all_posts = session.query(BlogPost).all()
    for post in all_posts:
        print(f"  {post.id}. {post.title} ({post.likes_count} likes)")
    
    session.close()
    print("\n✅ Migration successful!")
    print("\n🧪 Test it:")
    print("curl https://ai-stanbul.onrender.com/api/blog/posts")

if __name__ == "__main__":
    try:
        migrate_sample_posts()
    except Exception as e:
        print(f"\n\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
