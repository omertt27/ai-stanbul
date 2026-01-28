#!/usr/bin/env python3
"""
Delete CRUD test posts from the GCP database.
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import SessionLocal
from models import BlogPost

def delete_test_posts():
    """Find and delete CRUD test posts from the database."""
    db = SessionLocal()
    
    try:
        # Find posts with "CRUD" or "Test" in title
        test_posts = db.query(BlogPost).filter(
            BlogPost.title.ilike('%CRUD%') | 
            BlogPost.title.ilike('%Test Post%')
        ).all()
        
        if not test_posts:
            print("✅ No test posts found in database.")
            return
        
        print(f"🔍 Found {len(test_posts)} test post(s):")
        for post in test_posts:
            print(f"  - ID {post.id}: '{post.title}' by {post.author}")
        
        # Confirm deletion
        confirm = input("\n❓ Delete these posts? (y/n): ")
        if confirm.lower() != 'y':
            print("❌ Cancelled.")
            return
        
        # Delete the posts
        for post in test_posts:
            db.delete(post)
            print(f"  🗑️  Deleted: '{post.title}'")
        
        db.commit()
        print(f"\n✅ Successfully deleted {len(test_posts)} test post(s)!")
        
        # Show remaining posts
        remaining = db.query(BlogPost).count()
        print(f"📊 Remaining posts in database: {remaining}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🧹 Cleaning up CRUD test posts from database...\n")
    delete_test_posts()
