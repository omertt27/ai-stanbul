#!/usr/bin/env python3
"""
Delete test blog posts from the database
Removes posts with "CRUD Test" or "Test Post" in the title
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import SessionLocal
from models import BlogPost

def delete_test_posts():
    """Delete blog posts that contain test-related keywords in title"""
    db = SessionLocal()
    
    try:
        # Find test posts by title patterns
        test_keywords = [
            "CRUD Test",
            "Test Post",
            "test post",
            "CRUD Tester",
            "Updated CRUD",
        ]
        
        print("🔍 Searching for test blog posts...")
        
        # Get all posts that match test patterns
        all_posts = db.query(BlogPost).all()
        test_posts = []
        
        for post in all_posts:
            for keyword in test_keywords:
                if keyword.lower() in post.title.lower():
                    test_posts.append(post)
                    break
        
        if not test_posts:
            print("✅ No test posts found in the database.")
            return
        
        print(f"\n📋 Found {len(test_posts)} test post(s):")
        for post in test_posts:
            print(f"  - ID: {post.id} | Title: {post.title[:50]}...")
        
        # Confirm deletion
        response = input("\n❓ Do you want to delete these posts? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled.")
            return
        
        # Delete the posts
        deleted_count = 0
        for post in test_posts:
            db.delete(post)
            deleted_count += 1
            print(f"🗑️  Deleted: {post.title[:50]}...")
        
        db.commit()
        print(f"\n✅ Successfully deleted {deleted_count} test post(s)!")
        
        # Show remaining posts count
        remaining = db.query(BlogPost).count()
        print(f"📊 Remaining blog posts: {remaining}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🧹 Cleaning up test blog posts...\n")
    delete_test_posts()
