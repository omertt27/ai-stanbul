#!/usr/bin/env python3
"""Check blog_likes table in database"""
import os
import sys
sys.path.insert(0, '/Users/omer/Desktop/ai-stanbul/backend')

from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

# Load environment
load_dotenv('/Users/omer/Desktop/ai-stanbul/.env')

# Database URL
DATABASE_URL = "postgresql://postgres:NewSecurePassword123!@localhost:5433/postgres"

try:
    # Create engine
    engine = create_engine(DATABASE_URL)
    
    # Connect
    with engine.connect() as conn:
        print("✅ Connected to database!")
        
        # Get all tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\n📊 Total tables: {len(tables)}")
        print("\nAll tables:")
        for table in sorted(tables):
            print(f"  - {table}")
        
        # Check if blog_likes exists
        if 'blog_likes' in tables:
            print("\n✅ blog_likes table EXISTS")
            
            # Get table info
            columns = inspector.get_columns('blog_likes')
            print("\nColumns:")
            for col in columns:
                print(f"  - {col['name']}: {col['type']}")
            
            # Count rows
            result = conn.execute(text("SELECT COUNT(*) FROM blog_likes"))
            count = result.fetchone()[0]
            print(f"\n📈 Total likes in database: {count}")
            
            # Show recent likes
            if count > 0:
                result = conn.execute(text("""
                    SELECT id, blog_post_id, user_id, created_at 
                    FROM blog_likes 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """))
                print("\n🔥 Recent likes:")
                for row in result:
                    print(f"  - Post {row[1]} liked by {row[2][:15]}... at {row[3]}")
        else:
            print("\n❌ blog_likes table DOES NOT EXIST")
            print("\n💡 Need to create it? Run:")
            print("   cd /Users/omer/Desktop/ai-stanbul/backend")
            print("   python -c 'from database import Base, engine; Base.metadata.create_all(engine)'")
        
        # Also check blog_posts
        if 'blog_posts' in tables:
            result = conn.execute(text("SELECT COUNT(*) FROM blog_posts"))
            count = result.fetchone()[0]
            print(f"\n📝 Total blog posts: {count}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
