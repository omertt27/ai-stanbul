#!/usr/bin/env python3
"""
Add missing columns to blog_posts table in GCP Cloud SQL
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

def add_missing_columns():
    """Add missing columns to blog_posts table"""
    
    print("\n" + "="*80)
    print("🔧 Adding Missing Columns to blog_posts Table")
    print("="*80)
    
    # Use the connection from backend/.env
    database_url = "postgresql://postgres:NewSecurePassword123!@127.0.0.1:5433/postgres"
    engine = create_engine(database_url)
    
    # List of columns to add
    columns_to_add = [
        ("slug", "VARCHAR(250) UNIQUE"),
        ("excerpt", "TEXT"),
        ("status", "VARCHAR(20) DEFAULT 'published'"),
        ("featured_image", "VARCHAR(500)"),
        ("category", "VARCHAR(100)"),
        ("tags", "JSONB DEFAULT '[]'::jsonb"),
        ("views", "INTEGER DEFAULT 0"),
        ("likes", "INTEGER DEFAULT 0"),
        ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("published_at", "TIMESTAMP"),
    ]
    
    with engine.connect() as conn:
        for col_name, col_type in columns_to_add:
            try:
                # Check if column exists
                result = conn.execute(text(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='blog_posts' AND column_name='{col_name}'
                """))
                
                if result.fetchone() is None:
                    # Column doesn't exist, add it
                    print(f"  Adding column: {col_name} ({col_type})")
                    conn.execute(text(f"ALTER TABLE blog_posts ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    print(f"    ✅ Added: {col_name}")
                else:
                    print(f"    ⏭️  Skipped: {col_name} (already exists)")
                    
            except Exception as e:
                print(f"    ❌ Error adding {col_name}: {e}")
                conn.rollback()
    
    print("\n" + "="*80)
    print("✅ Column Migration Complete!")
    print("="*80 + "\n")
    
    # Show final table structure
    print("📊 Final Table Structure:")
    print("-"*80)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'blog_posts'
            ORDER BY ordinal_position
        """))
        
        for row in result:
            print(f"  {row[0]:<20} {row[1]:<20} Nullable: {row[2]}")
    
    print("-"*80 + "\n")

if __name__ == "__main__":
    add_missing_columns()
