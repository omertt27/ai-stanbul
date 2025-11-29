#!/usr/bin/env python3
"""
Production database migration - Add missing photo columns to restaurants table
Run this on Render or set DATABASE_URL to production database
"""

import os
import sys

def run_migration():
    """Run the migration using raw SQL"""
    
    # Import here to ensure dependencies are available
    try:
        from sqlalchemy import create_engine, text, inspect
    except ImportError:
        print("❌ ERROR: SQLAlchemy not installed")
        print("   Run: pip install sqlalchemy psycopg2-binary")
        return False
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        return False
    
    # Fix postgres:// to postgresql:// (Render sometimes uses old format)
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    print(f"🔗 Connecting to database...")
    
    try:
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Add columns if they don't exist (PostgreSQL safe syntax)
            print("🔧 Adding photo_url column...")
            conn.execute(text("""
                ALTER TABLE restaurants 
                ADD COLUMN IF NOT EXISTS photo_url VARCHAR;
            """))
            conn.commit()
            print("   ✅ photo_url column ready")
            
            print("🔧 Adding photo_reference column...")
            conn.execute(text("""
                ALTER TABLE restaurants 
                ADD COLUMN IF NOT EXISTS photo_reference VARCHAR;
            """))
            conn.commit()
            print("   ✅ photo_reference column ready")
        
        print("\n✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Production Database Migration")
    print("=" * 60 + "\n")
    
    success = run_migration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SUCCESS - Database schema updated")
    else:
        print("❌ FAILED - Check error messages above")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
