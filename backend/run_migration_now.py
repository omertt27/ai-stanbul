#!/usr/bin/env python3
"""
Quick migration script to add missing columns to production database
"""
import psycopg2
import sys

# Production database URL
DATABASE_URL = "postgresql://aistanbul_postgre_user:FEddnYmd0ymR2HKBJIax3mqWkfTB0XZe@dpg-d4jg45e3jp1c73b6gas0-a.frankfurt-postgres.render.com/aistanbul_postgre"

def run_migration():
    """Add photo_url and photo_reference columns to restaurants table"""
    try:
        print("🔗 Connecting to production database...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        print("📋 Checking current schema...")
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'restaurants'
            ORDER BY ordinal_position;
        """)
        columns_before = cursor.fetchall()
        print("\n✅ Current columns:")
        for col_name, col_type in columns_before:
            print(f"   - {col_name}: {col_type}")
        
        print("\n🔧 Adding missing columns...")
        cursor.execute("""
            ALTER TABLE restaurants 
            ADD COLUMN IF NOT EXISTS photo_url TEXT,
            ADD COLUMN IF NOT EXISTS photo_reference TEXT;
        """)
        
        conn.commit()
        print("✅ Migration executed successfully!")
        
        print("\n📋 Verifying new schema...")
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'restaurants'
            ORDER BY ordinal_position;
        """)
        columns_after = cursor.fetchall()
        print("\n✅ Updated columns:")
        for col_name, col_type in columns_after:
            print(f"   - {col_name}: {col_type}")
        
        # Check if columns were added
        column_names = [col[0] for col in columns_after]
        if 'photo_url' in column_names and 'photo_reference' in column_names:
            print("\n🎉 SUCCESS! Both columns added successfully!")
            print("   ✓ photo_url: TEXT")
            print("   ✓ photo_reference: TEXT")
        else:
            print("\n⚠️ Warning: Columns may already exist or migration failed")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if your IP is whitelisted in Render dashboard")
        print("   2. Verify the database URL is correct")
        print("   3. Try using Render CLI: render psql dpg-d4jg45e3jp1c73b6gas0-a")
        return False
        
    except Exception as e:
        print(f"\n❌ Migration Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🗄️  Database Migration: Add Photo Columns")
    print("=" * 60)
    
    success = run_migration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Migration completed successfully!")
        print("\n📌 Next steps:")
        print("   1. Test your backend API endpoints")
        print("   2. Try the chat app - it should work now!")
        sys.exit(0)
    else:
        print("❌ Migration failed. See error above.")
        print("\n📌 Alternative: Use Render Dashboard Shell")
        print("   1. Go to: https://dashboard.render.com")
        print("   2. Select your PostgreSQL database")
        print("   3. Click 'Shell' tab")
        print("   4. Run this SQL:")
        print("      ALTER TABLE restaurants")
        print("      ADD COLUMN IF NOT EXISTS photo_url TEXT,")
        print("      ADD COLUMN IF NOT EXISTS photo_reference TEXT;")
        sys.exit(1)
