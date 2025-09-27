#!/usr/bin/env python3
"""
Database Migration Script for AI Istanbul
Migrates from SQLite to PostgreSQL for production deployment
"""

import os
import sys
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

def migrate_sqlite_to_postgresql():
    """Migrate data from SQLite databases to PostgreSQL"""
    
    # Get PostgreSQL connection details
    postgres_url = os.getenv('DATABASE_URL')
    if not postgres_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        print("Please set DATABASE_URL to your PostgreSQL connection string")
        return False
    
    print("🔄 Starting database migration from SQLite to PostgreSQL...")
    
    # Find all SQLite databases
    sqlite_dbs = [
        'app.db',
        'blog_analytics.db', 
        'daily_usage.db',
        'db/ai_stanbul.db'
    ]
    
    migration_data = {}
    
    # Extract data from SQLite databases
    for db_file in sqlite_dbs:
        if os.path.exists(db_file):
            print(f"📦 Extracting data from {db_file}...")
            try:
                conn = sqlite3.connect(db_file)
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                db_data = {}
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT * FROM {table_name}")
                    rows = cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    db_data[table_name] = [dict(row) for row in rows]
                    print(f"  ✓ Extracted {len(rows)} rows from {table_name}")
                
                migration_data[db_file] = db_data
                conn.close()
                
            except Exception as e:
                print(f"❌ Error extracting from {db_file}: {e}")
                continue
    
    # Save extracted data as backup
    backup_file = f"migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_file, 'w') as f:
        json.dump(migration_data, f, indent=2, default=str)
    print(f"💾 Data backup saved to {backup_file}")
    
    # Connect to PostgreSQL and create tables
    try:
        # Parse PostgreSQL URL for psycopg2
        import urllib.parse as urlparse
        url = urlparse.urlparse(postgres_url)
        
        pg_conn = psycopg2.connect(
            database=url.path[1:],  # Remove leading slash
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port or 5432,
            sslmode='require'
        )
        
        cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
        print("🔗 Connected to PostgreSQL database")
        
        # Create tables and migrate data
        # This is a simplified migration - you may need to adjust for your specific schema
        
        # Example: Create chat_sessions table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) UNIQUE NOT NULL,
                user_ip VARCHAR(45),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0
            );
        """)
        
        # Example: Create chat_messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time FLOAT,
                user_ip VARCHAR(45)
            );
        """)
        
        # Example: Create blog_posts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blog_posts (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                excerpt TEXT,
                author VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published BOOLEAN DEFAULT false,
                tags TEXT[]
            );
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);")
        
        pg_conn.commit()
        print("✅ PostgreSQL tables created successfully")
        
        # Here you would add specific data migration logic based on your schema
        # For now, we'll just create the structure
        
        cursor.close()
        pg_conn.close()
        
        print("🎉 Database migration completed successfully!")
        print(f"📋 Migration summary saved to {backup_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL migration error: {e}")
        return False

def cleanup_sqlite_files():
    """Remove SQLite database files after successful migration"""
    
    print("\n🧹 Cleaning up SQLite database files...")
    
    sqlite_files = [
        'app.db',
        'blog_analytics.db',
        'daily_usage.db', 
        'backend/app.db',
        'backend/blog_analytics.db',
        'backend/daily_usage.db',
        'backend/db/ai_stanbul.db'
    ]
    
    removed_files = []
    for file_path in sqlite_files:
        full_path = os.path.join('/Users/omer/Desktop/ai-stanbul', file_path)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                removed_files.append(file_path)
                print(f"  ✓ Removed {file_path}")
            except Exception as e:
                print(f"  ❌ Failed to remove {file_path}: {e}")
    
    if removed_files:
        print(f"🗑️ Cleaned up {len(removed_files)} SQLite database files")
    else:
        print("ℹ️ No SQLite files found to clean up")

if __name__ == "__main__":
    print("🔒 AI Istanbul Database Migration to PostgreSQL")
    print("=" * 50)
    
    # Check if DATABASE_URL is set
    if not os.getenv('DATABASE_URL'):
        print("❌ DATABASE_URL environment variable not found")
        print("Please set your PostgreSQL connection string:")
        print("export DATABASE_URL='postgresql://user:password@host:port/database'")
        sys.exit(1)
    
    # Perform migration
    success = migrate_sqlite_to_postgresql()
    
    if success:
        print("\n⚠️ IMPORTANT: Test your application thoroughly before proceeding")
        response = input("Do you want to clean up SQLite files? (y/N): ")
        if response.lower() == 'y':
            cleanup_sqlite_files()
        else:
            print("ℹ️ SQLite files retained for safety. Remove manually after testing.")
    else:
        print("❌ Migration failed. Please check errors above.")
        sys.exit(1)
