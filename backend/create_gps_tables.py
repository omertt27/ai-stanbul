"""
Create GPS Navigation Tables on Production Database
===================================================

This script creates all GPS navigation tables on the production PostgreSQL database.
Run this ONCE after deploying to Render.

Usage:
    python backend/create_gps_tables.py
    
Or from Render Shell:
    cd /opt/render/project/src
    python backend/create_gps_tables.py
"""

import sys
import os

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

print("=" * 80)
print("🗄️  GPS NAVIGATION TABLES - CREATION SCRIPT")
print("=" * 80)
print("")

try:
    # Import database and models
    print("📦 Importing database models...")
    from database import Base, engine
    from models import (
        LocationHistory,
        NavigationSession,
        RouteHistory,
        NavigationEvent,
        UserPreferences,
        ChatSession,
        ConversationHistory
    )
    print("✅ Models imported successfully")
    print("")
    
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("")
    print("💡 Troubleshooting:")
    print("   1. Make sure you're in the project root directory")
    print("   2. Check that all dependencies are installed")
    print("   3. Verify DATABASE_URL environment variable is set")
    print("")
    sys.exit(1)

try:
    # Check database connection
    print("🔌 Testing database connection...")
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        print(f"✅ Connected to: {version[:80]}...")
        conn.commit()
    print("")
    
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("")
    print("💡 Troubleshooting:")
    print("   1. Check DATABASE_URL environment variable")
    print("   2. Verify database is running")
    print("   3. Check network connectivity")
    print("")
    sys.exit(1)

try:
    # Check existing tables
    print("📊 Checking existing tables...")
    from sqlalchemy import inspect
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    print(f"   Found {len(existing_tables)} existing tables")
    print("")
    
    # GPS navigation tables to create
    gps_tables = [
        'location_history',
        'navigation_sessions',
        'route_history',
        'navigation_events',
        'user_preferences',
        'chat_sessions',
        'conversation_history'
    ]
    
    # Check which tables already exist
    existing_gps = [t for t in gps_tables if t in existing_tables]
    missing_gps = [t for t in gps_tables if t not in existing_tables]
    
    if existing_gps:
        print("ℹ️  GPS tables already exist:")
        for table in existing_gps:
            print(f"   ✓ {table}")
        print("")
    
    if missing_gps:
        print("📝 GPS tables to create:")
        for table in missing_gps:
            print(f"   • {table}")
        print("")
    else:
        print("✅ All GPS navigation tables already exist!")
        print("")
        print("=" * 80)
        print("🎉 NOTHING TO DO - TABLES ALREADY CREATED")
        print("=" * 80)
        sys.exit(0)
    
except Exception as e:
    print(f"⚠️  Warning: Could not check existing tables: {e}")
    print("   Proceeding with table creation...")
    print("")

try:
    # Create GPS navigation tables
    print("🔨 Creating GPS navigation tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Table creation completed")
    print("")
    
except Exception as e:
    print(f"❌ Error creating tables: {e}")
    print("")
    sys.exit(1)

try:
    # Verify tables were created
    print("✅ Verifying GPS navigation tables...")
    inspector = inspect(engine)
    all_tables = inspector.get_table_names()
    
    gps_tables_status = []
    for table in gps_tables:
        exists = table in all_tables
        status = "✅" if exists else "❌"
        gps_tables_status.append((table, exists))
        print(f"   {status} {table}")
    
    print("")
    
    # Summary
    created_count = sum(1 for _, exists in gps_tables_status if exists)
    total_tables = len(all_tables)
    
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"   GPS Navigation Tables: {created_count}/{len(gps_tables)} ✅")
    print(f"   Total Database Tables: {total_tables}")
    print("")
    
    if created_count == len(gps_tables):
        print("🎉 SUCCESS! All GPS navigation tables are ready!")
        print("")
        print("📝 Next steps:")
        print("   1. Test navigation: Send 'How can I go to Taksim?' in chat")
        print("   2. Grant GPS permission in browser")
        print("   3. See turn-by-turn navigation!")
        print("")
        print("=" * 80)
        sys.exit(0)
    else:
        print("⚠️  WARNING: Some tables may not have been created")
        print("   Please check the error messages above")
        print("")
        print("=" * 80)
        sys.exit(1)
        
except Exception as e:
    print(f"⚠️  Could not verify tables: {e}")
    print("   Tables may have been created successfully")
    print("")
    print("=" * 80)
    sys.exit(0)
