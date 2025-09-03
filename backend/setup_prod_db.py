#!/usr/bin/env python3
"""
Production Database Setup Script
Run this script to set up the database for production deployment
"""

import os
import sqlite3
from sqlalchemy.orm import sessionmaker
from database import engine, Base
from models import User, Place, Museum, Restaurant, Event

def setup_production_database():
    """Set up the production database with all necessary tables and data"""
    
    print("🗄️ Setting up production database...")
    
    # Create all tables
    print("📋 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully")
    
    # Check if we need to populate seed data
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    try:
        place_count = session.query(Place).count()
        if place_count == 0:
            print("📦 Database is empty, loading seed data...")
            
            # Load seed data from SQL file
            db_path = os.path.join(os.path.dirname(__file__), 'app.db')
            seed_path = os.path.join(os.path.dirname(__file__), 'db', 'seed.sql')
            
            if os.path.exists(seed_path):
                with open(seed_path, 'r') as f:
                    seed_sql = f.read()
                
                # Execute seed SQL
                conn = sqlite3.connect(db_path)
                conn.executescript(seed_sql)
                conn.commit()
                conn.close()
                
                print("✅ Seed data loaded successfully")
            else:
                print("⚠️ Seed file not found, database will be empty")
        else:
            print(f"✅ Database already has {place_count} places")
        
        # Verify data
        place_count = session.query(Place).count()
        user_count = session.query(User).count()
        
        print(f"📊 Database status:")
        print(f"   - Places: {place_count}")
        print(f"   - Users: {user_count}")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        session.rollback()
        raise
    finally:
        session.close()
    
    print("🎉 Production database setup complete!")

if __name__ == "__main__":
    setup_production_database()
