"""
Database Setup and Migration Script
====================================

Sets up database tables and runs migrations for AI Istanbul application.

Usage:
    python setup_database.py              # Setup database
    python setup_database.py --test       # Test connection only
    python setup_database.py --migrate    # Run migrations
    python setup_database.py --reset      # Reset database (DANGER!)

Author: AI Istanbul Team
Date: December 2025
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_url():
    """Get database URL from environment"""
    db_url = os.getenv('DATABASE_URL')
    
    if not db_url:
        # Try to construct from individual parameters
        host = os.getenv('POSTGRES_HOST')
        port = os.getenv('POSTGRES_PORT', '5432')
        db_name = os.getenv('POSTGRES_DB')
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        
        if all([host, db_name, user, password]):
            db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            logger.info("✅ Constructed DATABASE_URL from individual parameters")
        else:
            logger.error("❌ No database configuration found in environment")
            return None
    
    # Fix postgres:// to postgresql://
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    
    return db_url


def test_connection():
    """Test database connection"""
    logger.info("🔍 Testing database connection...")
    
    db_url = get_database_url()
    if not db_url:
        return False
    
    try:
        # Add SSL for Render PostgreSQL
        connect_args = {}
        if 'render.com' in db_url or 'dpg-' in db_url:
            connect_args['sslmode'] = 'require'
        
        engine = create_engine(db_url, connect_args=connect_args)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ Connected to database")
            logger.info(f"📊 Database version: {version[:80]}...")
            conn.commit()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False


def create_tables():
    """Create all database tables"""
    logger.info("📝 Creating database tables...")
    
    db_url = get_database_url()
    if not db_url:
        return False
    
    try:
        # Import database and base first
        from backend.database import Base, engine
        
        # Import all models to register them with Base
        from backend.models import (
            User, Place, Museum, Restaurant, Event, ChatHistory, BlogPost,
            FeedbackEvent, UserInteractionAggregate, ItemFeatureVector, OnlineLearningModel,
            LocationHistory, NavigationSession, RouteHistory, NavigationEvent,
            UserPreferences, ChatSession, ConversationHistory
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        logger.info(f"✅ Created {len(tables)} tables:")
        for table in sorted(tables):
            logger.info(f"   - {table}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        return False


def check_existing_tables():
    """Check what tables already exist"""
    logger.info("🔍 Checking existing tables...")
    
    db_url = get_database_url()
    if not db_url:
        return []
    
    try:
        connect_args = {}
        if 'render.com' in db_url or 'dpg-' in db_url:
            connect_args['sslmode'] = 'require'
        
        engine = create_engine(db_url, connect_args=connect_args)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if tables:
            logger.info(f"📊 Found {len(tables)} existing tables:")
            for table in sorted(tables):
                logger.info(f"   - {table}")
        else:
            logger.info("📊 No existing tables found")
        
        return tables
        
    except Exception as e:
        logger.error(f"❌ Failed to check tables: {e}")
        return []


def reset_database():
    """Reset database (DROP ALL TABLES - USE WITH CAUTION!)"""
    logger.warning("⚠️  WARNING: This will DELETE ALL DATA!")
    
    response = input("Type 'YES' to confirm database reset: ")
    if response != 'YES':
        logger.info("❌ Database reset cancelled")
        return False
    
    logger.info("🗑️  Dropping all tables...")
    
    db_url = get_database_url()
    if not db_url:
        return False
    
    try:
        from backend.database import Base, engine
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ All tables dropped")
        
        # Recreate tables
        return create_tables()
        
    except Exception as e:
        logger.error(f"❌ Failed to reset database: {e}")
        return False


def setup_initial_data():
    """Setup initial data (admin user, etc.)"""
    logger.info("👤 Setting up initial data...")
    
    try:
        from backend.database import SessionLocal
        from backend.models import User
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        db = SessionLocal()
        
        # Check if admin exists
        admin = db.query(User).filter(User.email == "admin@aistanbul.com").first()
        
        if not admin:
            # Create admin user
            admin = User(
                username="admin",
                email="admin@aistanbul.com",
                full_name="AI Istanbul Admin",
                hashed_password=pwd_context.hash("admin123"),
                is_admin=True,
                created_at=datetime.utcnow()
            )
            db.add(admin)
            db.commit()
            logger.info("✅ Created admin user (email: admin@aistanbul.com, password: admin123)")
        else:
            logger.info("ℹ️  Admin user already exists")
        
        db.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup initial data: {e}")
        return False


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Database setup and migration')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    parser.add_argument('--migrate', action='store_true', help='Run migrations')
    parser.add_argument('--reset', action='store_true', help='Reset database (DANGER!)')
    parser.add_argument('--init-data', action='store_true', help='Initialize data')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("AI ISTANBUL - DATABASE SETUP")
    logger.info("=" * 80)
    
    # Test connection
    if not test_connection():
        logger.error("❌ Cannot proceed without database connection")
        return 1
    
    if args.test:
        logger.info("✅ Connection test complete")
        return 0
    
    if args.reset:
        if not reset_database():
            return 1
        logger.info("✅ Database reset complete")
        return 0
    
    # Check existing tables
    existing_tables = check_existing_tables()
    
    # Create tables if they don't exist
    if not existing_tables or args.migrate:
        if not create_tables():
            return 1
    
    # Setup initial data
    if args.init_data or not existing_tables:
        if not setup_initial_data():
            logger.warning("⚠️  Failed to setup initial data (non-critical)")
    
    logger.info("=" * 80)
    logger.info("✅ DATABASE SETUP COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start the backend server: cd backend && uvicorn main:app --reload")
    logger.info("2. Access admin dashboard: http://localhost:8000/admin")
    logger.info("3. Login with: admin@aistanbul.com / admin123")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    exit(main())
