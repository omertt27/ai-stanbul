"""
Database Migration for Real-Time Learning Tables
Creates tables for feedback events and user interaction aggregates
"""

import sys
import os

# Add parent directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, backend_dir)

from db.base import Base
from database import engine
from models import (
    FeedbackEvent,
    UserInteractionAggregate,
    ItemFeatureVector,
    OnlineLearningModel
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_realtime_learning_tables():
    """
    Create tables for real-time learning system
    """
    try:
        logger.info("🔧 Creating real-time learning tables...")
        
        # Create all tables defined in models
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Successfully created tables:")
        logger.info("   - feedback_events")
        logger.info("   - user_interaction_aggregates")
        logger.info("   - item_feature_vectors")
        logger.info("   - online_learning_models")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating tables: {str(e)}")
        return False


def verify_tables():
    """
    Verify that tables were created successfully
    """
    from sqlalchemy import inspect
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        required_tables = [
            'feedback_events',
            'user_interaction_aggregates',
            'item_feature_vectors',
            'online_learning_models'
        ]
        
        logger.info("\n📊 Verifying tables...")
        for table in required_tables:
            if table in tables:
                logger.info(f"   ✅ {table} exists")
                
                # Show columns
                columns = inspector.get_columns(table)
                logger.info(f"      Columns: {', '.join([col['name'] for col in columns])}")
            else:
                logger.warning(f"   ⚠️ {table} not found")
        
        return all(table in tables for table in required_tables)
        
    except Exception as e:
        logger.error(f"❌ Error verifying tables: {str(e)}")
        return False


def main():
    """Main migration function"""
    logger.info("=" * 60)
    logger.info("Real-Time Learning Database Migration")
    logger.info("=" * 60)
    
    # Create tables
    if create_realtime_learning_tables():
        logger.info("\n✅ Table creation successful!")
        
        # Verify tables
        if verify_tables():
            logger.info("\n✅ All tables verified successfully!")
            logger.info("\n🚀 Real-time learning system is ready!")
            return True
        else:
            logger.error("\n❌ Table verification failed")
            return False
    else:
        logger.error("\n❌ Table creation failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
