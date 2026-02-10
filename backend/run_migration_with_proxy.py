"""
Run migration with Cloud SQL Proxy - bypassing SSL configuration
"""
import os
import sys

# Set environment to bypass SSL when using proxy
os.environ['DATABASE_URL'] = 'postgresql://postgres:NewSecurePassword123!@localhost:5433/postgres'

# Import after setting env
from sqlalchemy import create_engine, text
from db.base import Base
from models import (
    FeedbackEvent,
    UserInteractionAggregate,
    ItemFeatureVector,
    OnlineLearningModel
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Create tables without SSL for Cloud SQL Proxy"""
    try:
        # Create engine without SSL (proxy provides encryption)
        engine = create_engine(
            os.environ['DATABASE_URL'],
            connect_args={'sslmode': 'disable'},
            pool_pre_ping=True
        )
        
        logger.info("🔧 Creating real-time learning tables via Cloud SQL Proxy...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Successfully created tables:")
        logger.info("   - feedback_events")
        logger.info("   - user_interaction_aggregates")
        logger.info("   - item_feature_vectors")
        logger.info("   - online_learning_models")
        
        # Verify tables
        with engine.connect() as conn:
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public';"))
            tables = [row[0] for row in result]
            logger.info(f"\n📊 Tables in database: {', '.join(tables)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating tables: {str(e)}")
        return False

if __name__ == "__main__":
    success = create_tables()
    sys.exit(0 if success else 1)
