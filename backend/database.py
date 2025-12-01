import os
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add config directory to path
config_path = os.path.join(os.path.dirname(__file__), 'config')
if config_path not in sys.path:
    sys.path.insert(0, config_path)

# Import database configuration
try:
    from config.database_config import get_database_config, db_config
    USE_CENTRALIZED_CONFIG = True
    logger.info("✅ Using centralized database configuration")
except ImportError:
    USE_CENTRALIZED_CONFIG = False
    logger.warning("⚠️ Using legacy database configuration")

Base = declarative_base()

# Database configuration
if USE_CENTRALIZED_CONFIG:
    # Use centralized configuration
    db_config.log_configuration()
    DATABASE_URL = db_config.get_sqlalchemy_url()
    
    # Create engine with proper configuration
    if db_config.is_postgres:
        engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            connect_args=db_config.connection_params.get('connect_args', {})
        )
        logger.info(f"� PostgreSQL engine created with connection pooling")
    else:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False
        )
        logger.info(f"🗃️ SQLite engine created")
else:
    # Legacy configuration (fallback)
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    
    if not DATABASE_URL:
        DATABASE_URL = "sqlite:///./app.db"
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        connect_args = {}
        if 'render.com' in DATABASE_URL or 'dpg-' in DATABASE_URL:
            connect_args['sslmode'] = 'require'
        
        engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            connect_args=connect_args
        )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency function to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


