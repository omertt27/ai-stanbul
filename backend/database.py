import os
import sys
import logging
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Track if database has been configured (prevent duplicate logging)
_db_config_logged = False

# Add config directory to path
config_path = os.path.join(os.path.dirname(__file__), 'config')
if config_path not in sys.path:
    sys.path.insert(0, config_path)

# PostgreSQL connection event listeners for better reliability
@event.listens_for(Engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Set PostgreSQL connection parameters on connect"""
    try:
        cursor = dbapi_conn.cursor()
        # Set statement timeout (30 seconds)
        cursor.execute("SET statement_timeout = 30000")
        # Set timezone
        cursor.execute("SET timezone = 'UTC'")
        cursor.close()
    except Exception as e:
        logger.warning(f"Could not set connection parameters: {e}")

# Import database configuration
try:
    from config.database_config import get_database_config, db_config
    USE_CENTRALIZED_CONFIG = True
    logger.info("✅ Using centralized database configuration")
except ImportError:
    USE_CENTRALIZED_CONFIG = False
    logger.warning("⚠️ Using legacy database configuration")

Base = declarative_base()

# Database configuration - only log once
if USE_CENTRALIZED_CONFIG:
    # Use centralized configuration (log only once)
    if not _db_config_logged:
        db_config.log_configuration()
        _db_config_logged = True
    
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
        logger.info(f"🗃️ SQLite engine created (fallback)")
    else:
        # AWS RDS or other PostgreSQL
        connect_args = {
            'connect_timeout': 10,
            'keepalives': 1,
            'keepalives_idle': 30,
            'keepalives_interval': 10,
            'keepalives_count': 5,
        }
        
        # Add SSL for specific providers
        if 'render.com' in DATABASE_URL or 'dpg-' in DATABASE_URL:
            connect_args['sslmode'] = 'require'
        elif 'rds.amazonaws.com' in DATABASE_URL or 'amazonaws.com' in DATABASE_URL:
            # AWS RDS - SSL recommended but not required for public access
            connect_args['sslmode'] = 'prefer'
            logger.info("🔒 AWS RDS detected - SSL preferred")
        
        # Use QueuePool for PostgreSQL, NullPool for serverless environments
        is_serverless = os.getenv('ENVIRONMENT') == 'serverless' or os.getenv('IS_CLOUD_RUN') == 'true'
        
        engine = create_engine(
            DATABASE_URL,
            poolclass=NullPool if is_serverless else QueuePool,
            pool_size=5 if not is_serverless else None,
            max_overflow=10 if not is_serverless else None,
            pool_pre_ping=True,
            pool_recycle=1800,  # Recycle connections every 30 minutes
            echo=False,
            connect_args=connect_args
        )
        
        pool_type = "NullPool (serverless)" if is_serverless else "QueuePool"
        logger.info(f"🐘 PostgreSQL engine created with {pool_type}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Log database backend information
try:
    logger.info("=" * 60)
    logger.info("📊 DATABASE BACKEND INFORMATION")
    logger.info(f"   Type: {engine.dialect.name}")
    logger.info(f"   Driver: {engine.driver}")
    if hasattr(engine.url, 'database'):
        logger.info(f"   Database: {engine.url.database}")
    if hasattr(engine.url, 'host') and engine.url.host:
        logger.info(f"   Host: {engine.url.host}")
        logger.info(f"   Port: {engine.url.port or 'default'}")
    logger.info(f"   Pool: {engine.pool.__class__.__name__}")
    logger.info("=" * 60)
except Exception as e:
    logger.warning(f"⚠️ Could not log database info: {e}")

# Dependency function to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


