import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

Base = declarative_base()

# Production-ready database configuration with PostgreSQL support
DATABASE_URL = os.getenv('DATABASE_URL')

# Fallback to SQLite for local development only
if not DATABASE_URL:
    print("⚠️ WARNING: No DATABASE_URL found in environment, using SQLite for local development")
    DATABASE_URL = "sqlite:///./app.db"
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False},
        echo=False  # Set to True only for debugging
    )
else:
    print(f"🔒 Using PostgreSQL database connection")
    # PostgreSQL configuration with connection pooling and security settings
    
    # Check if SSL mode is already specified in DATABASE_URL
    connect_args = {
        "connect_timeout": 30,
        "application_name": "ai_istanbul_backend"
    }
    
    # Only set sslmode if not already in DATABASE_URL
    if "sslmode=" not in DATABASE_URL:
        if os.getenv('ENVIRONMENT') == 'production':
            connect_args["sslmode"] = "require"  # Require SSL in production
        elif DATABASE_URL.startswith('postgresql://localhost') or '127.0.0.1' in DATABASE_URL:
            connect_args["sslmode"] = "disable"  # Disable SSL for local development
        else:
            connect_args["sslmode"] = "prefer"  # Default to prefer SSL but don't require it
    
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,  # Number of connections to keep open
        max_overflow=20,  # Additional connections when needed
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=3600,  # Recycle connections every hour
        echo=False,  # Never log SQL in production
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


