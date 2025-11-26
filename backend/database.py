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
original_database_url = DATABASE_URL
print(f"🔍 DATABASE_URL from environment: {repr(original_database_url)}")

# Fix postgres:// to postgresql:// for newer SQLAlchemy versions
if original_database_url and original_database_url.startswith('postgres://'):
    DATABASE_URL = original_database_url.replace('postgres://', 'postgresql://', 1)
    print(f"🔧 Fixed DATABASE_URL scheme: postgres:// -> postgresql://")
elif original_database_url:
    DATABASE_URL = original_database_url

if not DATABASE_URL:
    print("⚠️ WARNING: No DATABASE_URL found in environment, using SQLite for local development")
    DATABASE_URL = "sqlite:///./app.db"
    print(f"🗃️ Using SQLite database: {DATABASE_URL}")
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False},
        echo=False  # Set to True only for debugging
    )
else:
    print(f"🔒 Using PostgreSQL database connection: {DATABASE_URL[:50]}...")
    # PostgreSQL configuration with connection pooling and security settings
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,  # Number of connections to keep open
        max_overflow=20,  # Additional connections when needed
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=3600,  # Recycle connections every hour
        echo=False  # Never log SQL in production
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency function to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


