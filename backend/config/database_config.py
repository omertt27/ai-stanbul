"""
Database Configuration for PostgreSQL
======================================

Centralized database configuration with support for:
- Render PostgreSQL (Production)
- Local PostgreSQL (Development)
- SQLite (Fallback)

Author: AI Istanbul Team
Date: December 2025
"""

import os
import logging
from typing import Dict, Optional
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DatabaseConfig:
    """Database configuration manager"""
    
    def __init__(self):
        """Initialize database configuration"""
        self.database_url = self._get_database_url()
        self.is_postgres = self._is_postgresql()
        self.is_render = self._is_render_postgres()
        self.connection_params = self._get_connection_params()
        
    def _get_database_url(self) -> str:
        """Get database URL from environment"""
        db_url = os.getenv('DATABASE_URL')
        
        if not db_url:
            # Try to construct from individual parameters
            host = os.getenv('POSTGRES_HOST', os.getenv('DATABASE_HOST', 'localhost'))
            # Default to 5433 for Cloud SQL Proxy, fallback to 5432 for local
            port = os.getenv('POSTGRES_PORT', os.getenv('DATABASE_PORT', os.getenv('CLOUDSQL_PORT', '5433')))
            db_name = os.getenv('POSTGRES_DB', os.getenv('DATABASE_NAME', 'postgres'))
            user = os.getenv('POSTGRES_USER', os.getenv('DATABASE_USER', 'postgres'))
            password = os.getenv('POSTGRES_PASSWORD', os.getenv('DATABASE_PASSWORD', ''))
            
            if all([host, db_name, user]):
                if password:
                    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
                else:
                    db_url = f"postgresql://{user}@{host}:{port}/{db_name}"
                logger.info(f"✅ Constructed DATABASE_URL from individual parameters (port={port})")
            else:
                logger.warning("⚠️ No DATABASE_URL found, using SQLite")
                db_url = "sqlite:///./app.db"
        
        # Fix postgres:// to postgresql:// for SQLAlchemy
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
            logger.info("🔧 Fixed DATABASE_URL scheme: postgres:// -> postgresql://")
        
        return db_url
    
    def _is_postgresql(self) -> bool:
        """Check if using PostgreSQL"""
        return self.database_url.startswith('postgresql://')
    
    def _is_render_postgres(self) -> bool:
        """Check if using Render PostgreSQL"""
        return 'render.com' in self.database_url or 'dpg-' in self.database_url
    
    def _is_cloud_sql_unix_socket(self) -> bool:
        """Check if using Cloud SQL Unix socket connection"""
        return '/cloudsql/' in self.database_url
    
    def _get_connection_params(self) -> Dict:
        """Get SQLAlchemy connection parameters"""
        if not self.is_postgres:
            # SQLite configuration
            return {
                'connect_args': {'check_same_thread': False},
                'echo': False
            }
        
        # PostgreSQL configuration
        params = {
            'poolclass': 'QueuePool',
            'pool_size': 10,
            'max_overflow': 20,
            'pool_pre_ping': True,
            'pool_recycle': 3600,
            'echo': False,
            'connect_args': {}
        }
        
        # Add SSL for Render (not for Cloud SQL Unix sockets)
        if self.is_render:
            params['connect_args']['sslmode'] = 'require'
        elif not self._is_cloud_sql_unix_socket():
            # For TCP connections (not Unix sockets), add connection timeout
            params['connect_args']['connect_timeout'] = 5
        
        # For Cloud SQL with Unix sockets, we don't need connect_args
        # PostgreSQL driver will handle Unix sockets via the host parameter
        
        return params
    
    def get_sqlalchemy_url(self) -> str:
        """Get SQLAlchemy-compatible database URL"""
        return self.database_url
    
    def get_connection_info(self) -> Dict:
        """Get database connection information"""
        parsed = urlparse(self.database_url)
        
        # Extract Cloud SQL socket path if present
        query_params = parse_qs(parsed.query)
        socket_path = query_params.get('host', [None])[0]
        
        return {
            'type': 'postgresql' if self.is_postgres else 'sqlite',
            'host': parsed.hostname or 'localhost',
            'port': parsed.port,
            'database': parsed.path.lstrip('/').split('?')[0],  # Remove query string
            'username': parsed.username,
            'is_render': self.is_render,
            'is_cloud_sql_socket': self._is_cloud_sql_unix_socket(),
            'socket_path': socket_path,
            'has_ssl': self.is_render
        }
    
    def log_configuration(self):
        """Log database configuration (without sensitive info)"""
        info = self.get_connection_info()
        
        logger.info("=" * 60)
        logger.info("DATABASE CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Database Type: {info['type'].upper()}")
        
        if self.is_postgres:
            if info['is_cloud_sql_socket']:
                logger.info(f"Connection: Cloud SQL Unix Socket")
                logger.info(f"Socket Path: {info['socket_path']}")
            else:
                logger.info(f"Host: {info['host']}")
                logger.info(f"Port: {info['port']}")
            logger.info(f"Database: {info['database']}")
            logger.info(f"Username: {info['username']}")
            logger.info(f"Render PostgreSQL: {'Yes' if info['is_render'] else 'No'}")
            logger.info(f"SSL Enabled: {'Yes' if info['has_ssl'] else 'No'}")
        else:
            logger.info(f"Database File: {info['database']}")
        
        logger.info("=" * 60)


# Global configuration instance
db_config = DatabaseConfig()


def get_database_config() -> DatabaseConfig:
    """Get database configuration instance"""
    return db_config


def test_database_connection() -> bool:
    """Test database connection"""
    try:
        from sqlalchemy import create_engine, text
        
        engine = create_engine(
            db_config.get_sqlalchemy_url(),
            **db_config.connection_params
        )
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            conn.commit()
        
        logger.info("✅ Database connection successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log configuration
    db_config.log_configuration()
    
    # Test connection
    print("\n🔍 Testing database connection...")
    if test_database_connection():
        print("✅ Database is ready to use!")
    else:
        print("❌ Database connection failed. Check your configuration.")
