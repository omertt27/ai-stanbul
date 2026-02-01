"""
Database Configuration for PostgreSQL
======================================

Production-ready database configuration with support for:
- GCP Cloud SQL (Production - Cloud Run)
- GCP Cloud SQL Proxy (Development)
- Local PostgreSQL (Development)
- SQLite (Fallback)

Features:
- Cloud Run Unix socket optimization
- Connection pooling and retry logic
- SSL/TLS security
- Hybrid cloud support (GCP DB + AWS Redis)

Author: AI Istanbul Team
Date: February 2026
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
        self.is_gcp = self._is_gcp_postgres()
        self.is_production = self._is_production()
        self.is_cloud_run = self._is_cloud_run()
        self.connection_params = self._get_connection_params()
        
    def _is_production(self) -> bool:
        """Check if running in production environment"""
        return (
            os.getenv('ENVIRONMENT', '').lower() in ['production', 'prod'] or
            os.getenv('K_SERVICE') is not None or  # Cloud Run
            os.getenv('GAE_APPLICATION') is not None or  # App Engine
            'production' in os.getenv('NODE_ENV', '').lower()
        )
    
    def _is_cloud_run(self) -> bool:
        """Check if running on Cloud Run"""
        return os.getenv('K_SERVICE') is not None
        
    def _get_database_url(self) -> str:
        """Get database URL from environment with Cloud Run optimization"""
        db_url = os.getenv('DATABASE_URL')
        
        if not db_url:
            # Try to construct from individual parameters
            host = os.getenv('POSTGRES_HOST', os.getenv('DATABASE_HOST'))
            port = os.getenv('POSTGRES_PORT', os.getenv('DATABASE_PORT', '5432'))
            db_name = os.getenv('POSTGRES_DB', os.getenv('DATABASE_NAME', 'postgres'))
            user = os.getenv('POSTGRES_USER', os.getenv('DATABASE_USER', 'postgres'))
            password = os.getenv('POSTGRES_PASSWORD', os.getenv('DATABASE_PASSWORD', ''))
            
            # GCP Cloud SQL Production Configuration
            cloud_sql_instance = os.getenv('CLOUD_SQL_INSTANCE')
            gcp_project = os.getenv('GOOGLE_CLOUD_PROJECT', os.getenv('GCP_PROJECT_ID'))
            
            if self.is_production or self.is_cloud_run:
                # Production: Prefer Cloud SQL Connection Name (Unix Socket) for Cloud Run
                if cloud_sql_instance and gcp_project:
                    # Cloud Run uses Unix socket connection for optimal performance
                    socket_path = f"/cloudsql/{cloud_sql_instance}"
                    if password:
                        db_url = f"postgresql://{user}:{password}@/{db_name}?host={socket_path}"
                    else:
                        db_url = f"postgresql://{user}@/{db_name}?host={socket_path}"
                    logger.info(f"🚀 Cloud Run Production: Using Cloud SQL Unix Socket: {socket_path}")
                elif host and host.startswith(('34.', '35.')):
                    # Fallback to public IP for production
                    logger.warning("⚠️ Production using public IP - Unix socket recommended for Cloud Run")
                    if password:
                        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}?sslmode=require"
                    else:
                        db_url = f"postgresql://{user}@{host}:{port}/{db_name}?sslmode=require"
                else:
                    logger.error("❌ Production requires CLOUD_SQL_INSTANCE or valid DATABASE_HOST")
                    raise ValueError("Production database configuration incomplete")
            else:
                # Development: Use Cloud SQL Proxy or direct connection
                if not host and gcp_project and cloud_sql_instance:
                    host = '127.0.0.1'
                    port = '5433'  # Cloud SQL Proxy default
                    logger.info("🔧 Development: Using Cloud SQL Proxy (127.0.0.1:5433)")
                elif not host:
                    host = 'localhost'
                    port = '5432'
                    logger.info("🔧 Development: Using local PostgreSQL")
                
                if all([host, db_name, user]):
                    if password:
                        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
                    else:
                        db_url = f"postgresql://{user}@{host}:{port}/{db_name}"
                    logger.info(f"✅ Development: Constructed DATABASE_URL (host={host}, port={port})")
                else:
                    logger.warning("⚠️ No DATABASE_URL found, using SQLite fallback")
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
    
    def _is_gcp_postgres(self) -> bool:
        """Check if using GCP PostgreSQL (Cloud SQL)"""
        # Check for GCP IP patterns or explicit GCP indicators
        parsed = urlparse(self.database_url)
        host = parsed.hostname or ''
        
        # GCP Cloud SQL indicators
        return (
            host.startswith('34.') or host.startswith('35.') or  # Common GCP IP ranges
            host == '34.38.193.1' or  # Specific GCP instance IP
            'cloudsql' in self.database_url.lower() or
            'gcp' in self.database_url.lower() or
            os.getenv('GCP_PROJECT_ID') is not None or  # GCP environment
            os.getenv('GCP_CLOUDSQL_INSTANCE') is not None
        )
    
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
        
        # PostgreSQL configuration - Production ready
        params = {
            'pool_size': 15,                # Increased for production
            'max_overflow': 30,             # Handle traffic spikes
            'pool_pre_ping': True,          # Test connections before use
            'pool_recycle': 1800,           # Recycle connections every 30min
            'echo': False,                  # No SQL logging in production
            'connect_args': {},
            'pool_timeout': 30,             # Connection acquisition timeout
            'pool_reset_on_return': 'commit'  # Clean state on return
        }
        
        # Hybrid Cloud Configuration: GCP Database + AWS Redis
        if self.is_render:
            params['connect_args']['sslmode'] = 'require'
        elif self.is_gcp and not self._is_cloud_sql_unix_socket():
            # GCP Cloud SQL with Public IP - Production Configuration
            params['connect_args'].update({
                'connect_timeout': 60,               # Increased timeout for GCP
                'application_name': 'ai-istanbul-backend',  # For monitoring
                'sslmode': 'require',                # Force SSL for GCP
                'keepalives_idle': 600,              # Keep connection alive (10min)
                'keepalives_interval': 30,           # Ping every 30 seconds
                'keepalives_count': 3,               # Max 3 failed pings
                'tcp_user_timeout': 60000,          # 60 second TCP timeout
                'statement_timeout': 30000,         # 30 second statement timeout
                'idle_in_transaction_session_timeout': 300000,  # 5 min idle timeout
            })
            logger.info("🔐 Production: GCP Cloud SQL connection with SSL and monitoring")
        elif self.is_gcp and self._is_cloud_sql_unix_socket():
            # GCP Cloud SQL Unix Socket - Optimal for Cloud Run
            params['connect_args'].update({
                'application_name': 'ai-istanbul-backend',
                'statement_timeout': 30000,
                'idle_in_transaction_session_timeout': 300000,
            })
            logger.info("🚀 Production: GCP Cloud SQL Unix Socket (optimal performance)")
        else:
            # Default PostgreSQL configuration
            params['connect_args'].update({
                'connect_timeout': 30,
                'application_name': 'ai-istanbul-backend',
                'sslmode': 'prefer'
            })
        
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
    """Test database connection with retry logic"""
    import time
    from sqlalchemy import create_engine, text
    
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 Connection attempt {attempt}/{max_retries}...")
            
            engine = create_engine(
                db_config.get_sqlalchemy_url(),
                **db_config.connection_params
            )
            
            with engine.connect() as conn:
                # Test with a simple query
                result = conn.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                if row and row[0] == 1:
                    logger.info("✅ Database connection successful!")
                    return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Connection attempt {attempt} failed: {error_msg}")
            
            # Check for specific error types
            if "timeout expired" in error_msg:
                logger.error("🕐 Connection timeout - check network connectivity to GCP")
            elif "authentication failed" in error_msg:
                logger.error("🔑 Authentication failed - check username/password")
            elif "Connection refused" in error_msg:
                logger.error("🚫 Connection refused - check if database is running and accessible")
            elif "SSL" in error_msg:
                logger.error("🔐 SSL error - check SSL configuration")
            
            if attempt < max_retries:
                logger.info(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                logger.error("❌ All connection attempts failed")
                return False
    
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
