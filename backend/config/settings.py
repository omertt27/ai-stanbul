"""
Configuration Settings Module

Centralized configuration and environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""
    
    # App Configuration
    APP_NAME = "Istanbul AI Guide API"
    APP_VERSION = "2.1.0"
    APP_DESCRIPTION = "AI-powered Istanbul travel guide with enhanced authentication and production infrastructure"
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    # Cloud Run sets PORT, fallback to API_PORT, then 8000
    API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    # Admin Credentials
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_stanbul.db")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "ai-istanbul-media")
    AWS_S3_BLOG_PATH = "blog/images/"
    AWS_S3_FEATURED_PATH = "blog/featured-images/"
    AWS_S3_CONTENT_PATH = "blog/content-images/"
    AWS_S3_THUMBNAILS_PATH = "blog/thumbnails/"
    
    # LLM Configuration
    LLM_API_URL = os.getenv("LLM_API_URL", "")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    PURE_LLM_MODE = os.getenv("PURE_LLM_MODE", "false").lower() == "true"
    
    # ML Service
    ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8000")
    ML_USE_LLM_DEFAULT = os.getenv("ML_USE_LLM_DEFAULT", "true").lower() == "true"
    
    # CORS Origins
    CORS_ORIGINS = [
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://ai-stanbul.vercel.app",
        "*"  # Remove in production!
    ]
    
    # Feature Flags
    RATE_LIMITING_ENABLED = False
    ENHANCED_AUTH_AVAILABLE = False
    INFRASTRUCTURE_AVAILABLE = False
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production"""
        return os.getenv("ENVIRONMENT", "development") == "production"


settings = Settings()
