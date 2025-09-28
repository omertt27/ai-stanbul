"""
Authentication module for AI Istanbul Backend
Provides JWT-based authentication for admin access
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

# Security configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
JWT_REFRESH_SECRET_KEY = os.getenv("JWT_REFRESH_SECRET_KEY", "your-super-secret-refresh-key-change-in-production")
JWT_ALGORITHM = "HS256"

# Token expiration settings (enhanced security)
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short-lived access tokens
REFRESH_TOKEN_EXPIRE_DAYS = 7     # Longer-lived refresh tokens
JWT_EXPIRATION_HOURS = 24  # Backward compatibility

# Admin credentials (in production, store hashed passwords in database)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "b1d85ec52f84955ef2b6aefa0cb29421042eb1f8e9e20425d3185422e54ca039")

# HTTPBearer for extracting JWT tokens
security = HTTPBearer(auto_error=False)

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

def create_access_token(data: dict) -> str:
    """Create JWT access token with short expiration"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access_token"})
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token with longer expiration"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh_token"})
    
    encoded_jwt = jwt.encode(to_encode, JWT_REFRESH_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str, is_refresh_token: bool = False) -> Optional[dict]:
    """Verify JWT token and return payload"""
    try:
        secret_key = JWT_REFRESH_SECRET_KEY if is_refresh_token else JWT_SECRET_KEY
        payload = jwt.decode(token, secret_key, algorithms=[JWT_ALGORITHM])
        
        # Verify token type
        expected_type = "refresh_token" if is_refresh_token else "access_token"
        if payload.get("type") != expected_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def refresh_access_token(refresh_token: str) -> dict:
    """Create new access token from valid refresh token"""
    payload = verify_token(refresh_token, is_refresh_token=True)
    
    # Create new access token with same user data
    new_access_token = create_access_token({
        "sub": payload.get("sub"),
        "username": payload.get("username"),
        "role": payload.get("role")
    })
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    }

async def get_current_admin(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Dependency to verify admin authentication"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    payload = verify_token(credentials.credentials)
    
    # Verify admin role
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Verify user still exists and is valid
    username = payload.get("sub") or payload.get("username")  # Handle both "sub" and "username" fields
    if username != ADMIN_USERNAME:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user"
        )
    
    return payload

def authenticate_admin(username: str, password: str) -> Optional[dict]:
    """Authenticate admin user and return user info"""
    if username != ADMIN_USERNAME:
        return None
    
    if not verify_password(password, ADMIN_PASSWORD_HASH):
        return None
    
    return {
        "username": username,
        "role": "admin",
        "authenticated_at": datetime.utcnow().isoformat()
    }
