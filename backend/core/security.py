"""
Security Middleware Module

Provides authentication and authorization middleware for the API:
- JWT Token validation
- API Key verification
- Role-based access control (RBAC)

Author: AI Istanbul Team
Date: December 2024
"""

import logging
import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from functools import wraps

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt

# Try to import bcrypt directly, fallback to hashlib if not available
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ===========================================
# Configuration
# ===========================================

# JWT Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# API Key Settings
API_KEY_HEADER = "X-API-Key"
VALID_API_KEYS = set(filter(None, os.getenv("VALID_API_KEYS", "").split(",")))

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


# ===========================================
# User Roles
# ===========================================

class UserRole:
    """User roles for RBAC"""
    ADMIN = "admin"
    USER = "user"
    ANONYMOUS = "anonymous"
    API_CLIENT = "api_client"


# ===========================================
# Token Models
# ===========================================

class TokenData:
    """Data extracted from JWT token"""
    def __init__(
        self,
        user_id: str,
        email: Optional[str] = None,
        roles: List[str] = None,
        exp: Optional[datetime] = None
    ):
        self.user_id = user_id
        self.email = email
        self.roles = roles or [UserRole.USER]
        self.exp = exp

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role"""
        return role in self.roles

    def is_admin(self) -> bool:
        """Check if user is admin"""
        return UserRole.ADMIN in self.roles


# ===========================================
# JWT Token Functions
# ===========================================

def create_access_token(
    user_id: str,
    email: Optional[str] = None,
    roles: List[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a new JWT access token"""
    to_encode = {
        "sub": user_id,
        "email": email,
        "roles": roles or [UserRole.USER],
        "iat": datetime.utcnow()
    }
    
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        user_id = payload.get("sub")
        if user_id is None:
            return None
            
        return TokenData(
            user_id=user_id,
            email=payload.get("email"),
            roles=payload.get("roles", [UserRole.USER]),
            exp=datetime.fromtimestamp(payload.get("exp", 0))
        )
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (or fallback to SHA256)"""
    if BCRYPT_AVAILABLE:
        # Use bcrypt directly
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password_bytes, salt).decode('utf-8')
    else:
        # Fallback to SHA256 with salt (less secure but works everywhere)
        salt = secrets.token_hex(16)
        hash_obj = hashlib.sha256((salt + password).encode('utf-8'))
        return f"sha256${salt}${hash_obj.hexdigest()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    if BCRYPT_AVAILABLE and not hashed_password.startswith("sha256$"):
        # bcrypt verification
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.warning(f"Password verification failed: {e}")
            return False
    elif hashed_password.startswith("sha256$"):
        # SHA256 fallback verification
        parts = hashed_password.split("$")
        if len(parts) != 3:
            return False
        salt = parts[1]
        stored_hash = parts[2]
        hash_obj = hashlib.sha256((salt + plain_password).encode('utf-8'))
        return hash_obj.hexdigest() == stored_hash
    else:
        return False



# ===========================================
# API Key Verification
# ===========================================

def verify_api_key(api_key: str) -> bool:
    """Verify if an API key is valid"""
    if not VALID_API_KEYS:
        # No API keys configured - allow all (dev mode)
        logger.warning("⚠️ No API keys configured - API key verification disabled")
        return True
    return api_key in VALID_API_KEYS


# ===========================================
# FastAPI Dependencies
# ===========================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[TokenData]:
    """
    Get current authenticated user from JWT token or API key.
    Returns None if no authentication provided (for optional auth).
    """
    # Try JWT token first
    if credentials and credentials.credentials:
        token_data = verify_token(credentials.credentials)
        if token_data:
            return token_data
    
    # Try API key
    if api_key and verify_api_key(api_key):
        return TokenData(
            user_id="api_client",
            roles=[UserRole.API_CLIENT]
        )
    
    return None


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> TokenData:
    """
    Require authentication - raises 401 if not authenticated.
    Use as a dependency for protected endpoints.
    """
    user = await get_current_user(credentials, api_key)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Authentication required",
                "code": "AUTH_REQUIRED",
                "details": {
                    "message": "Please provide a valid JWT token or API key"
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def require_admin(
    user: TokenData = Depends(require_auth)
) -> TokenData:
    """
    Require admin role - raises 403 if not admin.
    Use as a dependency for admin-only endpoints.
    """
    if not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Admin access required",
                "code": "ADMIN_REQUIRED",
                "details": {
                    "message": "This endpoint requires admin privileges",
                    "user_roles": user.roles
                }
            }
        )
    
    return user


def require_roles(*required_roles: str):
    """
    Factory for role-based access control dependency.
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(user: TokenData = Depends(require_roles("admin", "moderator"))):
            ...
    """
    async def check_roles(user: TokenData = Depends(require_auth)) -> TokenData:
        if not any(user.has_role(role) for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "code": "ROLE_REQUIRED",
                    "details": {
                        "required_roles": list(required_roles),
                        "user_roles": user.roles
                    }
                }
            )
        return user
    
    return check_roles


# ===========================================
# Security Middleware
# ===========================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add user information to request state.
    Does not block requests - just extracts auth info if present.
    """
    
    # Paths that don't need any auth processing
    SKIP_PATHS = {"/", "/health", "/api/health", "/docs", "/openapi.json", "/redoc"}
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth processing for certain paths
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)
        
        # Try to extract user info
        user = None
        
        # Check for Bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = verify_token(token)
        
        # Check for API key
        if not user:
            api_key = request.headers.get(API_KEY_HEADER)
            if api_key and verify_api_key(api_key):
                user = TokenData(user_id="api_client", roles=[UserRole.API_CLIENT])
        
        # Store user in request state
        request.state.user = user
        
        return await call_next(request)


# ===========================================
# Utility Functions
# ===========================================

def get_user_from_request(request: Request) -> Optional[TokenData]:
    """Get user from request state (set by AuthenticationMiddleware)"""
    return getattr(request.state, 'user', None)


def is_authenticated(request: Request) -> bool:
    """Check if request is authenticated"""
    user = get_user_from_request(request)
    return user is not None


def generate_api_key() -> str:
    """Generate a new secure API key"""
    return f"aist_{secrets.token_urlsafe(32)}"


# ===========================================
# Setup Function
# ===========================================

def setup_security(app):
    """Set up security middleware on the app"""
    app.add_middleware(AuthenticationMiddleware)
    logger.info("✅ Security middleware configured")
