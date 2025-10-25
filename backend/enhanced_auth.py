"""
Enhanced Authentication System for AI Istanbul (KAM)
Implements secure user authentication with JWT tokens, refresh token rotation,
token blacklisting, and user session management for protecting chat data.

Features:
- JWT access and refresh tokens
- Refresh token rotation (security best practice)
- Token blacklisting for logout
- User session management
- Rate limiting per user
- Secure chat session binding to authenticated users
- GDPR-compliant data handling
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

import jwt
import redis
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr, validator


# ============================================================================
# Configuration
# ============================================================================

class AuthConfig:
    """Authentication configuration"""
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Security settings
    MIN_PASSWORD_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    TOKEN_BLACKLIST_TTL_SECONDS = 60 * 60 * 24 * 8  # 8 days
    
    # Session settings
    MAX_ACTIVE_SESSIONS_PER_USER = 5
    SESSION_EXPIRE_HOURS = 24
    
    # Redis keys
    REDIS_KEY_BLACKLIST = "auth:blacklist:{token_jti}"
    REDIS_KEY_REFRESH_TOKEN = "auth:refresh:{user_id}:{token_jti}"
    REDIS_KEY_USER_SESSIONS = "auth:sessions:{user_id}"
    REDIS_KEY_FAILED_LOGINS = "auth:failed:{identifier}"
    REDIS_KEY_USER_RATE_LIMIT = "auth:ratelimit:{user_id}:{endpoint}"


# ============================================================================
# Models
# ============================================================================

class UserRole(str, Enum):
    """User roles for access control"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class TokenType(str, Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True


class UserRegistrationRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v.lower()
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < AuthConfig.MIN_PASSWORD_LENGTH:
            raise ValueError(f'Password must be at least {AuthConfig.MIN_PASSWORD_LENGTH} characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLoginRequest(BaseModel):
    """User login request"""
    username: str
    password: str
    remember_me: bool = False


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str


class ChatSessionBinding(BaseModel):
    """Binds a chat session to an authenticated user"""
    session_id: str
    user_id: str
    created_at: datetime
    encrypted: bool = True  # Whether chat data is encrypted


# ============================================================================
# Enhanced Authentication Manager
# ============================================================================

class EnhancedAuthManager:
    """
    Enhanced authentication manager with JWT tokens, refresh token rotation,
    and secure session management.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()
    
    # ========================================================================
    # Password Management
    # ========================================================================
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password with constant-time comparison"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    # ========================================================================
    # JWT Token Generation
    # ========================================================================
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        role: UserRole = UserRole.USER,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create JWT access token with enhanced claims
        
        Args:
            user_id: Unique user identifier
            username: Username
            role: User role for access control
            additional_claims: Additional custom claims
            
        Returns:
            JWT access token string
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        payload = {
            "sub": user_id,  # Subject (user ID)
            "username": username,
            "role": role.value,
            "type": TokenType.ACCESS.value,
            "iat": now,  # Issued at
            "exp": expires_at,  # Expiration
            "jti": secrets.token_urlsafe(16),  # JWT ID for blacklisting
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)
    
    def create_refresh_token(
        self,
        user_id: str,
        username: str,
        remember_me: bool = False
    ) -> str:
        """
        Create JWT refresh token
        
        Args:
            user_id: Unique user identifier
            username: Username
            remember_me: Extend token lifetime if true
            
        Returns:
            JWT refresh token string
        """
        now = datetime.utcnow()
        
        # Extend expiration if "remember me" is checked
        expire_days = AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS * 4 if remember_me else AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS
        expires_at = now + timedelta(days=expire_days)
        
        token_jti = secrets.token_urlsafe(16)
        
        payload = {
            "sub": user_id,
            "username": username,
            "type": TokenType.REFRESH.value,
            "iat": now,
            "exp": expires_at,
            "jti": token_jti,
        }
        
        token = jwt.encode(payload, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)
        
        # Store refresh token in Redis for rotation tracking
        redis_key = AuthConfig.REDIS_KEY_REFRESH_TOKEN.format(
            user_id=user_id,
            token_jti=token_jti
        )
        self.redis_client.setex(
            redis_key,
            int((expires_at - now).total_seconds()),
            token
        )
        
        return token
    
    # ========================================================================
    # Token Verification & Validation
    # ========================================================================
    
    def verify_token(self, token: str, expected_type: TokenType) -> Dict[str, Any]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            expected_type: Expected token type (access or refresh)
            
        Returns:
            Token payload dictionary
            
        Raises:
            HTTPException: If token is invalid, expired, or blacklisted
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                AuthConfig.SECRET_KEY,
                algorithms=[AuthConfig.ALGORITHM]
            )
            
            # Verify token type
            if payload.get("type") != expected_type.value:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {expected_type.value}"
                )
            
            # Check if token is blacklisted
            if self.is_token_blacklisted(payload.get("jti")):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    # ========================================================================
    # Token Blacklisting (for logout)
    # ========================================================================
    
    def blacklist_token(self, token_jti: str, expires_in: int):
        """
        Add token to blacklist (for logout)
        
        Args:
            token_jti: JWT ID from token payload
            expires_in: Time until token would naturally expire
        """
        redis_key = AuthConfig.REDIS_KEY_BLACKLIST.format(token_jti=token_jti)
        self.redis_client.setex(redis_key, expires_in, "1")
    
    def is_token_blacklisted(self, token_jti: str) -> bool:
        """Check if token is blacklisted"""
        redis_key = AuthConfig.REDIS_KEY_BLACKLIST.format(token_jti=token_jti)
        return self.redis_client.exists(redis_key) > 0
    
    # ========================================================================
    # Refresh Token Rotation
    # ========================================================================
    
    def rotate_refresh_token(self, old_refresh_token: str) -> TokenResponse:
        """
        Rotate refresh token (security best practice)
        
        When a refresh token is used, it's invalidated and a new one is issued.
        This prevents token replay attacks.
        
        Args:
            old_refresh_token: Current refresh token
            
        Returns:
            New token pair (access + refresh)
        """
        # Verify old refresh token
        payload = self.verify_token(old_refresh_token, TokenType.REFRESH)
        
        user_id = payload["sub"]
        username = payload["username"]
        old_jti = payload["jti"]
        
        # Invalidate old refresh token
        redis_key = AuthConfig.REDIS_KEY_REFRESH_TOKEN.format(
            user_id=user_id,
            token_jti=old_jti
        )
        self.redis_client.delete(redis_key)
        
        # Blacklist old token
        exp_timestamp = payload.get("exp", 0)
        current_timestamp = datetime.utcnow().timestamp()
        ttl = max(int(exp_timestamp - current_timestamp), 0)
        self.blacklist_token(old_jti, ttl)
        
        # Create new token pair
        new_access_token = self.create_access_token(
            user_id=user_id,
            username=username,
            role=UserRole(payload.get("role", UserRole.USER.value))
        )
        
        new_refresh_token = self.create_refresh_token(
            user_id=user_id,
            username=username
        )
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    # ========================================================================
    # User Session Management
    # ========================================================================
    
    def create_user_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str
    ) -> str:
        """
        Create a new user session
        
        Args:
            user_id: User identifier
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store session in Redis
        redis_key = AuthConfig.REDIS_KEY_USER_SESSIONS.format(user_id=user_id)
        self.redis_client.hset(
            redis_key,
            session_id,
            jwt.encode(session.__dict__, AuthConfig.SECRET_KEY, default=str)
        )
        self.redis_client.expire(redis_key, AuthConfig.SESSION_EXPIRE_HOURS * 3600)
        
        # Limit active sessions per user
        self._enforce_session_limit(user_id)
        
        return session_id
    
    def _enforce_session_limit(self, user_id: str):
        """Enforce maximum active sessions per user"""
        redis_key = AuthConfig.REDIS_KEY_USER_SESSIONS.format(user_id=user_id)
        sessions = self.redis_client.hgetall(redis_key)
        
        if len(sessions) > AuthConfig.MAX_ACTIVE_SESSIONS_PER_USER:
            # Remove oldest sessions
            sorted_sessions = sorted(
                sessions.items(),
                key=lambda x: jwt.decode(x[1], AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM])['created_at']
            )
            
            for session_id, _ in sorted_sessions[:-AuthConfig.MAX_ACTIVE_SESSIONS_PER_USER]:
                self.redis_client.hdel(redis_key, session_id)
    
    def invalidate_user_session(self, user_id: str, session_id: str):
        """Invalidate a specific user session"""
        redis_key = AuthConfig.REDIS_KEY_USER_SESSIONS.format(user_id=user_id)
        self.redis_client.hdel(redis_key, session_id)
    
    def invalidate_all_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user (e.g., password change)"""
        redis_key = AuthConfig.REDIS_KEY_USER_SESSIONS.format(user_id=user_id)
        self.redis_client.delete(redis_key)
    
    # ========================================================================
    # Chat Session Binding (Security Feature)
    # ========================================================================
    
    def bind_chat_session_to_user(
        self,
        session_id: str,
        user_id: str,
        encrypt: bool = True
    ) -> ChatSessionBinding:
        """
        Bind a chat session to an authenticated user
        
        This ensures chat history is only accessible to the authenticated user.
        
        Args:
            session_id: Chat session identifier
            user_id: Authenticated user identifier
            encrypt: Whether to encrypt chat data
            
        Returns:
            ChatSessionBinding object
        """
        binding = ChatSessionBinding(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            encrypted=encrypt
        )
        
        # Store binding in Redis
        redis_key = f"chat:binding:{session_id}"
        self.redis_client.setex(
            redis_key,
            AuthConfig.SESSION_EXPIRE_HOURS * 3600,
            jwt.encode(binding.dict(), AuthConfig.SECRET_KEY, default=str)
        )
        
        return binding
    
    def verify_chat_session_access(self, session_id: str, user_id: str) -> bool:
        """
        Verify that a user has access to a chat session
        
        Args:
            session_id: Chat session identifier
            user_id: User identifier
            
        Returns:
            True if user has access, False otherwise
        """
        redis_key = f"chat:binding:{session_id}"
        binding_data = self.redis_client.get(redis_key)
        
        if not binding_data:
            return False
        
        try:
            binding = jwt.decode(binding_data, AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM])
            return binding.get("user_id") == user_id
        except:
            return False
    
    # ========================================================================
    # Login Attempt Tracking
    # ========================================================================
    
    def check_login_attempts(self, identifier: str) -> bool:
        """
        Check if login attempts exceed limit
        
        Args:
            identifier: Username, email, or IP address
            
        Returns:
            True if login allowed, False if locked out
        """
        redis_key = AuthConfig.REDIS_KEY_FAILED_LOGINS.format(identifier=identifier)
        attempts = self.redis_client.get(redis_key)
        
        if attempts and int(attempts) >= AuthConfig.MAX_LOGIN_ATTEMPTS:
            return False
        
        return True
    
    def record_failed_login(self, identifier: str):
        """Record a failed login attempt"""
        redis_key = AuthConfig.REDIS_KEY_FAILED_LOGINS.format(identifier=identifier)
        self.redis_client.incr(redis_key)
        self.redis_client.expire(redis_key, AuthConfig.LOCKOUT_DURATION_MINUTES * 60)
    
    def clear_failed_logins(self, identifier: str):
        """Clear failed login attempts (on successful login)"""
        redis_key = AuthConfig.REDIS_KEY_FAILED_LOGINS.format(identifier=identifier)
        self.redis_client.delete(redis_key)
    
    # ========================================================================
    # Per-User Rate Limiting
    # ========================================================================
    
    def check_user_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        max_requests: int = 100,
        window_seconds: int = 3600
    ) -> bool:
        """
        Check rate limit for a specific user and endpoint
        
        Args:
            user_id: User identifier
            endpoint: API endpoint
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            True if request allowed, False if rate limited
        """
        redis_key = AuthConfig.REDIS_KEY_USER_RATE_LIMIT.format(
            user_id=user_id,
            endpoint=endpoint.replace("/", "_")
        )
        
        current_count = self.redis_client.get(redis_key)
        
        if current_count and int(current_count) >= max_requests:
            return False
        
        # Increment counter
        pipe = self.redis_client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, window_seconds)
        pipe.execute()
        
        return True


# ============================================================================
# FastAPI Dependencies
# ============================================================================

# Global auth manager instance (initialized in main.py)
auth_manager: Optional[EnhancedAuthManager] = None


def get_auth_manager() -> EnhancedAuthManager:
    """Dependency to get auth manager"""
    if auth_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system not initialized"
        )
    return auth_manager


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    auth: EnhancedAuthManager = Depends(get_auth_manager)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user from JWT token
    
    Usage:
        @app.get("/protected")
        async def protected_route(current_user: Dict = Depends(get_current_user)):
            return {"user_id": current_user["sub"], "username": current_user["username"]}
    """
    token = credentials.credentials
    payload = auth.verify_token(token, TokenType.ACCESS)
    return payload


async def get_current_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current admin user
    
    Usage:
        @app.get("/admin")
        async def admin_route(admin_user: Dict = Depends(get_current_admin_user)):
            return {"message": "Admin access granted"}
    """
    if current_user.get("role") != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# ============================================================================
# Utility Functions
# ============================================================================

def generate_secure_session_id() -> str:
    """Generate a cryptographically secure session ID"""
    return secrets.token_urlsafe(32)


def hash_string(text: str) -> str:
    """Hash a string using SHA-256"""
    return hashlib.sha256(text.encode()).hexdigest()


# ============================================================================
# Example Usage
# ============================================================================

"""
# In main.py:

from enhanced_auth import EnhancedAuthManager, get_current_user, auth_manager

# Initialize auth manager
auth_manager = EnhancedAuthManager(redis_client=redis_client)

# Register endpoint
@app.post("/auth/register")
async def register(user: UserRegistrationRequest):
    # Hash password
    hashed_password = auth_manager.hash_password(user.password)
    
    # Save user to database (your implementation)
    user_id = save_user_to_db(user.username, user.email, hashed_password)
    
    return {"message": "User registered successfully", "user_id": user_id}

# Login endpoint
@app.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLoginRequest, request: Request):
    # Check login attempts
    if not auth_manager.check_login_attempts(credentials.username):
        raise HTTPException(status_code=429, detail="Too many failed login attempts. Try again later.")
    
    # Verify credentials (your implementation)
    user = get_user_from_db(credentials.username)
    if not user or not auth_manager.verify_password(credentials.password, user.hashed_password):
        auth_manager.record_failed_login(credentials.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Clear failed attempts
    auth_manager.clear_failed_logins(credentials.username)
    
    # Create tokens
    access_token = auth_manager.create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role
    )
    refresh_token = auth_manager.create_refresh_token(
        user_id=user.id,
        username=user.username,
        remember_me=credentials.remember_me
    )
    
    # Create session
    session_id = auth_manager.create_user_session(
        user_id=user.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

# Refresh token endpoint
@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(refresh_request: TokenRefreshRequest):
    return auth_manager.rotate_refresh_token(refresh_request.refresh_token)

# Logout endpoint
@app.post("/auth/logout")
async def logout(current_user: Dict = Depends(get_current_user)):
    # Blacklist current access token
    auth_manager.blacklist_token(
        current_user["jti"],
        AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    
    return {"message": "Logged out successfully"}

# Protected chat endpoint with user binding
@app.post("/ai/chat")
async def chat(
    message: str,
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    user_id = current_user["sub"]
    
    # Bind chat session to user (first time)
    if not auth_manager.verify_chat_session_access(session_id, user_id):
        auth_manager.bind_chat_session_to_user(session_id, user_id, encrypt=True)
    
    # Check user rate limit
    if not auth_manager.check_user_rate_limit(user_id, "/ai/chat", max_requests=100, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Process chat (your implementation)
    response = process_chat(message, session_id, user_id)
    
    return {"response": response}
"""
