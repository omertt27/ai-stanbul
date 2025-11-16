"""
Authentication Endpoints Module

User registration, login, logout, and token management
"""

from fastapi import APIRouter, HTTPException, Depends, Body, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import os

from database import get_db
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# Request/Response Models
class UserRegistrationRequest(BaseModel):
    email: str
    password: str
    username: str
    full_name: Optional[str] = None


class UserLoginRequest(BaseModel):
    email: str
    password: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    username: str


@router.post("/register", response_model=TokenResponse)
async def register_user(
    request: UserRegistrationRequest,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    try:
        from enhanced_auth import EnhancedAuthManager
        auth_manager = EnhancedAuthManager()
        
        result = await auth_manager.register_user(
            email=request.email,
            password=request.password,
            username=request.username,
            full_name=request.full_name,
            db=db
        )
        
        logger.info(f"✅ User registered: {request.email}")
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    request: UserLoginRequest,
    db: Session = Depends(get_db)
):
    """Login with email and password"""
    try:
        from enhanced_auth import EnhancedAuthManager
        auth_manager = EnhancedAuthManager()
        
        result = await auth_manager.login_user(
            email=request.email,
            password=request.password,
            db=db
        )
        
        logger.info(f"✅ User logged in: {request.email}")
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/admin-login")
async def admin_login(
    username: str = Body(...),
    password: str = Body(...)
):
    """Admin login for dashboard access"""
    import jwt
    import bcrypt
    from datetime import datetime, timedelta
    
    if username != settings.ADMIN_USERNAME:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    if not settings.ADMIN_PASSWORD_HASH:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin authentication not configured"
        )
    
    try:
        if not bcrypt.checkpw(
            password.encode('utf-8'),
            settings.ADMIN_PASSWORD_HASH.encode('utf-8')
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Generate JWT token
    token_data = {
        "username": username,
        "role": "admin",
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    
    token = jwt.encode(
        token_data,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    logger.info(f"✅ Admin logged in: {username}")
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": username,
            "role": "admin"
        }
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: TokenRefreshRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token"""
    try:
        from enhanced_auth import EnhancedAuthManager
        auth_manager = EnhancedAuthManager()
        
        result = await auth_manager.refresh_token(
            refresh_token=request.refresh_token,
            db=db
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(
    current_user: Dict = Depends(lambda: {"id": 1}),
    db: Session = Depends(get_db)
):
    """Logout user"""
    logger.info(f"✅ User logged out: {current_user.get('id')}")
    return {"message": "Logged out successfully"}


@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    current_user: Dict = Depends(lambda: {"id": 1, "email": "user@example.com", "username": "user"}),
    db: Session = Depends(get_db)
):
    """Get current user profile"""
    return current_user
