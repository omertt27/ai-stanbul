"""
FastAPI Dependencies Module

Centralized dependency injection for FastAPI endpoints
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session

from database import get_db


# Placeholder for authentication
async def get_current_user_optional(
    authorization: Optional[str] = Header(None)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, None otherwise
    Used for optional authentication
    """
    if not authorization:
        return None
    
    try:
        # Import here to avoid circular imports
        from enhanced_auth import get_current_user
        return await get_current_user(authorization)
    except:
        return None


async def get_current_user_required(
    authorization: str = Header(..., description="Bearer token")
) -> Dict[str, Any]:
    """
    Get current user, raise 401 if not authenticated
    Used for protected endpoints
    """
    try:
        from enhanced_auth import get_current_user
        return await get_current_user(authorization)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_db_session() -> Session:
    """Get database session"""
    return Depends(get_db)


# Language detection dependency
def get_language(
    accept_language: Optional[str] = Header(None),
    lang: Optional[str] = None
) -> str:
    """
    Detect user's preferred language
    Priority: query param > header > default
    """
    if lang:
        return lang.lower()
    
    if accept_language:
        # Parse Accept-Language header
        languages = accept_language.split(",")
        if languages:
            primary = languages[0].split(";")[0].strip()
            if "-" in primary:
                primary = primary.split("-")[0]
            return primary.lower()
    
    return "en"  # Default to English
