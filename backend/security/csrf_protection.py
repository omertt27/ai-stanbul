"""
CSRF Protection Module for AI Istanbul Backend
Provides Cross-Site Request Forgery protection for state-changing operations
"""

import secrets
import time
from typing import Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

# In-memory CSRF token store (in production, use Redis or database)
csrf_tokens: Dict[str, float] = {}

# CSRF token configuration
CSRF_TOKEN_LIFETIME = 3600  # 1 hour in seconds
CSRF_HEADER_NAME = "X-CSRF-Token"
CSRF_COOKIE_NAME = "csrf_token"

def generate_csrf_token() -> str:
    """Generate a secure CSRF token"""
    token = secrets.token_urlsafe(32)
    csrf_tokens[token] = time.time()
    return token

def cleanup_expired_tokens():
    """Remove expired CSRF tokens"""
    current_time = time.time()
    expired_tokens = [
        token for token, created_time in csrf_tokens.items()
        if current_time - created_time > CSRF_TOKEN_LIFETIME
    ]
    for token in expired_tokens:
        del csrf_tokens[token]

def validate_csrf_token(request: Request) -> bool:
    """Validate CSRF token from request header"""
    cleanup_expired_tokens()
    
    # Get token from header
    token = request.headers.get(CSRF_HEADER_NAME)
    if not token:
        return False
    
    # Check if token exists and is valid
    if token not in csrf_tokens:
        return False
    
    # Check if token is expired
    created_time = csrf_tokens[token]
    if time.time() - created_time > CSRF_TOKEN_LIFETIME:
        del csrf_tokens[token]
        return False
    
    return True

def require_csrf_token(request: Request):
    """Dependency to require valid CSRF token for state-changing operations"""
    if not validate_csrf_token(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing CSRF token"
        )

def csrf_token_endpoint():
    """Endpoint to provide CSRF token to authenticated users"""
    token = generate_csrf_token()
    response = JSONResponse({
        "csrf_token": token,
        "expires_in": CSRF_TOKEN_LIFETIME
    })
    
    # Set secure cookie (optional - can also use header-only approach)
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        max_age=CSRF_TOKEN_LIFETIME,
        httponly=True,
        secure=True,  # HTTPS only
        samesite="strict"
    )
    
    return response

# Example usage in main.py:
"""
from backend.security.csrf_protection import require_csrf_token, csrf_token_endpoint

# Add CSRF token endpoint
@app.get("/api/csrf-token")
async def get_csrf_token(current_admin=Depends(get_current_admin)):
    return csrf_token_endpoint()

# Protect state-changing endpoints
@app.post("/api/admin/some-action")
async def protected_action(
    request: Request,
    csrf_check=Depends(require_csrf_token),
    current_admin=Depends(get_current_admin)
):
    # Your protected action here
    pass
"""
