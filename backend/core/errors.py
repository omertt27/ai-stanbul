"""
Standardized Error Handling Module

Provides consistent error responses across all API endpoints:
- Custom exception classes
- Global exception handlers
- Standardized error format: {error, code, details}

Author: AI Istanbul Team
Date: December 2024
"""

import logging
import traceback
from typing import Optional, Dict, Any, Union
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from config.settings import settings

logger = logging.getLogger(__name__)


# ===========================================
# Error Codes
# ===========================================

class ErrorCode:
    """Standardized error codes"""
    # Authentication errors (1xxx)
    AUTH_REQUIRED = "AUTH_1001"
    AUTH_INVALID_TOKEN = "AUTH_1002"
    AUTH_EXPIRED_TOKEN = "AUTH_1003"
    AUTH_INVALID_CREDENTIALS = "AUTH_1004"
    
    # Authorization errors (2xxx)
    FORBIDDEN = "AUTHZ_2001"
    ROLE_REQUIRED = "AUTHZ_2002"
    ADMIN_REQUIRED = "AUTHZ_2003"
    
    # Validation errors (3xxx)
    VALIDATION_ERROR = "VAL_3001"
    INVALID_INPUT = "VAL_3002"
    MISSING_FIELD = "VAL_3003"
    INVALID_FORMAT = "VAL_3004"
    MESSAGE_TOO_LONG = "VAL_3005"
    MESSAGE_TOO_SHORT = "VAL_3006"
    
    # Resource errors (4xxx)
    NOT_FOUND = "RES_4001"
    ALREADY_EXISTS = "RES_4002"
    RESOURCE_EXHAUSTED = "RES_4003"
    
    # Rate limiting errors (5xxx)
    RATE_LIMITED = "RATE_5001"
    QUOTA_EXCEEDED = "RATE_5002"
    
    # Server errors (6xxx)
    INTERNAL_ERROR = "SRV_6001"
    SERVICE_UNAVAILABLE = "SRV_6002"
    EXTERNAL_API_ERROR = "SRV_6003"
    DATABASE_ERROR = "SRV_6004"
    LLM_ERROR = "SRV_6005"
    
    # Chat-specific errors (7xxx)
    CHAT_SESSION_NOT_FOUND = "CHAT_7001"
    CHAT_CONTEXT_ERROR = "CHAT_7002"
    CHAT_GENERATION_ERROR = "CHAT_7003"


# ===========================================
# Custom Exceptions
# ===========================================

class APIError(Exception):
    """Base API exception with standardized format"""
    
    def __init__(
        self,
        message: str,
        code: str = ErrorCode.INTERNAL_ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to standardized error response"""
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details
        }


class AuthenticationError(APIError):
    """Authentication failed"""
    def __init__(self, message: str = "Authentication required", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code=ErrorCode.AUTH_REQUIRED,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class AuthorizationError(APIError):
    """Authorization/permission denied"""
    def __init__(self, message: str = "Access denied", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code=ErrorCode.FORBIDDEN,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class ValidationError(APIError):
    """Request validation failed"""
    def __init__(self, message: str = "Validation failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class NotFoundError(APIError):
    """Resource not found"""
    def __init__(self, message: str = "Resource not found", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code=ErrorCode.NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class RateLimitError(APIError):
    """Rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code=ErrorCode.RATE_LIMITED,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


class ExternalServiceError(APIError):
    """External service (LLM, Google API, etc.) failed"""
    def __init__(self, message: str = "External service error", service: str = "unknown", details: Optional[Dict] = None):
        details = details or {}
        details["service"] = service
        super().__init__(
            message=message,
            code=ErrorCode.EXTERNAL_API_ERROR,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details
        )


class LLMError(APIError):
    """LLM service error"""
    def __init__(self, message: str = "LLM service error", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code=ErrorCode.LLM_ERROR,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


class ChatError(APIError):
    """Chat-specific error"""
    def __init__(self, message: str = "Chat error", code: str = ErrorCode.CHAT_GENERATION_ERROR, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


# ===========================================
# Error Response Builder
# ===========================================

class ErrorResponse:
    """Builder for standardized error responses"""
    
    @staticmethod
    def build(
        error: str,
        code: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        include_timestamp: bool = True
    ) -> Dict[str, Any]:
        """Build a standardized error response"""
        response = {
            "error": error,
            "code": code,
            "details": details or {}
        }
        
        if include_timestamp:
            response["timestamp"] = datetime.utcnow().isoformat()
        
        if request_id:
            response["request_id"] = request_id
        
        return response
    
    @staticmethod
    def from_exception(exc: Exception, request: Optional[Request] = None) -> Dict[str, Any]:
        """Build error response from any exception"""
        request_id = None
        if request:
            request_id = getattr(request.state, 'request_id', None)
        
        if isinstance(exc, APIError):
            return ErrorResponse.build(
                error=exc.message,
                code=exc.code,
                details=exc.details,
                request_id=request_id
            )
        
        # Generic exception
        return ErrorResponse.build(
            error=str(exc) if not settings.is_production() else "Internal server error",
            code=ErrorCode.INTERNAL_ERROR,
            details={"type": type(exc).__name__} if not settings.is_production() else None,
            request_id=request_id
        )


# ===========================================
# Exception Handlers
# ===========================================

async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors"""
    logger.warning(f"API Error: {exc.code} - {exc.message}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse.from_exception(exc, request)
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions with standardized format"""
    
    # Map status codes to error codes
    code_map = {
        400: ErrorCode.INVALID_INPUT,
        401: ErrorCode.AUTH_REQUIRED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        429: ErrorCode.RATE_LIMITED,
        500: ErrorCode.INTERNAL_ERROR,
        502: ErrorCode.EXTERNAL_API_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
    }
    
    error_code = code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
    
    # Handle detail as dict or string
    details = {}
    error_message = "An error occurred"
    
    if isinstance(exc.detail, dict):
        error_message = exc.detail.get("error", exc.detail.get("message", str(exc.detail)))
        details = exc.detail.get("details", {})
        if "code" in exc.detail:
            error_code = exc.detail["code"]
    elif isinstance(exc.detail, str):
        error_message = exc.detail
    
    logger.warning(f"HTTP {exc.status_code}: {error_message}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse.build(
            error=error_message,
            code=error_code,
            details=details
        )
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors with detailed field information"""
    
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(f"Validation error: {errors}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse.build(
            error="Request validation failed",
            code=ErrorCode.VALIDATION_ERROR,
            details={
                "validation_errors": errors,
                "error_count": len(errors)
            }
        )
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    
    # Log full traceback in non-production
    if not settings.is_production():
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
    else:
        logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse.build(
            error="Internal server error" if settings.is_production() else str(exc),
            code=ErrorCode.INTERNAL_ERROR,
            details={
                "type": type(exc).__name__,
                "traceback": traceback.format_exc().split("\n")[-5:]
            } if not settings.is_production() else None
        )
    )


# ===========================================
# Setup Function
# ===========================================

def setup_error_handlers(app: FastAPI):
    """Register all error handlers on the FastAPI app"""
    
    # Custom API errors
    app.add_exception_handler(APIError, api_error_handler)
    
    # HTTP exceptions (404, 401, etc.)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Pydantic validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Catch-all for unexpected errors
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("✅ Standardized error handlers configured")
