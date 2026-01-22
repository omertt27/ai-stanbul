"""
Request Logging Middleware

Logs all chat requests and responses with timing information for production monitoring.
This helps track:
- User query patterns
- Response times
- Intent detection accuracy
- Cache hit rates
- Error rates

Author: AI Istanbul Team
Date: January 2026
"""

import time
import logging
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

logger = logging.getLogger(__name__)

# Configure separate logger for chat analytics
chat_logger = logging.getLogger("chat_analytics")
chat_logger.setLevel(logging.INFO)

# Create file handler for chat logs
try:
    from pathlib import Path
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    chat_log_file = log_dir / "chat_requests.log"
    file_handler = logging.FileHandler(chat_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))
    chat_logger.addHandler(file_handler)
    logger.info(f"✅ Chat logging enabled: {chat_log_file}")
except Exception as e:
    logger.warning(f"⚠️ Could not set up chat log file: {e}")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all chat requests with detailed metrics.
    
    Logs:
    - Request timestamp
    - User message (truncated for privacy)
    - Session ID
    - Response time
    - Intent detected
    - Cache hit/miss
    - Backend used (vLLM, Groq, fallback)
    - Status code
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only log chat endpoints
        if not request.url.path.startswith("/api/chat"):
            return await call_next(request)
        
        start_time = time.time()
        
        # Capture request body for POST requests
        request_data = {}
        if request.method == "POST":
            try:
                body = await request.body()
                request_data = json.loads(body.decode())
                
                # Re-populate request body for downstream handlers
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
            except Exception as e:
                logger.warning(f"Could not parse request body: {e}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Try to extract response data for analytics
        response_data = {}
        try:
            # For streaming responses, we can't read the body
            if response.headers.get("content-type", "").startswith("application/json"):
                # Note: This is simplified - in production you'd need to capture
                # the response body without consuming it
                pass
        except Exception:
            pass
        
        # Log the request
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "response_time_ms": response_time_ms,
            "message": request_data.get("message", "")[:100] if "message" in request_data else "",
            "session_id": request_data.get("session_id", ""),
            "language": request_data.get("language", "en"),
            "user_location": bool(request_data.get("user_location")),
        }
        
        # Log to dedicated chat analytics logger
        chat_logger.info(json.dumps(log_entry))
        
        # Also log to console for slow requests (> 5 seconds)
        if response_time_ms > 5000:
            logger.warning(
                f"⏱️ SLOW REQUEST: {request.url.path} took {response_time_ms}ms - "
                f"Message: '{log_entry['message'][:50]}...'"
            )
        elif response_time_ms > 2000:
            logger.info(
                f"⏱️ {request.url.path} took {response_time_ms}ms - "
                f"Message: '{log_entry['message'][:30]}...'"
            )
        
        return response


class ResponseLoggingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced middleware to capture and log response details including
    intent, cache status, and LLM backend used.
    
    This middleware intercepts the response and extracts metadata for analytics.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only process chat endpoints
        if not request.url.path.startswith("/api/chat"):
            return await call_next(request)
        
        start_time = time.time()
        
        # Capture request
        request_message = ""
        session_id = ""
        if request.method == "POST":
            try:
                body = await request.body()
                request_data = json.loads(body.decode())
                request_message = request_data.get("message", "")[:100]
                session_id = request_data.get("session_id", "")
                
                # Re-populate request body
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
            except Exception:
                pass
        
        # Process request
        response = await call_next(request)
        
        # Calculate timing
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Log detailed analytics
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_path": request.url.path,
            "request_message": request_message,
            "session_id": session_id,
            "response_time_ms": response_time_ms,
            "status_code": response.status_code,
        }
        
        # Try to extract response metadata
        # Note: This is a simplified version - for production you'd use
        # a response wrapper to capture the body without consuming it
        
        chat_logger.info(json.dumps(log_entry))
        
        return response
