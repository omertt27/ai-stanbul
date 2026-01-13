"""
Rate limiting middleware for the Istanbul AI chatbot API.
Implements user-based and endpoint-based rate limiting with Redis backend.
"""

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import time
import json
import logging

from fastapi import HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    print("[WARNING] Redis not available for rate limiting, using in-memory storage")

logger = logging.getLogger(__name__)

class RateLimitManager:
    """
    Advanced rate limiting with Redis backend and fallback to memory storage.
    Supports per-user, per-endpoint, and global rate limits.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.memory_storage: Dict[str, Dict] = {}
        self.cleanup_interval = 3600  # Clean memory storage every hour
        self.last_cleanup = time.time()
        
        # Initialize Redis if available
        if REDIS_AVAILABLE and redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("✅ Redis connected for rate limiting")
            except Exception as e:
                logger.warning(f"❌ Redis connection failed: {e}, using memory storage")
                self.redis_client = None
        
        # Rate limit configurations
        self.rate_limits = {
            "ai_chat": {
                "requests": 30,      # 30 requests
                "window": 60,        # per minute
                "burst": 50          # burst limit
            },
            "ai_stream": {
                "requests": 10,      # 10 requests
                "window": 60,        # per minute
                "burst": 15          # burst limit
            },
            "image_analysis": {
                "requests": 5,       # 5 requests
                "window": 60,        # per minute
                "burst": 8           # burst limit
            },
            "global": {
                "requests": 100,     # 100 requests
                "window": 60,        # per minute
                "burst": 150         # burst limit
            }
        }
    
    def _get_storage_key(self, identifier: str, endpoint: str) -> str:
        """Generate storage key for rate limit tracking."""
        return f"rate_limit:{identifier}:{endpoint}"
    
    def _get_current_window(self, window_seconds: int) -> int:
        """Get current time window for rate limiting."""
        return int(time.time()) // window_seconds
    
    def _cleanup_memory_storage(self):
        """Clean up expired entries from memory storage."""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return
        
        current_time = time.time()
        expired_keys = []
        
        for key, data in self.memory_storage.items():
            if current_time - data.get("last_updated", 0) > 3600:  # 1 hour expiry
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_storage[key]
        
        self.last_cleanup = current_time
        logger.debug(f"Cleaned {len(expired_keys)} expired rate limit entries")
    
    def _store_data(self, key: str, data: Dict, ttl: int = 3600):
        """Store data in Redis or memory."""
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(data))
                return True
            except Exception as e:
                logger.warning(f"Redis storage failed: {e}, falling back to memory")
        
        # Fallback to memory storage
        self._cleanup_memory_storage()
        data["last_updated"] = time.time()
        self.memory_storage[key] = data
        return True
    
    def _get_data(self, key: str) -> Optional[Dict]:
        """Get data from Redis or memory."""
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(str(data))
                return None
            except Exception as e:
                logger.warning(f"Redis retrieval failed: {e}, falling back to memory")
        
        # Fallback to memory storage
        self._cleanup_memory_storage()
        return self.memory_storage.get(key)
    
    def check_rate_limit(self, identifier: str, endpoint: str) -> Dict[str, Any]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: User identifier (IP, user_id, etc.)
            endpoint: API endpoint name
            
        Returns:
            Dictionary with rate limit status and metadata
        """
        # Get rate limit config for endpoint
        config = self.rate_limits.get(endpoint, self.rate_limits["global"])
        
        window_seconds = config["window"]
        max_requests = config["requests"]
        burst_limit = config["burst"]
        
        current_window = self._get_current_window(window_seconds)
        storage_key = self._get_storage_key(identifier, endpoint)
        
        # Get current usage data
        usage_data = self._get_data(storage_key) or {
            "window": current_window,
            "count": 0,
            "burst_count": 0,
            "first_request": time.time()
        }
        
        # Reset if new window
        if usage_data["window"] != current_window:
            usage_data = {
                "window": current_window,
                "count": 1,
                "burst_count": 1,
                "first_request": time.time()
            }
        else:
            usage_data["count"] += 1
            usage_data["burst_count"] += 1
        
        # Check burst limit (short-term)
        time_since_first = time.time() - usage_data["first_request"]
        if time_since_first < 10 and usage_data["burst_count"] > burst_limit:
            return {
                "allowed": False,
                "reason": "burst_limit_exceeded",
                "retry_after": 10 - time_since_first,
                "limit": burst_limit,
                "used": usage_data["burst_count"],
                "reset_time": usage_data["first_request"] + 10
            }
        
        # Reset burst counter if enough time has passed
        if time_since_first >= 10:
            usage_data["burst_count"] = 1
            usage_data["first_request"] = time.time()
        
        # Check regular rate limit
        if usage_data["count"] > max_requests:
            next_window_start = (current_window + 1) * window_seconds
            return {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "retry_after": next_window_start - time.time(),
                "limit": max_requests,
                "used": usage_data["count"],
                "reset_time": next_window_start
            }
        
        # Store updated usage data
        self._store_data(storage_key, usage_data, window_seconds + 60)
        
        return {
            "allowed": True,
            "limit": max_requests,
            "used": usage_data["count"],
            "remaining": max_requests - usage_data["count"],
            "reset_time": (current_window + 1) * window_seconds
        }
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get current rate limiting statistics."""
        stats = {
            "storage_backend": "redis" if self.redis_client else "memory",
            "memory_entries": len(self.memory_storage),
            "rate_limits": self.rate_limits,
            "redis_connected": bool(self.redis_client)
        }
        
        if self.redis_client:
            try:
                # Just check if redis is working, don't try to get detailed info
                self.redis_client.ping()
                stats["redis_status"] = "connected"
            except Exception as e:
                stats["redis_error"] = str(e)
        
        return stats

# Global rate limiter instance
rate_limit_manager = None

def get_rate_limiter() -> RateLimitManager:
    """Get or create the global rate limiter instance."""
    global rate_limit_manager
    if rate_limit_manager is None:
        # Try to get Redis URL from environment
        import os
        redis_url = os.getenv("REDIS_URL")
        rate_limit_manager = RateLimitManager(redis_url)
    return rate_limit_manager

def create_slowapi_limiter():
    """Create SlowAPI limiter for FastAPI integration."""
    def get_user_id(request: Request):
        """Extract user identifier for rate limiting."""
        # Try to get user ID from headers, session, or fall back to IP
        user_id = request.headers.get("X-User-ID") or \
                 request.headers.get("X-Session-ID") or \
                 get_remote_address(request)
        return user_id
    
    # Create limiter with Redis backend if available
    redis_url = os.getenv("REDIS_URL") if REDIS_AVAILABLE else None
    
    if redis_url:
        try:
            import redis
            # Add connection timeout to prevent hanging in Cloud Run
            limiter = Limiter(
                key_func=get_user_id,
                storage_uri=redis_url,
                storage_options={
                    "socket_connect_timeout": 2,
                    "socket_timeout": 2
                }
            )
            logger.info("✅ SlowAPI limiter created with Redis backend")
        except Exception as e:
            logger.warning(f"Redis limiter failed: {e}, using memory backend")
            limiter = Limiter(
                key_func=get_user_id
            )
    else:
        limiter = Limiter(
            key_func=get_user_id
        )
    
    return limiter

# Create the limiter instance
import os
limiter = create_slowapi_limiter()

# Exception handler for rate limit exceeded
async def rate_limit_handler(request: Request, exc: Exception):
    """Custom handler for rate limit exceeded errors."""
    from fastapi.responses import JSONResponse
    
    # Check if this is actually a RateLimitExceeded error
    if hasattr(exc, 'detail'):
        detail = str(getattr(exc, 'detail'))
        retry_after = getattr(exc, 'retry_after', 60)
    else:
        detail = "Rate limit exceeded"
        retry_after = 60
    
    response_data = {
        "error": "rate_limit_exceeded",
        "message": f"Rate limit exceeded: {detail}",
        "retry_after": retry_after,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Log rate limit hit
    user_id = request.headers.get("X-User-ID") or \
             request.headers.get("X-Session-ID") or \
             get_remote_address(request)
    
    logger.warning(f"Rate limit exceeded for user {user_id} on {request.url.path}")
    
    return JSONResponse(
        status_code=429,
        content=response_data,
        headers={"Retry-After": str(getattr(exc, 'retry_after', 60))}
    )
