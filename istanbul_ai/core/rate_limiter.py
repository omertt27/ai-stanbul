"""
Rate Limiter for Istanbul AI
Token bucket rate limiting to protect against abuse
"""

from typing import Dict, Optional
import time
import logging
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Token bucket rate limiter
    Protects against abuse and ensures fair usage
    """
    
    def __init__(self):
        # Rate limits by user tier (TESTING: Limits disabled)
        self.rate_limits = {
            "free": {"requests": 100000, "window": 3600},      # 100k/hour (testing)
            "basic": {"requests": 100000, "window": 3600},     # 100k/hour (testing)
            "premium": {"requests": 100000, "window": 3600},   # 100k/hour (testing)
            "anonymous": {"requests": 100000, "window": 3600}  # 100k/hour (testing)
        }
        
        # Track requests
        self.request_buckets: Dict[str, list] = defaultdict(list)
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("✅ RateLimiter initialized")
    
    async def start_cleanup(self):
        """Start background cleanup task"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self):
        """Periodically cleanup old request records"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                current_time = time.time()
                
                # Remove requests older than 1 hour
                for user_id in list(self.request_buckets.keys()):
                    self.request_buckets[user_id] = [
                        ts for ts in self.request_buckets[user_id]
                        if current_time - ts < 3600
                    ]
                    
                    # Remove empty buckets
                    if not self.request_buckets[user_id]:
                        del self.request_buckets[user_id]
                
                logger.debug(f"✅ Cleaned up rate limit buckets")
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def check_rate_limit(self, user_id: str, 
                              user_tier: str = "free") -> tuple[bool, dict]:
        """
        Check if user is within rate limits
        Returns: (allowed, info)
        """
        current_time = time.time()
        
        # Get rate limit for user tier
        limit_config = self.rate_limits.get(user_tier, self.rate_limits["free"])
        max_requests = limit_config["requests"]
        window = limit_config["window"]
        
        # Get user's request history
        user_requests = self.request_buckets[user_id]
        
        # Remove old requests outside the window
        cutoff_time = current_time - window
        user_requests = [ts for ts in user_requests if ts > cutoff_time]
        self.request_buckets[user_id] = user_requests
        
        # Check if within limit
        if len(user_requests) >= max_requests:
            oldest_request = min(user_requests)
            retry_after = int(oldest_request + window - current_time)
            
            info = {
                "allowed": False,
                "limit": max_requests,
                "remaining": 0,
                "reset": retry_after,
                "retry_after": retry_after
            }
            
            logger.warning(f"⚠️ Rate limit exceeded for {user_id} ({user_tier})")
            return False, info
        
        # Add current request
        user_requests.append(current_time)
        
        info = {
            "allowed": True,
            "limit": max_requests,
            "remaining": max_requests - len(user_requests),
            "reset": int(window),
            "retry_after": 0
        }
        
        return True, info
    
    async def get_user_stats(self, user_id: str) -> dict:
        """Get rate limit stats for user"""
        current_time = time.time()
        user_requests = self.request_buckets.get(user_id, [])
        
        # Count requests in last hour
        hour_ago = current_time - 3600
        requests_last_hour = len([ts for ts in user_requests if ts > hour_ago])
        
        # Count requests in last minute
        minute_ago = current_time - 60
        requests_last_minute = len([ts for ts in user_requests if ts > minute_ago])
        
        return {
            "user_id": user_id,
            "requests_last_hour": requests_last_hour,
            "requests_last_minute": requests_last_minute,
            "total_tracked": len(user_requests)
        }


# Global rate limiter
_rate_limiter: Optional[RateLimiter] = None

async def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        await _rate_limiter.start_cleanup()
    return _rate_limiter
