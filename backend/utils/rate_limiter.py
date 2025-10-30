"""
Rate Limiter - Sliding window rate limiter with per-service limits
Protects APIs from abuse and quota exhaustion
"""

from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, message: str, wait_time: float, service: str):
        self.message = message
        self.wait_time = wait_time
        self.service = service
        super().__init__(self.message)


class RateLimiter:
    """
    Sliding window rate limiter with per-service and per-user limits.
    
    Features:
    - Per-service rate limits (OpenAI, Google Places, etc.)
    - Per-user rate limits
    - Sliding window algorithm (more accurate than fixed window)
    - Thread-safe operations
    - Detailed statistics
    
    Usage:
        limiter = RateLimiter()
        
        if limiter.is_allowed("openai", user_id):
            # Make API call
            response = call_openai_api()
        else:
            wait_time = limiter.wait_time("openai", user_id)
            raise RateLimitExceeded(f"Wait {wait_time:.1f}s")
    """
    
    def __init__(self):
        """Initialize rate limiter with default limits"""
        # Service limits: (max_requests, window_seconds)
        self.service_limits: Dict[str, Tuple[int, int]] = {
            "openai": (50, 60),  # 50 requests per minute
            "openai_gpt4": (20, 60),  # 20 GPT-4 requests per minute (more expensive)
            "openai_gpt35": (50, 60),  # 50 GPT-3.5 requests per minute
            "google_places": (100, 60),  # 100 requests per minute
            "google_maps": (100, 60),
            "ibb_api": (200, 60),  # 200 requests per minute (local API)
            "user_global": (30, 60),  # 30 requests per minute per user (global)
            "user_chat": (20, 60),  # 20 chat messages per minute per user
        }
        
        # Request history: {service:identifier: [timestamps]}
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "blocks_by_service": defaultdict(int)
        }
        
        # Thread safety
        self.lock = Lock()
        
        logger.info("🚦 Rate Limiter initialized")
        logger.info(f"   Service limits: {len(self.service_limits)} services")
    
    def set_limit(self, service: str, max_requests: int, window_seconds: int) -> None:
        """
        Set or update rate limit for a service
        
        Args:
            service: Service name
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        """
        with self.lock:
            self.service_limits[service] = (max_requests, window_seconds)
            logger.info(f"🚦 Updated limit for {service}: {max_requests} requests / {window_seconds}s")
    
    def is_allowed(self, service: str, identifier: str) -> bool:
        """
        Check if request is allowed under rate limit
        
        Args:
            service: Service name (e.g., "openai", "google_places")
            identifier: Unique identifier (e.g., user_id, ip_address)
            
        Returns:
            True if request is allowed, False otherwise
        """
        with self.lock:
            self.stats["total_requests"] += 1
            
            key = f"{service}:{identifier}"
            limit, window = self.service_limits.get(service, (100, 60))
            
            now = datetime.now()
            cutoff = now - timedelta(seconds=window)
            
            # Remove old requests outside the window
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > cutoff
            ]
            
            # Check if limit exceeded
            if len(self.requests[key]) >= limit:
                self.stats["blocked_requests"] += 1
                self.stats["blocks_by_service"][service] += 1
                logger.warning(f"🚫 Rate limit exceeded: {service} for {identifier} ({len(self.requests[key])}/{limit})")
                return False
            
            # Allow request and record timestamp
            self.requests[key].append(now)
            self.stats["allowed_requests"] += 1
            logger.debug(f"✅ Rate limit OK: {service} for {identifier} ({len(self.requests[key])}/{limit})")
            return True
    
    def wait_time(self, service: str, identifier: str) -> float:
        """
        Get wait time in seconds until next request is allowed
        
        Args:
            service: Service name
            identifier: Unique identifier
            
        Returns:
            Wait time in seconds (0.0 if request would be allowed now)
        """
        with self.lock:
            key = f"{service}:{identifier}"
            limit, window = self.service_limits.get(service, (100, 60))
            
            if not self.requests[key]:
                return 0.0
            
            now = datetime.now()
            cutoff = now - timedelta(seconds=window)
            
            # Get requests in current window
            recent_requests = [r for r in self.requests[key] if r > cutoff]
            
            if len(recent_requests) < limit:
                return 0.0
            
            # Calculate when oldest request will fall outside window
            oldest_request = min(recent_requests)
            time_until_reset = window - (now - oldest_request).total_seconds()
            
            return max(0.0, time_until_reset)
    
    def get_current_usage(self, service: str, identifier: str) -> Dict[str, any]:
        """
        Get current rate limit usage for a service/identifier
        
        Args:
            service: Service name
            identifier: Unique identifier
            
        Returns:
            Dictionary with usage information
        """
        with self.lock:
            key = f"{service}:{identifier}"
            limit, window = self.service_limits.get(service, (100, 60))
            
            now = datetime.now()
            cutoff = now - timedelta(seconds=window)
            
            # Get requests in current window
            recent_requests = [r for r in self.requests[key] if r > cutoff]
            current_count = len(recent_requests)
            
            return {
                "service": service,
                "identifier": identifier,
                "current_count": current_count,
                "limit": limit,
                "window_seconds": window,
                "remaining": max(0, limit - current_count),
                "usage_percent": (current_count / limit * 100) if limit > 0 else 0,
                "will_reset_in": self.wait_time(service, identifier) if current_count >= limit else None
            }
    
    def reset(self, service: str = None, identifier: str = None) -> None:
        """
        Reset rate limit tracking
        
        Args:
            service: Service name (optional, resets all if None)
            identifier: Unique identifier (optional)
        """
        with self.lock:
            if service is None and identifier is None:
                # Reset everything
                self.requests.clear()
                logger.info("🔄 Reset all rate limits")
            elif service and identifier:
                # Reset specific service+identifier
                key = f"{service}:{identifier}"
                if key in self.requests:
                    del self.requests[key]
                    logger.info(f"🔄 Reset rate limit: {service} for {identifier}")
            elif service:
                # Reset all for service
                keys_to_delete = [k for k in self.requests.keys() if k.startswith(f"{service}:")]
                for key in keys_to_delete:
                    del self.requests[key]
                logger.info(f"🔄 Reset rate limits for service: {service}")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get rate limiter statistics
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            total = self.stats["total_requests"]
            allowed = self.stats["allowed_requests"]
            blocked = self.stats["blocked_requests"]
            
            block_rate = (blocked / total * 100) if total > 0 else 0
            
            return {
                "total_requests": total,
                "allowed_requests": allowed,
                "blocked_requests": blocked,
                "block_rate": f"{block_rate:.1f}%",
                "blocks_by_service": dict(self.stats["blocks_by_service"]),
                "active_trackers": len(self.requests),
                "configured_services": list(self.service_limits.keys())
            }
    
    def check_and_raise(self, service: str, identifier: str) -> None:
        """
        Check rate limit and raise exception if exceeded
        
        Args:
            service: Service name
            identifier: Unique identifier
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        if not self.is_allowed(service, identifier):
            wait = self.wait_time(service, identifier)
            raise RateLimitExceeded(
                f"Rate limit exceeded for {service}. Try again in {wait:.1f} seconds.",
                wait_time=wait,
                service=service
            )


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("🧪 Testing RateLimiter...")
    
    # Test 1: Basic rate limiting
    print("\n📝 Test 1: Basic rate limiting")
    limiter = RateLimiter()
    limiter.set_limit("test_service", 3, 5)  # 3 requests per 5 seconds
    
    user_id = "test_user_123"
    
    # First 3 requests should be allowed
    for i in range(3):
        assert limiter.is_allowed("test_service", user_id), f"Request {i+1} should be allowed"
        print(f"✅ Request {i+1} allowed")
    
    # 4th request should be blocked
    assert not limiter.is_allowed("test_service", user_id), "4th request should be blocked"
    print("✅ 4th request blocked as expected")
    
    # Test 2: Wait time calculation
    print("\n📝 Test 2: Wait time")
    wait = limiter.wait_time("test_service", user_id)
    print(f"⏱️ Need to wait: {wait:.1f} seconds")
    assert wait > 0 and wait <= 5, "Wait time should be between 0 and 5 seconds"
    print("✅ Test 2 passed")
    
    # Test 3: Usage tracking
    print("\n📝 Test 3: Usage tracking")
    usage = limiter.get_current_usage("test_service", user_id)
    print(f"📊 Current usage: {usage}")
    assert usage['current_count'] == 3
    assert usage['limit'] == 3
    assert usage['remaining'] == 0
    print("✅ Test 3 passed")
    
    # Test 4: Window expiration
    print("\n📝 Test 4: Window expiration (waiting 5 seconds...)")
    time.sleep(5.5)
    assert limiter.is_allowed("test_service", user_id), "Request should be allowed after window expires"
    print("✅ Test 4 passed")
    
    # Test 5: Statistics
    print("\n📝 Test 5: Statistics")
    stats = limiter.get_stats()
    print(f"📈 Stats: {stats}")
    assert stats['total_requests'] == 5  # 4 from test 1 + 1 from test 4
    assert stats['blocked_requests'] == 1
    print("✅ Test 5 passed")
    
    # Test 6: Exception raising
    print("\n📝 Test 6: Exception raising")
    limiter.set_limit("test_service2", 1, 5)
    limiter.check_and_raise("test_service2", user_id)  # Should pass
    try:
        limiter.check_and_raise("test_service2", user_id)  # Should raise
        assert False, "Should have raised RateLimitExceeded"
    except RateLimitExceeded as e:
        print(f"✅ Caught expected exception: {e.message}")
        assert e.wait_time > 0
    
    print("\n🎉 All tests passed!")
    print(f"\nFinal stats: {limiter.get_stats()}")
