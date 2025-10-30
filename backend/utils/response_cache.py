"""
Smart Response Cache with Redis and local fallback
Implements intelligent caching for API responses with TTL and invalidation support
"""

import hashlib
import json
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SmartResponseCache:
    """Intelligent response caching with Redis fallback to local cache"""
    
    def __init__(self, redis_client=None, max_local_size: int = 1000):
        """
        Initialize cache with optional Redis client
        
        Args:
            redis_client: Redis client instance (optional)
            max_local_size: Maximum size of local fallback cache
        """
        self.redis = redis_client
        self.local_cache = OrderedDict()
        self.local_timestamps = {}
        self.max_local_size = max_local_size
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "redis_hits": 0,
            "local_hits": 0,
            "redis_errors": 0
        }
        logger.info(f"✅ SmartResponseCache initialized (Redis: {redis_client is not None})")
    
    def _generate_cache_key(self, query: str, intent: str = "", location: str = None) -> str:
        """
        Generate consistent cache key from query parameters
        
        Args:
            query: User query string
            intent: Detected intent
            location: Location context (optional)
            
        Returns:
            MD5 hash of normalized key components
        """
        key_parts = [query.lower().strip(), intent]
        if location:
            key_parts.append(location.lower().strip())
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, intent: str = "", location: str = None) -> Optional[Dict[str, Any]]:
        """
        Get cached response
        
        Args:
            query: User query string
            intent: Detected intent
            location: Location context (optional)
            
        Returns:
            Cached response dict or None if not found/expired
        """
        key = self._generate_cache_key(query, intent, location)
        
        # Try Redis first if available
        if self.redis:
            try:
                cached = self.redis.get(key)
                if cached:
                    self.cache_stats["hits"] += 1
                    self.cache_stats["redis_hits"] += 1
                    logger.debug(f"✅ Redis cache hit: {key[:16]}...")
                    return json.loads(cached)
            except Exception as e:
                self.cache_stats["redis_errors"] += 1
                logger.warning(f"⚠️ Redis error on get: {e}")
        
        # Fallback to local cache
        if key in self.local_cache:
            # Check if expired
            timestamp = self.local_timestamps.get(key)
            if timestamp and datetime.now() < timestamp:
                self.cache_stats["hits"] += 1
                self.cache_stats["local_hits"] += 1
                
                # Move to end (LRU)
                self.local_cache.move_to_end(key)
                
                logger.debug(f"✅ Local cache hit: {key[:16]}...")
                return self.local_cache[key]
            else:
                # Expired - remove it
                del self.local_cache[key]
                del self.local_timestamps[key]
        
        self.cache_stats["misses"] += 1
        logger.debug(f"❌ Cache miss: {key[:16]}...")
        return None
    
    def set(self, query: str, intent: str, response: Dict[str, Any], 
            location: str = None, ttl: int = 3600):
        """
        Cache response with TTL
        
        Args:
            query: User query string
            intent: Detected intent
            response: Response data to cache
            location: Location context (optional)
            ttl: Time to live in seconds (default 1 hour)
        """
        key = self._generate_cache_key(query, intent, location)
        
        # Try Redis with TTL
        if self.redis:
            try:
                self.redis.setex(key, ttl, json.dumps(response))
                logger.debug(f"✅ Cached to Redis: {key[:16]}... (TTL: {ttl}s)")
            except Exception as e:
                self.cache_stats["redis_errors"] += 1
                logger.warning(f"⚠️ Redis error on set: {e}")
        
        # Always update local cache as fallback
        if len(self.local_cache) >= self.max_local_size:
            # Remove oldest (LRU)
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
            del self.local_timestamps[oldest_key]
        
        self.local_cache[key] = response
        self.local_timestamps[key] = datetime.now() + timedelta(seconds=ttl)
        logger.debug(f"✅ Cached locally: {key[:16]}... (TTL: {ttl}s)")
    
    def invalidate(self, query: str, intent: str = "", location: str = None):
        """
        Invalidate specific cache entry
        
        Args:
            query: User query string
            intent: Detected intent
            location: Location context (optional)
        """
        key = self._generate_cache_key(query, intent, location)
        
        # Remove from Redis
        if self.redis:
            try:
                self.redis.delete(key)
            except Exception as e:
                logger.warning(f"⚠️ Redis error on invalidate: {e}")
        
        # Remove from local cache
        if key in self.local_cache:
            del self.local_cache[key]
            del self.local_timestamps[key]
        
        logger.info(f"🗑️ Cache invalidated: {key[:16]}...")
    
    def invalidate_pattern(self, pattern: str):
        """
        Invalidate all cache entries matching pattern
        
        Args:
            pattern: Pattern to match (e.g., "restaurant", "museum")
        """
        count = 0
        
        # Redis pattern invalidation
        if self.redis:
            try:
                keys = self.redis.keys(f"*{pattern}*")
                if keys:
                    self.redis.delete(*keys)
                    count += len(keys)
            except Exception as e:
                logger.warning(f"⚠️ Redis error on pattern invalidate: {e}")
        
        # Local cache pattern invalidation
        keys_to_delete = [
            key for key in self.local_cache.keys()
            if pattern in key
        ]
        
        for key in keys_to_delete:
            del self.local_cache[key]
            if key in self.local_timestamps:
                del self.local_timestamps[key]
            count += 1
        
        logger.info(f"🗑️ Pattern invalidation '{pattern}': {count} entries removed")
    
    def clear(self):
        """Clear all cache entries"""
        # Clear Redis
        if self.redis:
            try:
                self.redis.flushdb()
            except Exception as e:
                logger.warning(f"⚠️ Redis error on clear: {e}")
        
        # Clear local cache
        self.local_cache.clear()
        self.local_timestamps.clear()
        
        logger.info("🗑️ Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache performance metrics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "redis_hits": self.cache_stats["redis_hits"],
            "local_hits": self.cache_stats["local_hits"],
            "redis_errors": self.cache_stats["redis_errors"],
            "local_cache_size": len(self.local_cache),
            "max_local_size": self.max_local_size
        }
    
    def cleanup_expired(self):
        """Remove expired entries from local cache"""
        now = datetime.now()
        expired_keys = [
            key for key, timestamp in self.local_timestamps.items()
            if now >= timestamp
        ]
        
        for key in expired_keys:
            del self.local_cache[key]
            del self.local_timestamps[key]
        
        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired entries")


# Cache TTL recommendations by intent type
CACHE_TTL_RECOMMENDATIONS = {
    "restaurant": 3600,      # 1 hour - moderate change frequency
    "museum": 86400,         # 24 hours - static information
    "event": 1800,           # 30 minutes - dynamic schedules
    "transportation": 900,   # 15 minutes - real-time data
    "weather": 600,          # 10 minutes - frequently changing
    "general": 7200,         # 2 hours - general information
    "location": 3600,        # 1 hour - location-based queries
    "attraction": 43200,     # 12 hours - tourist attractions
}


def get_recommended_ttl(intent: str) -> int:
    """
    Get recommended TTL for an intent type
    
    Args:
        intent: Intent type
        
    Returns:
        Recommended TTL in seconds
    """
    return CACHE_TTL_RECOMMENDATIONS.get(intent, 3600)


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing SmartResponseCache...")
    
    # Test without Redis
    cache = SmartResponseCache()
    
    # Test basic operations
    test_query = "Best restaurants in Beyoğlu"
    test_response = {
        "response": "Here are the top restaurants...",
        "intent": "restaurant",
        "confidence": 0.95
    }
    
    # Cache miss
    result = cache.get(test_query, "restaurant", "Beyoğlu")
    assert result is None, "Should be cache miss"
    print("✅ Cache miss test passed")
    
    # Set cache
    cache.set(test_query, "restaurant", test_response, "Beyoğlu", ttl=10)
    print("✅ Cache set test passed")
    
    # Cache hit
    result = cache.get(test_query, "restaurant", "Beyoğlu")
    assert result == test_response, "Should be cache hit"
    print("✅ Cache hit test passed")
    
    # Test LRU eviction
    small_cache = SmartResponseCache(max_local_size=3)
    for i in range(5):
        small_cache.set(f"query_{i}", "test", {"data": i}, ttl=60)
    
    assert len(small_cache.local_cache) == 3, "Should respect max size"
    print("✅ LRU eviction test passed")
    
    # Test stats
    stats = cache.get_stats()
    assert stats["hits"] > 0, "Should have hits"
    assert stats["misses"] > 0, "Should have misses"
    print("✅ Stats test passed")
    print(f"📊 Cache stats: {stats}")
    
    # Test invalidation
    cache.invalidate(test_query, "restaurant", "Beyoğlu")
    result = cache.get(test_query, "restaurant", "Beyoğlu")
    assert result is None, "Should be invalidated"
    print("✅ Invalidation test passed")
    
    print("\n✅ All SmartResponseCache tests passed!")
