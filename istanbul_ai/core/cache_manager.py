"""
Cache Manager for Istanbul AI
High-performance Redis-based caching layer
"""

import redis.asyncio as redis
import json
import hashlib
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    High-performance caching layer for Istanbul AI
    Reduces GPU load and improves response times
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis: Optional[redis.Redis] = None
        self.redis_url = redis_url
        
        # Cache TTLs (Time To Live)
        self.ttl_config = {
            "intent": 3600,        # 1 hour
            "entity": 3600,        # 1 hour
            "language": 7200,      # 2 hours
            "response": 1800,      # 30 minutes
            "venue": 3600,         # 1 hour
            "route": 1800,         # 30 minutes (routes change)
            "events": 1800,        # 30 minutes
            "transportation": 300  # 5 minutes (dynamic)
        }
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50
            )
            await self.redis.ping()
            logger.info("✅ Connected to Redis cache")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis = None
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("✅ Disconnected from Redis")
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            key_data = data
        else:
            key_data = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.md5(key_data.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                logger.debug(f"✅ Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"❌ Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value"""
        if not self.redis:
            return
        
        try:
            await self.redis.setex(
                key,
                ttl,
                json.dumps(value)
            )
            logger.debug(f"✅ Cache SET: {key} (ttl={ttl}s)")
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
    
    async def get_or_compute(self, cache_type: str, data: Any, 
                            compute_func, **kwargs) -> Any:
        """
        Get from cache or compute and cache
        This is the main interface for caching
        """
        key = self._generate_cache_key(cache_type, data)
        
        # Try to get from cache
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        # Compute if not in cache
        result = await compute_func(**kwargs)
        
        # Cache the result
        ttl = self.ttl_config.get(cache_type, 3600)
        await self.set(key, result, ttl)
        
        return result
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        if not self.redis:
            return
        
        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match=pattern, count=100
                )
                if keys:
                    await self.redis.delete(*keys)
                if cursor == 0:
                    break
            logger.info(f"✅ Invalidated cache pattern: {pattern}")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.redis:
            return {}
        
        try:
            info = await self.redis.info("stats")
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / 
                           (info.get("keyspace_hits", 0) + 
                            info.get("keyspace_misses", 1)) * 100
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# Global cache manager
_cache_manager: Optional[CacheManager] = None

async def get_cache_manager() -> CacheManager:
    """Get or create global cache manager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.connect()
    return _cache_manager
