# AI Istanbul - Query Caching System
# Implements Redis-based caching for AI responses to reduce costs by 60-80%

import redis
import hashlib
import json
import logging
from typing import Optional
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class QueryCache:
    """Redis-based query caching system for AI responses"""
    
    def __init__(self):
        """Initialize Redis connection with fallback to in-memory cache"""
        self.redis_client = None
        self.memory_cache = {}  # Fallback cache
        self.cache_ttl = 86400 * 30  # 30 days TTL
        self.max_memory_cache_size = 1000  # Limit memory cache size
        
        # Try to connect to Redis
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"⚠️  Redis not available, using memory cache: {e}")
            self.redis_client = None
    
    def get_cache_key(self, query: str, context: str = "") -> str:
        """Generate cache key from query and context"""
        # Normalize query for better cache hits
        normalized_query = query.lower().strip()
        content = f"{normalized_query}:{context[:100]}"  # Limit context for key generation
        return f"query_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_cached_response(self, query: str, context: str = "") -> Optional[dict]:
        """Retrieve cached response if available"""
        try:
            cache_key = self.get_cache_key(query, context)
            
            if self.redis_client:
                # Try Redis first
                cached_data = self.redis_client.get(cache_key)
                if cached_data and isinstance(cached_data, str):
                    data = json.loads(cached_data)
                    logger.info(f"✅ Redis cache hit for query: {query[:50]}...")
                    return data
            else:
                # Fallback to memory cache
                if cache_key in self.memory_cache:
                    data = self.memory_cache[cache_key]
                    # Check if expired
                    if datetime.fromisoformat(data['expires_at']) > datetime.now():
                        logger.info(f"✅ Memory cache hit for query: {query[:50]}...")
                        return data
                    else:
                        # Remove expired entry
                        del self.memory_cache[cache_key]
                        
        except Exception as e:
            logger.error(f"❌ Cache retrieval error: {e}")
        
        logger.debug(f"🔍 Cache miss for query: {query[:50]}...")
        return None
    
    def cache_response(self, query: str, response: str, context: str = "", source: str = "ai"):
        """Cache successful AI response"""
        try:
            cache_key = self.get_cache_key(query, context)
            cache_data = {
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "query": query[:200],  # Truncate for storage efficiency
                "source": source,
                "context_hash": hashlib.md5(context.encode()).hexdigest()[:8] if context else None,
                "expires_at": (datetime.now() + timedelta(seconds=self.cache_ttl)).isoformat()
            }
            
            if self.redis_client:
                # Store in Redis
                self.redis_client.setex(
                    cache_key, 
                    self.cache_ttl, 
                    json.dumps(cache_data)
                )
                logger.info(f"💾 Response cached in Redis for query: {query[:50]}...")
            else:
                # Store in memory cache with size limit
                if len(self.memory_cache) >= self.max_memory_cache_size:
                    # Remove oldest entries
                    oldest_keys = sorted(
                        self.memory_cache.keys(), 
                        key=lambda k: self.memory_cache[k]['timestamp']
                    )[:100]
                    for old_key in oldest_keys:
                        del self.memory_cache[old_key]
                
                self.memory_cache[cache_key] = cache_data
                logger.info(f"💾 Response cached in memory for query: {query[:50]}...")
                
        except Exception as e:
            logger.error(f"❌ Cache storage error: {e}")
    
    def invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern (Redis only)"""
        if not self.redis_client:
            logger.warning("⚠️  Cache invalidation only available with Redis")
            return
        
        try:
            keys = self.redis_client.keys(f"query_cache:*{pattern}*")
            if keys and isinstance(keys, list):
                self.redis_client.delete(*keys)
                logger.info(f"🗑️  Invalidated {len(keys)} cache entries matching: {pattern}")
        except Exception as e:
            logger.error(f"❌ Cache invalidation error: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "cache_type": "redis" if self.redis_client else "memory",
            "total_entries": 0,
            "memory_usage_mb": 0,
            "hit_rate": "N/A"
        }
        
        try:
            if self.redis_client:
                # Redis stats
                info = self.redis_client.info()
                cache_keys = self.redis_client.keys("query_cache:*")
                if isinstance(info, dict) and isinstance(cache_keys, list):
                    stats.update({
                        "total_entries": len(cache_keys),
                        "memory_usage_mb": round(info.get('used_memory', 0) / 1024 / 1024, 2),
                        "connected_clients": info.get('connected_clients', 0)
                    })
            else:
                # Memory cache stats
                import sys
                cache_size = sum(sys.getsizeof(str(v)) for v in self.memory_cache.values())
                stats.update({
                    "total_entries": len(self.memory_cache),
                    "memory_usage_mb": round(cache_size / 1024 / 1024, 2)
                })
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
        
        return stats
    
    def clear_cache(self):
        """Clear all cache entries"""
        try:
            if self.redis_client:
                keys = self.redis_client.keys("query_cache:*")
                if keys and isinstance(keys, list):
                    self.redis_client.delete(*keys)
                    logger.info(f"🗑️  Cleared {len(keys)} Redis cache entries")
            else:
                count = len(self.memory_cache)
                self.memory_cache.clear()
                logger.info(f"🗑️  Cleared {count} memory cache entries")
        except Exception as e:
            logger.error(f"❌ Cache clear error: {e}")

# Global cache instance
query_cache = QueryCache()

# Cache hit/miss tracking for analytics
cache_stats = {
    "hits": 0,
    "misses": 0,
    "errors": 0
}

def track_cache_operation(operation: str):
    """Track cache operations for analytics"""
    if operation in cache_stats:
        cache_stats[operation] += 1

def get_cache_hit_rate() -> float:
    """Calculate cache hit rate percentage"""
    total = cache_stats["hits"] + cache_stats["misses"]
    if total == 0:
        return 0.0
    return round((cache_stats["hits"] / total) * 100, 2)
