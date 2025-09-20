"""
AI Rate Limiting and Caching Service for Istanbul AI Chatbot

This module implements intelligent rate limiting and caching for OpenAI API calls
to reduce costs, improve response times, and prevent API throttling.

Features:
- Per-user/session rate limiting
- Intelligent response caching based on query similarity
- Cache invalidation strategies
- Fallback mechanisms
- Performance metrics tracking
"""

import hashlib
import json
import time
from typing import Dict, Optional, Any, List, Tuple, Union
from datetime import datetime, timedelta
import redis
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIRateLimitCache:
    """
    Intelligent rate limiting and caching system for AI API calls.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 rate_limit_per_user: int = 20,  # requests per hour
                 rate_limit_per_ip: int = 100,    # requests per hour 
                 cache_ttl: int = 3600):          # cache TTL in seconds (1 hour)
        """
        Initialize the AI cache and rate limiting system.
        
        Args:
            redis_url: Redis connection URL
            rate_limit_per_user: Max requests per user per hour
            rate_limit_per_ip: Max requests per IP per hour
            cache_ttl: Cache time-to-live in seconds
        """
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("✅ Redis cache connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Running without cache.")
            self.redis_client = None
            self.cache_enabled = False
        
        self.rate_limit_per_user = rate_limit_per_user
        self.rate_limit_per_ip = rate_limit_per_ip
        self.cache_ttl = cache_ttl
        
        # In-memory fallback for rate limiting when Redis is unavailable
        self.memory_rate_limits = {}
        self.memory_cache = {}
        
    def _generate_cache_key(self, query: str, user_context: Optional[Dict] = None) -> str:
        """
        Generate a cache key based on normalized query and context.
        """
        # Normalize the query for better cache hits
        normalized_query = query.lower().strip()
        
        # Remove common variations and typos for better matching
        normalized_query = normalized_query.replace("restaurant", "restaurants")
        normalized_query = normalized_query.replace("restarunt", "restaurants")
        normalized_query = normalized_query.replace("resturant", "restaurants")
        
        # Include important context in cache key
        context_key = ""
        if user_context:
            # Only include location preferences in cache key
            location = user_context.get("location", "")
            language = user_context.get("language", "en")
            context_key = f"_{location}_{language}"
        
        # Create hash of normalized query + context
        cache_key = f"ai_cache:{hashlib.md5((normalized_query + context_key).encode()).hexdigest()}"
        return cache_key
    
    def _get_rate_limit_key(self, identifier: str, limit_type: str) -> str:
        """Generate rate limit key for user/IP tracking."""
        hour = datetime.now().strftime("%Y%m%d%H")
        return f"rate_limit:{limit_type}:{identifier}:{hour}"
    
    def check_rate_limit(self, session_id: str, ip_address: str) -> Tuple[bool, Dict]:
        """
        Check if the request should be rate limited.
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        hour_key = int(current_time // 3600)  # Current hour
        
        if self.cache_enabled and self.redis_client:
            try:
                # Check user rate limit
                user_key = self._get_rate_limit_key(session_id, "user")
                user_count_raw = self.redis_client.get(user_key)
                user_count = int(str(user_count_raw)) if user_count_raw else 0
                
                # Check IP rate limit
                ip_key = self._get_rate_limit_key(ip_address, "ip")
                ip_count_raw = self.redis_client.get(ip_key)
                ip_count = int(str(ip_count_raw)) if ip_count_raw else 0
                
                # Check limits
                user_exceeded = user_count >= self.rate_limit_per_user
                ip_exceeded = ip_count >= self.rate_limit_per_ip
                
                if user_exceeded or ip_exceeded:
                    return False, {
                        "rate_limited": True,
                        "user_requests": user_count,
                        "ip_requests": ip_count,
                        "reset_time": (hour_key + 1) * 3600,
                        "message": f"Rate limit exceeded. Try again in {60 - datetime.now().minute} minutes."
                    }
                
                return True, {
                    "rate_limited": False,
                    "user_requests": user_count,
                    "ip_requests": ip_count,
                    "remaining_user": self.rate_limit_per_user - user_count,
                    "remaining_ip": self.rate_limit_per_ip - ip_count
                }
                
            except Exception as e:
                logger.warning(f"Redis rate limit check failed: {e}")
                # Fall back to memory-based rate limiting
                
        # Memory-based fallback rate limiting
        user_key = f"user_{session_id}_{hour_key}"
        ip_key = f"ip_{ip_address}_{hour_key}"
        
        # Clean old entries (older than 2 hours)
        cutoff_hour = hour_key - 2
        keys_to_remove = []
        for k in self.memory_rate_limits.keys():
            # Extract hour from key (format: type_identifier_hour)
            parts = k.split('_')
            if len(parts) >= 3:
                try:
                    key_hour = int(parts[-1])
                    if key_hour < cutoff_hour:
                        keys_to_remove.append(k)
                except ValueError:
                    pass
        
        for k in keys_to_remove:
            del self.memory_rate_limits[k]
        
        user_count = self.memory_rate_limits.get(user_key, 0)
        ip_count = self.memory_rate_limits.get(ip_key, 0)
        
        if user_count >= self.rate_limit_per_user or ip_count >= self.rate_limit_per_ip:
            return False, {
                "rate_limited": True,
                "user_requests": user_count,
                "ip_requests": ip_count,
                "message": "Rate limit exceeded. Try again later."
            }
        
        return True, {
            "rate_limited": False,
            "user_requests": user_count,
            "ip_requests": ip_count,
            "remaining_user": self.rate_limit_per_user - user_count,
            "remaining_ip": self.rate_limit_per_ip - ip_count
        }
    
    def increment_rate_limit(self, session_id: str, ip_address: str):
        """Increment rate limit counters for user and IP."""
        current_time = time.time()
        hour_key = int(current_time // 3600)
        
        if self.cache_enabled and self.redis_client:
            try:
                user_key = self._get_rate_limit_key(session_id, "user")
                ip_key = self._get_rate_limit_key(ip_address, "ip")
                
                # Increment with expiration
                pipe = self.redis_client.pipeline()
                pipe.incr(user_key)
                pipe.expire(user_key, 3600)  # Expire after 1 hour
                pipe.incr(ip_key)
                pipe.expire(ip_key, 3600)
                pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis rate limit increment failed: {e}")
        
        # Memory fallback
        user_key = f"user_{session_id}_{hour_key}"
        ip_key = f"ip_{ip_address}_{hour_key}"
        
        self.memory_rate_limits[user_key] = self.memory_rate_limits.get(user_key, 0) + 1
        self.memory_rate_limits[ip_key] = self.memory_rate_limits.get(ip_key, 0) + 1
    
    def get_cached_response(self, query: str, user_context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Retrieve cached AI response if available.
        
        Args:
            query: User query string
            user_context: User context for cache key generation
            
        Returns:
            Cached response dict or None if not found
        """
        if not self.cache_enabled:
            return None
            
        cache_key = self._generate_cache_key(query, user_context)
        
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    response = json.loads(str(cached_data))
                    response["from_cache"] = True
                    response["cache_hit_time"] = datetime.now().isoformat()
                    logger.info(f"✅ Cache HIT for query: {query[:50]}...")
                    return response
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}")
        
        # Memory fallback
        if cache_key in self.memory_cache:
            cached_entry = self.memory_cache[cache_key]
            if time.time() - cached_entry["timestamp"] < self.cache_ttl:
                response = cached_entry["data"]
                response["from_cache"] = True
                response["cache_hit_time"] = datetime.now().isoformat()
                logger.info(f"✅ Memory cache HIT for query: {query[:50]}...")
                return response
            else:
                # Remove expired entry
                del self.memory_cache[cache_key]
        
        logger.info(f"❌ Cache MISS for query: {query[:50]}...")
        return None
    
    def cache_response(self, query: str, response: Dict, user_context: Optional[Dict] = None):
        """
        Cache AI response for future use.
        
        Args:
            query: User query string
            response: AI response to cache
            user_context: User context for cache key generation
        """
        if not self.cache_enabled:
            return
            
        cache_key = self._generate_cache_key(query, user_context)
        
        # Prepare response for caching (remove non-serializable data)
        cache_data = {
            "message": response.get("message", ""),
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Store partial query for debugging
            "from_cache": False
        }
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(cache_data)
                )
                logger.info(f"✅ Cached response for query: {query[:50]}...")
                return
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {e}")
        
        # Memory fallback
        self.memory_cache[cache_key] = {
            "data": cache_data,
            "timestamp": time.time()
        }
        
        # Clean old memory cache entries (keep only last 100)
        if len(self.memory_cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            for old_key, _ in sorted_items[:-80]:  # Keep 80 most recent
                del self.memory_cache[old_key]
        
        logger.info(f"✅ Memory cached response for query: {query[:50]}...")
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        stats = {
            "cache_enabled": self.cache_enabled,
            "redis_available": self.redis_client is not None,
            "rate_limits": {
                "per_user_hour": self.rate_limit_per_user,
                "per_ip_hour": self.rate_limit_per_ip
            },
            "cache_ttl_seconds": self.cache_ttl
        }
        
        if self.redis_client:
            try:
                redis_info_raw = self.redis_client.info()
                if isinstance(redis_info_raw, dict):
                    redis_info = redis_info_raw
                else:
                    redis_info = {"used_memory_human": "unknown", "connected_clients": 0}
                    
                stats["redis_info"] = {
                    "connected": True,
                    "memory_usage": redis_info.get("used_memory_human", "unknown"),
                    "connected_clients": redis_info.get("connected_clients", 0)
                }
            except Exception:
                stats["redis_info"] = {"connected": False}
        
        if hasattr(self, 'memory_cache'):
            stats["memory_cache_size"] = len(self.memory_cache)
            stats["memory_rate_limit_entries"] = len(self.memory_rate_limits)
        
        return stats
    
    def clear_cache(self, pattern: str = "ai_cache:*"):
        """Clear cached responses matching pattern."""
        if self.redis_client:
            try:
                keys_raw = self.redis_client.keys(pattern)
                if isinstance(keys_raw, list):
                    keys = keys_raw
                    if keys:
                        self.redis_client.delete(*keys)
                        logger.info(f"✅ Cleared {len(keys)} cache entries from Redis")
                else:
                    logger.warning("Unable to retrieve cache keys from Redis")
            except Exception as e:
                logger.warning(f"Redis cache clear failed: {e}")
        
        # Clear memory cache
        self.memory_cache.clear()
        logger.info("✅ Cleared memory cache")

    def optimize_database_queries(self, db: Session):
        """
        Apply additional database optimizations for better performance.
        """
        try:
            # Analyze database for query optimization
            db.execute(text("ANALYZE;"))
            
            # Update table statistics
            db.execute(text("PRAGMA optimize;"))
            
            logger.info("✅ Database optimization completed")
        except Exception as e:
            logger.warning(f"Database optimization failed: {e}")

# Global instance
ai_cache_service = None

def get_ai_cache_service() -> AIRateLimitCache:
    """Get or create the global AI cache service instance."""
    global ai_cache_service
    if ai_cache_service is None:
        ai_cache_service = AIRateLimitCache()
    return ai_cache_service

def init_ai_cache_service(redis_url: str = "redis://localhost:6379", 
                          rate_limit_per_user: int = 5,  # Lower for testing
                          rate_limit_per_ip: int = 50):   # Lower for testing
    """Initialize the AI cache service with custom configuration."""
    global ai_cache_service
    ai_cache_service = AIRateLimitCache(
        redis_url=redis_url,
        rate_limit_per_user=rate_limit_per_user,
        rate_limit_per_ip=rate_limit_per_ip
    )
    return ai_cache_service
