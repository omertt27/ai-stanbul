"""
Cache Manager
Redis caching strategies for responses and signals

Responsibilities:
- Semantic cache management
- Exact match cache
- Signal caching
- Cache key generation
- Cache statistics

Author: Istanbul AI Team
Date: November 14, 2025
"""

import logging
import hashlib
import time
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages all caching strategies
    
    Features:
    - Semantic cache (similarity-based)
    - Exact match cache
    - Signal caching
    - TTL management
    - Cache analytics
    """
    
    def __init__(
        self,
        redis_client=None,
        semantic_cache=None,
        default_ttl: int = 3600,
        enable_semantic: bool = True,
        enable_exact: bool = True
    ):
        """
        Initialize cache manager
        
        Args:
            redis_client: Redis client for exact match caching
            semantic_cache: Semantic cache instance for similarity search
            default_ttl: Default cache TTL in seconds (default: 1 hour)
            enable_semantic: Enable semantic cache (default: True)
            enable_exact: Enable exact match cache (default: True)
        """
        self.redis = redis_client
        self.semantic_cache = semantic_cache
        self.default_ttl = default_ttl
        self.enable_semantic = enable_semantic
        self.enable_exact = enable_exact
        
        # Statistics
        self.stats = {
            "semantic_hits": 0,
            "exact_hits": 0,
            "misses": 0,
            "stores": 0,
            "errors": 0
        }
        
        logger.info("✅ Cache Manager initialized")
        logger.info(f"   Semantic cache: {'✅ Enabled' if self.enable_semantic and self.semantic_cache else '❌ Disabled'}")
        logger.info(f"   Exact cache: {'✅ Enabled' if self.enable_exact and self.redis else '❌ Disabled'}")
        logger.info(f"   Default TTL: {default_ttl}s")
    
    def get_cache_key(self, query: str, language: str) -> str:
        """
        Generate cache key for exact match caching.
        
        Args:
            query: User query string
            language: Language code (en, tr, etc.)
            
        Returns:
            Cache key string (hash-based)
        """
        # Normalize query for consistent caching
        normalized = query.lower().strip()
        
        # Create cache key from query + language
        cache_string = f"{normalized}:{language}"
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        
        return f"llm_response:{cache_hash}"
    
    async def get_cached_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response using semantic or exact match.
        
        Strategy:
        1. Try semantic cache first (finds similar queries)
        2. Fall back to exact match cache
        3. Return None if no cache hit
        
        Args:
            query: User query string
            context: Query context with language, user_id, session_id
            
        Returns:
            Cached response dict or None if not found
        """
        start_time = time.time()
        language = context.get('language', 'en')
        user_id = context.get('user_id', 'anonymous')
        session_id = context.get('session_id')
        
        # Try semantic cache first (Priority 3.4)
        if self.enable_semantic and self.semantic_cache:
            try:
                cached_response = await self.semantic_cache.get_similar_response(
                    query=query,
                    context=context
                )
                
                if cached_response:
                    self.stats["semantic_hits"] += 1
                    retrieval_time = (time.time() - start_time) * 1000
                    
                    logger.info(f"✅ Semantic cache hit! (similarity-based)")
                    logger.info(f"   Retrieval time: {retrieval_time:.2f}ms")
                    
                    return {
                        "status": "success",
                        "response": cached_response['response']['response'],
                        "map_data": cached_response['response'].get('map_data'),
                        "signals": cached_response['response'].get('signals', {}),
                        "metadata": {
                            **cached_response['response'].get('metadata', {}),
                            "cached": True,
                            "cache_type": "semantic",
                            "original_cached_query": cached_response.get('query'),
                            "cache_retrieval_time": retrieval_time
                        }
                    }
            except Exception as e:
                logger.warning(f"⚠️ Semantic cache lookup failed: {e}")
                self.stats["errors"] += 1
        
        # Fall back to exact match cache
        if self.enable_exact and self.redis:
            try:
                cache_key = self.get_cache_key(query, language)
                cached = self.redis.get(cache_key)
                
                if cached:
                    self.stats["exact_hits"] += 1
                    retrieval_time = (time.time() - start_time) * 1000
                    
                    logger.info(f"✅ Exact cache hit!")
                    logger.info(f"   Retrieval time: {retrieval_time:.2f}ms")
                    
                    response = json.loads(cached)
                    
                    # Add cache metadata
                    if "metadata" not in response:
                        response["metadata"] = {}
                    response["metadata"]["cached"] = True
                    response["metadata"]["cache_type"] = "exact"
                    response["metadata"]["cache_retrieval_time"] = retrieval_time
                    
                    return response
            except Exception as e:
                logger.warning(f"⚠️ Exact cache lookup failed: {e}")
                self.stats["errors"] += 1
        
        # Cache miss
        self.stats["misses"] += 1
        return None
    
    async def cache_response(
        self,
        query: str,
        response: Dict[str, Any],
        context: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store response in both semantic and exact caches.
        
        Args:
            query: User query string
            response: Response dict to cache
            context: Query context with language, user_id, session_id
            ttl: Cache TTL in seconds (uses default if None)
            
        Returns:
            True if stored successfully, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
        
        language = context.get('language', 'en')
        success = False
        
        # Store in semantic cache
        if self.enable_semantic and self.semantic_cache:
            try:
                await self.semantic_cache.cache_response(
                    query=query,
                    response=response,
                    context=context,
                    ttl=ttl
                )
                
                logger.debug(f"✅ Stored in semantic cache (TTL: {ttl}s)")
                success = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to store in semantic cache: {e}")
                self.stats["errors"] += 1
        
        # Store in exact match cache
        if self.enable_exact and self.redis:
            try:
                cache_key = self.get_cache_key(query, language)
                cached_data = json.dumps(response)
                self.redis.setex(cache_key, ttl, cached_data)
                
                logger.debug(f"✅ Stored in exact cache (TTL: {ttl}s)")
                success = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to store in exact cache: {e}")
                self.stats["errors"] += 1
        
        if success:
            self.stats["stores"] += 1
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache performance metrics
        """
        total_requests = (
            self.stats["semantic_hits"] +
            self.stats["exact_hits"] +
            self.stats["misses"]
        )
        
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (
                (self.stats["semantic_hits"] + self.stats["exact_hits"]) /
                total_requests * 100
            )
        
        return {
            "semantic_hits": self.stats["semantic_hits"],
            "exact_hits": self.stats["exact_hits"],
            "misses": self.stats["misses"],
            "stores": self.stats["stores"],
            "errors": self.stats["errors"],
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 2),
            "semantic_enabled": self.enable_semantic and self.semantic_cache is not None,
            "exact_enabled": self.enable_exact and self.redis is not None
        }
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match keys (Redis only)
                    If None, clears all LLM response caches
        
        Returns:
            Number of entries cleared
        """
        cleared = 0
        
        # Clear exact cache
        if self.enable_exact and self.redis:
            try:
                if pattern is None:
                    pattern = "llm_response:*"
                
                keys = self.redis.keys(pattern)
                if keys:
                    cleared = self.redis.delete(*keys)
                    logger.info(f"✅ Cleared {cleared} exact cache entries")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clear exact cache: {e}")
                self.stats["errors"] += 1
        
        return cleared
