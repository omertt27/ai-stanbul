"""
caching.py - Caching System

Dual-layer caching system with semantic similarity and exact matching.

Features:
- Semantic cache: Find similar queries using embeddings
- Exact match cache: Traditional key-value caching
- Redis backend support
- TTL management
- Cache statistics
- Automatic invalidation

Author: AI Istanbul Team
Date: November 2025
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Dual-layer caching system.
    
    Layer 1: Semantic Cache (similarity-based, slower but finds related queries)
    Layer 2: Exact Match Cache (key-value, fast but requires exact match)
    """
    
    def __init__(
        self,
        redis_client=None,
        enable_semantic_cache: bool = True,
        cache_ttl: int = 3600,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client (optional, falls back to in-memory)
            enable_semantic_cache: Enable semantic similarity caching
            cache_ttl: Cache time-to-live in seconds
            similarity_threshold: Minimum similarity for semantic cache hits
        """
        self.redis = redis_client
        self.enable_semantic = enable_semantic_cache
        self.cache_ttl = cache_ttl
        self.similarity_threshold = similarity_threshold
        
        # In-memory fallback cache
        self.memory_cache = {}
        self.semantic_cache = {}
        
        # Statistics
        self.stats = {
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'stores': 0
        }
        
        logger.info(f"✅ Cache Manager initialized (semantic={'enabled' if enable_semantic_cache else 'disabled'})")
    
    async def get_cached_response(
        self,
        query: str,
        language: str = "en",
        similarity_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query.
        
        Tries in order:
        1. Exact match cache
        2. Semantic cache (if enabled)
        
        Args:
            query: User query
            language: Language code
            similarity_threshold: Override default similarity threshold
            
        Returns:
            Cached response dict or None
        """
        # Try exact match first (fastest)
        cache_key = self._generate_cache_key(query, language)
        
        cached = await self._get_exact_match(cache_key)
        if cached:
            self.stats['exact_hits'] += 1
            logger.debug(f"✅ Exact cache hit for: {query[:50]}...")
            return cached
        
        # Try semantic cache (slower but finds similar queries)
        if self.enable_semantic:
            threshold = similarity_threshold or self.similarity_threshold
            
            cached = await self._get_semantic_match(
                query=query,
                language=language,
                threshold=threshold
            )
            
            if cached:
                self.stats['semantic_hits'] += 1
                logger.debug(f"✅ Semantic cache hit for: {query[:50]}...")
                return cached
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    async def cache_response(
        self,
        query: str,
        language: str,
        response: Dict[str, Any]
    ):
        """
        Cache a response.
        
        Args:
            query: User query
            language: Language code
            response: Response dict to cache
        """
        cache_key = self._generate_cache_key(query, language)
        
        # Add metadata
        cache_entry = {
            **response,
            '_cached_at': datetime.now().isoformat(),
            '_query': query,
            '_language': language
        }
        
        # Store in exact match cache
        await self._store_exact_match(cache_key, cache_entry)
        
        # Store in semantic cache (if enabled)
        if self.enable_semantic:
            await self._store_semantic(query, language, cache_entry)
        
        self.stats['stores'] += 1
        logger.debug(f"💾 Cached response for: {query[:50]}...")
    
    def _generate_cache_key(self, query: str, language: str) -> str:
        """
        Generate cache key from query and language.
        
        Args:
            query: User query
            language: Language code
            
        Returns:
            Cache key string
        """
        # Normalize query (lowercase, strip whitespace)
        normalized = query.lower().strip()
        
        # Create key with language
        key_string = f"{language}:{normalized}"
        
        # Hash for consistent length
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"llm_cache:{key_hash}"
    
    async def _get_exact_match(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get exact match from cache."""
        # Try Redis first
        if self.redis:
            try:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fallback to in-memory cache
        return self.memory_cache.get(cache_key)
    
    async def _store_exact_match(self, cache_key: str, data: Dict[str, Any]):
        """Store in exact match cache."""
        # Try Redis first
        if self.redis:
            try:
                await self.redis.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Always store in memory as fallback
        self.memory_cache[cache_key] = data
        
        # Implement simple LRU for memory cache
        if len(self.memory_cache) > 1000:
            # Remove oldest entries
            to_remove = list(self.memory_cache.keys())[:100]
            for key in to_remove:
                del self.memory_cache[key]
    
    async def _get_semantic_match(
        self,
        query: str,
        language: str,
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """
        Find semantically similar cached query.
        
        Args:
            query: User query
            language: Language code
            threshold: Similarity threshold (0-1)
            
        Returns:
            Cached response or None
        """
        # TODO: Implement proper semantic search with embeddings
        # For now, this is a placeholder
        
        # In a real implementation, you would:
        # 1. Generate embedding for query
        # 2. Search for similar embeddings in vector database
        # 3. Return cached response if similarity > threshold
        
        return None
    
    async def _store_semantic(
        self,
        query: str,
        language: str,
        data: Dict[str, Any]
    ):
        """
        Store in semantic cache with embedding.
        
        Args:
            query: User query
            language: Language code
            data: Data to cache
        """
        # TODO: Implement semantic storage
        # For now, this is a placeholder
        
        # In a real implementation, you would:
        # 1. Generate embedding for query
        # 2. Store embedding + data in vector database
        # 3. Index for fast similarity search
        
        pass
    
    async def invalidate_cache(
        self,
        pattern: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Key pattern to match (e.g., "restaurant*")
            language: Invalidate specific language only
        """
        if self.redis:
            try:
                if pattern:
                    # Find matching keys
                    keys = await self.redis.keys(f"llm_cache:*{pattern}*")
                    if keys:
                        await self.redis.delete(*keys)
                        logger.info(f"🗑️ Invalidated {len(keys)} cache entries")
                else:
                    # Clear all
                    await self.redis.flushdb()
                    logger.info("🗑️ Cleared entire cache")
            except Exception as e:
                logger.error(f"Cache invalidation failed: {e}")
        
        # Clear in-memory cache
        if pattern:
            # Simple pattern matching for in-memory cache
            to_remove = [
                key for key in self.memory_cache.keys()
                if pattern.lower() in key.lower()
            ]
            for key in to_remove:
                del self.memory_cache[key]
        else:
            self.memory_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = (
            self.stats['exact_hits'] +
            self.stats['semantic_hits'] +
            self.stats['misses']
        )
        
        if total_requests == 0:
            hit_rate = 0.0
        else:
            hit_rate = (
                (self.stats['exact_hits'] + self.stats['semantic_hits']) /
                total_requests
            )
        
        return {
            'exact_hits': self.stats['exact_hits'],
            'semantic_hits': self.stats['semantic_hits'],
            'misses': self.stats['misses'],
            'stores': self.stats['stores'],
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'semantic_cache_size': len(self.semantic_cache)
        }
    
    async def get_cache_size(self) -> Dict[str, Any]:
        """Get cache size information."""
        redis_size = 0
        
        if self.redis:
            try:
                # Count keys matching our pattern
                keys = await self.redis.keys("llm_cache:*")
                redis_size = len(keys)
            except Exception as e:
                logger.warning(f"Failed to get Redis size: {e}")
        
        return {
            'redis_entries': redis_size,
            'memory_entries': len(self.memory_cache),
            'semantic_entries': len(self.semantic_cache),
            'total_entries': redis_size + len(self.memory_cache)
        }
    
    def cleanup_expired(self):
        """Clean up expired entries from in-memory cache."""
        # In-memory cache doesn't track expiry by default
        # This is a simple cleanup that removes entries older than TTL
        
        now = datetime.now()
        to_remove = []
        
        for key, entry in self.memory_cache.items():
            cached_at_str = entry.get('_cached_at')
            if cached_at_str:
                try:
                    cached_at = datetime.fromisoformat(cached_at_str)
                    age = (now - cached_at).total_seconds()
                    
                    if age > self.cache_ttl:
                        to_remove.append(key)
                except Exception:
                    pass
        
        for key in to_remove:
            del self.memory_cache[key]
        
        if to_remove:
            logger.debug(f"🗑️ Cleaned up {len(to_remove)} expired cache entries")
