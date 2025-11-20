"""
Advanced 3-Tier Caching System
L1: In-Memory (Redis) - Ultra-fast, 5min TTL
L2: Semantic Cache - Similar queries, 1hr TTL  
L3: Long-term Cache - Persistent, 24hr TTL
"""
import hashlib
import json
import time
from typing import Optional, Dict, Any, List
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class AdvancedCacheSystem:
    """Multi-tier caching with semantic similarity matching"""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        similarity_threshold: float = 0.85,
        enable_semantic: bool = True
    ):
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold
        self.enable_semantic = enable_semantic
        
        # Load semantic model for L2 cache
        if enable_semantic:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.enable_semantic = False
        
        # Cache statistics
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0
        }
    
    def _generate_key(self, query: str, language: str, context: Dict = None) -> str:
        """Generate cache key from query and context"""
        cache_input = {
            'query': query.lower().strip(),
            'language': language,
            'context': context or {}
        }
        key_str = json.dumps(cache_input, sort_keys=True)
        return f"cache:l1:{hashlib.sha256(key_str.encode()).hexdigest()}"
    
    def _get_l1_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """L1: In-memory cache (fastest)"""
        try:
            cached = self.redis.get(key)
            if cached:
                self.stats['l1_hits'] += 1
                logger.info(f"L1 cache hit: {key[:16]}...")
                return json.loads(cached)
        except Exception as e:
            logger.error(f"L1 cache error: {e}")
        return None
    
    def _set_l1_cache(self, key: str, value: Dict[str, Any], ttl: int = 300):
        """Store in L1 cache with 5min TTL"""
        try:
            self.redis.setex(
                key,
                ttl,
                json.dumps(value)
            )
            logger.debug(f"L1 cache set: {key[:16]}...")
        except Exception as e:
            logger.error(f"L1 cache set error: {e}")
    
    def _get_l2_semantic_cache(
        self,
        query: str,
        language: str
    ) -> Optional[Dict[str, Any]]:
        """L2: Semantic similarity cache"""
        if not self.enable_semantic:
            return None
        
        try:
            # Generate query embedding
            query_embedding = self.semantic_model.encode(query)
            
            # Search for similar queries
            pattern = f"cache:l2:{language}:*"
            for key in self.redis.scan_iter(match=pattern):
                cached_data = self.redis.get(key)
                if not cached_data:
                    continue
                
                cached = json.loads(cached_data)
                cached_embedding = np.array(cached['embedding'])
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, cached_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                )
                
                if similarity >= self.similarity_threshold:
                    self.stats['l2_hits'] += 1
                    logger.info(f"L2 semantic cache hit (similarity: {similarity:.2f})")
                    return cached['response']
        
        except Exception as e:
            logger.error(f"L2 semantic cache error: {e}")
        
        return None
    
    def _set_l2_semantic_cache(
        self,
        query: str,
        language: str,
        response: Dict[str, Any],
        ttl: int = 3600
    ):
        """Store in L2 semantic cache with 1hr TTL"""
        if not self.enable_semantic:
            return
        
        try:
            # Generate embedding
            embedding = self.semantic_model.encode(query)
            
            key = f"cache:l2:{language}:{hashlib.sha256(query.encode()).hexdigest()}"
            cache_data = {
                'query': query,
                'embedding': embedding.tolist(),
                'response': response,
                'timestamp': time.time()
            }
            
            self.redis.setex(key, ttl, json.dumps(cache_data))
            logger.debug(f"L2 semantic cache set: {key[:32]}...")
        
        except Exception as e:
            logger.error(f"L2 semantic cache set error: {e}")
    
    def _get_l3_persistent_cache(
        self,
        query: str,
        language: str
    ) -> Optional[Dict[str, Any]]:
        """L3: Long-term persistent cache"""
        try:
            key = f"cache:l3:{language}:{hashlib.sha256(query.encode()).hexdigest()}"
            cached = self.redis.get(key)
            
            if cached:
                self.stats['l3_hits'] += 1
                logger.info(f"L3 persistent cache hit")
                return json.loads(cached)
        
        except Exception as e:
            logger.error(f"L3 cache error: {e}")
        
        return None
    
    def _set_l3_persistent_cache(
        self,
        query: str,
        language: str,
        response: Dict[str, Any],
        ttl: int = 86400
    ):
        """Store in L3 cache with 24hr TTL"""
        try:
            key = f"cache:l3:{language}:{hashlib.sha256(query.encode()).hexdigest()}"
            cache_data = {
                'query': query,
                'response': response,
                'timestamp': time.time(),
                'access_count': 1
            }
            
            self.redis.setex(key, ttl, json.dumps(cache_data))
            logger.debug(f"L3 persistent cache set")
        
        except Exception as e:
            logger.error(f"L3 cache set error: {e}")
    
    async def get(
        self,
        query: str,
        language: str,
        context: Dict = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get from cache (checks L1 -> L2 -> L3)
        
        Args:
            query: User query
            language: Language code
            context: Additional context
        
        Returns:
            Cached response or None
        """
        # Try L1 cache first
        key = self._generate_key(query, language, context)
        result = self._get_l1_cache(key)
        if result:
            return result
        
        # Try L2 semantic cache
        result = self._get_l2_semantic_cache(query, language)
        if result:
            # Promote to L1
            self._set_l1_cache(key, result)
            return result
        
        # Try L3 persistent cache
        result = self._get_l3_persistent_cache(query, language)
        if result:
            # Promote to L1 and L2
            self._set_l1_cache(key, result)
            self._set_l2_semantic_cache(query, language, result)
            return result
        
        self.stats['misses'] += 1
        return None
    
    async def set(
        self,
        query: str,
        language: str,
        response: Dict[str, Any],
        context: Dict = None,
        importance: str = 'normal'
    ):
        """
        Store in cache (all tiers)
        
        Args:
            query: User query
            language: Language code
            response: LLM response
            context: Additional context
            importance: 'low', 'normal', 'high' - affects TTL
        """
        # Set TTLs based on importance
        ttls = {
            'low': (180, 1800, 43200),      # 3min, 30min, 12hr
            'normal': (300, 3600, 86400),    # 5min, 1hr, 24hr
            'high': (600, 7200, 172800)      # 10min, 2hr, 48hr
        }
        l1_ttl, l2_ttl, l3_ttl = ttls.get(importance, ttls['normal'])
        
        # Store in all cache tiers
        key = self._generate_key(query, language, context)
        self._set_l1_cache(key, response, ttl=l1_ttl)
        self._set_l2_semantic_cache(query, language, response, ttl=l2_ttl)
        self._set_l3_persistent_cache(query, language, response, ttl=l3_ttl)
        
        logger.info(f"Cached response across all tiers (importance: {importance})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'total_requests': total,
            'hit_rate': (total - self.stats['misses']) / total * 100,
            'l1_hit_rate': self.stats['l1_hits'] / total * 100,
            'l2_hit_rate': self.stats['l2_hits'] / total * 100,
            'l3_hit_rate': self.stats['l3_hits'] / total * 100,
        }
    
    async def warm_cache(self, queries: List[Dict[str, str]]):
        """
        Pre-populate cache with common queries
        
        Args:
            queries: List of {'query': str, 'language': str} dicts
        """
        logger.info(f"Warming cache with {len(queries)} queries...")
        
        for item in queries:
            query = item['query']
            language = item['language']
            
            # Check if already cached
            cached = await self.get(query, language)
            if cached:
                continue
            
            # This would normally call the LLM to generate response
            # For cache warming, you'd integrate with your LLM service
            logger.debug(f"Cache warming: {query[:50]}...")
    
    def invalidate(self, pattern: str = None):
        """
        Invalidate cache entries
        
        Args:
            pattern: Redis key pattern (default: all cache keys)
        """
        pattern = pattern or "cache:*"
        count = 0
        
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)
            count += 1
        
        logger.info(f"Invalidated {count} cache entries")
        return count


# Usage example
async def example_usage():
    """Example of how to use the advanced cache system"""
    import redis.asyncio as redis_async
    
    # Initialize Redis connection
    redis_client = await redis_async.from_url("redis://localhost:6379")
    
    # Create cache system
    cache = AdvancedCacheSystem(
        redis_client=redis_client,
        similarity_threshold=0.85,
        enable_semantic=True
    )
    
    # Try to get from cache
    query = "Best restaurants in Taksim"
    language = "en"
    
    cached_response = await cache.get(query, language)
    
    if cached_response:
        print("Cache hit!", cached_response)
    else:
        # Generate response (call LLM)
        response = {
            "answer": "Here are some great restaurants...",
            "confidence": 0.9
        }
        
        # Store in cache
        await cache.set(query, language, response, importance='high')
    
    # Get statistics
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
