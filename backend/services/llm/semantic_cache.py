"""
Semantic Cache for LLM Responses

Uses embedding-based similarity to cache responses for semantically similar queries.
Unlike MD5 hash caching, this can match:
- "How do I get to Sultanahmet?" ≈ "What's the best way to reach Sultanahmet?"
- "Weather in Istanbul" ≈ "What's the weather like in Istanbul today?"

Author: AI Istanbul Team
Date: December 2024
"""

import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response with metadata"""
    query: str
    response: str
    embedding: Optional[np.ndarray]
    timestamp: float
    hit_count: int = 0
    language: str = "en"
    context_hash: str = ""  # Hash of context used
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """
    Embedding-based semantic cache for LLM responses.
    
    Features:
    - Cosine similarity matching for semantic queries
    - Context-aware caching (different context = different cache)
    - TTL-based expiration
    - LRU eviction when cache is full
    - Fallback to exact match if embeddings unavailable
    """
    
    def __init__(
        self,
        max_entries: int = 500,
        similarity_threshold: float = 0.92,
        ttl_seconds: float = 300.0,  # 5 minutes default
        embedding_dim: int = 384  # sentence-transformers default
    ):
        """
        Initialize semantic cache.
        
        Args:
            max_entries: Maximum cache entries
            similarity_threshold: Minimum cosine similarity for cache hit
            ttl_seconds: Time-to-live for cache entries
            embedding_dim: Dimension of embeddings (for validation)
        """
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.embedding_dim = embedding_dim
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embedding_index: List[Tuple[str, np.ndarray]] = []  # (key, embedding) pairs
        
        # Embedding model (lazy loaded)
        self._embedding_model = None
        self._embeddings_available = False
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'semantic_hits': 0,
            'exact_hits': 0,
            'evictions': 0,
            'expirations': 0
        }
        
        logger.info(f"🧠 Semantic Cache initialized (max={max_entries}, threshold={similarity_threshold})")
    
    def _load_embedding_model(self):
        """Lazy load embedding model - DISABLED for performance in production"""
        import os
        
        # Check environment variable to explicitly disable embeddings
        disable_embeddings = os.environ.get('DISABLE_EMBEDDINGS', 'true').lower() == 'true'
        is_production = os.environ.get('ENVIRONMENT', '').lower() == 'production'
        
        if disable_embeddings or is_production:
            logger.info("⚡ Semantic cache using exact match only (embeddings disabled for speed)")
            self._embeddings_available = False
            return
        
        # Only load embeddings in development with explicit opt-in
        if self._embedding_model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight, fast model
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._embeddings_available = True
            logger.info("✅ Embedding model loaded for semantic cache")
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available, using exact match only")
            self._embeddings_available = False
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self._embeddings_available = False
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text"""
        if not self._embeddings_available:
            self._load_embedding_model()
        
        if not self._embeddings_available or self._embedding_model is None:
            return None
        
        try:
            embedding = self._embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        # Normalize
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def _get_cache_key(self, query: str, context_hash: str = "") -> str:
        """Generate cache key from query and context"""
        combined = f"{query.lower().strip()}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _hash_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Hash context to detect changes"""
        if not context:
            return ""
        
        # Create a stable string representation
        context_str = str(sorted(context.items())) if isinstance(context, dict) else str(context)
        return hashlib.md5(context_str.encode()).hexdigest()[:12]
    
    def _evict_expired(self):
        """Remove expired entries"""
        now = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if now - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            # Remove from embedding index
            self._embedding_index = [
                (k, e) for k, e in self._embedding_index if k != key
            ]
            self.stats['expirations'] += 1
        
        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self, count: int = 1):
        """Evict least recently used entries"""
        for _ in range(count):
            if self._cache:
                # OrderedDict pops oldest first
                key, _ = self._cache.popitem(last=False)
                # Remove from embedding index
                self._embedding_index = [
                    (k, e) for k, e in self._embedding_index if k != key
                ]
                self.stats['evictions'] += 1
    
    def get(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> Optional[str]:
        """
        Get cached response for query.
        
        Tries:
        1. Exact match (fastest)
        2. Semantic similarity match (if embeddings available)
        
        Args:
            query: User query
            context: Query context (for context-aware caching)
            language: Expected language
            
        Returns:
            Cached response or None
        """
        # Clean expired entries periodically
        if len(self._cache) > 0 and len(self._cache) % 50 == 0:
            self._evict_expired()
        
        context_hash = self._hash_context(context)
        cache_key = self._get_cache_key(query, context_hash)
        
        # Try exact match first
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            
            # Check expiration
            if time.time() - entry.timestamp > self.ttl_seconds:
                del self._cache[cache_key]
                self.stats['expirations'] += 1
            else:
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                entry.hit_count += 1
                self.stats['hits'] += 1
                self.stats['exact_hits'] += 1
                logger.info(f"✅ Cache EXACT HIT: '{query[:30]}...'")
                return entry.response
        
        # Try semantic match if embeddings available
        if self._embeddings_available or self._embedding_model is None:
            self._load_embedding_model()
        
        if self._embeddings_available and self._embedding_index:
            query_embedding = self._get_embedding(query)
            
            if query_embedding is not None:
                best_match = None
                best_similarity = 0.0
                
                for key, cached_embedding in self._embedding_index:
                    # Check if entry still exists and not expired
                    if key not in self._cache:
                        continue
                    
                    entry = self._cache[key]
                    
                    # Skip if expired
                    if time.time() - entry.timestamp > self.ttl_seconds:
                        continue
                    
                    # Skip if different context
                    if entry.context_hash != context_hash:
                        continue
                    
                    # Skip if different language
                    if entry.language != language:
                        continue
                    
                    similarity = self._compute_similarity(query_embedding, cached_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry
                
                if best_match and best_similarity >= self.similarity_threshold:
                    # Move to end (most recently used)
                    match_key = self._get_cache_key(best_match.query, best_match.context_hash)
                    if match_key in self._cache:
                        self._cache.move_to_end(match_key)
                    
                    best_match.hit_count += 1
                    self.stats['hits'] += 1
                    self.stats['semantic_hits'] += 1
                    logger.info(f"✅ Cache SEMANTIC HIT: '{query[:30]}...' ≈ '{best_match.query[:30]}...' (sim={best_similarity:.3f})")
                    return best_match.response
        
        self.stats['misses'] += 1
        return None
    
    def put(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Cache a response.
        
        Args:
            query: User query
            response: LLM response
            context: Query context
            language: Response language
            metadata: Additional metadata
        """
        if not response or len(response) < 10:
            return  # Don't cache empty/tiny responses
        
        context_hash = self._hash_context(context)
        cache_key = self._get_cache_key(query, context_hash)
        
        # Evict if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_lru(max(1, self.max_entries // 10))
        
        # Generate embedding
        embedding = self._get_embedding(query)
        
        # Create entry
        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            timestamp=time.time(),
            language=language,
            context_hash=context_hash,
            metadata=metadata or {}
        )
        
        # Store
        self._cache[cache_key] = entry
        
        # Update embedding index
        if embedding is not None:
            self._embedding_index.append((cache_key, embedding))
        
        logger.debug(f"📝 Cached response for: '{query[:30]}...'")
    
    def invalidate(self, query: str, context: Optional[Dict[str, Any]] = None):
        """Invalidate a specific cache entry"""
        context_hash = self._hash_context(context)
        cache_key = self._get_cache_key(query, context_hash)
        
        if cache_key in self._cache:
            del self._cache[cache_key]
            self._embedding_index = [
                (k, e) for k, e in self._embedding_index if k != cache_key
            ]
            logger.debug(f"🗑️ Invalidated cache for: '{query[:30]}...'")
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._embedding_index.clear()
        logger.info("🗑️ Semantic cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(total_requests, 1)
        
        return {
            'total_entries': len(self._cache),
            'max_entries': self.max_entries,
            'total_requests': total_requests,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': round(hit_rate * 100, 1),
            'semantic_hits': self.stats['semantic_hits'],
            'exact_hits': self.stats['exact_hits'],
            'evictions': self.stats['evictions'],
            'expirations': self.stats['expirations'],
            'embeddings_available': self._embeddings_available
        }


# Global instance
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create global semantic cache instance"""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache
