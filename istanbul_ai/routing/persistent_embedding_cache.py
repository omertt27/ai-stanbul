"""
Persistent Embedding Cache - LRU Cache with Disk Persistence

This module provides a persistent LRU cache for neural embeddings with:
- LRU (Least Recently Used) eviction policy
- Disk persistence (save/load)
- Thread-safe operations
- Statistics tracking
- Automatic memory management

Author: Istanbul AI Team
Date: October 31, 2025
"""

import os
import pickle
import numpy as np
import logging
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class PersistentEmbeddingCache:
    """
    LRU cache with disk persistence for neural embeddings
    
    Features:
    - LRU eviction (keeps most frequently used)
    - Disk persistence (survives restarts)
    - Max size limit (prevents memory leaks)
    - Thread-safe operations
    - Automatic save/load
    - Statistics tracking
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        cache_dir: str = 'cache/embeddings',
        auto_save: bool = True,
        save_interval: int = 300
    ):
        """
        Initialize persistent cache
        
        Args:
            max_size: Maximum number of embeddings to cache
            cache_dir: Directory for cache persistence
            auto_save: Enable automatic periodic saves
            save_interval: Seconds between auto-saves
        """
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # LRU cache (OrderedDict maintains insertion order)
        self.cache = OrderedDict()
        
        # Thread lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'saves': 0,
            'loads': 0
        }
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache from disk
        self._load_from_disk()
        
        # Start auto-save thread
        if auto_save:
            self._start_auto_save()
        
        logger.info(f"✅ Persistent cache initialized: {len(self.cache)} embeddings loaded")
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache (LRU: move to end on access)
        
        Args:
            key: Cache key (usually the text)
            
        Returns:
            Embedding array or None if not found
        """
        with self.lock:
            if key in self.cache:
                self.stats['hits'] += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            else:
                self.stats['misses'] += 1
                return None
    
    def set(self, key: str, value: np.ndarray):
        """
        Set embedding in cache (LRU: evict oldest if full)
        
        Args:
            key: Cache key (usually the text)
            value: Embedding array
        """
        with self.lock:
            if key in self.cache:
                # Update existing: move to end
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new entry
                self.cache[key] = value
                
                # Evict oldest if over max_size (LRU)
                if len(self.cache) > self.max_size:
                    evicted_key = next(iter(self.cache))
                    del self.cache[evicted_key]
                    self.stats['evictions'] += 1
    
    def get_many(self, keys: list) -> Dict[str, np.ndarray]:
        """
        Get multiple embeddings (efficient batch operation)
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dict of key -> embedding (only cached items)
        """
        result = {}
        with self.lock:
            for key in keys:
                if key in self.cache:
                    self.stats['hits'] += 1
                    self.cache.move_to_end(key)
                    result[key] = self.cache[key]
                else:
                    self.stats['misses'] += 1
        return result
    
    def set_many(self, items: Dict[str, np.ndarray]):
        """
        Set multiple embeddings (efficient batch operation)
        
        Args:
            items: Dict of key -> embedding
        """
        with self.lock:
            for key, value in items.items():
                if key in self.cache:
                    self.cache.move_to_end(key)
                    self.cache[key] = value
                else:
                    self.cache[key] = value
                    
                    # Evict oldest if over max_size
                    if len(self.cache) > self.max_size:
                        evicted_key = next(iter(self.cache))
                        del self.cache[evicted_key]
                        self.stats['evictions'] += 1
    
    def save_to_disk(self):
        """
        Save cache to disk (pickle format)
        
        Creates checkpoint file with timestamp
        """
        try:
            # Create checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = os.path.join(self.cache_dir, 'embeddings_latest.pkl')
            backup_file = os.path.join(self.cache_dir, f'embeddings_{timestamp}.pkl')
            
            with self.lock:
                # Save to temporary file first (atomic write)
                temp_file = checkpoint_file + '.tmp'
                
                with open(temp_file, 'wb') as f:
                    # Save cache data and metadata
                    data = {
                        'cache': dict(self.cache),
                        'stats': self.stats.copy(),
                        'max_size': self.max_size,
                        'timestamp': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Atomic rename (safer)
                os.replace(temp_file, checkpoint_file)
                
                # Keep backup (optional)
                if len(self.cache) > 100:  # Only backup if significant size
                    import shutil
                    shutil.copy2(checkpoint_file, backup_file)
                
                self.stats['saves'] += 1
            
            logger.info(f"💾 Cache saved: {len(self.cache)} embeddings, {self._get_memory_size_mb():.1f} MB")
            
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
    
    def _load_from_disk(self):
        """
        Load cache from disk (latest checkpoint)
        """
        checkpoint_file = os.path.join(self.cache_dir, 'embeddings_latest.pkl')
        
        if not os.path.exists(checkpoint_file):
            logger.info("📝 No existing cache found, starting fresh")
            return
        
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            # Load cache and metadata
            with self.lock:
                self.cache = OrderedDict(data['cache'])
                
                # Load stats (keep current if not in file)
                if 'stats' in data:
                    self.stats.update(data['stats'])
                
                # Validate max_size
                if len(self.cache) > self.max_size:
                    # Truncate to max_size (keep most recent)
                    items = list(self.cache.items())
                    self.cache = OrderedDict(items[-self.max_size:])
                
                self.stats['loads'] += 1
            
            logger.info(f"✅ Cache loaded: {len(self.cache)} embeddings from {data.get('timestamp', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}")
            logger.info("📝 Starting with empty cache")
            with self.lock:
                self.cache = OrderedDict()
    
    def _start_auto_save(self):
        """
        Start background thread for automatic periodic saves
        """
        def save_periodically():
            import time
            while True:
                time.sleep(self.save_interval)
                self.save_to_disk()
        
        thread = threading.Thread(target=save_periodically, daemon=True)
        thread.start()
        logger.info(f"🔄 Auto-save enabled: every {self.save_interval}s")
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
        logger.info("🗑️  Cache cleared")
    
    def get_size(self) -> int:
        """Get number of cached embeddings"""
        with self.lock:
            return len(self.cache)
    
    def _get_memory_size_mb(self) -> float:
        """
        Estimate memory usage in MB
        
        Returns:
            Approximate memory usage in megabytes
        """
        if not self.cache:
            return 0.0
        
        # Estimate: 768 floats (DistilBERT) * 4 bytes + overhead
        embedding_size = 768 * 4 / 1024 / 1024  # MB per embedding
        num_embeddings = len(self.cache)
        
        # Add overhead for keys and dict structure (~100 bytes per entry)
        overhead = num_embeddings * 100 / 1024 / 1024
        
        return num_embeddings * embedding_size + overhead
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'saves': self.stats['saves'],
                'loads': self.stats['loads'],
                'hit_rate': hit_rate,
                'memory_mb': self._get_memory_size_mb(),
                'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0.0
            }
    
    def reset_stats(self):
        """Reset statistics (keep cache)"""
        with self.lock:
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'saves': self.stats['saves'],
                'loads': self.stats['loads']
            }
        logger.info("📊 Cache stats reset")
    
    def __len__(self):
        """Get cache size"""
        return self.get_size()
    
    def __contains__(self, key):
        """Check if key in cache"""
        with self.lock:
            return key in self.cache
    
    def __del__(self):
        """Save cache on object destruction"""
        try:
            if self.auto_save and len(self.cache) > 0:
                self.save_to_disk()
        except:
            pass


def create_persistent_cache(
    max_size: int = 10000,
    cache_dir: str = 'cache/embeddings',
    auto_save: bool = True
) -> PersistentEmbeddingCache:
    """
    Factory function to create persistent cache
    
    Args:
        max_size: Maximum cache size
        cache_dir: Cache directory
        auto_save: Enable auto-save
        
    Returns:
        PersistentEmbeddingCache instance
    """
    return PersistentEmbeddingCache(
        max_size=max_size,
        cache_dir=cache_dir,
        auto_save=auto_save
    )
