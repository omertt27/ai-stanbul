"""
TTL Cache - Time-to-Live cache with automatic cleanup
Prevents memory leaks by automatically expiring old entries
"""

from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class TTLCache:
    """
    Time-to-live cache with automatic cleanup and LRU eviction.
    
    Features:
    - Automatic expiration based on TTL
    - LRU eviction when max_size is reached
    - Thread-safe operations
    - Periodic cleanup of expired entries
    
    Usage:
        cache = TTLCache(max_size=1000, ttl_minutes=60)
        cache.set("user_123", user_profile)
        profile = cache.get("user_123")
    """
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 60):
        """
        Initialize TTL cache
        
        Args:
            max_size: Maximum number of entries (LRU eviction when exceeded)
            ttl_minutes: Time-to-live in minutes
        """
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.lock = Lock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
        
        logger.info(f"✅ TTLCache initialized: max_size={max_size}, ttl={ttl_minutes}min")
    
    def set(self, key: str, value: Any) -> None:
        """
        Add or update entry with automatic cleanup
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Cleanup expired entries first
            self._cleanup_expired()
            
            # Check if we need to evict (LRU)
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest entry
                oldest_key, _ = self.cache.popitem(last=False)
                if oldest_key in self.timestamps:
                    del self.timestamps[oldest_key]
                self.stats["evictions"] += 1
                logger.debug(f"🗑️ LRU eviction: {oldest_key}")
            
            # Add/update entry
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
            
            self.cache[key] = value
            self.timestamps[key] = datetime.now()
            logger.debug(f"💾 Cache set: {key}")
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get entry with expiry check
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                logger.debug(f"❌ Cache miss: {key}")
                return default
            
            # Check if expired
            if datetime.now() - self.timestamps[key] > self.ttl:
                # Expired - remove it
                del self.cache[key]
                del self.timestamps[key]
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                logger.debug(f"⏰ Cache expired: {key}")
                return default
            
            # Valid entry - move to end (LRU)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            logger.debug(f"✅ Cache hit: {key}")
            return self.cache[key]
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.timestamps:
                    del self.timestamps[key]
                logger.debug(f"🗑️ Cache delete: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            logger.info("🗑️ Cache cleared")
    
    def _cleanup_expired(self) -> int:
        """
        Remove expired entries
        
        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            k for k, t in self.timestamps.items()
            if now - t > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
            self.stats["expirations"] += 1
        
        if expired_keys:
            logger.debug(f"🗑️ Cleaned up {len(expired_keys)} expired entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_minutes": self.ttl.total_seconds() / 60,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": f"{hit_rate:.1f}%",
                "evictions": self.stats["evictions"],
                "expirations": self.stats["expirations"],
                "total_requests": total_requests
            }
    
    def keys(self) -> List[str]:
        """Get all cache keys"""
        with self.lock:
            return list(self.cache.keys())
    
    def __len__(self) -> int:
        """Get number of entries in cache"""
        return len(self.cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("🧪 Testing TTLCache...")
    
    # Test 1: Basic operations
    print("\n📝 Test 1: Basic set/get")
    cache = TTLCache(max_size=3, ttl_minutes=1)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("nonexistent") is None
    print("✅ Test 1 passed")
    
    # Test 2: LRU eviction
    print("\n📝 Test 2: LRU eviction")
    cache.set("key3", "value3")
    cache.set("key4", "value4")  # Should evict key1
    
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"
    print("✅ Test 2 passed")
    
    # Test 3: TTL expiration
    print("\n📝 Test 3: TTL expiration (waiting 2 seconds...)")
    cache_ttl = TTLCache(max_size=10, ttl_minutes=0.03)  # 2 seconds
    cache_ttl.set("temp", "value")
    assert cache_ttl.get("temp") == "value"
    
    time.sleep(2.5)
    assert cache_ttl.get("temp") is None  # Expired
    print("✅ Test 3 passed")
    
    # Test 4: Statistics
    print("\n📝 Test 4: Statistics")
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    assert stats["size"] == 3
    assert stats["hits"] > 0
    assert stats["misses"] > 0
    print("✅ Test 4 passed")
    
    print("\n🎉 All tests passed!")
    print(f"\nFinal cache state: {cache.get_stats()}")
