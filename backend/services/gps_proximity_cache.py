"""
GPS Proximity Cache Service

Caches GPS-based query results to avoid redundant distance calculations.
Uses grid-based bucketing for efficient spatial queries.

Features:
- Grid cell-based caching (100m x 100m default)
- TTL-based expiration (5 minutes default)
- Redis-backed with in-memory fallback
- Category-aware caching (stations, restaurants, attractions)
- Thread-safe operations
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import threading
import math

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached result with timestamp."""
    data: Any
    timestamp: float
    hit_count: int = 0


class GPSProximityCache:
    """
    Grid-based proximity cache for GPS queries.
    
    Caches results by grid cell to avoid recalculating distances
    for queries within the same small area.
    """
    
    # Grid cell size in degrees (approximately 100m at Istanbul's latitude)
    # At 41°N: 1° latitude ≈ 111km, 1° longitude ≈ 85km
    # So 0.001° ≈ 111m latitude, 85m longitude
    GRID_SIZE_LAT = 0.001  # ~111 meters
    GRID_SIZE_LON = 0.0012  # ~100 meters at Istanbul's latitude
    
    # Cache settings
    DEFAULT_TTL = 300  # 5 minutes
    MAX_MEMORY_ENTRIES = 1000  # Max entries in memory cache
    
    def __init__(self, redis_client=None, ttl: int = None):
        """
        Initialize GPS proximity cache.
        
        Args:
            redis_client: Optional Redis client for distributed caching
            ttl: Cache TTL in seconds (default: 300)
        """
        self.redis = redis_client
        self.ttl = ttl or self.DEFAULT_TTL
        
        # In-memory fallback cache with LRU eviction
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        
        cache_type = "Redis" if self.redis else "In-Memory"
        logger.info(f"🗺️ GPS Proximity Cache initialized ({cache_type}, TTL={self.ttl}s)")
    
    def _get_grid_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert GPS coordinates to grid cell indices.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple of (lat_cell, lon_cell) indices
        """
        lat_cell = int(lat / self.GRID_SIZE_LAT)
        lon_cell = int(lon / self.GRID_SIZE_LON)
        return (lat_cell, lon_cell)
    
    def _make_key(self, lat: float, lon: float, category: str, params: Dict = None) -> str:
        """
        Create a cache key from GPS coordinates, category, and optional params.
        
        Args:
            lat: Latitude
            lon: Longitude
            category: Cache category (e.g., 'nearest_station', 'nearby_restaurants')
            params: Optional additional parameters to include in key
            
        Returns:
            Cache key string
        """
        lat_cell, lon_cell = self._get_grid_cell(lat, lon)
        key = f"gps_cache:{category}:{lat_cell}:{lon_cell}"
        
        if params:
            # Sort params for consistent key generation
            param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            key = f"{key}:{param_str}"
        
        return key
    
    def get(self, lat: float, lon: float, category: str, params: Dict = None) -> Optional[Any]:
        """
        Get cached result for GPS coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            category: Cache category
            params: Optional additional parameters
            
        Returns:
            Cached data or None if not found/expired
        """
        key = self._make_key(lat, lon, category, params)
        
        # Try Redis first
        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    self._stats['hits'] += 1
                    logger.debug(f"📍 GPS cache HIT (Redis): {key}")
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}")
        
        # Fallback to memory cache
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check expiration
                if time.time() - entry.timestamp < self.ttl:
                    # Move to end for LRU
                    self._memory_cache.move_to_end(key)
                    entry.hit_count += 1
                    self._stats['hits'] += 1
                    logger.debug(f"📍 GPS cache HIT (Memory): {key}")
                    return entry.data
                else:
                    # Expired, remove it
                    del self._memory_cache[key]
        
        self._stats['misses'] += 1
        logger.debug(f"📍 GPS cache MISS: {key}")
        return None
    
    def set(self, lat: float, lon: float, category: str, data: Any, params: Dict = None) -> bool:
        """
        Cache a result for GPS coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            category: Cache category
            data: Data to cache
            params: Optional additional parameters
            
        Returns:
            True if cached successfully
        """
        key = self._make_key(lat, lon, category, params)
        
        # Try Redis first
        if self.redis:
            try:
                self.redis.setex(key, self.ttl, json.dumps(data))
                self._stats['sets'] += 1
                logger.debug(f"📍 GPS cache SET (Redis): {key}")
                return True
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Fallback to memory cache
        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._memory_cache) >= self.MAX_MEMORY_ENTRIES:
                self._memory_cache.popitem(last=False)
                self._stats['evictions'] += 1
            
            self._memory_cache[key] = CacheEntry(
                data=data,
                timestamp=time.time()
            )
            self._stats['sets'] += 1
            logger.debug(f"📍 GPS cache SET (Memory): {key}")
            return True
    
    def invalidate(self, lat: float, lon: float, category: str = None, params: Dict = None) -> int:
        """
        Invalidate cache entries for GPS coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            category: Optional category to invalidate (None = all categories)
            params: Optional additional parameters
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        lat_cell, lon_cell = self._get_grid_cell(lat, lon)
        pattern = f"gps_cache:{category or '*'}:{lat_cell}:{lon_cell}"
        
        # Invalidate in Redis
        if self.redis:
            try:
                if category:
                    keys = [self._make_key(lat, lon, category, params)]
                else:
                    keys = list(self.redis.scan_iter(match=f"{pattern}*"))
                
                for key in keys:
                    self.redis.delete(key)
                    count += 1
            except Exception as e:
                logger.warning(f"Redis cache invalidate error: {e}")
        
        # Invalidate in memory cache
        with self._lock:
            keys_to_delete = []
            for key in self._memory_cache:
                if key.startswith(f"gps_cache:{category or ''}") or (
                    not category and f":{lat_cell}:{lon_cell}" in key
                ):
                    if category is None or category in key:
                        keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._memory_cache[key]
                count += 1
        
        logger.debug(f"📍 GPS cache INVALIDATE: {pattern} ({count} entries)")
        return count
    
    def get_nearby_cells(self, lat: float, lon: float, radius_cells: int = 1) -> List[Tuple[int, int]]:
        """
        Get grid cells within a radius of the given coordinates.
        
        Useful for finding cached results in adjacent cells.
        
        Args:
            lat: Center latitude
            lon: Center longitude  
            radius_cells: Number of cells to include in each direction
            
        Returns:
            List of (lat_cell, lon_cell) tuples
        """
        center_lat, center_lon = self._get_grid_cell(lat, lon)
        
        cells = []
        for dlat in range(-radius_cells, radius_cells + 1):
            for dlon in range(-radius_cells, radius_cells + 1):
                cells.append((center_lat + dlat, center_lon + dlon))
        
        return cells
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit/miss/set/eviction counts and hit rate
        """
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0
        
        with self._lock:
            memory_size = len(self._memory_cache)
        
        return {
            **self._stats,
            'hit_rate': round(hit_rate * 100, 2),
            'memory_entries': memory_size,
            'memory_limit': self.MAX_MEMORY_ENTRIES,
            'ttl_seconds': self.ttl
        }
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        
        # Clear Redis
        if self.redis:
            try:
                keys = list(self.redis.scan_iter(match="gps_cache:*"))
                for key in keys:
                    self.redis.delete(key)
                    count += 1
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")
        
        # Clear memory cache
        with self._lock:
            count += len(self._memory_cache)
            self._memory_cache.clear()
        
        logger.info(f"🗑️ GPS cache cleared ({count} entries)")
        return count


# Singleton instance
_gps_cache_instance: Optional[GPSProximityCache] = None


def get_gps_cache(redis_client=None) -> GPSProximityCache:
    """
    Get the singleton GPS proximity cache instance.
    
    Args:
        redis_client: Optional Redis client (only used on first call)
        
    Returns:
        GPSProximityCache instance
    """
    global _gps_cache_instance
    
    if _gps_cache_instance is None:
        _gps_cache_instance = GPSProximityCache(redis_client=redis_client)
    
    return _gps_cache_instance


# Convenience functions for common operations

def cache_nearest_station(lat: float, lon: float, station_id: str, station_name: str, 
                         distance_km: float) -> None:
    """Cache a nearest station result."""
    cache = get_gps_cache()
    cache.set(lat, lon, 'nearest_station', {
        'station_id': station_id,
        'station_name': station_name,
        'distance_km': distance_km
    })


def get_cached_nearest_station(lat: float, lon: float) -> Optional[Dict]:
    """Get cached nearest station result."""
    cache = get_gps_cache()
    return cache.get(lat, lon, 'nearest_station')


def cache_nearby_places(lat: float, lon: float, category: str, places: List[Dict], 
                       radius_km: float = 1.0) -> None:
    """Cache nearby places result."""
    cache = get_gps_cache()
    cache.set(lat, lon, f'nearby_{category}', places, params={'radius': radius_km})


def get_cached_nearby_places(lat: float, lon: float, category: str, 
                            radius_km: float = 1.0) -> Optional[List[Dict]]:
    """Get cached nearby places result."""
    cache = get_gps_cache()
    return cache.get(lat, lon, f'nearby_{category}', params={'radius': radius_km})
