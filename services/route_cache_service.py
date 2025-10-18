#!/usr/bin/env python3
"""
Route Cache Service for POI-Enhanced Route Planning
Implements multi-layer caching for performance optimization
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    ttl_seconds: int
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return (datetime.now() - self.timestamp).total_seconds()


class RouteCacheService:
    """
    Multi-layer caching service for route planning optimization
    
    Cache Layers:
    1. POI Query Cache (30 min TTL) - Spatial queries
    2. ML Prediction Cache (15 min TTL) - Crowding/travel time
    3. Distance Calculation Cache (60 min TTL) - Haversine distances
    4. Route Cache (10 min TTL) - Complete routes
    """
    
    def __init__(self, enable_stats: bool = True):
        """Initialize cache service"""
        self.poi_query_cache: Dict[str, CacheEntry] = {}
        self.ml_prediction_cache: Dict[str, CacheEntry] = {}
        self.distance_cache: Dict[str, CacheEntry] = {}
        self.route_cache: Dict[str, CacheEntry] = {}
        
        # Cache TTLs (seconds)
        self.POI_QUERY_TTL = 1800  # 30 minutes
        self.ML_PREDICTION_TTL = 900  # 15 minutes
        self.DISTANCE_TTL = 3600  # 60 minutes
        self.ROUTE_TTL = 600  # 10 minutes
        
        # Statistics
        self.enable_stats = enable_stats
        self.stats = {
            'poi_query': {'hits': 0, 'misses': 0, 'evictions': 0},
            'ml_prediction': {'hits': 0, 'misses': 0, 'evictions': 0},
            'distance': {'hits': 0, 'misses': 0, 'evictions': 0},
            'route': {'hits': 0, 'misses': 0, 'evictions': 0}
        }
        
        # Cache size limits
        self.MAX_POI_CACHE_SIZE = 1000
        self.MAX_ML_CACHE_SIZE = 500
        self.MAX_DISTANCE_CACHE_SIZE = 5000
        self.MAX_ROUTE_CACHE_SIZE = 100
        
        logger.info("✅ Route Cache Service initialized with multi-layer caching")
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate consistent cache key from data"""
        # Sort keys for consistency
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _evict_expired(self, cache: Dict[str, CacheEntry], cache_type: str) -> int:
        """Remove expired entries from cache"""
        expired_keys = [k for k, v in cache.items() if v.is_expired()]
        for key in expired_keys:
            del cache[key]
        
        if expired_keys and self.enable_stats:
            self.stats[cache_type]['evictions'] += len(expired_keys)
        
        return len(expired_keys)
    
    def _enforce_size_limit(self, cache: Dict[str, CacheEntry], max_size: int, cache_type: str):
        """Enforce cache size limit using LRU eviction"""
        if len(cache) <= max_size:
            return
        
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(cache.items(), key=lambda x: x[1].timestamp)
        
        # Remove oldest entries
        num_to_remove = len(cache) - max_size
        for key, _ in sorted_entries[:num_to_remove]:
            del cache[key]
        
        if self.enable_stats:
            self.stats[cache_type]['evictions'] += num_to_remove
    
    # ==================== POI Query Cache ====================
    
    def cache_poi_query(
        self, 
        center_lat: float, 
        center_lon: float, 
        radius_km: float,
        categories: Optional[List[str]],
        result: List[Any]
    ) -> str:
        """Cache POI spatial query result"""
        cache_key = self._generate_cache_key({
            'type': 'poi_query',
            'lat': round(center_lat, 4),
            'lon': round(center_lon, 4),
            'radius': radius_km,
            'categories': sorted(categories) if categories else []
        })
        
        self.poi_query_cache[cache_key] = CacheEntry(
            key=cache_key,
            value=result,
            timestamp=datetime.now(),
            ttl_seconds=self.POI_QUERY_TTL
        )
        
        self._enforce_size_limit(self.poi_query_cache, self.MAX_POI_CACHE_SIZE, 'poi_query')
        
        return cache_key
    
    def get_cached_poi_query(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        categories: Optional[List[str]] = None
    ) -> Optional[List[Any]]:
        """Retrieve cached POI query result"""
        self._evict_expired(self.poi_query_cache, 'poi_query')
        
        cache_key = self._generate_cache_key({
            'type': 'poi_query',
            'lat': round(center_lat, 4),
            'lon': round(center_lon, 4),
            'radius': radius_km,
            'categories': sorted(categories) if categories else []
        })
        
        if cache_key in self.poi_query_cache:
            entry = self.poi_query_cache[cache_key]
            if not entry.is_expired():
                entry.hits += 1
                if self.enable_stats:
                    self.stats['poi_query']['hits'] += 1
                logger.debug(f"✅ POI query cache HIT (age: {entry.age_seconds():.1f}s)")
                return entry.value
        
        if self.enable_stats:
            self.stats['poi_query']['misses'] += 1
        
        return None
    
    # ==================== ML Prediction Cache ====================
    
    def cache_ml_prediction(
        self,
        poi_id: str,
        prediction_type: str,  # 'crowding' or 'travel_time'
        datetime_key: datetime,
        result: Any
    ) -> str:
        """Cache ML prediction result"""
        cache_key = self._generate_cache_key({
            'type': 'ml_prediction',
            'poi_id': poi_id,
            'prediction_type': prediction_type,
            'datetime': datetime_key.isoformat()
        })
        
        self.ml_prediction_cache[cache_key] = CacheEntry(
            key=cache_key,
            value=result,
            timestamp=datetime.now(),
            ttl_seconds=self.ML_PREDICTION_TTL
        )
        
        self._enforce_size_limit(self.ml_prediction_cache, self.MAX_ML_CACHE_SIZE, 'ml_prediction')
        
        return cache_key
    
    def get_cached_ml_prediction(
        self,
        poi_id: str,
        prediction_type: str,
        datetime_key: datetime
    ) -> Optional[Any]:
        """Retrieve cached ML prediction"""
        self._evict_expired(self.ml_prediction_cache, 'ml_prediction')
        
        cache_key = self._generate_cache_key({
            'type': 'ml_prediction',
            'poi_id': poi_id,
            'prediction_type': prediction_type,
            'datetime': datetime_key.isoformat()
        })
        
        if cache_key in self.ml_prediction_cache:
            entry = self.ml_prediction_cache[cache_key]
            if not entry.is_expired():
                entry.hits += 1
                if self.enable_stats:
                    self.stats['ml_prediction']['hits'] += 1
                logger.debug(f"✅ ML prediction cache HIT (age: {entry.age_seconds():.1f}s)")
                return entry.value
        
        if self.enable_stats:
            self.stats['ml_prediction']['misses'] += 1
        
        return None
    
    # ==================== Distance Cache ====================
    
    def cache_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        distance_km: float
    ) -> str:
        """Cache calculated distance"""
        # Use rounded coordinates for cache key (to group nearby points)
        cache_key = self._generate_cache_key({
            'type': 'distance',
            'lat1': round(lat1, 3),
            'lon1': round(lon1, 3),
            'lat2': round(lat2, 3),
            'lon2': round(lon2, 3)
        })
        
        self.distance_cache[cache_key] = CacheEntry(
            key=cache_key,
            value=distance_km,
            timestamp=datetime.now(),
            ttl_seconds=self.DISTANCE_TTL
        )
        
        self._enforce_size_limit(self.distance_cache, self.MAX_DISTANCE_CACHE_SIZE, 'distance')
        
        return cache_key
    
    def get_cached_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> Optional[float]:
        """Retrieve cached distance"""
        cache_key = self._generate_cache_key({
            'type': 'distance',
            'lat1': round(lat1, 3),
            'lon1': round(lon1, 3),
            'lat2': round(lat2, 3),
            'lon2': round(lon2, 3)
        })
        
        if cache_key in self.distance_cache:
            entry = self.distance_cache[cache_key]
            if not entry.is_expired():
                entry.hits += 1
                if self.enable_stats:
                    self.stats['distance']['hits'] += 1
                return entry.value
        
        if self.enable_stats:
            self.stats['distance']['misses'] += 1
        
        return None
    
    # ==================== Route Cache ====================
    
    def cache_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        preferences: Dict[str, Any],
        route_result: Dict[str, Any]
    ) -> str:
        """Cache complete route result"""
        cache_key = self._generate_cache_key({
            'type': 'route',
            'start_lat': round(start_lat, 3),
            'start_lon': round(start_lon, 3),
            'end_lat': round(end_lat, 3),
            'end_lon': round(end_lon, 3),
            'preferences': preferences
        })
        
        self.route_cache[cache_key] = CacheEntry(
            key=cache_key,
            value=route_result,
            timestamp=datetime.now(),
            ttl_seconds=self.ROUTE_TTL
        )
        
        self._enforce_size_limit(self.route_cache, self.MAX_ROUTE_CACHE_SIZE, 'route')
        
        return cache_key
    
    def get_cached_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        preferences: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached route"""
        self._evict_expired(self.route_cache, 'route')
        
        cache_key = self._generate_cache_key({
            'type': 'route',
            'start_lat': round(start_lat, 3),
            'start_lon': round(start_lon, 3),
            'end_lat': round(end_lat, 3),
            'end_lon': round(end_lon, 3),
            'preferences': preferences
        })
        
        if cache_key in self.route_cache:
            entry = self.route_cache[cache_key]
            if not entry.is_expired():
                entry.hits += 1
                if self.enable_stats:
                    self.stats['route']['hits'] += 1
                logger.info(f"✅ Route cache HIT (age: {entry.age_seconds():.1f}s)")
                return entry.value
        
        if self.enable_stats:
            self.stats['route']['misses'] += 1
        
        return None
    
    # ==================== Cache Management ====================
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.poi_query_cache.clear()
        self.ml_prediction_cache.clear()
        self.distance_cache.clear()
        self.route_cache.clear()
        logger.info("🧹 All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = (
            len(self.poi_query_cache) +
            len(self.ml_prediction_cache) +
            len(self.distance_cache) +
            len(self.route_cache)
        )
        
        stats = {
            'total_entries': total_entries,
            'cache_sizes': {
                'poi_query': len(self.poi_query_cache),
                'ml_prediction': len(self.ml_prediction_cache),
                'distance': len(self.distance_cache),
                'route': len(self.route_cache)
            },
            'performance': {}
        }
        
        # Calculate hit rates
        for cache_type in ['poi_query', 'ml_prediction', 'distance', 'route']:
            hits = self.stats[cache_type]['hits']
            misses = self.stats[cache_type]['misses']
            total = hits + misses
            
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            stats['performance'][cache_type] = {
                'hits': hits,
                'misses': misses,
                'hit_rate_percent': round(hit_rate, 2),
                'evictions': self.stats[cache_type]['evictions']
            }
        
        return stats
    
    def print_cache_stats(self):
        """Print formatted cache statistics"""
        stats = self.get_cache_stats()
        
        print("\n" + "="*60)
        print("📊 ROUTE CACHE SERVICE STATISTICS")
        print("="*60)
        
        print(f"\n📦 Cache Sizes:")
        for cache_type, size in stats['cache_sizes'].items():
            print(f"  - {cache_type}: {size} entries")
        
        print(f"\n🎯 Performance Metrics:")
        for cache_type, metrics in stats['performance'].items():
            print(f"\n  {cache_type.upper()}:")
            print(f"    Hits: {metrics['hits']}")
            print(f"    Misses: {metrics['misses']}")
            print(f"    Hit Rate: {metrics['hit_rate_percent']}%")
            print(f"    Evictions: {metrics['evictions']}")
        
        print("\n" + "="*60 + "\n")


# Singleton instance
_cache_service_instance = None

def get_cache_service() -> RouteCacheService:
    """Get singleton cache service instance"""
    global _cache_service_instance
    if _cache_service_instance is None:
        _cache_service_instance = RouteCacheService()
    return _cache_service_instance


if __name__ == "__main__":
    # Test cache service
    logging.basicConfig(level=logging.INFO)
    
    cache = RouteCacheService()
    
    # Test POI query caching
    print("\n🧪 Testing POI Query Cache...")
    cache.cache_poi_query(41.0082, 28.9784, 2.0, ['museum', 'mosque'], ['poi1', 'poi2'])
    result = cache.get_cached_poi_query(41.0082, 28.9784, 2.0, ['museum', 'mosque'])
    print(f"Cached result: {result}")
    
    # Test distance caching
    print("\n🧪 Testing Distance Cache...")
    cache.cache_distance(41.0082, 28.9784, 41.0, 29.0, 2.5)
    dist = cache.get_cached_distance(41.0082, 28.9784, 41.0, 29.0)
    print(f"Cached distance: {dist} km")
    
    # Print stats
    cache.print_cache_stats()
