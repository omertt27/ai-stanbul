#!/usr/bin/env python3
"""
Route Cache Service - Simple in-memory caching for generated routes
"""

import time
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict

try:
    from services.geo_utilities import geo_utils, cache_utils
    GEO_UTILS_AVAILABLE = True
except ImportError:
    GEO_UTILS_AVAILABLE = False
    # Create minimal fallbacks
    class MockGeoUtils:
        @staticmethod
        def generate_location_hash(lat, lon, precision=4):
            return f"{round(lat, precision)}_{round(lon, precision)}"
    
    class MockCacheUtils:
        @staticmethod
        def generate_route_key(start_lat, start_lon, end_lat=None, end_lon=None, style="balanced", max_distance=5.0):
            start_hash = MockGeoUtils.generate_location_hash(start_lat, start_lon)
            if end_lat and end_lon:
                end_hash = MockGeoUtils.generate_location_hash(end_lat, end_lon)
                return f"route_{start_hash}_to_{end_hash}_{style}_{max_distance}"
            else:
                return f"route_{start_hash}_loop_{style}_{max_distance}"
    
    geo_utils = MockGeoUtils()
    cache_utils = MockCacheUtils()


class RouteCache:
    """Simple in-memory route cache with TTL and size limits"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.access_times: Dict[str, float] = {}
        
    def _generate_cache_key(self, route_request) -> str:
        """Generate cache key from route request"""
        try:
            return cache_utils.generate_route_key(
                route_request.start_lat,
                route_request.start_lng,
                getattr(route_request, 'end_lat', None),
                getattr(route_request, 'end_lng', None),
                route_request.route_style.value if hasattr(route_request.route_style, 'value') else str(route_request.route_style),
                route_request.max_distance_km
            )
        except Exception as e:
            # Fallback to simple hash
            request_str = f"{route_request.start_lat}_{route_request.start_lng}_{route_request.max_distance_km}"
            return hashlib.md5(request_str.encode()).hexdigest()[:16]
    
    def get_cached_route(self, route_request) -> Optional[Any]:
        """Get cached route if available and not expired"""
        cache_key = self._generate_cache_key(route_request)
        
        if cache_key not in self.cache:
            return None
        
        cached_data = self.cache[cache_key]
        
        # Check if expired
        if time.time() > cached_data['expires_at']:
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            return None
        
        # Update access time
        self.access_times[cache_key] = time.time()
        
        # Return the cached route object
        return cached_data['route']
    
    def cache_route(self, route_request, generated_route, ttl: Optional[int] = None) -> bool:
        """Cache a generated route"""
        try:
            cache_key = self._generate_cache_key(route_request)
            
            # Clean cache if at max size
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Store in cache
            expires_at = time.time() + (ttl or self.default_ttl)
            
            self.cache[cache_key] = {
                'route': generated_route,
                'cached_at': time.time(),
                'expires_at': expires_at,
                'request_hash': cache_key
            }
            
            self.access_times[cache_key] = time.time()
            
            return True
            
        except Exception as e:
            print(f"Failed to cache route: {e}")
            return False
    
    def _evict_oldest(self):
        """Evict oldest cached route based on access time"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        if oldest_key in self.cache:
            del self.cache[oldest_key]
        if oldest_key in self.access_times:
            del self.access_times[oldest_key]
    
    def clear_cache(self):
        """Clear all cached routes"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        expired_count = sum(1 for data in self.cache.values() if current_time > data['expires_at'])
        
        return {
            'total_cached': len(self.cache),
            'expired_routes': expired_count,
            'active_routes': len(self.cache) - expired_count,
            'cache_usage': f"{len(self.cache)}/{self.max_size}",
            'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_request_count', 1), 1),
            'oldest_cache_age': min((current_time - data['cached_at']) / 3600 for data in self.cache.values()) if self.cache else 0,
            'geo_utils_available': GEO_UTILS_AVAILABLE
        }
    
    def cleanup_expired(self):
        """Remove expired routes from cache"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.cache.items() 
            if current_time > data['expires_at']
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        return len(expired_keys)


class PopularRouteCache:
    """Cache for popular/pre-computed routes"""
    
    def __init__(self):
        self.popular_routes = {
            'sultanahmet_historic': {
                'name': 'Historic Sultanahmet Tour',
                'description': 'Classic historical route covering main attractions',
                'start_area': 'sultanahmet',
                'duration_hours': 4.0,
                'distance_km': 3.2,
                'attractions': ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace', 'Basilica Cistern'],
                'cached_at': time.time()
            },
            'bosphorus_scenic': {
                'name': 'Bosphorus Scenic Route',
                'description': 'Scenic route with Bosphorus views',
                'start_area': 'galata',
                'duration_hours': 3.5,
                'distance_km': 4.1,
                'attractions': ['Galata Tower', 'Galata Bridge', 'Ortaköy', 'Dolmabahçe Palace'],
                'cached_at': time.time()
            },
            'modern_istanbul': {
                'name': 'Modern Istanbul Experience',
                'description': 'Contemporary culture and nightlife',
                'start_area': 'taksim',
                'duration_hours': 3.0,
                'distance_km': 2.8,
                'attractions': ['Taksim Square', 'İstiklal Street', 'Galata Tower', 'Karaköy'],
                'cached_at': time.time()
            }
        }
    
    def get_popular_route(self, area: str, style: str = 'balanced') -> Optional[Dict]:
        """Get a popular pre-computed route for an area"""
        area_lower = area.lower()
        
        # Simple matching logic
        if 'sultanahmet' in area_lower or 'historic' in style.lower():
            return self.popular_routes.get('sultanahmet_historic')
        elif 'galata' in area_lower or 'scenic' in style.lower():
            return self.popular_routes.get('bosphorus_scenic')
        elif 'taksim' in area_lower or 'modern' in area_lower:
            return self.popular_routes.get('modern_istanbul')
        
        return None
    
    def get_all_popular_routes(self) -> Dict[str, Dict]:
        """Get all popular routes"""
        return self.popular_routes.copy()


# Global cache instances
route_cache = RouteCache(max_size=500, default_ttl=1800)  # 30 minutes TTL
popular_cache = PopularRouteCache()

# Cleanup task - in production this would be a background task
def cleanup_caches():
    """Cleanup expired cache entries"""
    expired_count = route_cache.cleanup_expired()
    print(f"🧹 Cleaned up {expired_count} expired routes from cache")
    return expired_count
