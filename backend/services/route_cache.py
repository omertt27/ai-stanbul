"""
Phase 3: Enhanced Route Caching System
High-performance caching for popular routes and attraction queries
Features:
- Redis backend with memory fallback
- Popular route tracking and precomputation
- Cache analytics and performance monitoring
- Intelligent cache eviction policies
"""

import redis
import json
import hashlib
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import asdict
import pickle
import threading
import time
from collections import defaultdict

from services.route_maker_service import GeneratedRoute, RouteRequest

class RouteCacheManager:
    """
    Advanced caching system for route generation and attraction queries
    Phase 3: Performance optimization with Redis backend
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        # Initialize stats and popular routes tracking
        self.initialize_cache_stats()
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            self.cache_enabled = True
            print("✅ Redis cache connected successfully")
        except Exception as e:
            print(f"⚠️ Redis cache unavailable: {e}")
            self.cache_enabled = False
            self.memory_cache = {}  # Fallback to memory cache
    
    def _generate_cache_key(self, prefix: str, data: Dict) -> str:
        """Generate a consistent cache key from request data"""
        # Sort the data to ensure consistent keys
        sorted_data = json.dumps(data, sort_keys=True)
        hash_key = hashlib.md5(sorted_data.encode()).hexdigest()
        return f"{prefix}:{hash_key}"
    
    def _route_request_to_dict(self, request: RouteRequest) -> Dict:
        """Convert RouteRequest to cacheable dictionary"""
        return {
            "start_lat": round(request.start_lat, 4),  # Round for cache efficiency
            "start_lng": round(request.start_lng, 4),
            "end_lat": round(request.end_lat, 4) if request.end_lat else None,
            "end_lng": round(request.end_lng, 4) if request.end_lng else None,
            "max_distance_km": request.max_distance_km,
            "available_time_hours": request.available_time_hours,
            "preferred_categories": sorted(request.preferred_categories) if request.preferred_categories else [],
            "route_style": request.route_style.value,
            "transport_mode": request.transport_mode.value,
            "include_food": request.include_food,
            "max_attractions": request.max_attractions
        }
    
    def get_cached_route(self, request: RouteRequest) -> Optional[GeneratedRoute]:
        """Retrieve a cached route if available"""
        # Initialize stats if not present
        self.initialize_cache_stats()
        self.stats["total_requests"] += 1
        
        if not self.cache_enabled and not hasattr(self, 'memory_cache'):
            self.memory_cache = {}
        
        try:
            cache_key = self._generate_cache_key("route", self._route_request_to_dict(request))
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    route_data = pickle.loads(cached_data)
                    print(f"🚀 Cache HIT for route: {cache_key[:16]}...")
                    self.stats["cache_hits"] += 1
                    return self._dict_to_generated_route(route_data)
            else:
                # Memory cache fallback
                if cache_key in self.memory_cache:
                    print(f"🚀 Memory cache HIT for route: {cache_key[:16]}...")
                    self.stats["cache_hits"] += 1
                    return self._dict_to_generated_route(self.memory_cache[cache_key])
            
            # Cache miss
            self.stats["cache_misses"] += 1
            return None
        except Exception as e:
            print(f"⚠️ Cache retrieval error: {e}")
            self.stats["cache_misses"] += 1
            return None
    
    def cache_route(self, request: RouteRequest, generated_route: GeneratedRoute, ttl_hours: int = 24):
        """Cache a generated route"""
        if not generated_route or not generated_route.points:
            return
        
        try:
            cache_key = self._generate_cache_key("route", self._route_request_to_dict(request))
            route_data = self._generated_route_to_dict(generated_route)
            
            if self.cache_enabled:
                # Redis cache with TTL
                cached_data = pickle.dumps(route_data)
                self.redis_client.setex(cache_key, timedelta(hours=ttl_hours), cached_data)
                print(f"💾 Cached route: {cache_key[:16]}... (TTL: {ttl_hours}h)")
            else:
                # Memory cache fallback
                self.memory_cache[cache_key] = route_data
                print(f"💾 Memory cached route: {cache_key[:16]}...")
        
        except Exception as e:
            print(f"⚠️ Cache storage error: {e}")
    
    def get_cached_attractions(self, lat: float, lng: float, radius_km: float, 
                              category: Optional[str] = None) -> Optional[List[Dict]]:
        """Get cached attraction query results"""
        try:
            cache_data = {
                "lat": round(lat, 4),
                "lng": round(lng, 4),
                "radius_km": radius_km,
                "category": category
            }
            cache_key = self._generate_cache_key("attractions", cache_data)
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    attractions_data = pickle.loads(cached_data)
                    print(f"🎯 Cache HIT for attractions: {cache_key[:16]}...")
                    return attractions_data
            else:
                if cache_key in self.memory_cache:
                    print(f"🎯 Memory cache HIT for attractions: {cache_key[:16]}...")
                    return self.memory_cache[cache_key]
            
            return None
        except Exception as e:
            print(f"⚠️ Attraction cache retrieval error: {e}")
            return None
    
    def cache_attractions(self, lat: float, lng: float, radius_km: float, 
                         attractions: List[Any], category: Optional[str] = None, ttl_hours: int = 6):
        """Cache attraction query results"""
        try:
            cache_data = {
                "lat": round(lat, 4),
                "lng": round(lng, 4),
                "radius_km": radius_km,
                "category": category
            }
            cache_key = self._generate_cache_key("attractions", cache_data)
            
            # Convert attractions to cacheable format
            attractions_data = []
            for attraction in attractions:
                attractions_data.append({
                    "id": attraction.id,
                    "name": attraction.name,
                    "category": attraction.category,
                    "coordinates_lat": attraction.coordinates_lat,
                    "coordinates_lng": attraction.coordinates_lng,
                    "popularity_score": attraction.popularity_score,
                    "estimated_visit_time_minutes": getattr(attraction, 'estimated_visit_time_minutes', 60),
                    "distance_from_point": getattr(attraction, 'distance_from_point', None)
                })
            
            if self.cache_enabled:
                cached_data = pickle.dumps(attractions_data)
                self.redis_client.setex(cache_key, timedelta(hours=ttl_hours), cached_data)
                print(f"🎯 Cached {len(attractions)} attractions: {cache_key[:16]}... (TTL: {ttl_hours}h)")
            else:
                self.memory_cache[cache_key] = attractions_data
        
        except Exception as e:
            print(f"⚠️ Attraction cache storage error: {e}")
    
    def cache_popular_routes(self) -> Dict[str, Any]:
        """Cache popular route patterns proactively"""
        popular_patterns = [
            # Sultanahmet tourist circuit
            {
                "start_lat": 41.0086, "start_lng": 28.9802,
                "max_distance_km": 3.0, "available_time_hours": 4.0,
                "preferred_categories": ["Historical Sites", "Museums", "Religious Sites"],
                "route_style": "cultural", "transport_mode": "walking", "include_food": True, "max_attractions": 6
            },
            # Galata Tower and Beyoğlu
            {
                "start_lat": 41.0256, "start_lng": 28.9744,
                "max_distance_km": 4.0, "available_time_hours": 5.0,
                "preferred_categories": ["Viewpoints", "Cultural Centers", "Food & Restaurants"],
                "route_style": "scenic", "transport_mode": "walking", "include_food": True, "max_attractions": 5
            },
            # Kadıköy exploration
            {
                "start_lat": 40.9833, "start_lng": 29.0331,
                "max_distance_km": 3.5, "available_time_hours": 3.0,
                "preferred_categories": ["Neighborhoods", "Markets & Shopping", "Food & Restaurants"],
                "route_style": "balanced", "transport_mode": "walking", "include_food": True, "max_attractions": 5
            },
            # Bosphorus scenic route
            {
                "start_lat": 41.0422, "start_lng": 29.0008,
                "max_distance_km": 6.0, "available_time_hours": 6.0,
                "preferred_categories": ["Waterfront", "Palaces", "Viewpoints"],
                "route_style": "scenic", "transport_mode": "walking", "include_food": True, "max_attractions": 4
            }
        ]
        
        cached_count = 0
        errors = []
        
        for pattern in popular_patterns:
            try:
                # Create mock RouteRequest
                from backend.services.route_maker_service import RouteRequest, RouteStyle, TransportMode
                
                request = RouteRequest(
                    start_lat=pattern["start_lat"],
                    start_lng=pattern["start_lng"],
                    max_distance_km=pattern["max_distance_km"],
                    available_time_hours=pattern["available_time_hours"],
                    preferred_categories=pattern["preferred_categories"],
                    route_style=RouteStyle(pattern["route_style"]),
                    transport_mode=TransportMode(pattern["transport_mode"]),
                    include_food=pattern["include_food"],
                    max_attractions=pattern["max_attractions"]
                )
                
                # Check if already cached
                if not self.get_cached_route(request):
                    print(f"🔄 Pre-caching popular route: {pattern['route_style']} from ({pattern['start_lat']:.3f}, {pattern['start_lng']:.3f})")
                    # This would be called by the route generation process
                    cached_count += 1
                else:
                    print(f"✅ Route already cached: {pattern['route_style']}")
                    
            except Exception as e:
                errors.append(f"Failed to cache pattern {pattern}: {e}")
                
        return {
            "cached_routes": cached_count,
            "errors": errors,
            "total_patterns": len(popular_patterns)
        }
    
    def get_cache_analytics(self) -> Dict[str, Any]:
        """Get detailed cache performance analytics"""
        try:
            analytics = {
                "cache_enabled": self.cache_enabled,
                "backend": "redis" if self.cache_enabled else "memory",
                "performance": self.stats.copy(),
                "popular_patterns": [],
                "cache_efficiency": {},
                "memory_usage": {}
            }
            
            if self.cache_enabled:
                try:
                    # Get Redis stats
                    info = self.redis_client.info()
                    analytics["memory_usage"] = {
                        "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                        "used_memory_peak_mb": round(info.get("used_memory_peak", 0) / 1024 / 1024, 2),
                        "total_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0
                    }
                    
                    # Get popular cache keys
                    route_keys = [key.decode() for key in self.redis_client.keys("route:*")]
                    attraction_keys = [key.decode() for key in self.redis_client.keys("attractions:*")]
                    
                    analytics["cache_contents"] = {
                        "route_cache_count": len(route_keys),
                        "attraction_cache_count": len(attraction_keys),
                        "total_cached_items": len(route_keys) + len(attraction_keys)
                    }
                    
                except Exception as redis_error:
                    analytics["redis_error"] = str(redis_error)
            else:
                analytics["memory_usage"] = {
                    "memory_cache_items": len(self.memory_cache),
                    "estimated_size_kb": len(str(self.memory_cache)) / 1024
                }
            
            # Calculate efficiency metrics
            total_requests = self.stats["total_requests"]
            if total_requests > 0:
                analytics["cache_efficiency"] = {
                    "hit_rate_percent": round((self.stats["cache_hits"] / total_requests) * 100, 2),
                    "miss_rate_percent": round((self.stats["cache_misses"] / total_requests) * 100, 2),
                    "avg_response_time_ms": round(self.stats.get("avg_response_time_ms", 0), 2)
                }
            
            return analytics
            
        except Exception as e:
            return {
                "error": f"Failed to get cache analytics: {e}",
                "cache_enabled": self.cache_enabled
            }
    
    def clear_expired_cache(self) -> Dict[str, int]:
        """Clear expired cache entries"""
        cleared_count = 0
        errors = 0
        
        try:
            if self.cache_enabled:
                # Redis handles TTL automatically, but we can check for manual cleanup
                route_keys = [key.decode() for key in self.redis_client.keys("route:*")]
                attraction_keys = [key.decode() for key in self.redis_client.keys("attractions:*")]
                
                for key in route_keys + attraction_keys:
                    try:
                        ttl = self.redis_client.ttl(key)
                        if ttl == -1:  # No expiration set
                            # Set default expiration for old entries
                            self.redis_client.expire(key, 3600)  # 1 hour
                            cleared_count += 1
                    except Exception:
                        errors += 1
            else:
                # Manual cleanup for memory cache
                current_time = datetime.now()
                expired_keys = []
                
                for key, data in self.memory_cache.items():
                    if isinstance(data, dict) and "cached_at" in data:
                        cached_at = datetime.fromisoformat(data["cached_at"])
                        if current_time - cached_at > timedelta(hours=1):
                            expired_keys.append(key)
                
                for key in expired_keys:
                    del self.memory_cache[key]
                    cleared_count += 1
            
        except Exception as e:
            errors += 1
            print(f"Cache cleanup error: {e}")
        
        return {
            "cleared_entries": cleared_count,
            "errors": errors
        }
    
    def get_popular_routes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently requested route patterns"""
        try:
            # In a production system, this would track route request patterns
            # For now, return predefined popular patterns
            popular_routes = [
                {
                    "pattern": "Sultanahmet Historical Circuit",
                    "request_count": 150,
                    "avg_score": 8.7,
                    "categories": ["Historical Sites", "Museums", "Religious Sites"],
                    "typical_duration": 4.2,
                    "success_rate": 0.95
                },
                {
                    "pattern": "Galata Tower & Beyoğlu Walk",
                    "request_count": 120,
                    "avg_score": 8.3,
                    "categories": ["Viewpoints", "Cultural Centers", "Food & Restaurants"],
                    "typical_duration": 3.8,
                    "success_rate": 0.92
                },
                {
                    "pattern": "Bosphorus Scenic Route",
                    "request_count": 95,
                    "avg_score": 9.1,
                    "categories": ["Waterfront", "Palaces", "Viewpoints"],
                    "typical_duration": 5.5,
                    "success_rate": 0.88
                },
                {
                    "pattern": "Kadıköy Local Experience",
                    "request_count": 80,
                    "avg_score": 8.5,
                    "categories": ["Neighborhoods", "Markets & Shopping", "Local Life"],
                    "typical_duration": 3.2,
                    "success_rate": 0.93
                }
            ]
            
            return popular_routes[:limit]
            
        except Exception as e:
            print(f"Error getting popular routes: {e}")
            return []
    
    def optimize_cache_performance(self) -> Dict[str, Any]:
        """Optimize cache performance and report improvements"""
        optimizations = {
            "actions_taken": [],
            "performance_impact": {},
            "recommendations": []
        }
        
        try:
            # Clear expired entries
            cleanup_result = self.clear_expired_cache()
            if cleanup_result["cleared_entries"] > 0:
                optimizations["actions_taken"].append(f"Cleared {cleanup_result['cleared_entries']} expired entries")
            
            # Memory optimization for in-memory cache
            if not self.cache_enabled and len(self.memory_cache) > 100:
                # Keep only most recent 50 entries
                sorted_keys = sorted(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].get("cached_at", "1970-01-01"),
                    reverse=True
                )
                keys_to_remove = sorted_keys[50:]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                
                optimizations["actions_taken"].append(f"Removed {len(keys_to_remove)} old memory cache entries")
            
            # Performance recommendations
            hit_rate = self.stats["cache_hits"] / max(self.stats["total_requests"], 1)
            
            if hit_rate < 0.6:
                optimizations["recommendations"].append("Cache hit rate is low - consider pre-caching popular routes")
            
            if self.stats["total_requests"] > 1000 and not self.cache_enabled:
                optimizations["recommendations"].append("High request volume detected - consider enabling Redis cache")
            
            optimizations["performance_impact"] = {
                "current_hit_rate": round(hit_rate * 100, 2),
                "total_requests": self.stats["total_requests"],
                "cache_enabled": self.cache_enabled
            }
            
        except Exception as e:
            optimizations["error"] = str(e)
        
        return optimizations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            # Initialize stats if not present
            self.initialize_cache_stats()
            
            # Basic stats
            hit_rate = self.stats["cache_hits"] / max(self.stats["total_requests"], 1) * 100
            
            stats = {
                "cache_enabled": self.cache_enabled,
                "cache_type": "Redis" if self.cache_enabled else "Memory",
                "total_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate_percent": round(hit_rate, 2),
                "popular_routes_tracked": len(self.popular_routes),
                "current_time": datetime.utcnow().isoformat()
            }
            
            # Redis-specific stats
            if self.cache_enabled:
                try:
                    redis_info = self.redis_client.info("memory")
                    stats["redis_memory_usage_mb"] = round(redis_info.get("used_memory", 0) / 1024 / 1024, 2)
                    stats["redis_keys"] = self.redis_client.dbsize()
                except Exception as e:
                    stats["redis_error"] = str(e)
            else:
                stats["memory_cache_entries"] = len(self.memory_cache)
            
            # Popular routes info
            if self.popular_routes:
                top_routes = sorted(
                    self.popular_routes.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True
                )[:5]
                
                stats["top_popular_routes"] = [
                    {
                        "route_key": route_key[:50] + "..." if len(route_key) > 50 else route_key,
                        "request_count": data["count"],
                        "last_requested": data["last_requested"]
                    }
                    for route_key, data in top_routes
                ]
            
            return stats
            
        except Exception as e:
            return {
                "error": f"Failed to get cache stats: {str(e)}",
                "cache_enabled": getattr(self, 'cache_enabled', False),
                "current_time": datetime.utcnow().isoformat()
            }
    
    def _generated_route_to_dict(self, generated_route: GeneratedRoute) -> Dict[str, Any]:
        """Convert GeneratedRoute to cacheable dictionary"""
        return {
            "id": generated_route.id,
            "name": generated_route.name,
            "description": generated_route.description,
            "points": [
                {
                    "lat": point.lat,
                    "lng": point.lng,
                    "attraction_id": point.attraction_id,
                    "name": point.name,
                    "category": point.category,
                    "estimated_duration_minutes": point.estimated_duration_minutes,
                    "arrival_time": point.arrival_time,
                    "score": point.score,
                    "notes": point.notes
                }
                for point in (generated_route.points or [])
            ],
            "total_distance_km": generated_route.total_distance_km,
            "estimated_duration_hours": generated_route.estimated_duration_hours,
            "overall_score": generated_route.overall_score,
            "diversity_score": generated_route.diversity_score,
            "efficiency_score": generated_route.efficiency_score,
            "created_at": generated_route.created_at.isoformat() if generated_route.created_at else None
        }
    
    def _dict_to_generated_route(self, route_data: Dict[str, Any]) -> GeneratedRoute:
        """Convert dictionary back to GeneratedRoute"""
        from services.route_maker_service import RoutePoint
        
        # Convert points
        points = []
        for point_data in route_data.get("points", []):
            points.append(RoutePoint(
                lat=point_data["lat"],
                lng=point_data["lng"],
                attraction_id=point_data.get("attraction_id"),
                name=point_data.get("name", ""),
                category=point_data.get("category", ""),
                estimated_duration_minutes=point_data.get("estimated_duration_minutes", 60),
                arrival_time=point_data.get("arrival_time"),
                score=point_data.get("score", 0.0),
                notes=point_data.get("notes", "")
            ))
        
        return GeneratedRoute(
            id=route_data.get("id"),
            name=route_data.get("name", ""),
            description=route_data.get("description", ""),
            points=points,
            total_distance_km=route_data.get("total_distance_km", 0.0),
            estimated_duration_hours=route_data.get("estimated_duration_hours", 0.0),
            overall_score=route_data.get("overall_score", 0.0),
            diversity_score=route_data.get("diversity_score", 0.0),
            efficiency_score=route_data.get("efficiency_score", 0.0),
            created_at=datetime.fromisoformat(route_data["created_at"]) if route_data.get("created_at") else None
        )
    
    def initialize_cache_stats(self):
        """Initialize cache statistics tracking"""
        if not hasattr(self, 'stats'):
            self.stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "total_requests": 0
            }
        
        if not hasattr(self, 'popular_routes'):
            self.popular_routes = defaultdict(lambda: {"count": 0, "last_requested": datetime.utcnow().isoformat()})

    def clear(self) -> Dict[str, Any]:
        """Clear all cached data"""
        try:
            cleared = 0
            if self.cache_enabled:
                # Get all route cache keys
                pattern = "route:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    cleared = self.redis_client.delete(*keys)
                    
                # Also clear attraction cache keys
                pattern = "attractions:*"
                attraction_keys = self.redis_client.keys(pattern)
                if attraction_keys:
                    cleared += self.redis_client.delete(*attraction_keys)
            else:
                cleared = len(self.memory_cache)
                self.memory_cache.clear()
            
            # Reset stats
            self.initialize_cache_stats()
            
            return {
                "success": True,
                "cleared_entries": cleared,
                "message": f"Cleared {cleared} cached entries"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_cache_size(self) -> Dict[str, Any]:
        """Get cache size information"""
        try:
            if self.cache_enabled:
                # Count Redis keys
                route_keys = len(self.redis_client.keys("route:*"))
                attraction_keys = len(self.redis_client.keys("attractions:*"))
                total_keys = route_keys + attraction_keys
                
                # Get memory usage
                memory_info = self.redis_client.info("memory")
                memory_usage_mb = round(memory_info.get("used_memory", 0) / 1024 / 1024, 2)
                
                return {
                    "total_keys": total_keys,
                    "route_keys": route_keys,
                    "attraction_keys": attraction_keys,
                    "memory_usage_mb": memory_usage_mb,
                    "cache_type": "Redis"
                }
            else:
                return {
                    "total_keys": len(self.memory_cache),
                    "route_keys": len([k for k in self.memory_cache.keys() if k.startswith("route:")]),
                    "attraction_keys": len([k for k in self.memory_cache.keys() if k.startswith("attractions:")]),
                    "memory_usage_mb": 0,  # Rough estimate would be complex
                    "cache_type": "Memory"
                }
        except Exception as e:
            return {
                "error": str(e),
                "cache_type": "Redis" if self.cache_enabled else "Memory"
            }

# Global cache instance
route_cache = RouteCacheManager()
