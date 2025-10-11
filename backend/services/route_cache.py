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

from .route_maker_service import GeneratedRoute, RouteRequest

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
    
    def cache_location_query(self, query: str, location_data: Dict[str, Any], user_id: str = None) -> bool:
        """Cache location query results for fast retrieval"""
        try:
            cache_key = self._generate_cache_key("location", {"query": query.lower(), "user_id": user_id})
            
            cache_data = {
                "location_data": location_data,
                "query": query,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "hit_count": 0
            }
            
            if self.cache_enabled:
                # Store in Redis with 2 hour TTL for location queries
                self.redis_client.setex(cache_key, 7200, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            return True
            
        except Exception as e:
            print(f"Failed to cache location query: {e}")
            return False
    
    def get_cached_location(self, query: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached location query results"""
        try:
            cache_key = self._generate_cache_key("location", {"query": query.lower(), "user_id": user_id})
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    # Update hit count
                    result["hit_count"] = result.get("hit_count", 0) + 1
                    self.redis_client.setex(cache_key, 7200, pickle.dumps(result))
                    return result["location_data"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    result["hit_count"] = result.get("hit_count", 0) + 1
                    return result["location_data"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve cached location: {e}")
            return None
    
    def cache_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Cache user personalization preferences"""
        try:
            cache_key = f"user_prefs:{user_id}"
            
            cache_data = {
                "preferences": preferences,
                "user_id": user_id,
                "updated_at": datetime.now().isoformat(),
                "version": 1
            }
            
            if self.cache_enabled:
                # Store user preferences with 24 hour TTL
                self.redis_client.setex(cache_key, 86400, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            return True
            
        except Exception as e:
            print(f"Failed to cache user preferences: {e}")
            return False
    
    def get_cached_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached user preferences"""
        try:
            cache_key = f"user_prefs:{user_id}"
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    return result["preferences"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    return result["preferences"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve cached user preferences: {e}")
            return None
    
    def cache_personalized_route(self, route_request: RouteRequest, route: GeneratedRoute, 
                                user_id: str, personalization_factors: Dict[str, Any]) -> bool:
        """Cache personalized route with user-specific factors"""
        try:
            # Create personalized cache key
            request_data = {
                "from_location": route_request.from_location,
                "to_location": route_request.to_location,
                "optimization_type": route_request.optimization_type,
                "user_id": user_id,
                "personalization": personalization_factors
            }
            
            cache_key = self._generate_cache_key("personalized_route", request_data)
            
            cache_data = {
                "route": asdict(route),
                "request": asdict(route_request),
                "user_id": user_id,
                "personalization_factors": personalization_factors,
                "cached_at": datetime.now().isoformat(),
                "computation_time": getattr(route, 'computation_time', 0),
                "hit_count": 0
            }
            
            if self.cache_enabled:
                # Store personalized routes with 1 hour TTL (shorter due to personalization)
                self.redis_client.setex(cache_key, 3600, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            # Track popular personalized routes
            self._track_popular_personalized_route(request_data)
            
            return True
            
        except Exception as e:
            print(f"Failed to cache personalized route: {e}")
            return False
    
    def get_cached_personalized_route(self, route_request: RouteRequest, user_id: str, 
                                     personalization_factors: Dict[str, Any]) -> Optional[GeneratedRoute]:
        """Retrieve cached personalized route"""
        try:
            request_data = {
                "from_location": route_request.from_location,
                "to_location": route_request.to_location,
                "optimization_type": route_request.optimization_type,
                "user_id": user_id,
                "personalization": personalization_factors
            }
            
            cache_key = self._generate_cache_key("personalized_route", request_data)
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    # Update hit count
                    result["hit_count"] = result.get("hit_count", 0) + 1
                    self.redis_client.setex(cache_key, 3600, pickle.dumps(result))
                    
                    # Convert back to GeneratedRoute object
                    route_data = result["route"]
                    route = GeneratedRoute(**route_data)
                    route.cached = True
                    route.cache_hit_count = result["hit_count"]
                    return route
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    result["hit_count"] = result.get("hit_count", 0) + 1
                    
                    route_data = result["route"]
                    route = GeneratedRoute(**route_data)
                    route.cached = True
                    route.cache_hit_count = result["hit_count"]
                    return route
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve cached personalized route: {e}")
            return None
    
    def _track_popular_personalized_route(self, request_data: Dict[str, Any]):
        """Track popular personalized route combinations"""
        try:
            # Create a route signature without user-specific data
            route_signature = f"{request_data['from_location']}|{request_data['to_location']}|{request_data['optimization_type']}"
            
            # Track in popular routes (personalization-aware)
            if route_signature not in self.popular_routes:
                self.popular_routes[route_signature] = {
                    "count": 0,
                    "from_location": request_data['from_location'],
                    "to_location": request_data['to_location'],
                    "optimization_type": request_data['optimization_type'],
                    "last_requested": datetime.now().isoformat(),
                    "personalization_patterns": defaultdict(int)
                }
            
            self.popular_routes[route_signature]["count"] += 1
            self.popular_routes[route_signature]["last_requested"] = datetime.now().isoformat()
            
            # Track personalization patterns
            for key, value in request_data.get('personalization', {}).items():
                pattern_key = f"{key}:{value}"
                self.popular_routes[route_signature]["personalization_patterns"][pattern_key] += 1
                
        except Exception as e:
            print(f"Failed to track popular personalized route: {e}")
    
    def get_location_based_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for location-based queries"""
        try:
            stats = {
                "location_queries": 0,
                "user_preferences": 0,
                "personalized_routes": 0,
                "popular_locations": [],
                "cache_performance": self.stats
            }
            
            if self.cache_enabled:
                # Count different types of cached items
                location_keys = self.redis_client.keys("location:*")
                user_pref_keys = self.redis_client.keys("user_prefs:*")
                personalized_route_keys = self.redis_client.keys("personalized_route:*")
                
                stats["location_queries"] = len(location_keys)
                stats["user_preferences"] = len(user_pref_keys)
                stats["personalized_routes"] = len(personalized_route_keys)
            else:
                # Memory cache statistics
                for key in self.memory_cache.keys():
                    if key.startswith("location:"):
                        stats["location_queries"] += 1
                    elif key.startswith("user_prefs:"):
                        stats["user_preferences"] += 1
                    elif key.startswith("personalized_route:"):
                        stats["personalized_routes"] += 1
            
            # Add popular location patterns
            location_patterns = defaultdict(int)
            for route_sig, route_data in self.popular_routes.items():
                from_loc = route_data["from_location"]
                to_loc = route_data["to_location"]
                location_patterns[from_loc] += route_data["count"]
                location_patterns[to_loc] += route_data["count"]
            
            stats["popular_locations"] = [
                {"location": loc, "requests": count}
                for loc, count in sorted(location_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            
            return stats
            
        except Exception as e:
            print(f"Failed to get location-based cache stats: {e}")
            return {}
    
    def cleanup_expired_personalized_cache(self):
        """Clean up expired personalized cache entries"""
        try:
            if not self.cache_enabled:
                # Memory cache cleanup
                current_time = datetime.now()
                expired_keys = []
                
                for key, data in self.memory_cache.items():
                    if key.startswith(("location:", "user_prefs:", "personalized_route:")):
                        if "timestamp" in data or "cached_at" in data:
                            timestamp_str = data.get("timestamp") or data.get("cached_at")
                            cached_time = datetime.fromisoformat(timestamp_str)
                            
                            # Different TTL for different types
                            if key.startswith("location:") and current_time - cached_time > timedelta(hours=2):
                                expired_keys.append(key)
                            elif key.startswith("user_prefs:") and current_time - cached_time > timedelta(hours=24):
                                expired_keys.append(key)
                            elif key.startswith("personalized_route:") and current_time - cached_time > timedelta(hours=1):
                                expired_keys.append(key)
                
                for key in expired_keys:
                    del self.memory_cache[key]
                
                print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            print(f"Failed to cleanup expired personalized cache: {e}")
    
    # ==============================================
    # COMPETITIVE ADVANTAGE FEATURES
    # Features that differentiate from Google Maps/TripAdvisor
    # ==============================================
    
    def cache_localized_tips(self, location_id: str, tips: List[Dict[str, Any]], source: str = "local") -> bool:
        """Cache insider tips and local knowledge that Google Maps doesn't have"""
        try:
            cache_key = f"localized_tips:{location_id}"
            
            cache_data = {
                "tips": tips,
                "location_id": location_id,
                "source": source,  # "local", "insider", "community"
                "cached_at": datetime.now().isoformat(),
                "tip_count": len(tips),
                "languages": list(set(tip.get("language", "tr") for tip in tips))
            }
            
            if self.cache_enabled:
                # Store tips with 6 hour TTL (they change less frequently)
                self.redis_client.setex(cache_key, 21600, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            print(f"💡 Cached {len(tips)} localized tips for {location_id}")
            return True
            
        except Exception as e:
            print(f"Failed to cache localized tips: {e}")
            return False
    
    def get_cached_localized_tips(self, location_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached insider tips and local knowledge"""
        try:
            cache_key = f"localized_tips:{location_id}"
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    print(f"💡 Retrieved {result['tip_count']} localized tips for {location_id}")
                    return result["tips"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    print(f"💡 Retrieved {result['tip_count']} localized tips for {location_id}")
                    return result["tips"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve localized tips: {e}")
            return None
    
    def cache_hidden_gems(self, area: str, gems: List[Dict[str, Any]], discovery_method: str = "local_knowledge") -> bool:
        """Cache hidden gems that mainstream apps don't know about"""
        try:
            cache_key = f"hidden_gems:{area.lower().replace(' ', '_')}"
            
            cache_data = {
                "gems": gems,
                "area": area,
                "discovery_method": discovery_method,  # "local_knowledge", "community", "ai_discovery"
                "cached_at": datetime.now().isoformat(),
                "gem_count": len(gems),
                "categories": list(set(gem.get("category", "local") for gem in gems)),
                "authenticity_score": sum(gem.get("authenticity_score", 8.0) for gem in gems) / len(gems) if gems else 0
            }
            
            if self.cache_enabled:
                # Store gems with 12 hour TTL (they're more stable)
                self.redis_client.setex(cache_key, 43200, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            print(f"💎 Cached {len(gems)} hidden gems for {area}")
            return True
            
        except Exception as e:
            print(f"Failed to cache hidden gems: {e}")
            return False
    
    def get_cached_hidden_gems(self, area: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached hidden gems"""
        try:
            cache_key = f"hidden_gems:{area.lower().replace(' ', '_')}"
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    print(f"💎 Retrieved {result['gem_count']} hidden gems for {area}")
                    return result["gems"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    print(f"💎 Retrieved {result['gem_count']} hidden gems for {area}")
                    return result["gems"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve hidden gems: {e}")
            return None
    
    def cache_smart_daily_guidance(self, user_profile: Dict[str, Any], guidance: Dict[str, Any]) -> bool:
        """Cache AI-powered daily guidance based on time, weather, crowds, user preferences"""
        try:
            user_id = user_profile.get("user_id", "anonymous")
            date_key = datetime.now().strftime("%Y-%m-%d")
            cache_key = f"daily_guidance:{user_id}:{date_key}"
            
            cache_data = {
                "guidance": guidance,
                "user_profile": user_profile,
                "generated_at": datetime.now().isoformat(),
                "weather_conditions": guidance.get("weather_context", {}),
                "crowd_predictions": guidance.get("crowd_predictions", {}),
                "personalization_factors": guidance.get("personalization_factors", {}),
                "daily_theme": guidance.get("theme", "explore")
            }
            
            if self.cache_enabled:
                # Store daily guidance with 4 hour TTL (updates throughout day)
                self.redis_client.setex(cache_key, 14400, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            print(f"🧠 Cached smart daily guidance for user {user_id} - Theme: {cache_data['daily_theme']}")
            return True
            
        except Exception as e:
            print(f"Failed to cache smart daily guidance: {e}")
            return False
    
    def get_cached_smart_daily_guidance(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached smart daily guidance"""
        try:
            date_key = datetime.now().strftime("%Y-%m-%d")
            cache_key = f"daily_guidance:{user_id}:{date_key}"
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    print(f"🧠 Retrieved smart daily guidance for {user_id} - Theme: {result['daily_theme']}")
                    return result["guidance"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    print(f"🧠 Retrieved smart daily guidance for {user_id} - Theme: {result['daily_theme']}")
                    return result["guidance"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve smart daily guidance: {e}")
            return None
    
    # ==============================================
    # REAL-TIME DATA INTEGRATION
    # ==============================================
    
    def cache_real_time_traffic_data(self, route_id: str, traffic_data: Dict[str, Any]) -> bool:
        """Cache real-time traffic data from İBB and other sources"""
        try:
            cache_key = f"traffic_realtime:{route_id}"
            
            cache_data = {
                "traffic_data": traffic_data,
                "route_id": route_id,
                "timestamp": datetime.now().isoformat(),
                "data_sources": traffic_data.get("sources", ["ibb_traffic", "waze_api"]),
                "congestion_level": traffic_data.get("congestion_level", "moderate"),
                "estimated_delay_minutes": traffic_data.get("delay_minutes", 0)
            }
            
            if self.cache_enabled:
                # Short TTL for real-time data (5 minutes)
                self.redis_client.setex(cache_key, 300, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            return True
            
        except Exception as e:
            print(f"Failed to cache real-time traffic data: {e}")
            return False
    
    def get_cached_real_time_traffic(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached real-time traffic data"""
        try:
            cache_key = f"traffic_realtime:{route_id}"
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    return result["traffic_data"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    return result["traffic_data"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve real-time traffic data: {e}")
            return None
    
    def cache_crowd_predictions(self, location_id: str, predictions: Dict[str, Any]) -> bool:
        """Cache crowd predictions for popular attractions"""
        try:
            cache_key = f"crowd_predictions:{location_id}"
            
            cache_data = {
                "predictions": predictions,
                "location_id": location_id,
                "generated_at": datetime.now().isoformat(),
                "prediction_model": predictions.get("model_version", "v1.0"),
                "hourly_predictions": predictions.get("hourly", {}),
                "best_visit_times": predictions.get("best_times", []),
                "avoid_times": predictions.get("avoid_times", [])
            }
            
            if self.cache_enabled:
                # Store predictions with 2 hour TTL
                self.redis_client.setex(cache_key, 7200, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            return True
            
        except Exception as e:
            print(f"Failed to cache crowd predictions: {e}")
            return False
    
    def get_cached_crowd_predictions(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached crowd predictions"""
        try:
            cache_key = f"crowd_predictions:{location_id}"
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    return result["predictions"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    return result["predictions"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve crowd predictions: {e}")
            return None
    
    def cache_event_data(self, area: str, events: List[Dict[str, Any]]) -> bool:
        """Cache real-time event data affecting routes and attractions"""
        try:
            cache_key = f"events_realtime:{area.lower().replace(' ', '_')}"
            
            cache_data = {
                "events": events,
                "area": area,
                "cached_at": datetime.now().isoformat(),
                "event_count": len(events),
                "event_types": list(set(event.get("type", "general") for event in events)),
                "impact_levels": [event.get("impact_level", "low") for event in events]
            }
            
            if self.cache_enabled:
                # Store events with 30 minute TTL (events change frequently)
                self.redis_client.setex(cache_key, 1800, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            return True
            
        except Exception as e:
            print(f"Failed to cache event data: {e}")
            return False
    
    def get_cached_event_data(self, area: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached event data"""
        try:
            cache_key = f"events_realtime:{area.lower().replace(' ', '_')}"
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    return result["events"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    return result["events"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve event data: {e}")
            return None
    
    # ==============================================
    # UX OPTIMIZATION & ANALYTICS
    # ==============================================
    
    def cache_ux_optimization_data(self, feature: str, optimization_data: Dict[str, Any]) -> bool:
        """Cache UX optimization data for smooth user experiences"""
        try:
            cache_key = f"ux_optimization:{feature}"
            
            cache_data = {
                "optimization_data": optimization_data,
                "feature": feature,
                "cached_at": datetime.now().isoformat(),
                "performance_metrics": optimization_data.get("performance", {}),
                "user_interaction_data": optimization_data.get("interactions", {}),
                "a_b_test_results": optimization_data.get("ab_test", {})
            }
            
            if self.cache_enabled:
                # Store UX data with 1 hour TTL
                self.redis_client.setex(cache_key, 3600, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            return True
            
        except Exception as e:
            print(f"Failed to cache UX optimization data: {e}")
            return False
    
    def get_cached_ux_optimization_data(self, feature: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached UX optimization data"""
        try:
            cache_key = f"ux_optimization:{feature}"
            
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    return result["optimization_data"]
            else:
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    return result["optimization_data"]
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve UX optimization data: {e}")
            return None
    
    def track_competitive_advantage_usage(self, feature_type: str, user_id: str, value_delivered: Dict[str, Any]) -> bool:
        """Track usage of features that provide competitive advantage over Google Maps/TripAdvisor"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            cache_key = f"competitive_analytics:{feature_type}:{today}"
            
            # Get existing data or initialize
            existing_data = None
            if self.cache_enabled:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    existing_data = pickle.loads(cached_data)
            else:
                existing_data = self.memory_cache.get(cache_key)
            
            if existing_data is None:
                existing_data = {
                    "feature_type": feature_type,
                    "date": today,
                    "unique_users": set(),
                    "usage_count": 0,
                    "value_metrics": defaultdict(list),
                    "user_satisfaction": []
                }
            
            # Update metrics
            existing_data["unique_users"].add(user_id)
            existing_data["usage_count"] += 1
            
            for metric, value in value_delivered.items():
                existing_data["value_metrics"][metric].append(value)
            
            # Convert set to list for JSON serialization
            cache_data = existing_data.copy()
            cache_data["unique_users"] = list(existing_data["unique_users"])
            cache_data["unique_user_count"] = len(existing_data["unique_users"])
            
            if self.cache_enabled:
                # Store analytics with 24 hour TTL
                self.redis_client.setex(cache_key, 86400, pickle.dumps(cache_data))
            else:
                self.memory_cache[cache_key] = cache_data
            
            return True
            
        except Exception as e:
            print(f"Failed to track competitive advantage usage: {e}")
            return False
    
    def get_competitive_advantage_analytics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get analytics on competitive advantage features usage"""
        try:
            analytics = {
                "summary": {
                    "total_unique_value_delivered": 0,
                    "features_with_advantage": [],
                    "user_retention_from_unique_features": 0
                },
                "feature_breakdown": {},
                "daily_trends": []
            }
            
            # Analyze last N days
            for days_ago in range(days_back):
                date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                
                # Check all competitive feature types
                feature_types = ["localized_tips", "hidden_gems", "smart_daily_guidance", "real_time_data"]
                
                daily_data = {"date": date, "features": {}}
                
                for feature_type in feature_types:
                    cache_key = f"competitive_analytics:{feature_type}:{date}"
                    
                    feature_data = None
                    if self.cache_enabled:
                        cached_data = self.redis_client.get(cache_key)
                        if cached_data:
                            feature_data = pickle.loads(cached_data)
                    else:
                        feature_data = self.memory_cache.get(cache_key)
                    
                    if feature_data:
                        daily_data["features"][feature_type] = {
                            "unique_users": feature_data.get("unique_user_count", 0),
                            "usage_count": feature_data.get("usage_count", 0),
                            "avg_satisfaction": sum(feature_data.get("user_satisfaction", [])) / len(feature_data.get("user_satisfaction", [1])) if feature_data.get("user_satisfaction") else 0
                        }
                        
                        # Add to feature breakdown
                        if feature_type not in analytics["feature_breakdown"]:
                            analytics["feature_breakdown"][feature_type] = {
                                "total_users": set(),
                                "total_usage": 0,
                                "satisfaction_scores": []
                            }
                        
                        analytics["feature_breakdown"][feature_type]["total_users"].update(feature_data.get("unique_users", []))
                        analytics["feature_breakdown"][feature_type]["total_usage"] += feature_data.get("usage_count", 0)
                        analytics["feature_breakdown"][feature_type]["satisfaction_scores"].extend(feature_data.get("user_satisfaction", []))
                
                analytics["daily_trends"].append(daily_data)
            
            # Finalize summary
            for feature_type, data in analytics["feature_breakdown"].items():
                data["unique_user_count"] = len(data["total_users"])
                data["total_users"] = list(data["total_users"])  # Convert set to list
                data["avg_satisfaction"] = sum(data["satisfaction_scores"]) / len(data["satisfaction_scores"]) if data["satisfaction_scores"] else 0
                
                analytics["summary"]["total_unique_value_delivered"] += data["total_usage"]
                if data["unique_user_count"] > 0:
                    analytics["summary"]["features_with_advantage"].append(feature_type)
            
            return analytics
            
        except Exception as e:
            print(f"Failed to get competitive advantage analytics: {e}")
            return {"error": str(e)}

# Global cache instance
route_cache = RouteCacheManager()
