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


class WeatherAwareRouteCache:
    """Weather-aware route cache with dynamic recommendations based on weather"""
    
    def __init__(self):
        # Weather-aware route recommendations
        self.weather_route_adaptations = {
            'rainy': {
                'indoor_focus': True,
                'covered_routes': True,
                'underground_preferred': True,
                'duration_reduction': 0.7  # Reduce outdoor time by 30%
            },
            'hot': {
                'shade_preferred': True,
                'early_morning': True,
                'water_nearby': True,
                'duration_reduction': 0.8  # Reduce walking in heat
            },
            'cold': {
                'indoor_warmup_stops': True,
                'reduced_outdoor_time': True,
                'duration_reduction': 0.8
            },
            'windy': {
                'avoid_waterfront': True,
                'sheltered_routes': True,
                'bosphorus_warning': True
            }
        }
        
        # Base popular routes with weather adaptations
        self.popular_routes = {
            'sultanahmet_historic': {
                'name': 'Historic Sultanahmet Tour',
                'description': 'Classic historical route covering main attractions',
                'start_area': 'sultanahmet',
                'duration_hours': 4.0,
                'distance_km': 3.2,
                'attractions': ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace', 'Basilica Cistern'],
                'weather_adaptations': {
                    'rainy': {
                        'indoor_attractions': ['Hagia Sophia', 'Blue Mosque', 'Basilica Cistern', 'Istanbul Archaeological Museums'],
                        'covered_areas': ['Grand Bazaar', 'Underground Cistern'],
                        'advice': 'Perfect for rainy weather - mostly covered historical sites'
                    },
                    'hot': {
                        'best_time': 'early_morning',
                        'shade_spots': ['Sultanahmet Park', 'Gulhane Park'],
                        'indoor_breaks': ['Hagia Sophia', 'Blue Mosque'],
                        'advice': 'Start early morning, take breaks in shaded mosques'
                    },
                    'cold': {
                        'indoor_focus': ['Hagia Sophia', 'Topkapi Palace indoor sections'],
                        'warm_spots': ['Turkish baths', 'Traditional cafes'],
                        'advice': 'Focus on indoor palace rooms and warm up with Turkish tea'
                    }
                },
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
    
    def get_popular_route(self, area: str, style: str = 'balanced', weather_data=None) -> Optional[Dict]:
        """Get a popular pre-computed route for an area, adapted for current weather"""
        area_lower = area.lower()
        
        # Get base route
        base_route = None
        if 'sultanahmet' in area_lower or 'historic' in style.lower():
            base_route = self.popular_routes.get('sultanahmet_historic')
        elif 'galata' in area_lower or 'scenic' in style.lower():
            base_route = self.popular_routes.get('bosphorus_scenic')
        elif 'taksim' in area_lower or 'modern' in area_lower:
            base_route = self.popular_routes.get('modern_istanbul')
        
        if not base_route:
            return None
        
        # Adapt route based on weather
        if weather_data:
            return self.adapt_route_for_weather(base_route.copy(), weather_data)
        
        return base_route.copy()
    
    def adapt_route_for_weather(self, route: Dict, weather_data: Any) -> Dict:
        """Adapt a route based on current weather conditions"""
        try:
            # Extract weather condition
            condition = getattr(weather_data, 'condition', '').lower()
            temp = getattr(weather_data, 'current_temp', 20)
            wind_speed = getattr(weather_data, 'wind_speed', 0)
            rainfall = getattr(weather_data, 'rainfall_1h', 0) or 0
            
            # Determine weather category
            weather_category = self._categorize_weather(condition, temp, wind_speed, rainfall)
            
            # Apply weather adaptations
            if weather_category in self.weather_route_adaptations:
                adaptation = self.weather_route_adaptations[weather_category]
                route = self._apply_adaptation(route, adaptation, weather_category)
            
            # Add weather-specific recommendations
            route['weather_adapted'] = True
            route['current_weather'] = {
                'condition': condition,
                'temperature': temp,
                'category': weather_category,
                'adapted_at': datetime.now().isoformat()
            }
            
            # Route-specific weather adaptations
            route_name = route.get('name', '').lower()
            if 'weather_adaptations' in route and weather_category in route['weather_adaptations']:
                weather_adaptation = route['weather_adaptations'][weather_category]
                route.update(weather_adaptation)
                
                # Add weather advice
                if 'advice' in weather_adaptation:
                    route['weather_advice'] = weather_adaptation['advice']
            
            return route
            
        except Exception as e:
            print(f"Error adapting route for weather: {e}")
            return route
    
    def _categorize_weather(self, condition: str, temp: float, wind_speed: float, rainfall: float) -> str:
        """Categorize weather conditions for route adaptation"""
        if rainfall > 0.5 or 'rain' in condition or 'drizzle' in condition:
            return 'rainy'
        elif temp > 28:
            return 'hot'
        elif temp < 10:
            return 'cold'
        elif wind_speed > 20:
            return 'windy'
        else:
            return 'clear'
    
    def _apply_adaptation(self, route: Dict, adaptation: Dict, weather_category: str) -> Dict:
        """Apply general weather adaptations to route"""
        if adaptation.get('duration_reduction'):
            route['duration_hours'] *= adaptation['duration_reduction']
            route['weather_notes'] = route.get('weather_notes', [])
            route['weather_notes'].append(f"Duration reduced for {weather_category} weather")
        
        if adaptation.get('indoor_focus'):
            route['weather_notes'] = route.get('weather_notes', [])
            route['weather_notes'].append("Route adapted to focus on indoor attractions")
        
        if adaptation.get('shade_preferred'):
            route['weather_notes'] = route.get('weather_notes', [])
            route['weather_notes'].append("Route prioritizes shaded areas")
        
        return route
    
    def get_weather_recommendations(self, weather_data: Any) -> List[str]:
        """Get weather-specific route recommendations"""
        try:
            condition = getattr(weather_data, 'condition', '').lower()
            temp = getattr(weather_data, 'current_temp', 20)
            wind_speed = getattr(weather_data, 'wind_speed', 0)
            rainfall = getattr(weather_data, 'rainfall_1h', 0) or 0
            
            recommendations = []
            
            if rainfall > 0.5 or 'rain' in condition:
                recommendations.extend([
                    "Consider indoor attractions like Hagia Sophia and Basilica Cistern",
                    "Grand Bazaar and Spice Bazaar are perfect for rainy weather",
                    "Use covered walkways and underground passages when possible"
                ])
            
            if temp > 28:
                recommendations.extend([
                    "Start your tour early morning (before 10 AM) or late afternoon",
                    "Take frequent breaks in air-conditioned museums and cafes",
                    "Stay hydrated and seek shade in parks like Gülhane"
                ])
            
            if temp < 10:
                recommendations.extend([
                    "Focus on indoor palaces and museums",
                    "Warm up with traditional Turkish tea or coffee",
                    "Turkish baths (hamams) are perfect for cold weather"
                ])
            
            if wind_speed > 20:
                recommendations.extend([
                    "Avoid exposed waterfront areas like Galata Bridge",
                    "Choose sheltered streets in old city areas",
                    "Be cautious near the Bosphorus - strong winds expected"
                ])
            
            if not recommendations:
                recommendations.append("Perfect weather for exploring Istanbul!")
            
            return recommendations
            
        except Exception as e:
            return ["Weather data unavailable - enjoy your Istanbul exploration!"]
    
    def get_all_popular_routes(self, weather_data=None) -> Dict[str, Dict]:
        """Get all popular routes, optionally adapted for weather"""
        routes = {}
        for key, route in self.popular_routes.items():
            if weather_data:
                routes[key] = self.adapt_route_for_weather(route.copy(), weather_data)
            else:
                routes[key] = route.copy()
        return routes

def get_weather_aware_route_recommendations(area: str, style: str = 'balanced', weather_data=None) -> Dict[str, Any]:
    """Get weather-aware route recommendations using the cache"""
    try:
        # Get weather-adapted route
        route = weather_aware_cache.get_popular_route(area, style, weather_data)
        
        # Get weather recommendations
        weather_recommendations = []
        if weather_data:
            weather_recommendations = weather_aware_cache.get_weather_recommendations(weather_data)
        
        return {
            'route': route,
            'weather_recommendations': weather_recommendations,
            'weather_aware': weather_data is not None,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting weather-aware recommendations: {e}")
        # Fallback to basic route
        return {
            'route': weather_aware_cache.get_popular_route(area, style),
            'weather_recommendations': ["Weather data unavailable - enjoy exploring!"],
            'weather_aware': False,
            'error': str(e)
        }

def get_transportation_advice_for_weather(weather_data=None, start_location=None, end_location=None) -> Dict[str, Any]:
    """Get detailed transportation advice like Google Maps based on weather conditions"""
    
    # Default locations if not provided
    if not start_location:
        start_location = "Taksim"
    if not end_location:
        end_location = "Sultanahmet"
    
    base_advice = {
        "route_overview": f"Best routes from {start_location} to {end_location}",
        "weather_impact": "Current weather conditions analyzed",
        "recommended_routes": [],
        "alternative_routes": [],
        "real_time_alerts": [],
        "cost_breakdown": {},
        "accessibility_info": []
    }
    
    if not weather_data:
        # Default recommendations without weather data
        base_advice["recommended_routes"] = [
            {
                "route_name": "Metro Route",
                "duration": "25-30 min",
                "cost": "15₺",
                "steps": [
                    {"mode": "walk", "instruction": f"Walk to {start_location} Metro Station", "duration": "3 min"},
                    {"mode": "metro", "instruction": "Take M2 Line towards Vezneciler", "duration": "20 min"},
                    {"mode": "walk", "instruction": f"Walk to {end_location}", "duration": "5 min"}
                ],
                "comfort_level": "High",
                "weather_suitability": "All weather"
            }
        ]
        return base_advice
    
    try:
        condition = getattr(weather_data, 'condition', '').lower()
        temp = getattr(weather_data, 'current_temp', 20)
        wind_speed = getattr(weather_data, 'wind_speed', 0)
        rainfall = getattr(weather_data, 'rainfall_1h', 0) or 0
        
        # Analyze weather impact
        weather_category = _categorize_weather_for_transport(condition, temp, wind_speed, rainfall)
        base_advice["weather_impact"] = f"Current: {condition.title()}, {temp}°C - {weather_category} conditions"
        
        # Generate detailed route recommendations
        routes = _generate_detailed_routes(start_location, end_location, weather_category, temp, rainfall, wind_speed)
        base_advice["recommended_routes"] = routes["recommended"]
        base_advice["alternative_routes"] = routes["alternatives"]
        
        # Real-time alerts based on weather
        base_advice["real_time_alerts"] = _generate_weather_alerts(weather_category, temp, rainfall, wind_speed)
        
        # Cost breakdown
        base_advice["cost_breakdown"] = {
            "metro_tram": "15₺ (Istanbulkart)",
            "bus": "15₺ (Istanbulkart)", 
            "ferry": "25₺ (Istanbulkart)",
            "taxi_estimate": "45-60₺",
            "walking": "Free"
        }
        
        # Accessibility information
        base_advice["accessibility_info"] = [
            "🦽 Metro stations have elevator access",
            "🚌 Low-floor buses available on main routes",
            "⚠️ Ferry terminals have limited accessibility",
            "🚶‍♀️ Sidewalk conditions vary by district"
        ]
        
        return base_advice
        
    except Exception as e:
        base_advice["error"] = f"Transportation advice unavailable: {e}"
        return base_advice

def _categorize_weather_for_transport(condition: str, temp: float, wind_speed: float, rainfall: float) -> str:
    """Categorize weather for detailed transport planning"""
    if rainfall > 2.0:
        return "Heavy Rain"
    elif rainfall > 0.5:
        return "Light Rain"
    elif temp > 32:
        return "Very Hot"
    elif temp > 28:
        return "Hot"
    elif temp < 5:
        return "Very Cold"
    elif temp < 10:
        return "Cold"
    elif wind_speed > 30:
        return "Very Windy"
    elif wind_speed > 20:
        return "Windy"
    else:
        return "Pleasant"

def _generate_detailed_routes(start: str, end: str, weather_category: str, temp: float, rainfall: float, wind_speed: float) -> Dict[str, List]:
    """Generate detailed route options like Google Maps"""
    
    routes = {"recommended": [], "alternatives": []}
    
    # Weather-optimized primary route
    if weather_category in ["Heavy Rain", "Light Rain"]:
        # Prioritize underground/covered transport
        routes["recommended"].append({
            "route_name": "Metro + Tram (Weather Optimized)",
            "duration": "28-35 min",
            "cost": "15₺",
            "weather_rating": "⭐⭐⭐⭐⭐ Excellent for rain",
            "steps": [
                {
                    "mode": "walk",
                    "instruction": f"Walk to {start} Metro Station (use covered walkways)",
                    "duration": "4 min",
                    "distance": "300m",
                    "weather_tip": "Stay under building overhangs when possible"
                },
                {
                    "mode": "metro",
                    "instruction": "Take M2 Line (Red Line) towards Vezneciler",
                    "duration": "18 min",
                    "stations": "6 stops",
                    "weather_tip": "Completely dry and climate controlled"
                },
                {
                    "mode": "tram",
                    "instruction": "Transfer to T1 Tram towards Eminönü",
                    "duration": "8 min",
                    "stations": "3 stops",
                    "weather_tip": "Modern trams with good weather protection"
                },
                {
                    "mode": "walk",
                    "instruction": f"Walk to {end} (use covered Grand Bazaar if possible)",
                    "duration": "6 min",
                    "distance": "400m",
                    "weather_tip": "Grand Bazaar provides full rain protection"
                }
            ],
            "comfort_level": "Very High",
            "accessibility": "Full wheelchair access",
            "real_time_info": "Live departure times available via IBB Mobile app"
        })
        
        # Alternative bus route
        routes["alternatives"].append({
            "route_name": "Express Bus Route",
            "duration": "35-45 min",
            "cost": "15₺",
            "weather_rating": "⭐⭐⭐ Good with short walks",
            "steps": [
                {
                    "mode": "walk",
                    "instruction": f"Walk to {start} bus stop",
                    "duration": "5 min",
                    "weather_tip": "Bring umbrella for bus stop wait"
                },
                {
                    "mode": "bus",
                    "instruction": "Take Bus 28, 30M, or 46C towards Eminönü",
                    "duration": "25-30 min",
                    "weather_tip": "Modern buses with AC and heating"
                },
                {
                    "mode": "walk",
                    "instruction": f"Walk to {end}",
                    "duration": "8 min",
                    "weather_tip": "Final walk - stay on main streets with better drainage"
                }
            ],
            "comfort_level": "Medium",
            "traffic_warning": "May be delayed due to rain-related traffic"
        })
        
    elif weather_category in ["Very Hot", "Hot"]:
        # Prioritize air-conditioned transport
        routes["recommended"].append({
            "route_name": "Air-Conditioned Metro Route",
            "duration": "25-30 min",
            "cost": "15₺",
            "weather_rating": f"⭐⭐⭐⭐⭐ Perfect for {temp}°C weather",
            "steps": [
                {
                    "mode": "walk",
                    "instruction": f"Walk to {start} Metro (seek shade, avoid 12-16:00)",
                    "duration": "3 min",
                    "distance": "250m",
                    "weather_tip": f"Very hot ({temp}°C) - walk in building shadows"
                },
                {
                    "mode": "metro",
                    "instruction": "Take M2 Line - fully air conditioned",
                    "duration": "20 min",
                    "weather_tip": "Cool and comfortable - perfect escape from heat"
                },
                {
                    "mode": "walk",
                    "instruction": f"Walk to {end} (use shaded paths)",
                    "duration": "5 min",
                    "weather_tip": "Stick to shaded side of streets, carry water"
                }
            ],
            "comfort_level": "Very High",
            "health_tip": "Stay hydrated - drink water before and after journey"
        })
        
        # Scenic ferry alternative (if not too hot)
        if temp < 30:
            routes["alternatives"].append({
                "route_name": "Scenic Ferry Route",
                "duration": "40-50 min",
                "cost": "25₺",
                "weather_rating": "⭐⭐⭐⭐ Great with Bosphorus breeze",
                "steps": [
                    {
                        "mode": "walk",
                        "instruction": "Walk to Kabataş Ferry Terminal",
                        "duration": "12 min",
                        "weather_tip": "Morning or evening recommended for this walk"
                    },
                    {
                        "mode": "ferry",
                        "instruction": "Take Bosphorus ferry to Eminönü",
                        "duration": "25 min",
                        "weather_tip": "Sit on shaded side of ferry, enjoy sea breeze"
                    },
                    {
                        "mode": "walk",
                        "instruction": f"Walk to {end}",
                        "duration": "8 min",
                        "weather_tip": "Short final walk in historic area"
                    }
                ],
                "comfort_level": "High",
                "scenic_value": "Excellent Bosphorus views"
            })
    
    elif weather_category in ["Very Windy", "Windy"]:
        # Avoid ferries, prefer underground
        routes["recommended"].append({
            "route_name": "Underground Metro (Wind Protected)",
            "duration": "25-30 min", 
            "cost": "15₺",
            "weather_rating": f"⭐⭐⭐⭐⭐ Completely wind-free",
            "steps": [
                {
                    "mode": "walk",
                    "instruction": f"Walk to {start} Metro (stay away from open areas)",
                    "duration": "4 min",
                    "weather_tip": f"Strong winds ({wind_speed} m/s) - avoid waterfront areas"
                },
                {
                    "mode": "metro",
                    "instruction": "Take M2 Line - underground, no wind impact",
                    "duration": "20 min",
                    "weather_tip": "Completely protected from wind"
                },
                {
                    "mode": "walk",
                    "instruction": f"Walk to {end} via sheltered streets",
                    "duration": "5 min",
                    "weather_tip": "Use narrow streets between buildings for wind protection"
                }
            ],
            "comfort_level": "Very High",
            "safety_note": "Ferries may be cancelled due to strong winds"
        })
    
    else:
        # Pleasant weather - all options available
        routes["recommended"].append({
            "route_name": "Balanced Metro + Walking",
            "duration": "25-30 min",
            "cost": "15₺", 
            "weather_rating": "⭐⭐⭐⭐⭐ Perfect weather for any transport",
            "steps": [
                {
                    "mode": "walk",
                    "instruction": f"Walk to {start} Metro Station",
                    "duration": "3 min",
                    "distance": "250m",
                    "weather_tip": f"Pleasant {temp}°C - enjoy the walk!"
                },
                {
                    "mode": "metro",
                    "instruction": "Take M2 Line towards Vezneciler",
                    "duration": "20 min",
                    "stations": "6 stops"
                },
                {
                    "mode": "walk",
                    "instruction": f"Walk to {end}",
                    "duration": "5 min",
                    "distance": "400m",
                    "weather_tip": "Perfect weather for exploring on foot"
                }
            ],
            "comfort_level": "High",
            "scenic_value": "Good opportunity for street photography"
        })
        
        # Walking route for pleasant weather
        routes["alternatives"].append({
            "route_name": "Scenic Walking Route",
            "duration": "45-60 min",
            "cost": "Free",
            "weather_rating": f"⭐⭐⭐⭐⭐ Ideal walking weather ({temp}°C)",
            "steps": [
                {
                    "mode": "walk",
                    "instruction": "Walk via İstiklal Street → Galata Bridge → Eminönü",
                    "duration": "50 min",
                    "distance": "4.2 km",
                    "weather_tip": "Perfect temperature for sightseeing walk"
                }
            ],
            "comfort_level": "Medium",
            "scenic_value": "Excellent - passes major landmarks",
            "health_benefit": "Great exercise with beautiful views"
        })
    
    return routes

def _generate_weather_alerts(weather_category: str, temp: float, rainfall: float, wind_speed: float) -> List[str]:
    """Generate real-time weather alerts for transportation"""
    alerts = []
    
    if weather_category == "Heavy Rain":
        alerts.extend([
            "⚠️ Heavy rain alert: Metro and tram services recommended",
            "🚌 Bus services may experience delays due to flooding on some routes",
            "🚗 Traffic congestion expected - add 15-20 minutes to taxi times",
            "⛴️ Ferry services may be suspended - check IDO website"
        ])
    
    elif weather_category == "Light Rain":
        alerts.extend([
            "🌧️ Light rain: Slight delays possible on bus routes",
            "☂️ Bring umbrella for short walks between stations"
        ])
    
    elif weather_category == "Very Hot":
        alerts.extend([
            f"🌡️ Extreme heat warning: {temp}°C - avoid outdoor walking 12:00-16:00",
            "💧 Carry water and stay hydrated",
            "🚇 Metro stations provide air conditioning relief"
        ])
        
    elif weather_category == "Very Windy":
        alerts.extend([
            f"💨 Strong wind alert: {wind_speed} m/s - ferry services may be cancelled",
            "⚠️ Be cautious walking near Bosphorus and Golden Horn",
            "🌉 Avoid bridges during peak wind hours"
        ])
    
    elif weather_category == "Very Cold":
        alerts.extend([
            f"🥶 Very cold: {temp}°C - dress warmly for outdoor waiting",
            "🧥 Heated buses and metro provide warm transport",
            "⚠️ Watch for icy conditions on walkways"
        ])
    
    return alerts

# Global cache instances
route_cache = RouteCache(max_size=500, default_ttl=1800)  # 30 minutes TTL
weather_aware_cache = WeatherAwareRouteCache()

# Cleanup task - in production this would be a background task
def cleanup_caches():
    """Cleanup expired cache entries"""
    expired_count = route_cache.cleanup_expired()
    print(f"🧹 Cleaned up {expired_count} expired routes from cache")
    return expired_count
