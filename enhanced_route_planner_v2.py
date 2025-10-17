#!/usr/bin/env python3
"""
Enhanced GPS-Based Route Planner V2 for AI Istanbul
Advanced features: weather-aware, preference-based, AI-powered, time-aware, multi-stop optimized
"""

import json
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import asyncio
import heapq

# Import weather client
try:
    from backend.api_clients.enhanced_weather import EnhancedWeatherClient
    WEATHER_CLIENT_AVAILABLE = True
except ImportError:
    WEATHER_CLIENT_AVAILABLE = False
    logging.warning("Weather client not available")

# Import intelligent location detector
try:
    from istanbul_ai.services.intelligent_location_detector import (
        IntelligentLocationDetector,
        LocationDetectionResult,
        GPSContext,
        WeatherContext
    )
    INTELLIGENT_LOCATION_AVAILABLE = True
except ImportError:
    INTELLIGENT_LOCATION_AVAILABLE = False
    logging.warning("Intelligent Location Detector not available")

# Import original route planner
try:
    from enhanced_gps_route_planner import (
        TransportMode,
        GPSLocation,
        PersonalizedWaypoint,
        RouteSegment,
        PersonalizedRoute,
        EnhancedGPSRoutePlanner
    )
    ORIGINAL_PLANNER_AVAILABLE = True
except ImportError:
    ORIGINAL_PLANNER_AVAILABLE = False
    logging.warning("Original route planner not available")

logger = logging.getLogger(__name__)

# ==================== ENUMS & DATA CLASSES ====================

class WeatherCondition(Enum):
    """Weather conditions affecting routes"""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    WINDY = "windy"
    FOGGY = "foggy"

class UserMood(Enum):
    """User mood/intent for personalization"""
    ADVENTUROUS = "adventurous"
    RELAXED = "relaxed"
    ROMANTIC = "romantic"
    CULTURAL = "cultural"
    FOODIE = "foodie"
    PHOTOGRAPHER = "photographer"
    SHOPPING = "shopping"

class TimeOfDay(Enum):
    """Time periods with different characteristics"""
    EARLY_MORNING = "early_morning"  # 6-9 AM
    MORNING = "morning"              # 9-12 PM
    AFTERNOON = "afternoon"          # 12-5 PM
    EVENING = "evening"              # 5-8 PM
    NIGHT = "night"                  # 8 PM-12 AM
    LATE_NIGHT = "late_night"        # 12-6 AM

@dataclass
class WeatherContext:
    """Weather context for route planning"""
    condition: WeatherCondition
    temperature_celsius: float
    humidity: float
    wind_speed_kmh: float
    precipitation_mm: float
    visibility_km: float
    uv_index: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_suitable_for_outdoor(self) -> bool:
        """Check if weather is suitable for outdoor activities"""
        if self.condition in [WeatherCondition.RAINY, WeatherCondition.SNOWY]:
            return False
        if self.temperature_celsius < 5 or self.temperature_celsius > 35:
            return False
        if self.wind_speed_kmh > 40:
            return False
        return True
    
    def is_suitable_for_ferry(self) -> bool:
        """Check if weather is suitable for ferry travel"""
        if self.condition == WeatherCondition.FOGGY and self.visibility_km < 2:
            return False
        if self.wind_speed_kmh > 30:
            return False
        return True

@dataclass
class UserPreferences:
    """Enhanced user preferences"""
    interests: List[str] = field(default_factory=list)
    mood: Optional[UserMood] = None
    preferred_transport_modes: List[TransportMode] = field(default_factory=list)
    avoid_transport_modes: List[TransportMode] = field(default_factory=list)
    budget_per_person: float = 100.0  # Turkish Lira
    accessibility_needs: List[str] = field(default_factory=list)
    activity_level: str = "medium"  # low, medium, high
    max_walking_distance_km: float = 5.0
    prefer_scenic: bool = True
    prefer_indoor: Optional[bool] = None  # None = weather-dependent
    time_available_hours: float = 8.0
    dietary_restrictions: List[str] = field(default_factory=list)
    language_preference: str = "en"

@dataclass
class LocationAttributes:
    """Detailed location attributes"""
    name: str
    location: GPSLocation
    category: str
    subcategory: str = ""
    is_indoor: bool = False
    is_outdoor: bool = False
    is_waterfront: bool = False
    has_garden: bool = False
    has_cafe: bool = False
    is_air_conditioned: bool = True
    requires_booking: bool = False
    average_visit_duration_minutes: int = 60
    entrance_fee_tl: float = 0.0
    accessibility_features: List[str] = field(default_factory=list)
    best_time_of_day: List[TimeOfDay] = field(default_factory=list)
    crowd_level: str = "medium"  # low, medium, high
    photo_opportunities: int = 5  # 1-10 scale
    family_friendly: bool = True
    tags: List[str] = field(default_factory=list)

@dataclass
class RouteOptimizationResult:
    """Result of route optimization"""
    route: PersonalizedRoute
    optimization_score: float
    weather_compatibility: float
    preference_match: float
    time_efficiency: float
    cost_efficiency: float
    recommendations: List[str]
    warnings: List[str]
    alternative_routes: List[PersonalizedRoute] = field(default_factory=list)

# ==================== OPTIMIZERS ====================

class WeatherAwareRouter:
    """Weather-aware routing logic"""
    
    def __init__(self, weather_client: Optional[EnhancedWeatherClient] = None):
        self.weather_client = weather_client
        logger.info("🌤 Weather-Aware Router initialized")
    
    async def get_current_weather_context(self) -> WeatherContext:
        """Get current weather context for Istanbul"""
        if not self.weather_client:
            return self._get_mock_weather_context()
        
        try:
            weather_data = self.weather_client.get_current_weather("Istanbul", "TR")
            return self._parse_weather_data(weather_data)
        except Exception as e:
            logger.error(f"Failed to get weather: {e}")
            return self._get_mock_weather_context()
    
    def _parse_weather_data(self, weather_data: Dict) -> WeatherContext:
        """Parse weather API data into WeatherContext"""
        # Map weather condition
        condition_map = {
            'Clear': WeatherCondition.SUNNY,
            'Clouds': WeatherCondition.CLOUDY,
            'Rain': WeatherCondition.RAINY,
            'Snow': WeatherCondition.SNOWY,
            'Fog': WeatherCondition.FOGGY,
            'Mist': WeatherCondition.FOGGY
        }
        
        main_weather = weather_data.get('weather', [{}])[0].get('main', 'Clear')
        condition = condition_map.get(main_weather, WeatherCondition.CLOUDY)
        
        main_data = weather_data.get('main', {})
        wind_data = weather_data.get('wind', {})
        
        return WeatherContext(
            condition=condition,
            temperature_celsius=main_data.get('temp', 20.0),
            humidity=main_data.get('humidity', 60.0),
            wind_speed_kmh=wind_data.get('speed', 0) * 3.6,  # m/s to km/h
            precipitation_mm=weather_data.get('rain', {}).get('1h', 0.0),
            visibility_km=weather_data.get('visibility', 10000) / 1000,
            uv_index=weather_data.get('uvi', 5.0)
        )
    
    def _get_mock_weather_context(self) -> WeatherContext:
        """Get mock weather context"""
        return WeatherContext(
            condition=WeatherCondition.SUNNY,
            temperature_celsius=22.0,
            humidity=65.0,
            wind_speed_kmh=15.0,
            precipitation_mm=0.0,
            visibility_km=10.0,
            uv_index=5.0
        )
    
    def filter_locations_by_weather(
        self,
        locations: List[LocationAttributes],
        weather: WeatherContext,
        user_pref: UserPreferences
    ) -> List[LocationAttributes]:
        """Filter and prioritize locations based on weather"""
        scored_locations = []
        
        for location in locations:
            score = self._calculate_weather_suitability(location, weather, user_pref)
            if score > 0.3:  # Minimum suitability threshold
                scored_locations.append((score, location))
        
        # Sort by score (highest first)
        scored_locations.sort(reverse=True, key=lambda x: x[0])
        
        return [loc for score, loc in scored_locations]
    
    def _calculate_weather_suitability(
        self,
        location: LocationAttributes,
        weather: WeatherContext,
        user_pref: UserPreferences
    ) -> float:
        """Calculate weather suitability score for a location"""
        score = 0.5  # Base score
        
        # Check user preference override
        if user_pref.prefer_indoor is True:
            score += 0.4 if location.is_indoor else -0.2
            return max(0, min(1, score))
        elif user_pref.prefer_indoor is False:
            score += 0.4 if location.is_outdoor else -0.2
            return max(0, min(1, score))
        
        # Weather-based scoring
        if weather.condition == WeatherCondition.RAINY:
            if location.is_indoor:
                score += 0.4
            elif location.is_outdoor and not location.has_cafe:
                score -= 0.3
        
        elif weather.condition == WeatherCondition.SUNNY:
            if location.is_outdoor or location.has_garden:
                score += 0.3
            if location.is_waterfront:
                score += 0.2
        
        elif weather.condition == WeatherCondition.SNOWY:
            if location.is_indoor and location.is_air_conditioned:
                score += 0.4
            elif location.is_outdoor:
                score -= 0.2
        
        # Temperature considerations
        if weather.temperature_celsius > 30:
            if location.is_air_conditioned or location.is_waterfront:
                score += 0.2
        elif weather.temperature_celsius < 10:
            if location.is_indoor and location.is_air_conditioned:
                score += 0.2
        
        # Wind considerations
        if weather.wind_speed_kmh > 30:
            if location.is_indoor:
                score += 0.2
            elif location.is_outdoor:
                score -= 0.2
        
        return max(0, min(1, score))
    
    def select_transport_mode_by_weather(
        self,
        weather: WeatherContext,
        available_modes: List[TransportMode],
        user_pref: UserPreferences
    ) -> List[TransportMode]:
        """Select optimal transport modes based on weather"""
        scored_modes = []
        
        for mode in available_modes:
            # Skip avoided modes
            if mode in user_pref.avoid_transport_modes:
                continue
            
            score = self._score_transport_mode_weather(mode, weather)
            scored_modes.append((score, mode))
        
        # Sort by score
        scored_modes.sort(reverse=True, key=lambda x: x[0])
        
        return [mode for score, mode in scored_modes if score > 0.3]
    
    def _score_transport_mode_weather(
        self,
        mode: TransportMode,
        weather: WeatherContext
    ) -> float:
        """Score transport mode based on weather"""
        score = 0.5
        
        mode_weather_dep = mode.value.get('weather_dependent', False)
        
        if weather.condition in [WeatherCondition.RAINY, WeatherCondition.SNOWY]:
            if mode_weather_dep:
                score -= 0.3  # Penalize walking/cycling in bad weather
            else:
                score += 0.2  # Prefer covered transport
        
        if weather.condition == WeatherCondition.SUNNY and weather.temperature_celsius < 25:
            if mode in [TransportMode.WALKING, TransportMode.CYCLING, TransportMode.FERRY]:
                score += 0.3  # Perfect weather for outdoor transport
        
        if weather.wind_speed_kmh > 30:
            if mode == TransportMode.FERRY:
                score -= 0.4  # Ferry not suitable in high wind
            elif mode in [TransportMode.METRO, TransportMode.PUBLIC_TRANSPORT]:
                score += 0.2
        
        if weather.condition == WeatherCondition.FOGGY:
            if mode == TransportMode.FERRY and weather.visibility_km < 2:
                score -= 0.5
            elif mode == TransportMode.METRO:
                score += 0.2
        
        return max(0, min(1, score))


class PreferenceBasedRouter:
    """Preference-based route generation"""
    
    def __init__(self):
        logger.info("🎯 Preference-Based Router initialized")
    
    def score_location_by_preferences(
        self,
        location: LocationAttributes,
        preferences: UserPreferences
    ) -> float:
        """Score a location based on user preferences"""
        score = 0.0
        
        # Interest matching
        interest_match = self._calculate_interest_match(location, preferences.interests)
        score += interest_match * 0.3
        
        # Mood matching
        if preferences.mood:
            mood_match = self._calculate_mood_match(location, preferences.mood)
            score += mood_match * 0.25
        
        # Budget considerations
        budget_score = self._calculate_budget_score(location, preferences)
        score += budget_score * 0.15
        
        # Accessibility
        accessibility_score = self._calculate_accessibility_score(location, preferences)
        score += accessibility_score * 0.15
        
        # Activity level
        activity_score = self._calculate_activity_score(location, preferences)
        score += activity_score * 0.15
        
        return min(1.0, score)
    
    def _calculate_interest_match(
        self,
        location: LocationAttributes,
        interests: List[str]
    ) -> float:
        """Calculate interest matching score"""
        if not interests:
            return 0.5
        
        location_tags = set(location.tags + [location.category, location.subcategory])
        interest_set = set(interests)
        
        matches = len(location_tags.intersection(interest_set))
        return min(1.0, matches / max(len(interests), 1))
    
    def _calculate_mood_match(
        self,
        location: LocationAttributes,
        mood: UserMood
    ) -> float:
        """Calculate mood matching score"""
        mood_mappings = {
            UserMood.ADVENTUROUS: ['adventure', 'outdoor', 'active', 'sports'],
            UserMood.RELAXED: ['cafe', 'garden', 'spa', 'beach', 'park'],
            UserMood.ROMANTIC: ['sunset', 'waterfront', 'restaurant', 'garden', 'historic'],
            UserMood.CULTURAL: ['museum', 'historic', 'art', 'gallery', 'architecture'],
            UserMood.FOODIE: ['restaurant', 'cafe', 'food', 'market', 'cuisine'],
            UserMood.PHOTOGRAPHER: ['scenic', 'viewpoint', 'historic', 'architecture'],
            UserMood.SHOPPING: ['shopping', 'bazaar', 'market', 'boutique']
        }
        
        mood_tags = mood_mappings.get(mood, [])
        location_tags = set(location.tags + [location.category, location.subcategory])
        
        matches = len(set(mood_tags).intersection(location_tags))
        return min(1.0, matches / max(len(mood_tags), 1))
    
    def _calculate_budget_score(
        self,
        location: LocationAttributes,
        preferences: UserPreferences
    ) -> float:
        """Calculate budget compatibility score"""
        if location.entrance_fee_tl == 0:
            return 1.0
        
        if location.entrance_fee_tl <= preferences.budget_per_person * 0.2:
            return 1.0
        elif location.entrance_fee_tl <= preferences.budget_per_person * 0.5:
            return 0.7
        elif location.entrance_fee_tl <= preferences.budget_per_person:
            return 0.4
        else:
            return 0.1
    
    def _calculate_accessibility_score(
        self,
        location: LocationAttributes,
        preferences: UserPreferences
    ) -> float:
        """Calculate accessibility score"""
        if not preferences.accessibility_needs:
            return 1.0
        
        needs_set = set(preferences.accessibility_needs)
        features_set = set(location.accessibility_features)
        
        if needs_set.issubset(features_set):
            return 1.0
        
        matches = len(needs_set.intersection(features_set))
        return matches / len(needs_set) if needs_set else 1.0
    
    def _calculate_activity_score(
        self,
        location: LocationAttributes,
        preferences: UserPreferences
    ) -> float:
        """Calculate activity level compatibility"""
        activity_intensities = {
            'museum': 'low',
            'park': 'medium',
            'hiking': 'high',
            'beach': 'low',
            'shopping': 'medium'
        }
        
        location_intensity = activity_intensities.get(location.category, 'medium')
        
        if location_intensity == preferences.activity_level:
            return 1.0
        elif (location_intensity == 'low' and preferences.activity_level == 'medium') or \
             (location_intensity == 'medium' and preferences.activity_level == 'high'):
            return 0.7
        else:
            return 0.4


class TimeAwarePlanner:
    """Time-aware route planning"""
    
    def __init__(self):
        logger.info("⏰ Time-Aware Planner initialized")
    
    def optimize_route_by_time(
        self,
        locations: List[LocationAttributes],
        time_available_hours: float,
        start_time: datetime,
        preferences: UserPreferences
    ) -> Tuple[List[LocationAttributes], Dict[str, Any]]:
        """Optimize route to fit within available time"""
        
        total_time_minutes = time_available_hours * 60
        selected_locations = []
        total_time_used = 0
        time_breakdown = {
            'visit_time': 0,
            'travel_time': 0,
            'buffer_time': 0
        }
        
        # Sort locations by priority score
        scored_locations = [(self._calculate_priority_score(loc, start_time), loc) 
                           for loc in locations]
        scored_locations.sort(reverse=True, key=lambda x: x[0])
        
        current_time = start_time
        prev_location = None
        
        for score, location in scored_locations:
            # Calculate time for this location
            visit_time = location.average_visit_duration_minutes
            travel_time = 0
            
            if prev_location:
                travel_time = self._estimate_travel_time(prev_location, location)
            
            buffer_time = 15  # Buffer between locations
            
            total_segment_time = visit_time + travel_time + buffer_time
            
            # Check if we have time
            if total_time_used + total_segment_time <= total_time_minutes:
                # Check if location is open at this time
                if self._is_location_open(location, current_time):
                    selected_locations.append(location)
                    total_time_used += total_segment_time
                    time_breakdown['visit_time'] += visit_time
                    time_breakdown['travel_time'] += travel_time
                    time_breakdown['buffer_time'] += buffer_time
                    
                    current_time += timedelta(minutes=total_segment_time)
                    prev_location = location
            else:
                break  # No more time available
        
        return selected_locations, time_breakdown
    
    def _calculate_priority_score(
        self,
        location: LocationAttributes,
        current_time: datetime
    ) -> float:
        """Calculate priority score for location"""
        score = 0.5
        
        # Check best time of day
        current_period = self._get_time_period(current_time)
        if current_period in location.best_time_of_day:
            score += 0.3
        
        # Prefer locations with booking if early in day
        if location.requires_booking and current_time.hour < 12:
            score += 0.2
        
        # Photo opportunities during good lighting
        if 9 <= current_time.hour <= 17:
            score += location.photo_opportunities * 0.05
        
        return min(1.0, score)
    
    def _get_time_period(self, dt: datetime) -> TimeOfDay:
        """Get time period from datetime"""
        hour = dt.hour
        if 6 <= hour < 9:
            return TimeOfDay.EARLY_MORNING
        elif 9 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 20:
            return TimeOfDay.EVENING
        elif 20 <= hour < 24:
            return TimeOfDay.NIGHT
        else:
            return TimeOfDay.LATE_NIGHT
    
    def _is_location_open(
        self,
        location: LocationAttributes,
        check_time: datetime
    ) -> bool:
        """Check if location is open at given time"""
        # Simplified - in production, use real opening hours
        hour = check_time.hour
        
        if location.category == 'museum':
            return 9 <= hour < 17
        elif location.category == 'restaurant':
            return 11 <= hour < 23
        elif location.category == 'cafe':
            return 8 <= hour < 22
        elif location.category == 'park':
            return True  # Always open
        else:
            return 9 <= hour < 20
    
    def _estimate_travel_time(
        self,
        from_loc: LocationAttributes,
        to_loc: LocationAttributes
    ) -> int:
        """Estimate travel time between locations (minutes)"""
        # Calculate distance
        distance = self._calculate_distance(
            from_loc.location.latitude,
            from_loc.location.longitude,
            to_loc.location.latitude,
            to_loc.location.longitude
        )
        
        # Assume average speed of 20 km/h for mixed transport
        time_hours = distance / 20
        return int(time_hours * 60) + 5  # Add 5 min buffer
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points (Haversine formula)"""
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


class MultiStopOptimizer:
    """Multi-stop route optimization using graph algorithms"""
    
    def __init__(self):
        logger.info("🗺 Multi-Stop Optimizer initialized")
    
    def optimize_route_order(
        self,
        locations: List[LocationAttributes],
        start_location: GPSLocation,
        transport_mode: TransportMode,
        preferences: UserPreferences
    ) -> List[LocationAttributes]:
        """Optimize visit order for multiple stops"""
        
        if len(locations) <= 2:
            return locations
        
        # Use nearest neighbor heuristic for speed
        # For production, could use more sophisticated algorithms
        return self._nearest_neighbor_tsp(locations, start_location, transport_mode)
    
    def _nearest_neighbor_tsp(
        self,
        locations: List[LocationAttributes],
        start: GPSLocation,
        transport_mode: TransportMode
    ) -> List[LocationAttributes]:
        """Nearest neighbor algorithm for TSP"""
        if not locations:
            return []
        
        unvisited = set(range(len(locations)))
        route = []
        current_pos = start
        
        while unvisited:
            nearest_idx = None
            nearest_dist = float('inf')
            
            for idx in unvisited:
                dist = self._calculate_distance(
                    current_pos.latitude,
                    current_pos.longitude,
                    locations[idx].location.latitude,
                    locations[idx].location.longitude
                )
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = idx
            
            if nearest_idx is not None:
                route.append(locations[nearest_idx])
                unvisited.remove(nearest_idx)
                current_pos = locations[nearest_idx].location
        
        return route
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points"""
        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


class AIRecommendationEngine:
    """AI-powered recommendation engine"""
    
    def __init__(self):
        self.recommendation_cache = {}
        logger.info("🤖 AI Recommendation Engine initialized")
    
    async def generate_recommendations(
        self,
        user_preferences: UserPreferences,
        weather: WeatherContext,
        time_context: datetime,
        current_location: GPSLocation
    ) -> List[str]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        # Weather-based recommendations
        weather_recs = self._get_weather_recommendations(weather, user_preferences)
        recommendations.extend(weather_recs)
        
        # Time-based recommendations
        time_recs = self._get_time_recommendations(time_context, user_preferences)
        recommendations.extend(time_recs)
        
        # Mood-based recommendations
        if user_preferences.mood:
            mood_recs = self._get_mood_recommendations(user_preferences.mood)
            recommendations.extend(mood_recs)
        
        # Location-based recommendations
        location_recs = self._get_location_recommendations(current_location)
        recommendations.extend(location_recs)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _get_weather_recommendations(
        self,
        weather: WeatherContext,
        preferences: UserPreferences
    ) -> List[str]:
        """Generate weather-based recommendations"""
        recs = []
        
        if weather.condition == WeatherCondition.SUNNY and weather.temperature_celsius < 25:
            recs.append("🌞 Perfect weather for a Bosphorus ferry tour - enjoy the scenic views!")
            recs.append("🚶 Great day for walking the historic peninsula - comfortable temperature")
            
        elif weather.condition == WeatherCondition.RAINY:
            recs.append("☔ Visit world-class museums today - Topkapi Palace has extensive indoor exhibits")
            recs.append("☕ Perfect weather for Istanbul's cozy cafes and covered bazaars")
            
        elif weather.temperature_celsius > 30:
            recs.append("🌊 Head to the waterfront - Ortaköy and Bebek offer cool breezes")
            recs.append("❄️ Visit air-conditioned museums during peak heat hours (12-4 PM)")
            
        elif weather.condition == WeatherCondition.WINDY:
            recs.append("🏛 Explore indoor attractions today - Grand Bazaar is covered and wind-free")
            
        return recs
    
    def _get_time_recommendations(
        self,
        current_time: datetime,
        preferences: UserPreferences
    ) -> List[str]:
        """Generate time-based recommendations"""
        recs = []
        hour = current_time.hour
        
        if 6 <= hour < 9:
            recs.append("🌅 Early morning at Blue Mosque - fewer crowds and beautiful morning light")
            recs.append("🥐 Start with a traditional Turkish breakfast in Sultanahmet")
            
        elif 12 <= hour < 14:
            recs.append("🍽 Lunch time! Try local restaurants near your current route")
            
        elif 17 <= hour < 19:
            recs.append("🌆 Sunset views from Galata Tower or Pierre Loti Hill")
            
        elif hour >= 20:
            recs.append("🌃 Evening Bosphorus cruise with dinner and lights")
            recs.append("🎭 Check out evening cultural performances in Beyoğlu")
        
        return recs
    
    def _get_mood_recommendations(self, mood: UserMood) -> List[str]:
        """Generate mood-based recommendations"""
        mood_recs = {
            UserMood.ROMANTIC: [
                "💑 Sunset at Çamlıca Hill with panoramic city views",
                "🌹 Romantic dinner cruise on the Bosphorus",
                "🏰 Stroll through Gülhane Park's rose gardens"
            ],
            UserMood.ADVENTUROUS: [
                "🏃 Explore the hidden cisterns and underground passages",
                "🚴 Bike the Bosphorus waterfront from Ortaköy to Bebek",
                "🎿 If winter, try snow activities in Belgrad Forest"
            ],
            UserMood.RELAXED: [
                "🧘 Visit Sultanahmet's peaceful gardens",
                "☕ Spend afternoon in traditional tea gardens",
                "🛀 Experience authentic Turkish bath (hammam)"
            ],
            UserMood.CULTURAL: [
                "🏛 Explore lesser-known museums like Pera Museum",
                "🎨 Visit contemporary art galleries in Karaköy",
                "📚 Discover historic libraries and manuscripts"
            ],
            UserMood.FOODIE: [
                "🍴 Food tour in Kadıköy's street food scene",
                "🐟 Fresh fish at Kumkapı waterfront restaurants",
                "🥙 Authentic kebabs in historic Fatih district"
            ]
        }
        
        return mood_recs.get(mood, [])
    
    def _get_location_recommendations(self, location: GPSLocation) -> List[str]:
        """Generate location-based recommendations"""
        # Simplified - in production, use intelligent location detector
        recs = []
        
        lat, lon = location.latitude, location.longitude
        
        # Sultanahmet area
        if 28.97 <= lon <= 28.98 and 41.00 <= lat <= 41.01:
            recs.append("📍 You're in Sultanahmet! Don't miss the Basilica Cistern")
            
        # Beyoğlu area
        elif 28.97 <= lon <= 28.98 and 41.03 <= lat <= 41.04:
            recs.append("📍 Explore Istiklal Street and hidden passages nearby")
        
        return recs


# ==================== MAIN ENHANCED ROUTE PLANNER ====================

class EnhancedRoutePlannerV2:
    """
    Enhanced Route Planner V2 with all advanced features
    """
    
    def __init__(self):
        # Initialize sub-systems
        self.weather_client = None
        if WEATHER_CLIENT_AVAILABLE:
            self.weather_client = EnhancedWeatherClient()
        
        self.weather_router = WeatherAwareRouter(self.weather_client)
        self.preference_router = PreferenceBasedRouter()
        self.time_planner = TimeAwarePlanner()
        self.multi_stop_optimizer = MultiStopOptimizer()
        self.ai_engine = AIRecommendationEngine()
        
        # Initialize original planner for backward compatibility
        self.original_planner = None
        if ORIGINAL_PLANNER_AVAILABLE:
            self.original_planner = EnhancedGPSRoutePlanner()
        
        # Initialize intelligent location detector
        self.location_detector = None
        if INTELLIGENT_LOCATION_AVAILABLE:
            self.location_detector = IntelligentLocationDetector()
        
        # Mock location database (in production, use real database)
        self.location_database = self._initialize_location_database()
        
        logger.info("🚀 Enhanced Route Planner V2 initialized with all features")
    
    async def plan_route(
        self,
        user_id: str,
        start_location: GPSLocation,
        preferences: UserPreferences,
        context: Optional[Dict[str, Any]] = None
    ) -> RouteOptimizationResult:
        """
        Plan an optimized route with all enhancements
        """
        logger.info(f"🗺 Planning enhanced route for user {user_id}")
        
        # Get weather context
        weather = await self.weather_router.get_current_weather_context()
        logger.info(f"Weather: {weather.condition.value}, {weather.temperature_celsius}°C")
        
        # Get time context
        current_time = datetime.now()
        
        # Step 1: Filter locations by weather
        candidate_locations = self.weather_router.filter_locations_by_weather(
            self.location_database,
            weather,
            preferences
        )
        logger.info(f"Weather filtering: {len(candidate_locations)} suitable locations")
        
        # Step 2: Score by preferences
        preference_scored = []
        for location in candidate_locations:
            score = self.preference_router.score_location_by_preferences(location, preferences)
            if score > 0.4:  # Threshold
                preference_scored.append((score, location))
        
        preference_scored.sort(reverse=True, key=lambda x: x[0])
        top_locations = [loc for score, loc in preference_scored[:20]]
        logger.info(f"Preference scoring: {len(top_locations)} top matches")
        
        # Step 3: Time-aware optimization
        optimized_locations, time_breakdown = self.time_planner.optimize_route_by_time(
            top_locations,
            preferences.time_available_hours,
            current_time,
            preferences
        )
        logger.info(f"Time optimization: {len(optimized_locations)} locations fit in {preferences.time_available_hours}h")
        
        # Step 4: Multi-stop optimization
        if preferences.preferred_transport_modes:
            transport_mode = preferences.preferred_transport_modes[0]
        else:
            suitable_modes = self.weather_router.select_transport_mode_by_weather(
                weather,
                list(TransportMode),
                preferences
            )
            transport_mode = suitable_modes[0] if suitable_modes else TransportMode.WALKING
        
        ordered_locations = self.multi_stop_optimizer.optimize_route_order(
            optimized_locations,
            start_location,
            transport_mode,
            preferences
        )
        logger.info(f"Route optimization: Visit order optimized")
        
        # Step 5: Generate AI recommendations
        recommendations = await self.ai_engine.generate_recommendations(
            preferences,
            weather,
            current_time,
            start_location
        )
        
        # Step 6: Generate warnings
        warnings = self._generate_warnings(weather, ordered_locations, preferences)
        
        # Build route (convert to PersonalizedRoute if original planner available)
        if self.original_planner and ORIGINAL_PLANNER_AVAILABLE:
            # Use original planner's structure
            route = await self._build_personalized_route(
                user_id,
                start_location,
                ordered_locations,
                transport_mode,
                preferences
            )
        else:
            route = None
        
        # Calculate optimization scores
        optimization_result = RouteOptimizationResult(
            route=route,
            optimization_score=self._calculate_optimization_score(ordered_locations, preferences),
            weather_compatibility=self._calculate_weather_compatibility(ordered_locations, weather),
            preference_match=sum(self.preference_router.score_location_by_preferences(loc, preferences) 
                                for loc in ordered_locations) / len(ordered_locations) if ordered_locations else 0,
            time_efficiency=time_breakdown['visit_time'] / (preferences.time_available_hours * 60),
            cost_efficiency=self._calculate_cost_efficiency(ordered_locations, preferences),
            recommendations=recommendations,
            warnings=warnings
        )
        
        logger.info(f"✅ Route planning complete! Score: {optimization_result.optimization_score:.2f}")
        return optimization_result
    
    async def _build_personalized_route(
        self,
        user_id: str,
        start_location: GPSLocation,
        locations: List[LocationAttributes],
        transport_mode: TransportMode,
        preferences: UserPreferences
    ) -> PersonalizedRoute:
        """Build PersonalizedRoute object"""
        # Convert LocationAttributes to PersonalizedWaypoint
        waypoints = []
        for loc in locations:
            waypoint = PersonalizedWaypoint(
                name=loc.name,
                location=loc.location,
                category=loc.category,
                interest_match=self.preference_router.score_location_by_preferences(loc, preferences),
                popularity_score=0.8,  # Mock
                weather_suitability=0.9,  # Mock
                accessibility_score=len(loc.accessibility_features) / 5.0 if loc.accessibility_features else 0.5,
                estimated_duration=loc.average_visit_duration_minutes,
                transport_modes=[transport_mode],
                personalization_reasons=[],
                real_time_updates={}
            )
            waypoints.append(waypoint)
        
        # Create route
        route = PersonalizedRoute(
            route_id=f"route_{user_id}_{int(time.time())}",
            user_id=user_id,
            start_location=start_location,
            waypoints=waypoints,
            segments=[],
            total_distance_km=0,
            total_time_minutes=0,
            total_cost=0,
            personalization_score=0,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            real_time_enabled=True
        )
        
        return route
    
    def _generate_warnings(
        self,
        weather: WeatherContext,
        locations: List[LocationAttributes],
        preferences: UserPreferences
    ) -> List[str]:
        """Generate warnings for the route"""
        warnings = []
        
        if weather.condition == WeatherCondition.RAINY:
            outdoor_count = sum(1 for loc in locations if loc.is_outdoor)
            if outdoor_count > 0:
                warnings.append(f"⚠️ {outdoor_count} outdoor locations in rainy weather - bring umbrella")
        
        if weather.temperature_celsius > 32:
            warnings.append("🌡 Very hot weather - stay hydrated and take breaks")
        
        if weather.temperature_celsius < 5:
            warnings.append("❄️ Cold weather - dress warmly")
        
        booking_needed = [loc.name for loc in locations if loc.requires_booking]
        if booking_needed:
            warnings.append(f"📅 Booking recommended: {', '.join(booking_needed)}")
        
        return warnings
    
    def _calculate_optimization_score(
        self,
        locations: List[LocationAttributes],
        preferences: UserPreferences
    ) -> float:
        """Calculate overall optimization score"""
        if not locations:
            return 0.0
        
        scores = [
            self.preference_router.score_location_by_preferences(loc, preferences)
            for loc in locations
        ]
        return sum(scores) / len(scores)
    
    def _calculate_weather_compatibility(
        self,
        locations: List[LocationAttributes],
        weather: WeatherContext
    ) -> float:
        """Calculate weather compatibility score"""
        if not locations:
            return 0.0
        
        if weather.is_suitable_for_outdoor():
            return 1.0
        
        indoor_count = sum(1 for loc in locations if loc.is_indoor)
        return indoor_count / len(locations)
    
    def _calculate_cost_efficiency(
        self,
        locations: List[LocationAttributes],
        preferences: UserPreferences
    ) -> float:
        """Calculate cost efficiency"""
        total_cost = sum(loc.entrance_fee_tl for loc in locations)
        
        if total_cost == 0:
            return 1.0
        
        budget = preferences.budget_per_person
        if total_cost <= budget:
            return 1.0 - (total_cost / budget)
        else:
            return 0.5  # Over budget
    
    def _initialize_location_database(self) -> List[LocationAttributes]:
        """Initialize mock location database"""
        # In production, load from real database
        locations = [
            LocationAttributes(
                name="Hagia Sophia",
                location=GPSLocation(41.0086, 28.9802, 10.0),
                category="museum",
                subcategory="historic",
                is_indoor=True,
                is_outdoor=False,
                average_visit_duration_minutes=90,
                entrance_fee_tl=100,
                tags=["historic", "architecture", "cultural", "unesco"],
                photo_opportunities=10,
                best_time_of_day=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON]
            ),
            LocationAttributes(
                name="Blue Mosque",
                location=GPSLocation(41.0055, 28.9769, 10.0),
                category="mosque",
                subcategory="historic",
                is_indoor=True,
                is_outdoor=True,
                has_garden=True,
                average_visit_duration_minutes=45,
                entrance_fee_tl=0,
                tags=["historic", "architecture", "cultural", "religious"],
                photo_opportunities=9,
                best_time_of_day=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON]
            ),
            LocationAttributes(
                name="Topkapi Palace",
                location=GPSLocation(41.0115, 28.9833, 10.0),
                category="museum",
                subcategory="palace",
                is_indoor=True,
                is_outdoor=True,
                has_garden=True,
                average_visit_duration_minutes=150,
                entrance_fee_tl=200,
                requires_booking=True,
                tags=["historic", "palace", "cultural", "garden"],
                photo_opportunities=10,
                best_time_of_day=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON]
            ),
            LocationAttributes(
                name="Grand Bazaar",
                location=GPSLocation(41.0108, 28.9680, 10.0),
                category="shopping",
                subcategory="bazaar",
                is_indoor=True,
                average_visit_duration_minutes=120,
                entrance_fee_tl=0,
                tags=["shopping", "historic", "bazaar", "cultural"],
                photo_opportunities=8,
                best_time_of_day=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON]
            ),
            LocationAttributes(
                name="Bosphorus Ferry Tour",
                location=GPSLocation(41.0250, 28.9738, 10.0),
                category="tour",
                subcategory="ferry",
                is_outdoor=True,
                is_waterfront=True,
                average_visit_duration_minutes=90,
                entrance_fee_tl=50,
                tags=["scenic", "waterfront", "tour", "bosphorus"],
                photo_opportunities=10,
                best_time_of_day=[TimeOfDay.AFTERNOON, TimeOfDay.EVENING]
            ),
            LocationAttributes(
                name="Galata Tower",
                location=GPSLocation(41.0256, 28.9744, 10.0),
                category="viewpoint",
                subcategory="tower",
                is_indoor=True,
                average_visit_duration_minutes=60,
                entrance_fee_tl=150,
                tags=["historic", "viewpoint", "panoramic", "architecture"],
                photo_opportunities=10,
                best_time_of_day=[TimeOfDay.AFTERNOON, TimeOfDay.EVENING]
            ),
            LocationAttributes(
                name="Basilica Cistern",
                location=GPSLocation(41.0084, 28.9778, 10.0),
                category="museum",
                subcategory="historic",
                is_indoor=True,
                average_visit_duration_minutes=45,
                entrance_fee_tl=90,
                tags=["historic", "underground", "architecture", "cool"],
                photo_opportunities=9,
                best_time_of_day=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON, TimeOfDay.EVENING]
            ),
            LocationAttributes(
                name="Ortaköy",
                location=GPSLocation(41.0550, 29.0264, 10.0),
                category="neighborhood",
                subcategory="waterfront",
                is_outdoor=True,
                is_waterfront=True,
                has_cafe=True,
                average_visit_duration_minutes=90,
                entrance_fee_tl=0,
                tags=["waterfront", "cafe", "scenic", "relaxed", "bosphorus"],
                photo_opportunities=9,
                best_time_of_day=[TimeOfDay.AFTERNOON, TimeOfDay.EVENING]
            ),
            LocationAttributes(
                name="Princes' Islands",
                location=GPSLocation(40.8600, 29.1167, 10.0),
                category="island",
                subcategory="nature",
                is_outdoor=True,
                is_waterfront=True,
                average_visit_duration_minutes=300,
                entrance_fee_tl=50,
                tags=["nature", "island", "scenic", "relaxed", "beach"],
                photo_opportunities=10,
                best_time_of_day=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON]
            ),
            LocationAttributes(
                name="Dolmabahçe Palace",
                location=GPSLocation(41.0391, 29.0003, 10.0),
                category="museum",
                subcategory="palace",
                is_indoor=True,
                is_outdoor=True,
                has_garden=True,
                is_waterfront=True,
                average_visit_duration_minutes=120,
                entrance_fee_tl=150,
                requires_booking=True,
                tags=["historic", "palace", "cultural", "architecture", "waterfront"],
                photo_opportunities=10,
                best_time_of_day=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON]
            )
        ]
        
        return locations


# ==================== TESTING & DEMO ====================

async def demo_enhanced_planner():
    """Demo the enhanced route planner"""
    print("🚀 Enhanced Route Planner V2 Demo\n")
    
    planner = EnhancedRoutePlannerV2()
    
    # User preferences
    preferences = UserPreferences(
        interests=["historic", "cultural", "architecture"],
        mood=UserMood.CULTURAL,
        preferred_transport_modes=[TransportMode.WALKING, TransportMode.METRO],
        budget_per_person=500.0,
        time_available_hours=6.0,
        prefer_scenic=True,
        activity_level="medium"
    )
    
    # Start location (Sultanahmet)
    start_location = GPSLocation(41.0082, 28.9784, 10.0)
    
    # Plan route
    result = await planner.plan_route(
        user_id="demo_user",
        start_location=start_location,
        preferences=preferences
    )
    
    # Print results
    print(f"📊 Optimization Results:")
    print(f"  Overall Score: {result.optimization_score:.2f}")
    print(f"  Weather Compatibility: {result.weather_compatibility:.2f}")
    print(f"  Preference Match: {result.preference_match:.2f}")
    print(f"  Time Efficiency: {result.time_efficiency:.2f}")
    print(f"  Cost Efficiency: {result.cost_efficiency:.2f}")
    
    print(f"\n🤖 AI Recommendations:")
    for rec in result.recommendations[:5]:
        print(f"  {rec}")
    
    if result.warnings:
        print(f"\n⚠️ Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")
    
    if result.route:
        print(f"\n🗺 Route Details:")
        print(f"  Total Locations: {len(result.route.waypoints)}")
        for i, waypoint in enumerate(result.route.waypoints, 1):
            print(f"  {i}. {waypoint.name} ({waypoint.category}) - {waypoint.estimated_duration}min")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(demo_enhanced_planner())
