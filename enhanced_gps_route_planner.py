#!/usr/bin/env python3
"""
Enhanced GPS-Based Route Planner for AI Istanbul
Advanced location-aware route planning with real-time personalization
"""

import json
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio

# Import fallback location detection
try:
    from fallback_location_detector import (
        FallbackLocationDetector, LocationFallbackOption, 
        LocationDetectionMethod, UserLocationPrompt, get_location_prompts
    )
    FALLBACK_DETECTOR_AVAILABLE = True
except ImportError:
    FALLBACK_DETECTOR_AVAILABLE = False
    logging.warning("Fallback location detector not available")

# Import Intelligent Location Detector from main system
try:
    from istanbul_ai.services.intelligent_location_detector import (
        IntelligentLocationDetector,
        LocationDetectionResult,
        GPSContext,
        WeatherContext,
        EventContext,
        TransportationContext
    )
    INTELLIGENT_LOCATION_AVAILABLE = True
    logging.info("✅ Intelligent Location Detector integrated successfully!")
except ImportError:
    INTELLIGENT_LOCATION_AVAILABLE = False
    logging.warning("⚠️ Intelligent Location Detector not available")

logger = logging.getLogger(__name__)

class TransportMode(Enum):
    """Transportation modes with specific characteristics"""
    WALKING = {
        "speed_kmh": 4.5,
        "cost_per_km": 0.0,
        "accessibility": "high",
        "weather_dependent": True,
        "scenic_factor": 0.9
    }
    CYCLING = {
        "speed_kmh": 15.0,
        "cost_per_km": 0.0,
        "accessibility": "medium",
        "weather_dependent": True,
        "scenic_factor": 0.8
    }
    DRIVING = {
        "speed_kmh": 25.0,
        "cost_per_km": 0.8,
        "accessibility": "low",
        "weather_dependent": False,
        "scenic_factor": 0.4
    }
    PUBLIC_TRANSPORT = {
        "speed_kmh": 20.0,
        "cost_per_km": 0.15,
        "accessibility": "high",
        "weather_dependent": False,
        "scenic_factor": 0.3
    }
    METRO = {
        "speed_kmh": 35.0,
        "cost_per_km": 0.12,
        "accessibility": "high",
        "weather_dependent": False,
        "scenic_factor": 0.1
    }
    FERRY = {
        "speed_kmh": 18.0,
        "cost_per_km": 0.20,
        "accessibility": "medium",
        "weather_dependent": True,
        "scenic_factor": 1.0
    }

class RoutePersonalizationLevel(Enum):
    """Levels of route personalization"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    AI_POWERED = "ai_powered"
    REAL_TIME = "real_time"

@dataclass
class GPSLocation:
    """GPS location with metadata"""
    latitude: float
    longitude: float
    accuracy: float = 10.0  # meters
    timestamp: datetime = None
    address: str = ""
    district: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PersonalizedWaypoint:
    """Enhanced waypoint with personalization data"""
    name: str
    location: GPSLocation
    category: str
    interest_match: float  # 0.0 to 1.0
    popularity_score: float
    weather_suitability: float
    accessibility_score: float
    estimated_duration: int  # minutes
    transport_modes: List[TransportMode]
    personalization_reasons: List[str]
    real_time_updates: Dict[str, Any]

@dataclass
class RouteSegment:
    """Individual segment of a route"""
    from_point: GPSLocation
    to_point: GPSLocation
    transport_mode: TransportMode
    distance_km: float
    estimated_time_minutes: int
    cost_estimate: float
    scenic_score: float
    accessibility_score: float
    real_time_conditions: Dict[str, Any]

@dataclass
class PersonalizedRoute:
    """Complete personalized route"""
    route_id: str
    user_id: str
    start_location: GPSLocation
    waypoints: List[PersonalizedWaypoint]
    segments: List[RouteSegment]
    total_distance_km: float
    total_time_minutes: int
    total_cost: float
    personalization_score: float
    created_at: datetime
    last_updated: datetime
    real_time_enabled: bool

class EnhancedGPSRoutePlanner:
    """
    Advanced GPS-based route planner with real-time personalization
    """
    
    def __init__(self):
        self.ml_cache = self._initialize_ml_cache()
        self.user_profiles = {}
        self.active_routes = {}
        self.real_time_monitoring = {}
        
        # Initialize Intelligent Location Detector (PRIMARY)
        self.intelligent_location_detector = None
        if INTELLIGENT_LOCATION_AVAILABLE:
            try:
                self.intelligent_location_detector = IntelligentLocationDetector()
                logger.info("📍 Intelligent Location Detector initialized for GPS route planning")
            except Exception as e:
                logger.warning(f"Could not initialize Intelligent Location Detector: {e}")
                self.intelligent_location_detector = None
        
        # Initialize fallback location detection (SECONDARY)
        self.fallback_detector = None
        if FALLBACK_DETECTOR_AVAILABLE:
            self.fallback_detector = FallbackLocationDetector()
            logger.info("🔄 Fallback Location Detector initialized")
        
        # Initialize Istanbul-specific data
        self.istanbul_districts = self._load_istanbul_districts()
        self.transport_network = self._load_transport_network()
        self.poi_database = self._load_poi_database()
        
        # District local tips database
        self.district_tips = {
            'sultanahmet': [
                "Visit early morning (8-9 AM) to avoid crowds at major museums",
                "Wear comfortable shoes as the historic peninsula involves lots of walking on cobblestones",
                "Many museums offer combined tickets that can save money and time",
                "The district is very walkable - most attractions are within 10-15 minutes of each other",
                "Friday prayer times (around 12:30 PM) can affect visiting hours at some mosques"
            ],
            'beyoglu': [
                "Istiklal Street is pedestrian-only and perfect for walking",
                "Galata Tower area offers some of the best views of the city",
                "The historic tram (nostalgic tram) runs along Istiklal Street",
                "Many art galleries and cultural centers are free to visit",
                "Evening visits offer beautiful city lights and vibrant nightlife"
            ],
            'besiktas': [
                "Dolmabahce Palace requires timed entry tickets - book in advance",
                "The Bosphorus waterfront promenade is perfect for jogging or walking",
                "Ferry connections from Besiktas pier offer scenic Bosphorus views",
                "Many football (soccer) fans gather here on match days",
                "The Naval Museum showcases Turkey's rich maritime history"
            ],
            'kadikoy': [
                "Take the ferry from European side for a scenic journey",
                "The Asian side offers a more local, less touristy experience",
                "Moda neighborhood has beautiful seaside cafes and walking paths",
                "Tuesday and Saturday markets offer fresh local produce",
                "Great fish restaurants along the waterfront with Bosphorus views"
            ],
            'eminonu': [
                "Spice Bazaar is less crowded than Grand Bazaar but equally authentic",
                "Ferry terminal connects to all major Bosphorus destinations",
                "Street food around the area offers authentic Turkish flavors",
                "Galata Bridge offers great fishing spots and restaurant views underneath",
                "Morning fish market provides fresh seafood and local atmosphere"
            ],
            'fatih': [
                "Grand Bazaar closes on Sundays - plan accordingly",
                "Suleymaniye Mosque offers panoramic city views from its courtyard",
                "Traditional Turkish baths (hammams) provide authentic cultural experience",
                "Many budget-friendly restaurants serve traditional Ottoman cuisine",
                "Conservative dress code recommended when visiting mosques"
            ]
        }
        
        logger.info("🚀 Enhanced GPS Route Planner initialized")
    
    def _initialize_ml_cache(self):
        """Initialize ML cache for route optimization"""
        try:
            from ml_result_cache import get_ml_cache
            cache = get_ml_cache()
            
            # Add GPS-specific cache TTL
            cache.cache_ttl.update({
                'gps_route_planning': timedelta(minutes=30),  # GPS routes change frequently
                'real_time_updates': timedelta(minutes=5),    # Real-time data is very volatile
                'transport_optimization': timedelta(hours=2), # Transport conditions change
                'location_recommendations': timedelta(hours=4) # Location-based suggestions
            })
            
            return cache
        except Exception as e:
            logger.warning(f"ML cache not available: {e}")
            return None
    
    def _load_istanbul_districts(self) -> Dict[str, Any]:
        """Load Istanbul district data with GPS boundaries"""
        return {
            'sultanahmet': {
                'center': GPSLocation(41.0082, 28.9784),
                'radius_km': 2.0,
                'attractions_density': 'very_high',
                'transport_connectivity': 'excellent'
            },
            'beyoglu': {
                'center': GPSLocation(41.0369, 28.9744),
                'radius_km': 3.0,
                'attractions_density': 'high',
                'transport_connectivity': 'excellent'
            },
            'kadikoy': {
                'center': GPSLocation(40.9907, 29.0205),
                'radius_km': 4.0,
                'attractions_density': 'medium',
                'transport_connectivity': 'good'
            },
            'besiktas': {
                'center': GPSLocation(41.0422, 29.0084),
                'radius_km': 3.5,
                'attractions_density': 'high',
                'transport_connectivity': 'excellent'
            }
        }
    
    def _load_transport_network(self) -> Dict[str, Any]:
        """Load Istanbul transport network data"""
        return {
            'metro_lines': {
                'm1': {'color': 'red', 'stations': 23, 'operational': True},
                'm2': {'color': 'green', 'stations': 16, 'operational': True},
                'm3': {'color': 'blue', 'stations': 11, 'operational': True},
                'm4': {'color': 'pink', 'stations': 19, 'operational': True}
            },
            'ferry_routes': {
                'kadikoy_eminonu': {'duration_min': 25, 'frequency_min': 15},
                'besiktas_kadikoy': {'duration_min': 30, 'frequency_min': 20},
                'karakoy_kadikoy': {'duration_min': 20, 'frequency_min': 10}
            },
            'bus_network': {
                'coverage': 'comprehensive',
                'avg_frequency_min': 8,
                'night_service': True
            }
        }
    
    def _load_poi_database(self) -> Dict[str, Any]:
        """Load Points of Interest database with GPS coordinates"""
        return {
            'museums': [
                {
                    'name': 'Hagia Sophia',
                    'location': GPSLocation(41.0086, 28.9802),
                    'category': 'historical',
                    'popularity': 0.95,
                    'visit_duration_min': 90,
                    'district': 'sultanahmet'
                },
                {
                    'name': 'Topkapi Palace',
                    'location': GPSLocation(41.0115, 28.9833),
                    'category': 'palace',
                    'popularity': 0.92,
                    'visit_duration_min': 120,
                    'district': 'sultanahmet'
                },
                {
                    'name': 'Istanbul Archaeological Museums',
                    'location': GPSLocation(41.0117, 28.9813),
                    'category': 'archaeology',
                    'popularity': 0.85,
                    'visit_duration_min': 100,
                    'district': 'sultanahmet'
                },
                {
                    'name': 'Pera Museum',
                    'location': GPSLocation(41.0316, 28.9750),
                    'category': 'art',
                    'popularity': 0.82,
                    'visit_duration_min': 80,
                    'district': 'beyoglu'
                },
                {
                    'name': 'Dolmabahce Palace',
                    'location': GPSLocation(41.0391, 29.0000),
                    'category': 'palace',
                    'popularity': 0.90,
                    'visit_duration_min': 110,
                    'district': 'besiktas'
                },
                {
                    'name': 'Naval Museum',
                    'location': GPSLocation(41.0426, 29.0077),
                    'category': 'maritime',
                    'popularity': 0.78,
                    'visit_duration_min': 75,
                    'district': 'besiktas'
                },
                {
                    'name': 'Moda Museum',
                    'location': GPSLocation(40.9856, 29.0254),
                    'category': 'local_history',
                    'popularity': 0.72,
                    'visit_duration_min': 60,
                    'district': 'kadikoy'
                }
            ],
            'restaurants': [
                {
                    'name': 'Pandeli',
                    'location': GPSLocation(41.0171, 28.9700),
                    'category': 'ottoman_cuisine',
                    'popularity': 0.88,
                    'avg_meal_duration_min': 75,
                    'district': 'eminonu'
                }
            ],
            'viewpoints': [
                {
                    'name': 'Galata Tower',
                    'location': GPSLocation(41.0256, 28.9744),
                    'category': 'panoramic_view',
                    'popularity': 0.89,
                    'visit_duration_min': 45,
                    'district': 'beyoglu'
                }
            ]
        }
    
    async def create_personalized_route(
        self,
        user_id: str,
        current_location: GPSLocation,
        preferences: Dict[str, Any],
        constraints: Dict[str, Any] = None
    ) -> PersonalizedRoute:
        """
        Create a personalized route based on GPS location and user preferences
        
        Args:
            user_id: Unique user identifier
            current_location: User's current GPS location
            preferences: User preferences (interests, transport modes, etc.)
            constraints: Route constraints (time, budget, radius)
        
        Returns:
            PersonalizedRoute object with optimized waypoints and segments
        """
        
        logger.info(f"🗺️ Creating personalized route for user {user_id} at {current_location.latitude:.4f}, {current_location.longitude:.4f}")
        
        # Check ML cache for similar routes
        cache_key = f"gps_route_{user_id}_{current_location.latitude:.3f}_{current_location.longitude:.3f}"
        if self.ml_cache:
            # Convert preferences to JSON-serializable format
            serializable_preferences = self._make_json_serializable(preferences)
            
            cached_route = self.ml_cache.get(
                cache_key,
                serializable_preferences,
                ['gps_route_planning', 'location_recommendations']
            )
            if cached_route:
                logger.info("🎯 Using cached personalized route")
                return self._deserialize_route(cached_route)
        
        # Get or create user profile
        user_profile = self._get_user_profile(user_id, preferences)
        
        # Find nearby points of interest
        nearby_pois = self._find_nearby_pois(
            current_location,
            radius_km=constraints.get('radius_km', 5.0),
            interests=preferences.get('interests', [])
        )
        
        # Score and rank POIs based on personalization
        scored_pois = self._score_pois_for_user(nearby_pois, user_profile, current_location)
        
        # Select optimal waypoints
        selected_waypoints = self._select_optimal_waypoints(
            scored_pois,
            constraints or {}
        )
        
        # Create route segments with transport optimization
        route_segments = await self._optimize_route_segments(
            current_location,
            selected_waypoints,
            preferences.get('transport_modes', [TransportMode.WALKING])
        )
        
        # Calculate route metrics
        total_distance = sum(segment.distance_km for segment in route_segments)
        total_time = sum(segment.estimated_time_minutes for segment in route_segments)
        total_cost = sum(segment.cost_estimate for segment in route_segments)
        
        # Create personalized route
        route = PersonalizedRoute(
            route_id=f"route_{user_id}_{int(time.time())}",
            user_id=user_id,
            start_location=current_location,
            waypoints=selected_waypoints,
            segments=route_segments,
            total_distance_km=total_distance,
            total_time_minutes=total_time,
            total_cost=total_cost,
            personalization_score=self._calculate_personalization_score(selected_waypoints, user_profile),
            created_at=datetime.now(),
            last_updated=datetime.now(),
            real_time_enabled=preferences.get('real_time_updates', False)
        )
        
        # Cache the route
        if self.ml_cache:
            serializable_preferences = self._make_json_serializable(preferences)
            self.ml_cache.set(
                cache_key,
                self._serialize_route(route),
                route.personalization_score,
                ['gps_route_planning', 'location_recommendations'],
                serializable_preferences
            )
        
        # Store active route for real-time updates
        self.active_routes[route.route_id] = route
        
        # Start real-time monitoring if enabled
        if route.real_time_enabled:
            await self._start_real_time_monitoring(route)
        
        logger.info(f"✅ Created personalized route: {len(selected_waypoints)} waypoints, {total_distance:.1f}km, {total_time}min")
        
        return route
    
    async def create_enhanced_route_response(
        self,
        user_id: str,
        current_location: GPSLocation,
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a route and return enhanced response with museums and local tips"""
        # Create the route
        route = await self.create_personalized_route(user_id, current_location, preferences)
        
        # Enhance response with museums and local tips
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        # Add recommendation summary
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'high' if route.personalization_score > 0.8 else 'medium' if route.personalization_score > 0.6 else 'basic',
            'route_optimization': 'GPS-optimized with real-time updates' if route.real_time_enabled else 'GPS-optimized'
        }
        
        return enhanced_response
    
    async def create_route_with_fallback_location(
        self,
        user_id: str,
        user_input: Optional[str] = None,
        user_context: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        session_data: Optional[Dict] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a personalized route using fallback location detection
        when GPS is not available or permission is denied
        
        Args:
            user_id: Unique user identifier
            user_input: User's location description
            user_context: Previous conversation context
            ip_address: User's IP address for geolocation
            session_data: Previous session data
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response with museums and local tips
        """
        if not self.fallback_detector:
            return {
                'error': 'Fallback location detection not available',
                'fallback_required': True,
                'location_prompts': self._get_basic_location_prompts()
            }
        
        # Detect user location using fallback methods
        location_options = await self.fallback_detector.detect_fallback_location(
            user_input, user_context, ip_address, session_data
        )
        
        if not location_options:
            return {
                'no_location_detected': True,
                'location_prompts': self._get_user_location_prompts(),
                'suggested_districts': list(self.istanbul_districts.keys()),
                'popular_landmarks': list(self.fallback_detector.landmark_locations.keys())
            }
        
        # Use the highest confidence location option
        best_option = location_options[0]
        
        if best_option.requires_user_confirmation:
            return {
                'location_confirmation_required': True,
                'detected_location': best_option.user_friendly_name,
                'confidence': best_option.confidence,
                'method': best_option.method.value,
                'alternative_options': [
                    {
                        'name': opt.user_friendly_name,
                        'confidence': opt.confidence,
                        'method': opt.method.value
                    } for opt in location_options[:3]
                ],
                'confirm_prompt': f"I detected your location as {best_option.user_friendly_name}. Is this correct?"
            }
        
        # Create route using detected location
        route = await self.create_personalized_route(
            user_id, 
            best_option.location, 
            preferences or {}
        )
        
        # Enhance response with museums and local tips
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        # Add fallback location information
        enhanced_response['location_detection'] = {
            'method': best_option.method.value,
            'confidence': best_option.confidence,
            'detected_location': best_option.user_friendly_name,
            'district': best_option.district,
            'neighborhood': best_option.neighborhood,
            'alternative_options_available': len(location_options) > 1
        }
        
        # Add recommendation summary
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'medium',  # Fallback locations get medium personalization
            'route_optimization': f'Location-optimized using {best_option.method.value}',
            'location_accuracy': 'high' if best_option.confidence > 0.8 else 'medium' if best_option.confidence > 0.6 else 'estimated'
        }
        
        return enhanced_response
    
    async def confirm_location_and_create_route(
        self,
        user_id: str,
        confirmed_location: GPSLocation,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create route after user confirms the detected location
        
        Args:
            user_id: Unique user identifier
            confirmed_location: User-confirmed GPS location
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response with museums and local tips
        """
        # Create route using confirmed location
        route = await self.create_personalized_route(
            user_id, 
            confirmed_location, 
            preferences or {}
        )
        
        # Enhance response with museums and local tips
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        # Add confirmation info
        enhanced_response['location_detection'] = {
            'method': 'user_confirmed',
            'confidence': 1.0,
            'detected_location': f"Confirmed location in {confirmed_location.district}",
            'district': confirmed_location.district
        }
        
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'high',  # Confirmed locations get high personalization
            'route_optimization': 'User-confirmed location with GPS optimization'
        }
        
        return enhanced_response
    
    def _get_user_location_prompts(self) -> List[Dict[str, Any]]:
        """Get user-friendly location input prompts"""
        if not self.fallback_detector:
            return self._get_basic_location_prompts()
        
        prompts = self.fallback_detector.generate_location_prompts()
        return [
            {
                'message': prompt.message,
                'input_type': prompt.input_type.value,
                'examples': prompt.examples,
                'suggestions': prompt.suggestions
            } for prompt in prompts
        ]
    
    def _get_basic_location_prompts(self) -> List[Dict[str, Any]]:
        """Get basic location prompts when fallback detector is not available"""
        return [
            {
                'message': "Could you tell me which district you're in?",
                'input_type': 'district_name',
                'examples': ["Sultanahmet", "Beyoğlu", "Kadıköy", "Beşiktaş"],
                'suggestions': list(self.istanbul_districts.keys())
            },
            {
                'message': "What landmark are you closest to?",
                'input_type': 'landmark',
                'examples': ["Galata Tower", "Topkapi Palace", "Taksim Square"],
                'suggestions': ["Sultanahmet", "Galata Tower", "Taksim", "Dolmabahce Palace"]
            }
        ]
    
    async def create_route_from_manual_input(
        self,
        user_id: str,
        location_description: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create route from manual location input when all other methods fail
        
        Args:
            user_id: Unique user identifier  
            location_description: User's manual location description
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response or location clarification request
        """
        if not self.fallback_detector:
            # Basic parsing without fallback detector
            location = self._parse_basic_location(location_description)
        else:
            # Use fallback detector for sophisticated parsing
            location_options = await self.fallback_detector.detect_fallback_location(
                user_input=location_description
            )
            
            if not location_options:
                return {
                    'parsing_failed': True,
                    'clarification_needed': True,
                    'message': "I couldn't identify your location. Could you be more specific?",
                    'suggestions': [
                        "Try mentioning a district name (e.g., 'Sultanahmet', 'Beyoğlu')",
                        "Name a landmark (e.g., 'near Galata Tower', 'close to Taksim Square')",
                        "Describe your area (e.g., 'in the old city', 'near the Bosphorus')"
                    ]
                }
            
            location = location_options[0].location
        
        if not location:
            return {
                'parsing_failed': True,
                'message': "Could you provide more details about your location?",
                'location_prompts': self._get_user_location_prompts()
            }
        
        # Create route using parsed location
        route = await self.create_personalized_route(user_id, location, preferences or {})
        
        # Enhance response
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        enhanced_response['location_detection'] = {
            'method': 'manual_input',
            'confidence': 0.7,  # Manual input gets medium confidence
            'detected_location': location_description,
            'district': location.district
        }
        
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'medium',
            'route_optimization': 'Manual location with intelligent routing'
        }
        
        return enhanced_response
    
    def _parse_basic_location(self, description: str) -> Optional[GPSLocation]:
        """Basic location parsing when fallback detector is not available"""
        desc_lower = description.lower().strip()
        
        # Check for district names
        for district, info in self.istanbul_districts.items():
            if district in desc_lower:
                return info['center']
        
        # Check for common landmarks
        landmarks = {
            'sultanahmet': GPSLocation(41.0082, 28.9784, district='sultanahmet'),
            'taksim': GPSLocation(41.0369, 28.9857, district='beyoglu'),
            'galata': GPSLocation(41.0256, 28.9744, district='beyoglu'),
            'kadikoy': GPSLocation(40.9907, 29.0205, district='kadikoy'),
            'besiktas': GPSLocation(41.0422, 29.0084, district='besiktas')
        }
        
        for landmark, location in landmarks.items():
            if landmark in desc_lower:
                return location
        
        return None
    
    async def _optimize_route_segments(
        self,
        start_location: GPSLocation,
        waypoints: List[PersonalizedWaypoint],
        preferred_transport_modes: List[TransportMode]
    ) -> List[RouteSegment]:
        """Optimize route segments between waypoints"""
        
        segments = []
        current_location = start_location
        
        for waypoint in waypoints:
            # Find best transport mode for this segment
            best_mode = self._select_best_transport_mode(
                current_location,
                waypoint.location,
                preferred_transport_modes,
                waypoint.transport_modes
            )
            
            # Calculate segment details
            distance = self._calculate_distance(current_location, waypoint.location)
            travel_time = self._calculate_travel_time(distance, best_mode)
            cost = self._calculate_segment_cost(distance, best_mode)
            scenic_score = self._calculate_scenic_score(current_location, waypoint.location, best_mode)
            accessibility_score = waypoint.accessibility_score
            
            segment = RouteSegment(
                from_point=current_location,
                to_point=waypoint.location,
                transport_mode=best_mode,
                distance_km=distance,
                estimated_time_minutes=travel_time,
                cost_estimate=cost,
                scenic_score=scenic_score,
                accessibility_score=accessibility_score,
                real_time_conditions={}
            )
            
            segments.append(segment)
            current_location = waypoint.location
        
        return segments
    
    def _select_best_transport_mode(
        self,
        from_loc: GPSLocation,
        to_loc: GPSLocation,
        preferred_modes: List[TransportMode],
        available_modes: List[TransportMode]
    ) -> TransportMode:
        """Select the best transport mode for a segment"""
        
        # Find intersection of preferred and available modes
        viable_modes = [mode for mode in preferred_modes if mode in available_modes]
        
        if not viable_modes:
            # Fall back to available modes
            viable_modes = available_modes
        
        if not viable_modes:
            # Ultimate fallback
            return TransportMode.WALKING
        
        # Score each viable mode
        distance = self._calculate_distance(from_loc, to_loc)
        best_mode = viable_modes[0]
        best_score = 0
        
        for mode in viable_modes:
            score = self._score_transport_mode(mode, distance)
            if score > best_score:
                best_score = score
                best_mode = mode
        
        return best_mode
    
    def _score_transport_mode(self, mode: TransportMode, distance_km: float) -> float:
        """Score a transport mode for a given distance"""
        mode_data = mode.value
        
        # Base score from mode characteristics
        score = 0.5
        
        # Adjust based on distance efficiency
        optimal_distance = {
            TransportMode.WALKING: 1.5,
            TransportMode.CYCLING: 5.0,
            TransportMode.DRIVING: 10.0,
            TransportMode.PUBLIC_TRANSPORT: 8.0,
            TransportMode.METRO: 15.0,
            TransportMode.FERRY: 10.0
        }
        
        if mode in optimal_distance:
            distance_efficiency = 1.0 - abs(distance_km - optimal_distance[mode]) / optimal_distance[mode]
            score += distance_efficiency * 0.4
        
        # Add scenic factor
        score += mode_data.get('scenic_factor', 0.5) * 0.3
        
        # Add cost efficiency (lower cost = higher score)
        cost_per_km = mode_data.get('cost_per_km', 1.0)
        cost_efficiency = max(0, 1.0 - cost_per_km)
        score += cost_efficiency * 0.3
        
        return score
    
    def _calculate_travel_time(self, distance_km: float, mode: TransportMode) -> int:
        """Calculate travel time in minutes"""
        speed_kmh = mode.value.get('speed_kmh', 4.5)
        base_time_minutes = (distance_km / speed_kmh) * 60
        
        # Add buffer for waiting times, stops, etc.
        if mode in [TransportMode.PUBLIC_TRANSPORT, TransportMode.METRO]:
            base_time_minutes += 10  # Waiting time
        elif mode == TransportMode.FERRY:
            base_time_minutes += 15  # Boarding time
        
        return int(base_time_minutes)
    
    def _calculate_segment_cost(self, distance_km: float, mode: TransportMode) -> float:
        """Calculate cost for a route segment"""
        cost_per_km = mode.value.get('cost_per_km', 0.0)
        return distance_km * cost_per_km
    
    def _calculate_scenic_score(
        self,
        from_loc: GPSLocation,
        to_loc: GPSLocation,
        mode: TransportMode
    ) -> float:
        """Calculate scenic score for a route segment"""
        base_scenic = mode.value.get('scenic_factor', 0.5)
        
        # Boost scenic score for routes near water or historic areas
        avg_lat = (from_loc.latitude + to_loc.latitude) / 2
        avg_lng = (from_loc.longitude + to_loc.longitude) / 2
        
        # Bosphorus area
        if 28.95 <= avg_lng <= 29.05 and 41.0 <= avg_lat <= 41.1:
            base_scenic += 0.2
        
        # Historic peninsula
        if 28.92 <= avg_lng <= 28.99 and 41.0 <= avg_lat <= 41.02:
            base_scenic += 0.15
        
        return min(base_scenic, 1.0)
    
    def _get_user_profile(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'interests': preferences.get('interests', []),
                'preferred_transport': preferences.get('transport_modes', []),
                'mobility_needs': preferences.get('mobility_needs', []),
                'budget_preference': preferences.get('budget', 'medium'),
                'activity_level': preferences.get('activity_level', 'medium'),
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
        else:
            # Update with new preferences
            profile = self.user_profiles[user_id]
            profile.update(preferences)
            profile['last_updated'] = datetime.now()
        
        return self.user_profiles[user_id]
    
    def _calculate_personalization_score(
        self,
        waypoints: List[PersonalizedWaypoint],
        user_profile: Dict[str, Any]
    ) -> float:
        """Calculate overall personalization score for the route"""
        if not waypoints:
            return 0.0
        
        total_score = sum(
            waypoint.interest_match * 0.4 +
            waypoint.popularity_score * 0.2 +
            waypoint.accessibility_score * 0.2 +
            waypoint.weather_suitability * 0.2
            for waypoint in waypoints
        )
        
        return total_score / len(waypoints)
    
    async def _start_real_time_monitoring(self, route: PersonalizedRoute):
        """Start real-time monitoring for route updates"""
        logger.info(f"🔄 Starting real-time monitoring for route {route.route_id}")
        
        self.real_time_monitoring[route.route_id] = {
            'active': True,
            'last_update': datetime.now(),
            'update_frequency': 300  # 5 minutes
        }
        
        # This would be implemented as a background task in production
        # For now, we'll just log the setup
        logger.info(f"✅ Real-time monitoring active for route {route.route_id}")
    
    async def update_route_real_time(
        self,
        route_id: str,
        current_user_location: GPSLocation,
        context_updates: Dict[str, Any] = None
    ) -> Optional[PersonalizedRoute]:
        """Update route based on real-time conditions"""
        
        if route_id not in self.active_routes:
            logger.warning(f"Route {route_id} not found for real-time update")
            return None
        
        route = self.active_routes[route_id]
        
        # Check if significant location change
        distance_from_planned = self._calculate_distance(
            current_user_location,
            route.start_location
        )
        
        if distance_from_planned > 0.5:  # 500m threshold
            logger.info(f"📍 Significant location change detected, re-routing...")
            
            # Create updated route from new location
            updated_route = await self.create_personalized_route(
                route.user_id,
                current_user_location,
                {'interests': [], 'real_time_updates': True},  # Would extract from original route
                {'max_waypoints': len(route.waypoints)}
            )
            
            # Update the stored route
            self.active_routes[route_id] = updated_route
            
            logger.info(f"✅ Route {route_id} updated with new starting point")
            return updated_route
        
        # Update real-time conditions for existing segments
        for segment in route.segments:
            segment.real_time_conditions.update({
                'traffic_level': 'moderate',  # Would integrate with traffic API
                'weather_impact': 'minimal',
                'last_updated': datetime.now().isoformat()
            })
        
        route.last_updated = datetime.now()
        return route
    
    def get_museums_for_districts(self, districts: List[str]) -> List[Dict[str, Any]]:
        """Get museums for specific districts"""
        museums = []
        for museum in self.poi_database.get('museums', []):
            if museum.get('district') in districts:
                museums.append(museum)
        return museums
    
    def get_local_tips_for_districts(self, districts: List[str]) -> Dict[str, List[str]]:
        """Get local tips for specific districts"""
        tips = {}
        for district in districts:
            if district in self.district_tips:
                tips[district] = self.district_tips[district]
        return tips
    
    def enhance_route_with_museums_and_tips(self, route: PersonalizedRoute) -> Dict[str, Any]:
        """Enhance route response with museums and local tips for all districts"""
        # Extract districts from route waypoints
        route_districts = set()
        for waypoint in route.waypoints:
            if hasattr(waypoint.location, 'district') and waypoint.location.district:
                route_districts.add(waypoint.location.district)
            # Also try to determine district from coordinates
            district = self._get_district_from_coordinates(waypoint.location)
            if district:
                route_districts.add(district)
        
        # Get museums for all districts in the route
        district_museums = self.get_museums_for_districts(list(route_districts))
        
        # Get local tips for all districts
        district_tips = self.get_local_tips_for_districts(list(route_districts))
        
        # Create enhanced route response
        enhanced_response = {
            'route_info': {
                'route_id': route.route_id,
                'total_distance_km': route.total_distance_km,
                'total_time_minutes': route.total_time_minutes,
                'total_cost': route.total_cost,
                'personalization_score': route.personalization_score,
                'waypoints_count': len(route.waypoints)
            },
            'districts_covered': list(route_districts),
            'museums_in_route': district_museums,
            'local_tips_by_district': district_tips,
            'route_segments': [
                {
                    'from_location': segment.start_location.address if segment.start_location.address else f"({segment.start_location.latitude:.4f}, {segment.start_location.longitude:.4f})",
                    'to_location': segment.end_location.address if segment.end_location.address else f"({segment.end_location.latitude:.4f}, {segment.end_location.longitude:.4f})",
                    'transport_mode': segment.transport_mode.value,
                    'distance_km': segment.distance_km,
                    'time_minutes': segment.estimated_time_minutes,
                    'cost': segment.cost_estimate
                }
                for segment in route.segments
            ],
            'waypoints': [
                {
                    'name': wp.name,
                    'category': wp.category,
                    'location': f"({wp.location.latitude:.4f}, {wp.location.longitude:.4f})",
                    'district': wp.location.district,
                    'personalization_score': wp.personalization_score,
                    'visit_duration_minutes': wp.recommended_visit_duration
                }
                for wp in route.waypoints
            ]
        }
        
        return enhanced_response
    
    def _get_district_from_coordinates(self, location: GPSLocation) -> str:
        """Determine district from GPS coordinates"""
        min_distance = float('inf')
        closest_district = None
        
        for district_name, district_info in self.istanbul_districts.items():
            center = district_info['center']
            distance = self._calculate_distance(location, center)
            radius = district_info.get('radius_km', 2.0)
            
            if distance <= radius and distance < min_distance:
                min_distance = distance
                closest_district = district_name
        
        return closest_district or 'unknown'
    
    def _serialize_route(self, route: PersonalizedRoute) -> Dict[str, Any]:
        """Serialize route for caching"""
        return {
            'route_data': asdict(route),
            'serialization_version': '1.0'
        }
    
    def _deserialize_route(self, cached_data: Dict[str, Any]) -> PersonalizedRoute:
        """Deserialize route from cache"""
        route_data = cached_data['route_data']
        
        # Convert datetime strings back to datetime objects
        route_data['created_at'] = datetime.fromisoformat(route_data['created_at'])
        route_data['last_updated'] = datetime.fromisoformat(route_data['last_updated'])
        
        # Reconstruct complex objects
        route_data['start_location'] = GPSLocation(**route_data['start_location'])
        
        return PersonalizedRoute(**route_data)
    
    def get_route_suggestions(
        self,
        current_location: GPSLocation,
        user_id: str = None
    ) -> List[str]:
        """Generate contextual route suggestions based on current location"""
        
        suggestions = []
        district = self._get_district_from_location(current_location)
        
        # Location-specific suggestions
        if district == 'sultanahmet':
            suggestions.extend([
                "🏛️ You're in the historic heart! Perfect for a walking tour of Ottoman sites",
                "📸 Hagia Sophia and Blue Mosque are just steps away",
                "🍽️ Try traditional Ottoman cuisine at nearby historic restaurants"
            ])
        elif district == 'beyoglu':
            suggestions.extend([
                "🎨 Explore the artistic side of Istanbul with galleries and modern art",
                "🌉 Walk down to Galata Bridge for stunning Bosphorus views",
                "🛍️ Istiklal Street offers great shopping and street food"
            ])
        elif district == 'kadikoy':
            suggestions.extend([
                "🌊 Take a ferry ride for the best Bosphorus experience",
                "🍻 Explore the vibrant nightlife and local bars",
                "🏪 Visit the local markets for authentic Istanbul shopping"
            ])
        
        # Time-based suggestions
        hour = datetime.now().hour
        if 9 <= hour <= 17:
            suggestions.append("☀️ Perfect time for outdoor sightseeing and museum visits")
        elif 17 <= hour <= 21:
            suggestions.append("🌅 Golden hour - ideal for photography and sunset views")
        elif hour >= 21:
            suggestions.append("🌙 Evening is perfect for dining and experiencing Istanbul's nightlife")
        
        return suggestions[:4]  # Limit to top 4 suggestions
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, TransportMode):
            return data.name
        elif isinstance(data, GPSLocation):
            return {
                'latitude': data.latitude,
                'longitude': data.longitude,
                'accuracy': data.accuracy,
                'address': data.address,
                'district': data.district
            }
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
    
    def _restore_from_serializable(self, data: Any) -> Any:
        """Restore data from JSON-serializable format"""
        if isinstance(data, dict):
            # Check if it's a GPSLocation
            if all(key in data for key in ['latitude', 'longitude']):
                return GPSLocation(
                    latitude=data['latitude'],
                    longitude=data['longitude'],
                    accuracy=data.get('accuracy', 10.0),
                    address=data.get('address', ''),
                    district=data.get('district', '')
                )
            else:
                return {key: self._restore_from_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._restore_from_serializable(item) for item in data]
        elif isinstance(data, str) and data in [mode.name for mode in TransportMode]:
            return TransportMode[data]
        else:
            return data
    
    def _convert_location_to_gps(self, location_name: str, metadata: Dict[str, Any]) -> Optional[GPSLocation]:
        """Convert detected location name to GPS coordinates"""
        try:
            # Check if metadata contains coordinates
            if 'coordinates' in metadata:
                coords = metadata['coordinates']
                return GPSLocation(
                    latitude=coords[0],
                    longitude=coords[1],
                    address=location_name,
                    district=metadata.get('district', '')
                )
            
            # Check district centers
            location_lower = location_name.lower()
            for district_name, district_data in self.istanbul_districts.items():
                if district_name in location_lower or location_lower in district_name:
                    center = district_data['center']
                    return GPSLocation(
                        latitude=center.latitude,
                        longitude=center.longitude,
                        address=location_name,
                        district=district_name
                    )
            
            # Check POI database
            if hasattr(self, 'poi_database'):
                for poi in self.poi_database.get('attractions', []):
                    if poi['name'].lower() == location_lower:
                        return GPSLocation(
                            latitude=poi['location']['lat'],
                            longitude=poi['location']['lng'],
                            address=poi['name'],
                            district=poi.get('district', '')
                        )
            
            logger.warning(f"Could not find GPS coordinates for location: {location_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error converting location to GPS: {e}")
            return None
    
    async def create_route_with_fallback_location(
        self,
        user_id: str,
        user_input: Optional[str] = None,
        user_context: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        session_data: Optional[Dict] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a personalized route using fallback location detection
        when GPS is not available or permission is denied
        
        Args:
            user_id: Unique user identifier
            user_input: User's location description
            user_context: Previous conversation context
            ip_address: User's IP address for geolocation
            session_data: Previous session data
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response with museums and local tips
        """
        if not self.fallback_detector:
            return {
                'error': 'Fallback location detection not available',
                'fallback_required': True,
                'location_prompts': self._get_basic_location_prompts()
            }
        
        # Detect user location using fallback methods
        location_options = await self.fallback_detector.detect_fallback_location(
            user_input, user_context, ip_address, session_data
        )
        
        if not location_options:
            return {
                'no_location_detected': True,
                'location_prompts': self._get_user_location_prompts(),
                'suggested_districts': list(self.istanbul_districts.keys()),
                'popular_landmarks': list(self.fallback_detector.landmark_locations.keys())
            }
        
        # Use the highest confidence location option
        best_option = location_options[0]
        
        if best_option.requires_user_confirmation:
            return {
                'location_confirmation_required': True,
                'detected_location': best_option.user_friendly_name,
                'confidence': best_option.confidence,
                'method': best_option.method.value,
                'alternative_options': [
                    {
                        'name': opt.user_friendly_name,
                        'confidence': opt.confidence,
                        'method': opt.method.value
                    } for opt in location_options[:3]
                ],
                'confirm_prompt': f"I detected your location as {best_option.user_friendly_name}. Is this correct?"
            }
        
        # Create route using detected location
        route = await self.create_personalized_route(
            user_id, 
            best_option.location, 
            preferences or {}
        )
        
        # Enhance response with museums and local tips
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        # Add fallback location information
        enhanced_response['location_detection'] = {
            'method': best_option.method.value,
            'confidence': best_option.confidence,
            'detected_location': best_option.user_friendly_name,
            'district': best_option.district,
            'neighborhood': best_option.neighborhood,
            'alternative_options_available': len(location_options) > 1
        }
        
        # Add recommendation summary
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'medium',  # Fallback locations get medium personalization
            'route_optimization': f'Location-optimized using {best_option.method.value}',
            'location_accuracy': 'high' if best_option.confidence > 0.8 else 'medium' if best_option.confidence > 0.6 else 'estimated'
        }
        
        return enhanced_response
    
    async def confirm_location_and_create_route(
        self,
        user_id: str,
        confirmed_location: GPSLocation,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create route after user confirms the detected location
        
        Args:
            user_id: Unique user identifier
            confirmed_location: User-confirmed GPS location
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response with museums and local tips
        """
        # Create route using confirmed location
        route = await self.create_personalized_route(
            user_id, 
            confirmed_location, 
            preferences or {}
        )
        
        # Enhance response with museums and local tips
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        # Add confirmation info
        enhanced_response['location_detection'] = {
            'method': 'user_confirmed',
            'confidence': 1.0,
            'detected_location': f"Confirmed location in {confirmed_location.district}",
            'district': confirmed_location.district
        }
        
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'high',  # Confirmed locations get high personalization
            'route_optimization': 'User-confirmed location with GPS optimization'
        }
        
        return enhanced_response
    
    def _get_user_location_prompts(self) -> List[Dict[str, Any]]:
        """Get user-friendly location input prompts"""
        if not self.fallback_detector:
            return self._get_basic_location_prompts()
        
        prompts = self.fallback_detector.generate_location_prompts()
        return [
            {
                'message': prompt.message,
                'input_type': prompt.input_type.value,
                'examples': prompt.examples,
                'suggestions': prompt.suggestions
            } for prompt in prompts
        ]
    
    def _get_basic_location_prompts(self) -> List[Dict[str, Any]]:
        """Get basic location prompts when fallback detector is not available"""
        return [
            {
                'message': "Could you tell me which district you're in?",
                'input_type': 'district_name',
                'examples': ["Sultanahmet", "Beyoğlu", "Kadıköy", "Beşiktaş"],
                'suggestions': list(self.istanbul_districts.keys())
            },
            {
                'message': "What landmark are you closest to?",
                'input_type': 'landmark',
                'examples': ["Galata Tower", "Topkapi Palace", "Taksim Square"],
                'suggestions': ["Sultanahmet", "Galata Tower", "Taksim", "Dolmabahce Palace"]
            }
        ]
    
    async def create_route_from_manual_input(
        self,
        user_id: str,
        location_description: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create route from manual location input when all other methods fail
        
        Args:
            user_id: Unique user identifier  
            location_description: User's manual location description
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response or location clarification request
        """
        if not self.fallback_detector:
            # Basic parsing without fallback detector
            location = self._parse_basic_location(location_description)
        else:
            # Use fallback detector for sophisticated parsing
            location_options = await self.fallback_detector.detect_fallback_location(
                user_input=location_description
            )
            
            if not location_options:
                return {
                    'parsing_failed': True,
                    'clarification_needed': True,
                    'message': "I couldn't identify your location. Could you be more specific?",
                    'suggestions': [
                        "Try mentioning a district name (e.g., 'Sultanahmet', 'Beyoğlu')",
                        "Name a landmark (e.g., 'near Galata Tower', 'close to Taksim Square')",
                        "Describe your area (e.g., 'in the old city', 'near the Bosphorus')"
                    ]
                }
            
            location = location_options[0].location
        
        if not location:
            return {
                'parsing_failed': True,
                'message': "Could you provide more details about your location?",
                'location_prompts': self._get_user_location_prompts()
            }
        
        # Create route using parsed location
        route = await self.create_personalized_route(user_id, location, preferences or {})
        
        # Enhance response
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        enhanced_response['location_detection'] = {
            'method': 'manual_input',
            'confidence': 0.7,  # Manual input gets medium confidence
            'detected_location': location_description,
            'district': location.district
        }
        
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'medium',
            'route_optimization': 'Manual location with intelligent routing'
        }
        
        return enhanced_response
    
    def _parse_basic_location(self, description: str) -> Optional[GPSLocation]:
        """Basic location parsing when fallback detector is not available"""
        desc_lower = description.lower().strip()
        
        # Check for district names
        for district, info in self.istanbul_districts.items():
            if district in desc_lower:
                return info['center']
        
        # Check for common landmarks
        landmarks = {
            'sultanahmet': GPSLocation(41.0082, 28.9784, district='sultanahmet'),
            'taksim': GPSLocation(41.0369, 28.9857, district='beyoglu'),
            'galata': GPSLocation(41.0256, 28.9744, district='beyoglu'),
            'kadikoy': GPSLocation(40.9907, 29.0205, district='kadikoy'),
            'besiktas': GPSLocation(41.0422, 29.0084, district='besiktas')
        }
        
        for landmark, location in landmarks.items():
            if landmark in desc_lower:
                return location
        
        return None
    
    async def _optimize_route_segments(
        self,
        start_location: GPSLocation,
        waypoints: List[PersonalizedWaypoint],
        preferred_transport_modes: List[TransportMode]
    ) -> List[RouteSegment]:
        """Optimize route segments between waypoints"""
        
        segments = []
        current_location = start_location
        
        for waypoint in waypoints:
            # Find best transport mode for this segment
            best_mode = self._select_best_transport_mode(
                current_location,
                waypoint.location,
                preferred_transport_modes,
                waypoint.transport_modes
            )
            
            # Calculate segment details
            distance = self._calculate_distance(current_location, waypoint.location)
            travel_time = self._calculate_travel_time(distance, best_mode)
            cost = self._calculate_segment_cost(distance, best_mode)
            scenic_score = self._calculate_scenic_score(current_location, waypoint.location, best_mode)
            accessibility_score = waypoint.accessibility_score
            
            segment = RouteSegment(
                from_point=current_location,
                to_point=waypoint.location,
                transport_mode=best_mode,
                distance_km=distance,
                estimated_time_minutes=travel_time,
                cost_estimate=cost,
                scenic_score=scenic_score,
                accessibility_score=accessibility_score,
                real_time_conditions={}
            )
            
            segments.append(segment)
            current_location = waypoint.location
        
        return segments
    
    def _select_best_transport_mode(
        self,
        from_loc: GPSLocation,
        to_loc: GPSLocation,
        preferred_modes: List[TransportMode],
        available_modes: List[TransportMode]
    ) -> TransportMode:
        """Select the best transport mode for a segment"""
        
        # Find intersection of preferred and available modes
        viable_modes = [mode for mode in preferred_modes if mode in available_modes]
        
        if not viable_modes:
            # Fall back to available modes
            viable_modes = available_modes
        
        if not viable_modes:
            # Ultimate fallback
            return TransportMode.WALKING
        
        # Score each viable mode
        distance = self._calculate_distance(from_loc, to_loc)
        best_mode = viable_modes[0]
        best_score = 0
        
        for mode in viable_modes:
            score = self._score_transport_mode(mode, distance)
            if score > best_score:
                best_score = score
                best_mode = mode
        
        return best_mode
    
    def _score_transport_mode(self, mode: TransportMode, distance_km: float) -> float:
        """Score a transport mode for a given distance"""
        mode_data = mode.value
        
        # Base score from mode characteristics
        score = 0.5
        
        # Adjust based on distance efficiency
        optimal_distance = {
            TransportMode.WALKING: 1.5,
            TransportMode.CYCLING: 5.0,
            TransportMode.DRIVING: 10.0,
            TransportMode.PUBLIC_TRANSPORT: 8.0,
            TransportMode.METRO: 15.0,
            TransportMode.FERRY: 10.0
        }
        
        if mode in optimal_distance:
            distance_efficiency = 1.0 - abs(distance_km - optimal_distance[mode]) / optimal_distance[mode]
            score += distance_efficiency * 0.4
        
        # Add scenic factor
        score += mode_data.get('scenic_factor', 0.5) * 0.3
        
        # Add cost efficiency (lower cost = higher score)
        cost_per_km = mode_data.get('cost_per_km', 1.0)
        cost_efficiency = max(0, 1.0 - cost_per_km)
        score += cost_efficiency * 0.3
        
        return score
    
    def _calculate_travel_time(self, distance_km: float, mode: TransportMode) -> int:
        """Calculate travel time in minutes"""
        speed_kmh = mode.value.get('speed_kmh', 4.5)
        base_time_minutes = (distance_km / speed_kmh) * 60
        
        # Add buffer for waiting times, stops, etc.
        if mode in [TransportMode.PUBLIC_TRANSPORT, TransportMode.METRO]:
            base_time_minutes += 10  # Waiting time
        elif mode == TransportMode.FERRY:
            base_time_minutes += 15  # Boarding time
        
        return int(base_time_minutes)
    
    def _calculate_segment_cost(self, distance_km: float, mode: TransportMode) -> float:
        """Calculate cost for a route segment"""
        cost_per_km = mode.value.get('cost_per_km', 0.0)
        return distance_km * cost_per_km
    
    def _calculate_scenic_score(
        self,
        from_loc: GPSLocation,
        to_loc: GPSLocation,
        mode: TransportMode
    ) -> float:
        """Calculate scenic score for a route segment"""
        base_scenic = mode.value.get('scenic_factor', 0.5)
        
        # Boost scenic score for routes near water or historic areas
        avg_lat = (from_loc.latitude + to_loc.latitude) / 2
        avg_lng = (from_loc.longitude + to_loc.longitude) / 2
        
        # Bosphorus area
        if 28.95 <= avg_lng <= 29.05 and 41.0 <= avg_lat <= 41.1:
            base_scenic += 0.2
        
        # Historic peninsula
        if 28.92 <= avg_lng <= 28.99 and 41.0 <= avg_lat <= 41.02:
            base_scenic += 0.15
        
        return min(base_scenic, 1.0)
    
    def _get_user_profile(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'interests': preferences.get('interests', []),
                'preferred_transport': preferences.get('transport_modes', []),
                'mobility_needs': preferences.get('mobility_needs', []),
                'budget_preference': preferences.get('budget', 'medium'),
                'activity_level': preferences.get('activity_level', 'medium'),
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
        else:
            # Update with new preferences
            profile = self.user_profiles[user_id]
            profile.update(preferences)
            profile['last_updated'] = datetime.now()
        
        return self.user_profiles[user_id]
    
    def _calculate_personalization_score(
        self,
        waypoints: List[PersonalizedWaypoint],
        user_profile: Dict[str, Any]
    ) -> float:
        """Calculate overall personalization score for the route"""
        if not waypoints:
            return 0.0
        
        total_score = sum(
            waypoint.interest_match * 0.4 +
            waypoint.popularity_score * 0.2 +
            waypoint.accessibility_score * 0.2 +
            waypoint.weather_suitability * 0.2
            for waypoint in waypoints
        )
        
        return total_score / len(waypoints)
    
    async def _start_real_time_monitoring(self, route: PersonalizedRoute):
        """Start real-time monitoring for route updates"""
        logger.info(f"🔄 Starting real-time monitoring for route {route.route_id}")
        
        self.real_time_monitoring[route.route_id] = {
            'active': True,
            'last_update': datetime.now(),
            'update_frequency': 300  # 5 minutes
        }
        
        # This would be implemented as a background task in production
        # For now, we'll just log the setup
        logger.info(f"✅ Real-time monitoring active for route {route.route_id}")
    
    async def update_route_real_time(
        self,
        route_id: str,
        current_user_location: GPSLocation,
        context_updates: Dict[str, Any] = None
    ) -> Optional[PersonalizedRoute]:
        """Update route based on real-time conditions"""
        
        if route_id not in self.active_routes:
            logger.warning(f"Route {route_id} not found for real-time update")
            return None
        
        route = self.active_routes[route_id]
        
        # Check if significant location change
        distance_from_planned = self._calculate_distance(
            current_user_location,
            route.start_location
        )
        
        if distance_from_planned > 0.5:  # 500m threshold
            logger.info(f"📍 Significant location change detected, re-routing...")
            
            # Create updated route from new location
            updated_route = await self.create_personalized_route(
                route.user_id,
                current_user_location,
                {'interests': [], 'real_time_updates': True},  # Would extract from original route
                {'max_waypoints': len(route.waypoints)}
            )
            
            # Update the stored route
            self.active_routes[route_id] = updated_route
            
            logger.info(f"✅ Route {route_id} updated with new starting point")
            return updated_route
        
        # Update real-time conditions for existing segments
        for segment in route.segments:
            segment.real_time_conditions.update({
                'traffic_level': 'moderate',  # Would integrate with traffic API
                'weather_impact': 'minimal',
                'last_updated': datetime.now().isoformat()
            })
        
        route.last_updated = datetime.now()
        return route
    
    def get_museums_for_districts(self, districts: List[str]) -> List[Dict[str, Any]]:
        """Get museums for specific districts"""
        museums = []
        for museum in self.poi_database.get('museums', []):
            if museum.get('district') in districts:
                museums.append(museum)
        return museums
    
    def get_local_tips_for_districts(self, districts: List[str]) -> Dict[str, List[str]]:
        """Get local tips for specific districts"""
        tips = {}
        for district in districts:
            if district in self.district_tips:
                tips[district] = self.district_tips[district]
        return tips
    
    def enhance_route_with_museums_and_tips(self, route: PersonalizedRoute) -> Dict[str, Any]:
        """Enhance route response with museums and local tips for all districts"""
        # Extract districts from route waypoints
        route_districts = set()
        for waypoint in route.waypoints:
            if hasattr(waypoint.location, 'district') and waypoint.location.district:
                route_districts.add(waypoint.location.district)
            # Also try to determine district from coordinates
            district = self._get_district_from_coordinates(waypoint.location)
            if district:
                route_districts.add(district)
        
        # Get museums for all districts in the route
        district_museums = self.get_museums_for_districts(list(route_districts))
        
        # Get local tips for all districts
        district_tips = self.get_local_tips_for_districts(list(route_districts))
        
        # Create enhanced route response
        enhanced_response = {
            'route_info': {
                'route_id': route.route_id,
                'total_distance_km': route.total_distance_km,
                'total_time_minutes': route.total_time_minutes,
                'total_cost': route.total_cost,
                'personalization_score': route.personalization_score,
                'waypoints_count': len(route.waypoints)
            },
            'districts_covered': list(route_districts),
            'museums_in_route': district_museums,
            'local_tips_by_district': district_tips,
            'route_segments': [
                {
                    'from_location': segment.start_location.address if segment.start_location.address else f"({segment.start_location.latitude:.4f}, {segment.start_location.longitude:.4f})",
                    'to_location': segment.end_location.address if segment.end_location.address else f"({segment.end_location.latitude:.4f}, {segment.end_location.longitude:.4f})",
                    'transport_mode': segment.transport_mode.value,
                    'distance_km': segment.distance_km,
                    'time_minutes': segment.estimated_time_minutes,
                    'cost': segment.cost_estimate
                }
                for segment in route.segments
            ],
            'waypoints': [
                {
                    'name': wp.name,
                    'category': wp.category,
                    'location': f"({wp.location.latitude:.4f}, {wp.location.longitude:.4f})",
                    'district': wp.location.district,
                    'personalization_score': wp.personalization_score,
                    'visit_duration_minutes': wp.recommended_visit_duration
                }
                for wp in route.waypoints
            ]
        }
        
        return enhanced_response
    
    def _get_district_from_coordinates(self, location: GPSLocation) -> str:
        """Determine district from GPS coordinates"""
        min_distance = float('inf')
        closest_district = None
        
        for district_name, district_info in self.istanbul_districts.items():
            center = district_info['center']
            distance = self._calculate_distance(location, center)
            radius = district_info.get('radius_km', 2.0)
            
            if distance <= radius and distance < min_distance:
                min_distance = distance
                closest_district = district_name
        
        return closest_district or 'unknown'
    
    def _serialize_route(self, route: PersonalizedRoute) -> Dict[str, Any]:
        """Serialize route for caching"""
        return {
            'route_data': asdict(route),
            'serialization_version': '1.0'
        }
    
    def _deserialize_route(self, cached_data: Dict[str, Any]) -> PersonalizedRoute:
        """Deserialize route from cache"""
        route_data = cached_data['route_data']
        
        # Convert datetime strings back to datetime objects
        route_data['created_at'] = datetime.fromisoformat(route_data['created_at'])
        route_data['last_updated'] = datetime.fromisoformat(route_data['last_updated'])
        
        # Reconstruct complex objects
        route_data['start_location'] = GPSLocation(**route_data['start_location'])
        
        return PersonalizedRoute(**route_data)
    
    def get_route_suggestions(
        self,
        current_location: GPSLocation,
        user_id: str = None
    ) -> List[str]:
        """Generate contextual route suggestions based on current location"""
        
        suggestions = []
        district = self._get_district_from_location(current_location)
        
        # Location-specific suggestions
        if district == 'sultanahmet':
            suggestions.extend([
                "🏛️ You're in the historic heart! Perfect for a walking tour of Ottoman sites",
                "📸 Hagia Sophia and Blue Mosque are just steps away",
                "🍽️ Try traditional Ottoman cuisine at nearby historic restaurants"
            ])
        elif district == 'beyoglu':
            suggestions.extend([
                "🎨 Explore the artistic side of Istanbul with galleries and modern art",
                "🌉 Walk down to Galata Bridge for stunning Bosphorus views",
                "🛍️ Istiklal Street offers great shopping and street food"
            ])
        elif district == 'kadikoy':
            suggestions.extend([
                "🌊 Take a ferry ride for the best Bosphorus experience",
                "🍻 Explore the vibrant nightlife and local bars",
                "🏪 Visit the local markets for authentic Istanbul shopping"
            ])
        
        # Time-based suggestions
        hour = datetime.now().hour
        if 9 <= hour <= 17:
            suggestions.append("☀️ Perfect time for outdoor sightseeing and museum visits")
        elif 17 <= hour <= 21:
            suggestions.append("🌅 Golden hour - ideal for photography and sunset views")
        elif hour >= 21:
            suggestions.append("🌙 Evening is perfect for dining and experiencing Istanbul's nightlife")
        
        return suggestions[:4]  # Limit to top 4 suggestions
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, TransportMode):
            return data.name
        elif isinstance(data, GPSLocation):
            return {
                'latitude': data.latitude,
                'longitude': data.longitude,
                'accuracy': data.accuracy,
                'address': data.address,
                'district': data.district
            }
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
    
    def _restore_from_serializable(self, data: Any) -> Any:
        """Restore data from JSON-serializable format"""
        if isinstance(data, dict):
            # Check if it's a GPSLocation
            if all(key in data for key in ['latitude', 'longitude']):
                return GPSLocation(
                    latitude=data['latitude'],
                    longitude=data['longitude'],
                    accuracy=data.get('accuracy', 10.0),
                    address=data.get('address', ''),
                    district=data.get('district', '')
                )
            else:
                return {key: self._restore_from_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._restore_from_serializable(item) for item in data]
        elif isinstance(data, str) and data in [mode.name for mode in TransportMode]:
            return TransportMode[data]
        else:
            return data
    
    def _convert_location_to_gps(self, location_name: str, metadata: Dict[str, Any]) -> Optional[GPSLocation]:
        """Convert detected location name to GPS coordinates"""
        try:
            # Check if metadata contains coordinates
            if 'coordinates' in metadata:
                coords = metadata['coordinates']
                return GPSLocation(
                    latitude=coords[0],
                    longitude=coords[1],
                    address=location_name,
                    district=metadata.get('district', '')
                )
            
            # Check district centers
            location_lower = location_name.lower()
            for district_name, district_data in self.istanbul_districts.items():
                if district_name in location_lower or location_lower in district_name:
                    center = district_data['center']
                    return GPSLocation(
                        latitude=center.latitude,
                        longitude=center.longitude,
                        address=location_name,
                        district=district_name
                    )
            
            # Check POI database
            if hasattr(self, 'poi_database'):
                for poi in self.poi_database.get('attractions', []):
                    if poi['name'].lower() == location_lower:
                        return GPSLocation(
                            latitude=poi['location']['lat'],
                            longitude=poi['location']['lng'],
                            address=poi['name'],
                            district=poi.get('district', '')
                        )
            
            logger.warning(f"Could not find GPS coordinates for location: {location_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error converting location to GPS: {e}")
            return None
    
    async def create_route_with_fallback_location(
        self,
        user_id: str,
        user_input: Optional[str] = None,
        user_context: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        session_data: Optional[Dict] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a personalized route using fallback location detection
        when GPS is not available or permission is denied
        
        Args:
            user_id: Unique user identifier
            user_input: User's location description
            user_context: Previous conversation context
            ip_address: User's IP address for geolocation
            session_data: Previous session data
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response with museums and local tips
        """
        if not self.fallback_detector:
            return {
                'error': 'Fallback location detection not available',
                'fallback_required': True,
                'location_prompts': self._get_basic_location_prompts()
            }
        
        # Detect user location using fallback methods
        location_options = await self.fallback_detector.detect_fallback_location(
            user_input, user_context, ip_address, session_data
        )
        
        if not location_options:
            return {
                'no_location_detected': True,
                'location_prompts': self._get_user_location_prompts(),
                'suggested_districts': list(self.istanbul_districts.keys()),
                'popular_landmarks': list(self.fallback_detector.landmark_locations.keys())
            }
        
        # Use the highest confidence location option
        best_option = location_options[0]
        
        if best_option.requires_user_confirmation:
            return {
                'location_confirmation_required': True,
                'detected_location': best_option.user_friendly_name,
                'confidence': best_option.confidence,
                'method': best_option.method.value,
                'alternative_options': [
                    {
                        'name': opt.user_friendly_name,
                        'confidence': opt.confidence,
                        'method': opt.method.value
                    } for opt in location_options[:3]
                ],
                'confirm_prompt': f"I detected your location as {best_option.user_friendly_name}. Is this correct?"
            }
        
        # Create route using detected location
        route = await self.create_personalized_route(
            user_id, 
            best_option.location, 
            preferences or {}
        )
        
        # Enhance response with museums and local tips
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        # Add fallback location information
        enhanced_response['location_detection'] = {
            'method': best_option.method.value,
            'confidence': best_option.confidence,
            'detected_location': best_option.user_friendly_name,
            'district': best_option.district,
            'neighborhood': best_option.neighborhood,
            'alternative_options_available': len(location_options) > 1
        }
        
        # Add recommendation summary
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'medium',  # Fallback locations get medium personalization
            'route_optimization': f'Location-optimized using {best_option.method.value}',
            'location_accuracy': 'high' if best_option.confidence > 0.8 else 'medium' if best_option.confidence > 0.6 else 'estimated'
        }
        
        return enhanced_response
    
    async def confirm_location_and_create_route(
        self,
        user_id: str,
        confirmed_location: GPSLocation,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create route after user confirms the detected location
        
        Args:
            user_id: Unique user identifier
            confirmed_location: User-confirmed GPS location
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response with museums and local tips
        """
        # Create route using confirmed location
        route = await self.create_personalized_route(
            user_id, 
            confirmed_location, 
            preferences or {}
        )
        
        # Enhance response with museums and local tips
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        # Add confirmation info
        enhanced_response['location_detection'] = {
            'method': 'user_confirmed',
            'confidence': 1.0,
            'detected_location': f"Confirmed location in {confirmed_location.district}",
            'district': confirmed_location.district
        }
        
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'high',  # Confirmed locations get high personalization
            'route_optimization': 'User-confirmed location with GPS optimization'
        }
        
        return enhanced_response
    
    def _get_user_location_prompts(self) -> List[Dict[str, Any]]:
        """Get user-friendly location input prompts"""
        if not self.fallback_detector:
            return self._get_basic_location_prompts()
        
        prompts = self.fallback_detector.generate_location_prompts()
        return [
            {
                'message': prompt.message,
                'input_type': prompt.input_type.value,
                'examples': prompt.examples,
                'suggestions': prompt.suggestions
            } for prompt in prompts
        ]
    
    def _get_basic_location_prompts(self) -> List[Dict[str, Any]]:
        """Get basic location prompts when fallback detector is not available"""
        return [
            {
                'message': "Could you tell me which district you're in?",
                'input_type': 'district_name',
                'examples': ["Sultanahmet", "Beyoğlu", "Kadıköy", "Beşiktaş"],
                'suggestions': list(self.istanbul_districts.keys())
            },
            {
                'message': "What landmark are you closest to?",
                'input_type': 'landmark',
                'examples': ["Galata Tower", "Topkapi Palace", "Taksim Square"],
                'suggestions': ["Sultanahmet", "Galata Tower", "Taksim", "Dolmabahce Palace"]
            }
        ]
    
    async def create_route_from_manual_input(
        self,
        user_id: str,
        location_description: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create route from manual location input when all other methods fail
        
        Args:
            user_id: Unique user identifier  
            location_description: User's manual location description
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response or location clarification request
        """
        if not self.fallback_detector:
            # Basic parsing without fallback detector
            location = self._parse_basic_location(location_description)
        else:
            # Use fallback detector for sophisticated parsing
            location_options = await self.fallback_detector.detect_fallback_location(
                user_input=location_description
            )
            
            if not location_options:
                return {
                    'parsing_failed': True,
                    'clarification_needed': True,
                    'message': "I couldn't identify your location. Could you be more specific?",
                    'suggestions': [
                        "Try mentioning a district name (e.g., 'Sultanahmet', 'Beyoğlu')",
                        "Name a landmark (e.g., 'near Galata Tower', 'close to Taksim Square')",
                        "Describe your area (e.g., 'in the old city', 'near the Bosphorus')"
                    ]
                }
            
            location = location_options[0].location
        
        if not location:
            return {
                'parsing_failed': True,
                'message': "Could you provide more details about your location?",
                'location_prompts': self._get_user_location_prompts()
            }
        
        # Create route using parsed location
        route = await self.create_personalized_route(user_id, location, preferences or {})
        
        # Enhance response
        enhanced_response = self.enhance_route_with_museums_and_tips(route)
        
        enhanced_response['location_detection'] = {
            'method': 'manual_input',
            'confidence': 0.7,  # Manual input gets medium confidence
            'detected_location': location_description,
            'district': location.district
        }
        
        enhanced_response['recommendation_summary'] = {
            'total_museums_available': len(enhanced_response['museums_in_route']),
            'districts_with_tips': len(enhanced_response['local_tips_by_district']),
            'personalization_level': 'medium',
            'route_optimization': 'Manual location with intelligent routing'
        }
        
        return enhanced_response
    
    def _parse_basic_location(self, description: str) -> Optional[GPSLocation]:
        """Basic location parsing when fallback detector is not available"""
        desc_lower = description.lower().strip()
        
        # Check for district names
        for district, info in self.istanbul_districts.items():
            if district in desc_lower:
                return info['center']
        
        # Check for common landmarks
        landmarks = {
            'sultanahmet': GPSLocation(41.0082, 28.9784, district='sultanahmet'),
            'taksim': GPSLocation(41.0369, 28.9857, district='beyoglu'),
            'galata': GPSLocation(41.0256, 28.9744, district='beyoglu'),
            'kadikoy': GPSLocation(40.9907, 29.0205, district='kadikoy'),
            'besiktas': GPSLocation(41.0422, 29.0084, district='besiktas')
        }
        
        for landmark, location in landmarks.items():
            if landmark in desc_lower:
                return location
        
        return None
    
    async def _optimize_route_segments(
        self,
        start_location: GPSLocation,
        waypoints: List[PersonalizedWaypoint],
        preferred_transport_modes: List[TransportMode]
    ) -> List[RouteSegment]:
        """Optimize route segments between waypoints"""
        
        segments = []
        current_location = start_location
        
        for waypoint in waypoints:
            # Find best transport mode for this segment
            best_mode = self._select_best_transport_mode(
                current_location,
                waypoint.location,
                preferred_transport_modes,
                waypoint.transport_modes
            )
            
            # Calculate segment details
            distance = self._calculate_distance(current_location, waypoint.location)
            travel_time = self._calculate_travel_time(distance, best_mode)
            cost = self._calculate_segment_cost(distance, best_mode)
            scenic_score = self._calculate_scenic_score(current_location, waypoint.location, best_mode)
            accessibility_score = waypoint.accessibility_score
            
            segment = RouteSegment(
                from_point=current_location,
                to_point=waypoint.location,
                transport_mode=best_mode,
                distance_km=distance,
                estimated_time_minutes=travel_time,
                cost_estimate=cost,
                scenic_score=scenic_score,
                accessibility_score=accessibility_score,
                real_time_conditions={}
            )
            
            segments.append(segment)
            current_location = waypoint.location
        
        return segments
    
    def _select_best_transport_mode(
        self,
        from_loc: GPSLocation,
        to_loc: GPSLocation,
        preferred_modes: List[TransportMode],
        available_modes: List[TransportMode]
    ) -> TransportMode:
        """Select the best transport mode for a segment"""
        
        # Find intersection of preferred and available modes
        viable_modes = [mode for mode in preferred_modes if mode in available_modes]
        
        if not viable_modes:
            # Fall back to available modes
            viable_modes = available_modes
        
        if not viable_modes:
            # Ultimate fallback
            return TransportMode.WALKING
        
        # Score each viable mode
        distance = self._calculate_distance(from_loc, to_loc)
        best_mode = viable_modes[0]
        best_score = 0
        
        for mode in viable_modes:
            score = self._score_transport_mode(mode, distance)
            if score > best_score:
                best_score = score
                best_mode = mode
        
        return best_mode
    
    def _score_transport_mode(self, mode: TransportMode, distance_km: float) -> float:
        """Score a transport mode for a given distance"""
        mode_data = mode.value
        
        # Base score from mode characteristics
        score = 0.5
        
        # Adjust based on distance efficiency
        optimal_distance = {
            TransportMode.WALKING: 1.5,
            TransportMode.CYCLING: 5.0,
            TransportMode.DRIVING: 10.0,
            TransportMode.PUBLIC_TRANSPORT: 8.0,
            TransportMode.METRO: 15.0,
            TransportMode.FERRY: 10.0
        }
        
        if mode in optimal_distance:
            distance_efficiency = 1.0 - abs(distance_km - optimal_distance[mode]) / optimal_distance[mode]
            score += distance_efficiency * 0.4
        
        # Add scenic factor
        score += mode_data.get('scenic_factor', 0.5) * 0.3
        
        # Add cost efficiency (lower cost = higher score)
        cost_per_km = mode_data.get('cost_per_km', 1.0)
        cost_efficiency = max(0, 1.0 - cost_per_km)
        score += cost_efficiency * 0.3
        
        return score
    
    def _calculate_travel_time(self, distance_km: float, mode: TransportMode) -> int:
        """Calculate travel time in minutes"""
        speed_kmh = mode.value.get('speed_kmh', 4.5)
        base_time_minutes = (distance_km / speed_kmh) * 60
        
        # Add buffer for waiting times, stops, etc.
        if mode in [TransportMode.PUBLIC_TRANSPORT, TransportMode.METRO]:
            base_time_minutes += 10  # Waiting time
        elif mode == TransportMode.FERRY:
            base_time_minutes += 15  # Boarding time
        
        return int(base_time_minutes)
    
    def _calculate_segment_cost(self, distance_km: float, mode: TransportMode) -> float:
        """Calculate cost for a route segment"""
        cost_per_km = mode.value.get('cost_per_km', 0.0)
        return distance_km * cost_per_km
    
    def _calculate_scenic_score(
        self,
        from_loc: GPSLocation,
        to_loc: GPSLocation,
        mode: TransportMode
    ) -> float:
        """Calculate scenic score for a route segment"""
        base_scenic = mode.value.get('scenic_factor', 0.5)
        
        # Boost scenic score for routes near water or historic areas
        avg_lat = (from_loc.latitude + to_loc.latitude) / 2
        avg_lng = (from_loc.longitude + to_loc.longitude) / 2
        
        # Bosphorus area
        if 28.95 <= avg_lng <= 29.05 and 41.0 <= avg_lat <= 41.1:
            base_scenic += 0.2
        
        # Historic peninsula
        if 28.92 <= avg_lng <= 28.99 and 41.0 <= avg_lat <= 41.02:
            base_scenic += 0.15
        
        return min(base_scenic, 1.0)
    
    def _get_user_profile(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'interests': preferences.get('interests', []),
                'preferred_transport': preferences.get('transport_modes', []),
                'mobility_needs': preferences.get('mobility_needs', []),
                'budget_preference': preferences.get('budget', 'medium'),
                'activity_level': preferences.get('activity_level', 'medium'),
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
        else:
            # Update with new preferences
            profile = self.user_profiles[user_id]
            profile.update(preferences)
            profile['last_updated'] = datetime.now()
        
        return self.user_profiles[user_id]
    
    def _calculate_personalization_score(
        self,
        waypoints: List[PersonalizedWaypoint],
        user_profile: Dict[str, Any]
    ) -> float:
        """Calculate overall personalization score for the route"""
        if not waypoints:
            return 0.0
        
        total_score = sum(
            waypoint.interest_match * 0.4 +
            waypoint.popularity_score * 0.2 +
            waypoint.accessibility_score * 0.2 +
            waypoint.weather_suitability * 0.2
            for waypoint in waypoints
        )
        
        return total_score / len(waypoints)
    
    async def _start_real_time_monitoring(self, route: PersonalizedRoute):
        """Start real-time monitoring for route updates"""
        logger.info(f"🔄 Starting real-time monitoring for route {route.route_id}")
        
        self.real_time_monitoring[route.route_id] = {
            'active': True,
            'last_update': datetime.now(),
            'update_frequency': 300  # 5 minutes
        }
        
        # This would be implemented as a background task in production
        # For now, we'll just log the setup
        logger.info(f"✅ Real-time monitoring active for route {route.route_id}")
    
    async def update_route_real_time(
        self,
        route_id: str,
        current_user_location: GPSLocation,
        context_updates: Dict[str, Any] = None
    ) -> Optional[PersonalizedRoute]:
        """Update route based on real-time conditions"""
        
        if route_id not in self.active_routes:
            logger.warning(f"Route {route_id} not found for real-time update")
            return None
        
        route = self.active_routes[route_id]
        
        # Check if significant location change
        distance_from_planned = self._calculate_distance(
            current_user_location,
            route.start_location
        )
        
        if distance_from_planned > 0.5:  # 500m threshold
            logger.info(f"📍 Significant location change detected, re-routing...")
            
            # Create updated route from new location
            updated_route = await self.create_personalized_route(
                route.user_id,
                current_user_location,
                {'interests': [], 'real_time_updates': True},  # Would extract from original route
                {'max_waypoints': len(route.waypoints)}
            )
            
            # Update the stored route
            self.active_routes[route_id] = updated_route
            
            logger.info(f"✅ Route {route_id} updated with new starting point")
            return updated_route
        
        # Update real-time conditions for existing segments
        for segment in route.segments:
            segment.real_time_conditions.update({
                'traffic_level': 'moderate',  # Would integrate with traffic API
                'weather_impact': 'minimal',
                'last_updated': datetime.now().isoformat()
            })
        
        route.last_updated = datetime.now()
        return route
    
    def get_museums_for_districts(self, districts: List[str]) -> List[Dict[str, Any]]:
        """Get museums for specific districts"""
        museums = []
        for museum in self.poi_database.get('museums', []):
            if museum.get('district') in districts:
                museums.append(museum)
        return museums
    
    def get_local_tips_for_districts(self, districts: List[str]) -> Dict[str, List[str]]:
        """Get local tips for specific districts"""
        tips = {}
        for district in districts:
            if district in self.district_tips:
                tips[district] = self.district_tips[district]
        return tips
    
    def enhance_route_with_museums_and_tips(self, route: PersonalizedRoute) -> Dict[str, Any]:
        """Enhance route response with museums and local tips for all districts"""
        # Extract districts from route waypoints
        route_districts = set()
        for waypoint in route.waypoints:
            if hasattr(waypoint.location, 'district') and waypoint.location.district:
                route_districts.add(waypoint.location.district)
            # Also try to determine district from coordinates
            district = self._get_district_from_coordinates(waypoint.location)
            if district:
                route_districts.add(district)
        
        # Get museums for all districts in the route
        district_museums = self.get_museums_for_districts(list(route_districts))
        
        # Get local tips for all districts
        district_tips = self.get_local_tips_for_districts(list(route_districts))
        
        # Create enhanced route response
        enhanced_response = {
            'route_info': {
                'route_id': route.route_id,
                'total_distance_km': route.total_distance_km,
                'total_time_minutes': route.total_time_minutes,
                'total_cost': route.total_cost,
                'personalization_score': route.personalization_score,
                'waypoints_count': len(route.waypoints)
            },
            'districts_covered': list(route_districts),
            'museums_in_route': district_museums,
            'local_tips_by_district': district_tips,
            'route_segments': [
                {
                    'from_location': segment.start_location.address if segment.start_location.address else f"({segment.start_location.latitude:.4f}, {segment.start_location.longitude:.4f})",
                    'to_location': segment.end_location.address if segment.end_location.address else f"({segment.end_location.latitude:.4f}, {segment.end_location.longitude:.4f})",
                    'transport_mode': segment.transport_mode.value,
                    'distance_km': segment.distance_km,
                    'time_minutes': segment.estimated_time_minutes,
                    'cost': segment.cost_estimate
                }
                for segment in route.segments
            ],
            'waypoints': [
                {
                    'name': wp.name,
                    'category': wp.category,
                    'location': f"({wp.location.latitude:.4f}, {wp.location.longitude:.4f})",
                    'district': wp.location.district,
                    'personalization_score': wp.personalization_score,
                    'visit_duration_minutes': wp.recommended_visit_duration
                }
                for wp in route.waypoints
            ]
        }
        
        return enhanced_response
    
    def _get_district_from_coordinates(self, location: GPSLocation) -> str:
        """Determine district from GPS coordinates"""
        min_distance = float('inf')
        closest_district = None
        
        for district_name, district_info in self.istanbul_districts.items():
            center = district_info['center']
            distance = self._calculate_distance(location, center)
            radius = district_info.get('radius_km', 2.0)
            
            if distance <= radius and distance < min_distance:
                min_distance = distance
                closest_district = district_name
        
        return closest_district or 'unknown'
    
    def _serialize_route(self, route: PersonalizedRoute) -> Dict[str, Any]:
        """Serialize route for caching"""
        return {
            'route_data': asdict(route),
            'serialization_version': '1.0'
        }
    
    def _deserialize_route(self, cached_data: Dict[str, Any]) -> PersonalizedRoute:
        """Deserialize route from cache"""
        route_data = cached_data['route_data']
        
        # Convert datetime strings back to datetime objects
        route_data['created_at'] = datetime.fromisoformat(route_data['created_at'])
        route_data['last_updated'] = datetime.fromisoformat(route_data['last_updated'])
        
        # Reconstruct complex objects
        route_data['start_location'] = GPSLocation(**route_data['start_location'])
        
        return PersonalizedRoute(**route_data)
    
    def get_route_suggestions(
        self,
        current_location: GPSLocation,
        user_id: str = None
    ) -> List[str]:
        """Generate contextual route suggestions based on current location"""
        
        suggestions = []
        district = self._get_district_from_location(current_location)
        
        # Location-specific suggestions
        if district == 'sultanahmet':
            suggestions.extend([
                "🏛️ You're in the historic heart! Perfect for a walking tour of Ottoman sites",
                "📸 Hagia Sophia and Blue Mosque are just steps away",
                "🍽️ Try traditional Ottoman cuisine at nearby historic restaurants"
            ])
        elif district == 'beyoglu':
            suggestions.extend([
                "🎨 Explore the artistic side of Istanbul with galleries and modern art",
                "🌉 Walk down to Galata Bridge for stunning Bosphorus views",
                "🛍️ Istiklal Street offers great shopping and street food"
            ])
        elif district == 'kadikoy':
            suggestions.extend([
                "🌊 Take a ferry ride for the best Bosphorus experience",
                "🍻 Explore the vibrant nightlife and local bars",
                "🏪 Visit the local markets for authentic Istanbul shopping"
            ])
        
        # Time-based suggestions
        hour = datetime.now().hour
        if 9 <= hour <= 17:
            suggestions.append("☀️ Perfect time for outdoor sightseeing and museum visits")
        elif 17 <= hour <= 21:
            suggestions.append("🌅 Golden hour - ideal for photography and sunset views")
        elif hour >= 21:
            suggestions.append("🌙 Evening is perfect for dining and experiencing Istanbul's nightlife")
        
        return suggestions[:4]  # Limit to top 4 suggestions
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, TransportMode):
            return data.name
        elif isinstance(data, GPSLocation):
            return {
                'latitude': data.latitude,
                'longitude': data.longitude,
                'accuracy': data.accuracy,
                'address': data.address,
                'district': data.district
            }
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
    
    def _restore_from_serializable(self, data: Any) -> Any:
        """Restore data from JSON-serializable format"""
        if isinstance(data, dict):
            # Check if it's a GPSLocation
            if all(key in data for key in ['latitude', 'longitude']):
                return GPSLocation(
                    latitude=data['latitude'],
                    longitude=data['longitude'],
                    accuracy=data.get('accuracy', 10.0),
                    address=data.get('address', ''),
                    district=data.get('district', '')
                )
            else:
                return {key: self._restore_from_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._restore_from_serializable(item) for item in data]
        elif isinstance(data, str) and data in [mode.name for mode in TransportMode]:
            return TransportMode[data]
        else:
            return data
    
    def _convert_location_to_gps(self, location_name: str, metadata: Dict[str, Any]) -> Optional[GPSLocation]:
        """Convert detected location name to GPS coordinates"""
        try:
            # Check if metadata contains coordinates
            if 'coordinates' in metadata:
                coords = metadata['coordinates']
                return GPSLocation(
                    latitude=coords[0],
                    longitude=coords[1],
                    address=location_name,
                    district=metadata.get('district', '')
                )
            
            # Check district centers
            location_lower = location_name.lower()
            for district_name, district_data in self.istanbul_districts.items():
                if district_name in location_lower or location_lower in district_name:
                    center = district_data['center']
                    return GPSLocation(
                        latitude=center.latitude,
                        longitude=center.longitude,
                        address=location_name,
                        district=district_name
                    )
            
            # Check POI database
            if hasattr(self, 'poi_database'):
                for poi in self.poi_database.get('attractions', []):
                    if poi['name'].lower() == location_lower:
                        return GPSLocation(
                            latitude=poi['location']['lat'],
                            longitude=poi['location']['lng'],
                            address=poi['name'],
                            district=poi.get('district', '')
                        )
            
            logger.warning(f"Could not find GPS coordinates for location: {location_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error converting location to GPS: {e}")
            return None

# Global enhanced GPS route planner instance
enhanced_gps_planner = EnhancedGPSRoutePlanner()

def get_enhanced_gps_planner() -> EnhancedGPSRoutePlanner:
    """Get the global enhanced GPS route planner instance"""
    return enhanced_gps_planner

async def standalone_create_route_with_intelligent_location_detection(
        user_id: str,
        user_input: str,
        user_profile: Any,
        conversation_context: Any,
        gps_location: Optional[Tuple[float, float]] = None,
        weather_data: Optional[Dict] = None,
        event_data: Optional[Dict] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a personalized route using the Intelligent Location Detector
        Integrates GPS, weather, events, and user context for best results
        
        Args:
            user_id: Unique user identifier
            user_input: User's message/query
            user_profile: User profile from main system
            conversation_context: Conversation context from main system
            gps_location: Optional GPS coordinates (lat, lng)
            weather_data: Optional weather context data
            event_data: Optional event context data
            preferences: User preferences for the route
            
        Returns:
            Enhanced route response with intelligent location detection
        """
        planner = get_enhanced_gps_planner()
        if not planner.intelligent_location_detector:
            logger.warning("Intelligent Location Detector not available, falling back to standard detection")
            return await planner.create_route_with_fallback_location(
                user_id, user_input, {}, None, None, preferences
            )
        
        try:
            # Build GPS context if GPS location available
            gps_context = None
            if gps_location:
                gps_context = GPSContext(
                    user_location=gps_location,
                    accuracy=10.0,  # Assume reasonable GPS accuracy
                    movement_pattern='stationary',  # Default assumption
                    nearby_landmarks=[],
                    district_proximity={}
                )
            
            # Build weather context if available
            weather_context = None
            if weather_data:
                weather_context = WeatherContext(
                    current_weather=weather_data.get('current', {}),
                    forecast=weather_data.get('forecast', []),
                    temperature=weather_data.get('temperature', 20),
                    precipitation=weather_data.get('precipitation', 0),
                    wind_speed=weather_data.get('wind_speed', 0),
                    weather_type=weather_data.get('type', 'clear')
                )
            
            # Build event context if available
            event_context = None
            if event_data:
                event_context = EventContext(
                    current_events=event_data.get('current_events', []),
                    cultural_events=event_data.get('cultural_events', []),
                    festivals=event_data.get('festivals', []),
                    seasonal_activities=event_data.get('seasonal_activities', [])
                )
            
            # Use Intelligent Location Detector with full context
            logger.info(f"🎯 Using Intelligent Location Detector with context-aware detection...")
            location_result = planner.intelligent_location_detector.detect_location_with_context(
                user_input=user_input,
                user_profile=user_profile,
                context=conversation_context,
                gps_context=gps_context,
                weather_context=weather_context,
                event_context=event_context
            )
            
            if not location_result or not location_result.location:
                logger.warning("No location detected by Intelligent Location Detector, falling back...")
                return await planner.create_route_with_fallback_location(
                    user_id, user_input, {}, None, None, preferences
                )
            
            logger.info(f"✅ Location detected: {location_result.location} (confidence: {location_result.confidence:.2f}, method: {location_result.detection_method})")
            
            # Convert detected location to GPS coordinates
            detected_gps_location = planner._convert_location_to_gps(
                location_result.location,
                location_result.metadata
            )
            
            if not detected_gps_location:
                logger.warning("Could not convert detected location to GPS coordinates")
                return await planner.create_route_with_fallback_location(
                    user_id, user_input, {}, None, None, preferences
                )
            
            # Create personalized route using detected location
            route = await planner.create_personalized_route(
                user_id,
                detected_gps_location,
                preferences or {}
            )
            
            # Enhance response with museums and local tips
            enhanced_response = planner.enhance_route_with_museums_and_tips(route)
            
            # Add intelligent location detection information
            enhanced_response['location_detection'] = {
                'method': 'intelligent_location_detector',
                'detection_method': location_result.detection_method,
                'confidence': location_result.confidence,
                'detected_location': location_result.location,
                'explanation': location_result.explanation,
                'alternative_locations': list(location_result.alternative_scores.keys())[:3] if location_result.alternative_scores else [],
                'gps_aware': gps_context is not None,
                'weather_aware': weather_context is not None,
                'event_aware': event_context is not None,
                'context_match_scores': location_result.context_match
            }
            
            # Add enhanced recommendations from location detector
            if location_result.recommendations:
                enhanced_response['intelligent_recommendations'] = location_result.recommendations
            
            # Add recommendation summary with intelligent features
            enhanced_response['recommendation_summary'] = {
                'total_museums_available': len(enhanced_response['museums_in_route']),
                'districts_with_tips': len(enhanced_response['local_tips_by_district']),
                'personalization_level': 'ai_powered',  # Highest level with intelligent detection
                'route_optimization': f'AI-powered optimization using {location_result.detection_method}',
                'location_accuracy': 'very_high' if location_result.confidence > 0.9 else 'high' if location_result.confidence > 0.7 else 'good',
                'context_awareness': {
                    'gps_integrated': gps_context is not None,
                    'weather_integrated': weather_context is not None,
                    'events_integrated': event_context is not None,
                    'user_profile_integrated': True
                }
            }
            
            logger.info(f"🚀 Created intelligent route with {len(enhanced_response.get('waypoints', []))} waypoints")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Intelligent location detection failed: {e}")
            if hasattr(planner, 'ml_debug_mode') and planner.ml_debug_mode:
                import traceback
                logger.debug(f"Intelligent detection error:\n{traceback.format_exc()}")
            
            # Fallback to standard detection
            return await planner.create_route_with_fallback_location(
                user_id, user_input, {}, None, None, preferences
            )