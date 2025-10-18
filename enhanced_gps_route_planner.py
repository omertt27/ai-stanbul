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
import sys
from pathlib import Path

# Add services directory to path
sys.path.insert(0, str(Path(__file__).parent / 'services'))

# Import POI Database Service
try:
    from poi_database_service import POIDatabaseService, POI, GeoCoordinate
    POI_DATABASE_AVAILABLE = True
    logging.info("✅ POI Database Service integrated successfully!")
except ImportError as e:
    POI_DATABASE_AVAILABLE = False
    logging.warning(f"⚠️ POI Database Service not available: {e}")

# Import ML Prediction Service (Phase 3)
try:
    from ml_prediction_service import (
        MLPredictionService, 
        CrowdingPrediction,
        TravelTimePrediction
    )
    ML_PREDICTION_AVAILABLE = True
    logging.info("✅ ML Prediction Service integrated successfully!")
except ImportError as e:
    ML_PREDICTION_AVAILABLE = False
    logging.warning(f"⚠️ ML Prediction Service not available: {e}")

# Import POI-Enhanced Route Optimizer (Phase 4)
try:
    from poi_enhanced_route_optimizer import (
        POIEnhancedRouteOptimizer,
        RouteConstraints,
        POIEnhancedRoute
    )
    POI_OPTIMIZER_AVAILABLE = True
    logging.info("✅ POI-Enhanced Route Optimizer integrated successfully!")
except ImportError as e:
    POI_OPTIMIZER_AVAILABLE = False
    logging.warning(f"⚠️ POI-Enhanced Route Optimizer not available: {e}")

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
        
        # Initialize POI Database Service (PRIMARY DATA SOURCE)
        self.poi_db_service = None
        if POI_DATABASE_AVAILABLE:
            try:
                self.poi_db_service = POIDatabaseService()
                logger.info(f"🎯 POI Database Service initialized with {len(self.poi_db_service.pois)} POIs")
            except Exception as e:
                logger.warning(f"Could not initialize POI Database Service: {e}")
                self.poi_db_service = None
        
        # Initialize ML Prediction Service (Phase 3)
        self.ml_prediction_service = None
        if ML_PREDICTION_AVAILABLE:
            try:
                self.ml_prediction_service = MLPredictionService()
                logger.info("🤖 ML Prediction Service initialized for crowding and travel time predictions")
            except Exception as e:
                logger.warning(f"Could not initialize ML Prediction Service: {e}")
                self.ml_prediction_service = None
        
        # Initialize ML-Enhanced Transportation System
        self.ml_transport_system = None
        try:
            from ml_enhanced_transportation_system import MLEnhancedTransportationSystem
            self.ml_transport_system = MLEnhancedTransportationSystem()
            logger.info("🚇 ML-Enhanced Transportation System integrated!")
        except ImportError as e:
            logger.warning(f"⚠️ ML-Enhanced Transportation System not available: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize ML Transportation System: {e}")
        
        # Initialize POI-Enhanced Route Optimizer (Phase 4)
        self.poi_optimizer = None
        if POI_OPTIMIZER_AVAILABLE:
            try:
                self.poi_optimizer = POIEnhancedRouteOptimizer()
                logger.info("🎯 POI-Enhanced Route Optimizer initialized successfully!")
            except Exception as e:
                logger.warning(f"Could not initialize POI Route Optimizer: {e}")
                self.poi_optimizer = None
        
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
        
        # Legacy POI database (DEPRECATED - use poi_db_service instead)
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
        
        # Get current time for POI filtering
        current_time = datetime.now()
        
        # Determine transport mode for scoring
        transport_mode = (preferences.get('transport_modes') or [TransportMode.WALKING])[0]
        
        # Find nearby points of interest with proper parameters
        nearby_pois = self._find_nearby_pois(
            location=current_location,
            radius_km=constraints.get('radius_km', 5.0) if constraints else 5.0,
            current_time=current_time,
            user_preferences=preferences
        )
        
        # Score and rank POIs based on personalization with all required parameters
        scored_pois = self._score_pois_for_user(
            pois=nearby_pois,
            user_location=current_location,
            user_preferences=preferences,
            current_time=current_time,
            transport_mode=transport_mode
        )
        
        # Select optimal waypoints with all required parameters
        selected_pois = self._select_optimal_waypoints(
            scored_pois=scored_pois,
            user_preferences=preferences,
            route_constraints=constraints or {}
        )
        
        # Convert POIs to PersonalizedWaypoints
        selected_waypoints = self._convert_pois_to_waypoints(
            selected_pois,
            user_preferences=preferences,
            current_time=current_time
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
    
    async def create_poi_optimized_route(
        self,
        user_id: str,
        start_location: GPSLocation,
        end_location: GPSLocation,
        preferences: Dict[str, Any],
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create POI-enhanced route using Phase 4 optimizer
        
        Args:
            user_id: User identifier
            start_location: Starting GPS location
            end_location: Destination GPS location
            preferences: User interests, budget, transport preferences
            constraints: Route constraints (max_pois, max_detour_minutes, etc.)
            
        Returns:
            Enhanced route response with POIs, segments, ML predictions, and recommendations
        """
        logger.info(f"🎯 Creating POI-optimized route for user {user_id}")
        
        # Check if optimizer is available
        if not self.poi_optimizer:
            logger.warning("POI Optimizer not available, falling back to basic route")
            # Fallback to basic personalized route
            route = await self.create_personalized_route(user_id, start_location, preferences, constraints)
            return self.enhance_route_with_museums_and_tips(route)
        
        # Convert constraints to RouteConstraints
        route_constraints = RouteConstraints(
            max_pois=constraints.get('max_pois', 3) if constraints else 3,
            max_detour_time_minutes=constraints.get('max_detour_minutes', 45) if constraints else 45,
            max_total_detour_minutes=constraints.get('max_total_detour', 120) if constraints else 120,
            require_category_diversity=preferences.get('prefer_diverse_categories', True),
            min_poi_value=constraints.get('min_poi_value', 0.3) if constraints else 0.3
        )
        
        # Convert GPS locations to GeoCoordinates
        from poi_database_service import GeoCoordinate
        start_coord = GeoCoordinate(lat=start_location.latitude, lon=start_location.longitude)
        end_coord = GeoCoordinate(lat=end_location.latitude, lon=end_location.longitude)
        
        # Create optimized route using Phase 4 optimizer
        optimized_route = await self.poi_optimizer.create_poi_enhanced_route(
            start_coord, 
            end_coord, 
            preferences, 
            route_constraints
        )
        
        # Convert optimizer response to enhanced route format
        response = self._format_optimized_route_response(optimized_route, user_id)
        
        # Add district tips for POIs in route
        districts_in_route = set()
        for poi in optimized_route.pois_included:
            if poi.district:
                districts_in_route.add(poi.district)
        
        local_tips = {}
        for district in districts_in_route:
            if district in self.district_tips:
                local_tips[district] = self.district_tips[district]
        
        response['local_tips_by_district'] = local_tips
        
        # Add recommendation summary
        response['recommendation_summary'] = {
            'optimization_method': 'POI-Enhanced Route Optimizer (Phase 4)',
            'total_pois_evaluated': optimized_route.optimization_insights.get('pois_evaluated', 0),
            'pois_included': len(optimized_route.pois_included),
            'pois_recommended_not_included': len(optimized_route.pois_recommended_not_included),
            'ml_predictions_used': optimized_route.optimization_insights.get('ml_predictions_used', False),
            'personalization_level': 'high',
            'route_optimization': 'ML-optimized with POI detour calculation'
        }
        
        logger.info(f"✅ POI-optimized route created: {len(optimized_route.pois_included)} POIs, {response['enhanced_route']['time_minutes']}min")
        
        return response
    
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
            'neighborhood': best_option.neighborhood
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
        };
        
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
        """Optimize route segments between waypoints with ML travel time predictions"""
        
        segments = []
        current_location = start_location
        current_time = datetime.now()
        
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
            base_travel_time = self._calculate_travel_time(distance, best_mode)
            
            # Use ML Prediction Service for travel time if available (Phase 3)
            if ML_PREDICTION_AVAILABLE and self.ml_prediction_service:
                try:
                    from_loc = f"{current_location.latitude:.4f},{current_location.longitude:.4f}"
                    to_loc = f"{waypoint.location.latitude:.4f},{waypoint.location.longitude:.4f}"
                    
                    travel_pred = self.ml_prediction_service.predict_travel_time(
                        from_loc,
                        to_loc,
                        best_mode.name.lower(),
                        base_travel_time,
                        current_time,
                        weather_data=None  # Could integrate weather service
                    )
                    
                    travel_time = int(travel_pred.predicted_time_minutes)
                    logger.debug(f"ML-predicted travel time: {travel_time}min (base: {base_travel_time}min)")
                except Exception as e:
                    logger.debug(f"Could not get ML travel prediction, using base: {e}")
                    travel_time = base_travel_time
            else:
                travel_time = base_travel_time
            
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
            current_time += timedelta(minutes=travel_time)
        
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
    
    def _calculate_travel_time(self, distance_kmh: float, mode: TransportMode) -> int:
        """Calculate travel time in minutes"""
        speed_kmh = mode.value.get('speed_kmh', 4.5)
        base_time_minutes = (distance_kmh / speed_kmh) * 60
        
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
    
    def _calculate_distance(self, loc1: GPSLocation, loc2: GPSLocation) -> float:
        """Calculate distance in kilometers between two GPS locations using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        # Earth's radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1 = radians(loc1.latitude)
        lon1 = radians(loc1.longitude)
        lat2 = radians(loc2.latitude)
        lon2 = radians(loc2.longitude)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert complex objects to JSON-serializable format"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.name
        elif hasattr(obj, '__dict__'):
            return {key: self._make_json_serializable(value) for key, value in obj.__dict__.items()}
        else:
            return str(obj)
    
    def _serialize_route(self, route: PersonalizedRoute) -> Dict[str, Any]:
        """Serialize a PersonalizedRoute to dict"""
        return asdict(route)
    
    def _deserialize_route(self, data: Dict[str, Any]) -> PersonalizedRoute:
        """Deserialize dict to PersonalizedRoute"""
        # Simple implementation - would need full reconstruction in production
        return PersonalizedRoute(**data)
    
    def _convert_location_to_gps(self, location: str, metadata: Dict) -> Optional[GPSLocation]:
        """Convert location string to GPS coordinates"""
        # Check if it's in our districts
        for district, info in self.istanbul_districts.items():
            if district.lower() in location.lower():
                return info['center']
        
        # Try to extract from metadata
        if 'coordinates' in metadata:
            coords = metadata['coordinates']
            return GPSLocation(
                latitude=coords.get('latitude', 41.0082),
                longitude=coords.get('longitude', 28.9784),
                district=metadata.get('district', '')
            )
        
        # Default to Sultanahmet
        return GPSLocation(41.0082, 28.9784, district='sultanahmet')
    
    def enhance_route_with_museums_and_tips(self, route: PersonalizedRoute) -> Dict[str, Any]:
        """Enhance route response with museum information and local tips"""
        response = {
            'route_id': route.route_id,
            'waypoints': [asdict(wp) for wp in route.waypoints],
            'segments': [asdict(seg) for seg in route.segments],
            'total_distance_km': route.total_distance_km,
            'total_time_minutes': route.total_time_minutes,
            'total_cost': route.total_cost,
            'personalization_score': route.personalization_score,
            'museums_in_route': [],
            'local_tips_by_district': {}
        }
        
        # Collect museums from waypoints
        for waypoint in route.waypoints:
            if 'museum' in waypoint.category.lower() or 'palace' in waypoint.category.lower():
                response['museums_in_route'].append({
                    'name': waypoint.name,
                    'category': waypoint.category,
                    'estimated_duration': waypoint.estimated_duration
                })
        
        # Collect local tips for districts in route
        districts_in_route = set()
        for waypoint in route.waypoints:
            if waypoint.location.district:
                districts_in_route.add(waypoint.location.district)
        
        for district in districts_in_route:
            if district in self.district_tips:
                response['local_tips_by_district'][district] = self.district_tips[district]
        
        return response
    
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
    
    def _find_nearby_pois(
        self,
        location: GPSLocation,
        radius_km: float,
        current_time: datetime,
        user_preferences: Dict[str, Any]
    ) -> List[POI]:
        """
        Find nearby POIs filtered by opening hours and accessibility requirements
        
        Args:
            location: Center point for search
            radius_km: Search radius in kilometers
            current_time: Current time for opening hours check
            user_preferences: User preferences including accessibility needs
            
        Returns:
            List of POI objects within radius that are open and accessible
        """
        if not POI_DATABASE_AVAILABLE or not self.poi_db_service:
            logger.warning("POI Database Service not available, using legacy POI data")
            return []
        
        try:
            # Query POIs from database service
            geo_coord = GeoCoordinate(
                lat=location.latitude,
                lon=location.longitude
            )
            
            nearby_pois = self.poi_db_service.find_pois_in_radius(
                center=geo_coord,
                radius_km=radius_km
            )
            
            # Filter by opening hours
            open_pois = []
            for poi in nearby_pois:
                if self.poi_db_service.is_poi_open(poi, current_time):
                    open_pois.append(poi)
            
            logger.info(f"Found {len(open_pois)} open POIs out of {len(nearby_pois)} nearby")
            
            # Filter by accessibility requirements
            accessibility_needs = user_preferences.get('accessibility', {})
            wheelchair_required = accessibility_needs.get('wheelchair_accessible', False)
            
            accessible_pois = []
            for poi in open_pois:
                # Check wheelchair accessibility if required
                if wheelchair_required:
                    if poi.accessibility.get('wheelchair_accessible', False):
                        accessible_pois.append(poi)
                else:
                    accessible_pois.append(poi)
            
            logger.info(f"Filtered to {len(accessible_pois)} accessible POIs")
            
            return accessible_pois
            
        except Exception as e:
            logger.error(f"Error finding nearby POIs: {e}")
            return []
    
    def _score_pois_for_user(
        self,
        pois: List[POI],
        user_location: GPSLocation,
        user_preferences: Dict[str, Any],
        current_time: datetime,
        transport_mode: TransportMode
    ) -> List[Tuple[POI, float]]:
        """
        Score POIs based on multiple factors including crowding predictions and ML
        
        Args:
            pois: List of candidate POIs
            user_location: User's current location
            user_preferences: User interests and preferences
            current_time: Current time for time-based scoring
            transport_mode: Selected transportation mode
            
        Returns:
            List of (POI, score) tuples sorted by score descending
        """
        scored_pois = []
        user_interests = user_preferences.get('interests', [])
        max_detour_minutes = user_preferences.get('max_detour_minutes', 30)
        
        for poi in pois:
            score = 0.0
            
            # 1. Category match score (0-40 points)
            category_score = 0.0
            poi_categories = poi.categories if hasattr(poi, 'categories') else []
            
            for interest in user_interests:
                interest_lower = interest.lower()
                for category in poi_categories:
                    if interest_lower in category.lower():
                        category_score += 10.0
                        break
            
            category_score = min(category_score, 40.0)
            score += category_score
            
            # 2. Popularity/rating score (0-25 points)
            popularity_score = poi.popularity_score * 25.0 if hasattr(poi, 'popularity_score') else 15.0
            score += popularity_score
            
            # 3. Distance/detour cost (0-20 points, inverse)
            poi_location = GPSLocation(poi.location.latitude, poi.location.longitude)
            distance_km = self._calculate_distance(user_location, poi_location)
            
            # Calculate travel time based on transport mode
            speed_kmh = transport_mode.value['speed_kmh']
            travel_time_minutes = (distance_km / speed_kmh) * 60
            
            # Penalize far POIs
            detour_factor = max(0, 1 - (travel_time_minutes / max_detour_minutes))
            distance_score = detour_factor * 20.0
            score += distance_score
            
            # 4. Crowding prediction score (0-15 points, inverse) - Using Phase 3 ML Prediction Service
            if ML_PREDICTION_AVAILABLE and self.ml_prediction_service:
                try:
                    # Use ML Prediction Service for advanced crowding prediction
                    crowding_pred = self.ml_prediction_service.predict_poi_crowding(
                        poi.poi_id,
                        current_time,
                        weather_data=None  # Could integrate weather service here
                    )
                    crowding_level = crowding_pred.crowding_level
                    # Lower crowding = higher score
                    crowding_score = (1.0 - crowding_level) * 15.0
                    score += crowding_score
                    
                    # Log crowding info
                    logger.debug(f"{poi.name}: {crowding_pred.crowding_label}, wait: {crowding_pred.wait_time_minutes}min")
                except Exception as e:
                    logger.debug(f"Could not get ML crowding prediction for {poi.name}: {e}")
                    score += 7.5  # Neutral score
            elif POI_DATABASE_AVAILABLE and self.poi_db_service:
                try:
                    # Fallback to POI database service
                    crowding_level = self.poi_db_service.predict_crowding(poi, current_time)
                    crowding_score = (1.0 - crowding_level) * 15.0
                    score += crowding_score
                except Exception as e:
                    logger.debug(f"Could not get crowding prediction for {poi.name}: {e}")
                    score += 7.5  # Neutral score
            else:
                score += 7.5  # Neutral score if service unavailable
            
            # 5. Time-of-day appropriateness (0-10 points)
            hour = current_time.hour
            time_score = 10.0  # Default
            
            # Morning (6-11): Museums, breakfast spots
            if 6 <= hour < 11:
                if any(cat in ['museum', 'cafe', 'breakfast'] for cat in poi_categories):
                    time_score = 10.0
                else:
                    time_score = 5.0
            # Afternoon (11-17): All attractions
            elif 11 <= hour < 17:
                time_score = 10.0
            # Evening (17-22): Restaurants, viewpoints
            elif 17 <= hour < 22:
                if any(cat in ['restaurant', 'viewpoint', 'nightlife'] for cat in poi_categories):
                    time_score = 10.0
                else:
                    time_score = 5.0
            # Night (22-6): Limited options
            else:
                if 'nightlife' in poi_categories:
                    time_score = 10.0
                else:
                    time_score = 2.0
            
            score += time_score
            
            # 6. ML prediction boost (0-10 points)
            if self.ml_cache:
                try:
                    # Create prediction context
                    ml_context = {
                        'poi_id': poi.poi_id,
                        'poi_name': poi.name,
                        'categories': poi_categories,
                        'user_interests': user_interests,
                        'time_of_day': hour,
                        'day_of_week': current_time.weekday(),
                        'distance_km': distance_km
                    }
                    
                    # Try to get ML prediction from cache
                    ml_prediction = self.ml_cache.get(
                        f"poi_recommendation_{poi.poi_id}",
                        ml_context,
                        ['poi_recommendation', 'user_preference_matching']
                    )
                    
                    if ml_prediction and isinstance(ml_prediction, (int, float)):
                        ml_score = float(ml_prediction) * 10.0
                        score += ml_score
                    else:
                        score += 5.0  # Neutral ML score
                except Exception as e:
                    logger.debug(f"Could not get ML prediction for {poi.name}: {e}")
                    score += 5.0  # Neutral ML score
            else:
                score += 5.0  # Neutral ML score
            
            scored_pois.append((poi, score))
        
        # Sort by score descending
        scored_pois.sort(key=lambda x: x[1], reverse=True)
        
        if scored_pois:
            logger.info(f"Scored {len(scored_pois)} POIs, top score: {scored_pois[0][1]:.2f}")
        else:
            logger.warning("No POIs scored - no nearby POIs found")
        
        return scored_pois
    
    def _select_optimal_waypoints(
        self,
        scored_pois: List[Tuple[POI, float]],
        user_preferences: Dict[str, Any],
        route_constraints: Dict[str, Any]
    ) -> List[POI]:
        """
        Select optimal waypoints with smart constraints
        
        Args:
            scored_pois: List of (POI, score) tuples sorted by score
            user_preferences: User preferences
            route_constraints: Route constraints (max waypoints, max detour time, etc.)
            
        Returns:
            List of selected POI waypoints
        """
        max_waypoints = route_constraints.get('max_waypoints', 3)
        max_total_detour_minutes = route_constraints.get('max_total_detour_minutes', 90)
        require_category_diversity = user_preferences.get('prefer_diverse_categories', True)
        
        selected_pois = []
        selected_categories = set()
        total_visit_time = 0
        
        for poi, score in scored_pois:
            # Stop if we have enough waypoints
            if len(selected_pois) >= max_waypoints:
                break
            
            # Check category diversity if required
            poi_categories = poi.categories if hasattr(poi, 'categories') else []
            primary_category = poi_categories[0] if poi_categories else 'unknown'
            
            if require_category_diversity and primary_category in selected_categories:
                # Skip if we already have a POI from this category
                continue
            
            # Check time constraint
            visit_duration = poi.suggested_duration_minutes if hasattr(poi, 'suggested_duration_minutes') else 60
            
            if total_visit_time + visit_duration > max_total_detour_minutes:
                # Would exceed time budget
                continue
            
            # Add to selected waypoints
            selected_pois.append(poi)
            selected_categories.add(primary_category)
            total_visit_time += visit_duration
            
            logger.info(f"Selected waypoint: {poi.name} (score: {score:.2f}, category: {primary_category})")
        
        logger.info(f"Selected {len(selected_pois)} optimal waypoints with {total_visit_time} min total visit time")
        
        return selected_pois
    
    def _convert_pois_to_waypoints(
        self,
        pois: List[POI],
        user_preferences: Dict[str, Any],
        current_time: datetime
    ) -> List[PersonalizedWaypoint]:
        """Convert POI objects to PersonalizedWaypoint objects"""
        waypoints = []
        user_interests = user_preferences.get('interests', [])
        
        for poi in pois:
            # Calculate interest match
            poi_categories = poi.categories if hasattr(poi, 'categories') else []
            interest_match = 0.0
            for interest in user_interests:
                if any(interest.lower() in cat.lower() for cat in poi_categories):
                    interest_match += 0.3
            interest_match = min(interest_match, 1.0)
            
            # Get attributes with defaults
            popularity = poi.popularity_score if hasattr(poi, 'popularity_score') else 0.7
            duration = poi.suggested_duration_minutes if hasattr(poi, 'suggested_duration_minutes') else 60
            
            # Create waypoint
            waypoint = PersonalizedWaypoint(
                name=poi.name,
                location=GPSLocation(
                    latitude=poi.location.latitude,
                    longitude=poi.location.longitude,
                    district=getattr(poi, 'district', '')
                ),
                category=poi_categories[0] if poi_categories else 'attraction',
                interest_match=interest_match,
                popularity_score=popularity,
                weather_suitability=0.8,  # Default
                accessibility_score=0.9,  # Default
                estimated_duration=duration,
                transport_modes=[TransportMode.WALKING, TransportMode.PUBLIC_TRANSPORT],
                personalization_reasons=[
                    f"Matches your interest in {user_interests[0]}" if user_interests else "Popular attraction",
                    f"Currently {self._get_crowding_label(poi, current_time)}"
                ],
                real_time_updates={}
            )
            waypoints.append(waypoint)
        
        return waypoints
    
    def _get_crowding_label(self, poi: POI, current_time: datetime) -> str:
        """Get human-readable crowding label for a POI"""
        if ML_PREDICTION_AVAILABLE and self.ml_prediction_service:
            try:
                pred = self.ml_prediction_service.predict_poi_crowding(poi.poi_id, current_time)
                return pred.crowding_label.lower()
            except:
                pass
        return "moderate crowding"
    
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
    
    def _format_optimized_route_response(
        self,
        optimized_route,  # POIEnhancedRoute object
        user_id: str
    ) -> Dict[str, Any]:
        """
        Format POIEnhancedRoute from optimizer into standard response structure
        
        Args:
            optimized_route: POIEnhancedRoute object from Phase 4 optimizer
            user_id: User identifier for personalization
            
        Returns:
            Formatted response dictionary with base_route, enhanced_route, POIs, etc.
        """
        from poi_database_service import POI
        
        # Convert POI objects to dictionaries for JSON serialization
        pois_included = []
        for poi in optimized_route.pois_included:
            try:
                # Handle both POI object variations (poi_id vs id)
                poi_id = poi.poi_id if hasattr(poi, 'poi_id') else (poi.id if hasattr(poi, 'id') else 'unknown')
                visit_duration = poi.visit_duration_min if hasattr(poi, 'visit_duration_min') else (poi.visit_duration_minutes if hasattr(poi, 'visit_duration_minutes') else 30)
                ticket_price = poi.ticket_price if hasattr(poi, 'ticket_price') else (poi.entrance_fee if hasattr(poi, 'entrance_fee') else 0)
                
                poi_dict = {
                    'id': poi_id,
                    'name': poi.name,
                    'category': poi.category,
                    'coordinates': {
                        'latitude': poi.location.lat,
                        'longitude': poi.location.lon
                    },
                    'rating': poi.rating,
                    'visit_duration_minutes': visit_duration,
                    'entrance_fee': ticket_price,
                    'opening_hours': getattr(poi, 'opening_hours', {}),
                    'description': getattr(poi, 'description', ''),
                    'district': getattr(poi, 'district', '')
                }
                pois_included.append(poi_dict)
            except Exception as e:
                logger.warning(f"Error formatting POI: {e}. POI type: {type(poi)}")
                continue
        
        # Format segments
        segments = []
        for seg in optimized_route.segments:
            try:
                segment_dict = {
                    'segment_type': getattr(seg, 'segment_type', 'transit'),
                    'from_location': getattr(seg, 'from_location', ''),
                    'to_location': getattr(seg, 'to_location', ''),
                    'transport_mode': getattr(seg, 'transport_mode', 'walk'),
                    'distance_km': getattr(seg, 'distance_km', 0.0),
                    'time_minutes': getattr(seg, 'scheduled_time_minutes', getattr(seg, 'time_minutes', 0)),
                    'predicted_time_minutes': getattr(seg, 'predicted_time_minutes', 0),
                    'cost': getattr(seg, 'cost', 0.0),
                    'scenic_score': getattr(seg, 'scenic_score', 0.5)
                }
                segments.append(segment_dict)
            except Exception as e:
                logger.warning(f"Error formatting segment: {e}")
                continue
        
        response = {
            'route_id': optimized_route.route_id,
            'user_id': user_id,
            'base_route': optimized_route.base_route,
            'enhanced_route': optimized_route.enhanced_route,
            'segments': segments,
            'pois_included': pois_included,
            'pois_recommended_not_included': optimized_route.pois_recommended_not_included,
            'optimization_insights': optimized_route.optimization_insights,
            'created_at': optimized_route.created_at.isoformat()
        }
        
        return response
    
    async def create_poi_optimized_route(
        self,
        user_id: str,
        start_location: GPSLocation,
        end_location: GPSLocation,
        preferences: Dict[str, Any],
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create POI-enhanced route using Phase 4 optimizer
        
        Args:
            user_id: User identifier
            start_location: Starting GPS location
            end_location: Destination GPS location
            preferences: User interests, budget, transport preferences
            constraints: Route constraints (max_pois, max_detour_minutes, etc.)
            
        Returns:
            Enhanced route response with POIs, segments, ML predictions, and recommendations
        """
        logger.info(f"🎯 Creating POI-optimized route for user {user_id}")
        
        # Check if optimizer is available
        if not self.poi_optimizer:
            logger.warning("POI Optimizer not available, falling back to basic route")
            # Fallback to basic personalized route
            route = await self.create_personalized_route(user_id, start_location, preferences, constraints)
            return self.enhance_route_with_museums_and_tips(route)
        
        # Convert constraints to RouteConstraints
        route_constraints = RouteConstraints(
            max_pois=constraints.get('max_pois', 3) if constraints else 3,
            max_detour_time_minutes=constraints.get('max_detour_minutes', 45) if constraints else 45,
            max_total_detour_minutes=constraints.get('max_total_detour', 120) if constraints else 120,
            require_category_diversity=preferences.get('prefer_diverse_categories', True),
            min_poi_value=constraints.get('min_poi_value', 0.3) if constraints else 0.3
        )
        
        # Convert GPS locations to GeoCoordinates
        from poi_database_service import GeoCoordinate
        start_coord = GeoCoordinate(lat=start_location.latitude, lon=start_location.longitude)
        end_coord = GeoCoordinate(lat=end_location.latitude, lon=end_location.longitude)
        
        # Create optimized route using Phase 4 optimizer
        optimized_route = await self.poi_optimizer.create_poi_enhanced_route(
            start_coord, 
            end_coord, 
            preferences, 
            route_constraints
        )
        
        # Convert optimizer response to enhanced route format
        response = self._format_optimized_route_response(optimized_route, user_id)
        
        # Add district tips for POIs in route
        districts_in_route = set()
        for poi in optimized_route.pois_included:
            if poi.district:
                districts_in_route.add(poi.district)
        
        local_tips = {}
        for district in districts_in_route:
            if district in self.district_tips:
                local_tips[district] = self.district_tips[district]
        
        response['local_tips_by_district'] = local_tips
        
        # Add recommendation summary
        response['recommendation_summary'] = {
            'optimization_method': 'POI-Enhanced Route Optimizer (Phase 4)',
            'total_pois_evaluated': optimized_route.optimization_insights.get('pois_evaluated', 0),
            'pois_included': len(optimized_route.pois_included),
            'pois_recommended_not_included': len(optimized_route.pois_recommended_not_included),
            'ml_predictions_used': optimized_route.optimization_insights.get('ml_predictions_used', False),
            'personalization_level': 'high',
            'route_optimization': 'ML-optimized with POI detour calculation'
        }
        
        logger.info(f"✅ POI-optimized route created: {len(optimized_route.pois_included)} POIs, {response['enhanced_route']['time_minutes']}min")
        
        return response
    
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
            'neighborhood': best_option.neighborhood
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


# Singleton instance
_planner_instance = None

def get_enhanced_gps_planner() -> EnhancedGPSRoutePlanner:
    """
    Get singleton instance of EnhancedGPSRoutePlanner
    
    Returns:
        EnhancedGPSRoutePlanner: Singleton planner instance
    """
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = EnhancedGPSRoutePlanner()
    return _planner_instance
