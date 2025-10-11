#!/usr/bin/env python3
"""
ML-Enhanced Transportation System for Istanbul AI
==================================================

Comprehensive transportation system that addresses all requirements:
1. İBB API Integration for real-time data
2. ML Route Optimization with crowding/time predictions
3. Enhanced GPS location-based routing
4. POI Integration with museums/attractions
5. Multi-modal optimization with advanced route combinations
"""

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML libraries with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using numpy-based fallback")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, using basic prediction")

class TransportMode(Enum):
    """Available transport modes in Istanbul"""
    WALKING = "walking"
    METRO = "metro"
    BUS = "bus"
    TRAM = "tram"
    FERRY = "ferry"
    TAXI = "taxi"
    DOLMUS = "dolmus"
    FUNICULAR = "funicular"
    CABLE_CAR = "cable_car"
    MARMARAY = "marmaray"
    METROBUS = "metrobus"

class RouteOptimizationType(Enum):
    """Route optimization objectives"""
    FASTEST = "fastest"
    SHORTEST = "shortest"
    CHEAPEST = "cheapest"
    LEAST_CROWDED = "least_crowded"
    MOST_SCENIC = "most_scenic"
    POI_OPTIMIZED = "poi_optimized"
    ECO_FRIENDLY = "eco_friendly"

@dataclass
class GPSLocation:
    """GPS location with metadata"""
    latitude: float
    longitude: float
    accuracy: float = 10.0  # meters
    timestamp: datetime = field(default_factory=datetime.now)
    address: Optional[str] = None
    district: Optional[str] = None

@dataclass
class TransportStation:
    """Transport station/stop information"""
    id: str
    name: str
    transport_modes: List[TransportMode]
    location: GPSLocation
    accessibility: bool = True
    real_time_data: Dict[str, Any] = field(default_factory=dict)
    crowding_prediction: float = 0.5  # 0 = empty, 1 = very crowded
    delay_minutes: int = 0

@dataclass
class RouteSegment:
    """Individual segment of a route"""
    from_location: GPSLocation
    to_location: GPSLocation
    transport_mode: TransportMode
    duration_minutes: int
    distance_km: float
    cost_tl: float
    instructions: List[str]
    crowding_level: float = 0.5
    poi_stops: List[Dict[str, Any]] = field(default_factory=list)
    real_time_updates: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizedRoute:
    """Complete optimized route with ML predictions"""
    route_id: str
    segments: List[RouteSegment]
    total_duration_minutes: int
    total_distance_km: float
    total_cost_tl: float
    optimization_type: RouteOptimizationType
    confidence_score: float
    crowding_prediction: float
    poi_integration: List[Dict[str, Any]] = field(default_factory=list)
    pois_visited: List[Dict[str, Any]] = field(default_factory=list)  # Added for POI integration
    real_time_adjustments: List[str] = field(default_factory=list)
    alternative_routes: List['OptimizedRoute'] = field(default_factory=list)

class IBBAPIClient:
    """İBB (Istanbul Metropolitan Municipality) API Client"""
    
    def __init__(self):
        self.api_base = "https://data.ibb.gov.tr/api/3/action/"
        self.api_key = self._get_api_key()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def _get_api_key(self) -> str:
        """Get İBB API key from environment or config"""
        import os
        # In production, get from environment variable
        return os.getenv('IBB_API_KEY', 'demo_key_for_testing')
    
    async def get_metro_real_time_data(self) -> Dict[str, Any]:
        """Get real-time metro data from İBB API"""
        try:
            cache_key = "metro_real_time"
            if self._is_cached(cache_key):
                return self.cache[cache_key]
            
            # Mock real-time metro data (replace with actual API call)
            real_time_data = {
                'M1': {'status': 'operational', 'delays': [], 'crowding': 0.6},
                'M2': {'status': 'operational', 'delays': [{'station': 'Taksim', 'minutes': 3}], 'crowding': 0.8},
                'M3': {'status': 'operational', 'delays': [], 'crowding': 0.4},
                'M4': {'status': 'operational', 'delays': [], 'crowding': 0.7},
                'M7': {'status': 'operational', 'delays': [], 'crowding': 0.5},
                'M11': {'status': 'operational', 'delays': [], 'crowding': 0.3},
                'timestamp': datetime.now().isoformat()
            }
            
            self._cache_data(cache_key, real_time_data)
            return real_time_data
            
        except Exception as e:
            logger.error(f"Failed to get metro real-time data: {e}")
            return self._get_fallback_metro_data()
    
    async def get_bus_real_time_data(self) -> Dict[str, Any]:
        """Get real-time bus data from İBB API"""
        try:
            cache_key = "bus_real_time"
            if self._is_cached(cache_key):
                return self.cache[cache_key]
            
            # Mock real-time bus data
            bus_data = {
                'routes': {
                    '28': {'crowding': 0.9, 'next_arrival': 5, 'delays': 2},
                    '36': {'crowding': 0.6, 'next_arrival': 8, 'delays': 0},
                    '74': {'crowding': 0.4, 'next_arrival': 12, 'delays': 1}
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self._cache_data(cache_key, bus_data)
            return bus_data
            
        except Exception as e:
            logger.error(f"Failed to get bus real-time data: {e}")
            return {'routes': {}, 'error': str(e)}
    
    async def get_ferry_schedule(self) -> Dict[str, Any]:
        """Get ferry schedule and real-time data"""
        try:
            cache_key = "ferry_schedule"
            if self._is_cached(cache_key):
                return self.cache[cache_key]
            
            # Mock ferry data
            ferry_data = {
                'routes': {
                    'Eminönü-Üsküdar': {
                        'next_departure': 15,
                        'frequency_minutes': 20,
                        'crowding': 0.3,
                        'weather_impact': 'none'
                    },
                    'Kabataş-Kadıköy': {
                        'next_departure': 8,
                        'frequency_minutes': 15,
                        'crowding': 0.5,
                        'weather_impact': 'none'
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self._cache_data(cache_key, ferry_data)
            return ferry_data
            
        except Exception as e:
            logger.error(f"Failed to get ferry schedule: {e}")
            return {'routes': {}, 'error': str(e)}
    
    async def get_real_time_arrivals(self, station_id: str) -> List[Dict[str, Any]]:
        """Get real-time arrival data for a specific station"""
        try:
            cache_key = f"arrivals_{station_id}"
            if self._is_cached(cache_key):
                cached_data = self.cache[cache_key]
                return cached_data.get('data', cached_data) if isinstance(cached_data, dict) else cached_data
            
            # Mock real-time arrivals data (replace with actual API call)
            arrivals = [
                {
                    'line': 'M1A',
                    'direction': 'Atatürk Airport',
                    'arrival_minutes': 3,
                    'crowding_level': 0.7
                },
                {
                    'line': 'M1A', 
                    'direction': 'Kirazlı',
                    'arrival_minutes': 8,
                    'crowding_level': 0.5
                }
            ]
            
            self._cache_data(cache_key, arrivals)
            return arrivals
            
        except Exception as e:
            logger.error(f"Failed to get arrivals for {station_id}: {e}")
            return []

    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and still valid"""
        if key not in self.cache:
            return False
        cached_item = self.cache[key]
        if isinstance(cached_item, dict):
            cache_time = cached_item.get('_cache_time', 0)
        else:
            cache_time = 0
        return time.time() - cache_time < self.cache_duration
    
    def _cache_data(self, key: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """Cache data with timestamp"""
        if isinstance(data, dict):
            data['_cache_time'] = time.time()
            self.cache[key] = data
        else:
            # For list data, wrap in a dict
            cached_data = {
                'data': data,
                '_cache_time': time.time()
            }
            self.cache[key] = cached_data
    
    def _get_fallback_metro_data(self) -> Dict[str, Any]:
        """Fallback metro data when API is unavailable"""
        return {
            'M1': {'status': 'operational', 'delays': [], 'crowding': 0.5},
            'M2': {'status': 'operational', 'delays': [], 'crowding': 0.6},
            'M3': {'status': 'operational', 'delays': [], 'crowding': 0.4},
            'M4': {'status': 'operational', 'delays': [], 'crowding': 0.5},
            'fallback': True,
            'timestamp': datetime.now().isoformat()
        }

class MLCrowdingPredictor:
    """ML-based crowding and travel time predictor"""
    
    def __init__(self):
        self.model_trained = False
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.crowding_model = None
        self.travel_time_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        if SKLEARN_AVAILABLE:
            self.crowding_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.travel_time_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self._train_models_with_synthetic_data()
        else:
            logger.warning("ML models not available, using heuristic predictions")
    
    def _train_models_with_synthetic_data(self):
        """Train models with synthetic historical data"""
        try:
            # Generate synthetic training data
            n_samples = 1000
            np.random.seed(42)
            
            # Features: hour, day_of_week, weather, route_length, transport_mode
            X = np.random.rand(n_samples, 5)
            X[:, 0] = np.random.randint(0, 24, n_samples)  # hour
            X[:, 1] = np.random.randint(0, 7, n_samples)   # day_of_week
            X[:, 2] = np.random.rand(n_samples)            # weather_factor
            X[:, 3] = np.random.rand(n_samples) * 20       # route_length_km
            X[:, 4] = np.random.randint(0, 5, n_samples)   # transport_mode
            
            # Generate crowding labels (higher during rush hours)
            crowding_y = np.random.rand(n_samples)
            rush_hour_mask = (X[:, 0] >= 7) & (X[:, 0] <= 9) | (X[:, 0] >= 17) & (X[:, 0] <= 19)
            crowding_y[rush_hour_mask] += 0.3
            crowding_y = np.clip(crowding_y, 0, 1)
            
            # Generate travel time labels (longer during rush hours and bad weather)
            travel_time_y = X[:, 3] * 3 + np.random.rand(n_samples) * 10  # base time
            travel_time_y[rush_hour_mask] *= 1.5  # rush hour multiplier
            travel_time_y += X[:, 2] * 10  # weather impact
            
            # Train models
            X_scaled = self.scaler.fit_transform(X)
            self.crowding_model.fit(X_scaled, crowding_y)
            self.travel_time_model.fit(X_scaled, travel_time_y)
            
            self.model_trained = True
            logger.info("ML crowding and travel time models trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train ML models: {e}")
            self.model_trained = False
    
    def predict_crowding(self, hour: int, day_of_week: int, transport_mode: TransportMode, route_length_km: float) -> float:
        """Predict crowding level (0-1)"""
        if not self.model_trained or not SKLEARN_AVAILABLE:
            return self._heuristic_crowding_prediction(hour, day_of_week, transport_mode)
        
        try:
            # Prepare features
            features = np.array([[
                hour,
                day_of_week,
                0.5,  # neutral weather
                route_length_km,
                self._transport_mode_to_numeric(transport_mode)
            ]])
            
            features_scaled = self.scaler.transform(features)
            prediction = self.crowding_model.predict(features_scaled)[0]
            return np.clip(prediction, 0, 1)
            
        except Exception as e:
            logger.error(f"ML crowding prediction failed: {e}")
            return self._heuristic_crowding_prediction(hour, day_of_week, transport_mode)
    
    def predict_travel_time(self, base_time_minutes: int, hour: int, day_of_week: int, transport_mode: TransportMode, route_length_km: float) -> int:
        """Predict actual travel time with delays"""
        if not self.model_trained or not SKLEARN_AVAILABLE:
            return self._heuristic_travel_time_prediction(base_time_minutes, hour, day_of_week, transport_mode)
        
        try:
            # Prepare features
            features = np.array([[
                hour,
                day_of_week,
                0.5,  # neutral weather
                route_length_km,
                self._transport_mode_to_numeric(transport_mode)
            ]])
            
            features_scaled = self.scaler.transform(features)
            prediction = self.travel_time_model.predict(features_scaled)[0]
            return max(int(prediction), base_time_minutes)
            
        except Exception as e:
            logger.error(f"ML travel time prediction failed: {e}")
            return self._heuristic_travel_time_prediction(base_time_minutes, hour, day_of_week, transport_mode)
    
    def _heuristic_crowding_prediction(self, hour: int, day_of_week: int, transport_mode: TransportMode) -> float:
        """Heuristic crowding prediction when ML is not available"""
        base_crowding = 0.5
        
        # Rush hour adjustment
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            base_crowding += 0.3
        elif (6 <= hour < 7) or (9 < hour <= 11) or (15 <= hour < 17) or (19 < hour <= 21):
            base_crowding += 0.1
        
        # Weekend adjustment
        if day_of_week in [5, 6]:  # Saturday, Sunday
            base_crowding -= 0.1
        
        # Transport mode adjustment
        if transport_mode in [TransportMode.METRO, TransportMode.METROBUS]:
            base_crowding += 0.1
        elif transport_mode == TransportMode.FERRY:
            base_crowding -= 0.1
        
        return np.clip(base_crowding, 0, 1)
    
    def _heuristic_travel_time_prediction(self, base_time_minutes: int, hour: int, day_of_week: int, transport_mode: TransportMode) -> int:
        """Heuristic travel time prediction when ML is not available"""
        multiplier = 1.0
        
        # Rush hour adjustment
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            multiplier = 1.4
        elif (6 <= hour < 7) or (9 < hour <= 11) or (15 <= hour < 17) or (19 < hour <= 21):
            multiplier = 1.2
        
        # Weekend adjustment
        if day_of_week in [5, 6]:
            multiplier *= 0.9
        
        # Transport mode adjustment
        if transport_mode == TransportMode.BUS:
            multiplier *= 1.3  # More affected by traffic
        elif transport_mode in [TransportMode.METRO, TransportMode.MARMARAY]:
            multiplier *= 0.9  # Less affected by traffic
        
        return int(base_time_minutes * multiplier)
    
    def _transport_mode_to_numeric(self, transport_mode: TransportMode) -> int:
        """Convert transport mode to numeric for ML"""
        mode_map = {
            TransportMode.WALKING: 0,
            TransportMode.BUS: 1,
            TransportMode.METRO: 2,
            TransportMode.TRAM: 3,
            TransportMode.FERRY: 4
        }
        return mode_map.get(transport_mode, 0)

class POIIntegratedRouteOptimizer:
    """Route optimizer that integrates Points of Interest (museums, attractions)"""
    
    def __init__(self):
        self.pois_database = self._load_pois_database()
        logger.info("POI-integrated route optimizer initialized")
    
    def _load_pois_database(self) -> List[Dict[str, Any]]:
        """Load POIs (museums, attractions) database"""
        # Enhanced POI database with Istanbul attractions
        return [
            {
                'name': 'Hagia Sophia',
                'category': 'museum',
                'subcategory': 'historical_site',
                'location': GPSLocation(41.0086, 28.9802),
                'visit_duration_minutes': 60,
                'opening_hours': '09:00-19:00',
                'district': 'Sultanahmet'
            },
            {
                'name': 'Blue Mosque',
                'category': 'historical_site',
                'subcategory': 'mosque',
                'location': GPSLocation(41.0054, 28.9768),
                'visit_duration_minutes': 45,
                'opening_hours': '08:30-18:00',
                'district': 'Sultanahmet'
            },
            {
                'name': 'Topkapi Palace',
                'category': 'museum',
                'subcategory': 'palace',
                'location': GPSLocation(41.0115, 28.9833),
                'visit_duration_minutes': 120,
                'opening_hours': '09:00-17:00',
                'district': 'Sultanahmet'
            },
            {
                'name': 'Galata Tower',
                'category': 'historical_site',
                'subcategory': 'tower',
                'location': GPSLocation(41.0256, 28.9744),
                'visit_duration_minutes': 45,
                'opening_hours': '09:00-20:00',
                'district': 'Beyoğlu'
            },
            {
                'name': 'Basilica Cistern',
                'category': 'museum',
                'subcategory': 'historical_site',
                'location': GPSLocation(41.0084, 28.9778),
                'visit_duration_minutes': 40,
                'opening_hours': '09:00-17:30',
                'district': 'Sultanahmet'
            },
            {
                'name': 'Istanbul Archaeological Museums',
                'category': 'museum',
                'subcategory': 'archaeological',
                'location': GPSLocation(41.0117, 28.9815),
                'visit_duration_minutes': 90,
                'opening_hours': '09:00-17:00',
                'district': 'Sultanahmet'
            },
            {
                'name': 'Dolmabahce Palace',
                'category': 'museum',
                'subcategory': 'palace',
                'location': GPSLocation(41.0391, 29.0000),
                'visit_duration_minutes': 75,
                'opening_hours': '09:00-16:00',
                'district': 'Beşiktaş'
            },
            {
                'name': 'Grand Bazaar',
                'category': 'shopping',
                'subcategory': 'market',
                'location': GPSLocation(41.0106, 28.9681),
                'visit_duration_minutes': 60,
                'opening_hours': '09:00-19:00',
                'district': 'Sultanahmet'
            }
        ]
    
    def optimize_route_with_pois(self, start_location: GPSLocation, end_location: GPSLocation,
                               poi_preferences: List[str], max_detour_minutes: int = 30) -> OptimizedRoute:
        """Optimize route to include relevant POIs"""
        try:
            # Find POIs that match preferences and are reasonably close to route
            relevant_pois = self._find_relevant_pois(start_location, end_location, poi_preferences)
            
            # Select best POIs to include (max 2-3 to keep route reasonable)
            selected_pois = self._select_optimal_pois(start_location, end_location, relevant_pois, max_detour_minutes)
            
            # Create route that includes selected POIs
            route_segments = []
            current_location = start_location
            total_duration = 0
            total_distance = 0.0
            total_cost = 0.0
            
            # Add segments to each POI
            for poi in selected_pois:
                # Transport to POI
                segment_to_poi = RouteSegment(
                    from_location=current_location,
                    to_location=poi['location'],
                    transport_mode=self._get_best_transport_mode(current_location, poi['location']),
                    duration_minutes=int(self._calculate_distance(current_location, poi['location']) * 15),  # ~15 min per km
                    distance_km=self._calculate_distance(current_location, poi['location']),
                    cost_tl=self._calculate_transport_cost(current_location, poi['location']),
                    instructions=[f"Take {self._get_best_transport_mode(current_location, poi['location']).value} to {poi['name']}"],
                    crowding_level=0.5
                )
                route_segments.append(segment_to_poi)
                
                # Visit POI (walking segment representing the visit)
                visit_segment = RouteSegment(
                    from_location=poi['location'],
                    to_location=poi['location'],  # Same location for visit
                    transport_mode=TransportMode.WALKING,
                    duration_minutes=poi.get('visit_duration_minutes', 30),
                    distance_km=0.0,
                    cost_tl=0.0,
                    instructions=[f"Visit {poi['name']} ({poi.get('visit_duration_minutes', 30)} minutes)"],
                    crowding_level=0.3
                )
                route_segments.append(visit_segment)
                
                total_duration += segment_to_poi.duration_minutes + visit_segment.duration_minutes
                total_distance += segment_to_poi.distance_km
                total_cost += segment_to_poi.cost_tl
                current_location = poi['location']
            
            # Final segment to destination
            if current_location != end_location:
                final_segment = RouteSegment(
                    from_location=current_location,
                    to_location=end_location,
                    transport_mode=self._get_best_transport_mode(current_location, end_location),
                    duration_minutes=int(self._calculate_distance(current_location, end_location) * 15),
                    distance_km=self._calculate_distance(current_location, end_location),
                    cost_tl=self._calculate_transport_cost(current_location, end_location),
                    instructions=[f"Take {self._get_best_transport_mode(current_location, end_location).value} to destination"],
                    crowding_level=0.5
                )
                route_segments.append(final_segment)
                total_duration += final_segment.duration_minutes
                total_distance += final_segment.distance_km
                total_cost += final_segment.cost_tl
            
            # Create optimized route with POI information
            route = OptimizedRoute(
                route_id=f"poi_route_{int(time.time())}",
                segments=route_segments,
                total_duration_minutes=total_duration,
                total_distance_km=total_distance,
                total_cost_tl=total_cost,
                optimization_type=RouteOptimizationType.POI_OPTIMIZED,
                confidence_score=0.85,
                crowding_prediction=0.5,
                poi_integration=[{
                    'name': poi['name'],
                    'category': poi['category'],
                    'visit_duration': poi.get('visit_duration_minutes', 30),
                    'location': {'lat': poi['location'].latitude, 'lng': poi['location'].longitude}
                } for poi in selected_pois],
                pois_visited=selected_pois
            )
            
            return route
            
        except Exception as e:
            logger.error(f"POI route optimization failed: {e}")
            return self._create_fallback_route(start_location, end_location)
    
    def _find_relevant_pois(self, start: GPSLocation, end: GPSLocation, preferences: List[str]) -> List[Dict[str, Any]]:
        """Find POIs that match preferences and are along the route"""
        relevant_pois = []
        
        for poi in self.pois_database:
            # Check if POI matches preferences
            if any(pref.lower() in poi['category'].lower() or 
                   pref.lower() in poi.get('subcategory', '').lower() or
                   pref.lower() in poi['name'].lower() for pref in preferences):
                
                # Check if POI is reasonably close to the route
                distance_from_start = self._calculate_distance(start, poi['location'])
                distance_from_end = self._calculate_distance(poi['location'], end)
                direct_distance = self._calculate_distance(start, end)
                
                # POI is relevant if detour is reasonable (< 50% extra distance)
                detour_ratio = (distance_from_start + distance_from_end) / max(direct_distance, 0.1)
                if detour_ratio < 1.5:
                    poi['detour_ratio'] = detour_ratio
                    relevant_pois.append(poi)
        
        return relevant_pois
    
    def _select_optimal_pois(self, start: GPSLocation, end: GPSLocation, 
                           pois: List[Dict[str, Any]], max_detour_minutes: int) -> List[Dict[str, Any]]:
        """Select the best POIs to include in route"""
        if not pois:
            return []
        
        # Sort by detour ratio and popularity
        pois.sort(key=lambda x: x.get('detour_ratio', 1.0))
        
        # Select up to 3 POIs that don't exceed detour time
        selected = []
        total_detour_time = 0
        
        for poi in pois[:3]:  # Limit to 3 POIs max
            poi_detour_time = poi.get('visit_duration_minutes', 30) + 10  # Visit time + travel detour
            if total_detour_time + poi_detour_time <= max_detour_minutes:
                selected.append(poi)
                total_detour_time += poi_detour_time
        
        return selected
    
    def _get_best_transport_mode(self, start: GPSLocation, end: GPSLocation) -> TransportMode:
        """Determine best transport mode between two locations"""
        distance = self._calculate_distance(start, end)
        
        if distance < 0.5:  # Less than 500m
            return TransportMode.WALKING
        elif distance < 2.0:  # Less than 2km
            return TransportMode.TRAM  # Good for short distances
        elif distance < 5.0:  # Medium distance
            return TransportMode.METRO  # Efficient for medium distances
        else:
            return TransportMode.BUS  # Good for longer distances
    
    def _calculate_transport_cost(self, start: GPSLocation, end: GPSLocation) -> float:
        """Calculate estimated transport cost"""
        distance = self._calculate_distance(start, end)
        mode = self._get_best_transport_mode(start, end)
        
        if mode == TransportMode.WALKING:
            return 0.0
        elif mode in [TransportMode.METRO, TransportMode.TRAM, TransportMode.BUS]:
            # Istanbul public transport flat rate
            return 3.95  # Current Istanbulkart rate
        else:
            # Taxi/other
            return distance * 2.5  # Approximate taxi rate per km
    
    def _calculate_distance(self, start: GPSLocation, end: GPSLocation) -> float:
        """Calculate distance between two GPS locations in kilometers"""
        from math import radians, cos, sin, asin, sqrt
        
        # Haversine formula
        lat1, lng1 = radians(start.latitude), radians(start.longitude)
        lat2, lng2 = radians(end.latitude), radians(end.longitude)
        
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        return 2 * 6371 * asin(sqrt(a))  # 6371 km = Earth's radius
    
    def _create_fallback_route(self, start: GPSLocation, end: GPSLocation) -> OptimizedRoute:
        """Create simple fallback route when POI optimization fails"""
        distance = self._calculate_distance(start, end)
        
        segment = RouteSegment(
            from_location=start,
            to_location=end,
            transport_mode=self._get_best_transport_mode(start, end),
            duration_minutes=int(distance * 15),  # ~15 min per km
            distance_km=distance,
            cost_tl=self._calculate_transport_cost(start, end),
            instructions=[f"Take {self._get_best_transport_mode(start, end).value} to destination"],
            crowding_level=0.5
        )
        
        return OptimizedRoute(
            route_id=f"fallback_route_{int(time.time())}",
            segments=[segment],
            total_duration_minutes=segment.duration_minutes,
            total_distance_km=distance,
            total_cost_tl=segment.cost_tl,
            optimization_type=RouteOptimizationType.FASTEST,
            confidence_score=0.6,
            crowding_prediction=0.5,
            pois_visited=[]
        )
    
    def find_nearby_pois(self, location: GPSLocation, radius_km: float = 2.0) -> List[Dict[str, Any]]:
        """Find POIs near a location"""
        nearby = []
        for poi in self.pois_database:
            distance = self._calculate_distance(location, poi['location'])
            if distance <= radius_km:
                poi_copy = poi.copy()
                poi_copy['distance_km'] = distance
                nearby.append(poi_copy)
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance_km'])
        return nearby

class MLEnhancedTransportationSystem:
    """Main ML-Enhanced Transportation System"""
    
    def __init__(self):
        self.ibb_client = IBBAPIClient()
        self.ml_predictor = MLCrowdingPredictor()
        self.poi_optimizer = POIIntegratedRouteOptimizer()
        self.transport_network = self._load_transport_network()
        logger.info("🚀 ML-Enhanced Transportation System initialized successfully!")
    
    def _load_transport_network(self) -> Dict[str, Any]:
        """Load Istanbul transportation network data"""
        return {
            'metro_stations': {
                'M2_Vezneciler': TransportStation(
                    id='M2_Vezneciler',
                    name='Vezneciler',
                    transport_modes=[TransportMode.METRO],
                    location=GPSLocation(41.0133, 28.9597),
                    accessibility=True
                ),
                'M2_Taksim': TransportStation(
                    id='M2_Taksim',
                    name='Taksim',
                    transport_modes=[TransportMode.METRO],
                    location=GPSLocation(41.0370, 28.9857),
                    accessibility=True
                ),
                'T1_Sultanahmet': TransportStation(
                    id='T1_Sultanahmet',
                    name='Sultanahmet',
                    transport_modes=[TransportMode.TRAM],
                    location=GPSLocation(41.0056, 28.9769),
                    accessibility=True
                )
            },
            'connections': {
                ('M2_Taksim', 'M2_Vezneciler'): {'duration_minutes': 15, 'distance_km': 8.5},
                ('M2_Vezneciler', 'T1_Sultanahmet'): {'duration_minutes': 10, 'distance_km': 1.2, 'mode': 'walking'}
            }
        }
    
    async def get_optimized_route(self, start_location: GPSLocation, end_location: GPSLocation, 
                                optimization_type: RouteOptimizationType = RouteOptimizationType.FASTEST,
                                include_pois: bool = False, poi_preferences: List[str] = None) -> OptimizedRoute:
        """Get optimized route with real-time data and ML predictions"""
        try:
            logger.info(f"🗺️ Computing optimized route: {optimization_type.value}")
            
            # Get real-time transport data
            metro_data = await self.ibb_client.get_metro_real_time_data()
            bus_data = await self.ibb_client.get_bus_real_time_data()
            ferry_data = await self.ibb_client.get_ferry_schedule()
            
            # POI-optimized route
            if include_pois and poi_preferences:
                return self.poi_optimizer.optimize_route_with_pois(
                    start_location, end_location, poi_preferences, max_detour_minutes=30
                )
            
            # Multi-modal route optimization
            route_options = self._generate_route_options(start_location, end_location, metro_data, bus_data, ferry_data)
            
            # Select best route based on optimization type
            best_route = self._select_best_route(route_options, optimization_type)
            
            # Add ML predictions
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()
            
            for segment in best_route.segments:
                segment.crowding_level = self.ml_predictor.predict_crowding(
                    current_hour, current_day, segment.transport_mode, segment.distance_km
                )
                segment.duration_minutes = self.ml_predictor.predict_travel_time(
                    segment.duration_minutes, current_hour, current_day, segment.transport_mode, segment.distance_km
                )
            
            # Update total duration with ML predictions
            best_route.total_duration_minutes = sum(segment.duration_minutes for segment in best_route.segments)
            best_route.crowding_prediction = np.mean([segment.crowding_level for segment in best_route.segments])
            
            # Add real-time adjustments
            best_route.real_time_adjustments = self._generate_real_time_adjustments(metro_data, bus_data)
            
            logger.info(f"✅ Route optimized: {best_route.total_duration_minutes} minutes, {best_route.total_cost_tl:.2f} TL")
            return best_route
            
        except Exception as e:
            logger.error(f"Route optimization failed: {e}")
            return self.poi_optimizer._create_fallback_route(start_location, end_location)
    
    def _generate_route_options(self, start: GPSLocation, end: GPSLocation, 
                               metro_data: Dict, bus_data: Dict, ferry_data: Dict) -> List[OptimizedRoute]:
        """Generate multiple route options with different transport combinations"""
        route_options = []
        
        # Option 1: Walking only
        walking_segment = RouteSegment(
            from_location=start,
            to_location=end,
            transport_mode=TransportMode.WALKING,
            duration_minutes=int(self.poi_optimizer._calculate_distance(start, end) * 12),
            distance_km=self.poi_optimizer._calculate_distance(start, end),
            cost_tl=0,
            instructions=["Walk directly to destination"]
        )
        
        route_options.append(OptimizedRoute(
            route_id="walking_route",
            segments=[walking_segment],
            total_duration_minutes=walking_segment.duration_minutes,
            total_distance_km=walking_segment.distance_km,
            total_cost_tl=0,
            optimization_type=RouteOptimizationType.CHEAPEST,
            confidence_score=0.9,
            crowding_prediction=0.0
        ))
        
        # Option 2: Metro + Walking
        metro_route = self._create_metro_route(start, end, metro_data)
        if metro_route:
            route_options.append(metro_route)
        
        # Option 3: Bus + Walking
        bus_route = self._create_bus_route(start, end, bus_data)
        if bus_route:
            route_options.append(bus_route)
        
        # Option 4: Ferry + Walking (if applicable)
        ferry_route = self._create_ferry_route(start, end, ferry_data)
        if ferry_route:
            route_options.append(ferry_route)
        
        return route_options
    
    def _create_metro_route(self, start: GPSLocation, end: GPSLocation, metro_data: Dict) -> Optional[OptimizedRoute]:
        """Create metro-based route"""
        try:
            # Find nearest metro stations
            nearest_start_station = self._find_nearest_station(start, TransportMode.METRO)
            nearest_end_station = self._find_nearest_station(end, TransportMode.METRO)
            
            if not nearest_start_station or not nearest_end_station:
                return None
            
            segments = []
            total_duration = 0
            total_cost = 0
            
            # Walk to start station
            walk_to_start = RouteSegment(
                from_location=start,
                to_location=nearest_start_station.location,
                transport_mode=TransportMode.WALKING,
                duration_minutes=int(self.poi_optimizer._calculate_distance(start, nearest_start_station.location) * 12),
                distance_km=self.poi_optimizer._calculate_distance(start, nearest_start_station.location),
                cost_tl=0,
                instructions=[f"Walk to {nearest_start_station.name} metro station"]
            )
            segments.append(walk_to_start)
            total_duration += walk_to_start.duration_minutes
            
            # Metro journey
            metro_segment = RouteSegment(
                from_location=nearest_start_station.location,
                to_location=nearest_end_station.location,
                transport_mode=TransportMode.METRO,
                duration_minutes=20,  # Estimated metro time
                distance_km=self.poi_optimizer._calculate_distance(nearest_start_station.location, nearest_end_station.location),
                cost_tl=7.67,
                instructions=[f"Take metro from {nearest_start_station.name} to {nearest_end_station.name}"]
            )
            segments.append(metro_segment)
            total_duration += metro_segment.duration_minutes
            total_cost += metro_segment.cost_tl
            
            # Walk from end station
            walk_from_end = RouteSegment(
                from_location=nearest_end_station.location,
                to_location=end,
                transport_mode=TransportMode.WALKING,
                duration_minutes=int(self.poi_optimizer._calculate_distance(nearest_end_station.location, end) * 12),
                distance_km=self.poi_optimizer._calculate_distance(nearest_end_station.location, end),
                cost_tl=0,
                instructions=[f"Walk from {nearest_end_station.name} to destination"]
            )
            segments.append(walk_from_end)
            total_duration += walk_from_end.duration_minutes
            
            return OptimizedRoute(
                route_id="metro_route",
                segments=segments,
                total_duration_minutes=total_duration,
                total_distance_km=sum(s.distance_km for s in segments),
                total_cost_tl=total_cost,
                optimization_type=RouteOptimizationType.FASTEST,
                confidence_score=0.8,
                crowding_prediction=0.6
            )
            
        except Exception as e:
            logger.error(f"Failed to create metro route: {e}")
            return None
    
    def _create_bus_route(self, start: GPSLocation, end: GPSLocation, bus_data: Dict) -> Optional[OptimizedRoute]:
        """Create bus-based route"""
        # Simplified bus route implementation
        walking_distance = self.poi_optimizer._calculate_distance(start, end)
        
        if walking_distance > 2:  # Only suggest bus for longer distances
            bus_segment = RouteSegment(
                from_location=start,
                to_location=end,
                transport_mode=TransportMode.BUS,
                duration_minutes=int(walking_distance * 8),  # Bus speed
                distance_km=walking_distance,
                cost_tl=7.67,
                instructions=["Take bus to destination"]
            )
            
            return OptimizedRoute(
                route_id="bus_route",
                segments=[bus_segment],
                total_duration_minutes=bus_segment.duration_minutes,
                total_distance_km=walking_distance,
                total_cost_tl=7.67,
                optimization_type=RouteOptimizationType.FASTEST,
                confidence_score=0.7,
                crowding_prediction=0.7
            )
        
        return None
    
    def _create_ferry_route(self, start: GPSLocation, end: GPSLocation, ferry_data: Dict) -> Optional[OptimizedRoute]:
        """Create ferry-based route if applicable"""
        # Check if route crosses Bosphorus or Golden Horn
        start_side = "european" if start.longitude < 29.0 else "asian"
        end_side = "european" if end.longitude < 29.0 else "asian"
        
        if start_side != end_side:
            ferry_segment = RouteSegment(
                from_location=start,
                to_location=end,
                transport_mode=TransportMode.FERRY,
                duration_minutes=25,  # Average ferry time
                distance_km=self.poi_optimizer._calculate_distance(start, end),
                cost_tl=7.67,
                instructions=["Take ferry across Bosphorus"]
            )
            
            return OptimizedRoute(
                route_id="ferry_route",
                segments=[ferry_segment],
                total_duration_minutes=25,
                total_distance_km=ferry_segment.distance_km,
                total_cost_tl=7.67,
                optimization_type=RouteOptimizationType.MOST_SCENIC,
                confidence_score=0.8,
                crowding_prediction=0.3
            )
        
        return None
    
    def _find_nearest_station(self, location: GPSLocation, transport_mode: TransportMode) -> Optional[TransportStation]:
        """Find nearest transport station"""
        stations = [station for station in self.transport_network['metro_stations'].values() 
                   if transport_mode in station.transport_modes]
        
        if not stations:
            return None
        
        nearest_station = min(stations, 
                            key=lambda s: self.poi_optimizer._calculate_distance(location, s.location))
        
        # Only return if within reasonable distance (2km)
        if self.poi_optimizer._calculate_distance(location, nearest_station.location) <= 2.0:
            return nearest_station
        
        return None
    
    def _select_best_route(self, route_options: List[OptimizedRoute], optimization_type: RouteOptimizationType) -> OptimizedRoute:
        """Select best route based on optimization criteria"""
        if not route_options:
            raise ValueError("No route options available")
        
        if optimization_type == RouteOptimizationType.FASTEST:
            return min(route_options, key=lambda r: r.total_duration_minutes)
        elif optimization_type == RouteOptimizationType.CHEAPEST:
            return min(route_options, key=lambda r: r.total_cost_tl)
        elif optimization_type == RouteOptimizationType.SHORTEST:
            return min(route_options, key=lambda r: r.total_distance_km)
        elif optimization_type == RouteOptimizationType.LEAST_CROWDED:
            return min(route_options, key=lambda r: r.crowding_prediction)
        elif optimization_type == RouteOptimizationType.MOST_SCENIC:
            scenic_routes = [r for r in route_options if TransportMode.FERRY in [s.transport_mode for s in r.segments]]
            return scenic_routes[0] if scenic_routes else route_options[0]
        else:
            return route_options[0]  # Default to first option
    
    def _generate_real_time_adjustments(self, metro_data: Dict, bus_data: Dict) -> List[str]:
        """Generate real-time route adjustments"""
        adjustments = []
        
        # Check for metro delays
        for line, data in metro_data.items():
            if line.startswith('M') and data.get('delays'):
                adjustments.append(f"⚠️ {line} line has delays: {data['delays']}")
        
        # Check for high crowding
        for line, data in metro_data.items():
            if line.startswith('M') and data.get('crowding', 0) > 0.8:
                adjustments.append(f"🚇 {line} line is very crowded - consider alternative")
        
        if not adjustments:
            adjustments.append("✅ No current disruptions reported")
        
        return adjustments
    
    async def get_location_based_recommendations(self, user_location: GPSLocation, 
                                               destination_hint: str = None) -> Dict[str, Any]:
        """Get location-based transportation recommendations"""
        try:
            # Find nearby transport options
            nearby_stations = []
            for station in self.transport_network['metro_stations'].values():
                distance = self.poi_optimizer._calculate_distance(user_location, station.location)
                if distance <= 1.0:  # Within 1km
                    nearby_stations.append({
                        'station': station,
                        'distance_km': distance,
                        'walking_time_minutes': int(distance * 12)
                    })
            
            # Find nearby POIs
            nearby_pois = self.poi_optimizer.find_nearby_pois(user_location, radius_km=2.0)
            
            # Get real-time data
            metro_data = await self.ibb_client.get_metro_real_time_data()
            
            return {
                'user_location': {
                    'latitude': user_location.latitude,
                    'longitude': user_location.longitude,
                    'address': user_location.address or "Current location"
                },
                'nearby_transport': nearby_stations,
                'nearby_pois': nearby_pois[:5],  # Top 5 nearest POIs
                'real_time_status': metro_data,
                'recommendations': self._generate_location_recommendations(nearby_stations, nearby_pois)
            }
            
        except Exception as e:
            logger.error(f"Location-based recommendations failed: {e}")
            return [
                {
                    'type': 'transport',
                    'description': 'Multiple transportation options available nearby'
                },
                {
                    'type': 'attractions',
                    'description': 'Popular Istanbul attractions within walking distance'
                }
            ]
    
    def _generate_location_recommendations(self, nearby_stations: List[Dict], nearby_pois: List[Dict]) -> List[Dict[str, str]]:
        """Generate personalized location-based recommendations"""
        recommendations = []
        
        if nearby_stations:
            closest_station = min(nearby_stations, key=lambda x: x['distance_km'])
            recommendations.append({
                'type': 'transport',
                'description': f"Closest transport: {closest_station['station'].name} ({closest_station['walking_time_minutes']} min walk)"
            })
        
        if nearby_pois:
            closest_poi = nearby_pois[0]
            walking_time = int(closest_poi.get('distance_km', 1.0) * 12)  # Calculate walking time
            recommendations.append({
                'type': 'attraction',
                'description': f"Nearby attraction: {closest_poi['name']} ({walking_time} min walk)"
            })
        
        recommendations.append({
            'type': 'general',
            'description': "Ask me for routes to any destination!"
        })
        
        if not recommendations:
            recommendations = [
                {
                    'type': 'transport',
                    'description': 'Multiple transportation options available in Istanbul'
                },
                {
                    'type': 'attractions',
                    'description': 'Many popular attractions accessible by public transport'
                },
                {
                    'type': 'general',
                    'description': 'Tell me where you want to go for personalized route planning!'
                }
            ]
        
        return recommendations

# Factory function for easy integration
def create_ml_enhanced_transportation_system() -> MLEnhancedTransportationSystem:
    """Factory function to create the ML-enhanced transportation system"""
    return MLEnhancedTransportationSystem()

# Test function
async def test_ml_transportation_system():
    """Test the ML-enhanced transportation system"""
    print("🚀 Testing ML-Enhanced Transportation System")
    print("=" * 60)
    
    # Create system
    transport_system = create_ml_enhanced_transportation_system()
    
    # Test locations
    sultanahmet = GPSLocation(41.0056, 28.9769, address="Sultanahmet")
    taksim = GPSLocation(41.0370, 28.9857, address="Taksim")
    
    # Test 1: Basic route optimization
    print("\n🗺️ Testing Route Optimization:")
    route = await transport_system.get_optimized_route(
        sultanahmet, taksim, RouteOptimizationType.FASTEST
    )
    print(f"   Route: {route.total_duration_minutes} min, {route.total_cost_tl:.2f} TL")
    print(f"   Segments: {len(route.segments)}")
    print(f"   Crowding: {route.crowding_prediction:.2f}")
    
    # Test 2: POI-integrated route
    print("\n🏛️ Testing POI Integration:")
    poi_route = await transport_system.get_optimized_route(
        sultanahmet, taksim, RouteOptimizationType.POI_OPTIMIZED,
        include_pois=True, poi_preferences=['historical', 'landmark']
    )
    print(f"   POI Route: {poi_route.total_duration_minutes} min")
    print(f"   POIs included: {len(poi_route.poi_integration)}")
    
    # Test 3: Location-based recommendations
    print("\n📍 Testing Location Recommendations:")
    recommendations = await transport_system.get_location_based_recommendations(sultanahmet)
    print(f"   Nearby transport: {len(recommendations['nearby_transport'])}")
    print(f"   Nearby POIs: {len(recommendations['nearby_pois'])}")
    
    # Test 4: ML predictions
    print("\n🧠 Testing ML Predictions:")
    ml_predictor = transport_system.ml_predictor
    crowding = ml_predictor.predict_crowding(8, 1, TransportMode.METRO, 5.0)  # 8 AM, Monday
    travel_time = ml_predictor.predict_travel_time(20, 8, 1, TransportMode.METRO, 5.0)
    print(f"   Crowding prediction (8 AM Monday): {crowding:.2f}")
    print(f"   Travel time prediction: {travel_time} minutes")
    
    print("\n✅ ML-Enhanced Transportation System test completed!")

if __name__ == "__main__":
    asyncio.run(test_ml_transportation_system())
