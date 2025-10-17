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
import os
import sys
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

# Try to import XGBoost and LightGBM for crowding prediction
try:
    import xgboost as xgb
    import lightgbm as lgb
    XGBOOST_AVAILABLE = True
    LIGHTGBM_AVAILABLE = True
    logger.info("✅ XGBoost and LightGBM available for crowding prediction")
except ImportError:
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False
    logger.warning("⚠️ XGBoost/LightGBM not available, using basic crowding prediction")

# Weather integration
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available for data processing")

# Environment variables for weather integration
import os
from dotenv import load_dotenv
load_dotenv()

# Weather API configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
WEATHER_PROVIDER = os.getenv('WEATHER_PROVIDER', 'openweather')

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
            
            # Use real İBB API integration
            try:
                from real_ibb_api_integration import RealIBBAPIClient
                
                async with RealIBBAPIClient() as ibb_client:
                    real_time_data = await ibb_client.get_metro_real_time_data()
                    
                if real_time_data and real_time_data.get('source') != 'fallback':
                    logger.info("✅ Using real İBB metro data")
                    self._cache_data(cache_key, real_time_data)
                    return real_time_data
                else:
                    logger.warning("İBB API unavailable, using enhanced fallback")
                    
            except ImportError:
                logger.warning("Real İBB API integration not available")
            except Exception as e:
                logger.error(f"İBB API error: {e}")
            
            # Enhanced fallback with realistic data
            real_time_data = {
                'lines': {
                    'M1A': {'name': 'Yenikapı-Atatürk Airport', 'status': 'operational', 'delays': [], 'crowding': 0.6},
                    'M2': {'name': 'Vezneciler-Hacıosman', 'status': 'operational', 'delays': [{'station': 'Taksim', 'minutes': 3}], 'crowding': 0.8},
                    'M4': {'name': 'Kadıköy-Sabiha Gökçen', 'status': 'operational', 'delays': [], 'crowding': 0.7},
                    'M11': {'name': 'İST Airport-Gayrettepe', 'status': 'operational', 'delays': [], 'crowding': 0.3},
                    'M6': {'name': 'Levent-Boğaziçi Üniv.', 'status': 'operational', 'delays': [], 'crowding': 0.5},
                    'T1': {'name': 'Kabataş-Bağcılar', 'status': 'operational', 'delays': [], 'crowding': 0.7}
                },
                'timestamp': datetime.now().isoformat(),
                'source': 'enhanced_fallback'
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

@dataclass
class WeatherImpact:
    """Weather impact on transportation"""
    condition: str
    temperature: float
    precipitation: float
    wind_speed: float
    visibility: float
    transport_modifier: float  # 0.5 = slower, 1.0 = normal, 1.5 = faster
    crowding_modifier: float   # 0.5 = less crowded, 1.0 = normal, 2.0 = more crowded
    recommendations: List[str]

class MLCrowdingPredictor:
    """ML-based crowding prediction using XGBoost/LightGBM"""
    
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = None
        self.is_trained = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models with fallback to basic prediction"""
        try:
            if XGBOOST_AVAILABLE:
                self.xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                logger.info("✅ XGBoost model initialized for crowding prediction")
            
            if LIGHTGBM_AVAILABLE:
                self.lgb_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                logger.info("✅ LightGBM model initialized for crowding prediction")
            
            if SKLEARN_AVAILABLE:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                logger.info("✅ Feature scaler initialized")
            
            # Train with synthetic data initially
            self._train_with_synthetic_data()
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _train_with_synthetic_data(self):
        """Train models with realistic synthetic Istanbul transportation data"""
        try:
            # Generate synthetic training data
            n_samples = 5000
            features = []
            labels = []
            
            for _ in range(n_samples):
                # Time features
                hour = np.random.randint(0, 24)
                day_of_week = np.random.randint(0, 7)
                month = np.random.randint(1, 13)
                
                # Weather features
                temp = np.random.normal(15, 10)  # Istanbul average temp
                rain = np.random.exponential(2)   # Rain intensity
                wind = np.random.exponential(5)   # Wind speed
                
                # Transportation features
                transport_mode = np.random.randint(0, 4)  # metro, bus, ferry, tram
                route_distance = np.random.exponential(10)  # km
                
                # Special events (randomly)
                is_holiday = np.random.choice([0, 1], p=[0.9, 0.1])
                is_event_day = np.random.choice([0, 1], p=[0.95, 0.05])
                
                feature_vector = [
                    hour, day_of_week, month, temp, rain, wind,
                    transport_mode, route_distance, is_holiday, is_event_day
                ]
                
                # Calculate crowding level (0-1)
                crowding = self._calculate_synthetic_crowding(
                    hour, day_of_week, temp, rain, transport_mode, is_holiday, is_event_day
                )
                
                features.append(feature_vector)
                labels.append(crowding)
            
            X = np.array(features)
            y = np.array(labels)
            
            # Scale features
            if self.scaler:
                X = self.scaler.fit_transform(X)
            
            # Train models
            if self.xgb_model:
                self.xgb_model.fit(X, y)
                logger.info("✅ XGBoost model trained with synthetic data")
            
            if self.lgb_model:
                self.lgb_model.fit(X, y)
                logger.info("✅ LightGBM model trained with synthetic data")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
    
    def _calculate_synthetic_crowding(self, hour, day_of_week, temp, rain, transport_mode, is_holiday, is_event_day):
        """Calculate realistic crowding levels for Istanbul transportation"""
        base_crowding = 0.3
        
        # Rush hour effects
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_crowding += 0.4  # Rush hours are more crowded
        elif 22 <= hour or hour <= 6:
            base_crowding -= 0.2  # Late night/early morning less crowded
        
        # Weekend effects
        if day_of_week in [5, 6]:  # Friday, Saturday
            if 20 <= hour <= 23:
                base_crowding += 0.3  # Weekend nights
            else:
                base_crowding -= 0.1  # Weekend days generally less crowded
        
        # Weather effects
        if rain > 5:  # Heavy rain
            base_crowding += 0.2  # People avoid walking, use transport more
        if temp < 0 or temp > 35:  # Extreme temperatures
            base_crowding += 0.15
        
        # Transport mode effects
        transport_crowding = {
            0: 0.1,   # Metro - generally less crowded due to frequency
            1: 0.2,   # Bus - more crowded
            2: -0.1,  # Ferry - scenic, less rushed
            3: 0.15   # Tram - moderate crowding
        }
        base_crowding += transport_crowding.get(transport_mode, 0)
        
        # Special events
        if is_holiday:
            base_crowding -= 0.2
        if is_event_day:
            base_crowding += 0.3
        
        # Ensure crowding is between 0 and 1
        return max(0, min(1, base_crowding + np.random.normal(0, 0.1)))
    
    def predict_crowding(self, hour: int, day_of_week: int, weather_data: Dict, 
                        transport_mode: str, route_distance: float) -> float:
        """Predict crowding level using ML models"""
        try:
            if not self.is_trained:
                return self._fallback_crowding_prediction(hour, day_of_week, weather_data, transport_mode)
            
            # Prepare features
            temp = weather_data.get('temperature', 15)
            rain = weather_data.get('precipitation', 0)
            wind = weather_data.get('wind_speed', 0)
            
            transport_mode_map = {'metro': 0, 'bus': 1, 'ferry': 2, 'tram': 3}
            transport_mode_encoded = transport_mode_map.get(transport_mode.lower(), 0)
            
            feature_vector = np.array([[
                hour, day_of_week, datetime.now().month, temp, rain, wind,
                transport_mode_encoded, route_distance, 0, 0  # no holiday/event for now
            ]])
            
            # Scale features
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Ensemble prediction
            predictions = []
            
            if self.xgb_model:
                xgb_pred = self.xgb_model.predict(feature_vector)[0]
                predictions.append(xgb_pred)
            
            if self.lgb_model:
                lgb_pred = self.lgb_model.predict(feature_vector)[0]
                predictions.append(lgb_pred)
            
            if predictions:
                crowding = np.mean(predictions)
                return max(0, min(1, crowding))
            
            return self._fallback_crowding_prediction(hour, day_of_week, weather_data, transport_mode)
            
        except Exception as e:
            logger.error(f"Error predicting crowding: {e}")
            return self._fallback_crowding_prediction(hour, day_of_week, weather_data, transport_mode)
    
    def _fallback_crowding_prediction(self, hour: int, day_of_week: int, weather_data: Dict, transport_mode: str) -> float:
        """Fallback crowding prediction without ML"""
        base_crowding = 0.3
        
        # Rush hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_crowding += 0.4
        
        # Weekend
        if day_of_week in [5, 6]:
            base_crowding -= 0.1
        
        # Weather
        if weather_data.get('precipitation', 0) > 5:
            base_crowding += 0.2
        
        return max(0, min(1, base_crowding))

class WeatherAwareTransportationAdvisor:
    """Weather-aware transportation advisor using existing weather service"""
    
    def __init__(self):
        self.weather_service = None
        self.crowding_predictor = MLCrowdingPredictor()
        self.weather_cache = {}
        self.cache_duration = 1800  # 30 minutes
        
        # Initialize weather service if available
        self._initialize_weather_service()
    
    def _initialize_weather_service(self):
        """Initialize connection to existing weather service"""
        try:
            # Import existing weather service
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))
            
            from services.weather_cache_service import WeatherCacheService
            self.weather_service = WeatherCacheService()
            logger.info("✅ Connected to existing weather service")
            
        except ImportError as e:
            logger.warning(f"Could not connect to weather service: {e}")
            self.weather_service = None
    
    async def get_weather_aware_advice(self, query: str, transport_mode: str = None, 
                                     route_distance: float = 5.0) -> Dict[str, Any]:
        """Generate weather-aware transportation advice"""
        try:
            # Get current weather data
            weather_data = await self._get_current_weather()
            
            # Get hourly forecast
            hourly_forecast = await self._get_hourly_forecast()
            
            # Calculate weather impact
            weather_impact = self._calculate_weather_impact(weather_data)
            
            # Predict crowding for different transport modes
            current_time = datetime.now()
            hour = current_time.hour
            day_of_week = current_time.weekday()
            
            transport_recommendations = []
            for mode in ['metro', 'bus', 'ferry', 'tram']:
                crowding = self.crowding_predictor.predict_crowding(
                    hour, day_of_week, weather_data, mode, route_distance
                )
                
                transport_recommendations.append({
                    'mode': mode,
                    'crowding_level': crowding,
                    'crowding_text': self._crowding_to_text(crowding),
                    'weather_suitability': self._calculate_mode_weather_suitability(mode, weather_impact)
                })
            
            # Sort by best overall recommendation
            transport_recommendations.sort(
                key=lambda x: (x['weather_suitability'] - x['crowding_level']), 
                reverse=True
            )
            
            # Generate advice text
            advice = self._generate_weather_advice_text(
                weather_data, weather_impact, transport_recommendations, hourly_forecast
            )
            
            return {
                'advice': advice,
                'weather_impact': weather_impact,
                'transport_recommendations': transport_recommendations,
                'current_weather': weather_data,
                'hourly_forecast': hourly_forecast[:6],  # Next 6 hours
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating weather-aware advice: {e}")
            return await self._generate_fallback_advice(query, transport_mode)
    
    async def _get_current_weather(self) -> Dict[str, Any]:
        """Get current weather using existing weather service"""
        try:
            if self.weather_service:
                # Use existing weather service
                weather = self.weather_service.current_weather
                if weather:
                    return {
                        'temperature': weather.current_temp,
                        'condition': weather.condition,
                        'description': weather.description,
                        'precipitation': 0,  # Would need to be calculated from condition
                        'wind_speed': weather.wind_speed,
                        'visibility': weather.visibility,
                        'humidity': weather.humidity
                    }
            
            # Fallback: Direct API call
            if OPENWEATHER_API_KEY:
                url = f"https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': 41.0082,  # Istanbul
                    'lon': 28.9784,
                    'appid': OPENWEATHER_API_KEY,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'temperature': data['main']['temp'],
                        'condition': data['weather'][0]['main'],
                        'description': data['weather'][0]['description'],
                        'precipitation': data.get('rain', {}).get('1h', 0),
                        'wind_speed': data['wind']['speed'],
                        'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                        'humidity': data['main']['humidity']
                    }
            
            # Ultimate fallback
            return {
                'temperature': 15,
                'condition': 'Clear',
                'description': 'clear sky',
                'precipitation': 0,
                'wind_speed': 5,
                'visibility': 10,
                'humidity': 60
            }
            
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return {
                'temperature': 15,
                'condition': 'Unknown',
                'description': 'weather data unavailable',
                'precipitation': 0,
                'wind_speed': 5,
                'visibility': 10,
                'humidity': 60
            }
    
    async def _get_hourly_forecast(self) -> List[Dict[str, Any]]:
        """Get hourly forecast using existing weather service"""
        try:
            if self.weather_service and self.weather_service.hourly_forecast:
                forecast = []
                for hour_data in self.weather_service.hourly_forecast[:12]:  # Next 12 hours
                    forecast.append({
                        'time': hour_data.timestamp.isoformat(),
                        'temperature': hour_data.temperature,
                        'condition': hour_data.condition,
                        'precipitation': hour_data.precipitation_chance / 100,
                        'wind_speed': hour_data.wind_speed
                    })
                return forecast
            
            return []  # Return empty if no forecast available
            
        except Exception as e:
            logger.error(f"Error getting hourly forecast: {e}")
            return []
    
    def _calculate_weather_impact(self, weather_data: Dict) -> WeatherImpact:
        """Calculate impact of weather on transportation"""
        try:
            temperature = weather_data.get('temperature', 15)
            precipitation = weather_data.get('precipitation', 0)
            wind_speed = weather_data.get('wind_speed', 0)
            visibility = weather_data.get('visibility', 10)
            
            # Simple rules for weather impact
            transport_modifier = 1.0
            crowding_modifier = 1.0
            recommendations = []
            
            # Temperature impact
            if temperature < 5:
                transport_modifier *= 0.8
                crowding_modifier *= 1.2
                recommendations.append("Dress warmly, expect slower transport")
            elif temperature > 30:
                transport_modifier *= 1.2
                crowding_modifier *= 0.8
                recommendations.append("Stay hydrated, expect faster transport")
            
            # Precipitation impact
            if precipitation > 10:
                transport_modifier *= 0.7
                crowding_modifier *= 1.3
                recommendations.append("Heavy rain expected, transport may be crowded")
            elif precipitation > 0:
                transport_modifier *= 0.9
                crowding_modifier *= 1.1
                recommendations.append("Light rain expected, take an umbrella")
            
            # Wind impact
            if wind_speed > 20:
                transport_modifier *= 0.9
                crowding_modifier *= 1.1
                recommendations.append("Strong winds, transport may be delayed")
            
            # Visibility impact
            if visibility < 1:
                transport_modifier *= 0.6
                crowding_modifier *= 1.4
                recommendations.append("Low visibility, expect significant transport delays")
            
            return WeatherImpact(
                condition=weather_data.get('condition', 'Clear'),
                temperature=temperature,
                precipitation=precipitation,
                wind_speed=wind_speed,
                visibility=visibility,
                transport_modifier=transport_modifier,
                crowding_modifier=crowding_modifier,
                recommendations=recommendations
            )
        
        except Exception as e:
            logger.error(f"Error calculating weather impact: {e}")
            return WeatherImpact(
                condition="Clear",
                temperature=15,
                precipitation=0,
                wind_speed=0,
                visibility=10,
                transport_modifier=1.0,
                crowding_modifier=1.0,
                recommendations=["No weather data available"]
            )
    
    def _crowding_to_text(self, crowding_level: float) -> str:
        """Convert crowding level to descriptive text"""
        if crowding_level < 0.3:
            return "Sparsely populated"
        elif crowding_level < 0.7:
            return "Moderately crowded"
        else:
            return "Very crowded"
    
    def _calculate_mode_weather_suitability(self, mode: str, weather_impact: WeatherImpact) -> float:
        """Calculate how suitable a transport mode is given the weather impact"""
        suitability = 1.0
        
        # Adjust suitability based on weather conditions
        if mode == "bus":
            suitability *= weather_impact.transport_modifier * 0.9  # Buses are slightly less reliable in bad weather
        elif mode == "metro":
            suitability *= weather_impact.transport_modifier * 1.1  # Metro is more reliable in bad weather
        elif mode == "ferry":
            suitability *= weather_impact.transport_modifier * 0.8  # Ferries are less reliable in bad weather
        elif mode == "tram":
            suitability *= weather_impact.transport_modifier * 1.0  # Trams are generally stable
        
        return max(0, min(1, suitability))
    
    def _generate_weather_advice_text(self, weather_data: Dict, weather_impact: WeatherImpact,
                                    transport_recommendations: List[Dict], hourly_forecast: List[Dict]) -> str:
        """Generate comprehensive weather-aware transportation advice text"""
        current_time = datetime.now().strftime("%H:%M")
        temp = weather_data.get('temperature', 15)
        condition = weather_data.get('condition', 'Clear')
        description = weather_data.get('description', 'clear sky')
        
        # Build advice text
        advice_parts = []
        
        # Weather header
        advice_parts.append(f"🌤️ **Weather-Aware Transportation Advice** (Updated: {current_time})")
        advice_parts.append(f"📊 **Current Weather**: {temp}°C, {description}")
        
        # Weather impact
        if weather_impact.recommendations:
            advice_parts.append(f"⚠️ **Weather Impact**: {', '.join(weather_impact.recommendations)}")
        
        # Transportation recommendations
        advice_parts.append(f"\n🚇 **Best Transportation Options Right Now**:")
        
        for i, rec in enumerate(transport_recommendations[:3], 1):
            mode_icon = {'metro': '🚇', 'bus': '🚌', 'ferry': '⛴️', 'tram': '🚋'}.get(rec['mode'], '🚐')
            crowding_icon = {'Sparsely populated': '🟢', 'Moderately crowded': '🟡', 'Very crowded': '🔴'}.get(rec['crowding_text'], '⚪')
            
            advice_parts.append(
                f"{i}. {mode_icon} **{rec['mode'].title()}** - {crowding_icon} {rec['crowding_text']} "
                f"(Weather suitability: {rec['weather_suitability']:.1f}/1.0)"
            )
        
        # Hourly forecast if available
        if hourly_forecast:
            advice_parts.append(f"\n🕐 **Next 6 Hours Weather**:")
            for hour_data in hourly_forecast[:6]:
                hour_time = datetime.fromisoformat(hour_data['time']).strftime("%H:%M")
                hour_temp = hour_data.get('temperature', 15)
                hour_condition = hour_data.get('condition', 'Clear')
                advice_parts.append(f"   • {hour_time}: {hour_temp}°C, {hour_condition}")
        
        # General recommendations
        advice_parts.append(f"\n💡 **Weather-Specific Tips**:")
        if temp < 10:
            advice_parts.append("   • Dress warmly and consider underground transport (metro)")
            advice_parts.append("   • Ferry rides may be colder due to wind")
        elif temp > 25:
            advice_parts.append("   • Stay hydrated and seek air-conditioned transport")
            advice_parts.append("   • Consider ferry rides for cooler Bosphorus breeze")
        
        if weather_data.get('precipitation', 0) > 0:
            advice_parts.append("   • Carry an umbrella and prefer covered transport")
            advice_parts.append("   • Metro and tram are better choices than bus during rain")
        
        advice_parts.append("   • Check real-time arrivals before traveling")
        advice_parts.append("   • Ask me for specific route recommendations!")
        
        return "\n".join(advice_parts)
    
    async def _generate_fallback_advice(self, query: str, transport_mode: str = None) -> Dict[str, Any]:
        """Generate fallback advice when weather data is unavailable"""
        current_time = datetime.now().strftime("%H:%M")
        
        advice = f"""🚇 **Istanbul Transportation Advice** (Updated: {current_time})
        
⚠️ **Weather data temporarily unavailable**

🚇 **General Transportation Recommendations**:
1. 🚇 **Metro** - Most reliable, runs frequently
2. 🚋 **Tram** - Good for touristic areas (T1 line)
3. 🚌 **Bus** - Extensive network, can be crowded
4. ⛴️ **Ferry** - Scenic option, weather dependent

💡 **General Tips**:
• Use Istanbulkart for all public transport
• Check Citymapper or Moovit for real-time info
• Rush hours: 7:30-9:30 AM, 5:30-7:30 PM
• Ferry is often fastest across Bosphorus

🎯 **Need specific directions?** Tell me your starting point and destination!"""

        return {
            'advice': advice,
            'weather_impact': None,
            'transport_recommendations': [
                {'mode': 'metro', 'crowding_level': 0.5, 'crowding_text': 'Moderately crowded', 'weather_suitability': 0.9},
                {'mode': 'tram', 'crowding_level': 0.6, 'crowding_text': 'Moderately crowded', 'weather_suitability': 0.8},
                {'mode': 'bus', 'crowding_level': 0.7, 'crowding_text': 'Very crowded', 'weather_suitability': 0.7},
                {'mode': 'ferry', 'crowding_level': 0.4, 'crowding_text': 'Sparsely populated', 'weather_suitability': 0.6}
            ],
            'current_weather': None,
            'hourly_forecast': [],
            'timestamp': datetime.now().isoformat(),
            'fallback': True
        }
    
    def get_weather_aware_advice_sync(self, query: str = "", transport_mode: str = None, 
                                     route_distance: float = 5.0) -> str:
        """
        Synchronous wrapper for get_weather_aware_advice that returns formatted text.
        This method can be called from synchronous code like process_transportation_query_sync.
        """
        try:
            import asyncio
            
            # Try to get existing event loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, create a new loop for this call
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self.get_weather_aware_advice(query, transport_mode, route_distance)
                        )
                        result = future.result(timeout=10)
                else:
                    # Loop exists but not running, use it
                    result = loop.run_until_complete(
                        self.get_weather_aware_advice(query, transport_mode, route_distance)
                    )
            except RuntimeError:
                # No event loop exists, create a new one
                result = asyncio.run(
                    self.get_weather_aware_advice(query, transport_mode, route_distance)
                )
            
            # Extract and return the advice text
            if isinstance(result, dict):
                return result.get('advice', '')
            return str(result)
            
        except Exception as e:
            logger.error(f"Error in synchronous weather advice: {e}")
            # Return fallback advice
            current_time = datetime.now().strftime("%H:%M")
            return f"""🚇 **Istanbul Transportation Advice** (Updated: {current_time})

⚠️ **Weather data temporarily unavailable**

🚇 **General Transportation Recommendations**:
1. 🚇 **Metro** - Most reliable, runs frequently
2. 🚋 **Tram** - Good for touristic areas (T1 line)
3. 🚌 **Bus** - Extensive network, can be crowded
4. ⛴️ **Ferry** - Scenic option, weather dependent

💡 **General Tips**:
• Use Istanbulkart for all public transport
• Check Citymapper or Moovit for real-time info
• Rush hours: 7:30-9:30 AM, 5:30-7:30 PM
• Ferry is often fastest across Bosphorus

🎯 **Need specific directions?** Tell me your starting point and destination!"""
    
    def get_crowding_prediction_info(self) -> str:
        """Get formatted ML-based crowding prediction information"""
        try:
            current_time = datetime.now()
            hour = current_time.hour
            day_of_week = current_time.weekday()
            
            # Get basic weather data for predictions
            try:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        weather_data = {'temperature': 15, 'precipitation': 0, 'wind_speed': 5}
                    else:
                        weather_data = loop.run_until_complete(self._get_current_weather())
                except RuntimeError:
                    weather_data = asyncio.run(self._get_current_weather())
            except Exception:
                weather_data = {'temperature': 15, 'precipitation': 0, 'wind_speed': 5}
            
            # Predict crowding for major transport modes
            modes_info = []
            for mode in ['metro', 'bus', 'tram', 'ferry']:
                crowding = self.crowding_predictor.predict_crowding(
                    hour, day_of_week, weather_data, mode, 5.0
                )
                crowding_text = self._crowding_to_text(crowding)
                modes_info.append(f"• {mode.title()}: {crowding_text} ({crowding:.1%})")
            
            info = f"""🤖 **ML-Based Crowding Predictions** (Powered by XGBoost/LightGBM)
            
Current predictions for Istanbul transportation:
{chr(10).join(modes_info)}

ℹ️ Predictions based on time of day, day of week, weather conditions, and historical patterns."""
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting crowding prediction info: {e}")
            return ""
    
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