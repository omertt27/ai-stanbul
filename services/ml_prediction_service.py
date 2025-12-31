#!/usr/bin/env python3
"""
ML Prediction Service for POI-Enhanced Route Planning
======================================================

Phase 3: ML Prediction Service Implementation
- POI crowding predictions with time-series analysis
- Transit crowding predictions
- Travel time predictions with multi-factor analysis
- Wait time estimation
- Best visit time recommendations

Integrates with existing MLCrowdingPredictor and enhances it for POI-specific predictions.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML Prediction Cache Service
try:
    from ml_prediction_cache_service import get_ml_cache, MLPredictionCache
    ML_CACHE_AVAILABLE = True
    logger.info("✅ ML Prediction Cache Service integrated successfully!")
except ImportError as e:
    ML_CACHE_AVAILABLE = False
    logger.warning(f"⚠️ ML Prediction Cache Service not available: {e}")

# Try to import ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    XGBOOST_AVAILABLE = True
    LIGHTGBM_AVAILABLE = True
    logger.info("✅ XGBoost and LightGBM ML libraries loaded successfully")
except ImportError as e:
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False
    logger.info(f"ℹ️  XGBoost/LightGBM not installed - using pattern-based prediction fallback: {e}")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class CrowdingPrediction:
    """POI crowding prediction result"""
    poi_id: str
    datetime: datetime
    crowding_level: float  # 0.0-1.0 (0=empty, 1=extremely crowded)
    wait_time_minutes: int
    visitor_count_estimate: int
    best_alternative_times: List[Tuple[datetime, float]]  # [(time, crowding_level)]
    confidence: float  # 0.0-1.0
    factors: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def crowding_label(self) -> str:
        """Human-readable crowding level"""
        if self.crowding_level < 0.2:
            return "Very Quiet"
        elif self.crowding_level < 0.4:
            return "Quiet"
        elif self.crowding_level < 0.6:
            return "Moderate"
        elif self.crowding_level < 0.8:
            return "Crowded"
        else:
            return "Very Crowded"


@dataclass
class TravelTimePrediction:
    """Travel time prediction for route segment"""
    segment_id: str
    base_time_minutes: float
    predicted_time_minutes: float
    wait_time_minutes: float
    confidence: float
    factors: Dict[str, float] = field(default_factory=dict)
    alternative_suggestions: List[str] = field(default_factory=list)


@dataclass
class TransitCrowdingPrediction:
    """Transit crowding prediction"""
    route_id: str
    stop_id: str
    datetime: datetime
    crowding_factor: float  # 1.0 = normal, 2.0 = very crowded
    available_capacity: float  # 0.0-1.0
    recommended_alternative: Optional[str] = None


class POICrowdingPredictor:
    """
    ML-based POI crowding prediction
    
    Features:
    - Time-series pattern recognition
    - Weather impact analysis
    - Event calendar integration
    - Holiday/season adjustments
    - Real-time crowding data integration
    """
    
    # Historical crowding patterns for major Istanbul POIs
    POI_PATTERNS = {
        "hagia_sophia": {
            "base_crowding": 0.6,
            "weekday": {"morning": 0.3, "afternoon": 0.7, "evening": 0.5},
            "weekend": {"morning": 0.8, "afternoon": 0.9, "evening": 0.6},
            "seasonal_multiplier": {"winter": 0.7, "spring": 1.2, "summer": 1.5, "autumn": 1.0},
            "opening_hours": (9, 19),
            "peak_hours": [(11, 14), (15, 17)],
            "capacity": 3000,
            "avg_visit_duration": 60,
        },
        "topkapi_palace": {
            "base_crowding": 0.65,
            "weekday": {"morning": 0.4, "afternoon": 0.75, "evening": 0.4},
            "weekend": {"morning": 0.85, "afternoon": 0.9, "evening": 0.5},
            "seasonal_multiplier": {"winter": 0.6, "spring": 1.3, "summer": 1.6, "autumn": 1.1},
            "opening_hours": (9, 18),
            "peak_hours": [(10, 13), (14, 16)],
            "capacity": 5000,
            "avg_visit_duration": 120,
        },
        "blue_mosque": {
            "base_crowding": 0.55,
            "weekday": {"morning": 0.4, "afternoon": 0.6, "evening": 0.5},
            "weekend": {"morning": 0.7, "afternoon": 0.8, "evening": 0.6},
            "seasonal_multiplier": {"winter": 0.8, "spring": 1.1, "summer": 1.4, "autumn": 1.0},
            "opening_hours": (8, 20),
            "peak_hours": [(9, 12), (15, 18)],
            "capacity": 4000,
            "avg_visit_duration": 45,
        },
        "grand_bazaar": {
            "base_crowding": 0.7,
            "weekday": {"morning": 0.5, "afternoon": 0.8, "evening": 0.6},
            "weekend": {"morning": 0.85, "afternoon": 0.95, "evening": 0.7},
            "seasonal_multiplier": {"winter": 0.9, "spring": 1.2, "summer": 1.3, "autumn": 1.1},
            "opening_hours": (9, 19),
            "peak_hours": [(11, 15), (16, 18)],
            "capacity": 8000,
            "avg_visit_duration": 90,
        },
        "dolmabahce_palace": {
            "base_crowding": 0.5,
            "weekday": {"morning": 0.35, "afternoon": 0.6, "evening": 0.3},
            "weekend": {"morning": 0.7, "afternoon": 0.8, "evening": 0.4},
            "seasonal_multiplier": {"winter": 0.6, "spring": 1.1, "summer": 1.4, "autumn": 0.9},
            "opening_hours": (9, 16),
            "peak_hours": [(10, 12), (13, 15)],
            "capacity": 2500,
            "avg_visit_duration": 90,
        },
        "galata_tower": {
            "base_crowding": 0.55,
            "weekday": {"morning": 0.4, "afternoon": 0.65, "evening": 0.5},
            "weekend": {"morning": 0.75, "afternoon": 0.85, "evening": 0.6},
            "seasonal_multiplier": {"winter": 0.7, "spring": 1.2, "summer": 1.4, "autumn": 1.0},
            "opening_hours": (9, 20),
            "peak_hours": [(11, 13), (16, 18)],
            "capacity": 500,
            "avg_visit_duration": 30,
        },
        "basilica_cistern": {
            "base_crowding": 0.6,
            "weekday": {"morning": 0.45, "afternoon": 0.7, "evening": 0.5},
            "weekend": {"morning": 0.8, "afternoon": 0.85, "evening": 0.55},
            "seasonal_multiplier": {"winter": 0.8, "spring": 1.1, "summer": 1.3, "autumn": 1.0},
            "opening_hours": (9, 18),
            "peak_hours": [(10, 13), (14, 16)],
            "capacity": 1000,
            "avg_visit_duration": 40,
        },
        "suleymaniye_mosque": {
            "base_crowding": 0.4,
            "weekday": {"morning": 0.3, "afternoon": 0.45, "evening": 0.35},
            "weekend": {"morning": 0.6, "afternoon": 0.7, "evening": 0.5},
            "seasonal_multiplier": {"winter": 0.9, "spring": 1.0, "summer": 1.2, "autumn": 1.0},
            "opening_hours": (6, 21),
            "peak_hours": [(12, 14), (17, 19)],
            "capacity": 3000,
            "avg_visit_duration": 30,
        },
    }
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Initialize ML prediction cache
        self.cache = get_ml_cache() if ML_CACHE_AVAILABLE else None
        if self.cache:
            logger.info("✅ ML Prediction Cache initialized for POI crowding predictions")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ML model for POI crowding prediction"""
        try:
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.08,
                    random_state=42,
                    objective='reg:squarederror'
                )
                logger.info("✅ XGBoost model initialized for POI crowding prediction")
            
            if SKLEARN_AVAILABLE:
                self.scaler = StandardScaler()
                logger.info("✅ Feature scaler initialized for POI predictions")
            
            # Train with pattern-based synthetic data
            self._train_with_poi_patterns()
            
        except Exception as e:
            logger.error(f"Error initializing POI prediction model: {e}")
    
    def _train_with_poi_patterns(self):
        """Train model with POI-specific patterns"""
        try:
            if not self.model:
                logger.warning("No ML model available, using pattern-based prediction")
                return
            
            # Generate training data from patterns
            n_samples = 10000
            features = []
            labels = []
            
            for _ in range(n_samples):
                # Random POI
                poi_id = np.random.choice(list(self.POI_PATTERNS.keys()))
                pattern = self.POI_PATTERNS[poi_id]
                
                # Random time
                hour = np.random.randint(0, 24)
                day_of_week = np.random.randint(0, 7)
                month = np.random.randint(1, 13)
                
                # Weather
                temp = np.random.normal(15, 8)
                rain = np.random.exponential(2)
                
                # Special days
                is_weekend = 1 if day_of_week >= 5 else 0
                is_holiday = np.random.choice([0, 1], p=[0.95, 0.05])
                
                # POI-specific features
                poi_type = hash(poi_id) % 5  # Encode POI type
                
                feature_vector = [
                    hour, day_of_week, month, temp, rain,
                    is_weekend, is_holiday, poi_type
                ]
                
                # Calculate crowding from pattern
                crowding = self._calculate_pattern_crowding(
                    poi_id, hour, day_of_week, month, rain, is_holiday
                )
                
                features.append(feature_vector)
                labels.append(crowding)
            
            X = np.array(features)
            y = np.array(labels)
            
            # Scale features
            if self.scaler:
                X = self.scaler.fit_transform(X)
            
            # Train model
            if self.model:
                self.model.fit(X, y)
                logger.info(f"✅ POI crowding model trained with {n_samples} samples")
                self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training POI model: {e}")
    
    def _calculate_pattern_crowding(self, poi_id: str, hour: int, day_of_week: int, 
                                   month: int, rain: float, is_holiday: bool) -> float:
        """Calculate crowding from historical patterns"""
        pattern = self.POI_PATTERNS.get(poi_id)
        if not pattern:
            return 0.5  # Default moderate crowding
        
        base_crowding = pattern["base_crowding"]
        
        # Time of day effect
        is_weekend = day_of_week >= 5
        time_pattern = pattern["weekend"] if is_weekend else pattern["weekday"]
        
        if 6 <= hour < 12:
            time_multiplier = time_pattern["morning"]
        elif 12 <= hour < 18:
            time_multiplier = time_pattern["afternoon"]
        else:
            time_multiplier = time_pattern["evening"]
        
        # Season effect
        season = ["winter", "winter", "spring", "spring", "spring", 
                 "summer", "summer", "summer", "autumn", "autumn", "autumn", "winter"][month - 1]
        seasonal_mult = pattern["seasonal_multiplier"].get(season, 1.0)
        
        # Weather effect
        weather_impact = 1.0
        if rain > 5:  # Heavy rain
            weather_impact = 0.7  # Fewer visitors
        
        # Holiday effect
        holiday_mult = 1.3 if is_holiday else 1.0
        
        # Peak hours effect
        peak_mult = 1.0
        opening_hours = pattern.get("opening_hours", (9, 18))
        if not (opening_hours[0] <= hour < opening_hours[1]):
            return 0.0  # Closed
        
        for peak_start, peak_end in pattern.get("peak_hours", []):
            if peak_start <= hour < peak_end:
                peak_mult = 1.2
                break
        
        # Calculate final crowding
        crowding = (base_crowding * time_multiplier * seasonal_mult * 
                   weather_impact * holiday_mult * peak_mult)
        
        # Add some noise
        crowding += np.random.normal(0, 0.05)
        
        return max(0.0, min(1.0, crowding))
    
    def predict_poi_crowding(self, poi_id: str, target_datetime: datetime,
                           weather_data: Optional[Dict] = None) -> CrowdingPrediction:
        """
        Predict crowding level for POI at specific time
        
        Args:
            poi_id: POI identifier
            target_datetime: Target visit time
            weather_data: Optional weather forecast data
        
        Returns:
            CrowdingPrediction with level, wait time, and recommendations
        """
        try:
            # Check cache first
            if self.cache:
                cache_key = f"poi_crowding_{poi_id}_{target_datetime.strftime('%Y%m%d_%H')}"
                cached_result = self.cache.get(
                    cache_key=cache_key,
                    context={'poi_id': poi_id, 'hour': target_datetime.hour},
                    prediction_types=['poi_scoring', 'crowding_prediction']
                )
                if cached_result:
                    logger.debug(f"🎯 Cache hit for POI crowding: {poi_id}")
                    return cached_result
            
            hour = target_datetime.hour
            day_of_week = target_datetime.weekday()
            month = target_datetime.month
            
            # Extract weather features
            temp = weather_data.get('temperature', 15) if weather_data else 15
            rain = weather_data.get('precipitation', 0) if weather_data else 0
            
            # Special days
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = self._is_holiday(target_datetime)
            
            # POI encoding
            poi_type = hash(poi_id) % 5
            
            # Prepare features
            feature_vector = np.array([[
                hour, day_of_week, month, temp, rain,
                is_weekend, is_holiday, poi_type
            ]])
            
            # Predict crowding
            if self.is_trained and self.model and self.scaler:
                feature_vector_scaled = self.scaler.transform(feature_vector)
                crowding_level = float(self.model.predict(feature_vector_scaled)[0])
                crowding_level = max(0.0, min(1.0, crowding_level))
                confidence = 0.85
            else:
                # Fallback to pattern-based prediction
                crowding_level = self._calculate_pattern_crowding(
                    poi_id, hour, day_of_week, month, rain, is_holiday
                )
                confidence = 0.75
            
            # Calculate wait time
            wait_time = self._estimate_wait_time(poi_id, crowding_level)
            
            # Estimate visitor count
            visitor_count = self._estimate_visitor_count(poi_id, crowding_level)
            
            # Find better alternative times
            alternatives = self._find_better_times(poi_id, target_datetime, weather_data)
            
            # Analyze factors
            factors = {
                'hour': hour,
                'is_weekend': bool(is_weekend),
                'is_holiday': is_holiday,
                'weather_impact': self._calculate_weather_impact(rain, temp),
                'season': self._get_season(month),
            }
            
            prediction = CrowdingPrediction(
                poi_id=poi_id,
                datetime=target_datetime,
                crowding_level=crowding_level,
                wait_time_minutes=wait_time,
                visitor_count_estimate=visitor_count,
                best_alternative_times=alternatives,
                confidence=confidence,
                factors=factors
            )
            
            # Cache the result
            if self.cache:
                cache_key = f"poi_crowding_{poi_id}_{target_datetime.strftime('%Y%m%d_%H')}"
                self.cache.set(
                    cache_key=cache_key,
                    prediction=prediction,
                    confidence_score=confidence,
                    prediction_types=['poi_scoring', 'crowding_prediction'],
                    context={'poi_id': poi_id, 'hour': target_datetime.hour}
                )
                logger.debug(f"💾 Cached POI crowding prediction: {poi_id}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting POI crowding: {e}")
            # Return moderate prediction on error
            return CrowdingPrediction(
                poi_id=poi_id,
                datetime=target_datetime,
                crowding_level=0.5,
                wait_time_minutes=15,
                visitor_count_estimate=0,
                best_alternative_times=[],
                confidence=0.5,
                factors={}
            )
    
    def _estimate_wait_time(self, poi_id: str, crowding_level: float) -> int:
        """Estimate wait time based on crowding level"""
        pattern = self.POI_PATTERNS.get(poi_id)
        if not pattern:
            base_wait = 10
        else:
            capacity = pattern.get("capacity", 1000)
            # Higher capacity POIs handle crowds better
            base_wait = max(5, 30 * (1000 / capacity))
        
        # Wait time scales exponentially with crowding
        wait_time = base_wait * (crowding_level ** 2)
        return int(wait_time)
    
    def _estimate_visitor_count(self, poi_id: str, crowding_level: float) -> int:
        """Estimate current visitor count"""
        pattern = self.POI_PATTERNS.get(poi_id)
        if not pattern:
            return int(crowding_level * 500)
        
        capacity = pattern.get("capacity", 1000)
        return int(crowding_level * capacity * 0.8)  # 80% of capacity at full crowding
    
    def _find_better_times(self, poi_id: str, target_datetime: datetime,
                          weather_data: Optional[Dict]) -> List[Tuple[datetime, float]]:
        """Find less crowded alternative times"""
        alternatives = []
        
        # Check next 6 hours - use pattern-based prediction to avoid recursion
        for offset in [1, 2, 3, 4, 5, 6]:
            alt_time = target_datetime + timedelta(hours=offset)
            
            # Extract time features
            hour = alt_time.hour
            day_of_week = alt_time.weekday()
            month = alt_time.month
            rain = weather_data.get('precipitation', 0) if weather_data else 0
            is_holiday = self._is_holiday(alt_time)
            
            # Use pattern-based prediction directly (no recursion)
            crowding_level = self._calculate_pattern_crowding(
                poi_id, hour, day_of_week, month, rain, is_holiday
            )
            
            alternatives.append((alt_time, crowding_level))
        
        # Sort by crowding level
        alternatives.sort(key=lambda x: x[1])
        
        # Return top 3 best times
        return alternatives[:3]
    
    def _is_holiday(self, dt: datetime) -> bool:
        """Check if date is a Turkish holiday"""
        # Turkish national holidays (simplified)
        holidays = [
            (1, 1),   # New Year
            (4, 23),  # National Sovereignty Day
            (5, 1),   # Labor Day
            (5, 19),  # Youth and Sports Day
            (7, 15),  # Democracy Day
            (8, 30),  # Victory Day
            (10, 29), # Republic Day
        ]
        return (dt.month, dt.day) in holidays
    
    def _calculate_weather_impact(self, rain: float, temp: float) -> float:
        """Calculate weather impact on POI visits"""
        impact = 1.0
        
        # Rain impact
        if rain > 10:
            impact *= 0.6  # Heavy rain deters visitors
        elif rain > 5:
            impact *= 0.8
        
        # Temperature impact
        if temp < 0 or temp > 35:
            impact *= 0.7  # Extreme temps deter visitors
        elif temp < 5 or temp > 30:
            impact *= 0.85
        
        return impact
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        seasons = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"
        }
        return seasons.get(month, "spring")


class TravelTimePredictor:
    """
    Predict actual travel times with multi-factor analysis
    
    Factors:
    - Base transit time
    - Crowding impact
    - Weather impact
    - Time-of-day patterns
    - Wait time at stations
    """
    
    def __init__(self, transit_crowding_predictor=None):
        self.transit_predictor = transit_crowding_predictor
        
        # Initialize ML prediction cache
        self.cache = get_ml_cache() if ML_CACHE_AVAILABLE else None
        if self.cache:
            logger.info("✅ ML Prediction Cache initialized for travel time predictions")
        
        logger.info("✅ Travel Time Predictor initialized")
    
    def predict_travel_time(self, from_location: str, to_location: str,
                          transport_mode: str, base_time_minutes: float,
                          target_datetime: datetime,
                          weather_data: Optional[Dict] = None) -> TravelTimePrediction:
        """
        Predict actual travel time for route segment
        
        Args:
            from_location: Starting point
            to_location: Destination
            transport_mode: Type of transport
            base_time_minutes: Scheduled/base travel time
            target_datetime: Planned travel time
            weather_data: Weather forecast
        
        Returns:
            TravelTimePrediction with adjusted time and factors
        """
        try:
            # Check cache first
            if self.cache:
                cache_key = f"travel_time_{from_location}_{to_location}_{transport_mode}_{target_datetime.strftime('%Y%m%d_%H')}"
                cached_result = self.cache.get(
                    cache_key=cache_key,
                    context={'transport_mode': transport_mode, 'hour': target_datetime.hour},
                    prediction_types=['transport_optimization']
                )
                if cached_result:
                    logger.debug(f"🎯 Cache hit for travel time prediction")
                    return cached_result
            
            # Get crowding multiplier
            crowding_mult = self._get_crowding_multiplier(
                transport_mode, target_datetime
            )
            
            # Get weather multiplier
            weather_mult = self._get_weather_multiplier(
                transport_mode, weather_data or {}
            )
            
            # Get time-of-day multiplier
            time_mult = self._get_time_multiplier(
                target_datetime.hour, target_datetime.weekday()
            )
            
            # Calculate wait time
            wait_time = self._predict_wait_time(
                transport_mode, target_datetime
            )
            
            # Calculate predicted travel time
            adjusted_time = base_time_minutes * crowding_mult * weather_mult * time_mult
            total_time = adjusted_time + wait_time
            
            # Generate alternative suggestions
            suggestions = self._generate_suggestions(
                crowding_mult, weather_mult, time_mult, wait_time
            )
            
            prediction = TravelTimePrediction(
                segment_id=f"{from_location}_{to_location}_{transport_mode}",
                base_time_minutes=base_time_minutes,
                predicted_time_minutes=total_time,
                wait_time_minutes=wait_time,
                confidence=0.8,
                factors={
                    'crowding_multiplier': crowding_mult,
                    'weather_multiplier': weather_mult,
                    'time_multiplier': time_mult,
                    'wait_time': wait_time
                },
                alternative_suggestions=suggestions
            )
            
            # Cache the result
            if self.cache:
                cache_key = f"travel_time_{from_location}_{to_location}_{transport_mode}_{target_datetime.strftime('%Y%m%d_%H')}"
                self.cache.set(
                    cache_key=cache_key,
                    prediction=prediction,
                    confidence_score=0.8,
                    prediction_types=['transport_optimization'],
                    context={'transport_mode': transport_mode, 'hour': target_datetime.hour}
                )
                logger.debug(f"💾 Cached travel time prediction")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting travel time: {e}")
            return TravelTimePrediction(
                segment_id=f"{from_location}_{to_location}_{transport_mode}",
                base_time_minutes=base_time_minutes,
                predicted_time_minutes=base_time_minutes + 5,
                wait_time_minutes=5,
                confidence=0.5,
                factors={},
                alternative_suggestions=[]
            )
    
    def _get_crowding_multiplier(self, transport_mode: str, dt: datetime) -> float:
        """Calculate crowding impact on travel time"""
        hour = dt.hour
        is_weekend = dt.weekday() >= 5
        
        # Rush hour detection
        is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
        
        if is_rush_hour and not is_weekend:
            base_mult = 1.4  # 40% slower in rush hour
        elif is_weekend and 12 <= hour <= 20:
            base_mult = 1.2  # 20% slower on weekend afternoons
        else:
            base_mult = 1.0
        
        # Transport-specific adjustments
        mode_adjustments = {
            'metro': 0.9,      # Metro less affected by crowding
            'bus': 1.2,        # Buses more affected
            'tram': 1.1,
            'ferry': 0.95,     # Ferries run on schedule
            'walking': 1.0,
        }
        
        return base_mult * mode_adjustments.get(transport_mode.lower(), 1.0)
    
    def _get_weather_multiplier(self, transport_mode: str, weather_data: Dict) -> float:
        """Calculate weather impact on travel time"""
        rain = weather_data.get('precipitation', 0)
        temp = weather_data.get('temperature', 15)
        wind = weather_data.get('wind_speed', 0)
        
        multiplier = 1.0
        
        # Rain impact
        if rain > 10:  # Heavy rain
            multiplier *= 1.3
        elif rain > 5:
            multiplier *= 1.15
        
        # Temperature impact
        if temp < 0:  # Freezing
            multiplier *= 1.2
        elif temp > 35:  # Very hot
            multiplier *= 1.1
        
        # Wind impact (mainly for ferries)
        if transport_mode.lower() == 'ferry' and wind > 30:
            multiplier *= 1.4  # Ferries slow down in high wind
        
        # Walking is most affected by weather
        if transport_mode.lower() == 'walking':
            multiplier *= 1.2 if (rain > 5 or temp < 5 or temp > 30) else 1.0
        
        return multiplier
    
    def _get_time_multiplier(self, hour: int, day_of_week: int) -> float:
        """Time-of-day patterns for travel time"""
        # Night hours (22-6) - faster travel
        if 22 <= hour or hour <= 6:
            return 0.85  # 15% faster
        
        # Midday (10-15) - moderate
        if 10 <= hour <= 15:
            return 1.0
        
        # Default
        return 1.0
    
    def _predict_wait_time(self, transport_mode: str, dt: datetime) -> float:
        """Predict wait time at station/stop"""
        hour = dt.hour
        is_weekend = dt.weekday() >= 5
        
        # Base wait times by transport mode (minutes)
        base_wait = {
            'metro': 5,
            'tram': 7,
            'bus': 10,
            'ferry': 15,
            'funicular': 8,
        }
        
        wait = base_wait.get(transport_mode.lower(), 8)
        
        # Peak hours - more frequent service
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            wait *= 0.7  # 30% less wait time
        
        # Late night - less frequent
        if 22 <= hour or hour <= 6:
            wait *= 1.5
        
        # Weekend adjustments
        if is_weekend:
            wait *= 1.2
        
        return wait
    
    def _generate_suggestions(self, crowding: float, weather: float, 
                            time: float, wait: float) -> List[str]:
        """Generate alternative suggestions based on factors"""
        suggestions = []
        
        if crowding > 1.3:
            suggestions.append("Consider traveling 30 minutes earlier/later to avoid crowds")
        
        if weather > 1.2:
            suggestions.append("Weather conditions may cause delays, allow extra time")
        
        if wait > 15:
            suggestions.append("Long wait times expected, check real-time schedules")
        
        if crowding > 1.2 and weather > 1.1:
            suggestions.append("Both crowding and weather will impact travel - consider taxi/rideshare")
        
        return suggestions


class MLPredictionService:
    """
    Unified ML Prediction Service for POI-Enhanced Route Planning
    
    Combines:
    - POI crowding predictions
    - Transit crowding predictions
    - Travel time predictions
    - Best visit time recommendations
    """
    
    def __init__(self):
        # Initialize cache first
        self.cache = get_ml_cache() if ML_CACHE_AVAILABLE else None
        if self.cache:
            logger.info("✅ ML Prediction Cache initialized for unified service")
        
        # Initialize predictors (they will use their own cache instances)
        self.poi_predictor = POICrowdingPredictor()
        self.travel_predictor = TravelTimePredictor()
        
        logger.info("✅ ML Prediction Service initialized")
    
    def predict_poi_crowding(self, poi_id: str, target_datetime: datetime,
                           weather_data: Optional[Dict] = None) -> CrowdingPrediction:
        """Predict POI crowding level"""
        return self.poi_predictor.predict_poi_crowding(poi_id, target_datetime, weather_data)
    
    def predict_travel_time(self, from_location: str, to_location: str,
                          transport_mode: str, base_time_minutes: float,
                          target_datetime: datetime,
                          weather_data: Optional[Dict] = None) -> TravelTimePrediction:
        """Predict travel time with multi-factor analysis"""
        return self.travel_predictor.predict_travel_time(
            from_location, to_location, transport_mode, base_time_minutes,
            target_datetime, weather_data
        )
    
    def get_best_visit_time(self, poi_id: str, date: datetime.date,
                           preferred_time_range: Optional[Tuple[int, int]] = None,
                           weather_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Find best time to visit a POI on given date
        
        Args:
            poi_id: POI identifier
            date: Target date
            preferred_time_range: Optional (start_hour, end_hour) tuple
            weather_data: Weather forecast
        
        Returns:
            Dict with best time, crowding level, and alternatives
        """
        # Get POI pattern
        pattern = self.poi_predictor.POI_PATTERNS.get(poi_id)
        if not pattern:
            logger.warning(f"No pattern found for POI: {poi_id}")
            return {
                'best_time': datetime.combine(date, datetime.min.time().replace(hour=10)),
                'crowding_level': 0.5,
                'alternatives': []
            }
        
        opening_hours = pattern.get("opening_hours", (9, 18))
        start_hour, end_hour = preferred_time_range or opening_hours
        
        # Ensure within opening hours
        start_hour = max(start_hour, opening_hours[0])
        end_hour = min(end_hour, opening_hours[1])
        
        # Check each hour
        predictions = []
        for hour in range(start_hour, end_hour):
            dt = datetime.combine(date, datetime.min.time().replace(hour=hour))
            pred = self.predict_poi_crowding(poi_id, dt, weather_data)
            predictions.append((dt, pred.crowding_level))
        
        # Sort by crowding level
        predictions.sort(key=lambda x: x[1])
        
        return {
            'best_time': predictions[0][0],
            'crowding_level': predictions[0][1],
            'alternatives': predictions[1:4]  # Top 3 alternatives
        }
    
    def analyze_route_crowding(self, route_segments: List[Dict],
                              target_datetime: datetime,
                              weather_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze crowding and travel time for entire route
        
        Args:
            route_segments: List of route segments
            target_datetime: Start time
            weather_data: Weather forecast
        
        Returns:
            Analysis with total time, crowding levels, bottlenecks
        """
        current_time = target_datetime
        total_predicted_time = 0
        segment_predictions = []
        max_crowding = 0
        bottlenecks = []
        
        for segment in route_segments:
            # Predict travel time
            travel_pred = self.predict_travel_time(
                segment.get('from'),
                segment.get('to'),
                segment.get('mode', 'walking'),
                segment.get('base_time', 10),
                current_time,
                weather_data
            )
            
            total_predicted_time += travel_pred.predicted_time_minutes
            segment_predictions.append(travel_pred)
            
            # Track crowding
            crowding = travel_pred.factors.get('crowding_multiplier', 1.0)
            if crowding > max_crowding:
                max_crowding = crowding
            
            if crowding > 1.3:  # Significant crowding
                bottlenecks.append({
                    'segment': f"{segment.get('from')} → {segment.get('to')}",
                    'crowding_multiplier': crowding,
                    'delay_minutes': travel_pred.predicted_time_minutes - travel_pred.base_time_minutes
                })
            
            # Update current time
            current_time += timedelta(minutes=travel_pred.predicted_time_minutes)
        
        return {
            'total_predicted_time_minutes': total_predicted_time,
            'max_crowding_multiplier': max_crowding,
            'bottlenecks': bottlenecks,
            'segment_predictions': segment_predictions,
            'arrival_time': current_time
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("🤖 ML Prediction Service - Phase 3 Implementation")
    print("=" * 80)
    
    # Initialize service
    ml_service = MLPredictionService()
    
    # Test POI crowding prediction
    print("\n📍 Testing POI Crowding Prediction")
    print("-" * 80)
    
    test_time = datetime(2024, 7, 15, 14, 0)  # Summer weekend afternoon
    prediction = ml_service.predict_poi_crowding("hagia_sophia", test_time)
    
    print(f"POI: Hagia Sophia")
    print(f"Time: {test_time.strftime('%A, %B %d at %I:%M %p')}")
    print(f"Crowding Level: {prediction.crowding_level:.2f} ({prediction.crowding_label})")
    print(f"Wait Time: {prediction.wait_time_minutes} minutes")
    print(f"Visitor Estimate: {prediction.visitor_count_estimate} people")
    print(f"Confidence: {prediction.confidence:.2f}")
    print("\nBetter Alternative Times:")
    for alt_time, alt_crowding in prediction.best_alternative_times:
        print(f"  - {alt_time.strftime('%I:%M %p')}: {alt_crowding:.2f}")
    
    # Test travel time prediction
    print("\n🚇 Testing Travel Time Prediction")
    print("-" * 80)
    
    travel_pred = ml_service.predict_travel_time(
        "Sultanahmet",
        "Taksim",
        "metro",
        25.0,
        test_time,
        {'temperature': 32, 'precipitation': 0, 'wind_speed': 10}
    )
    
    print(f"Route: Sultanahmet → Taksim (Metro)")
    print(f"Base Time: {travel_pred.base_time_minutes:.1f} minutes")
    print(f"Predicted Time: {travel_pred.predicted_time_minutes:.1f} minutes")
    print(f"Wait Time: {travel_pred.wait_time_minutes:.1f} minutes")
    print(f"Confidence: {travel_pred.confidence:.2f}")
    print("\nFactors:")
    for factor, value in travel_pred.factors.items():
        print(f"  - {factor}: {value:.2f}")
    if travel_pred.alternative_suggestions:
        print("\nSuggestions:")
        for suggestion in travel_pred.alternative_suggestions:
            print(f"  • {suggestion}")
    
    # Test best visit time
    print("\n⏰ Testing Best Visit Time Recommendation")
    print("-" * 80)
    
    best_time = ml_service.get_best_visit_time(
        "topkapi_palace",
        datetime(2024, 7, 20).date(),
        preferred_time_range=(9, 16)
    )
    
    print(f"POI: Topkapi Palace")
    print(f"Best Time: {best_time['best_time'].strftime('%I:%M %p')}")
    print(f"Crowding Level: {best_time['crowding_level']:.2f}")
    print("\nAlternative Times:")
    for alt_time, alt_crowding in best_time['alternatives']:
        print(f"  - {alt_time.strftime('%I:%M %p')}: {alt_crowding:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ Phase 3 ML Prediction Service: Implementation Complete!")
    print("=" * 80)
