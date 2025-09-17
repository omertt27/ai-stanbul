"""
Predictive Analytics Module for AI Istanbul Travel Guide
Provides weather-based suggestions, seasonal adjustments, and peak time predictions
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import math
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class Season(Enum):
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"

class ActivityType(Enum):
    OUTDOOR = "outdoor"
    INDOOR = "indoor"
    MIXED = "mixed"

@dataclass
class WeatherPrediction:
    """Weather-based prediction for activities"""
    date: datetime
    temperature_range: Tuple[int, int]
    precipitation_chance: float
    recommended_activities: List[str]
    avoid_activities: List[str]
    optimal_times: List[str]
    clothing_suggestions: List[str]

@dataclass
class SeasonalAdjustment:
    """Seasonal adjustments for recommendations"""
    season: Season
    popular_activities: List[str]
    crowd_multiplier: float
    price_multiplier: float
    operating_hours_changes: Dict[str, str]
    special_events: List[str]

@dataclass
class PeakTimePrediction:
    """Peak time predictions for attractions and areas"""
    location: str
    peak_hours: List[Tuple[int, int]]  # (start_hour, end_hour)
    crowd_level_by_hour: Dict[int, str]  # hour -> crowd_level
    best_visit_times: List[str]
    average_wait_time: Dict[int, int]  # hour -> minutes
    seasonal_variations: Dict[Season, float]

class WeatherBasedPredictor:
    """Predicts optimal activities based on weather conditions"""
    
    def __init__(self):
        # Activity categories based on weather suitability
        self.weather_activities = {
            "sunny_warm": {
                "outdoor": [
                    "Bosphorus ferry cruise", "Galata Bridge walk", "Çamlıca Hill visit",
                    "Emirgan Park picnic", "Ortaköy waterfront", "Bebek coastal walk",
                    "Prince Islands day trip", "Dolmabahçe Palace gardens",
                    "Gülhane Park stroll", "Asian side exploration"
                ],
                "mixed": [
                    "Topkapi Palace visit", "Blue Mosque tour", "Hagia Sophia visit",
                    "Grand Bazaar shopping", "Spice Bazaar exploration"
                ]
            },
            "sunny_cool": {
                "outdoor": [
                    "Historic Peninsula walking tour", "Galata Tower area exploration",
                    "Taksim to Galata walk", "Sultanahmet square visit",
                    "Bosphorus Bridge walk", "Kadıköy street exploration"
                ],
                "mixed": [
                    "Museum visits", "Covered bazaar shopping", "Indoor-outdoor restaurants",
                    "Turkish bath experience", "Café hopping in Beyoğlu"
                ]
            },
            "cloudy_mild": {
                "outdoor": [
                    "Photography walks", "Neighborhood exploration", "Market visits",
                    "Short boat trips", "Balat colorful houses tour"
                ],
                "indoor": [
                    "Museum tours", "Shopping centers", "Traditional restaurants",
                    "Turkish cooking classes", "Art galleries"
                ]
            },
            "rainy": {
                "indoor": [
                    "Grand Bazaar extended visit", "Turkish bath (hammam)",
                    "Museum marathon", "Covered shopping areas", "Traditional tea houses",
                    "Underground Cistern visit", "Indoor cultural shows",
                    "Turkish cuisine restaurants", "Art galleries and exhibitions"
                ]
            },
            "hot": {
                "indoor": [
                    "Air-conditioned museums", "Shopping malls", "Cool restaurants",
                    "Turkish baths", "Underground attractions"
                ],
                "water": [
                    "Bosphorus cruise", "Ferry rides", "Waterfront restaurants",
                    "Beach clubs", "Swimming areas"
                ]
            },
            "cold": {
                "indoor": [
                    "Museums and galleries", "Traditional Turkish baths",
                    "Warm restaurants and cafés", "Covered markets",
                    "Tea and coffee houses", "Indoor cultural venues"
                ],
                "warm_outdoor": [
                    "Brief monument visits", "Quick photo stops",
                    "Warm clothing shopping", "Indoor-outdoor venues"
                ]
            }
        }
        
        # Weather condition mappings
        self.weather_mappings = {
            "temperature": {
                "cold": (-5, 8),
                "cool": (8, 18),
                "mild": (18, 25),
                "warm": (25, 30),
                "hot": (30, 40)
            },
            "conditions": {
                "sunny": ["clear", "sunny", "fair"],
                "cloudy": ["cloudy", "overcast", "partly cloudy"],
                "rainy": ["rain", "drizzle", "shower", "thunderstorm"],
                "snowy": ["snow", "sleet", "freezing"]
            }
        }
    
    async def predict_optimal_activities(
        self, 
        weather_data: Dict[str, Any], 
        date: Optional[datetime] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> WeatherPrediction:
        """Predict optimal activities based on weather"""
        
        if not date:
            date = datetime.now()
        
        temperature = weather_data.get("temperature", 20)
        description = weather_data.get("description", "").lower()
        humidity = weather_data.get("humidity", 50)
        wind_speed = weather_data.get("wind_speed", 0)
        is_raining = weather_data.get("is_raining", False)
        
        # Determine weather category
        weather_category = self._categorize_weather(temperature, description, is_raining, humidity, wind_speed)
        
        # Get base recommendations
        recommended_activities = self._get_weather_activities(weather_category)
        avoid_activities = self._get_activities_to_avoid(weather_category)
        optimal_times = self._get_optimal_times(weather_category, temperature)
        clothing_suggestions = self._get_clothing_suggestions(temperature, is_raining, wind_speed)
        
        # Apply user preferences if available
        if user_preferences:
            recommended_activities = self._apply_user_preferences(recommended_activities, user_preferences)
        
        # Predict temperature range for the day
        temp_range = self._predict_daily_temperature_range(temperature, date)
        
        # Calculate precipitation chance
        precipitation_chance = self._calculate_precipitation_chance(description, humidity)
        
        return WeatherPrediction(
            date=date,
            temperature_range=temp_range,
            precipitation_chance=precipitation_chance,
            recommended_activities=recommended_activities[:8],  # Top 8
            avoid_activities=avoid_activities,
            optimal_times=optimal_times,
            clothing_suggestions=clothing_suggestions
        )
    
    def _categorize_weather(self, temp: float, description: str, is_raining: bool, humidity: float, wind_speed: float) -> str:
        """Categorize weather into activity-friendly categories"""
        
        if is_raining or any(rain_word in description for rain_word in ["rain", "shower", "drizzle"]):
            return "rainy"
        
        if temp >= 30:
            return "hot"
        elif temp <= 8:
            return "cold"
        elif temp <= 18:
            if any(sunny_word in description for sunny_word in ["clear", "sunny", "fair"]):
                return "sunny_cool"
            else:
                return "cloudy_mild"
        else:  # temp > 18 and < 30
            if any(sunny_word in description for sunny_word in ["clear", "sunny", "fair"]):
                return "sunny_warm"
            else:
                return "cloudy_mild"
    
    def _get_weather_activities(self, weather_category: str) -> List[str]:
        """Get recommended activities for weather category"""
        activities = []
        
        if weather_category in self.weather_activities:
            category_activities = self.weather_activities[weather_category]
            for activity_type, activity_list in category_activities.items():
                activities.extend(activity_list)
        
        return activities
    
    def _get_activities_to_avoid(self, weather_category: str) -> List[str]:
        """Get activities to avoid based on weather"""
        avoid_map = {
            "rainy": ["Outdoor walking tours", "Park visits", "Bosphorus cruise", "Outdoor markets"],
            "hot": ["Long walking tours", "Outdoor markets in midday", "Non-air-conditioned venues"],
            "cold": ["Outdoor dining", "Long outdoor walks", "Boat trips", "Open-air venues"],
            "sunny_warm": [],  # Good weather - avoid very little
            "sunny_cool": ["Swimming", "Very long outdoor activities"],
            "cloudy_mild": []  # Moderate weather - avoid very little
        }
        
        return avoid_map.get(weather_category, [])
    
    def _get_optimal_times(self, weather_category: str, temperature: float) -> List[str]:
        """Get optimal times for activities based on weather"""
        
        if weather_category == "hot":
            return ["Early morning (8-10 AM)", "Late afternoon (5-7 PM)", "Evening (after 7 PM)"]
        elif weather_category == "cold":
            return ["Late morning (10 AM-12 PM)", "Early afternoon (1-3 PM)"]
        elif weather_category == "rainy":
            return ["Anytime (indoor activities)", "Between rain breaks"]
        else:
            return ["Morning (9-11 AM)", "Afternoon (2-5 PM)", "Early evening (5-7 PM)"]
    
    def _get_clothing_suggestions(self, temperature: float, is_raining: bool, wind_speed: float) -> List[str]:
        """Get clothing suggestions based on weather"""
        suggestions = []
        
        if is_raining:
            suggestions.extend(["Waterproof jacket or umbrella", "Non-slip shoes", "Quick-dry clothing"])
        
        if temperature <= 5:
            suggestions.extend(["Heavy winter coat", "Warm layers", "Insulated shoes", "Gloves and hat"])
        elif temperature <= 15:
            suggestions.extend(["Light jacket or sweater", "Long pants", "Closed shoes"])
        elif temperature <= 25:
            suggestions.extend(["Light layers", "Comfortable walking shoes", "Light jacket for evening"])
        else:
            suggestions.extend(["Light, breathable clothing", "Sun hat", "Sunscreen", "Comfortable walking shoes"])
        
        if wind_speed > 20:
            suggestions.append("Wind-resistant outer layer")
        
        suggestions.append("Comfortable walking shoes (Istanbul involves lots of walking)")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _apply_user_preferences(self, activities: List[str], preferences: Dict[str, Any]) -> List[str]:
        """Apply user preferences to filter activities"""
        interests = preferences.get("interests", [])
        travel_style = preferences.get("travel_style", "solo")
        budget_level = preferences.get("budget_level", "moderate")
        
        filtered_activities = []
        
        for activity in activities:
            activity_lower = activity.lower()
            
            # Filter by interests
            if interests:
                if any(interest in activity_lower for interest in interests):
                    filtered_activities.append(activity)
                elif not any(interest in activity_lower for interest in ["museum", "restaurant", "shopping", "outdoor"]):
                    # Include general activities that don't match specific interests
                    filtered_activities.append(activity)
            else:
                filtered_activities.append(activity)
        
        # Apply travel style filters
        if travel_style == "family":
            family_friendly = [act for act in filtered_activities if not any(word in act.lower() for word in ["bar", "nightlife", "adult"])]
            return family_friendly
        elif travel_style == "couple":
            romantic_activities = [act for act in filtered_activities if any(word in act.lower() for word in ["cruise", "walk", "restaurant", "sunset", "view"])]
            return romantic_activities if romantic_activities else filtered_activities
        
        return filtered_activities
    
    def _predict_daily_temperature_range(self, current_temp: float, date: datetime) -> Tuple[int, int]:
        """Predict daily temperature range"""
        # Simple model: current temp ± variation based on time of day
        hour = date.hour
        
        if 6 <= hour <= 10:  # Morning - current temp is likely near daily minimum
            min_temp = int(current_temp)
            max_temp = int(current_temp + 8)
        elif 12 <= hour <= 16:  # Afternoon - current temp is likely near daily maximum
            min_temp = int(current_temp - 8)
            max_temp = int(current_temp)
        else:  # Evening/night - current temp is between min and max
            min_temp = int(current_temp - 4)
            max_temp = int(current_temp + 4)
        
        return (min_temp, max_temp)
    
    def _calculate_precipitation_chance(self, description: str, humidity: float) -> float:
        """Calculate precipitation chance based on description and humidity"""
        if any(word in description.lower() for word in ["rain", "shower", "drizzle", "thunderstorm"]):
            return 0.8
        elif any(word in description.lower() for word in ["cloudy", "overcast"]):
            return 0.3 if humidity > 70 else 0.1
        else:
            return 0.05

class SeasonalAnalyzer:
    """Analyzes seasonal patterns and adjusts recommendations"""
    
    def __init__(self):
        # Seasonal data for Istanbul
        self.seasonal_data = {
            Season.WINTER: {
                "months": [12, 1, 2],
                "temperature_range": (5, 15),
                "crowd_multiplier": 0.6,  # Lower tourist crowds
                "price_multiplier": 0.8,  # Lower prices
                "popular_activities": [
                    "Museum visits", "Turkish baths", "Covered bazaars",
                    "Traditional restaurants", "Tea houses", "Shopping centers"
                ],
                "special_events": [
                    "New Year celebrations", "Winter festivals", "Indoor concerts"
                ],
                "operating_hours": {
                    "museums": "Shorter hours (9 AM - 5 PM)",
                    "outdoor_attractions": "Limited hours",
                    "ferries": "Reduced schedule"
                }
            },
            Season.SPRING: {
                "months": [3, 4, 5],
                "temperature_range": (12, 22),
                "crowd_multiplier": 0.9,
                "price_multiplier": 0.9,
                "popular_activities": [
                    "Walking tours", "Park visits", "Outdoor cafés",
                    "Bosphorus cruises", "Photography tours", "Market exploration"
                ],
                "special_events": [
                    "Spring festivals", "Tulip season", "Outdoor events"
                ],
                "operating_hours": {
                    "most_venues": "Extended hours",
                    "outdoor_attractions": "Full schedule"
                }
            },
            Season.SUMMER: {
                "months": [6, 7, 8],
                "temperature_range": (20, 30),
                "crowd_multiplier": 1.3,  # Peak tourist season
                "price_multiplier": 1.2,  # Higher prices
                "popular_activities": [
                    "Bosphorus activities", "Outdoor dining", "Ferry rides",
                    "Beach visits", "Evening tours", "Rooftop venues"
                ],
                "special_events": [
                    "Summer festivals", "Outdoor concerts", "Cultural events"
                ],
                "operating_hours": {
                    "most_venues": "Extended summer hours",
                    "outdoor_attractions": "Late hours",
                    "ferries": "Frequent schedule"
                }
            },
            Season.FALL: {
                "months": [9, 10, 11],
                "temperature_range": (15, 25),
                "crowd_multiplier": 0.8,
                "price_multiplier": 0.9,
                "popular_activities": [
                    "Walking tours", "Cultural visits", "Food tours",
                    "Photography", "Outdoor exploration", "Moderate hiking"
                ],
                "special_events": [
                    "Autumn festivals", "Cultural events", "Art exhibitions"
                ],
                "operating_hours": {
                    "most_venues": "Standard hours",
                    "outdoor_attractions": "Good availability"
                }
            }
        }
    
    def get_seasonal_adjustment(self, date: Optional[datetime] = None) -> SeasonalAdjustment:
        """Get seasonal adjustments for recommendations"""
        if not date:
            date = datetime.now()
        
        season = self._determine_season(date)
        season_data = self.seasonal_data[season]
        
        return SeasonalAdjustment(
            season=season,
            popular_activities=season_data["popular_activities"],
            crowd_multiplier=season_data["crowd_multiplier"],
            price_multiplier=season_data["price_multiplier"],
            operating_hours_changes=season_data["operating_hours"],
            special_events=season_data["special_events"]
        )
    
    def adjust_recommendations_for_season(
        self, 
        base_recommendations: List[str], 
        date: Optional[datetime] = None
    ) -> List[str]:
        """Adjust recommendations based on seasonal appropriateness"""
        adjustment = self.get_seasonal_adjustment(date)
        season = adjustment.season
        
        # Seasonal activity preferences
        seasonal_preferences = {
            Season.WINTER: {
                "prefer": ["indoor", "warm", "covered", "museum", "restaurant", "shopping"],
                "avoid": ["outdoor", "cruise", "park", "beach", "swimming"]
            },
            Season.SPRING: {
                "prefer": ["walk", "park", "outdoor", "garden", "tour", "explore"],
                "avoid": ["beach", "swimming"]
            },
            Season.SUMMER: {
                "prefer": ["cruise", "outdoor", "ferry", "beach", "water", "evening"],
                "avoid": ["indoor", "hot", "midday"]
            },
            Season.FALL: {
                "prefer": ["walk", "cultural", "photography", "tour", "explore"],
                "avoid": ["beach", "swimming", "very_outdoor"]
            }
        }
        
        preferences = seasonal_preferences.get(season, {"prefer": [], "avoid": []})
        
        # Score recommendations based on seasonal appropriateness
        scored_recommendations = []
        for rec in base_recommendations:
            rec_lower = rec.lower()
            score = 1.0
            
            # Boost for preferred activities
            for prefer_term in preferences["prefer"]:
                if prefer_term in rec_lower:
                    score += 0.3
            
            # Penalize avoided activities
            for avoid_term in preferences["avoid"]:
                if avoid_term in rec_lower:
                    score -= 0.5
            
            # Add seasonal popular activities
            for popular_activity in adjustment.popular_activities:
                if any(word in rec_lower for word in popular_activity.lower().split()):
                    score += 0.2
            
            scored_recommendations.append((rec, score))
        
        # Sort by score and return top recommendations
        scored_recommendations.sort(key=lambda x: x[1], reverse=True)
        return [rec for rec, score in scored_recommendations if score > 0.5]
    
    def _determine_season(self, date: datetime) -> Season:
        """Determine season based on date"""
        month = date.month
        
        if month in [12, 1, 2]:
            return Season.WINTER
        elif month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        else:  # 9, 10, 11
            return Season.FALL

class PeakTimePredictor:
    """Predicts peak times and crowd levels for attractions"""
    
    def __init__(self):
        # Peak time data for major Istanbul attractions
        self.attraction_patterns = {
            "hagia_sophia": {
                "typical_peaks": [(10, 12), (14, 16)],
                "weekend_multiplier": 1.4,
                "seasonal_variations": {
                    Season.SUMMER: 1.3,
                    Season.SPRING: 1.1,
                    Season.FALL: 0.9,
                    Season.WINTER: 0.7
                },
                "crowd_by_hour": {
                    8: "low", 9: "moderate", 10: "high", 11: "very_high",
                    12: "high", 13: "moderate", 14: "high", 15: "very_high",
                    16: "high", 17: "moderate", 18: "low"
                }
            },
            "blue_mosque": {
                "typical_peaks": [(9, 11), (13, 15), (16, 17)],
                "weekend_multiplier": 1.3,
                "seasonal_variations": {
                    Season.SUMMER: 1.2,
                    Season.SPRING: 1.0,
                    Season.FALL: 0.9,
                    Season.WINTER: 0.8
                },
                "prayer_times_affect": True
            },
            "grand_bazaar": {
                "typical_peaks": [(11, 13), (15, 17)],
                "weekend_multiplier": 1.5,
                "seasonal_variations": {
                    Season.SUMMER: 1.4,
                    Season.SPRING: 1.1,
                    Season.FALL: 1.0,
                    Season.WINTER: 0.8
                }
            },
            "galata_tower": {
                "typical_peaks": [(16, 19)],  # Sunset views
                "weekend_multiplier": 1.6,
                "seasonal_variations": {
                    Season.SUMMER: 1.5,
                    Season.SPRING: 1.2,
                    Season.FALL: 1.1,
                    Season.WINTER: 0.6
                }
            },
            "taksim_square": {
                "typical_peaks": [(12, 14), (18, 21)],
                "weekend_multiplier": 1.8,
                "seasonal_variations": {
                    Season.SUMMER: 1.3,
                    Season.SPRING: 1.1,
                    Season.FALL: 1.0,
                    Season.WINTER: 0.9
                }
            },
            "bosphorus_ferry": {
                "typical_peaks": [(10, 12), (14, 16), (17, 19)],
                "weekend_multiplier": 1.4,
                "seasonal_variations": {
                    Season.SUMMER: 1.6,
                    Season.SPRING: 1.2,
                    Season.FALL: 1.0,
                    Season.WINTER: 0.5
                }
            }
        }
    
    async def predict_peak_times(
        self, 
        location: str, 
        date: Optional[datetime] = None,
        season_adjustment: Optional[SeasonalAdjustment] = None
    ) -> PeakTimePrediction:
        """Predict peak times for a specific location"""
        
        if not date:
            date = datetime.now()
        
        location_key = location.lower().replace(" ", "_").replace("'", "")
        
        if location_key not in self.attraction_patterns:
            return self._generate_generic_peak_prediction(location, date)
        
        pattern = self.attraction_patterns[location_key]
        
        # Get base peak hours
        peak_hours = pattern["typical_peaks"]
        
        # Adjust for weekend
        is_weekend = date.weekday() >= 5
        weekend_multiplier = pattern["weekend_multiplier"] if is_weekend else 1.0
        
        # Adjust for season
        season = self._determine_season(date)
        seasonal_multiplier = pattern["seasonal_variations"].get(season, 1.0)
        
        # Calculate crowd levels by hour
        crowd_by_hour = self._calculate_hourly_crowds(
            pattern, weekend_multiplier, seasonal_multiplier
        )
        
        # Calculate average wait times
        wait_times = self._calculate_wait_times(crowd_by_hour)
        
        # Determine best visit times
        best_times = self._find_best_visit_times(crowd_by_hour)
        
        return PeakTimePrediction(
            location=location,
            peak_hours=peak_hours,
            crowd_level_by_hour=crowd_by_hour,
            best_visit_times=best_times,
            average_wait_time=wait_times,
            seasonal_variations=pattern["seasonal_variations"]
        )
    
    def _calculate_hourly_crowds(
        self, 
        pattern: Dict, 
        weekend_multiplier: float, 
        seasonal_multiplier: float
    ) -> Dict[int, str]:
        """Calculate crowd levels for each hour"""
        
        # Base crowd levels
        base_crowds = pattern.get("crowd_by_hour", {})
        
        if not base_crowds:
            # Generate based on peak hours
            base_crowds = {}
            for hour in range(8, 20):
                is_peak = any(start <= hour <= end for start, end in pattern["typical_peaks"])
                if is_peak:
                    base_crowds[hour] = "high"
                elif any(start - 1 <= hour <= end + 1 for start, end in pattern["typical_peaks"]):
                    base_crowds[hour] = "moderate"
                else:
                    base_crowds[hour] = "low"
        
        # Apply multipliers
        adjusted_crowds = {}
        crowd_levels = ["low", "moderate", "high", "very_high"]
        
        for hour, base_level in base_crowds.items():
            base_index = crowd_levels.index(base_level)
            
            # Calculate adjustment
            total_multiplier = weekend_multiplier * seasonal_multiplier
            
            if total_multiplier > 1.3:
                new_index = min(3, base_index + 1)
            elif total_multiplier > 1.1:
                new_index = base_index
            else:
                new_index = max(0, base_index - 1)
            
            adjusted_crowds[hour] = crowd_levels[new_index]
        
        return adjusted_crowds
    
    def _calculate_wait_times(self, crowd_by_hour: Dict[int, str]) -> Dict[int, int]:
        """Calculate average wait times based on crowd levels"""
        wait_time_map = {
            "low": 2,
            "moderate": 8,
            "high": 20,
            "very_high": 35
        }
        
        return {hour: wait_time_map.get(level, 10) for hour, level in crowd_by_hour.items()}
    
    def _find_best_visit_times(self, crowd_by_hour: Dict[int, str]) -> List[str]:
        """Find the best times to visit based on crowd levels"""
        best_times = []
        
        for hour, level in crowd_by_hour.items():
            if level == "low":
                best_times.append(f"{hour:02d}:00 - Low crowds")
            elif level == "moderate":
                best_times.append(f"{hour:02d}:00 - Moderate crowds")
        
        if not best_times:
            # If no low/moderate times, find the least crowded
            min_crowd_hours = [hour for hour, level in crowd_by_hour.items() 
                             if level != "very_high"]
            for hour in min_crowd_hours[:3]:
                best_times.append(f"{hour:02d}:00 - Best available time")
        
        return best_times[:3]
    
    def _generate_generic_peak_prediction(self, location: str, date: datetime) -> PeakTimePrediction:
        """Generate generic peak prediction for unknown locations"""
        
        # Generic patterns for different location types
        if any(word in location.lower() for word in ["restaurant", "café", "food"]):
            peak_hours = [(12, 14), (19, 21)]  # Meal times
            crowd_by_hour = {
                12: "high", 13: "very_high", 14: "high",
                19: "high", 20: "very_high", 21: "moderate"
            }
        elif any(word in location.lower() for word in ["market", "bazaar", "shopping"]):
            peak_hours = [(11, 13), (15, 17)]
            crowd_by_hour = {
                11: "moderate", 12: "high", 13: "high",
                15: "high", 16: "very_high", 17: "moderate"
            }
        elif any(word in location.lower() for word in ["museum", "gallery", "cultural"]):
            peak_hours = [(10, 12), (14, 16)]
            crowd_by_hour = {
                10: "moderate", 11: "high", 12: "moderate",
                14: "high", 15: "very_high", 16: "moderate"
            }
        else:
            # Default pattern
            peak_hours = [(11, 13), (15, 17)]
            crowd_by_hour = {
                11: "moderate", 12: "high", 13: "moderate",
                15: "high", 16: "high", 17: "moderate"
            }
        
        wait_times = self._calculate_wait_times(crowd_by_hour)
        best_times = self._find_best_visit_times(crowd_by_hour)
        
        return PeakTimePrediction(
            location=location,
            peak_hours=peak_hours,
            crowd_level_by_hour=crowd_by_hour,
            best_visit_times=best_times,
            average_wait_time=wait_times,
            seasonal_variations={
                Season.SUMMER: 1.2,
                Season.SPRING: 1.0,
                Season.FALL: 0.9,
                Season.WINTER: 0.8
            }
        )
    
    def _determine_season(self, date: datetime) -> Season:
        """Determine season based on date"""
        month = date.month
        
        if month in [12, 1, 2]:
            return Season.WINTER
        elif month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        else:  # 9, 10, 11
            return Season.FALL

class PredictiveAnalyticsService:
    """Main service combining all predictive analytics capabilities"""
    
    def __init__(self):
        self.weather_predictor = WeatherBasedPredictor()
        self.seasonal_analyzer = SeasonalAnalyzer()
        self.peak_predictor = PeakTimePredictor()
    
    async def get_comprehensive_predictions(
        self,
        weather_data: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None,
        locations_of_interest: Optional[List[str]] = None,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive predictive analytics"""
        
        if not date:
            date = datetime.now()
        
        try:
            # Get weather-based predictions
            weather_prediction = await self.weather_predictor.predict_optimal_activities(
                weather_data, date, user_preferences
            )
            
            # Get seasonal adjustments
            seasonal_adjustment = self.seasonal_analyzer.get_seasonal_adjustment(date)
            
            # Adjust recommendations for season
            seasonally_adjusted_activities = self.seasonal_analyzer.adjust_recommendations_for_season(
                weather_prediction.recommended_activities, date
            )
            
            # Get peak time predictions for locations of interest
            peak_predictions = []
            if locations_of_interest:
                for location in locations_of_interest[:5]:  # Limit to 5 locations
                    try:
                        peak_pred = await self.peak_predictor.predict_peak_times(
                            location, date, seasonal_adjustment
                        )
                        peak_predictions.append(self._peak_prediction_to_dict(peak_pred))
                    except Exception as e:
                        logger.warning(f"Peak prediction failed for {location}: {e}")
            
            # Combine all predictions
            comprehensive_prediction = {
                "weather_prediction": {
                    "date": weather_prediction.date.isoformat(),
                    "temperature_range": weather_prediction.temperature_range,
                    "precipitation_chance": weather_prediction.precipitation_chance,
                    "recommended_activities": seasonally_adjusted_activities,
                    "avoid_activities": weather_prediction.avoid_activities,
                    "optimal_times": weather_prediction.optimal_times,
                    "clothing_suggestions": weather_prediction.clothing_suggestions
                },
                "seasonal_insights": {
                    "season": seasonal_adjustment.season.value,
                    "crowd_multiplier": seasonal_adjustment.crowd_multiplier,
                    "price_multiplier": seasonal_adjustment.price_multiplier,
                    "popular_activities": seasonal_adjustment.popular_activities,
                    "special_events": seasonal_adjustment.special_events,
                    "operating_hours_info": seasonal_adjustment.operating_hours_changes
                },
                "peak_time_predictions": peak_predictions,
                "summary_recommendations": self._generate_summary_recommendations(
                    weather_prediction, seasonal_adjustment, peak_predictions
                )
            }
            
            return comprehensive_prediction
            
        except Exception as e:
            logger.error(f"Comprehensive prediction failed: {e}")
            return self._fallback_prediction()
    
    def _peak_prediction_to_dict(self, peak_pred: PeakTimePrediction) -> Dict[str, Any]:
        """Convert PeakTimePrediction to dictionary"""
        return {
            "location": peak_pred.location,
            "peak_hours": [f"{start:02d}:00-{end:02d}:00" for start, end in peak_pred.peak_hours],
            "crowd_level_by_hour": peak_pred.crowd_level_by_hour,
            "best_visit_times": peak_pred.best_visit_times,
            "average_wait_time": peak_pred.average_wait_time,
            "seasonal_variations": {season.value: mult for season, mult in peak_pred.seasonal_variations.items()}
        }
    
    def _generate_summary_recommendations(
        self,
        weather_pred: WeatherPrediction,
        seasonal_adj: SeasonalAdjustment,
        peak_preds: List[Dict]
    ) -> List[str]:
        """Generate summary recommendations combining all predictions"""
        
        recommendations = []
        
        # Weather-based recommendation
        temp_range = weather_pred.temperature_range
        if weather_pred.precipitation_chance > 0.6:
            recommendations.append(
                f"High chance of rain today - focus on indoor activities like {weather_pred.recommended_activities[0] if weather_pred.recommended_activities else 'museums'}"
            )
        elif temp_range[1] > 28:
            recommendations.append(
                f"Hot day expected ({temp_range[1]}°C) - plan outdoor activities for {weather_pred.optimal_times[0] if weather_pred.optimal_times else 'early morning'}"
            )
        else:
            recommendations.append(
                f"Good weather for outdoor exploration - temperature range {temp_range[0]}-{temp_range[1]}°C"
            )
        
        # Seasonal recommendation
        season_name = seasonal_adj.season.value
        if seasonal_adj.crowd_multiplier > 1.1:
            recommendations.append(
                f"Peak {season_name} season - expect higher crowds and prices. Book popular attractions in advance."
            )
        elif seasonal_adj.crowd_multiplier < 0.8:
            recommendations.append(
                f"Off-season {season_name} travel - enjoy lower crowds and better prices!"
            )
        
        # Peak time recommendations
        if peak_preds:
            busy_locations = [pred["location"] for pred in peak_preds 
                            if any("very_high" in level for level in pred["crowd_level_by_hour"].values())]
            if busy_locations:
                recommendations.append(
                    f"High crowds expected at {', '.join(busy_locations[:2])} - consider visiting during off-peak hours"
                )
        
        # Activity timing recommendation
        if weather_pred.optimal_times:
            recommendations.append(
                f"Best activity times today: {', '.join(weather_pred.optimal_times[:2])}"
            )
        
        return recommendations[:4]  # Return top 4 recommendations
    
    def _fallback_prediction(self) -> Dict[str, Any]:
        """Fallback prediction when analysis fails"""
        return {
            "weather_prediction": {
                "recommended_activities": ["Explore historic Sultanahmet", "Visit Grand Bazaar", "Take Bosphorus ferry"],
                "optimal_times": ["Morning (9-11 AM)", "Afternoon (2-4 PM)"],
                "clothing_suggestions": ["Comfortable walking shoes", "Weather-appropriate layers"]
            },
            "seasonal_insights": {
                "season": "unknown",
                "popular_activities": ["Sightseeing", "Cultural visits", "Local cuisine"]
            },
            "peak_time_predictions": [],
            "summary_recommendations": [
                "Classic Istanbul sightseeing recommended",
                "Visit major attractions during moderate crowd times",
                "Try local Turkish cuisine"
            ]
        }

# Global predictive analytics service instance
predictive_analytics_service = PredictiveAnalyticsService()
