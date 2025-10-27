"""
ML-Enhanced Weather Handler
Provides weather-aware activity recommendations with neural ranking
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class WeatherContext:
    """Context for weather-based recommendations"""
    user_query: str
    current_weather: Dict[str, Any]  # temp, condition, humidity, wind, etc.
    forecast: Optional[List[Dict[str, Any]]]  # multi-day forecast
    temperature_range: str  # 'hot', 'warm', 'mild', 'cool', 'cold'
    weather_condition: str  # 'clear', 'cloudy', 'rainy', 'snowy', etc.
    activity_preferences: List[str]
    indoor_outdoor_pref: Optional[str]
    comfort_sensitivity: float  # 0.0-1.0, how weather-sensitive is user
    interests: List[str]
    time_of_day: Optional[str]
    budget_level: Optional[str]
    user_sentiment: float


class MLEnhancedWeatherHandler:
    """
    ML-Enhanced Weather Handler
    
    Features:
    - Real-time weather integration
    - Temperature-appropriate activity recommendations
    - Condition-based filtering (rain, heat, cold)
    - Neural ranking of weather-appropriate activities
    - Indoor/outdoor smart routing
    - Comfort-level personalization
    - Multi-day forecast planning
    """
    
    def __init__(self, weather_service, weather_recommendations_service, 
                 ml_context_builder, ml_processor, response_generator):
        """
        Initialize handler with required services
        
        Args:
            weather_service: Current weather and forecast service
            weather_recommendations_service: Weather-aware recommendations service
            ml_context_builder: Centralized ML context builder
            ml_processor: Neural processor for embeddings and ranking
            response_generator: Response generator for natural language output
        """
        self.weather_service = weather_service
        self.weather_recommendations_service = weather_recommendations_service
        self.ml_context_builder = ml_context_builder
        self.ml_processor = ml_processor
        self.response_generator = response_generator
        
        logger.info("✅ ML-Enhanced Weather Handler initialized")
    
    async def handle_weather_query(
        self,
        user_query: str,
        user_profile: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle weather-based activity query with ML enhancement
        
        Args:
            user_query: User's natural language query
            user_profile: Optional user profile for personalization
            context: Optional additional context
        
        Returns:
            Dict with activities, weather info, and natural language response
        """
        try:
            # Step 1: Get current weather and forecast
            weather_data = await self._get_weather_data(context)
            
            # Step 2: Extract ML context (including weather)
            ml_context = await self.ml_context_builder.build_context(
                query=user_query,
                intent="weather_activities",
                user_profile=user_profile,
                additional_context={
                    **(context or {}),
                    "weather": weather_data
                }
            )
            
            # Step 3: Build weather-specific context
            weather_context = self._build_weather_context(
                ml_context,
                weather_data,
                context
            )
            
            # Step 4: Get candidate activities (weather-appropriate)
            candidates = await self._get_candidate_activities(weather_context)
            
            # Step 5: Neural ranking with weather appropriateness
            ranked_activities = await self._rank_activities_neural(
                activities=candidates,
                context=weather_context,
                ml_context=ml_context
            )
            
            # Step 6: Apply filters (extreme weather, comfort, indoor/outdoor)
            filtered_activities = self._apply_filters(
                ranked_activities,
                weather_context
            )
            
            # Step 7: Generate personalized response
            response = await self._generate_response(
                activities=filtered_activities[:5],
                weather_context=weather_context,
                ml_context=ml_context
            )
            
            return {
                "success": True,
                "activities": filtered_activities[:5],
                "weather": weather_data,
                "response": response,
                "context_used": {
                    "temperature_range": weather_context.temperature_range,
                    "weather_condition": weather_context.weather_condition,
                    "indoor_outdoor": weather_context.indoor_outdoor_pref,
                    "comfort_sensitivity": weather_context.comfort_sensitivity
                }
            }
            
        except Exception as e:
            logger.error(f"Error in weather handler: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I'm having trouble getting weather information. Let me know what you'd like to do and I can suggest activities!"
            }
    
    async def _get_weather_data(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get current weather and forecast"""
        
        try:
            # Try to get real weather data
            current = await self.weather_service.get_current_weather("Istanbul")
            forecast = await self.weather_service.get_forecast("Istanbul", days=3)
            
            return {
                "current": current,
                "forecast": forecast
            }
            
        except Exception as e:
            logger.warning(f"Weather service error: {e}, using mock data")
            # Return mock weather data
            return self._get_mock_weather()
    
    def _get_mock_weather(self) -> Dict[str, Any]:
        """Return mock weather data for development"""
        
        return {
            "current": {
                "temperature": 18,
                "feels_like": 16,
                "condition": "partly_cloudy",
                "humidity": 65,
                "wind_speed": 15,
                "description": "Partly cloudy with light breeze"
            },
            "forecast": [
                {"date": "today", "temp_high": 20, "temp_low": 14, "condition": "partly_cloudy"},
                {"date": "tomorrow", "temp_high": 22, "temp_low": 16, "condition": "clear"},
                {"date": "day_after", "temp_high": 19, "temp_low": 15, "condition": "rainy"}
            ]
        }
    
    def _build_weather_context(
        self,
        ml_context: Dict[str, Any],
        weather_data: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]]
    ) -> WeatherContext:
        """Build weather-specific context"""
        
        query_lower = ml_context.get("original_query", "").lower()
        current = weather_data.get("current", {})
        
        # Determine temperature range
        temp = current.get("temperature", 20)
        if temp >= 30:
            temp_range = "hot"
        elif temp >= 25:
            temp_range = "warm"
        elif temp >= 15:
            temp_range = "mild"
        elif temp >= 5:
            temp_range = "cool"
        else:
            temp_range = "cold"
        
        # Get weather condition
        weather_condition = current.get("condition", "clear")
        
        # Extract activity preferences
        activity_preferences = []
        activity_keywords = {
            "outdoor": ["outdoor", "outside", "fresh air", "nature"],
            "cultural": ["museum", "gallery", "historical", "culture"],
            "food": ["food", "restaurant", "eat", "dining"],
            "shopping": ["shopping", "bazaar", "market", "mall"],
            "scenic": ["view", "scenic", "panorama", "photo"],
            "active": ["walk", "hike", "exercise", "active"],
            "relaxing": ["relax", "calm", "peaceful", "chill"]
        }
        
        for activity, keywords in activity_keywords.items():
            if any(kw in query_lower for kw in keywords):
                activity_preferences.append(activity)
        
        # Determine indoor/outdoor preference
        indoor_outdoor_pref = None
        if "indoor" in query_lower or "inside" in query_lower:
            indoor_outdoor_pref = "indoor"
        elif "outdoor" in query_lower or "outside" in query_lower:
            indoor_outdoor_pref = "outdoor"
        else:
            # Infer from weather
            if weather_condition in ["rainy", "snowy", "stormy"]:
                indoor_outdoor_pref = "indoor"
            elif weather_condition in ["clear", "sunny"]:
                indoor_outdoor_pref = "outdoor"
        
        # Comfort sensitivity (how much weather affects choice)
        comfort_sensitivity = 0.7  # Default
        if any(kw in query_lower for kw in ["any weather", "don't care", "doesn't matter"]):
            comfort_sensitivity = 0.3
        elif any(kw in query_lower for kw in ["comfortable", "pleasant", "nice weather"]):
            comfort_sensitivity = 0.9
        
        # Time of day
        time_of_day = None
        if any(kw in query_lower for kw in ["morning", "early"]):
            time_of_day = "morning"
        elif any(kw in query_lower for kw in ["afternoon", "lunch"]):
            time_of_day = "afternoon"
        elif any(kw in query_lower for kw in ["evening", "sunset"]):
            time_of_day = "evening"
        
        return WeatherContext(
            user_query=ml_context.get("original_query", ""),
            current_weather=current,
            forecast=weather_data.get("forecast"),
            temperature_range=temp_range,
            weather_condition=weather_condition,
            activity_preferences=activity_preferences,
            indoor_outdoor_pref=indoor_outdoor_pref,
            comfort_sensitivity=comfort_sensitivity,
            interests=ml_context.get("detected_interests", []),
            time_of_day=time_of_day,
            budget_level=ml_context.get("budget_preference"),
            user_sentiment=ml_context.get("sentiment_score", 0.0)
        )
    
    async def _get_candidate_activities(
        self,
        context: WeatherContext
    ) -> List[Dict[str, Any]]:
        """Get weather-appropriate activities"""
        
        try:
            # Get activities from recommendations service
            activities = await self.weather_recommendations_service.get_activities_for_weather(
                temperature_range=context.temperature_range,
                condition=context.weather_condition
            )
            
            return activities
            
        except Exception as e:
            logger.warning(f"Error fetching activities: {e}, using mock data")
            return self._get_mock_activities(context)
    
    def _get_mock_activities(self, context: WeatherContext) -> List[Dict[str, Any]]:
        """Return mock activities based on weather"""
        
        # Base activities for mild, partly cloudy weather
        return [
            {
                "id": "act_001",
                "name": "Bosphorus Ferry Tour",
                "type": "outdoor",
                "description": "Scenic ferry ride with city views, perfect for mild weather",
                "weather_suitability": {
                    "mild": 0.95,
                    "warm": 0.90,
                    "cool": 0.70,
                    "clear": 0.95,
                    "partly_cloudy": 0.85,
                    "rainy": 0.30
                },
                "duration": "2-3 hours",
                "budget": "budget",
                "comfort_level": 0.8,
                "highlights": ["Fresh air", "Views", "Covered seating available"]
            },
            {
                "id": "act_002",
                "name": "Istanbul Modern Art Museum",
                "type": "indoor",
                "description": "Contemporary art museum with Bosphorus views and cafe",
                "weather_suitability": {
                    "hot": 0.90,
                    "cold": 0.90,
                    "rainy": 0.95,
                    "snowy": 0.95
                },
                "duration": "2-4 hours",
                "budget": "moderate",
                "comfort_level": 1.0,
                "highlights": ["Climate controlled", "Restaurant", "Art"]
            },
            {
                "id": "act_003",
                "name": "Gülhane Park Walking Tour",
                "type": "outdoor",
                "description": "Historic park with gardens, perfect for pleasant weather",
                "weather_suitability": {
                    "mild": 1.0,
                    "warm": 0.85,
                    "cool": 0.75,
                    "clear": 0.95,
                    "partly_cloudy": 0.90,
                    "rainy": 0.20
                },
                "duration": "1-2 hours",
                "budget": "free",
                "comfort_level": 0.7,
                "highlights": ["Free", "Nature", "Historical sites nearby"]
            },
            {
                "id": "act_004",
                "name": "Grand Bazaar Shopping",
                "type": "indoor",
                "description": "Covered historic market, great for any weather",
                "weather_suitability": {
                    "hot": 0.80,
                    "cold": 0.85,
                    "rainy": 0.95,
                    "mild": 0.75
                },
                "duration": "2-4 hours",
                "budget": "varies",
                "comfort_level": 0.85,
                "highlights": ["Covered", "Shopping", "Historical"]
            },
            {
                "id": "act_005",
                "name": "Çamlıca Hill Sunset",
                "type": "outdoor",
                "description": "Highest point in Istanbul with panoramic views",
                "weather_suitability": {
                    "mild": 0.95,
                    "warm": 0.90,
                    "cool": 0.70,
                    "clear": 1.0,
                    "partly_cloudy": 0.85,
                    "rainy": 0.10
                },
                "duration": "2-3 hours",
                "budget": "free",
                "comfort_level": 0.6,
                "highlights": ["Free", "Best views", "Cafe available"]
            },
            {
                "id": "act_006",
                "name": "Cooking Class at Local Home",
                "type": "indoor",
                "description": "Learn Turkish cooking in a local's apartment",
                "weather_suitability": {
                    "hot": 0.85,
                    "cold": 0.90,
                    "rainy": 1.0,
                    "snowy": 0.95
                },
                "duration": "3-4 hours",
                "budget": "moderate",
                "comfort_level": 0.95,
                "highlights": ["Indoor", "Cultural", "Food included"]
            }
        ]
    
    async def _rank_activities_neural(
        self,
        activities: List[Dict[str, Any]],
        context: WeatherContext,
        ml_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank activities using neural similarity + weather appropriateness"""
        
        # Get query embedding
        query_embedding = await self.ml_processor.get_embedding(context.user_query)
        
        scored_activities = []
        for activity in activities:
            # Create activity description
            desc = f"{activity['name']} {activity.get('type', '')} {activity.get('description', '')}"
            
            # Get activity embedding
            activity_embedding = await self.ml_processor.get_embedding(desc)
            
            # Calculate base similarity
            base_score = self.ml_processor.calculate_similarity(
                query_embedding,
                activity_embedding
            )
            
            # Apply weather and context adjustments
            adjusted_score = self._adjust_score_with_weather(
                base_score,
                activity,
                context
            )
            
            scored_activities.append({
                **activity,
                "ml_score": adjusted_score,
                "base_similarity": base_score
            })
        
        # Sort by ML score
        scored_activities.sort(key=lambda x: x["ml_score"], reverse=True)
        
        return scored_activities
    
    def _adjust_score_with_weather(
        self,
        base_score: float,
        activity: Dict[str, Any],
        context: WeatherContext
    ) -> float:
        """Adjust score heavily based on weather appropriateness"""
        
        score = base_score
        boost = 0.0
        
        # WEATHER SUITABILITY BOOST (most important!)
        suitability = activity.get("weather_suitability", {})
        
        # Temperature suitability
        temp_suit = suitability.get(context.temperature_range, 0.5)
        if temp_suit > 0.8:
            boost += 0.40
        elif temp_suit > 0.6:
            boost += 0.20
        elif temp_suit < 0.3:
            boost -= 0.30  # Penalty for poor fit
        
        # Condition suitability
        cond_suit = suitability.get(context.weather_condition, 0.5)
        if cond_suit > 0.8:
            boost += 0.35
        elif cond_suit > 0.6:
            boost += 0.15
        elif cond_suit < 0.3:
            boost -= 0.25
        
        # INDOOR/OUTDOOR MATCH
        if context.indoor_outdoor_pref:
            activity_type = activity.get("type", "")
            if context.indoor_outdoor_pref == activity_type:
                boost += 0.25
        
        # COMFORT LEVEL (weighted by user sensitivity)
        comfort = activity.get("comfort_level", 0.5)
        comfort_boost = (comfort - 0.5) * context.comfort_sensitivity * 0.3
        boost += comfort_boost
        
        # ACTIVITY PREFERENCE MATCH
        activity_type = activity.get("type", "")
        for pref in context.activity_preferences:
            if pref in activity_type or pref in activity.get("description", "").lower():
                boost += 0.15
        
        # BUDGET MATCH
        if context.budget_level:
            activity_budget = activity.get("budget", "")
            if activity_budget == context.budget_level or activity_budget == "free":
                boost += 0.10
        
        # TIME OF DAY MATCH
        if context.time_of_day:
            # Some activities are better at certain times
            activity_name = activity.get("name", "").lower()
            if context.time_of_day == "morning" and any(kw in activity_name for kw in ["park", "breakfast", "market"]):
                boost += 0.10
            elif context.time_of_day == "evening" and any(kw in activity_name for kw in ["sunset", "dinner", "night"]):
                boost += 0.10
        
        # EXTREME WEATHER SAFETY
        if context.weather_condition in ["stormy", "heavy_rain", "snowy"]:
            if activity.get("type") == "outdoor":
                boost -= 0.40  # Strong penalty for outdoor in extreme weather
            else:
                boost += 0.20  # Bonus for indoor in extreme weather
        
        final_score = score * (1 + boost)
        return max(min(final_score, 1.0), 0.0)  # Clamp between 0 and 1
    
    def _apply_filters(
        self,
        activities: List[Dict[str, Any]],
        context: WeatherContext
    ) -> List[Dict[str, Any]]:
        """Apply hard filters for safety and comfort"""
        
        filtered = activities
        
        # Filter out dangerous activities in extreme weather
        if context.weather_condition in ["stormy", "heavy_rain", "blizzard"]:
            filtered = [
                a for a in filtered
                if a.get("type") != "outdoor" or a.get("comfort_level", 0) > 0.8
            ]
        
        # Filter by comfort sensitivity
        if context.comfort_sensitivity > 0.8:
            filtered = [
                a for a in filtered
                if a.get("comfort_level", 0) > 0.6
            ]
        
        return filtered
    
    async def _generate_response(
        self,
        activities: List[Dict[str, Any]],
        weather_context: WeatherContext,
        ml_context: Dict[str, Any]
    ) -> str:
        """Generate weather-aware response"""
        
        if not activities:
            return "Given the current weather, I'm having trouble finding suitable activities. Would you like indoor or outdoor suggestions?"
        
        response_parts = []
        
        # Weather summary
        current = weather_context.current_weather
        temp = current.get("temperature", "N/A")
        condition = current.get("description", "Current weather")
        
        response_parts.append(f"🌤️ **Current Weather in Istanbul:**")
        response_parts.append(f"   Temperature: {temp}°C, {condition}")
        
        # Weather-appropriate intro
        if weather_context.weather_condition == "rainy":
            response_parts.append("\n☔ Perfect indoor activities for rainy weather:")
        elif weather_context.temperature_range == "hot":
            response_parts.append("\n☀️ Beat the heat with these activities:")
        elif weather_context.weather_condition == "clear":
            response_parts.append("\n✨ Great weather! Here are the best outdoor options:")
        else:
            response_parts.append("\n🎯 Here are the best activities for current conditions:")
        
        # Top activity
        top = activities[0]
        response_parts.append(f"\n\n🌟 **{top['name']}** (Match: {int(top['ml_score']*100)}%)")
        response_parts.append(f"   {top.get('description', '')}")
        response_parts.append(f"   ⏱️ Duration: {top.get('duration', 'Flexible')}")
        response_parts.append(f"   💰 Cost: {top.get('budget', 'Varies').title()}")
        
        # Weather appropriateness
        comfort = top.get('comfort_level', 0.5)
        if comfort > 0.8:
            response_parts.append(f"   ☀️ Weather comfort: Excellent")
        elif comfort > 0.6:
            response_parts.append(f"   ☀️ Weather comfort: Good")
        
        # Highlights
        if "highlights" in top and top["highlights"]:
            highlights = ", ".join(top["highlights"][:3])
            response_parts.append(f"   ✨ {highlights}")
        
        # More activities
        if len(activities) > 1:
            response_parts.append("\n\n📋 **More weather-appropriate activities:**")
            for activity in activities[1:4]:
                response_parts.append(
                    f"   • **{activity['name']}** ({activity.get('type', '').title()}) - {activity.get('duration', 'Flexible')}"
                )
        
        # Forecast tip
        if weather_context.forecast:
            tomorrow = weather_context.forecast[0] if weather_context.forecast else None
            if tomorrow:
                tom_cond = tomorrow.get("condition", "")
                tom_temp = tomorrow.get("temp_high", "")
                if tom_cond != weather_context.weather_condition:
                    response_parts.append(f"\n\n🔮 Tomorrow's forecast: {tom_temp}°C, {tom_cond}")
                    if tom_cond == "clear" and weather_context.weather_condition == "rainy":
                        response_parts.append("   Consider outdoor activities tomorrow!")
        
        # Safety tip for extreme weather
        if weather_context.weather_condition in ["rainy", "stormy"]:
            response_parts.append("\n\n☔ Weather tip: Bring an umbrella and wear waterproof shoes!")
        elif weather_context.temperature_range == "hot":
            response_parts.append("\n\n☀️ Weather tip: Stay hydrated and use sunscreen!")
        elif weather_context.temperature_range == "cold":
            response_parts.append("\n\n🧥 Weather tip: Dress warmly in layers!")
        
        return "\n".join(response_parts)


def create_ml_enhanced_weather_handler(
    weather_service,
    weather_recommendations_service,
    ml_context_builder,
    ml_processor,
    response_generator
):
    """Factory function to create ML-enhanced weather handler"""
    return MLEnhancedWeatherHandler(
        weather_service=weather_service,
        weather_recommendations_service=weather_recommendations_service,
        ml_context_builder=ml_context_builder,
        ml_processor=ml_processor,
        response_generator=response_generator
    )
