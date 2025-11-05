"""
Step 2.2: Context-Aware Prompt Engineering
Enhanced prompts that integrate weather, GPS, and real-time context

This module provides:
- Weather-aware prompts (temperature, conditions, impact on travel)
- GPS-aware prompts (user's exact location, nearby landmarks)
- Transportation-aware prompts (route optimization, alternatives)
- Real-time data integration (delays, traffic, events)
- Concise map-focused responses (2-3 sentences)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ContextAwarePromptEngine:
    """
    Enhanced prompt engine with full context awareness
    
    Features:
    - Weather impact analysis
    - GPS location context
    - Real-time transit data
    - Concise map-focused output (2-3 sentences)
    - Multi-modal route comparison
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize context-aware prompt engine
        
        Args:
            language: 'en' or 'tr'
        """
        self.language = language
    
    # ===== WEATHER-AWARE PROMPTS =====
    
    def create_weather_aware_query_prompt(
        self,
        query: str,
        weather_data: Dict[str, Any],
        user_location: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create weather-aware prompt for general queries
        
        Args:
            query: User's question
            weather_data: Current weather information
            user_location: Optional GPS location
            
        Returns:
            Weather-aware prompt string
        """
        temp = weather_data.get('temperature', 'N/A')
        conditions = weather_data.get('conditions', 'N/A')
        humidity = weather_data.get('humidity', 'N/A')
        wind_speed = weather_data.get('wind_speed', 'N/A')
        
        # Analyze weather impact
        weather_impact = self._analyze_weather_impact(weather_data)
        
        prompt = f"""You are Istanbul AI, a helpful guide for Istanbul, Turkey.

Current Weather in Istanbul:
- Temperature: {temp}°C
- Conditions: {conditions}
- Humidity: {humidity}%
- Wind Speed: {wind_speed} km/h
- Impact: {weather_impact}

User Question: {query}

Provide helpful advice considering the current weather conditions. Be specific about:
1. How weather affects the recommended activity
2. Practical tips (clothing, timing, alternatives)
3. Istanbul-specific local knowledge

Keep your response concise (2-3 sentences) and practical.

Response:"""
        
        return prompt
    
    def _analyze_weather_impact(self, weather_data: Dict[str, Any]) -> str:
        """Analyze weather impact on activities"""
        temp = weather_data.get('temperature', 20)
        conditions = weather_data.get('conditions', '').lower()
        
        if 'rain' in conditions or 'shower' in conditions:
            return "Rainy conditions - indoor activities or covered transportation recommended"
        elif 'snow' in conditions:
            return "Snowy conditions - delays possible, dress warmly"
        elif temp < 5:
            return "Cold weather - limited outdoor comfort, warm clothing essential"
        elif temp > 30:
            return "Hot weather - seek shade, stay hydrated, avoid midday activities"
        elif 'wind' in conditions:
            return "Windy conditions - ferries may be affected"
        else:
            return "Good weather for outdoor activities"
    
    # ===== TRANSPORTATION-AWARE PROMPTS =====
    
    def create_transportation_advice_prompt(
        self,
        origin: str,
        destination: str,
        route_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]] = None,
        gps_context: Optional[Dict[str, Any]] = None,
        live_transit_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create concise transportation advice prompt (map-focused)
        
        Note: The map shows the detailed route, so LLM provides:
        - Weather impact on this route
        - Marmaray/alternative suggestions
        - Key tips or warnings
        
        Args:
            origin: Starting point
            destination: Destination point
            route_data: Route information from OSRM
            weather_data: Current weather
            gps_context: User's GPS location and district
            live_transit_data: Real-time delays/issues
            
        Returns:
            Concise prompt for map-focused advice
        """
        duration = route_data.get('duration', 'unknown')
        distance = route_data.get('distance', 'unknown')
        modes = route_data.get('transit_types', [])
        
        # Build context sections
        context_parts = [
            f"You are an Istanbul transportation assistant.",
            f"The user will see a detailed map with the route visualization.",
            f"",
            f"Route: {origin} → {destination}",
            f"Duration: {duration} minutes",
            f"Distance: {distance} km",
            f"Transit: {', '.join(modes) if modes else 'Mixed'}",
        ]
        
        # Add weather context if available
        if weather_data:
            temp = weather_data.get('temperature', 'N/A')
            conditions = weather_data.get('conditions', 'N/A')
            context_parts.append(f"Weather: {temp}°C, {conditions}")
            
            # Check if weather impacts route
            weather_impact = self._get_weather_route_impact(weather_data, modes)
            if weather_impact:
                context_parts.append(f"Weather Impact: {weather_impact}")
        
        # Add GPS context if available
        if gps_context and gps_context.get('district'):
            district = gps_context['district']
            context_parts.append(f"User's Location: {district}")
        
        # Add live transit alerts if available
        if live_transit_data:
            delays = live_transit_data.get('delays', [])
            if delays:
                context_parts.append(f"Alerts: {', '.join(delays)}")
        
        # Add Marmaray check
        crosses_bosphorus = self._check_bosphorus_crossing(origin, destination)
        if crosses_bosphorus:
            context_parts.append("Note: Route crosses Bosphorus - Marmaray recommended")
        
        # Build final prompt
        prompt = "\n".join(context_parts) + """

Provide CONCISE advice (2-3 sentences maximum):
1. Why this route is good right now (considering weather/traffic)
2. Marmaray recommendation if crossing Bosphorus
3. One key practical tip or alternative

Keep it under 50 words. The map shows all route details.

Response:"""
        
        return prompt
    
    def _get_weather_route_impact(
        self,
        weather_data: Dict[str, Any],
        transit_modes: List[str]
    ) -> Optional[str]:
        """Determine if weather impacts this specific route"""
        conditions = weather_data.get('conditions', '').lower()
        
        if 'rain' in conditions or 'snow' in conditions:
            if 'ferry' in ' '.join(transit_modes).lower():
                return "Ferry may be delayed or uncomfortable"
            elif 'bus' in ' '.join(transit_modes).lower():
                return "Bus stops may be wet; consider metro/Marmaray"
        
        if 'wind' in conditions:
            if 'ferry' in ' '.join(transit_modes).lower():
                return "Ferry service may be affected by high winds"
        
        return None
    
    def _check_bosphorus_crossing(self, origin: str, destination: str) -> bool:
        """Check if route crosses Bosphorus (simplified)"""
        # European side indicators
        european = ['taksim', 'beyoğlu', 'beşiktaş', 'şişli', 'fatih', 
                   'sultanahmet', 'eminönü', 'galata', 'karaköy']
        
        # Asian side indicators
        asian = ['kadıköy', 'üsküdar', 'bostancı', 'maltepe', 'kartal',
                'ataşehir', 'ayrılık', 'kozyatağı']
        
        origin_lower = origin.lower()
        dest_lower = destination.lower()
        
        origin_european = any(loc in origin_lower for loc in european)
        origin_asian = any(loc in origin_lower for loc in asian)
        dest_european = any(loc in dest_lower for loc in european)
        dest_asian = any(loc in dest_lower for loc in asian)
        
        # Crosses if one is European and other is Asian
        return (origin_european and dest_asian) or (origin_asian and dest_european)
    
    # ===== GPS-AWARE PROMPTS =====
    
    def create_nearby_poi_prompt(
        self,
        query: str,
        gps_context: Dict[str, Any],
        poi_results: List[Dict[str, Any]],
        weather_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create GPS-aware POI recommendation prompt
        
        Args:
            query: User's query
            gps_context: User's GPS location with district info
            poi_results: Nearby points of interest from semantic search
            weather_data: Current weather
            
        Returns:
            GPS-aware POI recommendation prompt
        """
        district = gps_context.get('district', 'Istanbul')
        lat = gps_context.get('lat', 0)
        lng = gps_context.get('lng', 0)
        
        # Build POI list
        poi_list = []
        for i, poi in enumerate(poi_results[:5], 1):
            name = poi.get('name', 'Unknown')
            distance = poi.get('distance', 'N/A')
            poi_list.append(f"{i}. {name} ({distance}m away)")
        
        prompt = f"""You are Istanbul AI, helping a user find nearby places.

User's Location: {district} ({lat:.4f}, {lng:.4f})

Nearby Places:
{chr(10).join(poi_list)}

User Query: {query}
"""
        
        # Add weather context if available
        if weather_data:
            temp = weather_data.get('temperature', 'N/A')
            conditions = weather_data.get('conditions', 'N/A')
            prompt += f"\nCurrent Weather: {temp}°C, {conditions}"
        
        prompt += """

Provide a CONCISE recommendation (2-3 sentences):
1. Which place is best for them right now
2. Why it's a good choice (distance, weather, current conditions)
3. One practical tip (how to get there, what to bring)

Keep it under 50 words and mention specific distances.

Response:"""
        
        return prompt
    
    # ===== MARMARAY-AWARE PROMPTS =====
    
    def create_marmaray_comparison_prompt(
        self,
        origin: str,
        destination: str,
        regular_route: Dict[str, Any],
        marmaray_route: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create prompt comparing regular route with Marmaray option
        
        Args:
            origin: Starting point
            destination: Destination
            regular_route: Regular route data
            marmaray_route: Marmaray route data
            weather_data: Current weather
            
        Returns:
            Comparison prompt for Marmaray vs regular route
        """
        regular_time = regular_route.get('duration', 'unknown')
        marmaray_time = marmaray_route.get('duration', 'unknown')
        
        prompt = f"""You are an Istanbul transportation expert.

Route Request: {origin} → {destination}

Option 1 (Regular): {regular_time} minutes
Option 2 (via Marmaray): {marmaray_time} minutes
"""
        
        if weather_data:
            temp = weather_data.get('temperature', 'N/A')
            conditions = weather_data.get('conditions', 'N/A')
            prompt += f"\nWeather: {temp}°C, {conditions}"
        
        prompt += """

Marmaray Benefits:
- Weather-independent (underground)
- Avoids Bosphorus traffic
- Fixed schedule, reliable timing
- Fast underwater crossing

In 2-3 sentences, recommend which option is better considering:
1. Time difference
2. Weather conditions
3. Reliability and comfort

Be specific about which to choose and why.

Response:"""
        
        return prompt
    
    # ===== WEATHER-AWARE ACTIVITY PROMPTS =====
    
    def create_activity_recommendation_prompt(
        self,
        query: str,
        weather_data: Dict[str, Any],
        activities: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None,
        gps_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create weather-aware activity recommendation prompt
        
        Args:
            query: User's question about activities
            weather_data: Current weather and forecast
            activities: Top ranked activities with ML scores
            user_preferences: User preferences (indoor/outdoor, budget, etc.)
            gps_context: User's GPS location
            
        Returns:
            Weather-aware activity recommendation prompt
        """
        # Extract weather info
        temp = weather_data.get('temperature', 'N/A')
        conditions = weather_data.get('conditions', 'N/A')
        humidity = weather_data.get('humidity', 'N/A')
        
        # Analyze weather suitability
        weather_advice = self._get_weather_activity_advice(weather_data)
        
        prompt = f"""You are Istanbul AI, an expert guide helping visitors find the perfect activities.

Current Weather in Istanbul:
- Temperature: {temp}°C
- Conditions: {conditions}
- Humidity: {humidity}%
- Recommendation: {weather_advice}

User Question: {query}

Top Activity Recommendations (ML-ranked for weather suitability):
"""
        
        # Add top 3 activities with scores
        for i, activity in enumerate(activities[:3], 1):
            name = activity.get('name', 'Unknown')
            activity_type = activity.get('type', 'N/A')
            duration = activity.get('duration', 'N/A')
            weather_score = int(activity.get('ml_score', 0) * 100)
            comfort = activity.get('comfort_level', 0.5)
            
            prompt += f"\n{i}. {name} ({activity_type})"
            prompt += f"\n   - Duration: {duration}"
            prompt += f"\n   - Weather Match: {weather_score}%"
            prompt += f"\n   - Comfort Level: {self._comfort_to_text(comfort)}"
        
        # Add user preferences if available
        if user_preferences:
            indoor_outdoor = user_preferences.get('indoor_outdoor_pref')
            budget = user_preferences.get('budget_level')
            if indoor_outdoor:
                prompt += f"\n\nUser Preference: {indoor_outdoor.title()} activities"
            if budget:
                prompt += f"\nBudget: {budget.title()}"
        
        # Add GPS context if available
        if gps_context and gps_context.get('district'):
            district = gps_context['district']
            prompt += f"\nUser Location: {district}"
        
        prompt += """

Provide a CONCISE recommendation (2-3 sentences) that:
1. Recommends the BEST activity for current weather
2. Explains WHY it's perfect right now (weather + timing)
3. Adds ONE practical tip (what to bring, when to go, nearby options)

Focus on the weather context and be specific. Keep it under 60 words.

Response:"""
        
        return prompt
    
    def _get_weather_activity_advice(self, weather_data: Dict[str, Any]) -> str:
        """Get general weather advice for activities"""
        temp = weather_data.get('temperature', 20)
        conditions = weather_data.get('conditions', '').lower()
        
        if 'rain' in conditions or 'shower' in conditions:
            return "Indoor activities or covered venues recommended"
        elif 'snow' in conditions:
            return "Indoor cultural sites ideal; outdoor sites scenic but cold"
        elif temp < 5:
            return "Museums, covered bazaars, and indoor dining recommended"
        elif temp > 30:
            return "Morning/evening outdoor activities; midday indoor with A/C"
        elif 'wind' in conditions:
            return "Avoid hilltop viewpoints; ferries may be uncomfortable"
        elif 'clear' in conditions or 'sunny' in conditions:
            return "Perfect for outdoor sightseeing, parks, and ferry tours"
        else:
            return "Good conditions for most activities"
    
    def _comfort_to_text(self, comfort_level: float) -> str:
        """Convert comfort level to text"""
        if comfort_level > 0.8:
            return "Excellent"
        elif comfort_level > 0.6:
            return "Good"
        elif comfort_level > 0.4:
            return "Moderate"
        else:
            return "Challenging"
    
    def create_multi_day_weather_planning_prompt(
        self,
        activities: List[Dict[str, Any]],
        forecast: List[Dict[str, Any]],
        user_query: str
    ) -> str:
        """
        Create prompt for multi-day activity planning based on forecast
        
        Args:
            activities: List of activities to schedule
            forecast: 3-5 day weather forecast
            user_query: User's planning question
            
        Returns:
            Multi-day planning prompt
        """
        prompt = f"""You are Istanbul AI, helping plan activities across multiple days.

User Request: {user_query}

Weather Forecast:
"""
        
        # Add forecast
        for i, day in enumerate(forecast[:5], 1):
            date = day.get('date', f'Day {i}')
            temp_high = day.get('temp_high', 'N/A')
            temp_low = day.get('temp_low', 'N/A')
            conditions = day.get('condition', 'N/A')
            prompt += f"\n{date}: {temp_high}°C/{temp_low}°C, {conditions}"
        
        prompt += "\n\nActivities to Schedule:\n"
        
        # Add activities
        for i, activity in enumerate(activities, 1):
            name = activity.get('name', 'Unknown')
            activity_type = activity.get('type', 'N/A')
            weather_suit = activity.get('weather_suitability', {})
            prompt += f"\n{i}. {name} ({activity_type})"
        
        prompt += """

In 3-4 sentences, suggest:
1. Which activities to do on which days based on weather
2. Which day is BEST for outdoor activities
3. Indoor activity suggestions for poor weather days

Be specific about matching activities to forecast days.

Response:"""
        
        return prompt
    
    # ===== RESPONSE POST-PROCESSING =====
    
    @staticmethod
    def truncate_to_sentences(response: str, max_sentences: int = 3) -> str:
        """
        Ensure response is concise (max N sentences)
        
        Args:
            response: LLM generated response
            max_sentences: Maximum number of sentences
            
        Returns:
            Truncated response
        """
        # Split by periods
        sentences = response.split('.')
        
        # Take first N sentences
        truncated = '. '.join(sentences[:max_sentences]).strip()
        
        # Add period if missing
        if truncated and not truncated.endswith('.'):
            truncated += '.'
        
        return truncated
    
    @staticmethod
    def validate_response_length(response: str, max_words: int = 60) -> bool:
        """
        Validate response is appropriately concise
        
        Args:
            response: Generated response
            max_words: Maximum word count
            
        Returns:
            True if response is within limit
        """
        word_count = len(response.split())
        return word_count <= max_words


# ===== HELPER FUNCTIONS =====

def get_prompt_engine(language: str = 'en') -> ContextAwarePromptEngine:
    """
    Get context-aware prompt engine instance
    
    Args:
        language: 'en' or 'tr'
        
    Returns:
        ContextAwarePromptEngine instance
    """
    return ContextAwarePromptEngine(language=language)


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example: Weather-aware query
    engine = ContextAwarePromptEngine()
    
    weather = {
        'temperature': 15,
        'conditions': 'Light rain',
        'humidity': 75,
        'wind_speed': 12
    }
    
    prompt = engine.create_weather_aware_query_prompt(
        query="Should I visit the Blue Mosque today?",
        weather_data=weather
    )
    
    print("Weather-Aware Prompt:")
    print(prompt)
    print("\n" + "="*60 + "\n")
    
    # Example: Transportation advice
    route_data = {
        'duration': 35,
        'distance': 18,
        'transit_types': ['Metro', 'Marmaray', 'Metro']
    }
    
    gps_context = {
        'lat': 41.0370,
        'lng': 28.9857,
        'district': 'Taksim'
    }
    
    prompt = engine.create_transportation_advice_prompt(
        origin="Taksim",
        destination="Kadıköy",
        route_data=route_data,
        weather_data=weather,
        gps_context=gps_context
    )
    
    print("Transportation Advice Prompt:")
    print(prompt)
    print("\n" + "="*60 + "\n")
    
    # Example: Marmaray comparison
    regular_route = {
        'duration': 45,
        'distance': 20,
        'transit_types': ['Bus', 'Ferry', 'Metro']
    }
    
    marmaray_route = {
        'duration': 30,
        'distance': 15,
        'transit_types': ['Marmaray']
    }
    
    prompt = engine.create_marmaray_comparison_prompt(
        origin="Taksim",
        destination="Kadıköy",
        regular_route=regular_route,
        marmaray_route=marmaray_route,
        weather_data=weather
    )
    
    print("Marmaray Comparison Prompt:")
    print(prompt)
    print("\n" + "="*60 + "\n")
    
    # Example: Activity recommendation
    activities = [
        {'name': 'Hagia Sophia', 'type': 'Museum', 'duration': '1-2 hours', 'ml_score': 0.9, 'comfort_level': 0.8},
        {'name': 'Basilica Cistern', 'type': 'Historical Site', 'duration': '30 minutes', 'ml_score': 0.85, 'comfort_level': 0.7},
        {'name': 'Topkapi Palace', 'type': 'Museum', 'duration': '2-3 hours', 'ml_score': 0.8, 'comfort_level': 0.6},
        {'name': 'Grand Bazaar', 'type': 'Shopping', 'duration': '1-2 hours', 'ml_score': 0.75, 'comfort_level': 0.5},
        {'name': 'Spice Bazaar', 'type': 'Market', 'duration': '30 minutes', 'ml_score': 0.7, 'comfort_level': 0.4},
    ]
    
    prompt = engine.create_activity_recommendation_prompt(
        query="What indoor activities do you suggest for a rainy day?",
        weather_data=weather,
        activities=activities,
        user_preferences={'indoor_outdoor_pref': 'indoor', 'budget_level': 'mid'},
        gps_context=gps_context
    )
    
    print("Activity Recommendation Prompt:")
    print(prompt)
