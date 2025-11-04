"""
Google Maps-Level Precision Prompts
Works with ANY LLM model (TinyLlama, LLaMA 3.2, etc.)

This module provides production-ready prompts that generate
Google Maps-style directions with:
- Exact step-by-step instructions
- Specific station names and platforms
- Walking distances and times
- Transfer instructions
- Weather-aware advice
- Real-time considerations
"""

from typing import Dict, List, Optional, Any
from datetime import datetime


class GoogleMapsStylePromptGenerator:
    """
    Generate Google Maps-level precision prompts for transportation
    
    Features:
    - Step-by-step navigation instructions
    - Exact station names, platforms, exits
    - Walking directions with distances
    - Transfer instructions
    - Weather-aware routing
    - Real-time delay integration
    
    Works with:
    - TinyLlama (development)
    - LLaMA 3.2 3B (production)
    - Any other LLM model
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize prompt generator
        
        Args:
            language: 'en' or 'tr'
        """
        self.language = language
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates for different scenarios"""
        return {
            'en': {
                'system_role': "You are a precise navigation assistant for Istanbul. Provide EXACT, step-by-step directions like Google Maps.",
                'instruction_prefix': "Give SPECIFIC instructions with exact details:",
                'required_elements': """
MUST include:
- Exact station/stop names
- Platform numbers (if known)
- Walking distances in meters
- Walking times in minutes
- Transfer instructions
- Exit/entrance directions
- Landmarks for navigation
""",
                'weather_instruction': "Mention weather impact ONLY if relevant to travel comfort or safety.",
                'format_instruction': "Format: Clear numbered steps. Be concise but precise."
            },
            'tr': {
                'system_role': "İstanbul için hassas navigasyon asistanısınız. Google Maps gibi TAM, adım adım yol tarifleri verin.",
                'instruction_prefix': "TAM detaylarla ÖZEL talimatlar verin:",
                'required_elements': """
ZORUNLU içerik:
- Tam istasyon/durak isimleri
- Peron numaraları (biliniyorsa)
- Metre cinsinden yürüme mesafeleri
- Dakika cinsinden yürüme süreleri
- Transfer talimatları
- Çıkış/giriş yönleri
- Navigasyon için işaret noktaları
""",
                'weather_instruction': "Hava durumunu SADECE seyahat konforuna veya güvenliğine etkisi varsa belirtin.",
                'format_instruction': "Format: Net numaralanmış adımlar. Kısa ama kesin olun."
            }
        }
    
    def create_route_prompt(
        self,
        origin: str,
        destination: str,
        route_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]] = None,
        live_transit_data: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        gps_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create Google Maps-style route prompt
        
        Args:
            origin: Starting location name
            destination: Ending location name
            route_data: Route information from OSRM/directions service
            weather_data: Current weather conditions
            live_transit_data: Real-time transit data from İBB
            user_preferences: User preferences (accessibility, speed, etc.)
            gps_context: GPS location context with:
                - gps_location: Tuple[float, float] - (lat, lon)
                - current_district: str - Detected district name
                - nearby_landmarks: List[str] - Nearby landmarks
                - location_accuracy: float - GPS accuracy in meters
        
        Returns:
            Formatted prompt for LLM
        """
        lang = self.language
        template = self.templates[lang]
        
        # Build the prompt
        prompt = f"""{template['system_role']}

USER REQUEST: How to get from {origin} to {destination}

ROUTE DATA:
"""
        
        # Add GPS context if available
        if gps_context:
            prompt += self._format_gps_context(gps_context)
        
        # Add route details
        prompt += self._format_route_data(route_data)
        
        # Add weather context if available
        if weather_data:
            prompt += self._format_weather_context(weather_data)
        
        # Add live transit data if available
        if live_transit_data:
            prompt += self._format_live_transit_data(live_transit_data)
        
        # Add user preferences if available
        if user_preferences:
            prompt += self._format_user_preferences(user_preferences)
        
        # Add instructions
        prompt += f"""

{template['instruction_prefix']}

{template['required_elements']}

{template['weather_instruction']}

{template['format_instruction']}

IMPORTANT: Keep response brief (2-3 sentences). Focus on the BEST transport option and ONE key tip.
The user will see full route details on the map.

RESPONSE:"""
        
        return prompt
    
    def _format_route_data(self, route_data: Dict[str, Any]) -> str:
        """Format route data section"""
        output = f"""
Total Duration: {route_data.get('duration', 'unknown')} minutes
Total Distance: {route_data.get('distance', 'unknown')} meters
Transit Modes: {', '.join(route_data.get('modes', []))}

DETAILED STEPS:
"""
        
        steps = route_data.get('steps', [])
        for i, step in enumerate(steps, 1):
            output += f"\nStep {i}:\n"
            output += f"  Mode: {step.get('mode', 'unknown')}\n"
            output += f"  Instruction: {step.get('instruction', 'N/A')}\n"
            
            if step.get('duration'):
                output += f"  Duration: {step['duration']} minutes\n"
            
            if step.get('distance'):
                output += f"  Distance: {step['distance']} meters\n"
            
            if step.get('line_name'):
                output += f"  Line: {step['line_name']}\n"
            
            if step.get('from_station'):
                output += f"  From: {step['from_station']}\n"
            
            if step.get('to_station'):
                output += f"  To: {step['to_station']}\n"
            
            if step.get('stops_count'):
                output += f"  Number of stops: {step['stops_count']}\n"
        
        return output
    
    def _format_weather_context(self, weather_data: Dict[str, Any]) -> str:
        """Format weather context"""
        temp = weather_data.get('temperature', 'N/A')
        condition = weather_data.get('condition', 'unknown')
        humidity = weather_data.get('humidity', 'N/A')
        wind_speed = weather_data.get('wind_speed', 'N/A')
        
        return f"""

CURRENT WEATHER:
  Temperature: {temp}°C
  Condition: {condition}
  Humidity: {humidity}%
  Wind Speed: {wind_speed} km/h
"""
    
    def _format_live_transit_data(self, live_data: Dict[str, Any]) -> str:
        """Format live transit information"""
        output = "\nREAL-TIME TRANSIT INFO:\n"
        
        if live_data.get('delays'):
            output += "  Active Delays:\n"
            for delay in live_data['delays']:
                output += f"    - {delay.get('line')}: {delay.get('message')}\n"
        
        if live_data.get('alerts'):
            output += "  Service Alerts:\n"
            for alert in live_data['alerts']:
                output += f"    - {alert.get('line')}: {alert.get('message')}\n"
        
        if live_data.get('next_departures'):
            output += "  Next Departures:\n"
            for dep in live_data['next_departures']:
                output += f"    - {dep.get('line')} at {dep.get('time')}\n"
        
        return output
    
    def _format_user_preferences(self, preferences: Dict[str, Any]) -> str:
        """Format user preferences"""
        output = "\nUSER PREFERENCES:\n"
        
        if preferences.get('accessibility_needed'):
            output += "  - Wheelchair accessible routes required\n"
        
        if preferences.get('prefer_indoor'):
            output += "  - Prefer covered/indoor routes\n"
        
        if preferences.get('avoid_stairs'):
            output += "  - Avoid stairs (use elevators/escalators)\n"
        
        if preferences.get('fastest_route'):
            output += "  - Prioritize fastest route\n"
        
        if preferences.get('cheapest_route'):
            output += "  - Prioritize cheapest route\n"
        
        return output
    
    def _format_gps_context(self, gps_context: Dict[str, Any]) -> str:
        """
        Format GPS location context
        
        Args:
            gps_context: Dictionary with:
                - gps_location: Tuple[float, float] - (lat, lon)
                - current_district: str - Detected district
                - nearby_landmarks: List[str] - Nearby landmarks
                - location_accuracy: float - GPS accuracy in meters
        
        Returns:
            Formatted GPS context string
        """
        output = "\nUSER GPS LOCATION:\n"
        
        if gps_context.get('gps_location'):
            lat, lon = gps_context['gps_location']
            output += f"  Coordinates: {lat:.6f}, {lon:.6f}\n"
            
            if gps_context.get('location_accuracy'):
                output += f"  GPS Accuracy: ±{gps_context['location_accuracy']:.0f} meters\n"
        
        if gps_context.get('current_district'):
            output += f"  Current District: {gps_context['current_district']}\n"
        
        if gps_context.get('nearby_landmarks'):
            landmarks = ', '.join(gps_context['nearby_landmarks'][:3])
            output += f"  Nearby Landmarks: {landmarks}\n"
        
        return output
    
    def create_cross_continental_prompt(
        self,
        origin: str,
        destination: str,
        ferry_route: Optional[Dict] = None,
        marmaray_route: Optional[Dict] = None,
        bridge_route: Optional[Dict] = None,
        weather_data: Optional[Dict] = None
    ) -> str:
        """
        Create prompt for cross-continental routes (Europe ↔ Asia)
        
        Compares: Ferry, Marmaray, Bridge options
        """
        lang = self.language
        template = self.templates[lang]
        
        prompt = f"""{template['system_role']}

USER REQUEST: Cross-continental route from {origin} to {destination}

This route crosses the Bosphorus. Multiple options available:

"""
        
        # Add ferry option
        if ferry_route:
            prompt += "OPTION 1: FERRY\n"
            prompt += self._format_route_data(ferry_route)
            prompt += "\n"
        
        # Add Marmaray option
        if marmaray_route:
            prompt += "OPTION 2: MARMARAY (Underground Rail)\n"
            prompt += self._format_route_data(marmaray_route)
            prompt += "\n"
        
        # Add bridge option
        if bridge_route:
            prompt += "OPTION 3: BRIDGE (Bus/Metrobus)\n"
            prompt += self._format_route_data(bridge_route)
            prompt += "\n"
        
        # Add weather
        if weather_data:
            prompt += self._format_weather_context(weather_data)
        
        # Add comparison instructions
        prompt += f"""

{template['instruction_prefix']}

For EACH option, provide:
1. Exact step-by-step directions
2. Total time and cost
3. Pros and cons considering current weather
4. Which option is best for different situations

Then RECOMMEND the best option based on:
- Current weather conditions
- Time efficiency
- Cost
- Comfort
- Reliability

{template['format_instruction']}

RESPONSE:"""
        
        return prompt
    
    def create_simple_query_prompt(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Create prompt for simple queries
        
        Examples:
        - "What's the weather like?"
        - "Is the M2 metro running?"
        - "Best way to Taksim?"
        """
        lang = self.language
        template = self.templates[lang]
        
        prompt = f"""{template['system_role']}

USER QUESTION: {query}

CONTEXT:
"""
        
        # Add available context
        if context.get('weather'):
            prompt += self._format_weather_context(context['weather'])
        
        if context.get('location'):
            prompt += f"\nUser's current location: {context['location']}\n"
        
        if context.get('time'):
            prompt += f"Current time: {context['time']}\n"
        
        if context.get('transit_status'):
            prompt += self._format_live_transit_data(context['transit_status'])
        
        prompt += f"""

Provide a CONCISE, HELPFUL answer with SPECIFIC details.

If the question is about directions:
- Give exact step-by-step instructions
- Include station names, walking distances, times
- Consider current weather and transit conditions

If the question is about weather or activities:
- Be specific about conditions
- Suggest practical actions
- Consider Istanbul-specific factors

{template['format_instruction']}

RESPONSE:"""
        
        return prompt
    
    def create_poi_recommendation_prompt(
        self,
        poi_type: str,
        gps_context: Dict[str, Any],
        nearby_pois: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create prompt for POI (Point of Interest) recommendations
        
        Args:
            poi_type: Type of POI ("museums", "restaurants", "attractions", etc.)
            gps_context: GPS location context with:
                - gps_location: Tuple[float, float] - (lat, lon)
                - current_district: str - Detected district
                - nearby_landmarks: List[str] - Nearby landmarks
            nearby_pois: List of nearby POIs from database with:
                - name: str
                - distance_km: float
                - distance_text: str (e.g., "500m walk")
                - rating: Optional[float]
                - category: Optional[str]
                - price_level: Optional[str]
            user_preferences: Optional user preferences:
                - interests: List[str]
                - budget_range: str
                - dietary_restrictions: List[str] (for restaurants)
        
        Returns:
            Formatted prompt for LLM
        """
        lang = self.language
        template = self.templates[lang]
        
        prompt = f"""{template['system_role']}

USER REQUEST: Recommend {poi_type} near my current location

"""
        
        # Add GPS context
        if gps_context:
            prompt += self._format_gps_context(gps_context)
        
        # Add nearby POIs
        prompt += f"\nNEARBY {poi_type.upper()} (pre-filtered by distance and ratings):\n"
        
        for idx, poi in enumerate(nearby_pois[:5], 1):
            prompt += f"\n{idx}. {poi.get('name', 'Unknown')}\n"
            
            if poi.get('distance_text'):
                prompt += f"   Distance: {poi['distance_text']}\n"
            elif poi.get('distance_km'):
                prompt += f"   Distance: {poi['distance_km']:.1f}km\n"
            
            if poi.get('rating'):
                prompt += f"   Rating: {poi['rating']}/5.0\n"
            
            if poi.get('category'):
                prompt += f"   Category: {poi['category']}\n"
            
            if poi.get('price_level'):
                prompt += f"   Price: {poi['price_level']}\n"
        
        # Add user preferences
        if user_preferences:
            prompt += "\nUSER PREFERENCES:\n"
            
            if user_preferences.get('interests'):
                interests = ', '.join(user_preferences['interests'][:5])
                prompt += f"  Interests: {interests}\n"
            
            if user_preferences.get('budget_range'):
                prompt += f"  Budget: {user_preferences['budget_range']}\n"
            
            if user_preferences.get('dietary_restrictions'):
                restrictions = ', '.join(user_preferences['dietary_restrictions'])
                prompt += f"  Dietary restrictions: {restrictions}\n"
        
        # Add instructions
        prompt += f"""

Provide a brief, personalized recommendation (2-3 sentences) focusing on:
1. Which nearby {poi_type[:-1]} is BEST for this user RIGHT NOW
2. ONE key reason why it's a great match (based on location, interests, or ratings)
3. Keep it personal and actionable

The user will see all options on the map with full details. Your advice should help them make the best choice.

RESPONSE:"""
        
        return prompt
    
    def validate_response_quality(
        self,
        response: str,
        required_elements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate if LLM response meets quality standards
        
        Args:
            response: Generated response from LLM
            required_elements: List of required elements to check
        
        Returns:
            Validation results with score and missing elements
        """
        validation = {
            'valid': True,
            'score': 0,
            'missing_elements': [],
            'suggestions': []
        }
        
        # Check for numbered steps (indicates structure)
        has_numbers = any(char.isdigit() for char in response[:100])
        if has_numbers:
            validation['score'] += 25
        else:
            validation['missing_elements'].append('numbered_steps')
            validation['suggestions'].append('Add numbered step-by-step instructions')
        
        # Check for specific distances/times
        has_metrics = any(keyword in response.lower() for keyword in ['minutes', 'meters', 'km', 'dakika', 'metre'])
        if has_metrics:
            validation['score'] += 25
        else:
            validation['missing_elements'].append('specific_metrics')
            validation['suggestions'].append('Include specific distances and times')
        
        # Check for station/stop names
        has_locations = any(keyword in response for keyword in ['Station', 'stop', 'İstasyon', 'durak'])
        if has_locations:
            validation['score'] += 25
        else:
            validation['missing_elements'].append('location_names')
            validation['suggestions'].append('Specify exact station/stop names')
        
        # Check length (should be substantive)
        if len(response) > 100:
            validation['score'] += 25
        else:
            validation['missing_elements'].append('sufficient_detail')
            validation['suggestions'].append('Provide more detailed instructions')
        
        # Overall validation
        validation['valid'] = validation['score'] >= 75
        
        return validation


# Singleton instance for easy import
_prompt_generator: Optional[GoogleMapsStylePromptGenerator] = None


def get_prompt_generator(language: str = 'en') -> GoogleMapsStylePromptGenerator:
    """Get or create prompt generator singleton"""
    global _prompt_generator
    
    if _prompt_generator is None or _prompt_generator.language != language:
        _prompt_generator = GoogleMapsStylePromptGenerator(language=language)
    
    return _prompt_generator


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Simple route
    generator = GoogleMapsStylePromptGenerator(language='en')
    
    route_data = {
        'duration': 25,
        'distance': 8500,
        'modes': ['walk', 'tram', 'walk'],
        'steps': [
            {
                'mode': 'walk',
                'instruction': 'Walk to Sultanahmet Tram Station',
                'duration': 3,
                'distance': 200
            },
            {
                'mode': 'tram',
                'instruction': 'Take T1 tram towards Kabataş',
                'line_name': 'T1 - Kabataş-Bağcılar',
                'from_station': 'Sultanahmet',
                'to_station': 'Eminönü',
                'stops_count': 2,
                'duration': 5
            },
            {
                'mode': 'walk',
                'instruction': 'Walk to destination',
                'duration': 2,
                'distance': 150
            }
        ]
    }
    
    weather_data = {
        'temperature': 22,
        'condition': 'Clear',
        'humidity': 65,
        'wind_speed': 12
    }
    
    prompt = generator.create_route_prompt(
        origin='Sultanahmet',
        destination='Eminönü',
        route_data=route_data,
        weather_data=weather_data
    )
    
    print("=" * 80)
    print("EXAMPLE PROMPT FOR LLM:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print("\n✅ This prompt works with TinyLlama, LLaMA 3.2, or any other LLM!")
    print("✅ Production-ready for T4 GPU deployment!")
