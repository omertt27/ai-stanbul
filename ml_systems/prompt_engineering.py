"""
Prompt Engineering for Istanbul AI
Weather-aware and transportation-aware prompt generation
"""
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class IstanbulPromptGenerator:
    """
    Generate context-aware prompts for Istanbul AI
    
    Features:
    - Weather-aware recommendations
    - Transportation-aware routing advice
    - Marmaray integration
    - Live transit data integration
    - Bilingual support (EN/TR)
    """
    
    def __init__(self):
        self.system_context = """You are KAM, a friendly and knowledgeable Istanbul AI assistant.
You provide practical, accurate advice about Istanbul, including:
- Transportation (metro, tram, bus, ferry, Marmaray)
- Weather-appropriate recommendations
- Tourist attractions
- Local tips and cultural insights

Always be helpful, concise, and specific to Istanbul."""
    
    def create_weather_aware_prompt(
        self,
        query: str,
        weather_data: Dict,
        language: str = 'en'
    ) -> str:
        """
        Create prompt with weather context
        
        Args:
            query: User's query
            weather_data: Current weather information
            language: Response language ('en' or 'tr')
            
        Returns:
            Formatted prompt with weather context
        """
        temp = weather_data.get('temperature', 'N/A')
        conditions = weather_data.get('conditions', 'N/A')
        humidity = weather_data.get('humidity', 'N/A')
        wind_speed = weather_data.get('wind_speed', 'N/A')
        
        # Weather assessment
        weather_advice = self._get_weather_advice(
            temp, conditions, humidity, wind_speed
        )
        
        prompt = f"""{self.system_context}

Current Istanbul Weather:
- Temperature: {temp}°C
- Conditions: {conditions}
- Humidity: {humidity}%
- Wind: {wind_speed} km/h

Weather Assessment:
{weather_advice}

User Query: {query}

Provide helpful advice considering the current weather. Be specific about:
- What to wear
- Best activities for this weather
- Transportation recommendations
- Any weather-related warnings

"""
        
        if language == 'tr':
            prompt += "Respond in TURKISH with a friendly, helpful tone.\n"
        else:
            prompt += "Respond in ENGLISH with a friendly, helpful tone.\n"
        
        prompt += "\nResponse:"
        
        return prompt
    
    def create_transportation_prompt(
        self,
        query: str,
        route_data: Dict,
        weather_data: Optional[Dict] = None,
        live_transit_data: Optional[Dict] = None,
        marmaray_option: Optional[Dict] = None,
        language: str = 'en'
    ) -> str:
        """
        Create prompt with transportation and weather context
        
        Args:
            query: User's query
            route_data: Route information
            weather_data: Current weather (optional)
            live_transit_data: Real-time transit info (optional)
            marmaray_option: Marmaray route option (optional)
            language: Response language
            
        Returns:
            Formatted prompt with all context
        """
        # Build route info
        duration = route_data.get('duration', 'unknown')
        distance = route_data.get('distance', 'unknown')
        modes = route_data.get('modes', [])
        steps = route_data.get('steps', [])
        
        prompt = f"""{self.system_context}

Route Request: {query}

Main Route:
- From: {route_data.get('origin', 'N/A')}
- To: {route_data.get('destination', 'N/A')}
- Duration: {duration} minutes
- Distance: {distance} km
- Transit modes: {', '.join(modes)}

Steps:
"""
        
        for i, step in enumerate(steps[:5], 1):  # Limit to 5 steps
            prompt += f"{i}. {step.get('instruction', 'N/A')}"
            if step.get('duration'):
                prompt += f" ({step['duration']} min)"
            prompt += "\n"
        
        # Add weather context
        if weather_data:
            temp = weather_data.get('temperature', 'N/A')
            conditions = weather_data.get('conditions', 'N/A')
            
            prompt += f"\nCurrent Weather: {temp}°C, {conditions}\n"
            prompt += self._get_transportation_weather_advice(temp, conditions)
        
        # Add Marmaray option
        if marmaray_option and marmaray_option.get('use_marmaray'):
            prompt += f"""
Alternative: Marmaray Route
- Duration: {marmaray_option.get('travel_time', 'N/A')} minutes
- Stations: {', '.join(marmaray_option.get('stations', []))}
- Benefits: {', '.join(marmaray_option.get('advantages', []))}
"""
        
        # Add live transit data
        if live_transit_data:
            delays = live_transit_data.get('delays', [])
            alerts = live_transit_data.get('alerts', [])
            
            if delays or alerts:
                prompt += "\nReal-time Transit Information:\n"
                
                if delays:
                    prompt += "- Current delays: " + ", ".join(delays) + "\n"
                if alerts:
                    prompt += "- Service alerts: " + ", ".join(alerts) + "\n"
        
        # Add instructions
        prompt += """
Provide practical transportation advice considering:
1. Efficiency and time
2. Weather conditions (if applicable)
3. Real-time delays (if any)
4. Cost considerations
5. Comfort and convenience
6. Istanbul-specific tips (traffic patterns, peak hours, etc.)

"""
        
        if marmaray_option and marmaray_option.get('use_marmaray'):
            prompt += "Compare the main route with the Marmaray option and recommend the best.\n"
        
        if language == 'tr':
            prompt += "\nRespond in TURKISH with a friendly, helpful tone.\n"
        else:
            prompt += "\nRespond in ENGLISH with a friendly, helpful tone.\n"
        
        prompt += "Use emojis (🚇🚋🚶‍♂️⛴️🚂) to make it engaging!\n"
        prompt += "\nResponse:"
        
        return prompt
    
    def create_marmaray_comparison_prompt(
        self,
        query: str,
        regular_route: Dict,
        marmaray_route: Dict,
        weather_data: Optional[Dict] = None,
        language: str = 'en'
    ) -> str:
        """
        Create prompt comparing regular route with Marmaray
        
        Args:
            query: User's query
            regular_route: Standard route data
            marmaray_route: Marmaray route data
            weather_data: Current weather (optional)
            language: Response language
            
        Returns:
            Formatted comparison prompt
        """
        prompt = f"""{self.system_context}

Route Request: {query}

Option 1: Regular Route
- Duration: {regular_route.get('duration', 'N/A')} minutes
- Modes: {', '.join(regular_route.get('modes', []))}
- Transfers: {regular_route.get('transfers', 0)}

Option 2: Marmaray Route (Cross-Bosphorus Underground Railway)
- Duration: {marmaray_route.get('duration', 'N/A')} minutes
- Bosphorus crossing time: ~4 minutes underground
- Weather-independent
- Fixed schedule
- Modern, air-conditioned trains

"""
        
        if weather_data:
            temp = weather_data.get('temperature', 'N/A')
            conditions = weather_data.get('conditions', 'N/A')
            prompt += f"Current Weather: {temp}°C, {conditions}\n\n"
        
        prompt += """Compare both options and recommend the best based on:
- Time efficiency
- Weather impact (ferries can be delayed in bad weather)
- Reliability
- Comfort
- Cost

"""
        
        if language == 'tr':
            prompt += "Respond in TURKISH.\n"
        else:
            prompt += "Respond in ENGLISH.\n"
        
        prompt += "\nResponse:"
        
        return prompt
    
    def _get_weather_advice(
        self,
        temp: float,
        conditions: str,
        humidity: float,
        wind_speed: float
    ) -> str:
        """Generate weather-specific advice"""
        advice = []
        
        # Temperature advice
        if temp > 30:
            advice.append("🥵 Very hot - stay hydrated, seek shade/AC")
        elif temp > 25:
            advice.append("☀️ Hot - sunscreen recommended, light clothing")
        elif temp > 15:
            advice.append("🌤️ Pleasant - perfect weather for outdoor activities")
        elif temp > 10:
            advice.append("🧥 Cool - light jacket recommended")
        else:
            advice.append("🥶 Cold - warm clothing essential")
        
        # Condition advice
        condition_lower = str(conditions).lower()
        if any(x in condition_lower for x in ['rain', 'rainy', 'drizzle']):
            advice.append("☔ Rainy - bring umbrella, prefer indoor/covered transit")
        elif any(x in condition_lower for x in ['snow', 'snowy']):
            advice.append("❄️ Snowy - dress warmly, roads may be slippery")
        elif wind_speed > 30:
            advice.append("💨 Windy - secure belongings, ferry may be affected")
        
        # Humidity advice
        if humidity > 80:
            advice.append("💧 High humidity - feels more uncomfortable")
        
        return "\n".join(f"- {a}" for a in advice)
    
    def _get_transportation_weather_advice(
        self,
        temp: float,
        conditions: str
    ) -> str:
        """Generate transportation-specific weather advice"""
        advice = []
        
        condition_lower = str(conditions).lower()
        
        if any(x in condition_lower for x in ['rain', 'rainy', 'drizzle']):
            advice.append("- Metro/Tram preferred over walking")
            advice.append("- Ferry service may have delays")
            advice.append("- Marmaray unaffected by rain (underground)")
        elif temp > 30:
            advice.append("- Air-conditioned metro/Marmaray preferred")
            advice.append("- Avoid long walks in midday heat")
            advice.append("- Ferry provides cool breeze")
        elif temp < 5:
            advice.append("- Indoor waiting areas recommended")
            advice.append("- Metro faster than bus in cold")
            advice.append("- Marmaray preferred over ferry")
        
        if advice:
            return "\nWeather Impact on Transportation:\n" + "\n".join(advice)
        return ""


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("Testing Prompt Generator")
    print("="*80)
    
    generator = IstanbulPromptGenerator()
    
    # Test 1: Weather-aware prompt
    print("\n1️⃣ Weather-Aware Prompt:")
    print("-" * 80)
    weather_data = {
        'temperature': 18,
        'conditions': 'Rainy',
        'humidity': 85,
        'wind_speed': 15
    }
    prompt = generator.create_weather_aware_prompt(
        "What should I do in Istanbul today?",
        weather_data
    )
    print(prompt)
    
    # Test 2: Transportation prompt
    print("\n2️⃣ Transportation Prompt:")
    print("-" * 80)
    route_data = {
        'origin': 'Sultanahmet',
        'destination': 'Kadıköy',
        'duration': 25,
        'distance': 8.5,
        'modes': ['tram', 'ferry'],
        'steps': [
            {'instruction': 'Walk to Sultanahmet Tram Station', 'duration': 3},
            {'instruction': 'Take T1 tram to Eminönü', 'duration': 8},
            {'instruction': 'Take ferry to Kadıköy', 'duration': 20}
        ]
    }
    prompt = generator.create_transportation_prompt(
        "How do I get from Sultanahmet to Kadıköy?",
        route_data,
        weather_data=weather_data
    )
    print(prompt)
    
    print("\n✅ Tests complete!")
