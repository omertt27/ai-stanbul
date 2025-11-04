"""
Google Maps-Level Precision Prompt Template
Generates highly detailed, actionable route guidance
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class WalkingStep:
    """Detailed walking instruction"""
    from_location: str
    to_location: str
    distance_meters: int
    duration_minutes: int
    instructions: List[str]
    landmarks: Optional[List[str]] = None


@dataclass
class TransitStep:
    """Detailed transit instruction"""
    mode: str  # 'tram', 'metro', 'bus', 'ferry', 'marmaray'
    line: str
    from_station: str
    to_station: str
    stops_count: int
    stops_passed: List[str]
    duration_minutes: int
    frequency: str
    platform: Optional[str] = None
    price: str = "15 TL"
    next_departures: Optional[List[str]] = None
    direction: Optional[str] = None


class GoogleMapsPromptTemplate:
    """
    Create Google Maps-style precision prompts for LLM
    
    Ensures:
    - Exact station names
    - Turn-by-turn walking directions
    - Precise times and distances
    - Platform numbers
    - Next departure times
    - Weather-aware recommendations
    """
    
    def __init__(self):
        self.emoji_map = {
            'walk': '🚶‍♂️',
            'tram': '🚋',
            'metro': '🚇',
            'bus': '🚌',
            'ferry': '⛴️',
            'marmaray': '🚂',
            'funicular': '🚡'
        }
    
    def create_precision_prompt(
        self,
        origin: str,
        destination: str,
        route_steps: List[Dict],
        weather_data: Optional[Dict] = None,
        language: str = 'en'
    ) -> str:
        """
        Create Google Maps-level precision prompt
        
        Args:
            origin: Starting location
            destination: Ending location
            route_steps: Detailed route steps from OSRM/directions service
            weather_data: Current weather information
            language: 'en' or 'tr'
            
        Returns:
            Formatted prompt for LLM with strict structure requirements
        """
        
        # Build structured route breakdown
        route_breakdown = self._build_route_breakdown(route_steps)
        
        # Calculate totals
        total_time = sum(step.get('duration_minutes', 0) for step in route_steps)
        total_cost = self._calculate_total_cost(route_steps)
        
        # Weather context
        weather_context = self._build_weather_context(weather_data) if weather_data else ""
        
        # Create the prompt
        prompt = f"""You are a precision navigation assistant for Istanbul. You MUST provide EXACT, ACTIONABLE directions like Google Maps.

**CRITICAL REQUIREMENTS:**
1. Use EXACT station names (not "the tram station" but "Sultanahmet Tram Stop, Platform 1")
2. Provide turn-by-turn walking directions with street names
3. Include specific distances (meters) and times (minutes)
4. Mention platform numbers, exits, and transfer points
5. Add next departure times if available
6. Keep it concise but precise

**Route Request:**
From: {origin}
To: {destination}

**Detailed Route Data:**
{route_breakdown}

**Total Journey:**
- Duration: {total_time} minutes
- Cost: {total_cost}

{weather_context}

**YOUR TASK:**
Generate a step-by-step navigation response that includes:

1. **Each walking segment:**
   - Exact starting point
   - Turn-by-turn directions with street names
   - Distance in meters
   - Duration in minutes
   - Landmarks to look for

2. **Each transit segment:**
   - Exact line number/name
   - Boarding station with platform/exit info
   - Direction of travel
   - Stations passed
   - Alighting station with exit info
   - Duration
   - Frequency/next departures
   - Price

3. **Weather considerations:**
   - How weather affects this route
   - Covered vs outdoor segments
   - Recommendations based on conditions

4. **Practical tips:**
   - Which entrance/exit to use
   - Where to stand on platform
   - Transfer shortcuts
   - Crowd warnings if peak hour

**FORMAT EXAMPLE:**
🚶‍♂️ **Step 1: Walk to [EXACT STOP NAME]** (Xm, X min)
   → [Turn-by-turn with street names]
   → [Landmark to look for]

🚋 **Step 2: [LINE] from [STATION] to [STATION]** (X stops, X min)
   → Board at: [Platform/Exit details]
   → Direction: [Terminus station]
   → Stops: [List all intermediate stops]
   → Exit at: [Exit number/direction]
   → Next departures: [Times]
   → Price: [Exact price]

[Continue for all steps...]

⏱️ **Total: X minutes, X TL**
🌤️ **Weather note:** [Specific advice for this route]

**Language:** {"Turkish" if language == 'tr' else "English"}

Generate the response now:"""

        return prompt
    
    def _build_route_breakdown(self, route_steps: List[Dict]) -> str:
        """Build detailed breakdown of route steps"""
        breakdown = []
        
        for i, step in enumerate(route_steps, 1):
            step_type = step.get('mode', step.get('type', 'unknown'))
            emoji = self.emoji_map.get(step_type, '📍')
            
            if step_type == 'walk' or step_type == 'walking':
                breakdown.append(self._format_walking_step(i, step, emoji))
            else:
                breakdown.append(self._format_transit_step(i, step, emoji))
        
        return '\n\n'.join(breakdown)
    
    def _format_walking_step(self, num: int, step: Dict, emoji: str) -> str:
        """Format walking step details"""
        from_loc = step.get('from', step.get('start_location', 'Unknown'))
        to_loc = step.get('to', step.get('end_location', 'Unknown'))
        distance = step.get('distance_meters', step.get('distance', 0))
        duration = step.get('duration_minutes', step.get('duration', 0))
        
        text = f"{emoji} **Step {num}: Walk from {from_loc} to {to_loc}**\n"
        text += f"   Distance: {distance}m\n"
        text += f"   Duration: {duration} min\n"
        
        if 'instructions' in step:
            text += "   Directions:\n"
            for instruction in step['instructions']:
                text += f"   → {instruction}\n"
        
        if 'landmarks' in step:
            text += "   Landmarks: " + ", ".join(step['landmarks']) + "\n"
        
        return text
    
    def _format_transit_step(self, num: int, step: Dict, emoji: str) -> str:
        """Format transit step details"""
        mode = step.get('mode', 'transit')
        line = step.get('line', step.get('route', 'Unknown'))
        from_station = step.get('from_station', step.get('start_station', 'Unknown'))
        to_station = step.get('to_station', step.get('end_station', 'Unknown'))
        duration = step.get('duration_minutes', step.get('duration', 0))
        
        text = f"{emoji} **Step {num}: {line} from {from_station} to {to_station}**\n"
        text += f"   Duration: {duration} min\n"
        
        if 'stops_count' in step:
            text += f"   Stops: {step['stops_count']}\n"
        
        if 'stops_passed' in step:
            text += f"   Via: {', '.join(step['stops_passed'])}\n"
        
        if 'platform' in step:
            text += f"   Platform: {step['platform']}\n"
        
        if 'direction' in step:
            text += f"   Direction: {step['direction']}\n"
        
        if 'frequency' in step:
            text += f"   Frequency: {step['frequency']}\n"
        
        if 'next_departures' in step:
            text += f"   Next departures: {', '.join(step['next_departures'])}\n"
        
        if 'price' in step:
            text += f"   Price: {step['price']}\n"
        
        return text
    
    def _build_weather_context(self, weather_data: Dict) -> str:
        """Build weather context for route"""
        temp = weather_data.get('temperature', 0)
        condition = weather_data.get('condition', 'unknown')
        wind = weather_data.get('wind_speed', 0)
        
        context = f"\n**Current Weather:**\n"
        context += f"- Temperature: {temp}°C\n"
        context += f"- Condition: {condition}\n"
        context += f"- Wind: {wind} km/h\n"
        
        # Add weather-specific routing advice
        if condition.lower() in ['rain', 'rainy', 'drizzle']:
            context += "\n⚠️ **Weather Impact:** Rainy - recommend covered routes, ferry may have delays\n"
        elif temp > 30:
            context += "\n⚠️ **Weather Impact:** Very hot - prefer air-conditioned metro/tram, avoid long walks\n"
        elif temp < 5:
            context += "\n⚠️ **Weather Impact:** Cold - prefer indoor waiting, Marmaray over ferry\n"
        elif wind > 30:
            context += "\n⚠️ **Weather Impact:** Windy - ferry service may be affected\n"
        
        return context
    
    def _calculate_total_cost(self, route_steps: List[Dict]) -> str:
        """Calculate total cost of route"""
        # Istanbul uses Istanbulkart with flat fare per mode
        # Simplified: count unique transit modes
        transit_steps = [s for s in route_steps if s.get('mode', s.get('type')) != 'walk']
        
        if not transit_steps:
            return "0 TL (walking only)"
        
        # Approximate: 15 TL per Istanbulkart tap
        # (In reality, transfers within 2 hours get discount)
        cost = len(transit_steps) * 15
        
        return f"~{cost} TL (Istanbulkart)"


def create_google_maps_response(
    origin: str,
    destination: str,
    route_data: Dict,
    weather_data: Optional[Dict] = None,
    llm_service = None,
    language: str = 'en'
) -> str:
    """
    Generate Google Maps-level precision response
    
    Args:
        origin: Starting location
        destination: Destination
        route_data: Route data from directions service
        weather_data: Weather information
        llm_service: LLM service instance
        language: Response language
        
    Returns:
        Precise, actionable navigation instructions
    """
    
    # Create prompt template
    template = GoogleMapsPromptTemplate()
    
    # Generate precision prompt
    prompt = template.create_precision_prompt(
        origin=origin,
        destination=destination,
        route_steps=route_data.get('steps', []),
        weather_data=weather_data,
        language=language
    )
    
    # Generate with LLM
    if llm_service:
        response = llm_service.generate(
            prompt=prompt,
            max_tokens=500,  # Longer for detailed directions
            temperature=0.3   # Lower temperature for factual accuracy
        )
        return response
    
    return prompt


# Example usage
if __name__ == "__main__":
    # Example route data
    example_route = {
        'steps': [
            {
                'type': 'walk',
                'from': 'Sultanahmet Square',
                'to': 'Sultanahmet Tram Stop',
                'distance_meters': 150,
                'duration_minutes': 2,
                'instructions': [
                    'Head north on Divan Yolu Cd toward Alemdar Cd',
                    'Turn right at Alemdar Cd',
                    'Tram stop on your left after 50m'
                ],
                'landmarks': ['Blue Mosque on your right', 'Hagia Sophia behind you']
            },
            {
                'mode': 'tram',
                'line': 'T1',
                'from_station': 'Sultanahmet',
                'to_station': 'Eminönü',
                'stops_count': 3,
                'stops_passed': ['Gülhane', 'Sirkeci'],
                'duration_minutes': 8,
                'frequency': '5-7 minutes',
                'platform': 'Platform 1 (Kabataş direction)',
                'direction': 'Kabataş',
                'price': '15 TL',
                'next_departures': ['14:15', '14:22', '14:29']
            },
            {
                'type': 'walk',
                'from': 'Eminönü Tram Stop',
                'to': 'Eminönü Ferry Terminal',
                'distance_meters': 200,
                'duration_minutes': 3,
                'instructions': [
                    'Exit tram from rear doors',
                    'Turn right toward waterfront',
                    'Walk along Ragıp Gümüşpala Cd',
                    'Ferry terminal on your right'
                ],
                'landmarks': ['Galata Bridge ahead', 'Spice Bazaar on left']
            },
            {
                'mode': 'ferry',
                'line': 'Kadıköy Ferry',
                'from_station': 'Eminönü Pier',
                'to_station': 'Kadıköy Pier',
                'duration_minutes': 20,
                'frequency': '20 minutes',
                'platform': 'Dock 3',
                'direction': 'Kadıköy',
                'price': '15 TL',
                'next_departures': ['14:30', '14:50', '15:10']
            }
        ]
    }
    
    example_weather = {
        'temperature': 22,
        'condition': 'Sunny',
        'wind_speed': 15
    }
    
    template = GoogleMapsPromptTemplate()
    prompt = template.create_precision_prompt(
        origin="Sultanahmet",
        destination="Kadıköy",
        route_steps=example_route['steps'],
        weather_data=example_weather,
        language='en'
    )
    
    print("="*80)
    print("GOOGLE MAPS-LEVEL PRECISION PROMPT")
    print("="*80)
    print(prompt)
    print("="*80)
