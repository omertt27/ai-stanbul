"""
Step 4.2: Marmaray Integration with LLM
Integrates Marmaray recommendations into transportation handler with weather-aware LLM advice

Features:
- Automatic Marmaray detection for Bosphorus crossings
- Weather-aware recommendation strength
- LLM-powered contextual advice
- Comparison with regular routes
"""

# This code should be added to istanbul_ai/handlers/transportation_handler.py

# Add these imports at the top:
"""
from backend.data.marmaray_stations import (
    get_marmaray_recommendation,
    find_nearest_marmaray_station,
    crosses_bosphorus,
    MARMARAY_INFO
)
"""

# Add this method to the TransportationHandler class:

def _check_marmaray_option(
    self,
    origin_coords: Dict[str, float],
    dest_coords: Dict[str, float],
    weather_data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Check if Marmaray is a good option for this route
    
    Args:
        origin_coords: {'lat': float, 'lng': float}
        dest_coords: {'lat': float, 'lng': float}
        weather_data: Current weather information
        
    Returns:
        Marmaray recommendation dict or None
    """
    try:
        # Get weather conditions string
        weather_conditions = None
        if weather_data:
            weather_conditions = weather_data.get('conditions', weather_data.get('description', ''))
        
        # Get Marmaray recommendation
        recommendation = get_marmaray_recommendation(
            origin_lat=origin_coords['lat'],
            origin_lon=origin_coords['lng'],
            dest_lat=dest_coords['lat'],
            dest_lon=dest_coords['lng'],
            weather_conditions=weather_conditions
        )
        
        if recommendation['use_marmaray']:
            logger.info(f"✅ Marmaray recommended: {recommendation['recommendation_strength']}")
            return recommendation
        else:
            logger.info(f"ℹ️  Marmaray not applicable: {recommendation.get('reason', 'Unknown')}")
            return None
            
    except Exception as e:
        logger.error(f"Error checking Marmaray option: {e}")
        return None


def _create_marmaray_llm_advice(
    self,
    query: str,
    regular_route: Dict[str, Any],
    marmaray_option: Dict[str, Any],
    weather_data: Dict[str, Any]
) -> str:
    """
    Generate LLM advice comparing regular route with Marmaray option
    
    Args:
        query: User's original query
        regular_route: Regular route data from OSRM
        marmaray_option: Marmaray recommendation data
        weather_data: Current weather
        
    Returns:
        LLM-generated advice string
    """
    if not self.has_llm or not self.prompt_engine:
        # Fallback to template-based advice
        return self._create_marmaray_template_advice(marmaray_option, weather_data)
    
    try:
        # Create prompt
        prompt = self.prompt_engine.create_marmaray_comparison_prompt(
            origin=query.split(' from ')[-1].split(' to ')[0] if ' from ' in query else 'origin',
            destination=query.split(' to ')[-1] if ' to ' in query else 'destination',
            regular_route={
                'duration': regular_route.get('duration', 'unknown'),
                'distance': regular_route.get('distance', 'unknown')
            },
            marmaray_route={
                'duration': marmaray_option.get('travel_time_minutes', 'unknown'),
                'undersea_time': marmaray_option.get('undersea_crossing_time', 4)
            },
            weather_data=weather_data
        )
        
        # Generate LLM response
        logger.info("🤖 Generating Marmaray comparison advice with LLM...")
        llm_advice = self.llm_service.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        
        # Validate and truncate if needed
        if self.prompt_engine.validate_response_length(llm_advice, max_words=80):
            return llm_advice
        else:
            truncated = self.prompt_engine.truncate_to_sentences(llm_advice, max_sentences=3)
            return truncated
            
    except Exception as e:
        logger.error(f"Error generating Marmaray LLM advice: {e}")
        return self._create_marmaray_template_advice(marmaray_option, weather_data)


def _create_marmaray_template_advice(
    self,
    marmaray_option: Dict[str, Any],
    weather_data: Dict[str, Any]
) -> str:
    """
    Create template-based Marmaray advice (fallback)
    
    Args:
        marmaray_option: Marmaray recommendation data
        weather_data: Current weather
        
    Returns:
        Template-based advice string
    """
    strength = marmaray_option.get('recommendation_strength', 'recommended')
    travel_time = marmaray_option.get('travel_time_minutes', 'unknown')
    origin_station = marmaray_option.get('origin_station', {}).get('name', 'nearest station')
    dest_station = marmaray_option.get('dest_station', {}).get('name', 'destination station')
    
    # Weather-based recommendations
    conditions = weather_data.get('conditions', '').lower()
    
    if 'rain' in conditions or 'snow' in conditions:
        weather_tip = "Perfect for rainy weather - completely underground and weather-independent!"
    elif 'wind' in conditions:
        weather_tip = "Better than ferry in windy conditions - smooth underground crossing."
    else:
        weather_tip = "Fast and reliable underground crossing."
    
    if strength == 'highly_recommended':
        return f"🚇 Marmaray is highly recommended! {origin_station} → {dest_station} takes ~{travel_time} minutes. {weather_tip}"
    elif strength == 'alternative':
        return f"🚇 Marmaray is a good alternative: {origin_station} → {dest_station} (~{travel_time} min). {weather_tip}"
    else:
        return f"🚇 Consider Marmaray: {origin_station} → {dest_station} takes ~{travel_time} minutes. {weather_tip}"


def _enhance_route_with_marmaray(
    self,
    route_response: Dict[str, Any],
    origin_coords: Dict[str, float],
    dest_coords: Dict[str, float],
    weather_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhance route response with Marmaray option if applicable
    
    Args:
        route_response: Original route response
        origin_coords: Origin coordinates
        dest_coords: Destination coordinates
        weather_data: Current weather data
        
    Returns:
        Enhanced route response with Marmaray option
    """
    # Check if Marmaray is applicable
    marmaray_option = self._check_marmaray_option(
        origin_coords=origin_coords,
        dest_coords=dest_coords,
        weather_data=weather_data
    )
    
    if not marmaray_option:
        # No Marmaray option, return original response
        return route_response
    
    # Get regular route data
    regular_route = route_response.get('route', {})
    
    # Generate LLM advice comparing options
    if weather_data:
        marmaray_advice = self._create_marmaray_llm_advice(
            query=route_response.get('query', ''),
            regular_route=regular_route,
            marmaray_option=marmaray_option,
            weather_data=weather_data
        )
    else:
        marmaray_advice = self._create_marmaray_template_advice(
            marmaray_option=marmaray_option,
            weather_data=weather_data or {}
        )
    
    # Add Marmaray option to response
    route_response['marmaray_option'] = {
        'available': True,
        'recommended': marmaray_option.get('use_marmaray', False),
        'strength': marmaray_option.get('recommendation_strength', 'recommended'),
        'travel_time_minutes': marmaray_option.get('travel_time_minutes'),
        'origin_station': marmaray_option.get('origin_station'),
        'dest_station': marmaray_option.get('dest_station'),
        'undersea_crossing_time': marmaray_option.get('undersea_crossing_time', 4),
        'advantages': marmaray_option.get('advantages', []),
        'llm_advice': marmaray_advice,
        'transfer_info': marmaray_option.get('transfer_info'),
        'weather_independent': True
    }
    
    logger.info(f"✅ Marmaray option added to route response ({marmaray_option.get('recommendation_strength')})")
    
    return route_response


# Example usage in get_route_plan method:
"""
async def get_route_plan(
    self,
    origin: str,
    destination: str,
    gps_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    '''Get route plan with Marmaray option if applicable'''
    
    # Get regular route
    route = await self.directions_service.get_route(origin, destination)
    
    # Get weather data
    weather_data = await self._get_weather_data()
    
    # Build response
    response = {
        'route': route,
        'weather': weather_data,
        'query': f'from {origin} to {destination}'
    }
    
    # Check for Marmaray option
    if gps_context and 'coordinates' in gps_context:
        origin_coords = gps_context['coordinates'].get('origin')
        dest_coords = gps_context['coordinates'].get('destination')
        
        if origin_coords and dest_coords:
            response = self._enhance_route_with_marmaray(
                route_response=response,
                origin_coords=origin_coords,
                dest_coords=dest_coords,
                weather_data=weather_data
            )
    
    # Generate general LLM advice for the route
    if self.has_llm and weather_data:
        route_advice = self._create_weather_aware_route_advice(
            origin=origin,
            destination=destination,
            route=route,
            weather_data=weather_data
        )
        response['llm_advice'] = route_advice
    
    return response
"""

print("""
✅ Step 4.2 Implementation Code Created

This code adds:
1. _check_marmaray_option() - Detects Marmaray applicability
2. _create_marmaray_llm_advice() - LLM-powered comparison
3. _create_marmaray_template_advice() - Template fallback
4. _enhance_route_with_marmaray() - Adds Marmaray to route response

Key Features:
- Automatic Bosphorus crossing detection
- Weather-aware recommendation strength
- LLM comparison between regular route and Marmaray
- Fallback to template-based advice
- Integration with existing route planning

Next: Add these methods to TransportationHandler class
""")
