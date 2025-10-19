"""
AI Chat Navigation Handler
Complete integration of OSRM routing with AI chat system

Handles navigation requests in natural language and returns rich responses
with turn-by-turn directions, POI recommendations, and map data.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import asyncio

# Import our services
try:
    from services.osrm_routing_service import get_osrm_service, OSRMRoute
    OSRM_AVAILABLE = True
except ImportError:
    OSRM_AVAILABLE = False

try:
    from services.navigation_intent_detector import get_navigation_detector
    INTENT_DETECTOR_AVAILABLE = True
except ImportError:
    INTENT_DETECTOR_AVAILABLE = False

try:
    from services.istanbul_geocoder import get_geocoder
    GEOCODER_AVAILABLE = True
except ImportError:
    GEOCODER_AVAILABLE = False

try:
    from services.poi_database_service import POIDatabaseService
    POI_DATABASE_AVAILABLE = True
except ImportError:
    POI_DATABASE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NavigationResponse:
    """Complete navigation response for AI chat"""
    success: bool
    message: str  # Natural language response
    route: Optional[Dict[str, Any]] = None
    map_display: bool = False
    suggestions: List[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'success': self.success,
            'message': self.message,
            'route': self.route,
            'map_display': self.map_display,
            'suggestions': self.suggestions or [],
            'error': self.error
        }


class AINavigationHandler:
    """
    AI Chat Navigation Handler
    
    Processes navigation requests and generates comprehensive responses
    with turn-by-turn directions, POI recommendations, and map data.
    
    Example Usage:
        handler = AINavigationHandler()
        response = await handler.handle_navigation_query(
            "How do I get from Sultanahmet to Taksim?"
        )
        print(response.message)  # Natural language response
        print(response.route)    # Route data with geometry
    """
    
    def __init__(self):
        """Initialize navigation handler with all required services"""
        self.osrm = get_osrm_service() if OSRM_AVAILABLE else None
        self.detector = get_navigation_detector() if INTENT_DETECTOR_AVAILABLE else None
        self.geocoder = get_geocoder() if GEOCODER_AVAILABLE else None
        
        logger.info("✅ AI Navigation Handler initialized")
        logger.info(f"   OSRM: {'✅' if OSRM_AVAILABLE else '❌'}")
        logger.info(f"   Intent Detector: {'✅' if INTENT_DETECTOR_AVAILABLE else '❌'}")
        logger.info(f"   Geocoder: {'✅' if GEOCODER_AVAILABLE else '❌'}")
        logger.info(f"   POI Database: {'✅' if POI_DATABASE_AVAILABLE else '❌'}")
    
    async def handle_navigation_query(
        self,
        query: str,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None
    ) -> NavigationResponse:
        """
        Handle a navigation query from AI chat
        
        Args:
            query: User's natural language query
            user_lat: User's current latitude (optional)
            user_lon: User's current longitude (optional)
        
        Returns:
            NavigationResponse with route and natural language description
        
        Example:
            >>> handler = AINavigationHandler()
            >>> response = await handler.handle_navigation_query(
            ...     "How do I get from Sultanahmet to Taksim?",
            ...     user_lat=41.0082,
            ...     user_lon=28.9784
            ... )
            >>> print(response.message)
        """
        try:
            # Check if all services are available
            if not all([OSRM_AVAILABLE, INTENT_DETECTOR_AVAILABLE, GEOCODER_AVAILABLE]):
                return NavigationResponse(
                    success=False,
                    message="Sorry, navigation services are currently unavailable. Please try again later.",
                    error="Missing required services"
                )
            
            # Detect navigation intent
            intent = self.detector.detect(query)
            
            if not intent:
                return NavigationResponse(
                    success=False,
                    message="I didn't understand that navigation request. Try asking like: 'How do I get from Sultanahmet to Taksim?'",
                    suggestions=[
                        "How do I get from Sultanahmet to Taksim?",
                        "Walking route to Blue Mosque",
                        "Drive me to Dolmabahce Palace",
                        "Route from Galata Tower to Grand Bazaar with museum stops"
                    ]
                )
            
            # Generate navigation response
            return await self._generate_navigation_response(
                origin_name=intent.origin,
                destination_name=intent.destination,
                mode=intent.mode,
                include_pois=intent.include_pois,
                poi_categories=intent.poi_categories,
                user_lat=user_lat,
                user_lon=user_lon
            )
        
        except Exception as e:
            logger.error(f"Navigation query error: {e}", exc_info=True)
            return NavigationResponse(
                success=False,
                message="Sorry, I encountered an error while planning your route. Please try again.",
                error=str(e)
            )
    
    async def _generate_navigation_response(
        self,
        origin_name: str,
        destination_name: str,
        mode: str = "walking",
        include_pois: bool = False,
        poi_categories: List[str] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None
    ) -> NavigationResponse:
        """
        Generate comprehensive navigation response
        
        Args:
            origin_name: Origin location name
            destination_name: Destination location name
            mode: Transportation mode (walking, driving, cycling)
            include_pois: Whether to include POI recommendations
            poi_categories: List of POI categories to include
            user_lat: User's current latitude
            user_lon: User's current longitude
        
        Returns:
            NavigationResponse with route and formatted message
        """
        # Geocode locations
        if origin_name == 'current_location' and user_lat and user_lon:
            origin_coords = (user_lat, user_lon)
            origin_display_name = "Your Location"
        else:
            origin_result = await self.geocoder.geocode_async(origin_name)
            if not origin_result:
                return NavigationResponse(
                    success=False,
                    message=f"Sorry, I couldn't find '{origin_name}'. Could you be more specific?",
                    suggestions=[
                        "Try using well-known landmarks like 'Sultanahmet', 'Taksim', 'Galata Tower'",
                        "Check the spelling of the location name"
                    ]
                )
            origin_coords = origin_result.to_tuple()
            origin_display_name = origin_result.name.title()
        
        destination_result = await self.geocoder.geocode_async(destination_name)
        if not destination_result:
            return NavigationResponse(
                success=False,
                message=f"Sorry, I couldn't find '{destination_name}'. Could you be more specific?",
                suggestions=[
                    "Try using well-known landmarks",
                    "Check the spelling of the location name"
                ]
            )
        destination_coords = destination_result.to_tuple()
        destination_display_name = destination_result.name.title()
        
        # Get route from OSRM
        route = await self.osrm.get_route(
            start=origin_coords,
            end=destination_coords,
            mode=mode,
            alternatives=True
        )
        
        if not route:
            return NavigationResponse(
                success=False,
                message=f"Sorry, I couldn't find a {mode} route between those locations.",
                suggestions=[
                    "Try a different transportation mode",
                    "Check if the locations are accessible"
                ]
            )
        
        # Add POIs along route if requested
        waypoints = []
        if include_pois and POI_DATABASE_AVAILABLE:
            waypoints = await self._find_pois_along_route(
                route=route,
                categories=poi_categories or ['museum', 'mosque', 'palace']
            )
        
        # Generate natural language response
        message = self._format_navigation_message(
            origin_name=origin_display_name,
            destination_name=destination_display_name,
            route=route,
            waypoints=waypoints,
            mode=mode
        )
        
        # Prepare route data for frontend
        route_data = {
            'origin': {
                'name': origin_display_name,
                'lat': origin_coords[0],
                'lon': origin_coords[1]
            },
            'destination': {
                'name': destination_display_name,
                'lat': destination_coords[0],
                'lon': destination_coords[1]
            },
            'distance_km': round(route.total_distance_km, 2),
            'duration_min': round(route.total_duration_min),
            'mode': mode,
            'steps': [
                {
                    'instruction': step.instruction,
                    'distance_m': round(step.distance_m),
                    'duration_s': round(step.duration_s),
                    'street_name': step.street_name
                }
                for step in route.steps
            ],
            'geometry': route.geometry,
            'waypoints': waypoints,
            'alternatives_count': len(route.alternatives) if route.alternatives else 0
        }
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            origin_name=origin_display_name,
            destination_name=destination_display_name,
            has_alternatives=bool(route.alternatives),
            has_pois=bool(waypoints)
        )
        
        return NavigationResponse(
            success=True,
            message=message,
            route=route_data,
            map_display=True,
            suggestions=suggestions
        )
    
    async def _find_pois_along_route(
        self,
        route: OSRMRoute,
        categories: List[str],
        max_distance_km: float = 0.5,
        max_pois: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find POIs near the route
        
        Args:
            route: OSRM route object
            categories: POI categories to search
            max_distance_km: Maximum distance from route (km)
            max_pois: Maximum number of POIs to return
        
        Returns:
            List of POI dictionaries
        """
        try:
            poi_db = POIDatabaseService()
            
            # Find POIs near route path
            nearby_pois = poi_db.find_pois_near_path(
                route.geometry,
                max_distance_km=max_distance_km,
                categories=categories
            )
            
            # Select top POIs by popularity
            top_pois = sorted(
                nearby_pois,
                key=lambda p: p.popularity_score,
                reverse=True
            )[:max_pois]
            
            return [
                {
                    'name': poi.name,
                    'name_en': poi.name_en,
                    'lat': poi.location.lat,
                    'lon': poi.location.lon,
                    'category': poi.category,
                    'visit_duration_min': poi.visit_duration_min,
                    'rating': poi.rating,
                    'description': poi.description_en
                }
                for poi in top_pois
            ]
        
        except Exception as e:
            logger.error(f"POI search error: {e}")
            return []
    
    def _format_navigation_message(
        self,
        origin_name: str,
        destination_name: str,
        route: OSRMRoute,
        waypoints: List[Dict],
        mode: str
    ) -> str:
        """
        Format natural language navigation message
        
        Args:
            origin_name: Origin location name
            destination_name: Destination location name
            route: OSRM route object
            waypoints: List of POI waypoints
            mode: Transportation mode
        
        Returns:
            Formatted message string
        """
        # Transport mode emojis
        mode_emoji = {
            'walking': '🚶',
            'driving': '🚗',
            'cycling': '🚴'
        }.get(mode, '🚶')
        
        message = f"🗺️ **Route from {origin_name} to {destination_name}**\n\n"
        message += f"📍 **Distance:** {route.total_distance_km:.2f} km\n"
        message += f"⏱️ **Duration:** {int(route.total_duration_min)} minutes {mode}\n\n"
        
        # Add POI recommendations
        if waypoints:
            message += f"🎯 **Interesting stops along the way:**\n"
            for i, poi in enumerate(waypoints, 1):
                message += f"{i}. **{poi['name_en']}** ({poi['category']})\n"
                message += f"   ⭐ {poi['rating']}/5 • ⏱️ {poi['visit_duration_min']} min visit\n"
            message += "\n"
        
        # Add turn-by-turn directions
        message += f"{mode_emoji} **Turn-by-turn directions:**\n"
        for i, step in enumerate(route.steps[:10], 1):  # Show first 10 steps
            message += f"{i}. {step.instruction}"
            if step.street_name and step.street_name not in step.instruction:
                message += f" on {step.street_name}"
            message += f" ({int(step.distance_m)}m)\n"
        
        if len(route.steps) > 10:
            remaining = len(route.steps) - 10
            message += f"\n... and {remaining} more step{'s' if remaining > 1 else ''}\n"
        
        # Add alternatives info
        if route.alternatives:
            message += f"\n💡 **{len(route.alternatives)} alternative route{'s' if len(route.alternatives) > 1 else ''} available**\n"
        
        # Add helpful tips
        message += f"\n💬 **Tips:**\n"
        if mode == 'walking':
            message += "• Wear comfortable shoes\n"
            message += "• Bring water for longer routes\n"
        elif mode == 'driving':
            message += "• Check traffic conditions before departing\n"
            message += "• Consider parking availability at destination\n"
        elif mode == 'cycling':
            message += "• Wear a helmet for safety\n"
            message += "• Follow bike lanes where available\n"
        
        return message
    
    def _generate_suggestions(
        self,
        origin_name: str,
        destination_name: str,
        has_alternatives: bool,
        has_pois: bool
    ) -> List[str]:
        """
        Generate follow-up suggestions for the user
        
        Args:
            origin_name: Origin location name
            destination_name: Destination location name
            has_alternatives: Whether alternative routes are available
            has_pois: Whether POIs were included
        
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        if has_alternatives:
            suggestions.append("Show me alternative routes")
        
        if not has_pois:
            suggestions.append(f"Show me attractions between {origin_name} and {destination_name}")
        
        suggestions.extend([
            "What's the fastest route?",
            "How do I get back?",
            "Tell me about the destination"
        ])
        
        return suggestions[:5]  # Limit to 5 suggestions


# ═══════════════════════════════════════════════════════════════════
# Global Handler Instance
# ═══════════════════════════════════════════════════════════════════

_handler_instance: Optional[AINavigationHandler] = None


def get_navigation_handler() -> AINavigationHandler:
    """Get or create global navigation handler instance"""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = AINavigationHandler()
    return _handler_instance


# ═══════════════════════════════════════════════════════════════════
# Testing & Examples
# ═══════════════════════════════════════════════════════════════════

async def test_navigation_handler():
    """Test navigation handler with various queries"""
    print("\n🤖 AI Navigation Handler - Test Suite\n")
    print("=" * 70)
    
    handler = AINavigationHandler()
    
    test_queries = [
        ("How do I get from Sultanahmet to Taksim?", None, None),
        ("Walking route from Galata Tower to Grand Bazaar", None, None),
        ("Drive me to Dolmabahce Palace from Blue Mosque", None, None),
        ("Take me to Ortakoy with museum stops", 41.0082, 28.9784),
    ]
    
    for i, (query, lat, lon) in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        print(f"   User location: {(lat, lon) if lat and lon else 'Not provided'}")
        
        response = await handler.handle_navigation_query(query, lat, lon)
        
        if response.success:
            print(f"\n   ✅ Navigation successful!")
            print(f"\n{response.message}")
            if response.route:
                print(f"\n   📊 Route details:")
                print(f"      Distance: {response.route['distance_km']} km")
                print(f"      Duration: {response.route['duration_min']} minutes")
                print(f"      Steps: {len(response.route['steps'])}")
        else:
            print(f"\n   ❌ Navigation failed: {response.message}")
        
        print("\n" + "-" * 70)
    
    print("\n" + "=" * 70)
    print("✅ Test suite completed!\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_navigation_handler())
