"""
AI Chat Route Integration
=========================

Connects intelligent routing to AI chat system for conversational route planning.
Users can ask for routes in natural language and get:
- Realistic walking routes with OpenStreetMap
- Multi-district optimization with ML
- Visual map display
- Smart recommendations

Usage in chat:
- "Show me walking route from Sultanahmet to Galata Tower"
- "Plan route visiting Taksim, Grand Bazaar, and Blue Mosque"
- "How do I get from my hotel to Hagia Sophia?"
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Import intelligent route integration
try:
    import sys
    import os
    # Add parent directories to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from intelligent_route_integration import (
        IntelligentRouteIntegration,
        IntelligentRoute,
        create_intelligent_route_integration
    )
    ROUTE_INTEGRATION_AVAILABLE = True
    logger.info("✅ Intelligent Route Integration available")
except ImportError as e:
    ROUTE_INTEGRATION_AVAILABLE = False
    logger.warning(f"⚠️ Intelligent Route Integration not available: {e}")


class AIChatRouteHandler:
    """
    AI Chat Route Handler
    
    Processes natural language route requests from chat and returns
    intelligent routes with visualization data.
    """
    
    # Famous Istanbul locations with coordinates
    KNOWN_LOCATIONS = {
        # Sultanahmet area
        'sultanahmet': (41.0054, 28.9768),
        'sultanahmet square': (41.0054, 28.9768),
        'blue mosque': (41.0054, 28.9768),
        'hagia sophia': (41.0086, 28.9802),
        'topkapi palace': (41.0115, 28.9833),
        'basilica cistern': (41.0084, 28.9778),
        
        # Beyoğlu area
        'taksim': (41.0370, 28.9850),
        'taksim square': (41.0370, 28.9850),
        'istiklal': (41.0332, 28.9775),
        'istiklal street': (41.0332, 28.9775),
        'galata tower': (41.0256, 28.9742),
        'galata': (41.0256, 28.9742),
        
        # Grand Bazaar area
        'grand bazaar': (41.0108, 28.9680),
        'kapali carsi': (41.0108, 28.9680),
        'spice bazaar': (41.0166, 28.9700),
        'misir carsisi': (41.0166, 28.9700),
        
        # Bosphorus
        'dolmabahce palace': (41.0392, 29.0000),
        'ortakoy': (41.0553, 29.0275),
        'bebek': (41.0797, 29.0425),
        
        # Asian side
        'kadikoy': (40.9900, 29.0250),
        'uskudar': (41.0226, 29.0150),
        'maiden tower': (41.0210, 29.0044),
        
        # Other districts
        'besiktas': (41.0426, 29.0050),
        'fatih': (41.0182, 28.9497),
        'eminonu': (41.0177, 28.9742),
        'balat': (41.0297, 28.9489),
    }
    
    def __init__(self):
        """Initialize chat route handler"""
        self.route_integration = None
        
        if ROUTE_INTEGRATION_AVAILABLE:
            self.route_integration = create_intelligent_route_integration(
                enable_osrm=True,
                enable_ml=True,
                enable_gps=True,
                osrm_profile='foot'
            )
            logger.info("✅ AI Chat Route Handler initialized")
        else:
            logger.warning("⚠️ Route integration not available")
    
    def handle_route_request(
        self,
        message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Handle route request from chat
        
        Args:
            message: User's chat message
            user_context: User preferences and current location
            
        Returns:
            Dict with route data and response message, or None if not a route request
        """
        # Check if this is a route request
        if not self._is_route_request(message):
            return None
        
        # Extract locations from message
        locations = self._extract_locations(message)
        
        if not locations or len(locations) < 2:
            return {
                'type': 'error',
                'message': "I couldn't identify the locations. Please specify at least a start and end point, like 'route from Sultanahmet to Galata Tower'."
            }
        
        # Determine transport mode
        transport_mode = self._detect_transport_mode(message)
        
        # Plan route
        try:
            if len(locations) == 2:
                # Single route
                route = self.route_integration.plan_intelligent_route(
                    start=locations[0],
                    end=locations[1],
                    transport_mode=transport_mode,
                    user_context=user_context
                )
                
                response = self._format_route_response(route, single=True)
                
            else:
                # Multi-location route
                districts = self._detect_districts(message)
                routes = self.route_integration.plan_multi_district_route(
                    locations=locations,
                    districts=districts,
                    transport_mode=transport_mode,
                    user_context=user_context
                )
                
                response = self._format_route_response(routes, single=False)
            
            return response
            
        except Exception as e:
            logger.error(f"Error planning route: {e}")
            return {
                'type': 'error',
                'message': f"Sorry, I encountered an error planning your route: {str(e)}"
            }
    
    def _is_route_request(self, message: str) -> bool:
        """Check if message is a route request"""
        message_lower = message.lower()
        
        route_keywords = [
            'route', 'directions', 'how to get', 'how do i get',
            'walk', 'walking', 'path', 'way to', 'navigate',
            'take me', 'show me', 'from', 'to'
        ]
        
        return any(keyword in message_lower for keyword in route_keywords)
    
    def _extract_locations(self, message: str) -> List[Tuple[float, float]]:
        """Extract location coordinates from message"""
        message_lower = message.lower()
        found_locations = []
        
        # Try to find known locations
        for location_name, coords in self.KNOWN_LOCATIONS.items():
            if location_name in message_lower:
                found_locations.append((location_name, coords))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_locations = []
        for name, coords in found_locations:
            if name not in seen:
                seen.add(name)
                unique_locations.append(coords)
        
        # If using "from X to Y" pattern, ensure correct order
        from_match = re.search(r'from\s+([^to]+)\s+to\s+(.+)', message_lower)
        if from_match:
            start_name = from_match.group(1).strip()
            end_name = from_match.group(2).strip()
            
            start_coords = None
            end_coords = None
            
            for location_name, coords in self.KNOWN_LOCATIONS.items():
                if location_name in start_name:
                    start_coords = coords
                if location_name in end_name:
                    end_coords = coords
            
            if start_coords and end_coords:
                return [start_coords, end_coords]
        
        return unique_locations
    
    def _detect_transport_mode(self, message: str) -> str:
        """Detect transport mode from message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['walk', 'walking', 'on foot']):
            return 'walking'
        elif any(word in message_lower for word in ['bus', 'metro', 'tram', 'transit', 'public transport']):
            return 'transit'
        elif any(word in message_lower for word in ['mix', 'combined']):
            return 'mixed'
        
        # Default to walking
        return 'walking'
    
    def _detect_districts(self, message: str) -> List[str]:
        """Detect districts mentioned in message"""
        message_lower = message.lower()
        districts = []
        
        district_names = [
            'Sultanahmet', 'Beyoğlu', 'Fatih', 'Beşiktaş', 
            'Kadıköy', 'Üsküdar', 'Eminönü', 'Balat'
        ]
        
        for district in district_names:
            if district.lower() in message_lower:
                districts.append(district)
        
        return districts
    
    def _format_route_response(
        self,
        route_or_routes: Any,
        single: bool = True
    ) -> Dict[str, Any]:
        """Format route response for chat"""
        
        if single:
            route = route_or_routes
            
            # Create readable message
            distance_km = route.visualization.total_distance / 1000
            duration_min = route.visualization.total_duration / 60
            
            message = f"🗺️ **Route Planned Successfully!**\n\n"
            message += f"📏 **Distance:** {distance_km:.2f} km\n"
            message += f"⏱️ **Duration:** {duration_min:.0f} minutes\n"
            message += f"🚶 **Mode:** {route.visualization.mode.title()}\n"
            
            if route.visualization.districts:
                message += f"🏛️ **Districts:** {', '.join(route.visualization.districts)}\n"
            
            if route.recommendations:
                message += f"\n💡 **Recommendations:**\n"
                for rec in route.recommendations:
                    message += f"  • {rec}\n"
            
            message += "\n📍 The route is displayed on the map above."
            
            # Export visualization data
            visualization_data = self.route_integration.export_for_frontend(route, format='leaflet')
            
            return {
                'type': 'route',
                'message': message,
                'route_data': {
                    'single': True,
                    'visualization': visualization_data,
                    'start': route.start_location,
                    'end': route.end_location,
                    'distance': route.visualization.total_distance,
                    'duration': route.visualization.total_duration,
                    'waypoints': route.visualization.waypoints,
                    'steps': route.visualization.steps,
                    'geojson': route.visualization.geojson
                }
            }
        
        else:
            routes = route_or_routes
            
            # Calculate totals
            total_distance = sum(r.visualization.total_distance for r in routes) / 1000
            total_duration = sum(r.visualization.total_duration for r in routes) / 60
            
            # Collect all districts
            all_districts = set()
            for route in routes:
                all_districts.update(route.visualization.districts)
            
            message = f"🗺️ **Multi-Stop Route Planned!**\n\n"
            message += f"📏 **Total Distance:** {total_distance:.2f} km\n"
            message += f"⏱️ **Total Duration:** {total_duration:.0f} minutes\n"
            message += f"🛑 **Stops:** {len(routes) + 1}\n"
            message += f"🏛️ **Districts:** {', '.join(all_districts)}\n"
            
            message += f"\n**Route Segments:**\n"
            for i, route in enumerate(routes, 1):
                seg_distance = route.visualization.total_distance / 1000
                seg_duration = route.visualization.total_duration / 60
                message += f"  {i}. {seg_distance:.1f}km, {seg_duration:.0f}min\n"
            
            message += "\n📍 The complete route is displayed on the map above."
            
            # Export all routes
            routes_data = [
                {
                    'visualization': self.route_integration.export_for_frontend(r, format='leaflet'),
                    'start': r.start_location,
                    'end': r.end_location
                }
                for r in routes
            ]
            
            return {
                'type': 'route',
                'message': message,
                'route_data': {
                    'single': False,
                    'routes': routes_data,
                    'total_distance': total_distance * 1000,
                    'total_duration': total_duration * 60,
                    'segments': len(routes)
                }
            }


# Global handler instance
_chat_route_handler = None


def get_chat_route_handler() -> AIChatRouteHandler:
    """Get or create chat route handler"""
    global _chat_route_handler
    if _chat_route_handler is None:
        _chat_route_handler = AIChatRouteHandler()
    return _chat_route_handler


def process_chat_route_request(message: str, user_context: Optional[Dict] = None) -> Optional[Dict]:
    """
    Process route request from chat message
    
    Args:
        message: User's chat message
        user_context: Optional user context
        
    Returns:
        Route response dict or None if not a route request
    """
    handler = get_chat_route_handler()
    return handler.handle_route_request(message, user_context)


# Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing AI Chat Route Integration...\n")
    
    # Initialize handler
    handler = get_chat_route_handler()
    
    # Test cases
    test_messages = [
        "Show me the walking route from Sultanahmet to Galata Tower",
        "How do I get from Taksim to Grand Bazaar?",
        "Plan a route visiting Blue Mosque, Hagia Sophia, and Topkapi Palace",
        "What's the best path from my hotel to Spice Bazaar?",
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: '{message}'")
        print('='*60)
        
        result = handler.handle_route_request(message)
        
        if result:
            print(f"\n{result['message']}")
            if result['type'] == 'route':
                route_data = result['route_data']
                if route_data['single']:
                    print(f"\nRoute visualization data ready for map display")
                    print(f"Waypoints: {len(route_data['waypoints'])}")
                else:
                    print(f"\nMulti-segment route ready for map display")
                    print(f"Segments: {route_data['segments']}")
        else:
            print("❌ Not recognized as a route request")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
