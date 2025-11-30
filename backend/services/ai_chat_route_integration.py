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

# Import multi-stop route planner
try:
    from multi_stop_route_planner import (
        MultiStopRoutePlanner,
        OptimizationStrategy,
        PointOfInterest
    )
    MULTI_STOP_AVAILABLE = True
    logger.info("✅ Multi-stop route planner available")
except ImportError as e:
    MULTI_STOP_AVAILABLE = False
    logger.warning(f"⚠️ Multi-stop route planner not available: {e}")


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
        global MULTI_STOP_AVAILABLE
        
        self.route_integration = None
        self.multi_stop_planner = None
        
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
        
        if MULTI_STOP_AVAILABLE:
            try:
                self.multi_stop_planner = MultiStopRoutePlanner()
                logger.info("✅ Multi-stop planner initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize multi-stop planner: {e}")
                MULTI_STOP_AVAILABLE = False
    
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
        
        # Check if it's a multi-stop request
        is_multi_stop = self._is_multi_stop_request(message)
        
        if is_multi_stop and MULTI_STOP_AVAILABLE and self.multi_stop_planner:
            # Handle multi-stop itinerary
            return self._handle_multi_stop_request(message, user_context)
        
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
                # Multi-location route (legacy handling)
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
            'take me', 'show me', 'from', 'to',
            'plan', 'itinerary', 'visit', 'tour', 'trip'
        ]
        
        return any(keyword in message_lower for keyword in route_keywords)
    
    def _is_multi_stop_request(self, message: str) -> bool:
        """Check if message is requesting multi-stop routing (itinerary/waypoints)"""
        message_lower = message.lower()
        
        # Strong multi-stop indicators (automatically multi-stop even with 0-1 locations)
        strong_indicators = [
            'itinerary', 'plan a tour', 'plan a day', 'plan my day',
            'day trip', 'half-day tour', 'full day tour'
        ]
        
        # Check for strong indicators first
        if any(indicator in message_lower for indicator in strong_indicators):
            # Exception: if it's clearly "itinerary from X to Y" (single route)
            from_to_pattern = re.search(r'itinerary\s+from\s+\w+.*\s+to\s+\w+', message_lower)
            if from_to_pattern and not any(word in message_lower for word in ['then', 'also', 'and then', ',']):
                return False
            return True
        
        # Keywords that indicate multi-stop planning (need 2+ locations)
        multi_stop_keywords = [
            'tour', 'visit all', 'visit multiple', 'multiple', 'several',
            'visiting', 'see all', 'stops', 'waypoints', 'trip'
        ]
        
        # Check for "from X to Y" pattern - this is single route
        from_to_pattern = re.search(r'\b(from|starting at|leave from)\s+\w+.*\b(to|going to|heading to)\s+\w+', message_lower)
        if from_to_pattern and not any(keyword in message_lower for keyword in ['then', 'also', 'and then']):
            return False
        
        # Check for connecting words that indicate multiple locations
        multi_location_patterns = [
            'and', 'then', 'also', 'plus', 'along with',
            ',', 'next', 'after that'
        ]
        
        # Count location mentions
        location_count = sum(1 for loc in self.KNOWN_LOCATIONS.keys() if loc in message_lower)
        
        # Detect list pattern: "A, B, and C" or "A, B, C"
        has_comma_list = ',' in message and location_count >= 2
        
        # Check for area/district references (suggests multiple locations)
        area_references = ['in sultanahmet', 'in beyoglu', 'in taksim', 'area', 'district', 'neighborhood']
        has_area_reference = any(ref in message_lower for ref in area_references)
        
        # Multi-stop conditions:
        has_multi_keyword = any(keyword in message_lower for keyword in multi_stop_keywords)
        has_multi_location_pattern = any(pattern in message_lower for pattern in multi_location_patterns) and location_count >= 2
        
        # Return True if:
        # 1. 3+ specific locations mentioned
        # 2. Multi-stop keyword + 2+ locations
        # 3. Multi-stop keyword + area reference (implies multiple POIs)
        # 4. Comma list with 2+ locations
        # 5. Multiple locations with connecting words
        return (location_count >= 3) or \
               (has_multi_keyword and location_count >= 2) or \
               (has_multi_keyword and has_area_reference) or \
               has_comma_list or \
               (has_multi_location_pattern and location_count >= 2)
    
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
    
    def _extract_poi_names(self, message: str) -> List[str]:
        """Extract POI names from message using natural language processing"""
        message_lower = message.lower()
        found_pois = []
        
        # Try to find known locations in the message
        # Sort by length (longest first) to match "Blue Mosque" before "Mosque"
        sorted_locations = sorted(self.KNOWN_LOCATIONS.keys(), key=len, reverse=True)
        
        for location_name in sorted_locations:
            if location_name in message_lower:
                # Avoid duplicates
                if location_name not in found_pois:
                    found_pois.append(location_name)
        
        # Also try the POI database if multi-stop planner is available
        if self.multi_stop_planner:
            poi_db = self.multi_stop_planner.poi_database
            for poi_key, poi in poi_db.items():
                poi_name_lower = poi.name.lower()
                if poi_name_lower in message_lower and poi_name_lower not in found_pois:
                    found_pois.append(poi.name)
        
        # If no specific POIs found, check for area/district references with keywords
        if len(found_pois) < 2:
            # Area-based suggestions
            area_mappings = {
                'sultanahmet': ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace', 'Basilica Cistern'],
                'beyoglu': ['Galata Tower', 'Istiklal Street', 'Taksim Square'],
                'taksim': ['Taksim Square', 'Istiklal Street'],
                'bosphorus': ['Dolmabahce Palace', 'Maiden Tower'],
                'shopping': ['Grand Bazaar', 'Spice Bazaar'],
                'museums': ['Hagia Sophia', 'Topkapi Palace', 'Basilica Cistern']
            }
            
            # Check for area keywords
            for area, pois in area_mappings.items():
                if area in message_lower:
                    # Add main attractions from that area (limit to 3-4)
                    for poi in pois[:4]:
                        if poi not in found_pois:
                            found_pois.append(poi)
                    break  # Only use first matching area
        
        return found_pois
    
    def _handle_multi_stop_request(
        self,
        message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle multi-stop itinerary planning request"""
        try:
            # Extract POI names from message
            poi_names = self._extract_poi_names(message)
            
            if len(poi_names) < 2:
                return {
                    'type': 'error',
                    'message': "I need at least 2 locations to plan an itinerary. Please mention specific places you want to visit, like 'Hagia Sophia, Blue Mosque, and Grand Bazaar'."
                }
            
            # Determine optimization strategy
            strategy = OptimizationStrategy.SHORTEST_TOTAL_TIME
            message_lower = message.lower()
            
            if 'accessible' in message_lower or 'wheelchair' in message_lower:
                strategy = OptimizationStrategy.ACCESSIBLE_FIRST
            elif 'distance' in message_lower or 'shortest distance' in message_lower:
                strategy = OptimizationStrategy.SHORTEST_TOTAL_DISTANCE
            elif 'nearest' in message_lower:
                strategy = OptimizationStrategy.NEAREST_NEIGHBOR
            
            # Extract preferences from user context
            preferences = []
            if user_context and user_context.get('accessible_mode'):
                preferences.append('accessible')
            
            # Plan multi-stop itinerary
            logger.info(f"Planning multi-stop itinerary for: {poi_names}")
            itinerary = self.multi_stop_planner.plan_multi_stop_route(
                poi_names=poi_names,
                strategy=strategy,
                preferences=preferences
            )
            
            # Format response
            return self._format_multi_stop_response(itinerary, message)
            
        except Exception as e:
            logger.error(f"Error planning multi-stop itinerary: {e}", exc_info=True)
            return {
                'type': 'error',
                'message': f"Sorry, I couldn't plan your itinerary: {str(e)}"
            }
    
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
    
    def _format_multi_stop_response(
        self,
        itinerary: Any,  # MultiStopItinerary
        original_message: str
    ) -> Dict[str, Any]:
        """Format multi-stop itinerary response for chat"""
        
        # Build response message
        message = "🗺️ **Multi-Stop Itinerary Planned!**\n\n"
        
        # Summary
        message += f"📍 **Stops:** {len(itinerary.stops)}\n"
        message += f"📏 **Total Distance:** {itinerary.total_distance_km:.2f} km\n"
        message += f"🚶 **Travel Time:** {itinerary.total_travel_time_minutes} min\n"
        message += f"⏱️ **Visit Time:** {itinerary.total_visit_time_minutes} min\n"
        message += f"⏰ **Total Time:** {itinerary.total_time_minutes} min (~{itinerary.total_time_minutes/60:.1f} hours)\n"
        message += f"💰 **Estimated Cost:** {itinerary.total_cost_tl:.2f} TL\n"
        
        if itinerary.optimization_strategy:
            strategy_name = itinerary.optimization_strategy.value.replace('_', ' ').title()
            message += f"🎯 **Strategy:** {strategy_name}\n"
        
        # Timeline
        message += "\n**📅 Itinerary Timeline:**\n"
        timeline = itinerary.get_timeline()
        for item in timeline[:10]:  # Show first 10 items to avoid too long message
            if item['type'] == 'arrival':
                message += f"  🏛️ **{item['time']}** - Arrive at {item['location']}\n"
            elif item['type'] == 'visit':
                message += f"     ⏱️ Visit for {item['duration']} min\n"
            elif item['type'] == 'travel':
                modes = ', '.join(item['modes'][:2])
                message += f"  🚶 **{item['time']}** - Travel to {item['to']} ({modes})\n"
        
        if len(timeline) > 10:
            message += f"\n  ... and {len(timeline) - 10} more steps\n"
        
        # Highlights
        if itinerary.highlights:
            message += "\n**✨ Highlights:**\n"
            for highlight in itinerary.highlights[:5]:
                message += f"  • {highlight}\n"
        
        # Warnings
        if itinerary.warnings:
            message += "\n**⚠️ Important Notes:**\n"
            for warning in itinerary.warnings[:3]:
                message += f"  • {warning}\n"
        
        # Accessibility
        if itinerary.accessibility_friendly:
            message += "\n♿ **Accessibility:** This route includes accessible options\n"
        
        message += "\n📍 The complete itinerary is displayed on the map above."
        
        # Prepare route data for visualization
        route_data = {
            'type': 'multi_stop_itinerary',
            'stops': [
                {
                    'name': stop.name,
                    'coordinates': stop.coordinates,
                    'category': stop.category,
                    'duration': stop.suggested_duration_minutes,
                    'accessibility': stop.accessibility_level
                }
                for stop in itinerary.stops
            ],
            'segments': [
                {
                    'from': seg.from_poi.name,
                    'to': seg.to_poi.name,
                    'distance_km': seg.distance_km,
                    'duration_min': seg.duration_minutes,
                    'modes': seg.modes_used,
                    'cost_tl': seg.cost_tl
                }
                for seg in itinerary.route_segments
            ],
            'summary': {
                'total_stops': len(itinerary.stops),
                'total_distance_km': itinerary.total_distance_km,
                'total_travel_time_min': itinerary.total_travel_time_minutes,
                'total_visit_time_min': itinerary.total_visit_time_minutes,
                'total_time_min': itinerary.total_time_minutes,
                'total_cost_tl': itinerary.total_cost_tl,
                'strategy': itinerary.optimization_strategy.value,
                'accessibility_friendly': itinerary.accessibility_friendly
            },
            'timeline': timeline
        }
        
        return {
            'type': 'multi_stop_itinerary',
            'message': message,
            'route_data': route_data
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
        "I want to visit Hagia Sophia, Blue Mosque, Grand Bazaar, and Spice Bazaar today",
        "Create an itinerary for Topkapi Palace, Basilica Cistern, and Dolmabahce Palace",
        "Plan my day: start at Galata Tower, then Istiklal Street, then Taksim Square",
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
