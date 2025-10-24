"""
Routing Service Adapter for Chat Integration
Bridges the journey planner with the chat system
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import the routing components
try:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from services.journey_planner import JourneyPlanner, JourneyRequest
    from services.route_network_builder import TransportationNetwork, TransportStop, TransportLine
    ROUTING_AVAILABLE = True
    logger.info("✅ Routing system components loaded successfully")
except ImportError as e:
    ROUTING_AVAILABLE = False
    logger.warning(f"⚠️ Routing system not available: {e}")

# Import enhanced ML transportation system for better location extraction
try:
    from enhanced_transportation_integration import TransportationQueryProcessor
    ML_TRANSPORT_AVAILABLE = True
    logger.info("✅ Enhanced ML transportation system available for location extraction")
except ImportError as e:
    ML_TRANSPORT_AVAILABLE = False
    logger.warning(f"⚠️ ML transportation system not available: {e}")


class RoutingServiceAdapter:
    """
    Adapter to connect the graph-based routing system with the chat interface
    """
    
    def __init__(self):
        """Initialize the routing service with the major routes network"""
        self.network = None
        self.journey_planner = None
        self.is_initialized = False
        self.ml_processor = None
        
        # Initialize ML-enhanced location extraction if available
        if ML_TRANSPORT_AVAILABLE:
            try:
                self.ml_processor = TransportationQueryProcessor()
                logger.info("🧠 ML-enhanced location extraction enabled")
            except Exception as e:
                logger.warning(f"⚠️ ML processor initialization failed: {e}")
                self.ml_processor = None
        
        if ROUTING_AVAILABLE:
            try:
                self._load_network()
                if self.network:
                    self.journey_planner = JourneyPlanner(self.network)
                    self.is_initialized = True
                    logger.info(f"🚇 Routing service initialized: {len(self.network.stops)} stops, {len(self.network.lines)} lines")
            except Exception as e:
                logger.error(f"❌ Failed to initialize routing service: {e}")
                self.is_initialized = False
    
    def _load_network(self):
        """Load the transportation network from the saved file"""
        try:
            network_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'major_routes_network.json'
            )
            
            if not os.path.exists(network_file):
                logger.warning(f"⚠️ Network file not found: {network_file}")
                return
            
            with open(network_file, 'r', encoding='utf-8') as f:
                network_data = json.load(f)
            
            self.network = TransportationNetwork()
            
            # Load stops
            for stop_id, stop_data in network_data['stops'].items():
                stop = TransportStop(
                    stop_id=stop_id,
                    name=stop_data['name'],
                    lat=stop_data['lat'],
                    lon=stop_data['lon'],
                    transport_type=stop_data['type']
                )
                self.network.add_stop(stop)
            
            # Load lines
            for line_id, line_data in network_data['lines'].items():
                line = TransportLine(
                    line_id=line_id,
                    name=line_data['name'],
                    transport_type=line_data['type'],
                    stops=line_data['stops'],
                    color=line_data.get('color', '#000000')
                )
                self.network.add_line(line)
            
            # Build network edges
            self.network.build_network()
            
            # Load transfers
            if 'transfers' in network_data:
                for transfer_data in network_data['transfers']:
                    self.network.add_transfer(
                        from_stop_id=transfer_data['from_stop_id'],
                        to_stop_id=transfer_data['to_stop_id'],
                        transfer_type="same_station" if transfer_data['is_step_free'] else "walking",
                        walking_meters=transfer_data['walking_distance'],
                        duration_minutes=transfer_data['walking_time']
                    )
                logger.info(f"✓ Loaded {len(network_data['transfers'])} transfers")
            
            logger.info(f"✓ Network loaded: {len(self.network.stops)} stops, {len(self.network.lines)} lines, {len(self.network.edges)} edges")
            
        except Exception as e:
            logger.error(f"❌ Failed to load network: {e}")
            self.network = None
    
    def is_routing_query(self, query: str) -> bool:
        """
        Detect if a query is asking for routing/directions
        """
        # Routing-specific keywords
        routing_keywords = [
            # English - route planning
            'how do i get', 'how to get', 'how can i go', 'how can i get',
            'route', 'directions', 'way to', 'path to',
            'from', 'to', 'go to', 'get to', 'take me', 'travel to',
            'navigate', 'journey from', 'best way to',
            
            # Turkish - route planning
            'nasıl gidebilirim', 'nasıl giderim', 'nasıl ulaşırım', 'nasıl ulaşabilirim',
            'nereden', 'nereye', 'yol', 'güzergah', 'rota',
            'aktarma yaparak', 'ile gitmek',
        ]
        
        # Exclude general info queries (not routing)
        info_keywords = [
            'what is', 'what are', 'tell me about', 'explain', 'describe',
            'which lines', 'what lines', 'how does', 'how many',
            'list', 'show me all', 'available', 'operating hours',
            'nedir', 'nelerdir', 'anlat', 'açıkla', 'hangi hatlar',
        ]
        
        query_lower = query.lower()
        
        # Check for info keywords first (higher priority)
        if any(keyword in query_lower for keyword in info_keywords):
            # This is likely a general info query, not routing
            return False
        
        # Check for routing keywords
        if any(keyword in query_lower for keyword in routing_keywords):
            return True
        
        # Additional check: does it contain "from X to Y" pattern?
        import re
        from_to_pattern = r'(?:from|dan|den)\s+\w+\s+(?:to|a|e)\s+\w+'
        if re.search(from_to_pattern, query_lower):
            return True
        
        return False
    
    def extract_locations(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract origin and destination from a routing query
        Uses ML-enhanced extraction if available, falls back to regex patterns
        """
        origin = None
        destination = None
        
        # Priority 1: Try ML-enhanced location extraction
        if self.ml_processor:
            try:
                ml_result = self._extract_locations_ml(query)
                if ml_result['origin'] or ml_result['destination']:
                    logger.info(f"🧠 ML extraction found: {ml_result['origin']} → {ml_result['destination']}")
                    return ml_result
            except Exception as e:
                logger.warning(f"⚠️ ML extraction failed: {e}, falling back to regex")
        
        # Priority 2: Regex-based pattern matching
        import re
        query_lower = query.lower()
        
        # Common patterns
        patterns = [
            # English - "from X to Y"
            (r'from\s+([^to]+?)\s+to\s+(.+?)(?:\s+by|\s+via|\s+mosque|\s+square|\?|$)', 'origin_dest'),
            # English - "get to X" or "go to X"  
            (r'(?:get|go)\s+to\s+(.+?)(?:\s+from|\s+mosque|\s+square|\?|$)', 'dest_only'),
            # English - "how do I/can I get from X to Y"
            (r'how\s+(?:do\s+i|can\s+i|to)\s+(?:get|go)(?:\s+from)?\s+(.+?)\s+to\s+(.+?)(?:\s+by|\s+via|\s+mosque|\s+square|\?|$)', 'origin_dest'),
            # English - "X to Y"
            (r'^(.+?)\s+to\s+(.+?)(?:\s+by|\s+via|\s+mosque|\s+square|\?|$)', 'origin_dest'),
            
            # Turkish - "X'den Y'e"
            (r'(\w+)\'?(?:dan|den)\s+(\w+)\'?(?:a|e)(?:\s+nasıl|\s+ne|\s+cami|\s+meydan|\?|$)', 'origin_dest'),
            (r'(\w+)(?:\'dan|\'den|dan|den)\s+(\w+)(?:\'a|\'e|a|e)(?:\s+nasıl|\s+ne|\s+cami|\s+meydan|\?|$)', 'origin_dest'),
        ]
        
        # Try to match patterns
        for pattern, pattern_type in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                if pattern_type == 'origin_dest' and len(match.groups()) >= 2:
                    origin = match.group(1).strip()
                    destination = match.group(2).strip()
                    # Clean up location names (remove "mosque", "square" etc)
                    origin = self._clean_location_name(origin)
                    destination = self._clean_location_name(destination)
                    break
                elif pattern_type == 'dest_only':
                    destination = match.group(1).strip()
                    destination = self._clean_location_name(destination)
                    break
        
        # Priority 3: Fallback - Look for known locations in the query
        if not origin and not destination and self.network:
            # Get all stop names
            location_names = [stop.name.lower() for stop in self.network.stops.values()]
            
            # Find locations mentioned in query
            found_locations = []
            for stop in self.network.stops.values():
                stop_name_lower = stop.name.lower()
                # Check for exact match or close match
                if stop_name_lower in query_lower or query_lower in stop_name_lower:
                    found_locations.append(stop.name)
            
            # If we found exactly 2 locations, assume first is origin, second is dest
            if len(found_locations) >= 2:
                origin = found_locations[0]
                destination = found_locations[1]
            elif len(found_locations) == 1:
                # Only one location found, assume it's the destination
                destination = found_locations[0]
        
        logger.info(f"📍 Extracted: {origin} → {destination}")
        return {
            'origin': origin,
            'destination': destination
        }
    
    def _extract_locations_ml(self, query: str) -> Dict[str, Optional[str]]:
        """
        Use ML-enhanced transportation system to extract locations
        This leverages the comprehensive location understanding
        """
        import asyncio
        
        # Use async processing
        try:
            # Get the event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Process query with ML system
            result = loop.run_until_complete(
                self.ml_processor.process_transportation_query_async(query)
            )
            
            # Parse the ML result to extract origin/destination
            # The ML system returns formatted text, we need to extract locations
            origin = None
            destination = None
            
            if isinstance(result, str):
                # Look for location names in the response
                import re
                
                # Pattern 1: "Taksim → Sultanahmet" (fix regex)
                arrow_match = re.search(r'(\w+(?:\s+\w+)?)\s*(?:→|->|–|—)\s*(\w+(?:\s+\w+)?)', result)
                if arrow_match:
                    origin = arrow_match.group(1).strip()
                    destination = arrow_match.group(2).strip()
                
                # Pattern 2: Look for "from X to Y" in response
                if not origin or not destination:
                    from_to_match = re.search(r'from\s+([^to]+?)\s+to\s+(\w+(?:\s+\w+)?)', result, re.IGNORECASE)
                    if from_to_match:
                        origin = from_to_match.group(1).strip()
                        destination = from_to_match.group(2).strip()
            
            return {
                'origin': origin,
                'destination': destination
            }
        except Exception as e:
            logger.warning(f"⚠️ ML location extraction error: {e}")
            return {'origin': None, 'destination': None}
    
    def _clean_location_name(self, location: str) -> str:
        """
        Clean location name by removing common suffixes like 'mosque', 'square'
        """
        if not location:
            return location
        
        # Remove common suffixes
        suffixes_to_remove = [
            'mosque', 'cami', 'camii',
            'square', 'meydanı', 'meydan',
            'station', 'istasyon', 'durağı', 'durak',
            'metro', 'metrosu',
            'stop', 'stops'
        ]
        
        location_lower = location.lower().strip()
        for suffix in suffixes_to_remove:
            if location_lower.endswith(suffix):
                location = location[:-(len(suffix))].strip()
                break
        
        return location.title() if location else location
    
    def plan_route(self, origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """
        Plan a route between origin and destination
        
        Returns:
            Route information dictionary or None if routing fails
        """
        if not self.is_initialized:
            return None
        
        try:
            # Create journey request
            request = JourneyRequest(
                origin=origin,
                destination=destination
            )
            
            # Plan the journey
            plan = self.journey_planner.plan_journey(request)
            
            if not plan:
                return None
            
            # Convert to chat-friendly format
            journey = plan.primary_journey
            
            result = {
                'success': True,
                'origin': origin,
                'destination': destination,
                'duration_minutes': journey.total_duration_minutes,
                'distance_km': journey.total_distance_km,
                'transfers': journey.total_transfers,
                'cost_tl': journey.estimated_cost_tl,
                'quality_score': journey.quality_score,
                'segments': []
            }
            
            # Add segments with coordinates
            for seg in journey.segments:
                # Get stop coordinates from network
                from_stop = self.network.stops.get(seg.from_stop)
                to_stop = self.network.stops.get(seg.to_stop)
                
                segment_data = {
                    'type': seg.transport_type,
                    'line': seg.line_id,
                    'line_name': seg.line_name,
                    'from_stop': seg.from_stop_name,
                    'to_stop': seg.to_stop_name,
                    'duration_minutes': seg.duration_minutes,
                    'stops_count': seg.stops_count
                }
                
                # Add coordinates if available
                if from_stop:
                    segment_data['from_coordinates'] = [from_stop.lat, from_stop.lon]
                if to_stop:
                    segment_data['to_coordinates'] = [to_stop.lat, to_stop.lon]
                
                result['segments'].append(segment_data)
            
            # Add alternatives if available
            if plan.alternative_journeys:
                result['alternatives'] = []
                for alt in plan.alternative_journeys[:2]:  # Top 2 alternatives
                    result['alternatives'].append({
                        'duration_minutes': alt.total_duration_minutes,
                        'transfers': alt.total_transfers,
                        'cost_tl': alt.estimated_cost_tl
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Route planning failed: {e}")
            return None
    
    def format_route_response(self, route_info: Dict[str, Any]) -> str:
        """
        Format route information into a user-friendly chat response
        """
        if not route_info or not route_info.get('success'):
            return "I couldn't find a route for that journey. Please check the location names and try again."
        
        # Build the response
        response_parts = []
        
        # Header
        response_parts.append(f"🗺️ **Route from {route_info['origin']} to {route_info['destination']}**\n")
        
        # Summary
        response_parts.append(f"⏱️ **Duration:** {route_info['duration_minutes']:.0f} minutes")
        response_parts.append(f"📏 **Distance:** {route_info['distance_km']:.1f} km")
        response_parts.append(f"🔄 **Transfers:** {route_info['transfers']}")
        response_parts.append(f"💰 **Estimated Cost:** ₺{route_info['cost_tl']:.2f}\n")
        
        # Route segments
        response_parts.append("**🚇 Your Journey:**")
        for i, seg in enumerate(route_info['segments'], 1):
            transport_emoji = {
                'metro': '🚇',
                'bus': '🚌',
                'ferry': '⛴️',
                'tram': '🚊'
            }.get(seg['type'], '🚆')
            
            response_parts.append(
                f"{i}. {transport_emoji} **{seg['line_name']}**\n"
                f"   From: {seg['from_stop']} → To: {seg['to_stop']}\n"
                f"   Duration: {seg['duration_minutes']:.0f} min | {seg['stops_count']} stops"
            )
        
        # Alternatives
        if route_info.get('alternatives'):
            response_parts.append("\n**Alternative Routes:**")
            for i, alt in enumerate(route_info['alternatives'], 1):
                response_parts.append(
                    f"{i}. {alt['duration_minutes']:.0f} min, {alt['transfers']} transfers, ₺{alt['cost_tl']:.2f}"
                )
        
        return "\n".join(response_parts)
    
    def process_routing_query(self, query: str) -> Optional[str]:
        """
        Main method to process a routing query and return a formatted response
        
        Args:
            query: User's routing question
        
        Returns:
            Formatted routing response or None if not a routing query
        """
        if not self.is_initialized:
            return None
        
        if not self.is_routing_query(query):
            return None
        
        # Extract locations
        locations = self.extract_locations(query)
        origin = locations.get('origin')
        destination = locations.get('destination')
        
        if not destination:
            return "I can help you find a route! Please tell me where you want to go. For example: 'How do I get from Taksim to Kadıköy?'"
        
        if not origin:
            # Ask for origin
            return f"Great! You want to go to {destination}. Where are you starting from?"
        
        # Plan the route
        route_info = self.plan_route(origin, destination)
        
        if not route_info:
            return f"I couldn't find a route from {origin} to {destination}. These locations might not be in my current network, or there's no direct connection. Please try different location names."
        
        # Format and return response
        return self.format_route_response(route_info)
    
    def get_route_map_data(self, origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """
        Get map visualization data for a route
        """
        if not self.is_initialized:
            return None
        
        try:
            from services.route_map_visualizer import get_map_visualizer
            route_info = self.plan_route(origin, destination)
            if not route_info:
                return None
            map_visualizer = get_map_visualizer()
            return map_visualizer.generate_route_map_data(route_info)
        except Exception as e:
            logger.error(f"Failed to generate map data: {e}")
            return None
    
    def generate_route_map_html(self, origin: str, destination: str) -> Optional[str]:
        """
        Generate complete HTML page with interactive map
        """
        if not self.is_initialized:
            return None
        
        try:
            from services.route_map_visualizer import get_map_visualizer
            route_info = self.plan_route(origin, destination)
            if not route_info:
                return None
            map_visualizer = get_map_visualizer()
            return map_visualizer.generate_map_html(route_info)
        except Exception as e:
            logger.error(f"Failed to generate map HTML: {e}")
            return None


# Global instance
_routing_service = None

def get_routing_service() -> Optional[RoutingServiceAdapter]:
    """Get or create the global routing service instance"""
    global _routing_service
    if _routing_service is None and ROUTING_AVAILABLE:
        _routing_service = RoutingServiceAdapter()
    return _routing_service
