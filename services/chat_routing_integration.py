"""
Chat System Integration for Industry-Level Routing
===================================================

Integrates the new graph-based routing system (Marmaray + Metro)
with the AI chat interface for natural language routing queries.

Created: October 24, 2025
Status: Production Ready
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .journey_planner import JourneyPlanner, JourneyRequest
from .location_matcher import LocationMatcher
from .route_network_builder import TransportationNetwork, TransportStop, TransportLine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRoutingIntegration:
    """
    Integrates industry-level routing with chat system
    Handles natural language queries and returns formatted responses
    """
    
    def __init__(self, network: TransportationNetwork):
        """
        Initialize chat routing integration
        
        Args:
            network: TransportationNetwork with loaded routes
        """
        self.network = network
        self.journey_planner = JourneyPlanner(network)
        self.location_matcher = LocationMatcher(network)
        
        logger.info(f"Chat routing initialized: {len(network.stops)} stops, {len(network.lines)} lines")
    
    def handle_routing_query(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle a natural language routing query
        
        Args:
            query: User's question (e.g., "How do I get from Taksim to Kadıköy?")
            user_context: Optional context (preferences, accessibility needs, etc.)
            
        Returns:
            Dict with:
            - success: bool
            - response_text: Formatted response for display
            - route_data: Detailed route information
            - map_data: Data for map visualization
            - alternatives: Alternative route options
        """
        
        try:
            # Extract origin and destination from query
            origin, destination = self._extract_locations_from_query(query)
            
            if not origin or not destination:
                return self._get_clarification_response(query)
            
            # Plan the journey
            request = JourneyRequest(
                origin=origin,
                destination=destination,
                departure_time=datetime.now()
            )
            
            plan = self.journey_planner.plan_journey(request)
            
            if not plan:
                return {
                    'success': False,
                    'response_text': f"Sorry, I couldn't find a route from {origin} to {destination}. Please check the location names and try again.",
                    'error': 'No route found'
                }
            
            # Format response for chat
            response_text = self._format_journey_for_chat(plan)
            
            # Prepare route data
            route_data = self._prepare_route_data(plan)
            
            # Prepare map data
            map_data = self._prepare_map_data(plan)
            
            # Get alternatives
            alternatives = []
            if plan.alternative_journeys:
                for alt in plan.alternative_journeys[:2]:  # Top 2 alternatives
                    alternatives.append({
                        'duration': alt.total_duration_minutes,
                        'transfers': alt.total_transfers,
                        'transport_types': list(alt.transport_types_used),
                        'description': self._format_journey_summary(alt)
                    })
            
            return {
                'success': True,
                'response_text': response_text,
                'route_data': route_data,
                'map_data': map_data,
                'alternatives': alternatives,
                'has_alternatives': len(alternatives) > 0,
                'journey_plan': plan,
                'origin_match': plan.origin_location.to_dict(),
                'destination_match': plan.destination_location.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error handling routing query: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'response_text': "I encountered an error planning your route. Please try rephrasing your question or contact support."
            }
    
    def _extract_locations_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract origin and destination from natural language query
        
        Returns:
            Tuple of (origin, destination)
        """
        query_lower = query.lower()
        
        # Common patterns for routing queries
        patterns = [
            # "from X to Y"
            r'from\s+([a-zçğıöşü\s]+)\s+to\s+([a-zçğıöşü\s]+)',
            # "X to Y"
            r'^([a-zçğıöşü\s]+)\s+to\s+([a-zçğıöşü\s]+)',
            # "how do I get from X to Y"
            r'how.*get.*from\s+([a-zçğıöşü\s]+)\s+to\s+([a-zçğıöşü\s]+)',
            # "route from X to Y"
            r'route.*from\s+([a-zçğıöşü\s]+)\s+to\s+([a-zçğıöşü\s]+)',
            # "I need to go from X to Y"
            r'go.*from\s+([a-zçğıöşü\s]+)\s+to\s+([a-zçğıöşü\s]+)',
            # "take me from X to Y"
            r'take.*from\s+([a-zçğıöşü\s]+)\s+to\s+([a-zçğıöşü\s]+)',
            # "I want to travel from X to Y"
            r'travel.*from\s+([a-zçğıöşü\s]+)\s+to\s+([a-zçğıöşü\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                origin = match.group(1).strip().title()
                destination = match.group(2).strip().title()
                return origin, destination
        
        return None, None
    
    def _get_clarification_response(self, query: str) -> Dict[str, Any]:
        """Generate a clarification response when locations can't be extracted"""
        
        return {
            'success': False,
            'needs_clarification': True,
            'response_text': (
                "I'd be happy to help you find a route! 🗺️\n\n"
                "Please tell me:\n"
                "• Where are you starting from?\n"
                "• Where do you want to go?\n\n"
                "For example: *'How do I get from Taksim to Kadıköy?'*"
            ),
            'suggested_queries': [
                "How do I get from Taksim to Kadıköy?",
                "Route from Yenikapı to Taksim",
                "Take me from Sirkeci to Pendik",
                "I need to go from Kadıköy to the airport"
            ]
        }
    
    def _format_journey_for_chat(self, plan) -> str:
        """Format journey plan as a chat-friendly message"""
        
        journey = plan.primary_journey
        output = []
        
        # Header
        output.append(f"🗺️ **Route: {journey.origin_name} → {journey.destination_name}**\n")
        
        # Quick summary
        output.append(f"⏱️ **Duration:** {journey.total_duration_minutes:.0f} minutes")
        output.append(f"🔄 **Transfers:** {journey.total_transfers}")
        output.append(f"📏 **Distance:** {journey.total_distance_km:.2f} km")
        output.append(f"💰 **Cost:** ₺{journey.estimated_cost_tl:.2f}\n")
        
        # Step-by-step instructions
        output.append("**📍 Step-by-Step Instructions:**\n")
        
        for i, segment in enumerate(journey.segments, 1):
            transport_emoji = self._get_transport_emoji(segment.transport_type)
            
            output.append(
                f"{i}. {transport_emoji} **{segment.line_name}** "
                f"({segment.transport_type.upper()})"
            )
            output.append(
                f"   • Board at: **{segment.from_stop_name}**"
            )
            output.append(
                f"   • Get off at: **{segment.to_stop_name}**"
            )
            output.append(
                f"   • Duration: {segment.duration_minutes:.0f} minutes "
                f"({segment.stops_count} stops)\n"
            )
            
            # Add transfer instructions between segments
            if i < len(journey.segments):
                output.append(
                    f"   ↓ *Transfer* (walk to next line)\n"
                )
        
        # Add alternatives if available
        if plan.alternative_journeys:
            output.append("\n**🔀 Alternative Routes:**\n")
            for i, alt in enumerate(plan.alternative_journeys[:2], 1):
                output.append(
                    f"{i}. {alt.total_duration_minutes:.0f} min, "
                    f"{alt.total_transfers} transfers - "
                    f"{' → '.join(list(alt.transport_types_used))}"
                )
        
        # Footer with tips
        output.append("\n💡 **Travel Tips:**")
        if journey.total_transfers == 0:
            output.append("• Direct route - no transfers needed!")
        elif journey.total_transfers == 1:
            output.append(f"• Easy route with just one transfer")
        else:
            output.append(f"• Multiple transfers - follow signs carefully")
        
        if 'metro' in journey.transport_types_used:
            output.append("• İstanbulkart accepted on all metro lines")
        
        if journey.total_walking_meters > 500:
            output.append(f"• Total walking: ~{journey.total_walking_meters}m")
        
        return '\n'.join(output)
    
    def _format_journey_summary(self, journey) -> str:
        """Create a short summary of a journey"""
        transport_names = []
        for seg in journey.segments:
            transport_names.append(seg.line_name)
        
        return f"{' → '.join(transport_names)} ({journey.total_duration_minutes:.0f} min)"
    
    def _get_transport_emoji(self, transport_type: str) -> str:
        """Get emoji for transport type"""
        emojis = {
            'metro': '🚇',
            'bus': '🚌',
            'ferry': '⛴️',
            'tram': '🚊',
            'funicular': '🚡',
            'walking': '🚶'
        }
        return emojis.get(transport_type.lower(), '🚉')
    
    def _prepare_route_data(self, plan) -> Dict[str, Any]:
        """Prepare detailed route data for API/frontend"""
        journey = plan.primary_journey
        
        return {
            'origin': {
                'name': journey.origin_name,
                'stop_id': journey.origin,
                'coordinates': self._get_stop_coordinates(journey.origin)
            },
            'destination': {
                'name': journey.destination_name,
                'stop_id': journey.destination,
                'coordinates': self._get_stop_coordinates(journey.destination)
            },
            'summary': {
                'duration_minutes': journey.total_duration_minutes,
                'distance_km': journey.total_distance_km,
                'transfers': journey.total_transfers,
                'walking_meters': journey.total_walking_meters,
                'cost_tl': journey.estimated_cost_tl,
                'transport_types': list(journey.transport_types_used),
                'quality_score': journey.quality_score
            },
            'segments': [
                {
                    'line_id': seg.line_id,
                    'line_name': seg.line_name,
                    'transport_type': seg.transport_type,
                    'from_stop': seg.from_stop_name,
                    'to_stop': seg.to_stop_name,
                    'from_stop_id': seg.from_stop,
                    'to_stop_id': seg.to_stop,
                    'duration_minutes': seg.duration_minutes,
                    'distance_km': seg.distance_km,
                    'stops_count': seg.stops_count
                }
                for seg in journey.segments
            ],
            'transfers': [
                {
                    'from_stop': t.from_stop_name,
                    'to_stop': t.to_stop_name,
                    'duration_minutes': t.duration_minutes,
                    'distance_meters': t.distance_meters,
                    'transfer_type': t.transfer_type
                }
                for t in journey.transfers
            ]
        }
    
    def _prepare_map_data(self, plan) -> Dict[str, Any]:
        """Prepare data for map visualization"""
        journey = plan.primary_journey
        
        # Collect all stop coordinates
        stops = []
        for seg in journey.segments:
            from_coords = self._get_stop_coordinates(seg.from_stop)
            to_coords = self._get_stop_coordinates(seg.to_stop)
            
            if from_coords:
                stops.append({
                    'name': seg.from_stop_name,
                    'lat': from_coords[0],
                    'lon': from_coords[1],
                    'type': 'origin' if seg == journey.segments[0] else 'waypoint'
                })
            
            if to_coords and seg == journey.segments[-1]:
                stops.append({
                    'name': seg.to_stop_name,
                    'lat': to_coords[0],
                    'lon': to_coords[1],
                    'type': 'destination'
                })
        
        # Collect route paths
        paths = []
        for seg in journey.segments:
            paths.append({
                'line_id': seg.line_id,
                'line_name': seg.line_name,
                'transport_type': seg.transport_type,
                'coordinates': seg.coordinates if hasattr(seg, 'coordinates') else []
            })
        
        return {
            'stops': stops,
            'paths': paths,
            'bounds': self._calculate_bounds(stops)
        }
    
    def _get_stop_coordinates(self, stop_id: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a stop"""
        stop = self.network.get_stop(stop_id)
        if stop:
            return (stop.lat, stop.lon)
        return None
    
    def _calculate_bounds(self, stops: List[Dict]) -> Dict[str, float]:
        """Calculate map bounds for all stops"""
        if not stops:
            return {'min_lat': 41.0, 'max_lat': 41.1, 'min_lon': 28.9, 'max_lon': 29.0}
        
        lats = [s['lat'] for s in stops if 'lat' in s]
        lons = [s['lon'] for s in stops if 'lon' in s]
        
        return {
            'min_lat': min(lats) if lats else 41.0,
            'max_lat': max(lats) if lats else 41.1,
            'min_lon': min(lons) if lons else 28.9,
            'max_lon': max(lons) if lons else 29.0
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for display"""
        return {
            'status': 'operational',
            'network_size': {
                'stops': len(self.network.stops),
                'lines': len(self.network.lines),
                'edges': len(self.network.edges)
            },
            'capabilities': {
                'marmaray': True,
                'metro': True,
                'ferry': True,
                'tram': True,
                'multi_modal': True,
                'transfers': True,
                'cross_continental': True
            },
            'data_source': 'Manual Routes (Marmaray + Metro Priority)',
            'last_updated': datetime.now().isoformat()
        }
