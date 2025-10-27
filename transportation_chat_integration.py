#!/usr/bin/env python3
"""
Transfer Instructions & Map Visualization Integration for AI Chat
=================================================================

Integrates the Google Maps-style transfer instructions and interactive map
visualization into the Istanbul AI chat system.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Import our new transfer instructions and map visualization features
try:
    from services.live_ibb_transportation_service import LiveIBBTransportationService
    from services.transfer_instructions_generator import TransferInstructionsGenerator, TransferInstruction
    TRANSFER_MAP_AVAILABLE = True
    logger.info("✅ Transfer Instructions & Map Visualization loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Transfer Instructions & Map Visualization not available: {e}")
    TRANSFER_MAP_AVAILABLE = False


class TransportationChatIntegration:
    """
    Integrates transfer instructions and map visualization into AI chat responses
    """
    
    def __init__(self):
        """Initialize the transportation chat integration"""
        if TRANSFER_MAP_AVAILABLE:
            self.transportation_service = LiveIBBTransportationService(use_mock_data=False)
            self.transfer_generator = TransferInstructionsGenerator()
            logger.info("🗺️ Transportation Chat Integration initialized")
        else:
            self.transportation_service = None
            self.transfer_generator = None
            logger.warning("⚠️ Transportation features not available")
    
    def is_available(self) -> bool:
        """Check if transportation features are available"""
        return TRANSFER_MAP_AVAILABLE and self.transportation_service is not None
    
    async def handle_transportation_query(
        self,
        query: str,
        user_location: Optional[str] = None,
        destination: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle transportation queries with detailed instructions and map data
        
        Args:
            query: User's question (e.g., "How do I get from Taksim to Kadıköy?")
            user_location: Starting location (if available)
            destination: Destination location (if available)
            user_context: Additional context (has_luggage, time_sensitive, etc.)
            
        Returns:
            Dict with response text, map data, and detailed instructions
        """
        
        if not self.is_available():
            return {
                'success': False,
                'error': 'Transportation features not available',
                'response_text': "I apologize, but the transportation system is currently unavailable. Please try again later."
            }
        
        try:
            # Extract origin and destination from query if not provided
            if not user_location or not destination:
                locations = self._extract_locations(query)
                user_location = user_location or locations.get('origin')
                destination = destination or locations.get('destination')
            
            if not user_location or not destination:
                return self._get_clarification_response(query)
            
            # Get enhanced recommendations with live data
            recommendations = await self.transportation_service.get_enhanced_recommendations(
                user_location,
                destination
            )
            
            if not recommendations or not recommendations.get('recommendations'):
                return {
                    'success': False,
                    'response_text': f"I couldn't find a route from {user_location} to {destination}. Please check the location names and try again."
                }
            
            # Get the best route
            best_route = recommendations.get('best_route')
            if not best_route:
                best_route = recommendations['recommendations'][0]
            
            # Determine which routes to take
            route_ids = self._determine_route_sequence(user_location, destination, best_route)
            
            # Generate detailed route with transfer instructions
            detailed_route = self.transportation_service.generate_detailed_route_with_transfers(
                origin=user_location,
                destination=destination,
                selected_routes=route_ids
            )
            
            # Format for chat display
            response_text = self._format_for_chat(detailed_route, recommendations)
            
            # Prepare map visualization data
            map_data = detailed_route.get('map_data')
            
            # Get data source information
            data_source_info = self.transportation_service.get_data_source_info()
            data_source_display = self.transportation_service.get_data_source_display()
            
            return {
                'success': True,
                'response_text': response_text,
                'map_data': map_data,
                'detailed_route': detailed_route,
                'alternatives': recommendations.get('recommendations', [])[:3],
                'fare_info': detailed_route.get('fare_info'),
                'transfer_count': len(detailed_route.get('transfers', [])),
                'total_time': detailed_route.get('total_time_estimate', 30),
                'has_map_visualization': map_data is not None,
                'data_source': data_source_info,
                'data_source_display': data_source_display
            }
            
        except Exception as e:
            logger.error(f"Error handling transportation query: {e}")
            return {
                'success': False,
                'error': str(e),
                'response_text': "I encountered an error processing your transportation request. Please try rephrasing your question."
            }
    
    def _extract_locations(self, query: str) -> Dict[str, Optional[str]]:
        """Extract origin and destination from query"""
        # Common patterns
        query_lower = query.lower()
        
        # First, try to extract known locations (most reliable)
        known_locations = [
            'Taksim', 'Kadıköy', 'Sultanahmet', 'Beşiktaş', 'Üsküdar',
            'Airport', 'Istanbul Airport', 'Sabiha Gökçen', 'Atatürk Airport',
            'Levent', 'Mecidiyeköy', 'Şişli', 'Beyoğlu', 'Kabataş',
            'Eminönü', 'Karaköy', 'Galata', 'Ortaköy', 'Sarıyer',
            'Galata Tower', 'Grand Bazaar', 'Spice Bazaar', 'Blue Mosque',
            'Hagia Sophia', 'Topkapi Palace', 'Dolmabahçe', 'Dolmabahce',
            'Ortaköy Mosque', 'Maiden Tower', 'Bosphorus', 'Golden Horn',
            'Bebek', 'Arnavutköy', 'Rumeli Fortress', 'Anadolu Fortress'
        ]
        
        found_locations = []
        for location in known_locations:
            if location.lower() in query_lower:
                found_locations.append(location)
        
        # Pattern: "from X to Y" - use regex only if we have found locations
        import re
        
        if len(found_locations) >= 2:
            # Try to determine order using patterns
            patterns = [
                (r'from\s+(.+?)\s+to\s+(.+?)(?:\s|$|[?.,!])', 'from_to'),
                (r'(.+?)\s+to\s+(.+?)(?:\s|$|[?.,!])', 'to_pattern')
            ]
            
            for pattern, pattern_type in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    origin_text = match.group(1).strip()
                    dest_text = match.group(2).strip()
                    
                    # Find which locations match
                    origin_loc = None
                    dest_loc = None
                    
                    for loc in found_locations:
                        if loc.lower() in origin_text:
                            origin_loc = loc
                        if loc.lower() in dest_text:
                            dest_loc = loc
                    
                    if origin_loc and dest_loc:
                        return {
                            'origin': origin_loc,
                            'destination': dest_loc
                        }
            
            # If pattern matching failed, use order of appearance
            return {
                'origin': found_locations[0],
                'destination': found_locations[1]
            }
        
        elif len(found_locations) == 1:
            # If only one location found, assume it's the destination
            return {
                'origin': None,
                'destination': found_locations[0]
            }
        
        return {'origin': None, 'destination': None}
    
    def _determine_route_sequence(
        self,
        origin: str,
        destination: str,
        best_route: Dict[str, Any]
    ) -> List[str]:
        """Determine the sequence of routes to take"""
        # Logic to determine which lines to take based on origin/destination
        origin_lower = origin.lower()
        dest_lower = destination.lower()
        
        route_sequences = {
            ('taksim', 'kadıköy'): ['M2', 'MARMARAY'],
            ('taksim', 'airport'): ['M2', 'M11'],
            ('taksim', 'istanbul airport'): ['M2', 'M11'],
            ('sultanahmet', 'airport'): ['T1', 'M2', 'M11'],
            ('sultanahmet', 'kadıköy'): ['T1', 'MARMARAY'],
            ('levent', 'mahmutbey'): ['M2', 'M7'],
            ('kabataş', 'taksim'): ['F1'],
            ('kadıköy', 'sabiha'): ['M4'],
            ('kadıköy', 'sabiha gökçen'): ['M4'],
        }
        
        # Try to find exact match
        for (orig, dest), routes in route_sequences.items():
            if orig in origin_lower and dest in dest_lower:
                return routes
        
        # Fallback to best route
        route_id = best_route.get('route', {}).route_id if hasattr(best_route.get('route', {}), 'route_id') else 'M2'
        return [route_id]
    
    def _format_for_chat(
        self,
        detailed_route: Dict[str, Any],
        recommendations: Dict[str, Any]
    ) -> str:
        """Format detailed route for chat display"""
        
        output = []
        
        # Header with data source indicator
        data_source = recommendations.get('data_source', 'unknown')
        if data_source == 'live_ibb_api':
            source_badge = '📡 Live İBB Data'
        elif data_source == 'live_ibb_mock':
            source_badge = '🧪 Mock Data'
        else:
            source_badge = '📊 Static Data'
        
        output.append(f"🗺️ **Route from {detailed_route['origin']} to {detailed_route['destination']}** ({source_badge})\n")
        
        # Quick summary
        route_names = ' → '.join(detailed_route['routes'])
        output.append(f"📍 **Take:** {route_names}")
        
        # Time and fare
        total_time = detailed_route.get('total_time_estimate', 'N/A')
        output.append(f"⏱️ **Time:** ~{total_time} minutes")
        
        fare_info = detailed_route.get('fare_info', {})
        if fare_info:
            total_cost = fare_info.get('breakdown', {}).get('total', 0)
            output.append(f"💳 **Fare:** {total_cost:.2f} TL (Istanbulkart)")
        
        output.append("")  # Blank line
        
        # Step-by-step instructions
        output.append("**📋 Step-by-Step Instructions:**\n")
        
        for i, step in enumerate(detailed_route.get('steps', []), 1):
            icon = step.get('mode_icon', '🚇')
            route_id = step.get('route_id', '')
            output.append(f"{icon} **Step {i}:** Take {route_id}")
            
            for detail in step.get('details', []):
                output.append(f"   • {detail}")
            output.append("")
        
        # Transfer instructions
        transfers = detailed_route.get('transfers', [])
        if transfers:
            output.append(f"**🔄 Transfers ({len(transfers)}):**\n")
            
            for transfer in transfers:
                instruction = transfer.get('instruction')
                if instruction:
                    output.append(f"**Transfer at {instruction.station_name}:**")
                    output.append(f"• From {instruction.from_line} to {instruction.to_line}")
                    output.append(f"• ⏱️ {instruction.estimated_time} min • 🚶 {instruction.walking_distance}m")
                    
                    # Add key steps
                    steps = instruction.detailed_steps[:3]  # First 3 steps
                    for step in steps:
                        output.append(f"• {step}")
                    output.append("")
        
        # Accessibility info
        accessibility = detailed_route.get('accessibility', [])
        if accessibility:
            output.append("**♿ Accessibility:**")
            for info in accessibility[:2]:  # First 2 items
                output.append(f"• {info}")
            output.append("")
        
        # Map visualization note
        if detailed_route.get('map_data'):
            output.append("🗺️ *Interactive map available - showing route on map*")
            output.append("")
        
        # Alternatives
        alternatives = recommendations.get('recommendations', [])[1:3]  # Get 2 alternatives
        if alternatives:
            output.append("**🔄 Alternative Options:**")
            for i, alt in enumerate(alternatives, 1):
                route = alt.get('route')
                if route:
                    output.append(f"{i}. {route.route_id} - {alt.get('reason', 'Alternative route')}")
        
        return '\n'.join(output)
    
    def _get_clarification_response(self, query: str) -> Dict[str, Any]:
        """Generate clarification request"""
        return {
            'success': False,
            'needs_clarification': True,
            'response_text': (
                "I'd be happy to help you with directions! 🗺️\n\n"
                "To give you the best route with detailed transfer instructions and map visualization, "
                "please tell me:\n"
                "• Where are you starting from?\n"
                "• Where do you want to go?\n\n"
                "For example: *'How do I get from Taksim to Kadıköy?'*"
            )
        }
    
    def format_transfer_instruction(self, instruction: TransferInstruction) -> str:
        """Format a single transfer instruction for chat"""
        if not self.transfer_generator:
            return ""
        
        return self.transfer_generator.format_transfer_instruction_for_display(instruction)
    
    def get_station_info(self, station_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific transfer station"""
        if not self.transfer_generator:
            return None
        
        station_data = self.transfer_generator.get_station_info(station_name)
        if not station_data:
            return None
        
        return {
            'name': station_data.station_name,
            'lines': station_data.transfer_available,
            'layout': station_data.platform_layout,
            'accessibility': station_data.accessibility,
            'amenities': station_data.amenities,
            'coordinates': station_data.coordinates
        }
    
    def get_all_transfer_stations(self) -> List[str]:
        """Get list of all transfer stations"""
        if not self.transfer_generator:
            return []
        
        return self.transfer_generator.get_all_transfer_stations()


# Factory function
def get_transportation_chat_integration() -> TransportationChatIntegration:
    """Get instance of transportation chat integration"""
    return TransportationChatIntegration()


# Quick test
async def test_integration():
    """Test the transportation chat integration"""
    print("=" * 80)
    print("  🗺️ TRANSPORTATION CHAT INTEGRATION TEST")
    print("=" * 80)
    
    integration = TransportationChatIntegration()
    
    if not integration.is_available():
        print("\n❌ Transportation features not available")
        return
    
    # Test queries
    test_queries = [
        "How do I get from Taksim to Kadıköy?",
        "Route from Sultanahmet to Istanbul Airport",
        "How to go from Levent to Mahmutbey?",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print('─'*80)
        
        result = await integration.handle_transportation_query(query)
        
        if result['success']:
            print(result['response_text'])
            print(f"\n✅ Map visualization: {result['has_map_visualization']}")
            print(f"✅ Transfers: {result['transfer_count']}")
            print(f"✅ Estimated time: {result['total_time']} minutes")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("✅ Integration test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_integration())
