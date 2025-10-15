"""
GPS Location Processing
======================

Handle GPS coordinates, location detection, and integration with transportation services.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime

from ..models.transportation_models import GPSLocation

# Import intelligent location detector and related classes  
if TYPE_CHECKING:
    from istanbul_ai.services.intelligent_location_detector import (
        IntelligentLocationDetector, 
        GPSContext,
        LocationDetectionResult
    )
    from istanbul_ai.core.user_profile import UserProfile
    from istanbul_ai.core.conversation_context import ConversationContext

# Runtime imports with fallback
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from istanbul_ai.services.intelligent_location_detector import (
        IntelligentLocationDetector, 
        GPSContext,
        LocationDetectionResult
    )
    from istanbul_ai.core.user_profile import UserProfile
    from istanbul_ai.core.conversation_context import ConversationContext
    LOCATION_DETECTOR_AVAILABLE = True
except ImportError:
    LOCATION_DETECTOR_AVAILABLE = False


class GPSLocationProcessor:
    """Process GPS locations and integrate with intelligent location detector"""
    
    def __init__(self):
        self.istanbul_bounds = {
            'lat_min': 40.8,
            'lat_max': 41.3,
            'lng_min': 28.6,
            'lng_max': 29.3
        }
        
        # Major transportation hubs with GPS coordinates
        self.transport_hubs = {
            'taksim': GPSLocation(41.0363, 28.9851, 'Taksim Square', 'Beyoğlu', 'Taksim Metro/Bus Hub'),
            'sultanahmet': GPSLocation(41.0086, 28.9802, 'Sultanahmet Square', 'Fatih', 'Historic Peninsula'),
            'kadikoy': GPSLocation(40.9969, 29.0264, 'Kadıköy Center', 'Kadıköy', 'Ferry Terminal'),
            'eminonu': GPSLocation(41.0176, 28.9706, 'Eminönü', 'Fatih', 'Ferry Terminal'),
            'galata_tower': GPSLocation(41.0256, 28.9744, 'Galata Tower', 'Beyoğlu', 'Historic Tower'),
            'ist_airport': GPSLocation(41.2753, 28.7519, 'Istanbul Airport', 'Arnavutköy', 'IST Airport'),
            'sabiha_gokcen': GPSLocation(40.8986, 29.3092, 'Sabiha Gökçen Airport', 'Pendik', 'SAW Airport'),
            'besiktas': GPSLocation(41.0422, 29.0094, 'Beşiktaş', 'Beşiktaş', 'Ferry Terminal'),
            'ortakoy': GPSLocation(41.0553, 29.0265, 'Ortaköy', 'Beşiktaş', 'Bosphorus Waterfront'),
            'karakoy': GPSLocation(41.0201, 28.9744, 'Karaköy', 'Beyoğlu', 'Ferry Terminal'),
            'uskudar': GPSLocation(41.0214, 29.0106, 'Üsküdar', 'Üsküdar', 'Ferry Terminal'),
            'levent': GPSLocation(41.0815, 28.9978, 'Levent', 'Beşiktaş', 'Business District'),
            'maslak': GPSLocation(41.1086, 29.0247, 'Maslak', 'Sarıyer', 'Business District')
        }
        
        self.logger = logging.getLogger(__name__)
    
    def is_in_istanbul(self, location: GPSLocation) -> bool:
        """Check if GPS coordinates are within Istanbul bounds"""
        return (self.istanbul_bounds['lat_min'] <= location.latitude <= self.istanbul_bounds['lat_max'] and
                self.istanbul_bounds['lng_min'] <= location.longitude <= self.istanbul_bounds['lng_max'])
    
    def find_nearest_transport_hub(self, location: GPSLocation) -> Tuple[str, GPSLocation, float]:
        """Find the nearest major transport hub to given GPS location"""
        if not self.is_in_istanbul(location):
            return None, None, float('inf')
        
        nearest_hub = None
        nearest_location = None
        min_distance = float('inf')
        
        for hub_name, hub_location in self.transport_hubs.items():
            distance = self._calculate_distance(location, hub_location)
            if distance < min_distance:
                min_distance = distance
                nearest_hub = hub_name
                nearest_location = hub_location
        
        return nearest_hub, nearest_location, min_distance
    
    def _calculate_distance(self, loc1: GPSLocation, loc2: GPSLocation) -> float:
        """Calculate distance between two GPS locations in meters"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [loc1.latitude, loc1.longitude, 
                                              loc2.latitude, loc2.longitude])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in meters
        r = 6371000
        return c * r
    
    def get_location_context(self, location: GPSLocation) -> Dict[str, Any]:
        """Get context information about a GPS location"""
        nearest_hub, hub_location, distance = self.find_nearest_transport_hub(location)
        
        context = {
            'coordinates': {
                'latitude': location.latitude,
                'longitude': location.longitude
            },
            'in_istanbul': self.is_in_istanbul(location),
            'nearest_transport_hub': {
                'name': nearest_hub,
                'distance_meters': round(distance) if distance != float('inf') else None,
                'walking_time_minutes': max(1, round(distance / 80)) if distance != float('inf') else None,
                'location': hub_location
            }
        }
        
        # Add district information if available
        if location.district:
            context['district'] = location.district
        if location.address:
            context['address'] = location.address
        if location.landmark:
            context['landmark'] = location.landmark
            
        return context
    
    def suggest_transport_options_from_gps(self, location: GPSLocation) -> Dict[str, Any]:
        """Suggest transportation options from a GPS location"""
        context = self.get_location_context(location)
        
        if not context['in_istanbul']:
            return {
                'error': 'Location is outside Istanbul',
                'suggestion': 'Please provide a location within Istanbul city limits'
            }
        
        nearest_hub = context['nearest_transport_hub']
        suggestions = {
            'current_location': {
                'coordinates': context['coordinates'],
                'description': location.address or f"Location near {nearest_hub['name']}"
            },
            'nearest_transport_hub': nearest_hub,
            'transport_options': []
        }
        
        # Add walking option to nearest hub
        if nearest_hub['distance_meters'] <= 2000:  # Within 2km
            suggestions['transport_options'].append({
                'type': 'walking',
                'description': f"Walk to {nearest_hub['name']}",
                'duration_minutes': nearest_hub['walking_time_minutes'],
                'distance_meters': nearest_hub['distance_meters'],
                'cost': 'Free',
                'recommendation': 'Good option for short distances'
            })
        
        # Add taxi/rideshare option
        suggestions['transport_options'].append({
            'type': 'taxi',
            'description': f"Taxi to {nearest_hub['name']} or any destination",
            'duration_minutes': max(5, nearest_hub['distance_meters'] // 200),  # ~200m/min in traffic
            'apps': ['BiTaksi', 'Uber', 'Taxi'],
            'cost_estimate': f"{max(20, nearest_hub['distance_meters'] * 0.01):.0f}-{max(30, nearest_hub['distance_meters'] * 0.015):.0f} TL",
            'recommendation': 'Best for direct routes or heavy luggage'
        })
        
        return suggestions


class GPSTransportationQueryProcessor:
    """Enhanced transportation processor with GPS location support and intelligent location detector integration"""
    
    def __init__(self):
        from ..services.transportation_service import EnhancedTransportationSystem
        from ..processors.comprehensive_processor import ComprehensiveTransportProcessor
        
        self.enhanced_system = EnhancedTransportationSystem()
        self.comprehensive_processor = ComprehensiveTransportProcessor()
        self.gps_processor = GPSLocationProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Initialize intelligent location detector if available
        if LOCATION_DETECTOR_AVAILABLE:
            try:
                self.location_detector = IntelligentLocationDetector()
                self.has_location_detector = True
                self.logger.info("📍 Integrated with IntelligentLocationDetector for GPS enhancement")
            except Exception as e:
                self.location_detector = None
                self.has_location_detector = False
                self.logger.warning(f"⚠️ Failed to initialize IntelligentLocationDetector: {e}")
        else:
            self.location_detector = None
            self.has_location_detector = False
            self.logger.warning("⚠️ IntelligentLocationDetector not available")
    
    async def process_gps_transportation_query(
        self, 
        user_input: str, 
        user_gps: Optional[GPSLocation] = None,
        destination: Optional[str] = None,
        entities: Dict[str, Any] = None,
        user_profile: Any = None
    ) -> str:
        """Process transportation query with GPS location support and intelligent detection"""
        
        if entities is None:
            entities = {}
        
        try:
            # Handle GPS location input with intelligent location detector enhancement
            if user_gps:
                self.logger.info(f"Processing GPS transportation query: {user_gps.latitude}, {user_gps.longitude}")
                
                # First, check if GPS location is within Istanbul bounds
                if not self.gps_processor.is_in_istanbul(user_gps):
                    return """📍 **Location Outside Istanbul**
                    
Your current location appears to be outside Istanbul city limits. 

🚗 **Getting to Istanbul:**
• **From Airports**: Use airport shuttles or metro connections
• **From Other Cities**: Check intercity bus or train services
• **From Nearby Areas**: Consider taxi or ride-sharing services

Once you're in Istanbul, I can provide detailed public transport directions!"""
                
                # Get location context using GPS processor
                location_context = self.gps_processor.get_location_context(user_gps)
                
                # Enhanced response with GPS context
                response = await self._generate_gps_enhanced_response(
                    user_input, user_gps, location_context, destination, entities, user_profile
                )
                
                return response
            
            else:
                # Fallback to comprehensive processor for text-based queries
                return await self.comprehensive_processor.process_transportation_query(
                    user_input=user_input,
                    entities=entities,
                    user_profile=user_profile
                )
                
        except Exception as e:
            self.logger.error(f"Error in GPS transportation query processing: {e}")
            return f"⚠️ Error processing transportation query: {str(e)}"
    
    async def _generate_gps_enhanced_response(
        self, 
        user_input: str, 
        user_gps: GPSLocation,
        location_context: Dict[str, Any],
        destination: Optional[str] = None,
        entities: Dict[str, Any] = None,
        user_profile: Any = None
    ) -> str:
        """Generate enhanced response using GPS location context"""
        
        nearest_hub = location_context['nearest_transport_hub']
        transport_suggestions = self.gps_processor.suggest_transport_options_from_gps(user_gps)
        
        # Build response based on query type
        if destination:
            # Route planning query
            return await self._generate_gps_route_response(
                user_gps, destination, location_context, transport_suggestions
            )
        else:
            # General transportation info from GPS location
            return self._generate_gps_general_response(
                user_input, location_context, transport_suggestions
            )
    
    async def _generate_gps_route_response(
        self,
        user_gps: GPSLocation,
        destination: str,
        location_context: Dict[str, Any],
        transport_suggestions: Dict[str, Any]
    ) -> str:
        """Generate route planning response from GPS location to destination"""
        
        nearest_hub = location_context['nearest_transport_hub']
        
        # Use comprehensive processor to get route from nearest hub to destination
        route_query = f"How to get from {nearest_hub['name']} to {destination}"
        route_response = await self.comprehensive_processor.process_transportation_query(
            user_input=route_query,
            entities={'origin': nearest_hub['name'], 'destination': destination}
        )
        
        # Combine GPS context with route information
        response = f"""📍 **Route from Your Location to {destination}**

🗺️ **Your Current Location:**
• Coordinates: {user_gps.latitude:.4f}, {user_gps.longitude:.4f}
• Nearest Transport Hub: {nearest_hub['name']} ({nearest_hub['distance_meters']}m away)
• Walking Time to Hub: ~{nearest_hub['walking_time_minutes']} minutes

🚶‍♂️ **Step 1: Get to {nearest_hub['name']}**
"""
        
        # Add transport options to nearest hub
        for option in transport_suggestions['transport_options']:
            if option['type'] == 'walking' and nearest_hub['distance_meters'] <= 1000:
                response += f"• **Walk ({option['duration_minutes']} min)**: {option['description']} - {option['recommendation']}\n"
            elif option['type'] == 'taxi':
                response += f"• **Taxi (~{option['duration_minutes']} min)**: {option['cost_estimate']} - {option['recommendation']}\n"
        
        response += f"\n🚇 **Step 2: Continue to {destination}**\n{route_response}"
        
        return response
    
    def _generate_gps_general_response(
        self,
        user_input: str,
        location_context: Dict[str, Any],
        transport_suggestions: Dict[str, Any]
    ) -> str:
        """Generate general transportation information response"""
        
        nearest_hub = location_context['nearest_transport_hub']
        
        response = f"""📍 **Transportation Options from Your Location**

🗺️ **Current Location Analysis:**
• Nearest Transport Hub: **{nearest_hub['name']}**
• Distance: {nearest_hub['distance_meters']}m ({nearest_hub['walking_time_minutes']} min walk)

🚇 **Available Transport Options:**
"""
        
        for option in transport_suggestions['transport_options']:
            response += f"• **{option['type'].title()}**: {option['description']}\n"
            if 'duration_minutes' in option:
                response += f"  - Duration: ~{option['duration_minutes']} minutes\n"
            if 'cost' in option:
                response += f"  - Cost: {option.get('cost', option.get('cost_estimate', 'Variable'))}\n"
            if 'recommendation' in option:
                response += f"  - Note: {option['recommendation']}\n"
            response += "\n"
        
        response += f"""💡 **Next Steps:**
• Walk to {nearest_hub['name']} to access metro/tram connections
• Use taxi/rideshare for direct routes
• Ask me for specific destinations: "How to get to [destination]?"

🗺️ **Need directions to a specific place?** Just tell me where you want to go!"""
        
        return response
