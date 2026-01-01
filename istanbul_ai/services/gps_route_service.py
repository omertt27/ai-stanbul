# -*- coding: utf-8 -*-
"""
GPS Route Service
Handles GPS-based route planning and directions for the Istanbul AI system.
Extracts GPS routing logic from the main system for better modularity.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.models import UserProfile, ConversationContext
from ..utils.gps_utils import (
    calculate_distance,
    find_nearest_hub,
    estimate_walking_time,
    format_gps_coordinates,
    get_transport_recommendations
)

logger = logging.getLogger(__name__)


class GPSRouteService:
    """
    Service for generating GPS-aware route responses.
    Provides personalized directions based on user's current location.
    """
    
    def __init__(self, transport_processor=None):
        """
        Initialize GPS Route Service.
        
        Args:
            transport_processor: Transportation processor for getting detailed routes
        """
        self.transport_processor = transport_processor
        self.location_service = self._load_location_service()
        self.entity_extractor = self._load_entity_extractor()
        self.transportation_rag = self._load_transportation_rag()
        logger.info("✅ GPS Route Service initialized")
    
    def _load_location_service(self):
        """Load the location database service for nearby attractions/museums."""
        try:
            from .location_database_service import LocationDatabaseService
            service = LocationDatabaseService()
            logger.info("✅ Location Database Service loaded in GPS Route Service")
            return service
        except Exception as e:
            logger.warning(f"⚠️ Location Database Service not available: {e}")
            return None
    
    def _load_entity_extractor(self):
        """Load the comprehensive entity extractor for Istanbul locations."""
        try:
            from backend.services.entity_extractor import get_entity_extractor
            extractor = get_entity_extractor()
            logger.info("✅ Comprehensive Entity Extractor loaded in GPS Route Service")
            return extractor
        except Exception as e:
            logger.warning(f"⚠️ Entity Extractor not available: {e}")
            return None
    
    def generate_route_response(
        self,
        message: str,
        entities: Dict[str, Any],
        user_profile: UserProfile,
        context: ConversationContext
    ) -> str:
        """
        Generate a GPS-aware route response based on user location.
        Now integrates with Transportation RAG system for comprehensive routing.
        
        Args:
            message: User's query message
            entities: Extracted entities from the query
            user_profile: User's profile with GPS data
            context: Current conversation context
            
        Returns:
            Formatted route response with GPS-aware directions
        """
        try:
            logger.info("🗺️ Generating GPS-aware route response with RAG Transportation")
            
            # Extract GPS coordinates
            user_gps = self._extract_user_gps(user_profile, context)
            destination = self._extract_destination(entities, context, message)
            
            # Check if we have GPS data
            if not user_gps:
                return self._generate_no_gps_response(destination)
            
            # Check if we have a valid destination
            if not destination:
                return self._generate_no_destination_response(user_gps)
            
            # Try to use Transportation RAG system first
            rag_route = self._get_rag_route(user_gps, destination, message)
            if rag_route:
                logger.info("✅ Using Transportation RAG system for route")
                return rag_route
            
            # Fallback to traditional GPS route
            logger.info("⚠️ RAG route not available, using traditional GPS route")
            return self._generate_detailed_route(user_gps, destination, user_profile, context)
            
        except Exception as e:
            logger.error(f"Error generating GPS route response: {e}", exc_info=True)
            return self._generate_fallback_response(entities)
    
    def _extract_user_gps(
        self,
        user_profile: UserProfile,
        context: ConversationContext
    ) -> Optional[tuple]:
        """Extract user GPS coordinates from profile or context."""
        # Try context first (most recent)
        if hasattr(context, 'gps_location') and context.gps_location:
            lat = context.gps_location.get('latitude')
            lon = context.gps_location.get('longitude')
            if lat and lon:
                logger.info(f"📍 Using GPS from context: {format_gps_coordinates((lat, lon))}")
                return (lat, lon)
        
        # Try user profile
        if hasattr(user_profile, 'gps_location') and user_profile.gps_location:
            lat = user_profile.gps_location.get('latitude')
            lon = user_profile.gps_location.get('longitude')
            if lat and lon:
                logger.info(f"📍 Using GPS from profile: {format_gps_coordinates((lat, lon))}")
                return (lat, lon)
        
        logger.warning("⚠️ No GPS coordinates found in context or profile")
        return None
    
    def _extract_destination(
        self,
        entities: Dict[str, Any],
        context: ConversationContext,
        message: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract destination from entities, context, or using comprehensive entity extractor.
        This ensures we catch destinations even when patterns don't match.
        """
        destination = None
        
        # Try various entity keys
        # First check for direct destination/location
        destination = entities.get('destination') or entities.get('location')
        
        # If not found, check to_location (from entity extractor)
        if not destination:
            destination = entities.get('to_location')
            if destination:
                logger.info(f"🎯 Destination from to_location: {destination}")
        
        # If not found, check districts (common for Istanbul queries)
        if not destination:
            districts = entities.get('districts', [])
            if districts and len(districts) > 0:
                # Use the last district mentioned (usually the destination in "from X to Y" queries)
                destination = districts[-1] if len(districts) > 1 else districts[0]
                logger.info(f"🎯 Destination from districts: {destination}")
        
        # If not found, check attractions
        if not destination:
            attractions = entities.get('attractions', [])
            if attractions and len(attractions) > 0:
                destination = attractions[-1] if len(attractions) > 1 else attractions[0]
                logger.info(f"🎯 Destination from attractions: {destination}")
        
        # If not found, check locations list
        if not destination:
            locations = entities.get('locations', [])
            if locations and len(locations) > 0:
                destination = locations[-1] if len(locations) > 1 else locations[0]
                logger.info(f"🎯 Destination from locations: {destination}")
        
        # If still not found and we have the message, use comprehensive entity extractor
        if not destination and message and self.entity_extractor:
            try:
                logger.info("🔍 Using comprehensive entity extractor as fallback...")
                extracted = self.entity_extractor.extract_entities(message, intent='transportation')
                
                # Check to_location first
                if extracted.get('to_location'):
                    destination = extracted['to_location']
                    logger.info(f"✅ Entity extractor found to_location: {destination}")
                # Then check locations
                elif extracted.get('locations') and len(extracted['locations']) > 0:
                    destination = extracted['locations'][-1]
                    logger.info(f"✅ Entity extractor found location: {destination}")
            except Exception as e:
                logger.warning(f"⚠️ Entity extractor fallback failed: {e}")
        
        # Try context as last resort
        if not destination and hasattr(context, 'last_location'):
            destination = context.last_location
        
        if destination:
            logger.info(f"🎯 Final destination extracted: {destination}")
        else:
            logger.warning("⚠️ No destination found in entities, context, or entity extractor")
        
        return destination
    
    def _generate_detailed_route(
        self,
        user_gps: tuple,
        destination: str,
        user_profile: UserProfile,
        context: ConversationContext
    ) -> str:
        """Generate detailed route with GPS awareness."""
        lat, lon = user_gps
        
        # Find nearest transport hub
        nearest_hubs = find_nearest_hub((lat, lon), max_results=1)
        
        if not nearest_hubs:
            return self._generate_direct_route(user_gps, destination)
        
        nearest_hub = nearest_hubs[0]
        hub_name = nearest_hub['name']
        hub_type = nearest_hub.get('transport_types', ['transport'])[0]
        distance_km = nearest_hub['distance_km']
        
        # Build response
        response_parts = []
        
        # Header with current location
        response_parts.append(
            f"🗺️ **Your Route to {destination}**\n"
            f"📍 From your location: {format_gps_coordinates((lat, lon))}\n"
        )
        
        # Step 1: Get to nearest hub
        walking_time = estimate_walking_time(distance_km * 1000)  # Convert km to meters
        transport_recs = get_transport_recommendations((lat, lon))
        transport_mode = 'walking' if distance_km <= 1.5 else 'taxi'
        
        response_parts.append(
            f"\n**Step 1: Get to {hub_name}** ({hub_type})"
        )
        
        if transport_mode == 'walking':
            response_parts.append(
                f"🚶 Walking distance: {distance_km:.1f} km (~{walking_time} minutes)\n"
                f"💡 This is walkable! Head towards {hub_name}."
            )
        elif transport_mode == 'taxi':
            response_parts.append(
                f"🚕 Distance: {distance_km:.1f} km (~{int(distance_km * 5)} minutes by taxi)\n"
                f"💡 Consider taking a taxi or using public transport to {hub_name}."
            )
        else:  # public transport
            response_parts.append(
                f"🚌 Distance: {distance_km:.1f} km\n"
                f"💡 Use public transport or taxi to reach {hub_name}."
            )
        
        # Step 2: Route from hub to destination
        if self.transport_processor:
            try:
                # Use transport processor for hub-to-destination route
                logger.info(f"🚇 Getting route from {hub_name} to {destination}")
                
                # Create a GPS location object for the hub
                hub_location = {
                    'latitude': nearest_hub.get('lat'),
                    'longitude': nearest_hub.get('lon')
                }
                
                # Get route details (this would call the transport processor)
                route_details = self._get_transport_route(hub_location, destination, user_profile)
                
                if route_details:
                    response_parts.append(f"\n**Step 2: From {hub_name} to {destination}**")
                    response_parts.append(route_details)
                else:
                    response_parts.append(
                        f"\n**Step 2: From {hub_name} to {destination}**\n"
                        f"Use public transport connections from {hub_name}. "
                        f"Ask for specific directions once you reach {hub_name}."
                    )
            except Exception as e:
                logger.error(f"Error getting transport route: {e}")
                response_parts.append(
                    f"\n**Step 2: From {hub_name} to {destination}**\n"
                    f"Once at {hub_name}, you can easily connect to {destination}."
                )
        else:
            response_parts.append(
                f"\n**Step 2: From {hub_name} to {destination}**\n"
                f"Use metro, tram, or bus connections from {hub_name}."
            )
        
        # Add helpful tips
        response_parts.append(
            f"\n💡 **Travel Tips:**\n"
            f"• Get an Istanbulkart for seamless public transport\n"
            f"• Download Citymapper or Moovit for real-time updates\n"
            f"• Current time: {datetime.now().strftime('%H:%M')}\n"
            f"• Estimated total journey: {self._estimate_total_time(distance_km, destination)}"
        )
        
        return "\n".join(response_parts)
    
    def _get_transport_route(
        self,
        hub_location: Dict[str, float],
        destination: str,
        user_profile: UserProfile
    ) -> Optional[str]:
        """Get transport route from hub to destination."""
        if not self.transport_processor:
            return None
        
        try:
            # This would integrate with the transport processor
            # For now, return a placeholder
            return f"🚇 Public transport directions will be provided by the transport system."
        except Exception as e:
            logger.error(f"Error getting transport route: {e}")
            return None
    
    def _generate_direct_route(self, user_gps: tuple, destination: str) -> str:
        """Generate a direct route when no transport hub is nearby."""
        lat, lon = user_gps
        
        return (
            f"🗺️ **Route to {destination}**\n"
            f"📍 From your location: {format_gps_coordinates(lat, lon)}\n\n"
            f"I'll help you get to {destination}. Let me provide you with transportation options.\n\n"
            f"💡 **Getting There:**\n"
            f"• Use a taxi or ride-sharing service for direct travel\n"
            f"• Check Citymapper or Moovit for public transport options\n"
            f"• Consider walking if it's a nearby location\n\n"
            f"Would you like specific metro, tram, or bus directions to {destination}?"
        )
    
    def _generate_no_gps_response(self, destination: Optional[str]) -> str:
        """Generate response when GPS is not available."""
        if destination:
            return (
                f"🗺️ **Route to {destination}**\n\n"
                f"I'd love to give you personalized directions, but I don't have your current location.\n\n"
                f"💡 **Options:**\n"
                f"• Share your GPS location for personalized route planning\n"
                f"• Tell me your starting point (e.g., 'Taksim', 'Sultanahmet')\n"
                f"• Ask for general directions to {destination}\n\n"
                f"Would you like me to provide general transportation info to {destination}?"
            )
        else:
            return (
                "🗺️ **Route Planning**\n\n"
                "I can help you plan your route! To provide personalized directions, I need:\n\n"
                "1. Your current location (or enable GPS)\n"
                "2. Your destination\n\n"
                "💡 Example: 'How do I get to Sultanahmet from Taksim?'\n"
                "Or share your GPS location for automatic route planning!"
            )
    
    def _generate_no_destination_response(self, user_gps: tuple) -> str:
        """Generate response when destination is not specified."""
        lat, lon = user_gps
        
        # Find nearby attractions or hubs
        nearest_hubs = find_nearest_hub((lat, lon), max_results=1)
        nearest_hub = nearest_hubs[0] if nearest_hubs else None
        
        response = (
            f"📍 **Current Location**: {format_gps_coordinates((lat, lon))}\n\n"
            f"I can see you're in Istanbul! Where would you like to go?\n\n"
        )
        
        if nearest_hub:
            hub_name = nearest_hub['name']
            distance = nearest_hub['distance_km']
            response += (
                f"🚇 **Nearby Transport**: {hub_name} ({distance:.1f} km away)\n\n"
            )
        
        response += (
            f"💡 **Popular Destinations:**\n"
            f"• Sultanahmet (Blue Mosque, Hagia Sophia)\n"
            f"• Taksim Square & Istiklal Street\n"
            f"• Grand Bazaar\n"
            f"• Galata Tower\n"
            f"• Bosphorus Ferry Tour\n\n"
            f"Tell me where you'd like to go, and I'll plan the best route for you!"
        )
        
        return response
    
    def _generate_fallback_response(self, entities: Dict[str, Any]) -> str:
        """Generate fallback response when route planning fails."""
        destination = entities.get('destination') or entities.get('location')
        
        if destination:
            return (
                f"🗺️ I can help you get to {destination}!\n\n"
                f"For the best route planning, please:\n"
                f"1. Share your GPS location, or\n"
                f"2. Tell me your starting point\n\n"
                f"I'll provide detailed directions with transportation options!"
            )
        else:
            return (
                "🗺️ I'm here to help you navigate Istanbul!\n\n"
                "Tell me:\n"
                "• Where you want to go\n"
                "• Your starting point (or share GPS)\n\n"
                "I'll provide the best route with public transport options!"
            )
    
    def _estimate_total_time(self, distance_to_hub_km: float, destination: str) -> str:
        """Estimate total travel time."""
        # Walking/taxi to hub
        time_to_hub = estimate_walking_time(distance_to_hub_km)
        if distance_to_hub_km > 2.0:
            time_to_hub = int(distance_to_hub_km * 5)  # Assume taxi
        
        # Rough estimate for hub to destination (30-45 min average)
        transit_time = 35
        
        total_time = time_to_hub + transit_time
        
        return f"{total_time}-{total_time + 15} minutes"
    
    def get_nearby_locations(
        self,
        user_gps: tuple,
        radius_km: float = 2.0,
        categories: Optional[List[str]] = None,
        max_results: int = 10
    ) -> str:
        """
        Get nearby attractions and museums with directions.
        
        Args:
            user_gps: User's GPS coordinates (lat, lon)
            radius_km: Search radius in kilometers
            categories: Filter by categories (e.g., ['culture', 'food'])
            max_results: Maximum number of results to return
            
        Returns:
            Formatted response with nearby locations and how to get there
        """
        if not user_gps or len(user_gps) != 2:
            return "❌ I need your GPS location to find nearby places. Please share your location."
        
        if not self.location_service:
            return "❌ Location search service is temporarily unavailable. Please try again later."
        
        try:
            lat, lon = user_gps
            
            # Get nearby locations with transport recommendations
            nearby = self.location_service.get_nearby_locations(
                lat, lon,
                radius_km=radius_km,
                categories=categories,
                max_results=max_results,
                include_transport=True
            )
            
            if not nearby:
                return self._format_no_results_response(user_gps, radius_km, categories)
            
            # Format the response
            return self._format_nearby_locations_response(nearby, user_gps, radius_km)
            
        except Exception as e:
            logger.error(f"Error getting nearby locations: {e}")
            return "❌ Sorry, I encountered an error searching for nearby places. Please try again."
    
    def _format_nearby_locations_response(
        self,
        locations: List[Dict],
        user_gps: tuple,
        radius_km: float
    ) -> str:
        """Format nearby locations into a user-friendly response."""
        lat, lon = user_gps
        
        response_parts = [
            f"📍 **Your Location**: {format_gps_coordinates(user_gps)}",
            f"🔍 **Found {len(locations)} places within {radius_km} km:**\n"
        ]
        
        for i, location in enumerate(locations, 1):
            # Basic info
            name = location['name']
            loc_type = location['type'].title()
            distance = location['distance_km']
            walk_time = location['walking_time_min']
            district = location.get('district', 'Istanbul')
            
            response_parts.append(f"\n**{i}. {name}** ({loc_type})")
            response_parts.append(f"   📏 {distance} km away • 🚶 ~{walk_time} min walk")
            response_parts.append(f"   📍 {district}")
            
            # Add rating if available (for attractions)
            if location.get('rating'):
                response_parts.append(f"   ⭐ {location['rating']}/5.0")
            
            # Add description snippet
            if location.get('description'):
                desc = location['description']
                desc_short = desc[:100] + '...' if len(desc) > 100 else desc
                response_parts.append(f"   ℹ️  {desc_short}")
            
            # Add transport recommendations
            if 'transport_route' in location:
                route = location['transport_route']
                options = route.get('options', [])
                
                if options:
                    # Find recommended option
                    recommended = next((opt for opt in options if opt.get('recommended')), options[0])
                    mode = recommended['mode'].replace('_', ' ').title()
                    duration = recommended['duration_min']
                    cost = recommended['cost_tl']
                    
                    response_parts.append(f"   🚇 **Best way**: {mode} (~{duration} min, {cost} TL)")
                    
                    # Show alternative if available
                    if len(options) > 1:
                        alt = options[1] if options[0] == recommended else options[0]
                        alt_mode = alt['mode'].replace('_', ' ').title()
                        alt_duration = alt['duration_min']
                        alt_cost = alt['cost_tl']
                        response_parts.append(f"   🔄 Alternative: {alt_mode} (~{alt_duration} min, {alt_cost} TL)")
            
            # Add opening hours if available
            if location.get('opening_hours'):
                hours = location['opening_hours']
                if isinstance(hours, dict):
                    # Show today's hours or general hours
                    today_key = datetime.now().strftime('%A').lower()
                    if today_key in hours:
                        response_parts.append(f"   🕐 Open today: {hours[today_key]}")
                    elif 'daily' in hours:
                        response_parts.append(f"   🕐 {hours['daily']}")
                else:
                    response_parts.append(f"   🕐 {hours}")
        
        # Add helpful footer
        response_parts.append("\n\n💡 **Tip**: Say 'How do I get to [place name]' for detailed directions!")
        
        return "\n".join(response_parts)
    
    def _format_no_results_response(
        self,
        user_gps: tuple,
        radius_km: float,
        categories: Optional[List[str]]
    ) -> str:
        """Format response when no locations are found."""
        response = f"📍 **Your Location**: {format_gps_coordinates(user_gps)}\n\n"
        response += f"🔍 No attractions or museums found within {radius_km} km"
        
        if categories:
            response += f" matching categories: {', '.join(categories)}"
        
        response += ".\n\n💡 **Try**:\n"
        response += f"- Expanding search radius (try {radius_km + 1} km)\n"
        response += "- Removing category filters\n"
        response += "- Asking 'What are the top attractions in Istanbul?'"
        
        return response
    
    def get_route_to_location(
        self,
        user_gps: tuple,
        destination_name: str,
        destination_gps: Optional[tuple] = None
    ) -> str:
        """
        Get detailed route from user's GPS to a specific location.
        
        Args:
            user_gps: User's current GPS coordinates
            destination_name: Name of the destination
            destination_gps: Destination GPS coordinates (will look up if not provided)
            
        Returns:
            Formatted route with step-by-step directions
        """
        if not user_gps or len(user_gps) != 2:
            return "❌ I need your GPS location to provide directions. Please share your location."
        
        try:
            # If destination GPS not provided, try to find it
            if not destination_gps and self.location_service:
                destination_gps = self._lookup_destination_gps(destination_name)
            
            if not destination_gps:
                return f"❌ I couldn't find GPS coordinates for '{destination_name}'. Please provide more details."
            
            # Get transport route
            if self.location_service:
                route = self.location_service.get_transport_route(
                    user_gps,
                    destination_gps,
                    destination_name
                )
                
                return self._format_route_response(route, user_gps, destination_name)
            else:
                # Fallback to basic route
                return self._generate_route(user_gps, destination_name)
                
        except Exception as e:
            logger.error(f"Error generating route: {e}")
            return f"❌ Sorry, I encountered an error generating directions to {destination_name}."
    
    def _lookup_destination_gps(self, destination_name: str) -> Optional[tuple]:
        """Look up GPS coordinates for a destination by name."""
        try:
            # Search in attractions
            for attraction in self.location_service.attractions:
                if destination_name.lower() in attraction['name'].lower():
                    return tuple(attraction['gps'])
            
            # Search in museums
            for museum_id, gps_info in self.location_service.museum_gps.items():
                museum = self.location_service.museum_db.museums.get(museum_id)
                if museum and destination_name.lower() in museum.name.lower():
                    return tuple(gps_info['gps'])
            
            return None
        except Exception as e:
            logger.error(f"Error looking up destination GPS: {e}")
            return None
    
    def _format_route_response(
        self,
        route: Dict[str, Any],
        user_gps: tuple,
        destination_name: str
    ) -> str:
        """Format transport route into user-friendly directions."""
        response_parts = [
            f"🗺️  **Route to {destination_name}**",
            f"📍 From: {format_gps_coordinates(user_gps)}",
            f"📏 Distance: {route.get('distance_km', 'N/A')} km\n"
        ]
        
        options = route.get('options', [])
        if not options:
            return f"❌ No route options available for {destination_name}."
        
        # Show all transport options
        response_parts.append("🚇 **Transport Options:**\n")
        
        for i, option in enumerate(options, 1):
            mode = option['mode'].replace('_', ' ').title()
            duration = option['duration_min']
            cost = option['cost_tl']
            desc = option.get('description', '')
            recommended = '⭐ ' if option.get('recommended') else '   '
            
            response_parts.append(f"{recommended}**Option {i}: {mode}**")
            response_parts.append(f"   ⏱️  Duration: ~{duration} minutes")
            response_parts.append(f"   💰 Cost: {cost} TL")
            if desc:
                response_parts.append(f"   📝 {desc}")
            response_parts.append("")
        
        # Add helpful tips
        response_parts.append("\n💡 **Tips:**")
        response_parts.append("- Istanbul public transport accepts İstanbulkart")
        response_parts.append("- Taxis use meters (starts at ~45 TL)")
        response_parts.append("- Walking is great for distances under 1-2 km")
        
        return "\n".join(response_parts)
