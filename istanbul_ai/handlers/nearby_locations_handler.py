"""
Nearby Locations Handler - GPS-based location discovery

This handler processes "What's near me?" and "What can I visit nearby?" queries
by leveraging GPS data to provide personalized, distance-aware recommendations
with integrated transport information.

Features:
- GPS-aware location search (museums, attractions, restaurants)
- Distance-based filtering and sorting
- Transport recommendations for each location
- Personalized suggestions based on user preferences
- Integration with accurate museum database and map visualization engine

Author: Istanbul AI Team
Date: December 2024
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Import accurate museum database
try:
    from backend.accurate_museum_database import IstanbulMuseumDatabase
    MUSEUM_DB_AVAILABLE = True
except ImportError:
    MUSEUM_DB_AVAILABLE = False
    logger.warning("⚠️ Accurate museum database not available")

# Import map visualization engine
try:
    from backend.services.map_visualization_engine import (
        MapVisualizationEngine, MapLocation, RouteSegment
    )
    MAP_ENGINE_AVAILABLE = True
except ImportError:
    MAP_ENGINE_AVAILABLE = False
    logger.warning("⚠️ Map visualization engine not available")


class NearbyLocationsHandler:
    """
    Handles nearby location queries with GPS-based recommendations.
    Integrates with accurate museum database and map visualization engine.
    """
    
    def __init__(self, gps_route_service=None, location_database_service=None,
                 neural_processor=None, user_manager=None, transport_service=None,
                 llm_service=None, gps_location_service=None):
        """
        Initialize the Nearby Locations Handler.
        
        Args:
            gps_route_service: GPS route planning service
            location_database_service: Unified location database service
            neural_processor: Optional ML model for semantic understanding
            user_manager: Optional user profile manager
            transport_service: Optional transport integration
            llm_service: Optional LLM service for GPS-aware POI recommendations
            gps_location_service: GPS location service for district detection
        """
        self.gps_route_service = gps_route_service
        self.location_database_service = location_database_service
        self.neural_processor = neural_processor
        self.user_manager = user_manager
        self.transport_service = transport_service
        
        # LLM + GPS integration
        self.llm_service = llm_service
        self.gps_location_service = gps_location_service
        
        # Initialize accurate museum database
        if MUSEUM_DB_AVAILABLE:
            try:
                self.museum_db = IstanbulMuseumDatabase()
                logger.info("✅ Accurate museum database integrated")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load museum database: {e}")
                self.museum_db = None
        else:
            self.museum_db = None
        
        # Initialize map visualization engine
        if MAP_ENGINE_AVAILABLE:
            try:
                self.map_engine = MapVisualizationEngine(use_osrm=True)
                logger.info("✅ Map visualization engine integrated")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load map engine: {e}")
                self.map_engine = None
        else:
            self.map_engine = None
        
        # Feature flags
        self.has_llm = llm_service is not None
        self.has_gps_location = gps_location_service is not None
        
        logger.info(
            f"✅ NearbyLocationsHandler initialized "
            f"(LLM: {self.has_llm}, GPSLocation: {self.has_gps_location})"
        )
    
    def can_handle(self, intent: str, entities: Dict, context: Any) -> bool:
        """
        Determine if this handler can process the query.
        
        Args:
            intent: Classified intent (should be 'nearby_locations')
            entities: Extracted entities
            context: Conversation context
            
        Returns:
            True if this is a nearby locations query
        """
        return intent == 'nearby_locations'
    
    def handle(
        self,
        message: str,
        entities: Dict,
        user_profile: Any,
        context: Any,
        neural_insights: Optional[Dict] = None,
        return_structured: bool = False
    ) -> Any:
        """
        Handle nearby locations query and generate response.
        
        Args:
            message: User's query message
            entities: Extracted entities (including GPS if available)
            user_profile: User profile with preferences
            context: Conversation context
            neural_insights: Optional ML-generated insights
            return_structured: Whether to return structured response
            
        Returns:
            Formatted response string or structured dict
        """
        try:
            logger.info(f"📍 Processing nearby locations query: {message[:50]}...")
            
            # Extract GPS coordinates from entities or context
            user_gps = self._extract_gps_coordinates(entities, context, user_profile)
            
            if not user_gps:
                return self._generate_no_gps_response(return_structured)
            
            # Extract search parameters
            radius_km = self._extract_radius(message, entities)
            location_types = self._extract_location_types(message, entities, neural_insights)
            max_results = self._extract_max_results(message, entities)
            
            # Get nearby locations from unified database
            nearby_results = self._get_nearby_locations(
                user_gps,
                radius_km,
                location_types,
                max_results
            )
            
            if not nearby_results:
                return self._generate_no_results_response(
                    user_gps, radius_km, return_structured
                )
            
            # Enhance with transport recommendations
            enhanced_results = self._enhance_with_transport(nearby_results, user_gps)
            
            # Apply personalization if available
            if self.user_manager and user_profile:
                enhanced_results = self._apply_personalization(
                    enhanced_results, user_profile
                )
            
            # Generate response
            response = self._generate_response(
                enhanced_results,
                user_gps,
                radius_km,
                location_types,
                return_structured
            )
            
            # Track query for analytics
            self._track_query(message, user_profile, enhanced_results)
            
            logger.info(f"✅ Nearby locations response generated: {len(enhanced_results)} recommendations")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in nearby locations handler: {e}", exc_info=True)
            return self._generate_error_response(str(e), return_structured)
    
    # ==================== INTERNAL METHODS ====================
    
    def _extract_gps_coordinates(
        self, entities: Dict, context: Any, user_profile: Any
    ) -> Optional[tuple]:
        """Extract GPS coordinates from various sources."""
        # Try entities first
        if entities.get('gps_coordinates'):
            return entities['gps_coordinates']
        
        # Try context
        if context and hasattr(context, 'gps_coordinates'):
            gps = context.gps_coordinates
            if gps:
                return gps
        
        # Try user profile
        if user_profile and hasattr(user_profile, 'current_location'):
            location = user_profile.current_location
            if location:
                return location
        
        logger.warning("⚠️ No GPS coordinates found in query context")
        return None
    
    def _extract_radius(self, message: str, entities: Dict) -> float:
        """Extract search radius from query (default 2km)."""
        # Check entities first
        if 'radius' in entities:
            return float(entities['radius'])
        
        # Parse from message
        import re
        message_lower = message.lower()
        
        # Look for distance patterns
        patterns = [
            r'within (\d+\.?\d*)\s*(km|kilometer|kilometre)',
            r'(\d+\.?\d*)\s*(km|kilometer|kilometre) radius',
            r'around (\d+\.?\d*)\s*(km|kilometer|kilometre)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return float(match.group(1))
        
        # Check for distance indicators
        if any(word in message_lower for word in ['walking', 'walk', 'nearby', 'close']):
            return 1.0  # 1km for walking distance
        elif any(word in message_lower for word in ['far', 'distant', 'area']):
            return 5.0  # 5km for wider area
        
        # Default radius
        return 2.0
    
    def _extract_location_types(
        self, message: str, entities: Dict, neural_insights: Optional[Dict]
    ) -> List[str]:
        """Extract desired location types from query."""
        types = []
        message_lower = message.lower()
        
        # Check for specific type mentions
        type_keywords = {
            'museum': ['museum', 'museums', 'müze'],
            'attraction': ['attraction', 'site', 'landmark', 'place to visit', 'tourist'],
            'restaurant': ['restaurant', 'cafe', 'eat', 'food', 'dining'],
            'shopping': ['shop', 'shopping', 'market', 'bazaar'],
            'park': ['park', 'garden', 'green space']
        }
        
        for loc_type, keywords in type_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                types.append(loc_type)
        
        # Use neural insights if available
        if neural_insights and 'detected_categories' in neural_insights:
            neural_types = neural_insights['detected_categories']
            types.extend([t for t in neural_types if t not in types])
        
        # Default: return all types if none specified
        if not types:
            types = ['museum', 'attraction']
        
        return types
    
    def _extract_max_results(self, message: str, entities: Dict) -> int:
        """Extract maximum number of results to return."""
        if 'max_results' in entities:
            return int(entities['max_results'])
        
        # Parse from message
        import re
        patterns = [
            r'top (\d+)',
            r'(\d+) places',
            r'(\d+) locations',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return int(match.group(1))
        
        # Default
        return 5
    
    def _get_nearby_locations(
        self,
        user_gps: tuple,
        radius_km: float,
        location_types: List[str],
        max_results: int
    ) -> List[Dict]:
        """Get nearby locations using multiple sources (museum DB, location service, etc.)."""
        all_results = []
        
        try:
            # 1. Get museums from accurate museum database
            if self.museum_db and ('museum' in location_types or 'attraction' in location_types):
                museum_results = self._get_nearby_museums_from_db(user_gps, radius_km)
                all_results.extend(museum_results)
                logger.info(f"📚 Found {len(museum_results)} museums from accurate database")
            
            # 2. Get from unified location database service
            if self.location_database_service:
                service_results = self.location_database_service.get_nearby_locations(
                    user_gps=user_gps,
                    radius_km=radius_km,
                    location_types=location_types,
                    max_results=max_results
                )
                # Merge results (avoid duplicates by name)
                existing_names = {r['name'] for r in all_results}
                for result in service_results:
                    if result['name'] not in existing_names:
                        all_results.append(result)
                logger.info(f"📍 Found {len(service_results)} locations from location service")
            
            # 3. Fallback to GPS route service
            elif self.gps_route_service and not all_results:
                results = self.gps_route_service.get_nearby_locations(
                    user_gps=user_gps,
                    radius_km=radius_km,
                    max_results=max_results
                )
                all_results.extend(results)
            
            # Sort by distance and limit results
            all_results.sort(key=lambda x: x.get('distance_km', 999))
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Error getting nearby locations: {e}")
            return []
    
    def _get_nearby_museums_from_db(
        self, user_gps: tuple, radius_km: float
    ) -> List[Dict]:
        """Get nearby museums from accurate museum database with GPS calculation."""
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lat1, lon1, lat2, lon2):
            """Calculate distance between two GPS coordinates in km."""
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return 6371 * c  # Earth radius in km
        
        # Museum GPS coordinates (from accurate_museum_database.py)
        MUSEUM_GPS = {
            "hagia_sophia": (41.0086, 28.9802),
            "topkapi_palace": (41.0115, 28.9833),
            "istanbul_archaeology_museums": (41.0117, 28.9806),
            "basilica_cistern": (41.0084, 28.9777),
            "turkish_and_islamic_arts_museum": (41.0055, 28.9765),
            "istanbul_modern": (40.9967, 28.9746),
            "pera_museum": (41.0315, 28.9744),
            "sakip_sabanci_museum": (41.1103, 29.0483),
            "rahmi_koc_museum": (41.0451, 28.9474),
            "dolmabahce_palace": (41.0391, 29.0000),
            "chora_church": (41.0306, 28.9380),
            "rumeli_fortress": (41.0849, 29.0559),
            "galata_tower": (41.0256, 28.9744),
            "suleymaniye_mosque": (41.0166, 28.9639),
            "blue_mosque": (41.0054, 28.9768)
        }
        
        nearby_museums = []
        user_lat, user_lon = user_gps
        
        for museum_id, museum_gps in MUSEUM_GPS.items():
            museum_lat, museum_lon = museum_gps
            distance = haversine(user_lat, user_lon, museum_lat, museum_lon)
            
            if distance <= radius_km:
                # Get museum info from database
                museum_info = self.museum_db.museums.get(museum_id)
                if museum_info:
                    nearby_museums.append({
                        'name': museum_info.name,
                        'type': 'museum',
                        'gps': museum_gps,
                        'distance_km': round(distance, 2),
                        'walking_time_min': int(distance * 12),  # ~12 min per km
                        'address': museum_info.location,
                        'description': museum_info.historical_significance[:200],
                        'opening_hours': museum_info.opening_hours,
                        'entrance_fee': museum_info.entrance_fee,
                        'visiting_duration': museum_info.visiting_duration,
                        'highlights': museum_info.must_see_highlights[:3],
                        'database': 'accurate_museum_database'
                    })
        
        return nearby_museums
    
    def _enhance_with_transport(
        self, locations: List[Dict], user_gps: tuple
    ) -> List[Dict]:
        """Enhance locations with transport recommendations."""
        enhanced = []
        
        for location in locations:
            try:
                # Get transport options if GPS route service is available
                if self.gps_route_service and 'gps' in location:
                    location_gps = location['gps']
                    transport_info = self.gps_route_service.get_route_to_location(
                        from_gps=user_gps,
                        to_gps=location_gps,
                        location_name=location.get('name', 'destination')
                    )
                    
                    if transport_info:
                        location['transport'] = transport_info
                
                enhanced.append(location)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to enhance location with transport: {e}")
                enhanced.append(location)
        
        return enhanced
    
    def _apply_personalization(
        self, locations: List[Dict], user_profile: Any
    ) -> List[Dict]:
        """Apply user personalization to location recommendations."""
        try:
            # Get user preferences
            preferences = getattr(user_profile, 'preferences', {})
            visited = getattr(user_profile, 'visited_locations', [])
            
            # Score and sort by personalization
            for location in locations:
                score = 1.0
                
                # Boost based on preferences
                if 'type' in location:
                    loc_type = location['type']
                    if loc_type in preferences.get('favorite_types', []):
                        score *= 1.5
                
                # Penalize already visited
                if location.get('name') in visited:
                    score *= 0.5
                
                location['personalization_score'] = score
            
            # Sort by combined distance and personalization score
            locations.sort(
                key=lambda x: x['distance_km'] / x.get('personalization_score', 1.0)
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Personalization failed: {e}")
        
        return locations
    
    def _generate_response(
        self,
        locations: List[Dict],
        user_gps: tuple,
        radius_km: float,
        location_types: List[str],
        return_structured: bool
    ) -> Any:
        """Generate formatted response with nearby locations and map visualization."""
        
        # ==================== LLM ENHANCEMENT ====================
        # Build GPS context and get LLM recommendation if available
        gps_context = {
            'gps_location': user_gps,
            'has_gps': True
        }
        
        # Try to detect district
        if self.has_gps_location:
            try:
                district_info = self.gps_location_service.get_district_from_coordinates(
                    user_gps[0], user_gps[1]
                )
                if district_info:
                    gps_context['district'] = district_info.get('district')
                    gps_context['confidence'] = district_info.get('confidence', 0.0)
            except Exception as e:
                logger.warning(f"District detection failed: {e}")
        
        # Get LLM recommendation
        llm_recommendation = None
        if self.has_llm and locations:
            try:
                search_criteria = {
                    'radius_km': radius_km,
                    'types': location_types,
                    'count': len(locations)
                }
                
                llm_recommendation = self._enhance_with_llm(
                    locations=locations,
                    gps_context=gps_context,
                    search_criteria=search_criteria,
                    user_preferences={}
                )
                
                if llm_recommendation:
                    logger.info(f"✨ LLM recommendation generated ({len(llm_recommendation)} chars)")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
        # ==================== END LLM ENHANCEMENT ====================
        
        # Build text response
        response_parts = []
        
        # Add LLM recommendation at the top if available
        if llm_recommendation:
            response_parts.append(f"✨ {llm_recommendation}\n")
        
        response_parts.append(
            f"📍 Here are {len(locations)} locations near you (within {radius_km}km):\n"
        )
        
        for i, location in enumerate(locations, 1):
            name = location.get('name', 'Unknown')
            loc_type = location.get('type', 'location')
            distance = location.get('distance_km', 0)
            walking_time = location.get('walking_time_min', 0)
            
            # Basic info
            response_parts.append(
                f"\n{i}. **{name}** ({loc_type.title()})"
            )
            response_parts.append(
                f"   📏 Distance: {distance:.1f}km (~{walking_time} min walk)"
            )
            
            # Address if available
            if 'address' in location:
                response_parts.append(f"   📍 {location['address']}")
            
            # Enhanced museum information from accurate database
            if 'entrance_fee' in location:
                response_parts.append(f"   🎫 {location['entrance_fee']}")
            
            if 'visiting_duration' in location:
                response_parts.append(f"   ⏱️ Visit: {location['visiting_duration']}")
            
            if 'highlights' in location:
                highlights = location['highlights']
                if highlights:
                    response_parts.append(f"   ⭐ Highlights: {', '.join(highlights[:2])}")
            
            # Transport recommendations
            if 'transport' in location and location['transport']:
                transport = location['transport']
                if 'recommended' in transport:
                    recommended = transport['recommended']
                    response_parts.append(
                        f"   🚇 Best route: {recommended.get('summary', 'Multiple options available')}"
                    )
            
            # Description if available
            elif 'description' in location:
                desc = location['description'][:150]
                response_parts.append(f"   ℹ️ {desc}...")
        
        response_text = '\n'.join(response_parts)
        
        # Generate map visualization data if map engine is available
        map_data = self._generate_map_data(locations, user_gps, radius_km)
        
        if return_structured:
            return {
                'response': response_text,
                'locations': locations,
                'user_gps': user_gps,
                'radius_km': radius_km,
                'count': len(locations),
                'map_data': map_data,
                'gps_context': gps_context,  # Include GPS context
                'llm_enhanced': llm_recommendation is not None,  # Flag for LLM enhancement
                'visualization_available': map_data.get('html') is not None
            }
        
        return response_text
    
    def _generate_map_data(
        self, locations: List[Dict], user_gps: tuple, radius_km: float
    ) -> Dict:
        """Generate interactive map visualization data using MapVisualizationEngine."""
        if not self.map_engine or not MAP_ENGINE_AVAILABLE:
            return {
                'user_location': {'lat': user_gps[0], 'lng': user_gps[1]},
                'locations': [
                    {
                        'name': loc.get('name'),
                        'lat': loc.get('gps', [0, 0])[0],
                        'lng': loc.get('gps', [0, 0])[1],
                        'type': loc.get('type'),
                        'distance_km': loc.get('distance_km')
                    }
                    for loc in locations
                ],
                'html': None
            }
        
        try:
            # Create map locations for all nearby places
            map_locations = []
            
            # Add user location
            user_location = self.map_engine.create_location(
                lat=user_gps[0],
                lon=user_gps[1],
                name="Your Location",
                type='start',
                metadata={'icon': '📍', 'color': 'red'}
            )
            map_locations.append(user_location)
            
            # Add nearby locations
            for loc in locations:
                if 'gps' in loc:
                    gps = loc['gps']
                    map_loc = self.map_engine.create_location(
                        lat=gps[0],
                        lon=gps[1],
                        name=loc.get('name', 'Unknown'),
                        type='poi',
                        metadata={
                            'type': loc.get('type', 'location'),
                            'distance_km': loc.get('distance_km', 0),
                            'walking_time_min': loc.get('walking_time_min', 0),
                            'address': loc.get('address', ''),
                            'description': loc.get('description', ''),
                            'entrance_fee': loc.get('entrance_fee', ''),
                            'icon': '🏛️' if loc.get('type') == 'museum' else '📍'
                        }
                    )
                    map_locations.append(map_loc)
            
            # Calculate map bounds
            bounds = self.map_engine.calculate_bounds(map_locations)
            
            # Generate HTML map using map engine
            map_html = self._generate_leaflet_map(map_locations, user_gps, bounds)
            
            return {
                'user_location': {'lat': user_gps[0], 'lng': user_gps[1]},
                'locations': [
                    {
                        'name': loc.get('name'),
                        'lat': loc.get('gps', [0, 0])[0],
                        'lng': loc.get('gps', [0, 0])[1],
                        'type': loc.get('type'),
                        'distance_km': loc.get('distance_km'),
                        'address': loc.get('address', ''),
                        'entrance_fee': loc.get('entrance_fee', '')
                    }
                    for loc in locations
                ],
                'bounds': bounds,
                'html': map_html,
                'center': user_gps,
                'zoom': self._calculate_zoom_level(radius_km)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating map data: {e}")
            return {
                'user_location': {'lat': user_gps[0], 'lng': user_gps[1]},
                'locations': [],
                'html': None,
                'error': str(e)
            }
    
    def _generate_leaflet_map(
        self, map_locations: List, user_gps: tuple, bounds: Dict
    ) -> Optional[str]:
        """Generate Leaflet.js HTML map."""
        try:
            # Use the existing map engine to generate the visualization
            # This integrates with the same system used by the route planner
            center = user_gps
            zoom = 14
            
            # Generate simplified HTML map (can be enhanced with full Leaflet template)
            locations_json = []
            for loc in map_locations:
                locations_json.append({
                    'lat': loc.lat,
                    'lon': loc.lon,
                    'name': loc.name,
                    'type': loc.type,
                    'metadata': loc.metadata
                })
            
            return {
                'type': 'leaflet',
                'center': center,
                'zoom': zoom,
                'locations': locations_json,
                'bounds': bounds
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating Leaflet map: {e}")
            return None
    
    def _calculate_zoom_level(self, radius_km: float) -> int:
        """Calculate appropriate zoom level based on radius."""
        if radius_km <= 1:
            return 15  # Street level
        elif radius_km <= 3:
            return 14  # Neighborhood
        elif radius_km <= 5:
            return 13  # District
        else:
            return 12  # City area
    
    def _build_gps_context(self, user_profile) -> Dict[str, Any]:
        """
        Build GPS context from user profile for LLM enhancement.
        
        Args:
            user_profile: User profile with optional GPS location
            
        Returns:
            Dictionary with GPS context
        """
        gps_context = {
            'gps_location': None,
            'district': None,
            'confidence': 0.0,
            'has_gps': False
        }
        
        # Extract GPS from user profile
        if user_profile and hasattr(user_profile, 'current_location'):
            gps_location = user_profile.current_location
            if gps_location and isinstance(gps_location, tuple) and len(gps_location) == 2:
                gps_context['gps_location'] = gps_location
                gps_context['has_gps'] = True
                
                # Detect district using GPS location service
                if self.has_gps_location:
                    try:
                        district_info = self.gps_location_service.get_district_from_coordinates(
                            gps_location[0], gps_location[1]
                        )
                        if district_info:
                            gps_context['district'] = district_info.get('district')
                            gps_context['confidence'] = district_info.get('confidence', 0.0)
                            logger.info(
                                f"📍 Detected district: {gps_context['district']} "
                                f"(confidence: {gps_context['confidence']:.2f})"
                            )
                    except Exception as e:
                        logger.warning(f"District detection failed: {e}")
        
        return gps_context
    
    def _enhance_with_llm(
        self,
        locations: List[Dict[str, Any]],
        gps_context: Dict[str, Any],
        search_criteria: Dict[str, Any],
        user_preferences: Optional[Dict] = None
    ) -> str:
        """
        Enhance nearby locations response with LLM-generated recommendations.
        
        Args:
            locations: List of nearby POIs
            gps_context: GPS context from _build_gps_context
            search_criteria: Search parameters (radius, types, etc.)
            user_preferences: Optional user preferences
            
        Returns:
            LLM-generated POI recommendation (concise, contextual)
        """
        if not self.has_llm:
            return ""
        
        try:
            # Get LLM recommendation
            llm_advice = self.llm_service.get_poi_recommendation(
                locations=locations,
                gps_context=gps_context,
                search_criteria=search_criteria,
                user_preferences=user_preferences or {}
            )
            
            logger.info(f"✨ LLM POI recommendation generated ({len(llm_advice)} chars)")
            return llm_advice
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return ""
    
    def _generate_no_gps_response(self, return_structured: bool) -> Any:
        """Generate response when GPS coordinates are not available."""
        response_text = (
            "📍 To show you nearby locations, I need your current GPS coordinates.\n\n"
            "You can:\n"
            "• Share your location with me\n"
            "• Specify a landmark (e.g., 'near Taksim Square')\n"
            "• Provide GPS coordinates\n\n"
            "Once I know where you are, I can recommend museums, attractions, "
            "restaurants and more in your area!"
        )
        
        if return_structured:
            return {
                'response': response_text,
                'error': 'no_gps_coordinates',
                'requires_location': True
            }
        
        return response_text
    
    def _generate_no_results_response(
        self, user_gps: tuple, radius_km: float, return_structured: bool
    ) -> Any:
        """Generate response when no locations found in range."""
        response_text = (
            f"📍 I couldn't find any locations within {radius_km}km of your position.\n\n"
            "You can try:\n"
            "• Expanding the search radius (e.g., 'within 5km')\n"
            "• Searching for a specific type (e.g., 'museums near me')\n"
            "• Moving to a different area of Istanbul\n\n"
            "Istanbul has many amazing places to discover!"
        )
        
        if return_structured:
            return {
                'response': response_text,
                'locations': [],
                'user_gps': user_gps,
                'radius_km': radius_km,
                'count': 0
            }
        
        return response_text
    
    def _generate_error_response(self, error: str, return_structured: bool) -> Any:
        """Generate error response."""
        response_text = (
            "❌ I encountered an error while searching for nearby locations.\n"
            "Please try again or ask me something else about Istanbul!"
        )
        
        if return_structured:
            return {
                'response': response_text,
                'error': error
            }
        
        return response_text
    
    def _track_query(
        self, message: str, user_profile: Any, results: List[Dict]
    ) -> None:
        """Track query for analytics (optional)."""
        try:
            # This could be extended to log to analytics service
            logger.info(
                f"📊 Nearby query tracked: {len(results)} results, "
                f"user: {getattr(user_profile, 'user_id', 'unknown')}"
            )
        except Exception as e:
            logger.debug(f"Analytics tracking failed: {e}")


def create_nearby_locations_handler(
    gps_route_service=None,
    location_database_service=None,
    neural_processor=None,
    user_manager=None,
    transport_service=None,
    llm_service=None,
    gps_location_service=None
):
    """
    Factory function to create a nearby locations response handler.
    
    This function returns a handler callable compatible with the response router.
    
    Args:
        gps_route_service: GPS route planning service
        location_database_service: Unified location database service
        neural_processor: Optional ML processor
        user_manager: Optional user manager
        transport_service: Optional transport service
        llm_service: Optional LLM service for GPS-aware POI recommendations
        gps_location_service: GPS location service for district detection
        
    Returns:
        Callable handler function
    """
    handler = NearbyLocationsHandler(
        gps_route_service=gps_route_service,
        location_database_service=location_database_service,
        neural_processor=neural_processor,
        user_manager=user_manager,
        transport_service=transport_service,
        llm_service=llm_service,
        gps_location_service=gps_location_service
    )
    
    # Return a closure that matches the expected handler signature
    def nearby_locations_response_handler(
        message: str,
        entities: Dict,
        user_profile: Any,
        context: Any,
        neural_insights: Optional[Dict] = None,
        return_structured: bool = False
    ):
        return handler.handle(
            message=message,
            entities=entities,
            user_profile=user_profile,
            context=context,
            neural_insights=neural_insights,
            return_structured=return_structured
        )
    
    return nearby_locations_response_handler
