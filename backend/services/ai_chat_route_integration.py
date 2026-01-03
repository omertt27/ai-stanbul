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
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Import custom exceptions
try:
    from .route_exceptions import (
        RouteServiceError,
        LocationExtractionError,
        InsufficientLocationsError,
        ServiceUnavailableError,
        GeocodingError,
        GPSPermissionRequiredError,
        NavigationError,
        FallbackRoutingUsedWarning
    )
    EXCEPTIONS_AVAILABLE = True
except ImportError:
    # Fallback to generic exceptions
    RouteServiceError = Exception
    LocationExtractionError = Exception
    InsufficientLocationsError = Exception
    ServiceUnavailableError = Exception
    GeocodingError = Exception
    GPSPermissionRequiredError = Exception
    NavigationError = Exception
    FallbackRoutingUsedWarning = UserWarning
    EXCEPTIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import LLM route preference detector
try:
    from .llm.route_preference_detector import detect_route_preferences
    LLM_PREFERENCES_AVAILABLE = True
    logger.info("✅ LLM Route Preference Detector available")
except ImportError as e:
    LLM_PREFERENCES_AVAILABLE = False
    logger.warning(f"⚠️ LLM Route Preference Detector not available: {e}")

# Import intelligent route integration
try:
    # Try relative import first (we're in services/)
    try:
        from .intelligent_route_integration import (
            IntelligentRouteIntegration,
            IntelligentRoute,
            create_intelligent_route_integration
        )
    except ImportError:
        # Fallback to absolute import
        from services.intelligent_route_integration import (
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
    try:
        from .multi_stop_route_planner import (
            MultiStopRoutePlanner,
            OptimizationStrategy,
            PointOfInterest
        )
    except ImportError:
        from services.multi_stop_route_planner import (
            MultiStopRoutePlanner,
            OptimizationStrategy,
            PointOfInterest
        )
    MULTI_STOP_AVAILABLE = True
    logger.info("✅ Multi-stop route planner available")
except ImportError as e:
    MULTI_STOP_AVAILABLE = False
    logger.warning(f"⚠️ Multi-stop route planner not available: {e}")

# Import GPS turn-by-turn navigation
try:
    try:
        from .gps_turn_by_turn_navigation import (
            GPSTurnByTurnNavigator,
            NavigationMode,
            GPSLocation,
            RouteStep,
            InstructionType,
            convert_osrm_to_steps
        )
    except ImportError:
        from services.gps_turn_by_turn_navigation import (
            GPSTurnByTurnNavigator,
            NavigationMode,
            GPSLocation,
            RouteStep,
            InstructionType,
            convert_osrm_to_steps
        )
    GPS_NAVIGATION_AVAILABLE = True
    logger.info("✅ GPS turn-by-turn navigation available")
except ImportError as e:
    GPS_NAVIGATION_AVAILABLE = False
    logger.warning(f"⚠️ GPS turn-by-turn navigation not available: {e}")

# Import Istanbul Geocoder for location fallback
try:
    import sys
    import os
    # Add parent directory to path for geocoder import
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from services.istanbul_geocoder import IstanbulGeocoder, GeocodedLocation
    GEOCODER_AVAILABLE = True
    logger.info("✅ Istanbul Geocoder available for location fallback")
except ImportError as e:
    GEOCODER_AVAILABLE = False
    logger.warning(f"⚠️ Istanbul Geocoder not available: {e}")

# Note: Using general LLM system Redis cache - no specialized route cache needed
ROUTE_CACHE_AVAILABLE = False


def normalize_turkish(text: str) -> str:
    """
    Normalize Turkish characters to ASCII equivalents for better matching.
    
    Turkish special characters:
    - ç → c
    - ğ → g
    - ı → i
    - İ → i  
    - ö → o
    - ş → s
    - ü → u
    
    Args:
        text: Text with potentially Turkish characters
        
    Returns:
        Normalized ASCII text
    """
    replacements = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U',
    }
    
    for turkish_char, ascii_char in replacements.items():
        text = text.replace(turkish_char, ascii_char)
    
    return text


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
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize chat route handler
        
        REFACTORED (Option 3): Now uses Transportation RAG System for all routing
        Only keeps NLP detection and response formatting.
        
        Args:
            redis_url: Redis connection URL for route caching (optional)
        """
        global MULTI_STOP_AVAILABLE
        
        # NEW: Use Transportation RAG System for actual routing
        try:
            from services.transportation_rag_system import get_transportation_rag
            self.transport_rag = get_transportation_rag()
            logger.info("✅ Using Transportation RAG System for route planning")
        except ImportError as e:
            logger.error(f"❌ Failed to import Transportation RAG: {e}")
            self.transport_rag = None
        
        # DEPRECATED: Old routing systems (kept for backwards compatibility)
        self.route_integration = None
        self.multi_stop_planner = None
        self.geocoder = None
        self.route_cache = None
        self.active_navigators: Dict[str, GPSTurnByTurnNavigator] = {}  # session_id -> navigator
        self.navigation_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> navigation state
        
        logger.info("✅ AI Chat Route Handler initialized (RAG-powered)")
        
        if MULTI_STOP_AVAILABLE:
            try:
                self.multi_stop_planner = MultiStopRoutePlanner()
                logger.info("✅ Multi-stop planner initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize multi-stop planner: {e}")
                MULTI_STOP_AVAILABLE = False
        
        # Initialize Istanbul Geocoder for location fallback
        if GEOCODER_AVAILABLE:
            try:
                self.geocoder = IstanbulGeocoder(use_external_geocoding=True)
                logger.info("✅ Istanbul Geocoder initialized with external fallback enabled")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize geocoder: {e}")
                self.geocoder = None
        
        # Initialize Route Cache System
        if ROUTE_CACHE_AVAILABLE:
            try:
                self.route_cache = RouteCacheManager(redis_url=redis_url)
                logger.info("✅ Route Cache System initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize route cache: {e}")
                self.route_cache = None
    
    async def handle_route_request(
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
        logger.info(f"🔍 [ROUTE HANDLER] Checking message: '{message}'")
        is_route = self._is_route_request(message)
        logger.info(f"🔍 [ROUTE HANDLER] Is route request: {is_route}")
        
        if not is_route:
            logger.info(f"❌ [ROUTE HANDLER] Not a route request, returning None")
            return None
        
        logger.info(f"✅ [ROUTE HANDLER] Detected as route request! Proceeding with location extraction...")
        
        # Check if it's a multi-stop request (uses different planner)
        is_multi_stop = self._is_multi_stop_request(message)
        logger.info(f"🔍 [ROUTE HANDLER] Is multi-stop: {is_multi_stop}")
        
        if is_multi_stop and MULTI_STOP_AVAILABLE and self.multi_stop_planner:
            return self._handle_multi_stop_request(message, user_context)
        
        # Main route planning flow with comprehensive error handling
        try:
            # Step 1: Validate and prepare locations
            locations, _ = await self._validate_and_prepare_locations(message, user_context)
            
            # Step 2: Extract route preferences using LLM
            route_preferences = await self._extract_route_preferences(message, locations, user_context)
            
            # Step 3: Determine transport mode
            transport_mode = self._determine_transport_mode(message, route_preferences)
            
            # Step 4: Build routing parameters
            routing_params = self._build_routing_params(route_preferences, user_context)
            
            # Step 5: Plan route (single or multi-location)
            if len(locations) == 2:
                response = await self._plan_single_route(
                    locations, transport_mode, routing_params, route_preferences
                )
            else:
                response = await self._plan_multi_location_route(
                    message, locations, transport_mode, user_context
                )
            
            return response
            
        except (GPSPermissionRequiredError, InsufficientLocationsError, 
                LocationExtractionError, NavigationError, ServiceUnavailableError) as e:
            # Handle expected custom exceptions
            return self._handle_route_planning_error(e)
        
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in route planning: {e}", exc_info=True)
            return self._handle_route_planning_error(e)

    # ========== Route Planning Helper Methods ==========
    
    async def _validate_and_prepare_locations(
        self,
        message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        """
        Validate and prepare locations from message
        
        Returns:
            Tuple of (locations list, extraction metadata)
        
        Raises:
            InsufficientLocationsError: If less than 2 locations found
            LocationExtractionError: If extraction fails
        """
        # Extract locations using existing _extract_locations method
        locations = await self._extract_locations(message)
        
        user_gps = None
        if len(locations) < 2:
            # Check if user has GPS location for single-location queries
            user_gps = self._get_user_gps_location(user_context)
            if user_gps and len(locations) == 1:
                locations.insert(0, user_gps)  # Add as starting point
            else:
                raise InsufficientLocationsError(
                    "I need at least 2 locations to plan a route. "
                    "Please specify both a starting point and destination, "
                    "or enable GPS location to use your current location."
                )
        
        metadata = {
            'total_locations': len(locations),
            'extraction_method': 'nlp_pattern_matching',
            'has_gps_fallback': user_gps is not None
        }
        
        return locations, metadata
    
    async def _extract_route_preferences(
        self,
        message: str,
        locations: List[Tuple[float, float]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract route preferences from message using LLM or patterns
        
        Returns:
            Dict with preferences: {
                'avoid_hills': bool,
                'prefer_scenic': bool,
                'accessible': bool,
                'fastest': bool,
                'time_limit': Optional[int]
            }
        """
        preferences = {}
        
        if LLM_PREFERENCES_AVAILABLE:
            # Use LLM to detect preferences
            try:
                llm_prefs = await detect_route_preferences(message)
                preferences.update(llm_prefs)
            except Exception as e:
                logger.warning(f"LLM preference detection failed: {e}")
        
        # Pattern-based fallback
        message_lower = message.lower()
        
        preferences['avoid_hills'] = any(word in message_lower for word in ['flat', 'avoid hills', 'no hills'])
        preferences['prefer_scenic'] = any(word in message_lower for word in ['scenic', 'beautiful', 'nice views'])
        preferences['accessible'] = any(word in message_lower for word in ['accessible', 'wheelchair'])
        preferences['fastest'] = any(word in message_lower for word in ['fastest', 'quickest', 'hurry'])
        
        # Extract time limit
        time_match = re.search(r'(\d+)\s*(min|minute|hour)', message_lower)
        if time_match:
            value = int(time_match.group(1))
            unit = time_match.group(2)
            preferences['time_limit'] = value * 60 if 'hour' in unit else value
        
        # Merge with user context preferences
        if user_context and user_context.get('preferences'):
            preferences.update(user_context['preferences'])
        
        return preferences
    
    def _determine_transport_mode(
        self,
        message: str,
        route_preferences: Dict[str, Any]
    ) -> str:
        """
        Determine transport mode from message and preferences
        
        Returns:
            'foot' | 'bicycle' | 'car' | 'transit'
        """
        message_lower = message.lower()
        
        # Check for explicit mode keywords
        if any(word in message_lower for word in ['walk', 'walking', 'on foot', 'pedestrian']):
            return 'foot'
        elif any(word in message_lower for word in ['bike', 'bicycle', 'cycling', 'cycle']):
            return 'bicycle'
        elif any(word in message_lower for word in ['drive', 'driving', 'car']):
            return 'car'
        elif any(word in message_lower for word in ['bus', 'metro', 'transit', 'public transport']):
            return 'transit'
        
        # Default based on preferences
        if route_preferences.get('fastest'):
            return 'car'
        elif route_preferences.get('prefer_scenic'):
            return 'foot'
        
        # Default to walking
        return 'foot'
    
    def _build_routing_params(
        self,
        route_preferences: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build routing parameters from preferences
        
        Returns:
            Dict of routing parameters for route integration
        """
        params = {
            'avoid_highways': route_preferences.get('avoid_highways', False),
            'avoid_tolls': route_preferences.get('avoid_tolls', False),
            'avoid_ferries': route_preferences.get('avoid_ferries', False),
            'prefer_scenic': route_preferences.get('prefer_scenic', False),
            'accessible': route_preferences.get('accessible', False)
        }
        
        # Add time constraint if specified
        if 'time_limit' in route_preferences:
            params['max_duration_seconds'] = route_preferences['time_limit'] * 60
        
        # Add user context settings
        if user_context:
            if user_context.get('language'):
                params['language'] = user_context['language']
        
        return params
    
    async def _plan_single_route(
        self,
        locations: List[Tuple[float, float]],
        transport_mode: str,
        routing_params: Dict[str, Any],
        route_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plan a single route between two locations WITH CACHING
        
        This is the primary route planning method with full cache integration.
        
        Args:
            locations: List of 2 coordinate tuples [(lat1, lon1), (lat2, lon2)]
            transport_mode: 'foot' | 'bicycle' | 'car' | 'transit'
            routing_params: Additional routing parameters
            route_preferences: User preferences
            
        Returns:
            Formatted route response with metadata
        """
        start_coords, end_coords = locations[0], locations[1]
        
        # ========== STEP 1: CHECK CACHE ==========
        cached_route = None
        cache_key = None
        
        if self.route_cache:
            try:
                cache_key = self.route_cache._generate_cache_key('route', {
                    'start': start_coords,
                    'end': end_coords,
                    'mode': transport_mode,
                    'params': json.dumps(routing_params, sort_keys=True)
                })
                
                # Try to get from cache
                cached_route = self.route_cache.get_cached_route(cache_key)
                if cached_route:
                    logger.info(f"🚀 CACHE HIT: Route from {start_coords} to {end_coords}")
                    
                    # Add cache metadata
                    cached_route['metadata'] = cached_route.get('metadata', {})
                    cached_route['metadata']['from_cache'] = True
                    cached_route['metadata']['cache_key'] = cache_key[:16] + '...'
                    
                    return self._format_route_response(
                        cached_route,
                        single=True
                    )
            except Exception as e:
                logger.warning(f"Cache lookup error: {e}")
        
        # ========== STEP 2: COMPUTE ROUTE (Cache miss) ==========
        logger.info(f"💾 CACHE MISS: Computing new route from {start_coords} to {end_coords}")
        
        # REFACTORED: Use Transportation RAG System instead of intelligent_route_integration
        if not self.transport_rag:
            raise ServiceUnavailableError("Transportation RAG System is not available")
        
        try:
            # Extract location names from coordinates for RAG query
            origin_name = self._get_location_name(start_coords)
            dest_name = self._get_location_name(end_coords)
            
            logger.info(f"🚇 Using Transportation RAG: {origin_name} → {dest_name}")
            logger.info(f"📍 GPS coordinates - Origin: {start_coords}, Destination: {end_coords}")
            
            # Prepare GPS coordinates for RAG (convert from tuple to dict)
            origin_gps_dict = {
                'lat': start_coords[0],
                'lon': start_coords[1]
            }
            dest_gps_dict = {
                'lat': end_coords[0],
                'lon': end_coords[1]
            }
            
            # Use Transportation RAG to find route with GPS coordinates
            rag_route = self.transport_rag.find_route(
                origin=origin_name,
                destination=dest_name,
                max_transfers=3,
                origin_gps=origin_gps_dict,
                destination_gps=dest_gps_dict
            )
            
            if not rag_route:
                raise NavigationError(
                    f"No route found between {origin_name} and {dest_name}. "
                    "Please try different locations or check if they are reachable."
                )
            
            logger.info(f"✅ Transportation RAG found route: {rag_route.total_duration} min, {len(rag_route.segments)} segments")
            
            # Convert RAG route to our format
            route_data = {
                'distance': rag_route.total_distance,
                'duration': rag_route.total_duration,
                'polyline': None,  # RAG doesn't provide polyline
                'steps': rag_route.steps,
                'segments': rag_route.segments,
                'origin': rag_route.origin,
                'destination': rag_route.destination,
                'waypoints': rag_route.waypoints if hasattr(rag_route, 'waypoints') else [],
                'metadata': {
                    'routing_method': 'transportation_rag',
                    'confidence': rag_route.confidence if hasattr(rag_route, 'confidence') else 0.95,
                    'warnings': rag_route.warnings if hasattr(rag_route, 'warnings') else [],
                    'from_cache': False,
                    'lines_used': [seg.line for seg in rag_route.segments] if rag_route.segments else []
                }
            }
            
            # ========== STEP 3: STORE IN CACHE ==========
            if self.route_cache and cache_key:
                try:
                    # Cache for 24 hours (popular routes) or 6 hours (less popular)
                    is_popular = self._is_popular_route(start_coords, end_coords)
                    ttl_hours = 24 if is_popular else 6
                    
                    self.route_cache.cache_route(cache_key, route_data, ttl_hours=ttl_hours)
                    logger.info(f"💾 Cached route {cache_key[:16]}... (TTL: {ttl_hours}h)")
                    
                except Exception as e:
                    logger.warning(f"Failed to cache route: {e}")
            
            # ========== STEP 4: FORMAT AND RETURN ==========
            return self._format_route_response(route_data, single=True)
            
        except Exception as e:
            logger.error(f"Route planning failed: {e}", exc_info=True)
            raise NavigationError(f"Route planning failed: {str(e)}")
    
    async def _plan_multi_location_route(
        self,
        message: str,
        locations: List[Tuple[float, float]],
        transport_mode: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plan route visiting multiple locations (3+)
        
        Falls back to multi-stop planner if available
        """
        if MULTI_STOP_AVAILABLE and self.multi_stop_planner:
            # Delegate to multi-stop handler
            return self._handle_multi_stop_request(message, user_context)
        else:
            # Simple multi-waypoint routing
            if not self.route_integration:
                raise ServiceUnavailableError("Route planning service is not available")
            
            try:
                # Plan route through all waypoints
                route = self.route_integration.plan_waypoint_route(
                    waypoints=locations,
                    mode=transport_mode
                )
                
                return self._format_route_response(route, {})
                
            except Exception as e:
                raise NavigationError(f"Multi-location routing failed: {str(e)}")
    
    def _handle_route_planning_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle route planning errors with user-friendly messages
        
        Args:
            error: The exception that occurred
            
        Returns:
            Error response dict
        """
        # Check if error has custom error_response method (from custom exceptions)
        if EXCEPTIONS_AVAILABLE and hasattr(error, 'error_response'):
            return error.error_response()
        
        # Fallback error handling
        error_msg = str(error)
        
        # Make error messages more user-friendly
        if "location" in error_msg.lower():
            return {
                'type': 'error',
                'message': f"❌ Location Error\n\n{error_msg}\n\n"
                          "**Tip:** Try using well-known landmarks like 'Sultanahmet' or 'Taksim Square'.",
                'error_code': 'LOCATION_ERROR'
            }
        elif "network" in error_msg.lower() or "unavailable" in error_msg.lower():
            return {
                'type': 'error',
                'message': f"🔌 Service Temporarily Unavailable\n\n"
                          "The routing service is currently unavailable. Please try again in a moment.",
                'error_code': 'SERVICE_UNAVAILABLE'
            }
        else:
            return {
                'type': 'error',
                'message': f"⚠️ Route Planning Error\n\n{error_msg}",
                'error_code': 'ROUTE_PLANNING_ERROR'
            }
    
    def _is_popular_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """
        Determine if a route is between popular tourist locations
        
        Popular routes get longer cache TTL
        """
        # Check if start and end are both in KNOWN_LOCATIONS
        start_is_known = any(
            abs(start[0] - coords[0]) < 0.001 and abs(start[1] - coords[1]) < 0.001
            for coords in self.KNOWN_LOCATIONS.values()
        )
        
        end_is_known = any(
            abs(end[0] - coords[0]) < 0.001 and abs(end[1] - coords[1]) < 0.001
            for coords in self.KNOWN_LOCATIONS.values()
        )
        
        return start_is_known and end_is_known
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get route cache performance statistics"""
        if not self.route_cache:
            return {
                'cache_enabled': False,
                'message': 'Route caching is not available'
            }
        
        stats = self.route_cache.get_cache_stats()
        
        return {
            'cache_enabled': True,
            'total_requests': stats['total_requests'],
            'cache_hits': stats['cache_hits'],
            'cache_misses': stats['cache_misses'],
            'hit_rate': stats.get('cache_hit_rate', 0),
            'cache_size': self.route_cache.get_cache_size(),
            'memory_usage_mb': stats.get('memory_usage_bytes', 0) / (1024 * 1024)
        }

    async def handle_gps_navigation_command(
        self,
        message: str,
        session_id: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Handle GPS navigation commands through chat
        
        Commands:
        - "start navigation to [location]"
        - "navigate to [location]" 
        - "stop navigation"
        - "where am I?"
        - "what's next?"
        - "repeat instruction"
        
        Args:
            message: User's chat message
            session_id: Unique session identifier
            user_location: Current GPS location {'lat': float, 'lon': float}
            
        Returns:
            Dict with navigation response or None if not a navigation command
        """
        if not GPS_NAVIGATION_AVAILABLE:
            return None
        
        message_lower = message.lower()
        
        # Check for navigation commands
        if self._is_start_navigation_command(message_lower):
            return await self._handle_start_navigation(message, session_id, user_location)
        
        elif any(cmd in message_lower for cmd in ['stop navigation', 'end navigation', 'cancel navigation', 'exit navigation']):
            return self._handle_stop_navigation(session_id)
        
        elif any(cmd in message_lower for cmd in ['where am i', 'current location', 'my location']):
            return self._handle_location_query(session_id, user_location)
        
        elif any(cmd in message_lower for cmd in ["what's next", 'next instruction', 'next step', 'continue']):
            return self._handle_next_instruction(session_id, user_location)
        
        elif any(cmd in message_lower for cmd in ['repeat', 'say again', 'what was that']):
            return self._handle_repeat_instruction(session_id)
        
        elif any(cmd in message_lower for cmd in ['navigation status', 'am i navigating']):
            return self._handle_navigation_status(session_id)
        
        elif any(cmd in message_lower for cmd in ['reroute', 'recalculate', 'new route']):
            return await self._handle_reroute(session_id, user_location)
        
        # Auto-update if navigation is active and location provided
        if session_id in self.active_navigators and user_location:
            return self._handle_auto_update(session_id, user_location)
        
        return None
    
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
    
    async def _extract_locations(self, message: str) -> List[Tuple[float, float]]:
        """
        Industry-level location extraction from natural language queries.
        
        Supports patterns:
        - "from X to Y" / "X to Y" 
        - "to Y from X" / "go to Y from X"
        - "between X and Y"
        - "X → Y" / "X -> Y"
        - Multiple locations: "X, Y, and Z"
        - Implicit patterns: "how do I get to X" (uses current location)
        
        Args:
            message: Natural language route query
            
        Returns:
            List of coordinate tuples in journey order
        """
        # Normalize Turkish characters before processing
        message_normalized = normalize_turkish(message.lower().strip())
        logger.info(f"🔍 Starting location extraction from: '{message_normalized}'")
        
        # PATTERN 1: "to Y from X" - destination before origin
        # Examples: "to taksim from kadikoy", "go to galata from sultanahmet", "how can I go to X from Y"
        # FIX: Use proper word boundary matching instead of [^from] which excludes individual chars
        to_from_patterns = [
            r'(?:how\s+(?:can|do)\s+i\s+)?(?:go\s+)?to\s+(.+?)\s+from\s+(.+?)(?:\s*[?.!]|$)',
            r'(?:get|travel|walk|drive)\s+to\s+(.+?)\s+from\s+(.+?)(?:\s*[?.!]|$)',
            r'(?:route|directions)\s+to\s+(.+?)\s+from\s+(.+?)(?:\s*[?.!]|$)',
        ]
        
        for pattern in to_from_patterns:
            match = re.search(pattern, message_normalized, re.IGNORECASE)
            if match:
                logger.info(f"✅ Matched to-from pattern: {pattern}")
                dest_str = match.group(1).strip()
                origin_str = match.group(2).strip()
                logger.info(f"   Raw extracted: origin='{origin_str}', dest='{dest_str}'")
                
                # Clean up common noise words
                origin_str = self._clean_location_string(origin_str)
                dest_str = self._clean_location_string(dest_str)
                logger.info(f"   After cleaning: origin='{origin_str}', dest='{dest_str}'")
                
                origin_coords = await self._find_best_location_match(origin_str)
                dest_coords = await self._find_best_location_match(dest_str)
                
                if origin_coords and dest_coords:
                    logger.info(f"✅ Extracted route (to-from): {origin_str} → {dest_str}")
                    return [origin_coords, dest_coords]
                else:
                    logger.warning(f"⚠️ Failed to find coords: origin={origin_coords}, dest={dest_coords}")
        
        # PATTERN 2: "from X to Y" - traditional direction pattern
        # Examples: "from sultanahmet to galata", "route from X to Y", "directions from A to B"
        # FIX: Use proper word boundary matching
        from_to_patterns = [
            r'(?:from|starting\s+from)\s+(.+?)\s+to\s+(.+?)(?:\s*[?.!]|$)',
            r'(?:route|directions|path|way)\s+from\s+(.+?)\s+to\s+(.+?)(?:\s*[?.!]|$)',
            r'(?:going|traveling|walking)\s+from\s+([^to]+?)\s+to\s+(.+?)(?:\s*[?.!]|$)',
        ]
        
        for pattern in from_to_patterns:
            match = re.search(pattern, message_normalized, re.IGNORECASE)
            if match:
                origin_str = match.group(1).strip()
                dest_str = match.group(2).strip()
                
                origin_str = self._clean_location_string(origin_str)
                dest_str = self._clean_location_string(dest_str)
                
                origin_coords = await self._find_best_location_match(origin_str)
                dest_coords = await self._find_best_location_match(dest_str)
                
                if origin_coords and dest_coords:
                    logger.info(f"✅ Extracted route (from-to): {origin_str} → {dest_str}")
                    return [origin_coords, dest_coords]
        
        # PATTERN 3: "between X and Y" - bidirectional query
        # Examples: "distance between X and Y", "route between A and B"
        between_pattern = r'between\s+([^and]+?)\s+and\s+(.+?)(?:\s*[?.!]|$)'
        match = re.search(between_pattern, message_normalized, re.IGNORECASE)
        if match:
            loc1_str = self._clean_location_string(match.group(1).strip())
            loc2_str = self._clean_location_string(match.group(2).strip())
            
            loc1_coords = await self._find_best_location_match(loc1_str)
            loc2_coords = await self._find_best_location_match(loc2_str)
            
            if loc1_coords and loc2_coords:
                logger.info(f"✅ Extracted route (between): {loc1_str} ↔ {loc2_str}")
                return [loc1_coords, loc2_coords]
        
        # PATTERN 4: Simple "X to Y" without prepositions
        # Examples: "taksim to kadikoy", "sultanahmet → galata"
        simple_patterns = [
            r'^([^to]+?)\s+(?:to|→|->)\s+(.+?)(?:\s*[?.!]|$)',
            r'(?:^|\s)([a-z\s]+)\s+(?:to|→|->)\s+([a-z\s]+)(?:\s*[?.!]|$)',
        ]
        
        for pattern in simple_patterns:
            match = re.search(pattern, message_normalized, re.IGNORECASE)
            if match:
                origin_str = self._clean_location_string(match.group(1).strip())
                dest_str = self._clean_location_string(match.group(2).strip())
                
                origin_coords = await self._find_best_location_match(origin_str)
                dest_coords = await self._find_best_location_match(dest_str)
                
                if origin_coords and dest_coords:
                    logger.info(f"✅ Extracted route (simple): {origin_str} → {dest_str}")
                    return [origin_coords, dest_coords]
        
        # PATTERN 5: Comma-separated list for multi-stop routes
        # Examples: "visit taksim, galata, and sultanahmet", "tour of X, Y, Z"
        if ',' in message_normalized:
            locations = await self._extract_comma_separated_locations(message_normalized)
            if len(locations) >= 2:
                logger.info(f"✅ Extracted multi-stop route: {len(locations)} locations")
                return locations
        
        # FALLBACK: Find all mentioned locations (preserve order)
        found_locations = []
        location_positions = []
        
        for location_name, coords in self.KNOWN_LOCATIONS.items():
            # Use word boundary matching for better accuracy
            pattern = r'\b' + re.escape(location_name) + r'\b'
            match = re.search(pattern, message_normalized)
            if match:
                location_positions.append((match.start(), location_name, coords))
        
        # Sort by position in message to preserve natural order
        location_positions.sort(key=lambda x: x[0])
        
        # Remove duplicates while preserving order
        seen = set()
        for _, name, coords in location_positions:
            if name not in seen:
                seen.add(name)
                found_locations.append(coords)
        
        if found_locations:
            logger.info(f"✅ Found {len(found_locations)} locations (fallback extraction)")
        
        return found_locations
    
    def _clean_location_string(self, loc_str: str) -> str:
        """
        Clean location string by removing noise words and punctuation.
        
        Args:
            loc_str: Raw location string from regex capture
            
        Returns:
            Cleaned location string
        """
        # Remove common noise words
        noise_words = [
            'the', 'a', 'an', 'my', 'your', 'our', 'this', 'that',
            'here', 'there', 'now', 'then', 'at', 'in', 'on', 'by',
            'near', 'around', 'area', 'place', 'location', 'spot'
        ]
        
        # Split into words
        words = loc_str.split()
        
        # Filter noise words (but keep if it's the only word)
        if len(words) > 1:
            words = [w for w in words if w not in noise_words]
        
        # Remove punctuation from ends
        cleaned = ' '.join(words).strip(',.!?;: ')
        
        return cleaned
    
    async def _find_best_location_match(self, query: str) -> Optional[Tuple[float, float]]:
        """
        Find best matching location from KNOWN_LOCATIONS using fuzzy matching.
        
        Strategies:
        1. Exact match (case-insensitive)
        2. Substring match (location name contains query or vice versa)
        3. Word-based partial match (all query words appear in location name)
        4. Geocoder fallback (100+ landmarks + Nominatim API)
        
        Args:
            query: Location query string (cleaned)
            
        Returns:
            Coordinate tuple if found, None otherwise
        """
        query = query.lower().strip()
        logger.info(f"🔍 Looking for location match for: '{query}'")
        
        if not query:
            return None
        
        # Strategy 1: Exact match
        if query in self.KNOWN_LOCATIONS:
            logger.info(f"✅ Found exact match in KNOWN_LOCATIONS for: '{query}'")
            return self.KNOWN_LOCATIONS[query]
        
        # Strategy 2: Check if query is substring of any location name
        for location_name, coords in self.KNOWN_LOCATIONS.items():
            if query in location_name or location_name in query:
                # Prefer shorter matches (more specific)
                logger.info(f"✅ Found substring match in KNOWN_LOCATIONS: '{query}' -> '{location_name}'")
                return coords
        
        # Strategy 3: Word-based partial matching
        query_words = set(query.split())
        best_match = None
        best_match_score = 0
        best_match_name = None
        
        for location_name, coords in self.KNOWN_LOCATIONS.items():
            location_words = set(location_name.split())
            
            # Count matching words
            matching_words = query_words & location_words
            match_score = len(matching_words)
            
            # Require at least one matching word
            if match_score > 0 and match_score > best_match_score:
                best_match = coords
                best_match_score = match_score
                best_match_name = location_name
        
        if best_match:
            logger.info(f"✅ Found word-based match in KNOWN_LOCATIONS: '{query}' -> '{best_match_name}'")
            return best_match
        
        # Strategy 4: Geocoder fallback (100+ landmarks + external geocoding)
        if self.geocoder:
            logger.info(f"🔍 No match in KNOWN_LOCATIONS, trying geocoder fallback for: '{query}'")
            try:
                geocoded = await self.geocoder.geocode_async(query)
                if geocoded:
                    logger.info(f"✅ Geocoder found location: '{query}' -> {geocoded.name} at ({geocoded.lat}, {geocoded.lon}) [source: {geocoded.source}, confidence: {geocoded.confidence}]")
                    return geocoded.to_tuple()
                else:
                    logger.info(f"⚠️ Geocoder could not find location: '{query}'")
            except Exception as e:
                logger.warning(f"⚠️ Error using geocoder for '{query}': {e}")
        
        # No match found
        logger.warning(f"❌ Could not find location match for: '{query}' (tried KNOWN_LOCATIONS and geocoder)")
        return None
    
    async def _extract_comma_separated_locations(self, message: str) -> List[Tuple[float, float]]:
        """
        Extract locations from comma-separated list.
        
        Examples:
        - "visit taksim, galata, and sultanahmet"
        - "tour of X, Y, Z"
        
        Args:
            message: Message containing comma-separated locations
            
        Returns:
            List of coordinates in order mentioned
        """
        locations = []
        
        # Split by commas and 'and'
        parts = re.split(r'[,]|\s+and\s+', message)
        
        for part in parts:
            part = self._clean_location_string(part.strip())
            if part:
                coords = await self._find_best_location_match(part)
                if coords and coords not in locations:
                    locations.append(coords)
        
        return locations
    
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
    
    def _determine_routing_method(self, route: Any) -> str:
        """
        Determine which routing method was used
        
        Returns:
            'precise' - OSRM realistic routing
            'estimated' - GPS-based routing
            'approximate' - Fallback Haversine calculation
        """
        # Check if route has metadata about routing method
        if hasattr(route, 'gps_route_data') and route.gps_route_data:
            osrm_data = route.gps_route_data.get('osrm_route')
            if osrm_data:
                return 'precise'  # OSRM was used
        
        # Check for GPS planner data
        if hasattr(route, 'gps_route_data') and route.gps_route_data:
            return 'estimated'
        
        # Otherwise, likely fallback Haversine
        return 'approximate'
    
    def _calculate_confidence(self, route: Any) -> float:
        """
        Calculate confidence score for route accuracy
        
        Returns:
            Confidence score from 0.0 to 1.0
        """
        score = 0.5  # Base score
        
        routing_method = self._determine_routing_method(route)
        
        if routing_method == 'precise':
            score += 0.4  # OSRM is highly accurate
        elif routing_method == 'estimated':
            score += 0.3  # GPS routing is good
        else:
            score += 0.1  # Haversine is rough estimate
        
        # Add bonus for ML predictions
        if hasattr(route, 'ml_predictions') and route.ml_predictions:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_warnings(self, route: Any) -> List[str]:
        """
        Generate helpful warnings for users
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        routing_method = self._determine_routing_method(route)
        total_distance = route.visualization.total_distance
        
        if routing_method == 'approximate':
            warnings.append(
                "⚠️ Route shown is approximate (straight-line). "
                "Actual walking route may be longer."
            )
        
        if total_distance > 5000:  # More than 5km
            warnings.append(
                "ℹ️ This is a long walk (>5km). Consider using public transit or taxi."
            )
        
        if total_distance > 10000:  # More than 10km
            warnings.append(
                "⚠️ Very long distance! Walking not recommended. "
                "Use metro, bus, or taxi instead."
            )
        
        # Check ML predictions for crowding
        if hasattr(route, 'ml_predictions') and route.ml_predictions:
            crowding = route.ml_predictions.get('crowding', {})
            max_crowding = crowding.get('max', 0)
            if max_crowding > 0.8:
                warnings.append(
                    "👥 Route may be crowded at this time."
                )
        
        return warnings
    
    def _generate_tips(self, route: Any) -> List[str]:
        """
        Generate helpful tips for users
        
        Returns:
            List of tip messages
        """
        tips = []
        
        # Use route recommendations if available
        if hasattr(route, 'recommendations') and route.recommendations:
            return route.recommendations[:3]  # Top 3 recommendations
        
        # Generate generic tips based on route
        total_distance = route.visualization.total_distance
        
        if total_distance > 3000:
            tips.append("💡 Consider stopping for breaks along the way")
        
        if total_distance < 2000:
            tips.append("💡 This is a pleasant walking distance")
        
        # Add district-specific tips
        if hasattr(route.visualization, 'districts') and route.visualization.districts:
            districts = route.visualization.districts
            if 'Sultanahmet' in districts:
                tips.append("📸 Great area for photos!")
            if 'Beyoğlu' in districts:
                tips.append("☕ Many cafes along the way")
        
        return tips
    
    def _create_natural_response(self, route: Any, preferences: Optional[Dict] = None) -> str:
        """
        Create natural language response with routing method awareness
        
        Args:
            route: Route object
            preferences: Optional user preferences
            
        Returns:
            Natural language message
        """
        distance_km = route.visualization.total_distance / 1000
        duration_min = route.visualization.total_duration / 60
        
        routing_method = self._determine_routing_method(route)
        
        # Start with routing method-aware greeting
        if routing_method == 'precise':
            response = f"🚶‍♂️ I found a walking route for you! "
        elif routing_method == 'estimated':
            response = f"🗺️ Here's an estimated route for you. "
        else:
            response = f"📍 Here's an approximate route (straight-line distance). "
        
        response += f"It's about **{distance_km:.1f} km** and will take "
        response += f"approximately **{duration_min:.0f} minutes** on foot.\n\n"
        
        # Add warnings
        warnings = self._generate_warnings(route)
        if warnings:
            for warning in warnings:
                response += f"{warning}\n"
            response += "\n"
        
        # Add tips
        tips = self._generate_tips(route)
        if tips:
            response += "**💡 Tips:**\n"
            for tip in tips:
                response += f"• {tip}\n"
        
        return response.strip()
    
    def _format_route_response(
        self,
        route_or_routes: Any,
        single: bool = True
    ) -> Dict[str, Any]:
        """Format route response for chat with enhanced metadata"""
        
        if single:
            route = route_or_routes
            
            # FALLBACK: If visualization engine is not available, extract from OSRM route
            # This happens when the Map Visualization Engine module is missing
            total_distance = route.visualization.total_distance
            total_duration = route.visualization.total_duration
            
            # If visualization values are 0, try to get from GPS route data (OSRM)
            if total_distance == 0 and hasattr(route, 'gps_route_data') and route.gps_route_data:
                osrm_data = route.gps_route_data.get('osrm_route', {})
                if isinstance(osrm_data, dict):
                    total_distance = osrm_data.get('total_distance', 0)
                    total_duration = osrm_data.get('total_duration', 0)
                elif hasattr(osrm_data, 'total_distance'):  # OSRMRoute object
                    total_distance = osrm_data.total_distance
                    total_duration = osrm_data.total_duration
            
                if total_distance > 0:
                    logger.info(f"✅ Using OSRM fallback: {total_distance}m, {total_duration}s")
            
            # Create natural language message with routing method awareness
            message = self._create_natural_response(route)
            
            # Determine routing method and confidence
            routing_method = self._determine_routing_method(route)
            confidence = self._calculate_confidence(route)
            
            # Export visualization data
            visualization_data = self.route_integration.export_for_frontend(route, format='leaflet')
            
            # Build response with enhanced metadata
            response = {
                'type': 'route',
                'message': message,
                'route_data': {
                    'single': True,
                    'visualization': visualization_data,
                    'start': route.start_location,
                    'end': route.end_location,
                    'origin': route.start_location,  # Alias for frontend compatibility
                    'destination': route.end_location,  # Alias for frontend compatibility
                    'distance': total_distance,
                    'duration': total_duration,
                    'total_distance': total_distance,
                    'total_time': total_duration,
                    'waypoints': route.visualization.waypoints,
                    'steps': route.visualization.steps,
                    'geojson': route.visualization.geojson,
                    'routing_method': routing_method,
                    'confidence': confidence
                },
                'metadata': {
                    'routing_method': routing_method,
                    'routing_method_description': {
                        'precise': 'Realistic walking route via OpenStreetMap',
                        'estimated': 'GPS-based route estimation',
                        'approximate': 'Straight-line distance approximation'
                    }.get(routing_method, 'Unknown'),
                    'confidence': confidence,
                    'warnings': self._generate_warnings(route),
                    'tips': self._generate_tips(route)
                }
            }
            
            return response
        
        else:
            routes = route_or_routes
            
            # Calculate totals with fallback for missing visualization data
            total_distance = 0
            total_duration = 0
            
            for r in routes:
                route_distance = r.visualization.total_distance
                route_duration = r.visualization.total_duration
                
                # Fallback to OSRM data if visualization is 0
                if route_distance == 0 and hasattr(r, 'gps_route_data') and r.gps_route_data:
                    osrm_data = r.gps_route_data.get('osrm_route', {})
                    if isinstance(osrm_data, dict):
                        route_distance = osrm_data.get('total_distance', 0)
                        route_duration = osrm_data.get('total_duration', 0)
                    elif hasattr(osrm_data, 'total_distance'):
                        route_distance = osrm_data.total_distance
                        route_duration = osrm_data.total_duration
                
                total_distance += route_distance
                total_duration += route_duration
            
            # Convert to km and minutes
            total_distance = total_distance / 1000
            total_duration = total_duration / 60
            
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
            
            # Add metadata for each route
            for i, r in enumerate(routes):
                r.routing_method = self._determine_routing_method(r)
                r.confidence = self._calculate_confidence(r)
                
                # Add warnings and tips to the first route only (summary route)
                if i == 0:
                    r.warnings = self._generate_warnings(r)
                    r.tips = self._generate_tips(r)
            
            return {
                'type': 'route',
                'message': message,
                'route_data': {
                    'single': False,
                    'routes': routes_data,
                    'origin': routes[0].start_location if routes else None,  # First route origin
                    'destination': routes[-1].end_location if routes else None,  # Last route destination
                    'total_distance': total_distance * 1000,
                    'total_duration': total_duration * 60,
                    'total_time': total_duration * 60,  # Alias for consistency
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
    
    # ========== GPS Turn-by-Turn Navigation Methods ==========
    
    def start_gps_navigation(
        self,
        session_id: str,
        route_data: Dict[str, Any],
        current_location: Dict[str, float],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Start GPS turn-by-turn navigation
        
        Args:
            session_id: Unique session identifier
            route_data: Route data from planning
            current_location: Current GPS location {lat, lon, accuracy}
            language: Language for instructions (en, tr)
            
        Returns:
            Initial navigation state
        """
        if not GPS_NAVIGATION_AVAILABLE:
            return {
                'type': 'error',
                'message': 'GPS navigation is not available'
            }
        
        try:
            # Extract route steps from route data
            if 'osrm_route' in route_data:
                # Convert OSRM route to steps
                steps = convert_osrm_to_steps(route_data['osrm_route'])
            elif 'steps' in route_data:
                # Direct route steps
                steps = route_data['steps']
            else:
                return {
                    'type': 'error',
                    'message': 'Invalid route data format'
                }
            
            if not steps:
                return {
                    'type': 'error',
                    'message': 'No route steps available'
                }
            
            # Determine navigation mode
            mode_map = {
                'walking': NavigationMode.WALKING,
                'cycling': NavigationMode.CYCLING,
                'driving': NavigationMode.DRIVING,
                'transit': NavigationMode.TRANSIT
            }
            nav_mode = mode_map.get(route_data.get('mode', 'walking'), NavigationMode.WALKING)
            
            # Create navigator
            navigator = GPSTurnByTurnNavigator(
                route_steps=steps,
                mode=nav_mode,
                language=language
            )
            
            # Store navigator
            self.active_navigators[session_id] = navigator
            
            # Create GPS location
            gps_location = GPSLocation(
                latitude=current_location['lat'],
                longitude=current_location['lon'],
                accuracy=current_location.get('accuracy', 10.0),
                speed=current_location.get('speed'),
                bearing=current_location.get('bearing')
            )
            
            # Start navigation
            state = navigator.start_navigation(gps_location)
            
            logger.info(f"🧭 GPS navigation started for session {session_id}")
            
            return {
                'type': 'navigation_started',
                'message': '🧭 **Turn-by-turn navigation started!**\n\nFollow the instructions below.',
                'session_id': session_id,
                'navigation_state': state.to_dict(),
                'route_overview': navigator.get_route_overview()
            }
            
        except Exception as e:
            logger.error(f"Error starting GPS navigation: {e}", exc_info=True)
            return {
                'type': 'error',
                'message': f'Failed to start navigation: {str(e)}'
            }
    
    def update_gps_navigation(
        self,
        session_id: str,
        current_location: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Update GPS navigation with new location
        
        Args:
            session_id: Session identifier
            current_location: Current GPS location {lat, lon, accuracy, speed, bearing}
            
        Returns:
            Updated navigation state
        """
        if not GPS_NAVIGATION_AVAILABLE:
            return {
                'type': 'error',
                'message': 'GPS navigation is not available'
            }
        
        # Check if navigator exists
        if session_id not in self.active_navigators:
            return {
                'type': 'error',
                'message': 'No active navigation session found. Please start navigation first.'
            }
        
        try:
            navigator = self.active_navigators[session_id]
            
            # Create GPS location
            gps_location = GPSLocation(
                latitude=current_location['lat'],
                longitude=current_location['lon'],
                accuracy=current_location.get('accuracy', 10.0),
                speed=current_location.get('speed'),
                bearing=current_location.get('bearing')
            )
            
            # Update navigation
            state = navigator.update_location(gps_location)
            
            # Check if arrived
            if state.has_arrived:
                # Clean up navigator
                del self.active_navigators[session_id]
                logger.info(f"🎯 Navigation completed for session {session_id}")
                
                return {
                    'type': 'navigation_completed',
                    'message': '🎯 **You have arrived at your destination!**\n\nNavigation completed successfully.',
                    'navigation_state': state.to_dict()
                }
            
            # Check if rerouting needed
            if state.off_route and len(state.warnings) > 0:
                return {
                    'type': 'navigation_update',
                    'message': '⚠️ **Off Route**\n\nYou are off the planned route. Would you like to recalculate?',
                    'navigation_state': state.to_dict(),
                    'rerouting_suggested': True
                }
            
            return {
                'type': 'navigation_update',
                'navigation_state': state.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error updating GPS navigation: {e}", exc_info=True)
            return {
                'type': 'error',
                'message': f'Failed to update navigation: {str(e)}'
            }
    
    def request_reroute(
        self,
        session_id: str,
        current_location: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Request rerouting from current location
        
        Args:
            session_id: Session identifier
            current_location: Current GPS location
            
        Returns:
            Rerouting result
        """
        if session_id not in self.active_navigators:
            return {
                'type': 'error',
                'message': 'No active navigation session found'
            }
        
        try:
            navigator = self.active_navigators[session_id]
            
            # Get destination
            destination = navigator.destination
            if not destination:
                return {
                    'type': 'error',
                    'message': 'No destination available for rerouting'
                }
            
            # Plan new route using route integration
            if not self.route_integration:
                return {
                    'type': 'error',
                    'message': 'Route planning not available'
                }
            
            new_route = self.route_integration.plan_intelligent_route(
                start=(current_location['lat'], current_location['lon']),
                end=destination,
                transport_mode='walking',
                user_context={}
            )
            
            # Convert to route steps
            if hasattr(new_route, 'osrm_route') and new_route.osrm_route:
                new_steps = convert_osrm_to_steps(new_route.osrm_route)
            else:
                return {
                    'type': 'error',
                    'message': 'Failed to generate new route'
                }
            
            # Update navigator with new route
            navigator.route_steps = new_steps
            navigator.current_step_index = 0
            navigator.total_distance = sum(step.distance for step in new_steps)
            navigator.off_route_count = 0
            
            # Get initial state with new route
            gps_location = GPSLocation(
                latitude=current_location['lat'],
                longitude=current_location['lon'],
                accuracy=current_location.get('accuracy', 10.0)
            )
            
            state = navigator.update_location(gps_location)
            
            logger.info(f"🔄 Rerouting completed for session {session_id}")
            
            return {
                'type': 'reroute_success',
                'message': '🔄 **Route Recalculated!**\n\nNew route calculated from your current location.',
                'navigation_state': state.to_dict(),
                'route_overview': navigator.get_route_overview()
            }
            
        except Exception as e:
            logger.error(f"Error rerouting: {e}", exc_info=True)
            return {
                'type': 'error',
                'message': f'Failed to recalculate route: {str(e)}'
            }
    
    def stop_gps_navigation(self, session_id: str) -> Dict[str, Any]:
        """
        Stop GPS navigation
        
        Args:
            session_id: Session identifier
            
        Returns:
            Stop confirmation
        """
        if session_id in self.active_navigators:
            navigator = self.active_navigators[session_id]
            navigator.stop_navigation()
            del self.active_navigators[session_id]
            
            logger.info(f"🛑 Navigation stopped for session {session_id}")
            
            return {
                'type': 'navigation_stopped',
                'message': '🛑 Navigation stopped'
            }
        
        return {
            'type': 'info',
            'message': 'No active navigation session found'
        }
    
    def get_navigation_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current navigation status
        
        Args:
            session_id: Session identifier
            
        Returns:
            Navigation status
        """
        if session_id not in self.active_navigators:
            return {
                'type': 'info',
                'active': False,
                'message': 'No active navigation session'
            }
        
        navigator = self.active_navigators[session_id]
        

        
        return {
            'type': 'navigation_status',
            'active': navigator.is_navigating,
            'arrived': navigator.has_arrived,
            'current_step': navigator.current_step_index,
            'total_steps': len(navigator.route_steps),
            'mode': navigator.mode.value,
            'language': navigator.language,
            'route_overview': navigator.get_route_overview()
        }
    
    def _is_start_navigation_command(self, message_lower: str) -> bool:
        """Check if message is a start navigation command"""
        start_keywords = [
            'start navigation', 'navigate to', 'navigate me',
            'take me to', 'directions to', 'guide me to',
            'turn by turn', 'gps to', 'route to'
        ]
        return any(keyword in message_lower for keyword in start_keywords)
    
    async def _handle_start_navigation(
        self,
        message: str,
        session_id: str,
        user_location: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Start GPS turn-by-turn navigation"""
        
        # Check if already navigating
        if session_id in self.active_navigators:
            return {
                'type': 'navigation_warning',
                'message': "You're already navigating! Say 'stop navigation' first, or I can reroute you to a new destination.",
                'navigation_active': True
            }
        
        # Validate user location
        if not user_location or 'lat' not in user_location or 'lon' not in user_location:
            return {
                'type': 'navigation_error',
                'message': "I need your current location to start navigation. Please enable GPS and try again.",
                'needs_location': True
            }
        
        # Extract destination
        locations = await self._extract_locations(message)
        if not locations or len(locations) == 0:
            return {
                'type': 'navigation_error',
                'message': "Where would you like to go? Please specify a destination like 'Navigate to Galata Tower' or 'Take me to Blue Mosque'."
            }
        
        destination = locations[0]  # First location is destination
        start_location = (user_location['lat'], user_location['lon'])
        
        try:
            # Plan route using intelligent routing
            route = self.route_integration.plan_intelligent_route(
                start=start_location,
                end=destination,
                transport_mode='walking',
                user_context={'gps_navigation': True}
            )
            
            # Get geometry from route
            route_geometry = None
            if hasattr(route, 'geometry') and route.geometry:
                route_geometry = route.geometry
            elif hasattr(route, 'osrm_route') and route.osrm_route and 'geometry' in route.osrm_route:
                # Extract geometry from OSRM response
                osrm_geom = route.osrm_route['geometry']
                if isinstance(osrm_geom, list):
                    route_geometry = osrm_geom
                elif isinstance(osrm_geom, dict) and 'coordinates' in osrm_geom:
                    route_geometry = osrm_geom['coordinates']
            elif hasattr(route, 'path') and route.path:
                route_geometry = route.path
            
            if not route or not route_geometry:
                return {
                    'type': 'navigation_error',
                    'message': "Sorry, I couldn't find a route to that location. Please try a different destination."
                }
            
            # Create GPS navigator
            gps_location = GPSLocation(
                latitude=start_location[0],
                longitude=start_location[1],
                accuracy=10.0,
                timestamp=datetime.now()
            )
            
            navigator = GPSTurnByTurnNavigator(
                route_geometry=route_geometry,
                mode=NavigationMode.WALKING
            )
            
            # Start navigation
            nav_state = navigator.start_navigation(gps_location)
            
            # Store navigator
            self.active_navigators[session_id] = navigator
            self.navigation_sessions[session_id] = {
                'destination': destination,
                'destination_name': self._get_location_name(destination),
                'start_time': datetime.now(),
                'route': route
            }
            
            # Format response
            current_instruction = nav_state.current_instruction
            distance_km = getattr(route, 'distance', 0) / 1000 if hasattr(route, 'distance') else 0
            duration_min = getattr(route, 'duration', 0) / 60 if hasattr(route, 'duration') else 0
            
            # Fallback values if route doesn't have distance/duration
            if distance_km == 0 and route_geometry:
                # Estimate from geometry length
                distance_km = len(route_geometry) * 0.1  # Rough estimate
                duration_min = distance_km * 12  # ~5 km/h walking speed
            
            response_message = f"""🧭 **Navigation Started!**

📍 **Destination:** {self._get_location_name(destination)}
📏 **Total Distance:** {distance_km:.2f} km
⏱️ **Estimated Time:** {int(duration_min)} minutes

**First Instruction:**
➡️ {current_instruction.text}
📍 In {current_instruction.distance:.0f} meters

Say "what's next" for updates or "stop navigation" to end."""
            
            return {
                'type': 'navigation_started',
                'message': response_message,
                'navigation_active': True,
                'navigation_data': {
                    'session_id': session_id,
                    'destination': destination,
                    'destination_name': self._get_location_name(destination),
                    'current_instruction': {
                        'text': current_instruction.text,
                        'distance': current_instruction.distance,
                        'type': current_instruction.instruction_type.value
                    },
                    'progress': {
                        'distance_remaining': nav_state.distance_remaining,
                        'time_remaining': nav_state.time_remaining,
                        'percent_complete': 0
                    },
                    'route_geometry': route_geometry,
                    'map_data': {
                        'type': 'navigation',
                        'start': start_location,
                        'end': destination,
                        'geometry': route_geometry,
                        'current_position': start_location
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error starting navigation: {e}")
            return {
                'type': 'navigation_error',
                'message': f"Sorry, I couldn't start navigation: {str(e)}"
            }
    
    def _handle_stop_navigation(self, session_id: str) -> Dict[str, Any]:
        """Stop active navigation"""
        if session_id not in self.active_navigators:
            return {
                'type': 'navigation_info',
                'message': "You're not currently navigating. Say 'navigate to [location]' to start!"
            }
        
        # Get session info
        session_info = self.navigation_sessions.get(session_id, {})
        destination_name = session_info.get('destination_name', 'your destination')
        
        # Stop navigation
        navigator = self.active_navigators[session_id]
        nav_state = navigator.stop_navigation()
        
        # Calculate statistics
        if session_info.get('start_time'):
            duration = (datetime.now() - session_info['start_time']).total_seconds() / 60
            duration_text = f"{int(duration)} minutes"
        else:
            duration_text = "unknown time"
        
        # Clean up
        del self.active_navigators[session_id]
        del self.navigation_sessions[session_id]
        
        return {
            'type': 'navigation_stopped',
            'message': f"✅ **Navigation Ended**\n\nYou were navigating to **{destination_name}** for {duration_text}.\n\nSafe travels! 🚶‍♂️",
            'navigation_active': False
        }
    
    def _handle_location_query(
        self,
        session_id: str,
        user_location: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Handle "where am I?" queries"""
        if not user_location:
            return {
                'type': 'location_error',
                'message': "I can't determine your location right now. Please enable GPS.",
                'needs_location': True
            }
        
        lat, lon = user_location['lat'], user_location['lon']
        
        # Check if navigating
        if session_id in self.active_navigators:
            navigator = self.active_navigators[session_id]
            session_info = self.navigation_sessions[session_id]
            nav_state = navigator.get_navigation_state()
            
            destination_name = session_info.get('destination_name', 'your destination')
            distance_km = nav_state.distance_remaining / 1000
            time_min = int(nav_state.time_remaining / 60)
            
            return {
                'type': 'location_with_navigation',
                'message': f"""📍 **Your Location:** {lat:.6f}, {lon:.6f}

🧭 **Navigating to:** {destination_name}
📏 **Distance Remaining:** {distance_km:.2f} km
⏱️ **Time Remaining:** {time_min} minutes

**Current Instruction:**
➡️ {nav_state.current_instruction.text}
""",
                'location': {'lat': lat, 'lon': lon},
                'navigation_active': True
            }
        else:
            # Find nearby landmarks
            nearby = self._find_nearby_locations(lat, lon, radius_km=0.5)
            
            if nearby:
                nearby_text = "\n".join([f"• {name} ({dist:.0f}m away)" for name, dist in nearby[:3]])
                message = f"""📍 **Your Location:** {lat:.6f}, {lon:.6f}

**Nearby Landmarks:**
{nearby_text}

Say 'navigate to [location]' to start turn-by-turn directions!"""
            else:
                message = f"""📍 **Your Location:** {lat:.6f}, {lon:.6f}

Say 'navigate to [location]' to start turn-by-turn directions!"""
            
            return {
                'type': 'location_info',
                'message': message,
                'location': {'lat': lat, 'lon': lon},
                'navigation_active': False
            }
    
    def _handle_next_instruction(
        self,
        session_id: str,
        user_location: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Handle "what's next?" queries"""
        if session_id not in self.active_navigators:
            return {
                'type': 'navigation_info',
                'message': "You're not currently navigating. Say 'navigate to [location]' to start!"
            }
        
        if not user_location:
            return {
                'type': 'navigation_error',
                'message': "I need your current location to provide navigation updates. Please enable GPS."
            }
        
        # Update navigation with current location
        navigator = self.active_navigators[session_id]
        gps_location = GPSLocation(
            latitude=user_location['lat'],
            longitude=user_location['lon'],
            accuracy=10.0,
            timestamp=datetime.now()
        )
        
        nav_state = navigator.update_location(gps_location)
        
        # Check if arrived
        if nav_state.has_arrived:
            session_info = self.navigation_sessions[session_id]
            destination_name = session_info.get('destination_name', 'your destination')
            
            # Clean up
            del self.active_navigators[session_id]
            del self.navigation_sessions[session_id]
            
            return {
                'type': 'navigation_complete',
                'message': f"🎉 **You've Arrived!**\n\nWelcome to **{destination_name}**!\n\nEnjoy your visit! 🚶‍♂️",
                'navigation_active': False,
                'arrived': True
            }
        
        # Format instruction
        instruction = nav_state.current_instruction
        distance_km = nav_state.distance_remaining / 1000
        time_min = int(nav_state.time_remaining / 60)
        
        # Check if off-route
        if nav_state.off_route:
            message = f"""⚠️ **Off Route!**

You're {nav_state.off_route_distance:.0f}m from the route.

**Suggested Action:**
{instruction.text}

Say 'reroute' to calculate a new route."""
        else:
            message = f"""➡️ **Next Instruction:**
{instruction.text}

📍 **In:** {instruction.distance:.0f} meters
📏 **Remaining:** {distance_km:.2f} km
⏱️ **ETA:** {time_min} minutes"""
        
        return {
            'type': 'navigation_update',
            'message': message,
            'navigation_active': True,
            'navigation_data': {
                'current_instruction': {
                    'text': instruction.text,
                    'distance': instruction.distance,
                    'type': instruction.instruction_type.value
                },
                'progress': {
                    'distance_remaining': nav_state.distance_remaining,
                    'time_remaining': nav_state.time_remaining,
                    'off_route': nav_state.off_route,
                    'off_route_distance': nav_state.off_route_distance
                },
                'current_position': [user_location['lat'], user_location['lon']]
            }
        }
    
    def _handle_repeat_instruction(self, session_id: str) -> Dict[str, Any]:
        """Handle "repeat instruction" queries"""
        if session_id not in self.active_navigators:
            return {
                'type': 'navigation_info',
                'message': "You're not currently navigating."
            }
        
        navigator = self.active_navigators[session_id]
        nav_state = navigator.get_navigation_state()
        instruction = nav_state.current_instruction
        
        return {
            'type': 'navigation_repeat',
            'message': f"🔁 **Repeating:**\n\n➡️ {instruction.text}\n📍 In {instruction.distance:.0f} meters",
            'navigation_active': True
        }
    
    def _handle_navigation_status(self, session_id: str) -> Dict[str, Any]:
        """Handle navigation status query"""
        if session_id not in self.active_navigators:
            return {
                'type': 'navigation_info',
                'message': "❌ **No Active Navigation**\n\nSay 'navigate to [location]' to start turn-by-turn directions!"
            }
        
        navigator = self.active_navigators[session_id]
        session_info = self.navigation_sessions[session_id]
        nav_state = navigator.get_navigation_state()
        
        destination_name = session_info.get('destination_name', 'your destination')
        distance_km = nav_state.distance_remaining / 1000
        time_min = int(nav_state.time_remaining / 60)
        
        # Calculate progress
        route = session_info.get('route')
        if route:
            progress_pct = (1 - nav_state.distance_remaining / route.distance) * 100
        else:
            progress_pct = 0
        
        return {
            'type': 'navigation_status',
            'message': f"""✅ **Navigation Active**



🎯 **Destination:** {destination_name}
📏 **Remaining:** {distance_km:.2f} km
⏱️ **ETA:** {time_min} minutes
📊 **Progress:** {progress_pct:.0f}%

**Current Instruction:**
➡️ {nav_state.current_instruction.text}

Say 'what's next' for updates or 'stop navigation' to end.""",
            'navigation_active': True
        }
    
    async def _handle_reroute(
        self,
        session_id: str,
        user_location: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Handle reroute request"""
        if session_id not in self.active_navigators:
            return {
                'type': 'navigation_info',
                'message': "You're not currently navigating."
            }
        
        if not user_location:
            return {
                'type': 'navigation_error',
                'message': "I need your current location to reroute. Please enable GPS."
            }
        
        session_info = self.navigation_sessions[session_id]
        destination = session_info['destination']
        
        # Stop current navigation
        del self.active_navigators[session_id]
        del self.navigation_sessions[session_id]
        
        # Start new navigation from current location
        fake_message = f"navigate to {self._get_location_name(destination)}"
        return await self._handle_start_navigation(fake_message, session_id, user_location)
    
    def _handle_auto_update(
        self,
        session_id: str,
        user_location: Dict[str, float]
    ) -> Dict[str, Any]:
        """Auto-update navigation when location changes (silent)"""
        navigator = self.active_navigators[session_id]
        
        gps_location = GPSLocation(
            latitude=user_location['lat'],
            longitude=user_location['lon'],
            accuracy=10.0,
            timestamp=datetime.now()
        )
        
        nav_state = navigator.update_location(gps_location)
        
        # Only return data, don't send automatic messages
        return {
            'type': 'navigation_auto_update',
            'silent': True,
            'navigation_active': True,
            'navigation_data': {
                'progress': {
                    'distance_remaining': nav_state.distance_remaining,
                    'time_remaining': nav_state.time_remaining,
                    'off_route': nav_state.off_route,
                    'has_arrived': nav_state.has_arrived
                },
                'current_position': [user_location['lat'], user_location['lon']]
            }
        }
    
    def _get_location_name(self, coords: Tuple[float, float]) -> str:
        """Get friendly name for coordinates"""
        # Reverse lookup in KNOWN_LOCATIONS
        for name, loc_coords in self.KNOWN_LOCATIONS.items():
            if abs(loc_coords[0] - coords[0]) < 0.001 and abs(loc_coords[1] - coords[1]) < 0.001:
                return name.title()
        
        # Return formatted coordinates
        return f"{coords[0]:.4f}, {coords[1]:.4f}"
    
    def _get_user_gps_location(self, user_context: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
        """
        Extract user's GPS location from context
        
        Args:
            user_context: User context dict that may contain GPS location
            
        Returns:
            Tuple of (lat, lon) if GPS available, None otherwise
        """
        logger.info(f"🔍 Checking user_context for GPS: {user_context}")
        
        if not user_context:
            logger.info("❌ No user_context provided")
            return None
        
        # Check for GPS location in various formats
        # Format 1: {'gps': {'lat': 41.0, 'lon': 28.9}}
        if 'gps' in user_context and isinstance(user_context['gps'], dict):
            gps = user_context['gps']
            logger.info(f"✅ Found 'gps' in context: {gps}")
            if 'lat' in gps and 'lon' in gps:
                result = (float(gps['lat']), float(gps['lon']))
                logger.info(f"✅ Returning GPS location from 'gps' field: {result}")
                return result
        
        # Format 2: {'location': {'lat': 41.0, 'lon': 28.9}}
        if 'location' in user_context and isinstance(user_context['location'], dict):
            location = user_context['location']
            if 'lat' in location and 'lon' in location:
                return (float(location['lat']), float(location['lon']))
        
        # Format 3: {'current_location': [41.0, 28.9]}
        if 'current_location' in user_context:
            loc = user_context['current_location']
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                return (float(loc[0]), float(loc[1]))
            elif isinstance(loc, dict) and 'lat' in loc and 'lon' in loc:
                return (float(loc['lat']), float(loc['lon']))
        
        # Format 4: Direct lat/lon in context
        if 'lat' in user_context and 'lon' in user_context:
            return (float(user_context['lat']), float(user_context['lon']))
        
        # Format 5: {'latitude': 41.0, 'longitude': 28.9}
        if 'latitude' in user_context and 'longitude' in user_context:
            return (float(user_context['latitude']), float(user_context['longitude']))
        
        logger.debug("No GPS location found in user context")
        return None
    
    def _find_nearby_locations(
        self,
        lat: float,
        lon: float,
        radius_km: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Find nearby known locations"""
        import math
        
        nearby = []
        
        for name, (loc_lat, loc_lon) in self.KNOWN_LOCATIONS.items():
            # Calculate distance using Haversine formula
            dlat = math.radians(loc_lat - lat)
            dlon = math.radians(loc_lon - lon)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(loc_lat)) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance_km = 6371 * c
            
            if distance_km <= radius_km:
                nearby.append((name.title(), distance_km * 1000))  # Convert to meters
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        return nearby
        

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


# ========== GPS Navigation Functions ==========

def start_navigation(
    session_id: str,
    route_data: Dict[str, Any],
    current_location: Dict[str, float],
    language: str = "en"
) -> Dict[str, Any]:
    """
    Start GPS turn-by-turn navigation
    
    Args:
        session_id: Unique session identifier (e.g., user_id)
        route_data: Route data from planning (must include 'osrm_route' or 'steps')
        current_location: Current GPS location dict with 'lat', 'lon', optional 'accuracy'
        language: Language for instructions ('en' or 'tr')
        
    Returns:
        Navigation start response with initial state
    """
    handler = get_chat_route_handler()
    return handler.start_gps_navigation(session_id, route_data, current_location, language)


def update_navigation(
    session_id: str,
    current_location: Dict[str, float]
) -> Dict[str, Any]:
    """
    Update GPS navigation with new location
    
    Args:
        session_id: Session identifier
        current_location: Current GPS location dict with 'lat', 'lon', optional 'accuracy', 'speed', 'bearing'
        
    Returns:
        Updated navigation state
    """
    handler = get_chat_route_handler()
    return handler.update_gps_navigation(session_id, current_location)


def request_reroute(
    session_id: str,
    current_location: Dict[str, float]
) -> Dict[str, Any]:
    """
    Request rerouting from current location to original destination
    
    Args:
        session_id: Session identifier
        current_location: Current GPS location
        
    Returns:
        Rerouting result with new navigation state
    """
    handler = get_chat_route_handler()
    return handler.request_reroute(session_id, current_location)


def stop_navigation(session_id: str) -> Dict[str, Any]:
    """
    Stop GPS navigation
    
    Args:
        session_id: Session identifier
        
    Returns:
        Stop confirmation
    """
    handler = get_chat_route_handler()
    return handler.stop_gps_navigation(session_id)


def get_navigation_status(session_id: str) -> Dict[str, Any]:
    """
    Get current navigation status
    
    Args:
        session_id: Session identifier
        
    Returns:
        Navigation status information
    """
    handler = get_chat_route_handler()
    return handler.get_navigation_status(session_id)


# Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing AI Chat Route Integration...\n")
    
    # Initialize handler
    handler = get_chat_route_handler()
    
    # Test cases for route planning
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
                    print(f"Waypoints: {len(route_data.get('waypoints', []))}")
                else:
                    print(f"\nMulti-segment route ready for map display")
                    print(f"Segments: {route_data.get('segments', 0)}")
        else:
            print("❌ Not recognized as a route request")
    
    print("\n" + "="*60)
    print("✅ Route planning tests completed!")
    
    # Test GPS navigation if available
    if GPS_NAVIGATION_AVAILABLE:
        print("\n" + "="*60)
        print("🧭 Testing GPS Turn-by-Turn Navigation...")
        print("="*60)
        
        # Mock route data for testing
        test_route = {
            'mode': 'walking',
            'steps': [
                RouteStep(
                    instruction_type=InstructionType.DEPART,
                    distance=100,
                    duration=72,
                    start_location=(41.0054, 28.9768),
                    end_location=(41.0064, 28.9768),
                    street_name="Sultanahmet Square"
                               ),
                RouteStep(
                    instruction_type=InstructionType.TURN_RIGHT,
                    distance=200,
                    duration=144,
                    start_location=(41.0064, 28.9768),
                    end_location=(41.0064, 28.9788),
                    street_name="Divanyolu Street"
                ),
                RouteStep(
                    instruction_type=InstructionType.ARRIVE,
                    distance=50,
                    duration=36,
                    start_location=(41.0064, 28.9788),
                    end_location=(41.0086, 28.9802),
                    street_name="Hagia Sophia"
                )
            ]
        }
        
        # Start navigation
        session_id = "test_session_001"
        start_location = {'lat': 41.0054, 'lon': 28.9768, 'accuracy': 10}
        
        print(f"\n1️⃣ Starting navigation from {start_location['lat']:.4f}, {start_location['lon']:.4f}...")
        start_result = start_navigation(session_id, test_route, start_location, language='en')
        
        if start_result['type'] == 'navigation_started':
            print("✅ Navigation started successfully!")
            nav_state = start_result['navigation_state']
            print(f"📢 Initial instruction: {nav_state['instruction']['text']}")
            print(f"📊 Progress: {nav_state['progress']['percent_complete']:.1f}%")
            print(f"📏 Remaining: {nav_state['progress']['distance_remaining']:.0f}m")
            
            # Simulate location updates
            test_locations = [
                {'lat': 41.0058, 'lon': 28.9768, 'accuracy': 8},
                {'lat': 41.0062, 'lon': 28.9768, 'accuracy': 7},
                {'lat': 41.0064, 'lon': 28.9778, 'accuracy': 9},
            ]
            
            for j, location in enumerate(test_locations, 2):
                print(f"\n{j}️⃣ Updating location to {location['lat']:.4f}, {location['lon']:.4f}...")
                update_result = update_navigation(session_id, location)
                
                if update_result['type'] == 'navigation_update':
                    nav_state = update_result['navigation_state']
                    print(f"📢 Current instruction: {nav_state['instruction']['text']}")
                    print(f"📊 Progress: {nav_state['progress']['percent_complete']:.1f}%")
                    
                    if nav_state['status']['off_route']:
                        print("⚠️ Off route detected!")
                
                elif update_result['type'] == 'navigation_completed':
                    print("🎉 Arrived at destination!")
                    break
            
            # Stop navigation
            print(f"\n4️⃣ Stopping navigation...")
            stop_result = stop_navigation(session_id)
            print(f"✅ {stop_result['message']}")
        
        else:
            print(f"❌ Failed to start navigation: {start_result.get('message')}")
        
        print("\n" + "="*60)
        print("✅ GPS navigation tests completed!")
    else:
        print("\n⚠️ GPS navigation not available for testing")
    
    print("\n" + "="*60)
    print("🎉 All tests completed!")
