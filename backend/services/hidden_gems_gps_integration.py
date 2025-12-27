"""
Hidden Gems GPS Integration
============================

Seamlessly integrates hidden gems discovery with GPS turn-by-turn navigation.
When users ask for hidden gems, they can immediately navigate to them with
full map visualization and turn-by-turn directions.

Features:
- Hidden gem recommendations with GPS coordinates
- One-click navigation from chat to hidden gem
- Map visualization of all recommended gems
- Route planning to multiple hidden gems
- Distance and time estimates from user location
- Context-aware recommendations based on current location

Usage:
    User: "Show me hidden cafes nearby"
    System: Shows hidden cafes with distances + "Navigate to..." buttons
    
    User: "Navigate to [hidden gem name]"
    System: Activates GPS navigation with turn-by-turn directions
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import GPS navigation components
try:
    from gps_turn_by_turn_navigation import (
        GPSTurnByTurnNavigator,
        NavigationMode,
        GPSLocation,
        RouteStep,
        convert_osrm_to_steps
    )
    GPS_NAV_AVAILABLE = True
except ImportError:
    GPS_NAV_AVAILABLE = False
    # GPS navigation module is optional

# Import route integration
try:
    from services.ai_chat_route_integration import (
        AIChatRouteHandler,
        create_chat_route_handler
    )
    ROUTE_HANDLER_AVAILABLE = True
except ImportError:
    try:
        from backend.services.ai_chat_route_integration import (
            AIChatRouteHandler,
            create_chat_route_handler
        )
        ROUTE_HANDLER_AVAILABLE = True
    except ImportError:
        ROUTE_HANDLER_AVAILABLE = False
        # Route handler module is optional

# Import hidden gems database
try:
    from data.hidden_gems_database import (
        HIDDEN_GEMS_DATABASE,
        get_gems_by_neighborhood,
        get_gems_by_type,
        get_all_hidden_gems
    )
    GEMS_DB_AVAILABLE = True
except ImportError:
    try:
        from backend.data.hidden_gems_database import (
            HIDDEN_GEMS_DATABASE,
            get_gems_by_neighborhood,
            get_gems_by_type,
            get_all_hidden_gems
        )
        GEMS_DB_AVAILABLE = True
    except ImportError as e:
        GEMS_DB_AVAILABLE = False
        logger.warning(f"⚠️ Hidden gems database not available: {e}")

# Import multilingual intent keywords
try:
    from services.multilingual_intent_keywords import (
        detect_hidden_gems_intent,
        extract_neighborhood,
        HIDDEN_GEMS_KEYWORDS,
        NEIGHBORHOOD_KEYWORDS
    )
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    try:
        from backend.services.multilingual_intent_keywords import (
            detect_hidden_gems_intent,
            extract_neighborhood,
            HIDDEN_GEMS_KEYWORDS,
            NEIGHBORHOOD_KEYWORDS
        )
        MULTILINGUAL_AVAILABLE = True
    except ImportError as e:
        MULTILINGUAL_AVAILABLE = False
        logger.warning(f"⚠️ Multilingual intent keywords not available: {e}")

# Istanbul neighborhoods with approximate GPS coordinates
NEIGHBORHOOD_COORDS = {
    'sultanahmet': (41.0054, 28.9768),
    'beyoğlu': (41.0370, 28.9850),
    'beşiktaş': (41.0420, 28.9905),
    'kadıköy': (40.9904, 29.0250),
    'üsküdar': (41.0224, 29.0154),
    'sarıyer': (41.1688, 29.0535),
    'ortaköy': (41.0551, 29.0294),
    'balat': (41.0289, 28.9489),
    'karaköy': (41.0236, 28.9765),
    'eminönü': (41.0174, 28.9706),
    'taksim': (41.0370, 28.9850),
    'galata': (41.0256, 28.9742),
    'şişli': (41.0602, 28.9872),
    'nişantaşı': (41.0466, 28.9920),
    'bebek': (41.0811, 29.0430),
    'arnavutköy': (41.0680, 29.0370),
}


@dataclass
class HiddenGemLocation:
    """Hidden gem with GPS coordinates for navigation"""
    name: str
    type: str
    description: str
    neighborhood: str
    latitude: float
    longitude: float
    local_tip: Optional[str] = None
    best_time: Optional[str] = None
    cost: Optional[str] = None
    how_to_find: Optional[str] = None
    distance_km: Optional[float] = None
    walking_time_min: Optional[int] = None
    directions_preview: Optional[str] = None
    distance_km: Optional[float] = None
    walking_time_min: Optional[int] = None
    directions_preview: Optional[str] = None
    local_tip: Optional[str] = None
    best_time: Optional[str] = None
    cost: Optional[str] = None
    how_to_find: Optional[str] = None


class HiddenGemsGPSIntegration:
    """
    Hidden Gems GPS Integration
    
    Bridges hidden gem discovery with GPS navigation.
    Users can:
    1. Ask for hidden gems (get recommendations with distances)
    2. Click "Navigate" to start GPS turn-by-turn
    3. Get route visualization on map
    4. Plan routes visiting multiple hidden gems
    """
    
    def __init__(self, db_session=None):
        """Initialize integration with GPS and routing services"""
        self.db = db_session
        
        # GPS navigator is created per-navigation-session, not at init time
        # GPSTurnByTurnNavigator requires route_steps, mode, language
        self.gps_nav_available = GPS_NAV_AVAILABLE
        
        # Initialize route handler
        if ROUTE_HANDLER_AVAILABLE:
            try:
                self.route_handler = create_chat_route_handler()
                logger.info("✅ Route handler initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize route handler: {e}")
                self.route_handler = None
        else:
            self.route_handler = None
        
        logger.info(f"🗺️ Hidden Gems GPS Integration initialized (GPS available: {self.gps_nav_available}, Routes: {self.route_handler is not None})")
    
    # Sub-neighborhood to parent neighborhood mapping
    SUB_NEIGHBORHOOD_MAP = {
        'balat': ['beyoğlu', 'fatih'],
        'fener': ['beyoğlu', 'fatih'],
        'cihangir': ['beyoğlu'],
        'galata': ['beyoğlu'],
        'karaköy': ['beyoğlu'],
        'taksim': ['beyoğlu'],
        'ortaköy': ['beşiktaş'],
        'bebek': ['beşiktaş'],
        'arnavutköy': ['beşiktaş'],
        'moda': ['kadıköy'],
        'yeldeğirmeni': ['kadıköy'],
        'sultanahmet': ['fatih'],
        'eminönü': ['fatih'],
        'kuzguncuk': ['üsküdar'],
        'kilyos': ['sarıyer'],
        'tarabya': ['sarıyer'],
    }
    
    def _resolve_neighborhood(self, neighborhood: str) -> List[str]:
        """Resolve sub-neighborhood to parent neighborhoods"""
        neighborhood = neighborhood.lower()
        if neighborhood in self.SUB_NEIGHBORHOOD_MAP:
            return self.SUB_NEIGHBORHOOD_MAP[neighborhood]
        return [neighborhood]
    
    def get_hidden_gems_with_navigation(
        self,
        user_location: Optional[Dict[str, float]] = None,
        gem_type: Optional[str] = None,
        neighborhood: Optional[str] = None,
        max_distance_km: Optional[float] = 5.0,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get hidden gems with GPS coordinates and navigation options
        
        Args:
            user_location: User's GPS location {lat, lon}
            gem_type: Type of gem (cafe, park, historical, etc.)
            neighborhood: Istanbul neighborhood
            max_distance_km: Maximum distance from user
            limit: Maximum number of gems to return
        
        Returns:
            {
                'gems': List of HiddenGemLocation objects,
                'map_data': Map visualization data,
                'navigation_ready': bool,
                'user_location': user location if provided
            }
        """
        if not GEMS_DB_AVAILABLE:
            return {
                'gems': [],
                'error': 'Hidden gems database not available',
                'navigation_ready': False
            }
        
        # Get gems from database
        gems = []
        if neighborhood:
            # Resolve sub-neighborhood to parent neighborhoods
            parent_neighborhoods = self._resolve_neighborhood(neighborhood)
            for parent in parent_neighborhoods:
                gems.extend(get_gems_by_neighborhood(parent))
            
            # Also filter by name/description containing the sub-neighborhood
            search_term = neighborhood.lower()
            if not gems:
                # Try searching all gems for mentions of the neighborhood
                all_gems = get_all_hidden_gems()
                gems = [g for g in all_gems if search_term in g.get('name', '').lower() 
                        or search_term in g.get('description', '').lower()]
        elif gem_type:
            gems = get_gems_by_type(gem_type.lower())
        else:
            gems = get_all_hidden_gems()
        
        # If still no gems found, return all gems as fallback
        if not gems:
            gems = get_all_hidden_gems()
        
        # Enrich gems with GPS coordinates
        enriched_gems = []
        map_markers = []
        
        for gem in gems[:limit * 2]:  # Get more than needed, filter by distance
            # Get neighborhood coordinates - use stored neighborhood or default
            gem_neighborhood = gem.get('neighborhood', '').lower()
            if gem_neighborhood not in NEIGHBORHOOD_COORDS:
                # Try to find a matching parent neighborhood
                found = False
                for parent in self._resolve_neighborhood(gem_neighborhood):
                    if parent in NEIGHBORHOOD_COORDS:
                        gem_neighborhood = parent
                        found = True
                        break
                if not found:
                    # Use Istanbul center as fallback
                    lat, lon = 41.0082, 28.9784
                else:
                    lat, lon = NEIGHBORHOOD_COORDS[gem_neighborhood]
            else:
                lat, lon = NEIGHBORHOOD_COORDS[gem_neighborhood]
            
            # Create location object
            gem_location = HiddenGemLocation(
                name=gem['name'],
                type=gem.get('type', 'unknown'),
                description=gem.get('description', ''),
                neighborhood=gem_neighborhood,
                latitude=lat,
                longitude=lon,
                local_tip=gem.get('local_tip'),
                best_time=gem.get('best_time'),
                cost=gem.get('cost'),
                how_to_find=gem.get('how_to_find')
            )
            
            # Calculate distance if user location provided
            if user_location:
                distance_km = self._calculate_distance(
                    user_location['lat'], user_location['lon'],
                    lat, lon
                )
                
                # Filter by max distance
                if max_distance_km and distance_km > max_distance_km:
                    continue
                
                gem_location.distance_km = distance_km
                gem_location.walking_time_min = int(distance_km * 15)  # ~4km/h walking speed
                gem_location.directions_preview = self._get_direction_preview(
                    user_location['lat'], user_location['lon'],
                    lat, lon
                )
            
            enriched_gems.append(gem_location)
            
            # Add map marker
            map_markers.append({
                'type': 'hidden_gem',
                'name': gem_location.name,
                'lat': lat,
                'lon': lon,
                'description': gem_location.description,
                'gem_type': gem_location.type,
                'distance_km': gem_location.distance_km,
                'icon': self._get_gem_icon(gem_location.type)
            })
            
            if len(enriched_gems) >= limit:
                break
        
        # Sort by distance if user location provided
        if user_location:
            enriched_gems.sort(key=lambda g: g.distance_km or 999)
        
        # Create map data
        map_data = {
            'markers': map_markers,
            'center': user_location or {'lat': 41.0082, 'lon': 28.9784},  # Default to Istanbul center
            'zoom': 13
        }
        
        # Add user location marker if provided
        if user_location:
            map_data['markers'].insert(0, {
                'type': 'user',
                'name': 'Your Location',
                'lat': user_location['lat'],
                'lon': user_location['lon'],
                'icon': 'user'
            })
        
        return {
            'gems': [self._gem_to_dict(g) for g in enriched_gems],
            'map_data': map_data,
            'navigation_ready': self.gps_nav_available,
            'user_location': user_location,
            'count': len(enriched_gems)
        }
    
    def navigate_to_hidden_gem(
        self,
        gem_name: str,
        user_location: Dict[str, float],
        session_id: str = 'default'
    ) -> Dict[str, Any]:
        """
        Start GPS navigation to a hidden gem
        
        Args:
            gem_name: Name of the hidden gem
            user_location: User's current GPS location
            session_id: Navigation session ID
        
        Returns:
            Navigation data with turn-by-turn instructions
        """
        if not user_location:
            return {
                'error': 'Location required',
                'message': 'Please enable GPS to navigate to hidden gems.'
            }
        
        # Find gem by name
        gem = self._find_gem_by_name(gem_name)
        if not gem:
            return {
                'error': 'Gem not found',
                'message': f'Could not find hidden gem: {gem_name}'
            }
        
        # Get gem coordinates
        gem_neighborhood = gem.get('neighborhood', '').lower()
        if gem_neighborhood not in NEIGHBORHOOD_COORDS:
            return {
                'error': 'Location unavailable',
                'message': f'GPS coordinates not available for {gem_name}'
            }
        
        dest_lat, dest_lon = NEIGHBORHOOD_COORDS[gem_neighborhood]
        
        # Calculate distance and direction
        distance_km = self._calculate_distance(
            user_location['lat'], user_location['lon'],
            dest_lat, dest_lon
        )
        direction = self._get_direction_preview(
            user_location['lat'], user_location['lon'],
            dest_lat, dest_lon
        )
        walking_time_min = int(distance_km * 15)  # ~4km/h walking
        
        # Create simple navigation data
        nav_data = {
            'destination': {
                'name': gem['name'],
                'lat': dest_lat,
                'lon': dest_lon,
                'neighborhood': gem_neighborhood
            },
            'distance_km': round(distance_km, 2),
            'walking_time_min': walking_time_min,
            'direction': direction,
            'how_to_find': gem.get('how_to_find', f'Head {direction} towards {gem_neighborhood}')
        }
        
        # Format response
        return {
            'success': True,
            'message': f"🗺️ {gem['name']} is {distance_km:.1f}km {direction} from you (~{walking_time_min} min walk). {gem.get('how_to_find', '')}",
            'gem': gem,
            'navigation_data': nav_data,
            'navigation_active': True,
            'map_data': self._create_navigation_map_data(
                nav_data,
                gem['name'],
                user_location,
                dest_lat,
                dest_lon
            )
        }
    
    def plan_hidden_gems_tour(
        self,
        user_location: Dict[str, float],
        gem_names: List[str],
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Plan a tour visiting multiple hidden gems
        
        Args:
            user_location: Starting location
            gem_names: List of gem names to visit
            optimize: Whether to optimize visit order
        
        Returns:
            Multi-stop route with map visualization
        """
        if not self.route_handler:
            return {
                'error': 'Route planning not available',
                'message': 'Multi-stop routing is currently unavailable.'
            }
        
        # Find all gems
        gems = []
        for name in gem_names:
            gem = self._find_gem_by_name(name)
            if gem:
                gems.append(gem)
        
        if not gems:
            return {
                'error': 'No gems found',
                'message': 'Could not find any of the specified hidden gems.'
            }
        
        # Create waypoints
        waypoints = []
        for gem in gems:
            neighborhood = gem.get('neighborhood', '').lower()
            if neighborhood in NEIGHBORHOOD_COORDS:
                lat, lon = NEIGHBORHOOD_COORDS[neighborhood]
                waypoints.append({
                    'name': gem['name'],
                    'lat': lat,
                    'lon': lon,
                    'type': 'hidden_gem'
                })
        
        # Plan route
        try:
            route_result = self.route_handler.plan_multi_stop_route(
                start_location=user_location,
                waypoints=waypoints,
                optimize=optimize
            )
            
            return {
                'success': True,
                'message': f"🗺️ Planned tour of {len(gems)} hidden gems!",
                'gems': gems,
                'route': route_result,
                'map_data': route_result.get('map_data')
            }
            
        except Exception as e:
            logger.error(f"Route planning error: {e}")
            return {
                'error': 'Route planning error',
                'message': f'Failed to plan tour: {str(e)}'
            }
    
    def handle_hidden_gem_chat_request(
        self,
        message: str,
        user_location: Optional[Dict[str, float]] = None,
        session_id: str = 'default'
    ) -> Optional[Dict[str, Any]]:
        """
        Handle hidden gem requests in chat - MULTILINGUAL
        
        Detects requests in English, Turkish, Russian, German, Arabic like:
        - "Show me hidden cafes" (EN)
        - "Gizli kafeler göster" (TR) 
        - "Покажи скрытые кафе" (RU)
        - "Zeig mir versteckte Cafés" (DE)
        - "أرني المقاهي المخفية" (AR)
        
        Also handles:
        - "Navigate to [gem name]"
        - "Hidden gems near me"
        - "Plan route to visit [gem1], [gem2], [gem3]"
        
        Returns None if not a hidden gem request
        """
        message_lower = message.lower()
        
        # Check for navigation to specific gem
        if 'navigate to' in message_lower or 'take me to' in message_lower:
            # Extract gem name
            gem_name = self._extract_gem_name(message)
            if gem_name:
                return self.navigate_to_hidden_gem(gem_name, user_location, session_id)
        
        # Check for tour planning
        if 'tour' in message_lower or 'visit' in message_lower and ',' in message:
            gem_names = self._extract_multiple_gem_names(message)
            if gem_names:
                return self.plan_hidden_gems_tour(user_location, gem_names)
        
        # Check for hidden gem discovery - MULTILINGUAL
        is_hidden_gems_request = False
        
        if MULTILINGUAL_AVAILABLE:
            # Use multilingual intent detection
            is_hidden_gems_request = detect_hidden_gems_intent(message)
            if is_hidden_gems_request:
                logger.info(f"🌍 Multilingual hidden gems intent detected for: {message[:50]}...")
        else:
            # Fallback to English-only detection
            is_hidden_gems_request = any(keyword in message_lower for keyword in [
                'hidden', 'secret', 'local spot', 'off the beaten'
            ])
        
        if is_hidden_gems_request:
            # Extract type and neighborhood
            gem_type = self._extract_gem_type(message)
            neighborhood = self._extract_neighborhood(message)
            
            result = self.get_hidden_gems_with_navigation(
                user_location=user_location,
                gem_type=gem_type,
                neighborhood=neighborhood
            )
            
            # Format response message for gems
            if result.get('gems') and not result.get('error'):
                gems = result['gems']
                gem_count = len(gems)
                
                # Build response message
                msg_parts = [f"🔮 Found {gem_count} hidden gem{'s' if gem_count > 1 else ''}"]
                if neighborhood:
                    msg_parts[0] += f" in {neighborhood}"
                msg_parts[0] += ":\n"
                
                for gem in gems[:5]:  # Show top 5
                    distance_info = ""
                    if gem.get('distance_km'):
                        distance_info = f" ({gem['distance_km']:.1f}km away)"
                    msg_parts.append(f"• **{gem['name']}** - {gem['description'][:100]}...{distance_info}\n")
                
                result['message'] = "\n".join(msg_parts)
                result['suggestions'] = [
                    f"Navigate to {gems[0]['name']}" if gems else "Show more hidden gems",
                    "Show hidden restaurants",
                    "Show hidden cafes",
                    "Show hidden spots in Kadıköy"
                ]
            
            return result
        
        return None
    
    # Helper methods
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _get_direction_preview(self, lat1: float, lon1: float, lat2: float, lon2: float) -> str:
        """Get compass direction (N, NE, E, etc.)"""
        from math import atan2, degrees
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        angle = degrees(atan2(dlon, dlat))
        
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = round(angle / 45) % 8
        
        return directions[index]
    
    def _get_gem_icon(self, gem_type: str) -> str:
        """Get map icon for gem type"""
        icons = {
            'cafe': 'coffee',
            'restaurant': 'restaurant',
            'park': 'tree',
            'historical': 'landmark',
            'view': 'mountain',
            'shopping': 'shopping-bag',
            'art': 'palette',
            'nightlife': 'moon',
            'nature': 'leaf'
        }
        return icons.get(gem_type, 'star')
    
    def _gem_to_dict(self, gem: HiddenGemLocation) -> Dict[str, Any]:
        """Convert HiddenGemLocation to dict"""
        return {
            'name': gem.name,
            'type': gem.type,
            'description': gem.description,
            'neighborhood': gem.neighborhood,
            'latitude': gem.latitude,
            'longitude': gem.longitude,
            'distance_km': gem.distance_km,
            'walking_time_min': gem.walking_time_min,
            'directions_preview': gem.directions_preview,
            'local_tip': gem.local_tip,
            'best_time': gem.best_time,
            'cost': gem.cost,
            'how_to_find': gem.how_to_find
        }
    
    def _find_gem_by_name(self, gem_name: str) -> Optional[Dict[str, Any]]:
        """Find gem in database by name"""
        if not GEMS_DB_AVAILABLE:
            return None
        
        gem_name_lower = gem_name.lower()
        
        for gems in HIDDEN_GEMS_DATABASE.values():
            for gem in gems:
                if gem['name'].lower() == gem_name_lower or gem_name_lower in gem['name'].lower():
                    return gem
        
        return None
    
    def _extract_gem_name(self, message: str) -> Optional[str]:
        """Extract gem name from navigation request"""
        # Simple extraction - can be enhanced with NLP
        patterns = [
            r'navigate to (.+)',
            r'take me to (.+)',
            r'directions to (.+)',
            r'go to (.+)'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_multiple_gem_names(self, message: str) -> List[str]:
        """Extract multiple gem names from tour request"""
        # Split by commas and 'and'
        import re
        parts = re.split(r',|\sand\s', message)
        return [p.strip() for p in parts if p.strip()]
    
    def _extract_gem_type(self, message: str) -> Optional[str]:
        """Extract gem type from message"""
        types = ['cafe', 'restaurant', 'park', 'historical', 'view', 'shopping', 'art', 'nightlife', 'nature']
        
        message_lower = message.lower()
        for gem_type in types:
            if gem_type in message_lower or f"{gem_type}s" in message_lower:
                return gem_type
        
        return None
    
    def _extract_neighborhood(self, message: str) -> Optional[str]:
        """Extract neighborhood from message (multilingual)"""
        message_lower = message.lower()
        
        # First try multilingual extraction if available
        if MULTILINGUAL_AVAILABLE:
            extracted = extract_neighborhood(message)
            if extracted:
                return extracted
        
        # Fallback to local neighborhood coords
        for neighborhood in NEIGHBORHOOD_COORDS.keys():
            if neighborhood in message_lower:
                return neighborhood
        
        return None
    
    def _create_navigation_map_data(
        self,
        nav_result: Dict[str, Any],
        gem_name: str,
        user_location: Dict[str, float],
        dest_lat: float,
        dest_lon: float
    ) -> Dict[str, Any]:
        """Create map data for navigation visualization"""
        return {
            'route': nav_result.get('route', {}).get('geometry', []),
            'markers': [
                {
                    'type': 'start',
                    'name': 'Your Location',
                    'lat': user_location['lat'],
                    'lon': user_location['lon'],
                    'icon': 'user'
                },
                {
                    'type': 'destination',
                    'name': gem_name,
                    'lat': dest_lat,
                    'lon': dest_lon,
                    'icon': 'star'
                }
            ],
            'center': user_location,
            'zoom': 14
        }


# Singleton instance
_hidden_gems_gps_integration = None


def get_hidden_gems_gps_integration(db_session=None) -> HiddenGemsGPSIntegration:
    """Get or create hidden gems GPS integration singleton"""
    global _hidden_gems_gps_integration
    
    if _hidden_gems_gps_integration is None:
        _hidden_gems_gps_integration = HiddenGemsGPSIntegration(db_session=db_session)
    
    return _hidden_gems_gps_integration
