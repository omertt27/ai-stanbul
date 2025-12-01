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
except ImportError as e:
    GPS_NAV_AVAILABLE = False
    logger.warning(f"⚠️ GPS navigation not available: {e}")

# Import route integration
try:
    from backend.services.ai_chat_route_integration import (
        AIChatRouteHandler,
        create_chat_route_handler
    )
    ROUTE_HANDLER_AVAILABLE = True
except ImportError as e:
    ROUTE_HANDLER_AVAILABLE = False
    logger.warning(f"⚠️ Route handler not available: {e}")

# Import hidden gems database
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
        
        # Initialize GPS navigator
        if GPS_NAV_AVAILABLE:
            try:
                self.gps_navigator = GPSTurnByTurnNavigator(db_session=db_session)
                logger.info("✅ GPS navigator initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize GPS navigator: {e}")
                self.gps_navigator = None
        else:
            self.gps_navigator = None
        
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
        
        logger.info(f"🗺️ Hidden Gems GPS Integration initialized (GPS: {self.gps_navigator is not None}, Routes: {self.route_handler is not None})")
    
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
        if neighborhood:
            gems = get_gems_by_neighborhood(neighborhood.lower())
        elif gem_type:
            gems = get_gems_by_type(gem_type.lower())
        else:
            gems = get_all_hidden_gems()
        
        # Enrich gems with GPS coordinates
        enriched_gems = []
        map_markers = []
        
        for gem in gems[:limit * 2]:  # Get more than needed, filter by distance
            # Get neighborhood coordinates
            gem_neighborhood = gem.get('neighborhood', '').lower()
            if gem_neighborhood not in NEIGHBORHOOD_COORDS:
                continue
            
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
            'navigation_ready': self.gps_navigator is not None,
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
        if not self.gps_navigator:
            return {
                'error': 'GPS navigation not available',
                'message': 'Please enable GPS navigation to use this feature.'
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
        
        # Start navigation
        try:
            nav_result = self.gps_navigator.start_navigation(
                start_location=GPSLocation(
                    latitude=user_location['lat'],
                    longitude=user_location['lon'],
                    timestamp=datetime.now()
                ),
                destination=GPSLocation(
                    latitude=dest_lat,
                    longitude=dest_lon,
                    timestamp=datetime.now()
                ),
                session_id=session_id,
                mode=NavigationMode.WALKING
            )
            
            if not nav_result['success']:
                return {
                    'error': 'Navigation failed',
                    'message': nav_result.get('error', 'Could not start navigation')
                }
            
            # Format response
            return {
                'success': True,
                'message': f"🗺️ Navigating to {gem['name']}! Follow the turn-by-turn directions.",
                'gem': gem,
                'navigation_data': nav_result,
                'navigation_active': True,
                'map_data': self._create_navigation_map_data(
                    nav_result,
                    gem['name'],
                    user_location,
                    dest_lat,
                    dest_lon
                )
            }
            
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return {
                'error': 'Navigation error',
                'message': f'Failed to start navigation: {str(e)}'
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
        Handle hidden gem requests in chat
        
        Detects requests like:
        - "Show me hidden cafes"
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
        
        # Check for hidden gem discovery
        if any(keyword in message_lower for keyword in ['hidden', 'secret', 'local spot', 'off the beaten']):
            # Extract type and neighborhood
            gem_type = self._extract_gem_type(message)
            neighborhood = self._extract_neighborhood(message)
            
            return self.get_hidden_gems_with_navigation(
                user_location=user_location,
                gem_type=gem_type,
                neighborhood=neighborhood
            )
        
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
        """Extract neighborhood from message"""
        message_lower = message.lower()
        
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
