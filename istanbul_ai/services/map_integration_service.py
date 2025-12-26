"""
Map Integration Service for Istanbul AI
Connects handlers with map_visualization_engine to provide visual maps in responses
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Import map visualization engine
try:
    from backend.services.map_visualization_engine import (
        MapVisualizationEngine,
        MapLocation,
        RouteSegment,
        MapVisualization
    )
    MAP_ENGINE_AVAILABLE = True
except ImportError:
    MAP_ENGINE_AVAILABLE = False
    logger.warning("Map visualization engine not available - maps will be disabled")


class MapIntegrationService:
    """
    Integrates map visualization into handler responses
    Provides easy methods for handlers to add maps to their responses
    """
    
    def __init__(self, use_gpu: bool = False, use_osrm: bool = True):
        """
        Initialize Map Integration Service
        
        Args:
            use_gpu: Whether to use GPU acceleration
            use_osrm: Whether to use OSRM for realistic routes
        """
        self.enabled = MAP_ENGINE_AVAILABLE
        self.engine = None
        
        if self.enabled:
            try:
                self.engine = MapVisualizationEngine(use_gpu=use_gpu, use_osrm=use_osrm)
                logger.info("Map Integration Service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize map engine: {e}")
                self.enabled = False
        else:
            logger.warning("Map Integration Service disabled - engine not available")
    
    def is_enabled(self) -> bool:
        """Check if map integration is enabled"""
        return self.enabled and self.engine is not None
    
    def create_attraction_map(
        self,
        attractions: List[Dict[str, Any]],
        user_location: Optional[Tuple[float, float]] = None
    ) -> Optional[Dict]:
        """
        Create a map for attraction recommendations
        
        Args:
            attractions: List of attraction dictionaries with 'name', 'lat', 'lon', etc.
            user_location: Optional user location as (lat, lon)
            
        Returns:
            Map data dictionary in Leaflet format, or None if disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            locations = []
            
            # Add user location if provided
            if user_location:
                locations.append(
                    self.engine.create_location(
                        user_location[0],
                        user_location[1],
                        "Your Location",
                        "start",
                        {"description": "Your current location"}
                    )
                )
            
            # Add attractions
            for attr in attractions:
                # Support both 'lat'/'lon' and 'latitude'/'longitude' field names
                lat = attr.get('lat') or attr.get('latitude')
                lon = attr.get('lon') or attr.get('lng') or attr.get('longitude')
                
                if lat is not None and lon is not None:
                    metadata = {
                        'description': attr.get('description', ''),
                        'address': attr.get('address', ''),
                        'rating': attr.get('rating', ''),
                        'category': attr.get('category', ''),
                        'district': attr.get('district', '')
                    }
                    locations.append(
                        self.engine.create_location(
                            lat,
                            lon,
                            attr.get('name', 'Attraction'),
                            'poi',
                            metadata
                        )
                    )
            
            if not locations:
                return None
            
            # Generate visualization
            viz = self.engine.generate_visualization(locations)
            
            # Return in Leaflet format
            return self.engine.to_leaflet_format(viz)
            
        except Exception as e:
            logger.error(f"Error creating attraction map: {e}")
            return None
    
    def create_restaurant_map(
        self,
        restaurants: List[Dict[str, Any]],
        user_location: Optional[Tuple[float, float]] = None
    ) -> Optional[Dict]:
        """
        Create a map for restaurant recommendations
        
        Args:
            restaurants: List of restaurant dictionaries with 'name', 'lat', 'lon', etc.
            user_location: Optional user location as (lat, lon)
            
        Returns:
            Map data dictionary in Leaflet format, or None if disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            locations = []
            
            # Add user location if provided
            if user_location:
                locations.append(
                    self.engine.create_location(
                        user_location[0],
                        user_location[1],
                        "Your Location",
                        "start",
                        {"description": "Your current location"}
                    )
                )
            
            # Add restaurants
            for rest in restaurants:
                # Support both 'lat'/'lon' and 'latitude'/'longitude' field names
                lat = rest.get('lat') or rest.get('latitude')
                lon = rest.get('lon') or rest.get('lng') or rest.get('longitude')
                
                if lat is not None and lon is not None:
                    metadata = {
                        'description': rest.get('description', ''),
                        'address': rest.get('address', ''),
                        'cuisine': rest.get('cuisine', '') or ', '.join(rest.get('cuisine_types', [])),
                        'price_range': rest.get('price_range', '') or rest.get('budget_category', ''),
                        'rating': rest.get('rating', ''),
                        'district': rest.get('district', ''),
                        'neighborhood': rest.get('neighborhood', '')
                    }
                    locations.append(
                        self.engine.create_location(
                            lat,
                            lon,
                            rest.get('name', 'Restaurant'),
                            'poi',
                            metadata
                        )
                    )
            
            if not locations:
                return None
            
            # Generate visualization
            viz = self.engine.generate_visualization(locations)
            
            # Return in Leaflet format
            return self.engine.to_leaflet_format(viz)
            
        except Exception as e:
            logger.error(f"Error creating restaurant map: {e}")
            return None
    
    def create_route_map(
        self,
        start_location: Tuple[float, float, str],  # (lat, lon, name)
        end_location: Tuple[float, float, str],    # (lat, lon, name)
        waypoints: Optional[List[Tuple[float, float, str]]] = None,
        route_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict]:
        """
        Create a map for route planning with realistic walking routes
        
        Args:
            start_location: Start point as (lat, lon, name)
            end_location: End point as (lat, lon, name)
            waypoints: Optional intermediate points as [(lat, lon, name), ...]
            route_info: Optional route information (distance, duration, etc.)
            
        Returns:
            Map data dictionary in Leaflet format, or None if disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            locations = []
            routes = []
            
            # Create start location
            start_loc = self.engine.create_location(
                start_location[0],
                start_location[1],
                start_location[2],
                "start",
                {"description": "Starting point"}
            )
            locations.append(start_loc)
            
            # Create end location
            end_loc = self.engine.create_location(
                end_location[0],
                end_location[1],
                end_location[2],
                "end",
                {"description": "Destination"}
            )
            locations.append(end_loc)
            
            # Create waypoint locations
            waypoint_locs = []
            if waypoints:
                for wp in waypoints:
                    wp_loc = self.engine.create_location(
                        wp[0], wp[1], wp[2],
                        "poi",
                        {"description": "Waypoint"}
                    )
                    locations.append(wp_loc)
                    waypoint_locs.append(wp_loc)
            
            # Create realistic walking route using OSRM
            route = self.engine.create_realistic_walking_route(
                start_loc,
                end_loc,
                waypoint_locs if waypoint_locs else None
            )
            
            if route:
                routes.append(route)
            
            # Generate visualization
            viz = self.engine.generate_visualization(locations, routes)
            
            # Return in Leaflet format
            return self.engine.to_leaflet_format(viz)
            
        except Exception as e:
            logger.error(f"Error creating route map: {e}")
            return None
    
    def create_transportation_map(
        self,
        start_location: Tuple[float, float, str],
        end_location: Tuple[float, float, str],
        transport_segments: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict]:
        """
        Create a map for transportation routes with multiple modes
        
        Args:
            start_location: Start point as (lat, lon, name)
            end_location: End point as (lat, lon, name)
            transport_segments: List of transport segment dictionaries
            
        Returns:
            Map data dictionary in Leaflet format, or None if disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            locations = []
            routes = []
            
            # Create start location
            start_loc = self.engine.create_location(
                start_location[0],
                start_location[1],
                start_location[2],
                "start",
                {"description": "Starting point"}
            )
            locations.append(start_loc)
            
            # Create end location
            end_loc = self.engine.create_location(
                end_location[0],
                end_location[1],
                end_location[2],
                "end",
                {"description": "Destination"}
            )
            locations.append(end_loc)
            
            # Process transport segments
            if transport_segments:
                for segment in transport_segments:
                    seg_start = self.engine.create_location(
                        segment.get('start_lat', start_location[0]),
                        segment.get('start_lon', start_location[1]),
                        segment.get('start_name', 'Start'),
                        'transit'
                    )
                    seg_end = self.engine.create_location(
                        segment.get('end_lat', end_location[0]),
                        segment.get('end_lon', end_location[1]),
                        segment.get('end_name', 'End'),
                        'transit'
                    )
                    
                    # Create route segment
                    route_seg = self.engine.create_route_segment(
                        seg_start,
                        seg_end,
                        segment.get('distance_km', 0),
                        segment.get('duration_min', 0),
                        segment.get('mode', 'walk'),
                        segment.get('instructions', [])
                    )
                    routes.append(route_seg)
                    
                    # Add transit stations to locations
                    if segment.get('show_station', True):
                        locations.append(seg_start)
                        locations.append(seg_end)
            else:
                # Create default walking route
                route = self.engine.create_realistic_walking_route(start_loc, end_loc)
                if route:
                    routes.append(route)
            
            # Generate visualization
            viz = self.engine.generate_visualization(locations, routes)
            
            # Return in Leaflet format
            return self.engine.to_leaflet_format(viz)
            
        except Exception as e:
            logger.error(f"Error creating transportation map: {e}")
            return None
    
    def create_multi_location_map(
        self,
        locations_data: List[Dict[str, Any]],
        center: Optional[Tuple[float, float]] = None
    ) -> Optional[Dict]:
        """
        Create a general-purpose map with multiple locations
        
        Args:
            locations_data: List of location dictionaries with 'lat', 'lon', 'name', etc.
            center: Optional center point override
            
        Returns:
            Map data dictionary in Leaflet format, or None if disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            locations = []
            
            for loc_data in locations_data:
                if 'lat' in loc_data and 'lon' in loc_data:
                    loc = self.engine.create_location(
                        loc_data['lat'],
                        loc_data['lon'],
                        loc_data.get('name', 'Location'),
                        loc_data.get('type', 'poi'),
                        loc_data.get('metadata', {})
                    )
                    locations.append(loc)
            
            if not locations:
                return None
            
            # Generate visualization
            viz = self.engine.generate_visualization(locations, center=center)
            
            # Return in Leaflet format
            return self.engine.to_leaflet_format(viz)
            
        except Exception as e:
            logger.error(f"Error creating multi-location map: {e}")
            return None
    
    def create_neighborhood_map(
        self,
        neighborhoods: List[Dict[str, Any]],
        user_location: Optional[Tuple[float, float]] = None
    ) -> Optional[Dict]:
        """
        Create a map for neighborhood recommendations
        
        Args:
            neighborhoods: List of neighborhood dictionaries with 'name', 'lat', 'lon', etc.
            user_location: Optional user location as (lat, lon)
            
        Returns:
            Map data dictionary in Leaflet format, or None if disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            locations = []
            
            # Add user location if provided
            if user_location:
                locations.append(
                    self.engine.create_location(
                        user_location[0],
                        user_location[1],
                        "Your Location",
                        "start",
                        {"description": "Your current location"}
                    )
                )
            
            # Add neighborhoods
            for neighborhood in neighborhoods:
                if 'lat' in neighborhood and 'lon' in neighborhood:
                    metadata = {
                        'description': neighborhood.get('character', {}).get('vibe', ''),
                        'district_type': neighborhood.get('district_type', ''),
                        'best_time': ', '.join(neighborhood.get('character', {}).get('best_time', [])),
                        'crowd': neighborhood.get('character', {}).get('crowd', '')
                    }
                    locations.append(
                        self.engine.create_location(
                            neighborhood['lat'],
                            neighborhood['lon'],
                            neighborhood.get('name', 'Neighborhood'),
                            'poi',
                            metadata
                        )
                    )
            
            if not locations:
                return None
            
            # Generate visualization
            viz = self.engine.generate_visualization(locations)
            
            # Return in Leaflet format
            return self.engine.to_leaflet_format(viz)
            
        except Exception as e:
            logger.error(f"Error creating neighborhood map: {e}")
            return None
    
    def create_hidden_gem_map(
        self,
        hidden_gems: List[Dict[str, Any]],
        user_location: Optional[Tuple[float, float]] = None
    ) -> Optional[Dict]:
        """
        Create a map for hidden gem recommendations
        
        Args:
            hidden_gems: List of hidden gem dictionaries with 'name', 'lat', 'lon', etc.
            user_location: Optional user location as (lat, lon)
            
        Returns:
            Map data dictionary in Leaflet format, or None if disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            locations = []
            
            # Add user location if provided
            if user_location:
                locations.append(
                    self.engine.create_location(
                        user_location[0],
                        user_location[1],
                        "Your Location",
                        "start",
                        {"description": "Your current location"}
                    )
                )
            
            # Add hidden gems with special marker
            for gem in hidden_gems:
                if 'lat' in gem and 'lon' in gem:
                    metadata = {
                        'description': gem.get('description', ''),
                        'category': gem.get('category', 'hidden_gem'),
                        'local_tip': gem.get('local_tip', ''),
                        'authenticity_score': gem.get('authenticity_score', '')
                    }
                    locations.append(
                        self.engine.create_location(
                            gem['lat'],
                            gem['lon'],
                            gem.get('name', 'Hidden Gem'),
                            'poi',  # Could use custom 'hidden_gem' type
                            metadata
                        )
                    )
            
            if not locations:
                return None
            
            # Generate visualization
            viz = self.engine.generate_visualization(locations)
            
            # Return in Leaflet format
            return self.engine.to_leaflet_format(viz)
            
        except Exception as e:
            logger.error(f"Error creating hidden gem map: {e}")
            return None
    
    def create_advanced_transportation_map(
        self,
        start_location: Tuple[float, float, str],
        end_location: Tuple[float, float, str],
        route_segments: List[Dict[str, Any]],
        route_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict]:
        """
        Create PRODUCTION-QUALITY transportation map (Google Maps/Moovit level)
        
        Features:
        - Multi-modal transport (metro, bus, tram, ferry, walking)
        - Color-coded routes by transport type
        - Station markers with names
        - Transfer points highlighted
        - Distance and duration per segment
        - Total route stats
        - Real-time alternative routes
        
        Args:
            start_location: Start point as (lat, lon, name)
            end_location: End point as (lat, lon, name)
            route_segments: List of route segment dictionaries with:
                - mode: 'metro', 'bus', 'tram', 'ferry', 'walk'
                - start_lat, start_lon, start_name
                - end_lat, end_lon, end_name
                - line_name: e.g., "M2 Metro", "28 Bus"
                - line_color: hex color for the line
                - duration_min: minutes for this segment
                - distance_km: kilometers for this segment
                - stations: list of intermediate stations (optional)
                - instructions: step-by-step directions
            route_metadata: Optional metadata with:
                - total_duration_min: total trip time
                - total_distance_km: total distance
                - transfer_count: number of transfers
                - fare_info: fare details
                - alternatives: alternative routes
        
        Returns:
            Enhanced map data dictionary in Leaflet format
        """
        if not self.is_enabled():
            return None
        
        try:
            locations = []
            routes = []
            
            # Color scheme for transport modes (Google Maps style)
            mode_colors = {
                'metro': '#0066CC',      # Blue
                'tram': '#FF3333',       # Red
                'bus': '#00AA00',        # Green
                'ferry': '#00CCCC',      # Cyan
                'funicular': '#9933FF',  # Purple
                'walk': '#888888',       # Gray
                'metrobus': '#FF9900'    # Orange
            }
            
            # Create start marker
            start_loc = self.engine.create_location(
                start_location[0],
                start_location[1],
                start_location[2],
                "start",
                {
                    "description": "Starting point",
                    "icon": "start",
                    "color": "#4CAF50"
                }
            )
            locations.append(start_loc)
            
            # Create end marker
            end_loc = self.engine.create_location(
                end_location[0],
                end_location[1],
                end_location[2],
                "end",
                {
                    "description": "Destination",
                    "icon": "destination",
                    "color": "#F44336"
                }
            )
            locations.append(end_loc)
            
            # Process each route segment
            previous_end = None
            segment_number = 1
            
            for idx, segment in enumerate(route_segments):
                mode = segment.get('mode', 'walk').lower()
                
                # Create segment start location
                seg_start = self.engine.create_location(
                    segment.get('start_lat'),
                    segment.get('start_lon'),
                    segment.get('start_name', f'Stop {segment_number}'),
                    'transit',
                    {
                        "mode": mode,
                        "segment_number": segment_number,
                        "is_transfer": previous_end is not None and mode != 'walk',
                        "line_name": segment.get('line_name', ''),
                        "icon": self._get_transport_icon(mode)
                    }
                )
                
                # Create segment end location
                seg_end = self.engine.create_location(
                    segment.get('end_lat'),
                    segment.get('end_lon'),
                    segment.get('end_name', f'Stop {segment_number + 1}'),
                    'transit',
                    {
                        "mode": mode,
                        "segment_number": segment_number + 1,
                        "line_name": segment.get('line_name', ''),
                        "icon": self._get_transport_icon(mode)
                    }
                )
                
                # Add locations (avoid duplicates at transfer points)
                if idx == 0 or mode != 'walk':
                    locations.append(seg_start)
                locations.append(seg_end)
                
                # Add intermediate stations if provided
                station_locations = []
                if 'stations' in segment and segment['stations']:
                    for station in segment['stations']:
                        station_loc = self.engine.create_location(
                            station.get('lat'),
                            station.get('lon'),
                            station.get('name', 'Station'),
                            'transit',
                            {
                                "mode": mode,
                                "is_intermediate": True,
                                "line_name": segment.get('line_name', ''),
                                "icon": self._get_transport_icon(mode)
                            }
                        )
                        station_locations.append(station_loc)
                        locations.append(station_loc)
                
                # Create route segment with proper styling
                route_color = segment.get('line_color', mode_colors.get(mode, '#888888'))
                
                # Create route (use OSRM for walking, straight lines for transit)
                if mode == 'walk':
                    # Realistic walking route
                    route_seg = self.engine.create_realistic_walking_route(
                        seg_start, seg_end
                    )
                else:
                    # Transit route (could be straight line or predefined route)
                    route_seg = self.engine.create_route_segment(
                        seg_start,
                        seg_end,
                        segment.get('distance_km', 0),
                        segment.get('duration_min', 0),
                        mode,
                        segment.get('instructions', [])
                    )
                
                if route_seg:
                    # Add route styling metadata
                    route_seg.metadata['color'] = route_color
                    route_seg.metadata['line_name'] = segment.get('line_name', mode.title())
                    route_seg.metadata['segment_number'] = segment_number
                    routes.append(route_seg)
                
                previous_end = seg_end
                segment_number += 1
            
            # Generate visualization
            viz = self.engine.generate_visualization(locations, routes)
            
            # Convert to Leaflet format
            map_data = self.engine.to_leaflet_format(viz)
            
            # Enhance with route metadata (Google Maps style)
            if route_metadata:
                map_data['route_info'] = {
                    'total_duration_min': route_metadata.get('total_duration_min', 0),
                    'total_distance_km': route_metadata.get('total_distance_km', 0),
                    'transfer_count': route_metadata.get('transfer_count', 0),
                    'fare_info': route_metadata.get('fare_info', {}),
                    'segments_count': len(route_segments),
                    'modes_used': list(set([s.get('mode', 'walk') for s in route_segments]))
                }
                
                # Add alternative routes if provided
                if 'alternatives' in route_metadata:
                    map_data['alternatives'] = route_metadata['alternatives']
            
            # Add transport legend
            map_data['legend'] = {
                mode: {
                    'color': color,
                    'icon': self._get_transport_icon(mode),
                    'label': mode.title()
                }
                for mode, color in mode_colors.items()
                if any(s.get('mode') == mode for s in route_segments)
            }
            
            logger.info(
                f"🗺️ Generated ADVANCED transportation map: "
                f"{len(route_segments)} segments, "
                f"{len(locations)} locations, "
                f"{route_metadata.get('total_duration_min', 0)} min"
            )
            
            return map_data
            
        except Exception as e:
            logger.error(f"Error creating advanced transportation map: {e}")
            return None
    
    def _get_transport_icon(self, mode: str) -> str:
        """Get appropriate icon for transport mode"""
        icon_map = {
            'metro': 'subway',
            'tram': 'tram',
            'bus': 'directions_bus',
            'ferry': 'directions_boat',
            'funicular': 'cable_car',
            'walk': 'directions_walk',
            'metrobus': 'directions_bus'
        }
        return icon_map.get(mode.lower(), 'place')


# Global instance (singleton)
_map_service_instance = None


def get_map_service(use_gpu: bool = False, use_osrm: bool = True) -> MapIntegrationService:
    """
    Get the global MapIntegrationService instance (singleton)
    
    Args:
        use_gpu: Whether to use GPU acceleration
        use_osrm: Whether to use OSRM for realistic routes
        
    Returns:
        MapIntegrationService instance
    """
    global _map_service_instance
    
    if _map_service_instance is None:
        _map_service_instance = MapIntegrationService(use_gpu=use_gpu, use_osrm=use_osrm)
    
    return _map_service_instance


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Get service
    service = get_map_service()
    
    if service.is_enabled():
        print("✅ Map Integration Service is enabled")
        
        # Test attraction map
        attractions = [
            {
                'name': 'Hagia Sophia',
                'lat': 41.0115,
                'lon': 28.9833,
                'description': 'Byzantine masterpiece',
                'category': 'Historical'
            },
            {
                'name': 'Blue Mosque',
                'lat': 41.0055,
                'lon': 28.9769,
                'description': 'Ottoman imperial mosque',
                'category': 'Religious'
            }
        ]
        
        map_data = service.create_attraction_map(attractions, user_location=(41.0082, 28.9784))
        
        if map_data:
            print(f"✅ Generated map with {len(map_data['markers'])} markers")
            print(f"   Center: {map_data['center']}")
            print(f"   Zoom: {map_data['zoom']}")
        else:
            print("❌ Failed to generate map")
    else:
        print("❌ Map Integration Service is disabled")
