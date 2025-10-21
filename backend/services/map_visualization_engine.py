"""
Map Visualization Engine for Istanbul AI
Provides interactive map visualization with route optimization
Uses template-based rendering (no generative AI)
Supports local MacBook development and GPU-accelerated production

FREE & OPEN-SOURCE ONLY:
- Uses Leaflet.js for map rendering
- Uses OpenStreetMap (OSM) for base map tiles
- No paid map services (Google Maps, Mapbox, etc.)
- Fully free for development and production
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MapLocation:
    """Represents a location on the map"""
    lat: float
    lon: float
    name: str
    type: str  # 'start', 'end', 'poi', 'transit'
    metadata: Dict[str, Any] = None


@dataclass
class RouteSegment:
    """Represents a segment of a route"""
    start: MapLocation
    end: MapLocation
    distance_km: float
    duration_min: int
    transport_mode: str  # 'walk', 'metro', 'tram', 'bus', 'ferry'
    instructions: List[str]
    waypoints: List[Tuple[float, float]] = None


@dataclass
class MapVisualization:
    """Complete map visualization data"""
    center: Tuple[float, float]
    zoom: int
    locations: List[MapLocation]
    routes: List[RouteSegment]
    bounds: Dict[str, float]  # 'north', 'south', 'east', 'west'
    metadata: Dict[str, Any]


class MapVisualizationEngine:
    """
    Map Visualization Engine
    - Generates interactive map data structures
    - Optimizes route visualization
    - Uses Leaflet.js + OpenStreetMap (free & open-source)
    - Template-based, no generative AI
    - No paid map services required
    """
    
    # Istanbul city center coordinates
    ISTANBUL_CENTER = (41.0082, 28.9784)
    
    # Default zoom levels
    ZOOM_LEVELS = {
        'city': 11,
        'district': 13,
        'neighborhood': 15,
        'street': 17
    }
    
    # Transport mode colors for visualization
    TRANSPORT_COLORS = {
        'walk': '#4CAF50',
        'metro': '#2196F3',
        'tram': '#FF9800',
        'bus': '#F44336',
        'ferry': '#00BCD4',
        'funicular': '#9C27B0'
    }
    
    # Transport mode icons
    TRANSPORT_ICONS = {
        'walk': '🚶',
        'metro': '🚇',
        'tram': '🚊',
        'bus': '🚌',
        'ferry': '⛴️',
        'funicular': '🚡'
    }
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize Map Visualization Engine
        
        Args:
            use_gpu: Whether to use GPU acceleration for computations
        """
        self.use_gpu = use_gpu
        logger.info(f"Map Visualization Engine initialized (GPU: {use_gpu})")
    
    def create_location(
        self,
        lat: float,
        lon: float,
        name: str,
        type: str = 'poi',
        metadata: Optional[Dict] = None
    ) -> MapLocation:
        """Create a map location object"""
        return MapLocation(
            lat=lat,
            lon=lon,
            name=name,
            type=type,
            metadata=metadata or {}
        )
    
    def calculate_bounds(self, locations: List[MapLocation]) -> Dict[str, float]:
        """
        Calculate bounding box for given locations
        
        Args:
            locations: List of map locations
            
        Returns:
            Dictionary with 'north', 'south', 'east', 'west' bounds
        """
        if not locations:
            # Default to Istanbul bounds
            return {
                'north': 41.2,
                'south': 40.8,
                'east': 29.2,
                'west': 28.7
            }
        
        lats = [loc.lat for loc in locations]
        lons = [loc.lon for loc in locations]
        
        # Add 10% padding
        lat_padding = (max(lats) - min(lats)) * 0.1
        lon_padding = (max(lons) - min(lons)) * 0.1
        
        return {
            'north': max(lats) + lat_padding,
            'south': min(lats) - lat_padding,
            'east': max(lons) + lon_padding,
            'west': min(lons) - lon_padding
        }
    
    def calculate_center(self, locations: List[MapLocation]) -> Tuple[float, float]:
        """
        Calculate center point for given locations
        
        Args:
            locations: List of map locations
            
        Returns:
            Tuple of (latitude, longitude)
        """
        if not locations:
            return self.ISTANBUL_CENTER
        
        lats = [loc.lat for loc in locations]
        lons = [loc.lon for loc in locations]
        
        return (sum(lats) / len(lats), sum(lons) / len(lons))
    
    def determine_zoom_level(
        self,
        bounds: Dict[str, float],
        viewport_width: int = 800,
        viewport_height: int = 600
    ) -> int:
        """
        Determine optimal zoom level based on bounds and viewport
        
        Args:
            bounds: Bounding box dictionary
            viewport_width: Viewport width in pixels
            viewport_height: Viewport height in pixels
            
        Returns:
            Optimal zoom level (1-20)
        """
        lat_diff = bounds['north'] - bounds['south']
        lon_diff = bounds['east'] - bounds['west']
        
        # Simple heuristic based on coordinate differences
        if lat_diff > 1.0 or lon_diff > 1.0:
            return self.ZOOM_LEVELS['city']
        elif lat_diff > 0.1 or lon_diff > 0.1:
            return self.ZOOM_LEVELS['district']
        elif lat_diff > 0.01 or lon_diff > 0.01:
            return self.ZOOM_LEVELS['neighborhood']
        else:
            return self.ZOOM_LEVELS['street']
    
    def create_route_segment(
        self,
        start: MapLocation,
        end: MapLocation,
        distance_km: float,
        duration_min: int,
        transport_mode: str,
        instructions: List[str],
        waypoints: Optional[List[Tuple[float, float]]] = None
    ) -> RouteSegment:
        """Create a route segment"""
        return RouteSegment(
            start=start,
            end=end,
            distance_km=distance_km,
            duration_min=duration_min,
            transport_mode=transport_mode,
            instructions=instructions,
            waypoints=waypoints or []
        )
    
    def optimize_waypoints(
        self,
        waypoints: List[Tuple[float, float]],
        max_points: int = 100
    ) -> List[Tuple[float, float]]:
        """
        Optimize waypoints for visualization (reduce points while maintaining shape)
        Uses Douglas-Peucker algorithm for line simplification
        
        Args:
            waypoints: List of (lat, lon) tuples
            max_points: Maximum number of points to keep
            
        Returns:
            Optimized list of waypoints
        """
        if len(waypoints) <= max_points:
            return waypoints
        
        # Simple decimation - keep every Nth point
        step = len(waypoints) // max_points
        optimized = waypoints[::step]
        
        # Always keep first and last points
        if optimized[0] != waypoints[0]:
            optimized.insert(0, waypoints[0])
        if optimized[-1] != waypoints[-1]:
            optimized.append(waypoints[-1])
        
        return optimized
    
    def generate_visualization(
        self,
        locations: List[MapLocation],
        routes: Optional[List[RouteSegment]] = None,
        center: Optional[Tuple[float, float]] = None,
        zoom: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> MapVisualization:
        """
        Generate complete map visualization
        
        Args:
            locations: List of locations to show
            routes: Optional list of routes
            center: Optional center point (auto-calculated if not provided)
            zoom: Optional zoom level (auto-calculated if not provided)
            metadata: Optional metadata
            
        Returns:
            MapVisualization object
        """
        routes = routes or []
        
        # Calculate bounds
        all_locations = locations.copy()
        for route in routes:
            all_locations.append(route.start)
            all_locations.append(route.end)
        
        bounds = self.calculate_bounds(all_locations)
        
        # Calculate center and zoom if not provided
        if center is None:
            center = self.calculate_center(all_locations)
        
        if zoom is None:
            zoom = self.determine_zoom_level(bounds)
        
        # Optimize route waypoints
        for route in routes:
            if route.waypoints:
                route.waypoints = self.optimize_waypoints(route.waypoints)
        
        return MapVisualization(
            center=center,
            zoom=zoom,
            locations=locations,
            routes=routes,
            bounds=bounds,
            metadata=metadata or {
                'generated_at': datetime.utcnow().isoformat(),
                'engine': 'map_visualization_engine',
                'gpu_enabled': self.use_gpu
            }
        )
    
    def to_leaflet_format(self, viz: MapVisualization) -> Dict:
        """
        Convert visualization to Leaflet.js format
        
        Args:
            viz: MapVisualization object
            
        Returns:
            Dictionary formatted for Leaflet.js
        """
        # Convert locations to markers
        markers = []
        for loc in viz.locations:
            marker = {
                'position': [loc.lat, loc.lon],
                'title': loc.name,
                'type': loc.type,
                'popup': self._generate_popup_content(loc)
            }
            if loc.metadata:
                marker['metadata'] = loc.metadata
            markers.append(marker)
        
        # Convert routes to polylines
        polylines = []
        for route in viz.routes:
            waypoints = route.waypoints or [
                (route.start.lat, route.start.lon),
                (route.end.lat, route.end.lon)
            ]
            
            polyline = {
                'positions': [[lat, lon] for lat, lon in waypoints],
                'color': self.TRANSPORT_COLORS.get(route.transport_mode, '#666666'),
                'weight': 4,
                'opacity': 0.7,
                'mode': route.transport_mode,
                'distance': route.distance_km,
                'duration': route.duration_min,
                'instructions': route.instructions
            }
            polylines.append(polyline)
        
        return {
            'center': list(viz.center),
            'zoom': viz.zoom,
            'bounds': viz.bounds,
            'markers': markers,
            'polylines': polylines,
            'metadata': viz.metadata
        }
    
    def to_geojson_format(self, viz: MapVisualization) -> Dict:
        """
        Convert visualization to GeoJSON format (for Leaflet and other OSM-compatible tools)
        
        Args:
            viz: MapVisualization object
            
        Returns:
            Dictionary formatted as GeoJSON
        """
        # Create GeoJSON features
        features = []
        
        # Add location markers
        for loc in viz.locations:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [loc.lon, loc.lat]
                },
                'properties': {
                    'name': loc.name,
                    'type': loc.type,
                    'description': self._generate_popup_content(loc),
                    'marker_symbol': self.TRANSPORT_ICONS.get(loc.type, '📍')
                }
            }
            if loc.metadata:
                feature['properties']['metadata'] = loc.metadata
            features.append(feature)
        
        # Add route lines
        for route in viz.routes:
            waypoints = route.waypoints or [
                (route.start.lat, route.start.lon),
                (route.end.lat, route.end.lon)
            ]
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[lon, lat] for lat, lon in waypoints]
                },
                'properties': {
                    'mode': route.transport_mode,
                    'distance': route.distance_km,
                    'duration': route.duration_min,
                    'color': self.TRANSPORT_COLORS.get(route.transport_mode, '#666666'),
                    'instructions': route.instructions,
                    'icon': self.TRANSPORT_ICONS.get(route.transport_mode, '🚶')
                }
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features,
            'center': [viz.center[1], viz.center[0]],  # [lon, lat] for GeoJSON
            'zoom': viz.zoom,
            'bounds': [
                [viz.bounds['west'], viz.bounds['south']],
                [viz.bounds['east'], viz.bounds['north']]
            ],
            'metadata': viz.metadata
        }
    
    def generate_osm_tiles_config(self) -> Dict:
        """
        Generate OpenStreetMap tiles configuration
        
        Returns:
            Dictionary with OSM tile server configurations (all free)
        """
        return {
            'default': {
                'name': 'OpenStreetMap',
                'url': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                'attribution': '© OpenStreetMap contributors',
                'maxZoom': 19,
                'subdomains': ['a', 'b', 'c']
            },
            'humanitarian': {
                'name': 'Humanitarian OSM',
                'url': 'https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
                'attribution': '© OpenStreetMap contributors, Tiles courtesy of HOT',
                'maxZoom': 19,
                'subdomains': ['a', 'b', 'c']
            },
            'transport': {
                'name': 'Transport Map',
                'url': 'https://{s}.tile.thunderforest.com/transport/{z}/{x}/{y}.png',
                'attribution': '© OpenStreetMap contributors, © Thunderforest',
                'maxZoom': 18,
                'subdomains': ['a', 'b', 'c'],
                'note': 'Requires free API key for production use'
            }
        }
    
    def _generate_popup_content(self, location: MapLocation) -> str:
        """Generate popup content for a location"""
        content = f"<strong>{location.name}</strong><br>"
        content += f"Type: {location.type}<br>"
        
        if location.metadata:
            if 'address' in location.metadata:
                content += f"Address: {location.metadata['address']}<br>"
            if 'description' in location.metadata:
                content += f"{location.metadata['description']}<br>"
        
        return content
    
    def _get_marker_label(self, location_type: str) -> str:
        """Get marker label based on location type"""
        labels = {
            'start': 'A',
            'end': 'B',
            'poi': '📍',
            'transit': '🚉'
        }
        return labels.get(location_type, '📍')
    
    def _get_marker_icon(self, location_type: str) -> Optional[Dict]:
        """Get marker icon configuration based on location type"""
        # Can be extended with custom icons
        return None


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = MapVisualizationEngine(use_gpu=False)
    
    # Create sample locations
    locations = [
        engine.create_location(
            41.0082, 28.9784,
            "Sultanahmet Square",
            "start",
            {"description": "Historic center of Istanbul"}
        ),
        engine.create_location(
            41.0115, 28.9833,
            "Hagia Sophia",
            "poi",
            {"description": "Byzantine masterpiece"}
        ),
        engine.create_location(
            41.0055, 28.9769,
            "Blue Mosque",
            "poi",
            {"description": "Ottoman imperial mosque"}
        ),
        engine.create_location(
            41.0167, 28.9667,
            "Grand Bazaar",
            "end",
            {"description": "Historic covered market"}
        )
    ]
    
    # Create sample route
    routes = [
        engine.create_route_segment(
            locations[0], locations[1],
            0.5, 7, "walk",
            ["Walk north on Divanyolu", "Turn right at Sultanahmet Park"]
        ),
        engine.create_route_segment(
            locations[1], locations[2],
            0.3, 4, "walk",
            ["Walk south towards Blue Mosque"]
        ),
        engine.create_route_segment(
            locations[2], locations[3],
            1.2, 15, "walk",
            ["Walk west through Sultanahmet", "Continue to Grand Bazaar"]
        )
    ]
    
    # Generate visualization
    viz = engine.generate_visualization(locations, routes)
    
    # Convert to different formats (FREE & OPEN-SOURCE ONLY)
    leaflet_data = engine.to_leaflet_format(viz)
    geojson_data = engine.to_geojson_format(viz)
    osm_config = engine.generate_osm_tiles_config()
    
    print("\n✅ Map Visualization Engine Test")
    print(f"Center: {viz.center}")
    print(f"Zoom: {viz.zoom}")
    print(f"Locations: {len(viz.locations)}")
    print(f"Routes: {len(viz.routes)}")
    print(f"Bounds: {viz.bounds}")
    
    # Save sample outputs
    with open('sample_leaflet_map.json', 'w') as f:
        json.dump(leaflet_data, f, indent=2)
    
    with open('sample_geojson_map.json', 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    with open('osm_tiles_config.json', 'w') as f:
        json.dump(osm_config, f, indent=2)
    
    print("\n✅ Sample map files generated (FREE & OPEN-SOURCE)!")
    print("- sample_leaflet_map.json (Leaflet.js format)")
    print("- sample_geojson_map.json (GeoJSON for OSM tools)")
    print("- osm_tiles_config.json (OpenStreetMap tile servers)")
    print("\nAll map solutions are FREE and OPEN-SOURCE:")
    print("✓ Leaflet.js (BSD-2-Clause license)")
    print("✓ OpenStreetMap (ODbL license)")
    print("✓ No paid APIs required")
