"""
Route Map Visualizer for Istanbul Transportation
Generates interactive map data for route visualization
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MapBounds:
    """Map bounds for centering"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates"""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2
        )
    
    @property
    def zoom(self) -> int:
        """Calculate appropriate zoom level"""
        lat_diff = self.max_lat - self.min_lat
        lon_diff = self.max_lon - self.min_lon
        max_diff = max(lat_diff, lon_diff)
        
        # Estimate zoom level based on coordinate difference
        if max_diff < 0.01:
            return 15  # Very close (neighborhood)
        elif max_diff < 0.05:
            return 13  # District
        elif max_diff < 0.1:
            return 12  # Multiple districts
        elif max_diff < 0.5:
            return 10  # City area
        else:
            return 9   # Metropolitan area


class RouteMapVisualizer:
    """
    Visualizer for generating map data from route information
    """
    
    # Transport type colors
    TRANSPORT_COLORS = {
        'metro': '#E31E24',      # Red
        'marmaray': '#9C27B0',   # Purple
        'bus': '#2196F3',        # Blue
        'ferry': '#00BCD4',      # Cyan
        'tram': '#4CAF50',       # Green
        'transfer': '#FF9800'    # Orange
    }
    
    # Line-specific colors (from Istanbul transport)
    LINE_COLORS = {
        # Metro lines
        'M1A': '#E31E24',  # Red
        'M1B': '#00A651',  # Light Green
        'M2': '#00A651',   # Green
        'M3': '#0033A0',   # Blue
        'M4': '#FF6B9D',   # Pink
        'M5': '#662D91',   # Purple
        'M6': '#CD5C5C',   # Brown
        'M7': '#F39C12',   # Orange
        'M9': '#FFD700',   # Yellow
        
        # Marmaray
        'Marmaray': '#9C27B0',  # Purple
        
        # Ferry routes
        'F1': '#00BCD4',
        'F2': '#00BCD4',
        'F3': '#00BCD4',
        'F4': '#00BCD4',
        
        # Tram lines
        'T1': '#4CAF50',
        'T4': '#4CAF50',
        'T5': '#4CAF50',
    }
    
    def __init__(self):
        """Initialize the map visualizer"""
        logger.info("🗺️ Route Map Visualizer initialized")
    
    def generate_route_map_data(self, route_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate map visualization data from route information
        
        Args:
            route_info: Route information from journey planner
        
        Returns:
            Map data with markers, polylines, and metadata
        """
        if not route_info or not route_info.get('success'):
            return self._generate_empty_map()
        
        map_data = {
            'success': True,
            'markers': [],
            'polylines': [],
            'bounds': None,
            'center': None,
            'zoom': 11,
            'metadata': {
                'origin': route_info['origin'],
                'destination': route_info['destination'],
                'duration': route_info['duration_minutes'],
                'distance': route_info['distance_km'],
                'transfers': route_info['transfers']
            }
        }
        
        try:
            # Collect all coordinates for bounds calculation
            all_coords = []
            
            # Process each segment
            for idx, segment in enumerate(route_info.get('segments', [])):
                segment_data = self._process_segment(segment, idx)
                
                # Add markers for stops
                if segment_data['start_marker']:
                    map_data['markers'].append(segment_data['start_marker'])
                    all_coords.append(segment_data['start_marker']['coordinates'])
                
                if segment_data['end_marker']:
                    map_data['markers'].append(segment_data['end_marker'])
                    all_coords.append(segment_data['end_marker']['coordinates'])
                
                # Add polyline for route segment
                if segment_data['polyline']:
                    map_data['polylines'].append(segment_data['polyline'])
            
            # Calculate map bounds and center
            if all_coords:
                bounds = self._calculate_bounds(all_coords)
                map_data['bounds'] = {
                    'southwest': [bounds.min_lat, bounds.min_lon],
                    'northeast': [bounds.max_lat, bounds.max_lon]
                }
                map_data['center'] = list(bounds.center)
                map_data['zoom'] = bounds.zoom
            
            logger.info(f"✓ Generated map data: {len(map_data['markers'])} markers, {len(map_data['polylines'])} lines")
            
        except Exception as e:
            logger.error(f"Error generating map data: {e}")
            return self._generate_empty_map()
        
        return map_data
    
    def _process_segment(self, segment: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a route segment into map data"""
        transport_type = segment.get('type', 'metro').lower()
        line_id = segment.get('line', '')
        
        # Get color for this line/type
        color = self.LINE_COLORS.get(line_id, self.TRANSPORT_COLORS.get(transport_type, '#666666'))
        
        # Get coordinates (if available)
        from_coords = segment.get('from_coordinates', [41.0082, 28.9784])  # Default to Istanbul center
        to_coords = segment.get('to_coordinates', [41.0082, 28.9784])
        
        # Create markers
        start_marker = {
            'type': 'stop',
            'coordinates': from_coords,
            'label': segment.get('from_stop', 'Start'),
            'icon': self._get_transport_icon(transport_type),
            'color': color,
            'index': index * 2,
            'popup': self._create_stop_popup(segment, 'start')
        }
        
        end_marker = {
            'type': 'stop',
            'coordinates': to_coords,
            'label': segment.get('to_stop', 'End'),
            'icon': self._get_transport_icon(transport_type),
            'color': color,
            'index': index * 2 + 1,
            'popup': self._create_stop_popup(segment, 'end')
        }
        
        # Create polyline
        polyline = {
            'type': 'route',
            'coordinates': [from_coords, to_coords],
            'color': color,
            'weight': 4,
            'opacity': 0.8,
            'transport_type': transport_type,
            'line_name': segment.get('line_name', ''),
            'popup': self._create_line_popup(segment)
        }
        
        return {
            'start_marker': start_marker,
            'end_marker': end_marker,
            'polyline': polyline
        }
    
    def _get_transport_icon(self, transport_type: str) -> str:
        """Get icon name for transport type"""
        icons = {
            'metro': '🚇',
            'marmaray': '🚄',
            'bus': '🚌',
            'ferry': '⛴️',
            'tram': '🚊',
            'transfer': '🔄'
        }
        return icons.get(transport_type, '📍')
    
    def _create_stop_popup(self, segment: Dict[str, Any], position: str) -> str:
        """Create popup content for a stop"""
        if position == 'start':
            stop_name = segment.get('from_stop', 'Stop')
        else:
            stop_name = segment.get('to_stop', 'Stop')
        
        line_name = segment.get('line_name', 'Line')
        transport_type = segment.get('type', 'metro').title()
        
        popup = f"""
        <div style="min-width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: #333;">
                {self._get_transport_icon(segment.get('type', 'metro'))} {stop_name}
            </h4>
            <div style="font-size: 13px; color: #666;">
                <strong>{line_name}</strong><br/>
                Type: {transport_type}
            </div>
        </div>
        """
        return popup.strip()
    
    def _create_line_popup(self, segment: Dict[str, Any]) -> str:
        """Create popup content for a route line"""
        line_name = segment.get('line_name', 'Line')
        duration = segment.get('duration_minutes', 0)
        stops = segment.get('stops_count', 0)
        
        popup = f"""
        <div style="min-width: 180px;">
            <h4 style="margin: 0 0 8px 0; color: #333;">
                {line_name}
            </h4>
            <div style="font-size: 13px; color: #666;">
                ⏱️ Duration: {duration:.0f} min<br/>
                🚏 Stops: {stops}
            </div>
        </div>
        """
        return popup.strip()
    
    def _calculate_bounds(self, coordinates: List[List[float]]) -> MapBounds:
        """Calculate map bounds from coordinate list"""
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        return MapBounds(
            min_lat=min(lats),
            max_lat=max(lats),
            min_lon=min(lons),
            max_lon=max(lons)
        )
    
    def _generate_empty_map(self) -> Dict[str, Any]:
        """Generate empty map data (Istanbul center)"""
        return {
            'success': False,
            'markers': [],
            'polylines': [],
            'bounds': None,
            'center': [41.0082, 28.9784],  # Istanbul center
            'zoom': 11,
            'metadata': {}
        }
    
    def generate_map_html(self, route_info: Dict[str, Any]) -> str:
        """
        Generate complete HTML page with interactive map
        
        Args:
            route_info: Route information from journey planner
        
        Returns:
            HTML string with embedded Leaflet map
        """
        map_data = self.generate_route_map_data(route_info)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Istanbul Route Map - {map_data['metadata'].get('origin', 'Start')} to {map_data['metadata'].get('destination', 'End')}</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
        }}
        
        #map-container {{
            position: relative;
            width: 100%;
            height: 100vh;
        }}
        
        #map {{
            width: 100%;
            height: 100%;
        }}
        
        #route-info {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            max-width: 350px;
        }}
        
        #route-info h2 {{
            font-size: 18px;
            margin-bottom: 12px;
            color: #333;
        }}
        
        .route-detail {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 14px;
            color: #666;
        }}
        
        .route-detail strong {{
            margin-right: 8px;
            color: #333;
        }}
        
        .legend {{
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #eee;
        }}
        
        .legend-title {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            margin-bottom: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
            font-size: 12px;
            color: #666;
        }}
        
        .legend-icon {{
            width: 20px;
            height: 3px;
            margin-right: 8px;
            border-radius: 2px;
        }}
        
        @media (max-width: 768px) {{
            #route-info {{
                left: 10px;
                top: 10px;
                max-width: calc(100% - 20px);
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div id="map-container">
        <!-- Route Information Panel -->
        <div id="route-info">
            <h2>🗺️ Route Information</h2>
            <div class="route-detail">
                <strong>From:</strong> {map_data['metadata'].get('origin', 'Start')}
            </div>
            <div class="route-detail">
                <strong>To:</strong> {map_data['metadata'].get('destination', 'End')}
            </div>
            <div class="route-detail">
                <strong>⏱️ Duration:</strong> {map_data['metadata'].get('duration', 0):.0f} minutes
            </div>
            <div class="route-detail">
                <strong>📏 Distance:</strong> {map_data['metadata'].get('distance', 0):.1f} km
            </div>
            <div class="route-detail">
                <strong>🔄 Transfers:</strong> {map_data['metadata'].get('transfers', 0)}
            </div>
            
            <div class="legend">
                <div class="legend-title">Transport Types:</div>
                <div class="legend-item">
                    <div class="legend-icon" style="background: #E31E24;"></div>
                    <span>Metro</span>
                </div>
                <div class="legend-item">
                    <div class="legend-icon" style="background: #9C27B0;"></div>
                    <span>Marmaray</span>
                </div>
                <div class="legend-item">
                    <div class="legend-icon" style="background: #00BCD4;"></div>
                    <span>Ferry</span>
                </div>
                <div class="legend-item">
                    <div class="legend-icon" style="background: #4CAF50;"></div>
                    <span>Tram</span>
                </div>
            </div>
        </div>
        
        <!-- Map -->
        <div id="map"></div>
    </div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // Map data from Python
        const mapData = {json.dumps(map_data, indent=2)};
        
        // Initialize map
        const map = L.map('map').setView(mapData.center, mapData.zoom);
        
        // Add tile layer (OpenStreetMap)
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);
        
        // Add polylines (route lines)
        mapData.polylines.forEach(function(line) {{
            const polyline = L.polyline(line.coordinates, {{
                color: line.color,
                weight: line.weight,
                opacity: line.opacity
            }}).addTo(map);
            
            if (line.popup) {{
                polyline.bindPopup(line.popup);
            }}
        }});
        
        // Add markers (stops)
        mapData.markers.forEach(function(marker, index) {{
            const icon = L.divIcon({{
                className: 'custom-marker',
                html: `<div style="
                    background: ${{marker.color}};
                    color: white;
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    border: 3px solid white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                ">${{marker.icon}}</div>`,
                iconSize: [32, 32],
                iconAnchor: [16, 16]
            }});
            
            const leafletMarker = L.marker(marker.coordinates, {{ icon: icon }}).addTo(map);
            
            if (marker.popup) {{
                leafletMarker.bindPopup(marker.popup);
            }}
        }});
        
        // Fit map to bounds if available
        if (mapData.bounds) {{
            map.fitBounds([
                mapData.bounds.southwest,
                mapData.bounds.northeast
            ], {{ padding: [50, 50] }});
        }}
    </script>
</body>
</html>
        """
        
        return html


# Global instance
_map_visualizer = None

def get_map_visualizer() -> RouteMapVisualizer:
    """Get or create the global map visualizer instance"""
    global _map_visualizer
    if _map_visualizer is None:
        _map_visualizer = RouteMapVisualizer()
    return _map_visualizer
