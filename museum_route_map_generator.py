"""
Museum Route Map Generator
==========================
Generates interactive HTML maps for museum routes using Leaflet.js + OpenStreetMap
Integrates with existing map visualization engine and museum database
Uses FREE & OPEN-SOURCE technologies only:
- Leaflet.js for interactive maps
- OpenStreetMap for base tiles
- OSRM for realistic walking routes
- Accurate museum database with verified information
"""

import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging

# Import existing map visualization engine
try:
    from backend.services.map_visualization_engine import MapVisualizationEngine, MapLocation, RouteSegment
    MAP_ENGINE_AVAILABLE = True
except ImportError:
    MAP_ENGINE_AVAILABLE = False

# Import accurate museum database
try:
    from backend.accurate_museum_database import IstanbulMuseumDatabase
    MUSEUM_DB_AVAILABLE = True
except ImportError:
    MUSEUM_DB_AVAILABLE = False

# Import OSRM routing for walking paths
try:
    from backend.services.osrm_routing_service import OSRMRoutingService
    OSRM_AVAILABLE = True
except ImportError:
    OSRM_AVAILABLE = False

logger = logging.getLogger(__name__)

class MuseumRouteMapGenerator:
    """Generate interactive maps for museum routes using existing systems"""
    
    def __init__(self):
        self.default_center = (41.0082, 28.9784)  # Istanbul center
        self.default_zoom = 13
        
        # Initialize map engine
        if MAP_ENGINE_AVAILABLE:
            self.map_engine = MapVisualizationEngine(use_osrm=OSRM_AVAILABLE)
            logger.info("✅ Map visualization engine loaded")
        else:
            self.map_engine = None
            logger.warning("⚠️ Map engine not available - using basic maps")
        
        # Initialize museum database
        if MUSEUM_DB_AVAILABLE:
            self.museum_db = IstanbulMuseumDatabase()
            logger.info("✅ Accurate museum database loaded")
        else:
            self.museum_db = None
            logger.warning("⚠️ Museum database not available")
        
        # Initialize OSRM for walking routes
        if OSRM_AVAILABLE:
            try:
                self.osrm = OSRMRoutingService(profile='foot')
                logger.info("✅ OSRM routing service loaded for walking paths")
            except:
                self.osrm = None
                logger.warning("⚠️ OSRM not available - using straight lines")
        else:
            self.osrm = None
        
    def create_interactive_map(
        self, 
        route_data: Dict[str, Any],
        width: str = "100%",
        height: str = "600px"
    ) -> str:
        """
        Create an interactive HTML map with Leaflet.js
        
        Args:
            route_data: Route data from EnhancedMuseumRoutePlanner
            width: Map width (CSS format)
            height: Map height (CSS format)
            
        Returns:
            Complete HTML string with embedded map
        """
        
        # Extract museums from route
        museums = route_data.get('museums', [])
        if not museums:
            return self._create_empty_map_message()
        
        # Calculate map center from museum coordinates
        center = self._calculate_map_center(museums)
        
        # Generate map HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Istanbul Museum Route</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossorigin=""/>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
            integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
            crossorigin=""></script>
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        #map {{
            width: {width};
            height: {height};
        }}
        .museum-popup {{
            font-size: 14px;
            max-width: 300px;
        }}
        .museum-popup h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 16px;
        }}
        .museum-popup .info-row {{
            margin: 5px 0;
            display: flex;
            align-items: center;
        }}
        .museum-popup .icon {{
            margin-right: 8px;
            min-width: 20px;
        }}
        .museum-popup .highlight {{
            background: #fff3cd;
            padding: 8px;
            border-radius: 4px;
            margin: 8px 0;
            border-left: 3px solid #ffc107;
        }}
        .museum-popup .tip {{
            background: #d1ecf1;
            padding: 8px;
            border-radius: 4px;
            margin: 8px 0;
            border-left: 3px solid #17a2b8;
            font-size: 12px;
        }}
        .route-info {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
        }}
        .route-info h4 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        .route-info .stat {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .route-info .stat strong {{
            color: #2c3e50;
        }}
        .legend {{
            position: absolute;
            bottom: 30px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .legend-item {{
            margin: 5px 0;
            display: flex;
            align-items: center;
        }}
        .legend-marker {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="route-info">
        <h4>🗺️ Museum Route</h4>
        <div class="stat">📍 <strong>Museums:</strong> {len(museums)}</div>
        <div class="stat">⏱️ <strong>Duration:</strong> {route_data.get('total_duration_hours', 0):.1f} hours</div>
        <div class="stat">💰 <strong>Total Cost:</strong> {route_data.get('total_cost_tl', 0):.0f} TL</div>
        <div class="stat">🚶 <strong>Walking:</strong> {route_data.get('total_walking_distance_km', 0):.1f} km</div>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-marker" style="background: #e74c3c;"></div>
            <span>Start</span>
        </div>
        <div class="legend-item">
            <div class="legend-marker" style="background: #3498db;"></div>
            <span>Museums</span>
        </div>
        <div class="legend-item">
            <div class="legend-marker" style="background: #2ecc71;"></div>
            <span>End</span>
        </div>
    </div>

    <script>
        // Initialize map
        var map = L.map('map').setView([{center[0]}, {center[1]}], {self.default_zoom});
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19
        }}).addTo(map);
        
        // Museum data
        var museums = {json.dumps(self._prepare_museum_data(museums))};
        
        // Add markers for each museum
        var markers = [];
        var bounds = [];
        
        museums.forEach(function(museum, index) {{
            var isFirst = index === 0;
            var isLast = index === museums.length - 1;
            
            // Choose marker color
            var markerColor = isFirst ? '#e74c3c' : (isLast ? '#2ecc71' : '#3498db');
            
            // Create custom icon
            var icon = L.divIcon({{
                className: 'custom-marker',
                html: '<div style="background-color: ' + markerColor + '; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">' + (index + 1) + '</div>',
                iconSize: [30, 30],
                iconAnchor: [15, 15]
            }});
            
            // Create marker
            var marker = L.marker([museum.lat, museum.lng], {{icon: icon}}).addTo(map);
            
            // Create popup content
            var popupContent = `
                <div class="museum-popup">
                    <h3>${{museum.name}}</h3>
                    <div class="info-row">
                        <span class="icon">📍</span>
                        <span>${{museum.district}} - ${{museum.neighborhood}}</span>
                    </div>
                    <div class="info-row">
                        <span class="icon">🕒</span>
                        <span>Arrival: ${{museum.arrival_time}}</span>
                    </div>
                    <div class="info-row">
                        <span class="icon">⏱️</span>
                        <span>Visit Duration: ${{museum.duration}} minutes</span>
                    </div>
                    <div class="info-row">
                        <span class="icon">💰</span>
                        <span>Entry Fee: ${{museum.fee}} TL</span>
                    </div>
                    ${{museum.highlights ? '<div class="highlight">✨ <strong>Don\\'t Miss:</strong> ' + museum.highlights + '</div>' : ''}}
                    ${{museum.tip ? '<div class="tip">💡 <strong>Tip:</strong> ' + museum.tip + '</div>' : ''}}
                </div>
            `;
            
            marker.bindPopup(popupContent);
            
            // Open first museum popup by default
            if (isFirst) {{
                marker.openPopup();
            }}
            
            markers.push(marker);
            bounds.push([museum.lat, museum.lng]);
        }});
        
        // Draw route lines between museums using OSRM walking paths
        if (museums.length > 1) {{
            for (var i = 0; i < museums.length - 1; i++) {{
                // Check if we have OSRM walking path
                if (museums[i].walking_path && museums[i].walking_path.length > 0) {{
                    // Use OSRM-generated walking path
                    var polyline = L.polyline(museums[i].walking_path, {{
                        color: '#4CAF50',  // Green for walking
                        weight: 4,
                        opacity: 0.8,
                        smoothFactor: 1
                    }}).addTo(map);
                    
                    // Add tooltip showing this is a real walking path
                    polyline.bindTooltip('🚶 Realistic walking path via streets', {{
                        permanent: false,
                        direction: 'top'
                    }});
                }} else {{
                    // Fallback to straight line
                    var straightLine = L.polyline([
                        [museums[i].lat, museums[i].lng],
                        [museums[i + 1].lat, museums[i + 1].lng]
                    ], {{
                        color: '#3498db',
                        weight: 3,
                        opacity: 0.6,
                        dashArray: '10, 10'
                    }}).addTo(map);
                }}
                
                // Add distance label at midpoint
                if (museums[i].walking_distance) {{
                    var midLat = (museums[i].lat + museums[i + 1].lat) / 2;
                    var midLng = (museums[i].lng + museums[i + 1].lng) / 2;
                    
                    L.marker([midLat, midLng], {{
                        icon: L.divIcon({{
                            className: 'distance-label',
                            html: '<div style="background: white; padding: 6px 10px; border-radius: 6px; border: 2px solid #4CAF50; font-size: 12px; font-weight: bold; white-space: nowrap; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🚶 ' + museums[i].walking_distance + '</div>',
                            iconSize: [80, 24]
                        }})
                    }}).addTo(map);
                }}
            }}
        }}
        
        // Fit map to show all museums
        if (bounds.length > 0) {{
            map.fitBounds(bounds, {{padding: [50, 50]}});
        }}
    </script>
</body>
</html>
"""
        return html
    
    def _calculate_map_center(self, museums: List[Dict]) -> Tuple[float, float]:
        """Calculate center point from museum coordinates"""
        if not museums:
            return self.default_center
        
        lats = []
        lngs = []
        
        for museum_info in museums:
            museum = museum_info.get('museum')
            if museum and hasattr(museum, 'coordinates'):
                lat, lng = museum.coordinates
                lats.append(lat)
                lngs.append(lng)
        
        if not lats:
            return self.default_center
        
        center_lat = sum(lats) / len(lats)
        center_lng = sum(lngs) / len(lngs)
        
        return (center_lat, center_lng)
    
    def _prepare_museum_data(self, museums: List[Dict]) -> List[Dict]:
        """
        Prepare museum data for JavaScript with enriched database information and OSRM paths
        """
        prepared = []
        
        for i, museum_info in enumerate(museums):
            museum = museum_info.get('museum')
            if not museum:
                continue
            
            # Get coordinates
            lat, lng = museum.coordinates if hasattr(museum, 'coordinates') else (0, 0)
            museum_name = museum.name if hasattr(museum, 'name') else 'Museum'
            
            # Get enriched data from accurate museum database
            db_info = self._enrich_museum_data_from_database(museum_name)
            
            # Get top highlight
            highlights = museum.highlights[0] if hasattr(museum, 'highlights') and museum.highlights else None
            
            # If database has must-see items, use those instead
            if db_info and db_info.get('must_see'):
                highlights = db_info['must_see'][0]
            
            # Get top tip
            tip = None
            if hasattr(museum, 'local_tips') and museum.local_tips:
                top_tips = [t for t in museum.local_tips if t.importance >= 4]
                if top_tips:
                    tip = top_tips[0].description
            
            # Add best time from database if available
            if db_info and db_info.get('best_time') and not tip:
                tip = f"Best time: {db_info['best_time']}"
            
            # Walking distance and path to next museum
            walking_distance = museum_info.get('walking_distance', None)
            walking_path = []
            
            # Get realistic walking path using OSRM if available
            if i < len(museums) - 1:
                next_museum = museums[i + 1].get('museum')
                if next_museum and hasattr(next_museum, 'coordinates'):
                    next_lat, next_lng = next_museum.coordinates
                    walking_path = self._get_walking_path_coordinates(
                        (lat, lng),
                        (next_lat, next_lng)
                    )
            
            # Add verified information from database
            verified_info = {}
            if db_info:
                verified_info = {
                    'hours': db_info.get('opening_hours', {}),
                    'verified_fee': db_info.get('entrance_fee'),
                    'photography': '📸 Photos allowed' if db_info.get('photography') else '🚫 No photos',
                    'accessibility': db_info.get('accessibility'),
                    'historical_note': db_info.get('historical_significance', '')[:150] + '...' if db_info.get('historical_significance') else ''
                }
            
            prepared.append({
                'name': museum_name,
                'lat': lat,
                'lng': lng,
                'district': museum.district if hasattr(museum, 'district') else '',
                'neighborhood': museum.neighborhood if hasattr(museum, 'neighborhood') else '',
                'arrival_time': museum_info.get('arrival_time', ''),
                'duration': museum.visit_duration_minutes if hasattr(museum, 'visit_duration_minutes') else 0,
                'fee': museum.entry_fee_tl if hasattr(museum, 'entry_fee_tl') else 0,
                'highlights': highlights,
                'tip': tip,
                'walking_distance': walking_distance,
                'walking_path': walking_path,  # NEW: OSRM walking path
                'verified_info': verified_info  # NEW: Database-verified info
            })
        
        return prepared
    
    def _create_empty_map_message(self) -> str:
        """Create message for empty route"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>No Route Found</title>
    <style>
        body {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .message {
            background: white;
            padding: 40px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .message h2 {
            color: #2c3e50;
            margin: 0 0 16px 0;
        }
        .message p {
            color: #7f8c8d;
            margin: 0;
        }
    </style>
</head>
<body>
    <div class="message">
        <h2>🗺️ No Route Available</h2>
        <p>Unable to generate museum route map.</p>
    </div>
</body>
</html>
"""
    
    def create_google_maps_url(self, museums: List[Dict]) -> str:
        """
        Create a Google Maps URL with multiple waypoints
        Useful as a fallback or alternative to embedded map
        """
        if not museums or len(museums) == 0:
            return ""
        
        base_url = "https://www.google.com/maps/dir/"
        
        waypoints = []
        for museum_info in museums:
            museum = museum_info.get('museum')
            if museum and hasattr(museum, 'coordinates'):
                lat, lng = museum.coordinates
                waypoints.append(f"{lat},{lng}")
        
        if not waypoints:
            return ""
        
        # Google Maps URL with waypoints
        url = base_url + "/".join(waypoints)
        
        return url
    
    def create_summary_html(self, route_data: Dict[str, Any]) -> str:
        """Create a beautiful HTML summary of the route without map"""
        museums = route_data.get('museums', [])
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Museum Route Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            margin: 0;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #2c3e50;
            margin: 0 0 10px 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-card .label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .museum-card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #3498db;
        }}
        .museum-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        .museum-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 10px 0;
        }}
        .info-item {{
            display: flex;
            align-items: center;
            font-size: 14px;
        }}
        .info-item .icon {{
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🗺️ Your Museum Route</h1>
        <p style="color: #7f8c8d;">Optimized itinerary for exploring Istanbul's museums</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="value">{len(museums)}</div>
                <div class="label">Museums</div>
            </div>
            <div class="stat-card">
                <div class="value">{route_data.get('total_duration_hours', 0):.1f}h</div>
                <div class="label">Total Duration</div>
            </div>
            <div class="stat-card">
                <div class="value">{route_data.get('total_cost_tl', 0):.0f} TL</div>
                <div class="label">Total Cost</div>
            </div>
            <div class="stat-card">
                <div class="value">{route_data.get('total_walking_distance_km', 0):.1f} km</div>
                <div class="label">Walking Distance</div>
            </div>
        </div>
        
        {''.join(self._format_museum_card(i, m) for i, m in enumerate(museums, 1))}
    </div>
</body>
</html>
"""
        return html
    
    def _format_museum_card(self, index: int, museum_info: Dict) -> str:
        """Format a single museum card"""
        museum = museum_info.get('museum')
        if not museum:
            return ""
        
        return f"""
        <div class="museum-card">
            <h3>{index}. {museum.name if hasattr(museum, 'name') else 'Museum'}</h3>
            <div class="museum-info">
                <div class="info-item">
                    <span class="icon">📍</span>
                    <span>{museum.district if hasattr(museum, 'district') else ''} - {museum.neighborhood if hasattr(museum, 'neighborhood') else ''}</span>
                </div>
                <div class="info-item">
                    <span class="icon">🕒</span>
                    <span>Arrival: {museum_info.get('arrival_time', '')}</span>
                </div>
                <div class="info-item">
                    <span class="icon">⏱️</span>
                    <span>Duration: {museum.visit_duration_minutes if hasattr(museum, 'visit_duration_minutes') else 0} min</span>
                </div>
                <div class="info-item">
                    <span class="icon">💰</span>
                    <span>Entry: {museum.entry_fee_tl if hasattr(museum, 'entry_fee_tl') else 0} TL</span>
                </div>
            </div>
        </div>
"""

    def _get_walking_path_coordinates(
        self,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """
        Get realistic walking path coordinates using OSRM
        
        Args:
            start_coords: (lat, lon) of start point
            end_coords: (lat, lon) of end point
            
        Returns:
            List of (lat, lon) coordinates along the walking path
        """
        if not self.osrm:
            # Fallback to straight line
            return [start_coords, end_coords]
        
        try:
            # Get route from OSRM
            route = self.osrm.get_route(
                start_coords=start_coords,
                end_coords=end_coords,
                alternatives=False
            )
            
            if route and route.geometry:
                # OSRM returns coordinates, convert to our format
                return route.geometry
            
        except Exception as e:
            logger.warning(f"OSRM routing failed: {e}, using straight line")
        
        # Fallback
        return [start_coords, end_coords]
    
    def _enrich_museum_data_from_database(self, museum_name: str) -> Optional[Dict[str, Any]]:
        """
        Get enriched museum information from accurate database
        
        Args:
            museum_name: Name of the museum
            
        Returns:
            Dict with verified museum information
        """
        if not self.museum_db:
            return None
        
        # Map museum names to database keys
        name_mapping = {
            'hagia sophia': 'hagia_sophia',
            'topkapi palace': 'topkapi_palace',
            'blue mosque': 'blue_mosque',
            'basilica cistern': 'basilica_cistern',
            'archaeological museums': 'archaeology_museums',
            'istanbul modern': 'istanbul_modern',
            'pera museum': 'pera_museum',
            'turkish islamic arts': 'turkish_islamic_arts'
        }
        
        # Find matching museum in database
        museum_lower = museum_name.lower()
        for key_name, db_key in name_mapping.items():
            if key_name in museum_lower:
                museum_info = self.museum_db.museums.get(db_key)
                if museum_info:
                    return {
                        'opening_hours': museum_info.opening_hours,
                        'entrance_fee': museum_info.entrance_fee,
                        'historical_significance': museum_info.historical_significance,
                        'must_see': museum_info.must_see_highlights[:2] if hasattr(museum_info, 'must_see_highlights') else [],
                        'best_time': museum_info.best_time_to_visit,
                        'photography': museum_info.photography_allowed,
                        'accessibility': museum_info.accessibility
                    }
        
        return None
    


# Test function
if __name__ == "__main__":
    # Test with sample data
    generator = MuseumRouteMapGenerator()
    
    # Sample route data
    sample_route = {
        'museums': [
            {
                'museum': type('Museum', (), {
                    'name': 'Hagia Sophia',
                    'coordinates': (41.0086, 28.9802),
                    'district': 'Fatih',
                    'neighborhood': 'Sultanahmet',
                    'visit_duration_minutes': 90,
                    'entry_fee_tl': 200,
                    'highlights': ['Byzantine mosaics', 'Ottoman calligraphy'],
                    'local_tips': [type('Tip', (), {'importance': 5, 'description': 'Visit early morning for fewer crowds'})()]
                })(),
                'arrival_time': '09:00',
                'walking_distance': '800m'
            }
        ],
        'total_duration_hours': 5.5,
        'total_cost_tl': 600,
        'total_walking_distance_km': 2.3
    }
    
    html = generator.create_interactive_map(sample_route)
    
    # Save to file for testing
    with open('/tmp/test_museum_map.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("✅ Map generator created successfully!")
    print("Test map saved to: /tmp/test_museum_map.html")
