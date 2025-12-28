"""
Enhanced Map Visualization Service - Moovit-Style Multi-Route Display
=====================================================================

Enhanced map service that integrates with the route optimizer to display
multiple route alternatives with visual comparison.

Features:
✅ Multi-route visualization with color coding
✅ Route comparison overlay (fastest, best, least transfers, etc.)
✅ Comfort score visualization
✅ Transfer point markers with quality indicators
✅ Walking segment visualization
✅ Peak hour crowding indicators
✅ Interactive route selection

Author: AI Istanbul Team
Date: January 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Color scheme for different route types
ROUTE_COLORS = {
    'fastest': '#FF4444',        # Red - fast/urgent
    'best': '#4CAF50',           # Green - recommended
    'least_transfers': '#2196F3', # Blue - simple
    'least_walking': '#FF9800',  # Orange - comfort
    'most_comfortable': '#9C27B0', # Purple - premium
    'accessible': '#00BCD4'      # Cyan - accessibility
}

# Transfer quality colors
TRANSFER_QUALITY_COLORS = {
    'excellent': '#4CAF50',  # Green
    'good': '#8BC34A',       # Light green
    'fair': '#FFC107',       # Yellow
    'poor': '#FF9800',       # Orange
    'difficult': '#F44336'   # Red
}


class EnhancedMapVisualizationService:
    """
    Enhanced map visualization service with Moovit-style multi-route display.
    
    Integrates with:
    - Route optimizer for multi-route alternatives
    - Transportation RAG for route data
    - Comfort scorer for quality indicators
    """
    
    def __init__(self):
        """Initialize enhanced map visualization service"""
        # Import route integration
        try:
            from services.transportation_route_integration import get_route_integration
            self.route_integration = get_route_integration()
            logger.info("✅ Enhanced Map Visualization Service initialized with Route Integration")
        except Exception as e:
            logger.error(f"Failed to initialize Route Integration: {e}")
            self.route_integration = None
        
        # Import transportation RAG for map data
        try:
            from services.transportation_rag_system import get_transportation_rag
            self.transport_rag = get_transportation_rag()
            logger.info("✅ Transportation RAG integrated")
        except Exception as e:
            logger.warning(f"Transportation RAG not available: {e}")
            self.transport_rag = None
    
    async def generate_multi_route_map(
        self,
        origin: str,
        destination: str,
        origin_gps: Optional[Dict[str, float]] = None,
        destination_gps: Optional[Dict[str, float]] = None,
        num_alternatives: int = 3,
        user_language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Generate Moovit-style multi-route map visualization.
        
        Args:
            origin: Origin location name
            destination: Destination location name
            origin_gps: Optional origin GPS coordinates
            destination_gps: Optional destination GPS coordinates
            num_alternatives: Number of route alternatives to display
            user_language: User's preferred language
            
        Returns:
            Enhanced map data with multiple routes:
            {
                'type': 'multi_route',
                'routes': [
                    {
                        'id': 'route_1',
                        'preference': 'best',
                        'color': '#4CAF50',
                        'highlighted': True,
                        'segments': [...],
                        'markers': [...],
                        'stats': {...}
                    },
                    ...
                ],
                'comparison': {...},
                'center': {'lat': ..., 'lng': ...},
                'zoom': 12,
                'bounds': {'north': ..., 'south': ..., 'east': ..., 'west': ...}
            }
        """
        if not self.route_integration:
            logger.error("Route integration not available")
            return self._error_map_data("Route integration not available")
        
        logger.info(f"🗺️ Generating multi-route map: {origin} → {destination}")
        
        # Get route alternatives
        route_result = self.route_integration.get_route_alternatives(
            origin=origin,
            destination=destination,
            origin_gps=origin_gps,
            destination_gps=destination_gps,
            num_alternatives=num_alternatives,
            generate_llm_summaries=False,  # Not needed for map
            user_language=user_language
        )
        
        if not route_result['success']:
            logger.error(f"Failed to get route alternatives: {route_result.get('error')}")
            return self._error_map_data(route_result.get('error', 'Route not found'))
        
        # Build multi-route map data
        map_data = {
            'type': 'multi_route',
            'routes': [],
            'comparison': route_result.get('route_comparison', {}),
            'metadata': {
                'origin': origin,
                'destination': destination,
                'num_routes': len(route_result.get('alternatives', [])),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        # Convert each alternative to map route
        alternatives = route_result.get('alternatives', [])
        for i, alt in enumerate(alternatives):
            route_viz = self._convert_route_to_map_format(
                route_option=alt,
                route_index=i,
                is_primary=(i == 0)  # First route is highlighted
            )
            map_data['routes'].append(route_viz)
        
        # Calculate map bounds and center
        map_data['bounds'] = self._calculate_map_bounds(map_data['routes'])
        map_data['center'] = self._calculate_map_center(map_data['bounds'])
        map_data['zoom'] = self._calculate_optimal_zoom(map_data['bounds'])
        
        logger.info(f"✅ Generated multi-route map with {len(map_data['routes'])} routes")
        return map_data
    
    def _convert_route_to_map_format(
        self,
        route_option: Dict[str, Any],
        route_index: int,
        is_primary: bool = False
    ) -> Dict[str, Any]:
        """
        Convert RouteOption to map visualization format.
        
        Returns:
            {
                'id': 'route_1',
                'preference': 'best',
                'color': '#4CAF50',
                'highlighted': True,
                'opacity': 0.8,
                'segments': [
                    {
                        'type': 'metro',
                        'line': 'M2',
                        'color': '#00AA00',
                        'coordinates': [[lat, lng], ...],
                        'duration': 15,
                        'stops': 8
                    },
                    {
                        'type': 'transfer',
                        'location': 'Yenikapı',
                        'quality': 'good',
                        'color': '#8BC34A',
                        'duration': 3
                    },
                    ...
                ],
                'markers': [
                    {
                        'type': 'origin',
                        'position': {'lat': ..., 'lng': ...},
                        'label': 'Start'
                    },
                    {
                        'type': 'transfer',
                        'position': {'lat': ..., 'lng': ...},
                        'label': 'Transfer at Yenikapı',
                        'quality': 'good',
                        'icon': 'transfer_good'
                    },
                    ...
                ],
                'stats': {
                    'duration': 23,
                    'transfers': 1,
                    'walking': 300,
                    'comfort': 85
                }
            }
        """
        preference = route_option.get('preference', 'best')
        route = route_option.get('route', {})
        
        # Get route color
        color = ROUTE_COLORS.get(preference, '#4CAF50')
        
        # Build segments from route steps
        segments = []
        markers = []
        
        route_steps = route.get('steps', [])
        for i, step in enumerate(route_steps):
            segment = self._convert_step_to_segment(step, i)
            if segment:
                segments.append(segment)
            
            # Add marker for significant points
            marker = self._create_marker_for_step(step, i, len(route_steps))
            if marker:
                markers.append(marker)
        
        return {
            'id': f'route_{route_index + 1}',
            'preference': preference,
            'color': color,
            'highlighted': is_primary,
            'opacity': 0.9 if is_primary else 0.6,
            'weight': 5 if is_primary else 3,
            'segments': segments,
            'markers': markers,
            'stats': {
                'duration': route_option.get('duration_minutes', 0),
                'transfers': route_option.get('num_transfers', 0),
                'walking': route_option.get('walking_meters', 0),
                'comfort': route_option.get('comfort_score', {}).get('overall_comfort', 0)
            },
            'highlights': route_option.get('highlights', []),
            'warnings': route_option.get('warnings', [])
        }
    
    def _convert_step_to_segment(self, step: Dict[str, Any], step_index: int) -> Optional[Dict[str, Any]]:
        """Convert a route step to a map segment"""
        step_mode = step.get('mode', 'walk')
        
        # Get segment color based on mode
        mode_colors = {
            'metro': '#00AA00',
            'marmaray': '#0066CC',
            'tram': '#CC0000',
            'ferry': '#0099CC',
            'bus': '#FF6600',
            'funicular': '#9966CC',
            'walk': '#666666'
        }
        
        segment = {
            'type': step_mode,
            'color': mode_colors.get(step_mode, '#666666'),
            'duration': step.get('duration', 0),
            'distance': step.get('distance', 0)
        }
        
        # Add mode-specific details
        if step_mode != 'walk':
            segment['line'] = step.get('line_name', 'Unknown')
            segment['stops'] = step.get('stops_count', 0)
        
        # Add coordinates if available
        if step.get('waypoints'):
            segment['coordinates'] = step['waypoints']
        elif step.get('start_location') and step.get('end_location'):
            segment['coordinates'] = [
                step['start_location'],
                step['end_location']
            ]
        
        return segment
    
    def _create_marker_for_step(
        self,
        step: Dict[str, Any],
        step_index: int,
        total_steps: int
    ) -> Optional[Dict[str, Any]]:
        """Create a marker for important points in the route"""
        
        # Origin marker (first step)
        if step_index == 0:
            return {
                'type': 'origin',
                'position': step.get('start_location', [0, 0]),
                'label': 'Start',
                'icon': 'origin',
                'color': '#4CAF50'
            }
        
        # Destination marker (last step)
        if step_index == total_steps - 1:
            return {
                'type': 'destination',
                'position': step.get('end_location', [0, 0]),
                'label': 'Destination',
                'icon': 'destination',
                'color': '#F44336'
            }
        
        # Transfer marker (non-walk steps in the middle)
        if step.get('mode') != 'walk' and step_index > 0:
            # This is likely a transfer point
            return {
                'type': 'transfer',
                'position': step.get('start_location', [0, 0]),
                'label': f"Transfer to {step.get('line_name', 'transit')}",
                'icon': 'transfer',
                'color': '#FF9800'
            }
        
        return None
    
    def _calculate_map_bounds(self, routes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate bounding box for all routes"""
        all_coords = []
        
        for route in routes:
            for segment in route.get('segments', []):
                coords = segment.get('coordinates', [])
                all_coords.extend(coords)
        
        if not all_coords:
            # Default to Istanbul center
            return {
                'north': 41.1,
                'south': 40.9,
                'east': 29.1,
                'west': 28.8
            }
        
        lats = [coord[0] for coord in all_coords if len(coord) >= 2]
        lngs = [coord[1] for coord in all_coords if len(coord) >= 2]
        
        if not lats or not lngs:
            return {
                'north': 41.1,
                'south': 40.9,
                'east': 29.1,
                'west': 28.8
            }
        
        return {
            'north': max(lats),
            'south': min(lats),
            'east': max(lngs),
            'west': min(lngs)
        }
    
    def _calculate_map_center(self, bounds: Dict[str, float]) -> Dict[str, float]:
        """Calculate center point of map bounds"""
        return {
            'lat': (bounds['north'] + bounds['south']) / 2,
            'lng': (bounds['east'] + bounds['west']) / 2
        }
    
    def _calculate_optimal_zoom(self, bounds: Dict[str, float]) -> int:
        """Calculate optimal zoom level based on bounds"""
        lat_diff = bounds['north'] - bounds['south']
        lng_diff = bounds['east'] - bounds['west']
        
        # Larger difference = zoom out more
        max_diff = max(lat_diff, lng_diff)
        
        if max_diff > 0.5:
            return 10
        elif max_diff > 0.2:
            return 11
        elif max_diff > 0.1:
            return 12
        elif max_diff > 0.05:
            return 13
        else:
            return 14
    
    def _error_map_data(self, error_message: str) -> Dict[str, Any]:
        """Generate error map data"""
        return {
            'type': 'error',
            'error': error_message,
            'routes': [],
            'center': {'lat': 41.0082, 'lng': 28.9784},  # Istanbul center
            'zoom': 11
        }
    
    async def generate_single_route_map(
        self,
        origin: str,
        destination: str,
        origin_gps: Optional[Dict[str, float]] = None,
        destination_gps: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate traditional single-route map (backward compatibility).
        
        Uses the transportation RAG system for single route display.
        """
        if not self.transport_rag:
            return self._error_map_data("Transportation RAG not available")
        
        logger.info(f"🗺️ Generating single route map: {origin} → {destination}")
        
        # Get route from RAG
        route = self.transport_rag.find_route(
            origin=origin,
            destination=destination,
            origin_gps=origin_gps,
            destination_gps=destination_gps
        )
        
        if not route:
            return self._error_map_data("Route not found")
        
        # Get map data from RAG
        map_data = self.transport_rag.get_map_data_for_last_route()
        
        if not map_data:
            return self._error_map_data("Map data not available")
        
        return map_data


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_enhanced_map_service_instance = None


def get_enhanced_map_service() -> EnhancedMapVisualizationService:
    """Get or create singleton EnhancedMapVisualizationService instance"""
    global _enhanced_map_service_instance
    if _enhanced_map_service_instance is None:
        _enhanced_map_service_instance = EnhancedMapVisualizationService()
        logger.info("✅ EnhancedMapVisualizationService initialized")
    return _enhanced_map_service_instance


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Test the enhanced map visualization service
    """
    import asyncio
    import json
    
    async def test_multi_route_map():
        print("🗺️ Testing Enhanced Map Visualization Service\n")
        
        service = get_enhanced_map_service()
        
        # Test multi-route map
        map_data = await service.generate_multi_route_map(
            origin="Taksim",
            destination="Kadıköy",
            origin_gps={"lat": 41.0370, "lon": 28.9850},
            destination_gps={"lat": 40.9900, "lon": 29.0250},
            num_alternatives=3
        )
        
        if map_data['type'] == 'multi_route':
            print("✅ Multi-route map generated successfully!\n")
            print(f"📍 {map_data['metadata']['origin']} → {map_data['metadata']['destination']}")
            print(f"🔢 Routes: {map_data['metadata']['num_routes']}\n")
            
            for route in map_data['routes']:
                print(f"Route {route['id']} - {route['preference'].upper()}")
                print(f"   Color: {route['color']}")
                print(f"   Highlighted: {route['highlighted']}")
                print(f"   Duration: {route['stats']['duration']} min")
                print(f"   Transfers: {route['stats']['transfers']}")
                print(f"   Comfort: {route['stats']['comfort']:.0f}/100")
                print(f"   Segments: {len(route['segments'])}")
                print(f"   Markers: {len(route['markers'])}")
                if route['highlights']:
                    print(f"   Highlights: {', '.join(route['highlights'])}")
                print()
            
            print(f"📐 Map Bounds:")
            print(f"   North: {map_data['bounds']['north']:.4f}")
            print(f"   South: {map_data['bounds']['south']:.4f}")
            print(f"   East: {map_data['bounds']['east']:.4f}")
            print(f"   West: {map_data['bounds']['west']:.4f}")
            print(f"   Center: {map_data['center']}")
            print(f"   Zoom: {map_data['zoom']}")
        else:
            print(f"❌ Error: {map_data.get('error')}")
    
    # Run test
    asyncio.run(test_multi_route_map())
