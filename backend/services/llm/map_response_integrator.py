"""
Map Response Integrator for AI-stanbul
Integrates map visualization data into LLM responses for route and location queries

Part of Phase 4.3+ (Multi-Intent Handler) - Map System Integration
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MapResponseData:
    """Structured map data for responses"""
    map_type: str  # 'route', 'marker', 'multi_point', 'combined'
    map_data: Dict[str, Any]
    visualization_url: Optional[str] = None
    description: str = ""
    priority: int = 1  # For multi-map scenarios


class MapResponseIntegrator:
    """
    Integrates map visualization into chatbot responses.
    
    Handles:
    - Single route queries → single route map
    - Multi-location queries → multi-point map
    - Multi-intent queries → aggregated/combined map
    - Location info queries → marker map
    """
    
    def __init__(self):
        """Initialize map response integrator"""
        self.map_service = None
        self._init_map_service()
    
    def _init_map_service(self):
        """Initialize map visualization service"""
        try:
            from services.map_visualization_service import MapVisualizationService
            self.map_service = MapVisualizationService()
            logger.info("✅ Map Response Integrator initialized with MapVisualizationService")
        except Exception as e:
            logger.error(f"Failed to initialize map service: {e}")
            self.map_service = None
    
    async def generate_map_for_intent(
        self,
        intent: Dict[str, Any],
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en"
    ) -> Optional[MapResponseData]:
        """
        Generate map data for a single intent.
        
        Args:
            intent: Intent with type and entities (from multi_intent_handler)
            user_location: User GPS coordinates
            language: Response language
            
        Returns:
            MapResponseData or None if no map needed/possible
        """
        if not self.map_service:
            logger.warning("Map service not available")
            return None
        
        intent_type = intent.get('type', '')
        entities = intent.get('entities', {})
        
        # Determine if this intent needs a map
        if intent_type not in ['route', 'info', 'transport', 'location']:
            logger.debug(f"Intent type '{intent_type}' doesn't need map visualization")
            return None
        
        try:
            # Extract query text for map generation
            query_text = entities.get('raw_text', '') or intent.get('raw_text', '')
            
            # Determine if routing is needed
            needs_routing = intent_type in ['route', 'transport']
            
            # Generate map
            map_data = await self.map_service.generate_map(
                query=query_text,
                user_location=user_location,
                language=language,
                routing=needs_routing
            )
            
            if not map_data:
                logger.debug(f"No map data generated for intent: {intent_type}")
                return None
            
            # Create description based on intent
            description = self._create_map_description(intent_type, entities, map_data)
            
            return MapResponseData(
                map_type=map_data.get('type', 'marker'),
                map_data=map_data,
                description=description,
                priority=intent.get('priority', 1)
            )
            
        except Exception as e:
            logger.error(f"Error generating map for intent: {e}")
            return None
    
    def _create_map_description(
        self,
        intent_type: str,
        entities: Dict[str, Any],
        map_data: Dict[str, Any]
    ) -> str:
        """Create human-readable description of the map"""
        
        if intent_type == 'route':
            origin = entities.get('origin', 'your location')
            destination = entities.get('destination', 'destination')
            return f"Route from {origin} to {destination}"
        
        elif intent_type == 'info' or intent_type == 'location':
            location = entities.get('location', 'location')
            return f"Location: {location}"
        
        elif intent_type == 'transport':
            return "Transportation options"
        
        return "Map visualization"
    
    async def aggregate_maps_for_multi_intent(
        self,
        intents: List[Dict[str, Any]],
        intent_responses: Dict[str, Any],
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate multiple maps into a combined visualization.
        
        For multi-intent queries like "route to Hagia Sophia and show restaurants nearby",
        this creates a combined map showing the route + restaurant markers.
        
        Args:
            intents: List of detected intents
            intent_responses: Dict of intent_type -> response data
            user_location: User GPS coordinates
            language: Response language
            
        Returns:
            Combined map data or None
        """
        if not self.map_service:
            return None
        
        map_responses = []
        
        # Generate map for each intent
        for intent in intents:
            map_response = await self.generate_map_for_intent(
                intent=intent,
                user_location=user_location,
                language=language
            )
            
            if map_response:
                map_responses.append(map_response)
        
        if not map_responses:
            logger.debug("No maps generated for any intent")
            return None
        
        # If only one map, return it directly
        if len(map_responses) == 1:
            return map_responses[0].map_data
        
        # Multiple maps: combine them
        try:
            combined_map = self._combine_maps(map_responses)
            return combined_map
        except Exception as e:
            logger.error(f"Error combining maps: {e}")
            # Fallback: return primary map
            return map_responses[0].map_data
    
    def _combine_maps(self, map_responses: List[MapResponseData]) -> Dict[str, Any]:
        """
        Combine multiple maps into one visualization.
        
        Strategy:
        - If one is a route and others are markers, overlay markers on route
        - If all are markers, create multi-marker map
        - If multiple routes, show primary route with secondary as suggestions
        """
        
        # Sort by priority
        map_responses.sort(key=lambda x: x.priority)
        
        # Find route maps vs marker maps
        route_maps = [m for m in map_responses if m.map_type == 'route']
        marker_maps = [m for m in map_responses if m.map_type == 'marker']
        
        # Case 1: Route + markers (most common for multi-intent)
        if route_maps and marker_maps:
            logger.info("Combining route with additional markers")
            return self._overlay_markers_on_route(route_maps[0], marker_maps)
        
        # Case 2: Multiple routes
        if len(route_maps) > 1:
            logger.info("Multiple routes detected, showing primary")
            # Return primary route, could be enhanced to show alternatives
            return route_maps[0].map_data
        
        # Case 3: Multiple markers
        if len(marker_maps) > 1:
            logger.info("Combining multiple marker maps")
            return self._combine_marker_maps(marker_maps)
        
        # Case 4: Single map
        return map_responses[0].map_data
    
    def _overlay_markers_on_route(
        self,
        route_map: MapResponseData,
        marker_maps: List[MapResponseData]
    ) -> Dict[str, Any]:
        """Overlay additional markers on a route map"""
        
        combined = route_map.map_data.copy()
        
        # Add additional markers
        existing_markers = combined.get('markers', [])
        
        for marker_map in marker_maps:
            additional_markers = marker_map.map_data.get('markers', [])
            
            # Add markers with different type to distinguish them
            for marker in additional_markers:
                marker_copy = marker.copy()
                marker_copy['type'] = f"secondary_{marker_map.map_type}"
                existing_markers.append(marker_copy)
        
        combined['markers'] = existing_markers
        combined['type'] = 'combined'
        combined['description'] = f"{route_map.description} with {len(marker_maps)} additional points"
        
        return combined
    
    def _combine_marker_maps(self, marker_maps: List[MapResponseData]) -> Dict[str, Any]:
        """Combine multiple marker maps into one"""
        
        all_markers = []
        all_lats = []
        all_lngs = []
        
        for marker_map in marker_maps:
            markers = marker_map.map_data.get('markers', [])
            all_markers.extend(markers)
            
            # Collect coordinates for center calculation
            for marker in markers:
                pos = marker.get('position', {})
                if 'lat' in pos and 'lng' in pos:
                    all_lats.append(pos['lat'])
                    all_lngs.append(pos['lng'])
        
        # Calculate center
        center = {
            'lat': sum(all_lats) / len(all_lats) if all_lats else 41.0082,
            'lng': sum(all_lngs) / len(all_lngs) if all_lngs else 28.9784
        }
        
        return {
            'type': 'multi_point',
            'markers': all_markers,
            'center': center,
            'zoom': 13,
            'description': f"Map with {len(all_markers)} locations"
        }
    
    def format_map_for_response(
        self,
        map_data: Optional[Dict[str, Any]],
        include_description: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Format map data for inclusion in chat response.
        
        Args:
            map_data: Raw map data
            include_description: Whether to include text description
            
        Returns:
            Formatted map data ready for frontend
        """
        if not map_data:
            return None
        
        formatted = {
            'type': map_data.get('type', 'marker'),
            'center': map_data.get('center', {}),
            'zoom': map_data.get('zoom', 13),
            'markers': map_data.get('markers', []),
        }
        
        # Add route data if present
        if 'route' in map_data:
            formatted['route'] = map_data['route']
        
        # Add description if requested
        if include_description and 'description' in map_data:
            formatted['description'] = map_data['description']
        
        # Add metadata
        formatted['metadata'] = {
            'has_origin': map_data.get('has_origin', False),
            'has_destination': map_data.get('has_destination', False),
            'origin_name': map_data.get('origin_name'),
            'destination_name': map_data.get('destination_name')
        }
        
        return formatted


# Singleton instance
_map_response_integrator = None


def get_map_response_integrator() -> MapResponseIntegrator:
    """Get or create the map response integrator singleton"""
    global _map_response_integrator
    
    if _map_response_integrator is None:
        _map_response_integrator = MapResponseIntegrator()
        logger.info("✅ Map Response Integrator singleton created")
    
    return _map_response_integrator
