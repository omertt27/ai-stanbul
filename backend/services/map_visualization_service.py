"""
Map Visualization Service
=========================

Service for generating map visualization data for the frontend.
Integrates with TransportationDirectionsService to provide routes and markers.

Features:
- GPS-based routing from user location
- Map data generation for frontend visualization
- Route optimization
- Marker generation for POIs
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class MapVisualizationService:
    """Service for generating map visualization data"""
    
    def __init__(self):
        """Initialize map visualization service"""
        # Import transportation service
        try:
            from .transportation_directions_service import TransportationDirectionsService
            self.transportation_service = TransportationDirectionsService()
            logger.info("✅ Map Visualization Service initialized with Transportation Service")
        except Exception as e:
            logger.error(f"Failed to initialize Transportation Service: {e}")
            self.transportation_service = None
    
    async def generate_map(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en",
        routing: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Generate map visualization data based on query and user location.
        
        Args:
            query: User query
            user_location: User GPS location {"lat": float, "lon": float}
            language: Response language
            routing: Whether to generate a route (True for GPS routing)
            
        Returns:
            Dict with map data:
            {
                "type": "route" | "marker",
                "route": {...} if routing,
                "markers": [...],
                "center": {"lat": float, "lng": float},
                "zoom": int
            }
        """
        # Parse destination from query
        destination = self._extract_destination_from_query(query)
        
        # If destination is None, it means user specified both origin and destination
        # In this case, let the LLM handle it without GPS routing
        if destination is None:
            logger.info("Query specifies explicit origin and destination - skipping GPS routing")
            return None
        
        if not destination:
            logger.warning(f"Could not extract destination from query: {query}")
            return None
        
        # Check if user has GPS location for routing
        if not user_location:
            logger.warning("No user location provided for map generation")
            return None
        
        # Get destination coordinates
        destination_coords = self._get_destination_coordinates(destination)
        
        if not destination_coords:
            logger.warning(f"Could not find coordinates for destination: {destination}")
            return None
        
        # Generate map data with GPS-based routing
        if routing and self.transportation_service:
            return await self._generate_route_map(
                user_location,
                destination_coords,
                destination,
                language
            )
        else:
            return self._generate_marker_map(destination_coords, destination)
    
    async def _generate_route_map(
        self,
        start_location: Dict[str, float],
        end_coords: Tuple[float, float],
        end_name: str,
        language: str
    ) -> Dict[str, Any]:
        """Generate route map data"""
        start = (start_location["lat"], start_location["lon"])
        
        try:
            # Get directions from transportation service
            route = self.transportation_service.get_directions(
                start=start,
                end=end_coords,
                start_name="Your Location",
                end_name=end_name
            )
            
            if not route:
                logger.warning("No route found")
                return None
            
            # Convert route to map data format
            waypoints = []
            markers = []
            
            # Add start marker
            markers.append({
                "position": {"lat": start[0], "lng": start[1]},
                "label": "Start",
                "type": "start"
            })
            
            # Process route steps
            for i, step in enumerate(route.steps):
                # Add waypoints
                if step.waypoints:
                    for wp in step.waypoints:
                        waypoints.append({"lat": wp[0], "lng": wp[1]})
                
                # Add step markers (for transit stations)
                if step.mode in ['metro', 'tram', 'bus', 'ferry'] and step.line_name:
                    markers.append({
                        "position": {"lat": step.start_location[0], "lng": step.start_location[1]},
                        "label": step.line_name,
                        "type": "transit"
                    })
            
            # Add end marker
            markers.append({
                "position": {"lat": end_coords[0], "lng": end_coords[1]},
                "label": end_name,
                "type": "end"
            })
            
            # Calculate center and zoom
            all_lats = [start[0], end_coords[0]]
            all_lngs = [start[1], end_coords[1]]
            center = {
                "lat": sum(all_lats) / len(all_lats),
                "lng": sum(all_lngs) / len(all_lngs)
            }
            
            # Generate route summary
            summary = {
                "distance": f"{route.total_distance / 1000:.1f} km",
                "duration": f"{route.total_duration} min",
                "modes": route.modes_used or []
            }
            
            return {
                "type": "route",
                "route": {
                    "waypoints": waypoints,
                    "summary": summary,
                    "steps": [
                        {
                            "instruction": step.instruction,
                            "mode": step.mode,
                            "distance": f"{step.distance / 1000:.1f} km" if step.distance > 1000 else f"{step.distance:.0f} m",
                            "duration": f"{step.duration} min"
                        }
                        for step in route.steps
                    ]
                },
                "markers": markers,
                "center": center,
                "zoom": 13
            }
            
        except Exception as e:
            logger.error(f"Error generating route map: {e}")
            return None
    
    def _generate_marker_map(
        self,
        coords: Tuple[float, float],
        name: str
    ) -> Dict[str, Any]:
        """Generate simple marker map data"""
        return {
            "type": "marker",
            "markers": [
                {
                    "position": {"lat": coords[0], "lng": coords[1]},
                    "label": name,
                    "type": "destination"
                }
            ],
            "center": {"lat": coords[0], "lng": coords[1]},
            "zoom": 15
        }
    
    def _extract_destination_from_query(self, query: str) -> Optional[str]:
        """Extract destination from user query"""
        query_lower = query.lower()
        
        # Check if query explicitly mentions both origin and destination
        # Patterns like: "from X to Y", "X to Y", "from X how to get to Y"
        from_to_patterns = [
            r'from\s+(.+?)\s+to\s+(.+?)[\?\.!]?$',
            r'from\s+(.+?)\s+how.*to.*get.*to\s+(.+?)[\?\.!]?$',
            r'(.+?)\s+to\s+(.+?)[\?\.!]?$',
        ]
        
        import re
        for pattern in from_to_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # User specified both origin and destination
                # Return None to signal that GPS should NOT be used
                # The LLM will handle this case directly
                origin = match.group(1).strip()
                destination = match.group(2).strip()
                
                # Check if origin is a known landmark (not "here", "my location", etc.)
                location_words = ['here', 'my location', 'current location', 'where i am', 'this place']
                if not any(word in origin for word in location_words):
                    logger.info(f"Query specifies both origin ('{origin}') and destination ('{destination}')")
                    # Return None to indicate GPS should be ignored
                    return None
        
        # Common patterns for destination-only queries
        patterns = [
            ("how to get to ", ""),
            ("how do i get to ", ""),
            ("directions to ", ""),
            ("take me to ", ""),
            ("navigate to ", ""),
            ("route to ", ""),
            ("way to ", ""),
            ("go to ", ""),
            ("get to ", ""),
        ]
        
        for pattern_start, pattern_end in patterns:
            if pattern_start in query_lower:
                idx = query_lower.index(pattern_start) + len(pattern_start)
                destination = query[idx:].strip()
                # Remove trailing questions and punctuation
                destination = destination.rstrip('?!.').strip()
                return destination
        
        # If no pattern matched, try to find known landmarks
        landmarks = self._get_known_landmarks()
        for landmark, coords in landmarks.items():
            if landmark.lower() in query_lower:
                return landmark
        
        return None
    
    def _get_destination_coordinates(self, destination: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a destination"""
        destination_lower = destination.lower()
        
        # Known landmarks and attractions in Istanbul
        landmarks = self._get_known_landmarks()
        
        # Try exact match first
        for name, coords in landmarks.items():
            if destination_lower == name.lower():
                return coords
        
        # Try partial match
        for name, coords in landmarks.items():
            if destination_lower in name.lower() or name.lower() in destination_lower:
                return coords
        
        return None
    
    def _get_known_landmarks(self) -> Dict[str, Tuple[float, float]]:
        """Get dictionary of known landmarks and their coordinates"""
        return {
            # Major Attractions
            "Hagia Sophia": (41.0086, 28.9802),
            "Blue Mosque": (41.0054, 28.9768),
            "Sultanahmet Mosque": (41.0054, 28.9768),
            "Topkapi Palace": (41.0115, 28.9833),
            "Grand Bazaar": (41.0108, 28.9680),
            "Spice Bazaar": (41.0166, 28.9709),
            "Galata Tower": (41.0256, 28.9744),
            "Basilica Cistern": (41.0084, 28.9778),
            "Dolmabahce Palace": (41.0391, 29.0000),
            
            # Neighborhoods/Districts
            "Taksim": (41.0370, 28.9857),
            "Taksim Square": (41.0370, 28.9857),
            "Sultanahmet": (41.0086, 28.9802),
            "Kadikoy": (40.9900, 29.0244),
            "Besiktas": (41.0422, 29.0061),
            "Beyoglu": (41.0315, 28.9754),
            "Ortakoy": (41.0553, 29.0297),
            "Eminonu": (41.0173, 28.9705),
            "Sisli": (41.0602, 28.9875),
            "Nisantasi": (41.0498, 28.9935),
            
            # Transportation Hubs
            "Istanbul Airport": (41.2753, 28.7519),
            "Sabiha Gokcen Airport": (40.8986, 29.3092),
            "Yenikapi": (41.0035, 28.9510),
            "Sirkeci": (41.0171, 28.9769),
            
            # Shopping
            "Istiklal Avenue": (41.0330, 28.9785),
            "Istiklal Street": (41.0330, 28.9785),
            
            # Bosphorus
            "Bosphorus Bridge": (41.0442, 29.0217),
            "Maiden's Tower": (41.0211, 29.0043),
            
            # Mosques
            "Suleymaniye Mosque": (41.0164, 28.9644),
            "Ortakoy Mosque": (41.0553, 29.0297),
            "Eyup Sultan Mosque": (41.0470, 28.9347),
        }
