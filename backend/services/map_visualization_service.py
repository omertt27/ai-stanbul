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

import re
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
                "zoom": int,
                "has_origin": bool,  # Whether origin was extracted from query
                "has_destination": bool,  # Whether destination was extracted from query
                "origin_name": str,  # Name of origin (if extracted)
                "destination_name": str  # Name of destination (if extracted)
            }
        """
        # Try to extract both origin and destination from query
        origin, destination = self._extract_locations_from_query(query)
        
        # Case 1: Both origin and destination specified (e.g., "Taksim to Kadıköy")
        if origin and destination:
            origin_coords = self._get_destination_coordinates(origin)
            dest_coords = self._get_destination_coordinates(destination)
            
            if origin_coords and dest_coords:
                logger.info(f"Generating route map: {origin} → {destination}")
                map_data = await self._generate_route_map_from_coords(
                    origin_coords,
                    dest_coords,
                    origin,
                    destination,
                    language
                )
                if map_data:
                    # Add location extraction flags
                    map_data['has_origin'] = True
                    map_data['has_destination'] = True
                    map_data['origin_name'] = origin
                    map_data['destination_name'] = destination
                return map_data
        
        # Case 2: Only destination specified, use user GPS location as origin
        if destination and user_location:
            dest_coords = self._get_destination_coordinates(destination)
            if dest_coords:
                logger.info(f"Generating GPS route map: user location → {destination}")
                map_data = await self._generate_route_map(
                    user_location,
                    dest_coords,
                    destination,
                    language
                )
                if map_data:
                    # GPS used as origin
                    map_data['has_origin'] = True  # GPS counts as origin
                    map_data['has_destination'] = True
                    map_data['origin_name'] = "Your Location"
                    map_data['destination_name'] = destination
                return map_data
        
        # Case 3: Just show destination marker (no route, no GPS)
        if destination:
            dest_coords = self._get_destination_coordinates(destination)
            if dest_coords:
                logger.info(f"Generating marker map for: {destination}")
                map_data = self._generate_marker_map(dest_coords, destination)
                # Only destination, no origin
                map_data['has_origin'] = False
                map_data['has_destination'] = True
                map_data['origin_name'] = None
                map_data['destination_name'] = destination
                return map_data
        
        logger.warning(f"Could not generate map for query: {query}")
        return None
    
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
    
    def _extract_locations_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract both origin and destination from user query.
        
        Handles patterns:
        - "from X to Y" / "X to Y"
        - "to Y from X" / "go to Y from X" (reversed order)
        - "how can I go to Y from X"
        
        Returns:
            Tuple of (origin, destination) - either can be None
        """
        query_lower = query.lower()
        
        import re
        
        # PATTERN 1: "to Y from X" - destination before origin (most common in natural speech)
        # Examples: "how can I go to taksim from kadikoy", "to sultanahmet from taksim"
        # Fixed: Use negative lookahead (?!from) instead of character class [^from]
        to_from_patterns = [
            r'(?:how\s+(?:can|do)\s+i\s+)?(?:go\s+)?to\s+((?:(?!\bfrom\b).)+?)\s+from\s+(.+?)(?:\s*[\?\.!]|$)',
            r'(?:get|travel|walk)\s+to\s+((?:(?!\bfrom\b).)+?)\s+from\s+(.+?)(?:\s*[\?\.!]|$)',
            r'(?:directions|route)\s+to\s+((?:(?!\bfrom\b).)+?)\s+from\s+(.+?)(?:\s*[\?\.!]|$)',
        ]
        
        for pattern in to_from_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                destination = match.group(1).strip()
                origin = match.group(2).strip()
                
                # Clean common noise words
                destination = self._clean_location_name(destination)
                origin = self._clean_location_name(origin)
                
                logger.info(f"✅ Extracted (to-from): origin='{origin}', destination='{destination}'")
                return (origin, destination)
        
        # PATTERN 2: "from X to Y" - traditional order
        # Examples: "from kadikoy to taksim", "route from X to Y"
        # Fixed: Use negative lookahead (?!to) instead of character class [^to]
        from_to_patterns = [
            r'from\s+((?:(?!\bto\b).)+?)\s+to\s+(.+?)(?:\s*[\?\.!]|$)',
            r'(?:route|directions|path)\s+from\s+((?:(?!\bto\b).)+?)\s+to\s+(.+?)(?:\s*[\?\.!]|$)',
            r'(?:going|traveling|walking)\s+from\s+((?:(?!\bto\b).)+?)\s+to\s+(.+?)(?:\s*[\?\.!]|$)',
        ]
        
        for pattern in from_to_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                origin = match.group(1).strip()
                destination = match.group(2).strip()
                
                origin = self._clean_location_name(origin)
                destination = self._clean_location_name(destination)
                
                logger.info(f"✅ Extracted (from-to): origin='{origin}', destination='{destination}'")
                return (origin, destination)
        
        # PATTERN 3: Turkish patterns
        # "X'den Y'ye nasıl giderim"
        tr_patterns = [
            (r'(.+?)(?:\'?den|\'?dan)\s+(.+?)(?:\'?ye|\'?ya|\'?e|\'?a)\s+nas[ıi]l', 'tr'),
            (r'(.+?)(?:\'?den|\'?dan)\s+(.+?)(?:\'?ye|\'?ya)\s+git', 'tr'),
        ]
        
        for pattern, lang in tr_patterns:
            match = re.search(pattern, query_lower)
            if match:
                origin = match.group(1).strip()
                destination = match.group(2).strip()
                
                # Clean Turkish suffixes
                origin = re.sub(r'\'?(den|dan)$', '', origin).strip()
                destination = re.sub(r'\'?(ye|ya|e|a)$', '', destination).strip()
                
                logger.info(f"✅ Extracted (Turkish): origin='{origin}', destination='{destination}'")
                return (origin, destination)
        
        # If no "from X to Y" pattern, try destination-only patterns
        destination = self._extract_destination_only(query)
        if destination:
            logger.info(f"ℹ️ Extracted destination only: '{destination}' (no origin specified)")
            return (None, destination)
        
        return (None, None)
    
    def _clean_location_name(self, location: str) -> str:
        """Clean location name by removing noise words"""
        # Remove common noise words
        noise_words = ['the', 'a', 'an', 'my', 'your', 'our', 'this', 'that',
                      'here', 'there', 'at', 'in', 'on', 'near', 'around']
        
        words = location.split()
        if len(words) > 1:
            words = [w for w in words if w not in noise_words]
        
        return ' '.join(words).strip()
    
    def _extract_destination_only(self, query: str) -> Optional[str]:
        """Extract only destination from queries like 'how to get to X'"""
        query_lower = query.lower()
        
        # English patterns
        en_patterns = [
            "how to get to ",
            "how do i get to ",
            "directions to ",
            "take me to ",
            "navigate to ",
            "route to ",
            "way to ",
            "go to ",
            "get to ",
        ]
        
        # Turkish patterns
        tr_patterns = [
            "nasıl giderim ",
            "nasıl gidilir ",
            "yol tarifi ",
            "götür ",
        ]
        
        for pattern in en_patterns + tr_patterns:
            if pattern in query_lower:
                idx = query_lower.index(pattern) + len(pattern)
                destination = query[idx:].strip()
                # Remove trailing punctuation
                destination = destination.rstrip('?!.').strip()
                # Remove Turkish suffixes
                destination = re.sub(r'\'?[ye|ya|e|a]$', '', destination).strip()
                return destination
        
        # Try to find known landmarks
        landmarks = self._get_known_landmarks()
        for landmark in landmarks.keys():
            if landmark.lower() in query_lower:
                return landmark
        
        return None
    
    async def _generate_route_map_from_coords(
        self,
        origin_coords: Tuple[float, float],
        dest_coords: Tuple[float, float],
        origin_name: str,
        dest_name: str,
        language: str
    ) -> Dict[str, Any]:
        """Generate route map from two coordinate points"""
        if not self.transportation_service:
            # Fallback: show both as markers
            return {
                "type": "markers",
                "markers": [
                    {
                        "lat": origin_coords[0],
                        "lng": origin_coords[1],
                        "title": origin_name,
                        "type": "origin"
                    },
                    {
                        "lat": dest_coords[0],
                        "lng": dest_coords[1],
                        "title": dest_name,
                        "type": "destination"
                    }
                ],
                "center": {
                    "lat": (origin_coords[0] + dest_coords[0]) / 2,
                    "lng": (origin_coords[1] + dest_coords[1]) / 2
                },
                "zoom": 12
            }
        
        try:
            # Get route from transportation service
            route_data = await self.transportation_service.get_route(
                start_lat=origin_coords[0],
                start_lon=origin_coords[1],
                end_lat=dest_coords[0],
                end_lon=dest_coords[1],
                language=language
            )
            
            if route_data:
                return {
                    "type": "route",
                    "route": route_data,
                    "origin": {"lat": origin_coords[0], "lng": origin_coords[1], "name": origin_name},
                    "destination": {"lat": dest_coords[0], "lng": dest_coords[1], "name": dest_name},
                    "center": {
                        "lat": (origin_coords[0] + dest_coords[0]) / 2,
                        "lng": (origin_coords[1] + dest_coords[1]) / 2
                    },
                    "zoom": 13
                }
        except Exception as e:
            logger.error(f"Route generation failed: {e}")
        
        # Fallback to markers
        return {
            "type": "markers",
            "markers": [
                {"lat": origin_coords[0], "lng": origin_coords[1], "title": origin_name, "type": "origin"},
                {"lat": dest_coords[0], "lng": dest_coords[1], "title": dest_name, "type": "destination"}
            ],
            "center": {
                "lat": (origin_coords[0] + dest_coords[0]) / 2,
                "lng": (origin_coords[1] + dest_coords[1]) / 2
            },
            "zoom": 12
        }
    
    def _extract_destination_from_query(self, query: str) -> Optional[str]:
        """DEPRECATED: Use _extract_locations_from_query instead"""
        _, destination = self._extract_locations_from_query(query)
        return destination
    
    def _get_destination_coordinates(self, destination: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a destination"""
        destination_lower = destination.lower()
        
        # Normalize Turkish characters for better matching
        destination_normalized = self._normalize_turkish(destination_lower)
        
        # Known landmarks and attractions in Istanbul
        landmarks = self._get_known_landmarks()
        
        # Try exact match first
        for name, coords in landmarks.items():
            name_lower = name.lower()
            name_normalized = self._normalize_turkish(name_lower)
            if destination_normalized == name_normalized or destination_lower == name_lower:
                return coords
        
        # Try partial match
        for name, coords in landmarks.items():
            name_lower = name.lower()
            name_normalized = self._normalize_turkish(name_lower)
            if (destination_normalized in name_normalized or name_normalized in destination_normalized or
                destination_lower in name_lower or name_lower in destination_lower):
                return coords
        
        return None
    
    def _normalize_turkish(self, text: str) -> str:
        """Normalize Turkish characters to ASCII equivalents for matching"""
        # Turkish character mappings
        turkish_map = {
            'ı': 'i', 'İ': 'i', 'ğ': 'g', 'Ğ': 'g',
            'ü': 'u', 'Ü': 'u', 'ş': 's', 'Ş': 's',
            'ö': 'o', 'Ö': 'o', 'ç': 'c', 'Ç': 'c'
        }
        for turkish_char, ascii_char in turkish_map.items():
            text = text.replace(turkish_char, ascii_char)
        return text
    
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
