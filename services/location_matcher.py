"""
Location Matcher - Fuzzy Stop Matching and Geocoding
Converts user locations to transportation network stops
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import re
from difflib import SequenceMatcher
import math

@dataclass
class LocationMatch:
    """Represents a matched location/stop"""
    stop_id: str
    stop_name: str
    transport_type: str
    latitude: float
    longitude: float
    distance_km: float  # Distance from query location
    confidence: float  # Match confidence (0-1)
    lines: List[str]  # Lines serving this stop
    
    def to_dict(self) -> Dict:
        return {
            'stop_id': self.stop_id,
            'name': self.stop_name,
            'transport_type': self.transport_type,
            'location': {'lat': self.latitude, 'lon': self.longitude},
            'distance_km': round(self.distance_km, 2),
            'confidence': round(self.confidence, 2),
            'lines': self.lines
        }

class LocationMatcher:
    """
    Intelligent location matching for transportation network
    Handles fuzzy text matching, coordinate-based search, and geocoding
    """
    
    def __init__(self, network_graph):
        """
        Initialize with transportation network
        
        Args:
            network_graph: TransportationNetwork from route_network_builder
        """
        self.network = network_graph
        self.earth_radius_km = 6371.0
        
        # Build search indices
        self._build_search_indices()
    
    def _build_search_indices(self):
        """Build efficient search indices for stops"""
        # Name-based index
        self.name_index: Dict[str, List[str]] = {}  # normalized_name -> [stop_ids]
        
        # Token-based index for partial matches
        self.token_index: Dict[str, List[str]] = {}  # token -> [stop_ids]
        
        # Area/district index
        self.area_index: Dict[str, List[str]] = {}  # area_name -> [stop_ids]
        
        for stop_id, stop in self.network.stops.items():
            # Normalize stop name
            normalized = self._normalize_text(stop.name)
            
            # Add to name index
            if normalized not in self.name_index:
                self.name_index[normalized] = []
            self.name_index[normalized].append(stop_id)
            
            # Add tokens to token index
            tokens = normalized.split()
            for token in tokens:
                if token not in self.token_index:
                    self.token_index[token] = []
                self.token_index[token].append(stop_id)
            
            # Extract area/district if present
            area = self._extract_area(stop.name)
            if area:
                if area not in self.area_index:
                    self.area_index[area] = []
                self.area_index[area].append(stop_id)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching (lowercase, remove special chars)
        """
        # Turkish character mapping
        turkish_map = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u'
        }
        
        text = text.lower()
        for tr_char, en_char in turkish_map.items():
            text = text.replace(tr_char, en_char)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_area(self, stop_name: str) -> Optional[str]:
        """Extract area/district name from stop name"""
        # Common patterns: "Stop Name (District)", "District - Stop Name"
        patterns = [
            r'\(([^)]+)\)',  # Extract text in parentheses
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\s*-',  # Extract before dash
        ]
        
        for pattern in patterns:
            match = re.search(pattern, stop_name)
            if match:
                return self._normalize_text(match.group(1))
        
        return None
    
    def calculate_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """
        Calculate great-circle distance between two coordinates
        
        Args:
            coord1: (latitude, longitude)
            coord2: (latitude, longitude)
            
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return self.earth_radius_km * c
    
    def find_stops_by_name(self, query: str, 
                          max_results: int = 5,
                          min_confidence: float = 0.3) -> List[LocationMatch]:
        """
        Find stops by fuzzy name matching
        
        Args:
            query: Search query (stop name)
            max_results: Maximum number of results
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of LocationMatch objects
        """
        normalized_query = self._normalize_text(query)
        query_tokens = set(normalized_query.split())
        
        candidates: Dict[str, float] = {}  # stop_id -> confidence score
        
        # Exact name match (highest priority)
        if normalized_query in self.name_index:
            for stop_id in self.name_index[normalized_query]:
                candidates[stop_id] = 1.0
        
        # Token-based matching
        for token in query_tokens:
            # Exact token match
            if token in self.token_index:
                for stop_id in self.token_index[token]:
                    if stop_id not in candidates:
                        candidates[stop_id] = 0.0
                    candidates[stop_id] += 0.3
            
            # Partial token match (prefix)
            for indexed_token, stop_ids in self.token_index.items():
                if indexed_token.startswith(token) or token.startswith(indexed_token):
                    similarity = SequenceMatcher(None, token, indexed_token).ratio()
                    if similarity > 0.7:
                        for stop_id in stop_ids:
                            if stop_id not in candidates:
                                candidates[stop_id] = 0.0
                            candidates[stop_id] += similarity * 0.2
        
        # Fuzzy matching on all stop names if needed
        if len(candidates) < max_results:
            for stop_id, stop in self.network.stops.items():
                if stop_id in candidates:
                    continue
                
                normalized_name = self._normalize_text(stop.name)
                similarity = SequenceMatcher(None, normalized_query, normalized_name).ratio()
                
                if similarity > min_confidence:
                    candidates[stop_id] = similarity
        
        # Convert to LocationMatch objects
        matches = []
        for stop_id, confidence in candidates.items():
            if confidence < min_confidence:
                continue
            
            stop = self.network.stops[stop_id]
            
            # Get lines serving this stop
            lines = self._get_stop_lines(stop_id)
            
            match = LocationMatch(
                stop_id=stop_id,
                stop_name=stop.name,
                transport_type=stop.transport_type,
                latitude=stop.lat,
                longitude=stop.lon,
                distance_km=0.0,  # No distance for name-based search
                confidence=min(1.0, confidence),
                lines=lines
            )
            matches.append(match)
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return matches[:max_results]
    
    def find_stops_by_coordinates(self, latitude: float, longitude: float,
                                  max_distance_km: float = 1.0,
                                  max_results: int = 5) -> List[LocationMatch]:
        """
        Find stops near given coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            max_distance_km: Maximum search radius
            max_results: Maximum number of results
            
        Returns:
            List of LocationMatch objects sorted by distance
        """
        query_coord = (latitude, longitude)
        nearby_stops = []
        
        for stop_id, stop in self.network.stops.items():
            stop_coord = (stop.lat, stop.lon)
            distance = self.calculate_distance(query_coord, stop_coord)
            
            if distance <= max_distance_km:
                lines = self._get_stop_lines(stop_id)
                
                # Calculate confidence based on distance (closer = higher confidence)
                confidence = max(0.0, 1.0 - (distance / max_distance_km))
                
                match = LocationMatch(
                    stop_id=stop_id,
                    stop_name=stop.name,
                    transport_type=stop.transport_type,
                    latitude=stop.lat,
                    longitude=stop.lon,
                    distance_km=distance,
                    confidence=confidence,
                    lines=lines
                )
                nearby_stops.append(match)
        
        # Sort by distance
        nearby_stops.sort(key=lambda m: m.distance_km)
        
        return nearby_stops[:max_results]
    
    def find_nearest_stops(self, 
                          gps_lat: float,
                          gps_lng: float, 
                          max_distance_km: float = 1.0,
                          transport_types: Optional[List[str]] = None,
                          limit: int = 5) -> List[Dict]:
        """
        Find nearest transportation stops to GPS coordinates
        Optimized for GPS-based routing with walking directions
        
        Args:
            gps_lat: User's GPS latitude
            gps_lng: User's GPS longitude
            max_distance_km: Maximum search radius (default 1km)
            transport_types: Filter by transport type (e.g., ['metro', 'bus'])
            limit: Maximum number of results
            
        Returns:
            List of nearest stops with walking info:
            {
                'stop_id': str,
                'stop_name': str,
                'transport_type': str,
                'distance_km': float,
                'distance_m': int,
                'walking_time_min': int,
                'coordinates': {'lat': float, 'lon': float},
                'bearing': float,  # Compass bearing (0-360)
                'direction': str,  # Human-readable direction (e.g., 'north')
                'lines': List[str]
            }
        """
        results = []
        
        for stop_id, stop in self.network.stops.items():
            # Skip if transport type filter doesn't match
            if transport_types and stop.transport_type not in transport_types:
                continue
            
            # Calculate distance using haversine
            distance_km = self._haversine_distance(
                gps_lat, gps_lng,
                stop.lat, stop.lon
            )
            
            # Skip if too far
            if distance_km > max_distance_km:
                continue
            
            # Calculate bearing for directions
            bearing = self._calculate_bearing(
                gps_lat, gps_lng,
                stop.lat, stop.lon
            )
            
            # Convert bearing to human-readable direction
            direction = self._bearing_to_direction(bearing)
            
            # Calculate walking time (80 meters/minute average walking speed)
            distance_m = int(distance_km * 1000)
            walking_time_min = max(1, int(distance_m / 80))
            
            # Get lines serving this stop
            lines = self._get_stop_lines(stop_id)
            
            results.append({
                'stop_id': stop_id,
                'stop_name': stop.name,
                'transport_type': stop.transport_type,
                'distance_km': round(distance_km, 3),
                'distance_m': distance_m,
                'walking_time_min': walking_time_min,
                'coordinates': {
                    'lat': stop.lat,
                    'lon': stop.lon
                },
                'bearing': round(bearing, 1),
                'direction': direction,
                'lines': lines
            })
        
        # Sort by distance (closest first)
        results.sort(key=lambda x: x['distance_km'])
        
        return results[:limit]
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two GPS points using Haversine formula
        More accurate than simple Euclidean distance for GPS coordinates
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
            
        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in kilometers
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon/2)**2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _calculate_bearing(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """
        Calculate compass bearing between two GPS points
        
        Args:
            lat1, lon1: Starting coordinate
            lat2, lon2: Ending coordinate
            
        Returns:
            Bearing in degrees (0-360, where 0 = North)
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        
        # Calculate bearing
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360
        return (bearing_deg + 360) % 360
    
    def _bearing_to_direction(self, bearing: float) -> str:
        """
        Convert compass bearing to human-readable direction
        
        Args:
            bearing: Bearing in degrees (0-360)
            
        Returns:
            Direction name (e.g., 'north', 'northeast', etc.)
        """
        directions = [
            (0, 22.5, 'north'),
            (22.5, 67.5, 'northeast'),
            (67.5, 112.5, 'east'),
            (112.5, 157.5, 'southeast'),
            (157.5, 202.5, 'south'),
            (202.5, 247.5, 'southwest'),
            (247.5, 292.5, 'west'),
            (292.5, 337.5, 'northwest'),
            (337.5, 360, 'north')
        ]
        
        for min_deg, max_deg, direction in directions:
            if min_deg <= bearing < max_deg:
                return direction
        
        return 'north'  # Default for 337.5-360 range

    def find_stops_by_area(self, area_name: str,
                          max_results: int = 10) -> List[LocationMatch]:
        """
        Find all stops in a specific area/district
        
        Args:
            area_name: Area or district name
            max_results: Maximum number of results
            
        Returns:
            List of LocationMatch objects
        """
        normalized_area = self._normalize_text(area_name)
        
        if normalized_area not in self.area_index:
            return []
        
        matches = []
        for stop_id in self.area_index[normalized_area][:max_results]:
            stop = self.network.stops[stop_id]
            lines = self._get_stop_lines(stop_id)
            
            match = LocationMatch(
                stop_id=stop_id,
                stop_name=stop.name,
                transport_type=stop.transport_type,
                latitude=stop.lat,
                longitude=stop.lon,
                distance_km=0.0,
                confidence=1.0,
                lines=lines
            )
            matches.append(match)
        
        return matches
    
    def geocode_location(self, location_query: str) -> Optional[Tuple[float, float]]:
        """
        Convert location query to coordinates (simplified geocoding)
        In production, this would use a real geocoding service
        
        Args:
            location_query: Location name or address
            
        Returns:
            (latitude, longitude) tuple or None
        """
        # For now, try to match to a known stop
        matches = self.find_stops_by_name(location_query, max_results=1, min_confidence=0.5)
        
        if matches:
            return (matches[0].latitude, matches[0].longitude)
        
        # Famous Istanbul landmarks (fallback)
        landmarks = {
            'taksim': (41.0370, 28.9857),
            'sultanahmet': (41.0082, 28.9784),
            'besiktas': (41.0422, 29.0094),
            'kadikoy': (40.9905, 29.0250),
            'uskudar': (41.0226, 29.0078),
            'sisli': (41.0602, 28.9879),
            'beyoglu': (41.0329, 28.9779),
            'fatih': (41.0192, 28.9497),
            'levent': (41.0782, 29.0070),
            'maslak': (41.1121, 29.0208)
        }
        
        normalized_query = self._normalize_text(location_query)
        for landmark, coords in landmarks.items():
            if landmark in normalized_query:
                return coords
        
        return None
    
    def _get_stop_lines(self, stop_id: str) -> List[str]:
        """Get list of line names serving a stop"""
        lines = set()
        
        for line_id, line_data in self.network.lines.items():
            # line_data is a TransportLine object
            if hasattr(line_data, 'stops') and stop_id in line_data.stops:
                line_name = line_data.name if hasattr(line_data, 'name') else line_id
                lines.add(line_name)
        
        return sorted(list(lines))
    
    def match_location(self, query: str,
                      prefer_coordinates: bool = False) -> Optional[LocationMatch]:
        """
        Smart location matching - tries multiple strategies
        
        Args:
            query: Location query (name, coordinates, or address)
            prefer_coordinates: Prefer coordinate-based matching
            
        Returns:
            Best LocationMatch or None
        """
        # Try parsing as coordinates (lat,lon)
        coord_match = re.match(r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', query)
        if coord_match:
            lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
            matches = self.find_stops_by_coordinates(lat, lon, max_results=1)
            if matches:
                return matches[0]
        
        # Try name-based matching
        name_matches = self.find_stops_by_name(query, max_results=1, min_confidence=0.5)
        if name_matches:
            return name_matches[0]
        
        # Try geocoding
        coords = self.geocode_location(query)
        if coords:
            coord_matches = self.find_stops_by_coordinates(
                coords[0], coords[1], max_results=1
            )
            if coord_matches:
                return coord_matches[0]
        
        return None
    
    def get_nearby_transport_options(self, latitude: float, longitude: float,
                                     max_distance_km: float = 0.5) -> Dict[str, List[LocationMatch]]:
        """
        Get all nearby transport options grouped by type
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            max_distance_km: Search radius
            
        Returns:
            Dictionary of transport_type -> [LocationMatch]
        """
        all_nearby = self.find_stops_by_coordinates(
            latitude, longitude, max_distance_km, max_results=50
        )
        
        grouped = {}
        for match in all_nearby:
            transport_type = match.transport_type
            if transport_type not in grouped:
                grouped[transport_type] = []
            grouped[transport_type].append(match)
        
        return grouped
