#!/usr/bin/env python3
"""
Intelligent Location Detection Service
Automatically detects user location from various sources without requiring manual coordinates
"""

import re
import logging
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from datetime import datetime

# Import route maker service to access OpenStreetMap data
try:
    from backend.services.route_maker_service import get_route_maker
    ROUTE_MAKER_AVAILABLE = True
except ImportError as e:
    ROUTE_MAKER_AVAILABLE = False
    print(f"⚠️ Route Maker Service not available for GPS validation: {e}")

logger = logging.getLogger(__name__)

class LocationConfidence(Enum):
    """Location detection confidence levels"""
    VERY_HIGH = "very_high"      # GPS coordinates, IP geolocation
    HIGH = "high"                # Named landmarks, specific addresses
    MEDIUM = "medium"            # Neighborhood names, general areas
    LOW = "low"                  # City-level, vague references
    UNKNOWN = "unknown"          # No location information detected

@dataclass
class DetectedLocation:
    """Detected location information"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    name: Optional[str] = None
    neighborhood: Optional[str] = None
    district: Optional[str] = None
    confidence: LocationConfidence = LocationConfidence.UNKNOWN
    source: str = "unknown"
    accuracy_meters: Optional[float] = None
    detected_at: datetime = None
    raw_input: str = ""
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()

class IntelligentLocationDetector:
    """
    Intelligent location detection system that automatically detects user location
    from various sources without requiring manual GPS coordinates
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.load_istanbul_landmarks()
        self.load_neighborhoods()
        self.load_transportation_hubs()
        
    def load_istanbul_landmarks(self):
        """Load Istanbul landmarks with precise coordinates"""
        self.landmarks = {
            # Historic Peninsula (Fatih)
            'hagia sophia': {'lat': 41.0086, 'lng': 28.9802, 'district': 'Fatih', 'neighborhood': 'Sultanahmet'},
            'blue mosque': {'lat': 41.0054, 'lng': 28.9768, 'district': 'Fatih', 'neighborhood': 'Sultanahmet'},
            'sultanahmet mosque': {'lat': 41.0054, 'lng': 28.9768, 'district': 'Fatih', 'neighborhood': 'Sultanahmet'},
            'topkapi palace': {'lat': 41.0115, 'lng': 28.9833, 'district': 'Fatih', 'neighborhood': 'Sultanahmet'},
            'grand bazaar': {'lat': 41.0106, 'lng': 28.9681, 'district': 'Fatih', 'neighborhood': 'Beyazıt'},
            'basilica cistern': {'lat': 41.0084, 'lng': 28.9778, 'district': 'Fatih', 'neighborhood': 'Sultanahmet'},
            'spice bazaar': {'lat': 41.0166, 'lng': 28.9707, 'district': 'Fatih', 'neighborhood': 'Eminönü'},
            
            # Beyoğlu
            'galata tower': {'lat': 41.0256, 'lng': 28.9744, 'district': 'Beyoğlu', 'neighborhood': 'Galata'},
            'taksim square': {'lat': 41.0369, 'lng': 28.9850, 'district': 'Beyoğlu', 'neighborhood': 'Taksim'},
            'istiklal street': {'lat': 41.0342, 'lng': 28.9784, 'district': 'Beyoğlu', 'neighborhood': 'Beyoğlu'},
            'dolmabahce palace': {'lat': 41.0391, 'lng': 29.0000, 'district': 'Beşiktaş', 'neighborhood': 'Kabataş'},
            
            # Beşiktaş
            'ortakoy': {'lat': 41.0553, 'lng': 29.0275, 'district': 'Beşiktaş', 'neighborhood': 'Ortaköy'},
            'bebek': {'lat': 41.0833, 'lng': 29.0439, 'district': 'Beşiktaş', 'neighborhood': 'Bebek'},
            'besiktas': {'lat': 41.0422, 'lng': 29.0064, 'district': 'Beşiktaş', 'neighborhood': 'Beşiktaş'},
            
            # Kadıköy (Asian Side)
            'kadikoy': {'lat': 41.0066, 'lng': 29.0297, 'district': 'Kadıköy', 'neighborhood': 'Kadıköy'},
            'moda': {'lat': 40.9897, 'lng': 29.0297, 'district': 'Kadıköy', 'neighborhood': 'Moda'},
            'bagdat street': {'lat': 40.9666, 'lng': 29.1067, 'district': 'Kadıköy', 'neighborhood': 'Bağdat Caddesi'},
            
            # Üsküdar
            'uskudar': {'lat': 41.0214, 'lng': 29.0106, 'district': 'Üsküdar', 'neighborhood': 'Üsküdar'},
            'maiden tower': {'lat': 41.0211, 'lng': 29.0042, 'district': 'Üsküdar', 'neighborhood': 'Salacak'},
            
            # Transportation Hubs
            'istanbul airport': {'lat': 41.2753, 'lng': 28.7519, 'district': 'Arnavutköy', 'neighborhood': 'Airport'},
            'sabiha gokcen airport': {'lat': 40.8986, 'lng': 29.3092, 'district': 'Pendik', 'neighborhood': 'Airport'},
            'ataturk airport': {'lat': 40.9769, 'lng': 28.8194, 'district': 'Bakırköy', 'neighborhood': 'Airport'},
        }
        
    def load_neighborhoods(self):
        """Load Istanbul neighborhoods with approximate centers"""
        self.neighborhoods = {
            # Fatih District
            'sultanahmet': {'lat': 41.0082, 'lng': 28.9784, 'district': 'Fatih'},
            'eminonu': {'lat': 41.0166, 'lng': 28.9707, 'district': 'Fatih'},
            'beyazit': {'lat': 41.0106, 'lng': 28.9681, 'district': 'Fatih'},
            'laleli': {'lat': 41.0106, 'lng': 28.9549, 'district': 'Fatih'},
            'aksaray': {'lat': 41.0064, 'lng': 28.9469, 'district': 'Fatih'},
            
            # Beyoğlu District
            'beyoglu': {'lat': 41.0342, 'lng': 28.9784, 'district': 'Beyoğlu'},
            'galata': {'lat': 41.0256, 'lng': 28.9744, 'district': 'Beyoğlu'},
            'karakoy': {'lat': 41.0256, 'lng': 28.9744, 'district': 'Beyoğlu'},
            'taksim': {'lat': 41.0369, 'lng': 28.9850, 'district': 'Beyoğlu'},
            'cihangir': {'lat': 41.0314, 'lng': 28.9778, 'district': 'Beyoğlu'},
            
            # Beşiktaş District
            'besiktas': {'lat': 41.0422, 'lng': 29.0064, 'district': 'Beşiktaş'},
            'ortakoy': {'lat': 41.0553, 'lng': 29.0275, 'district': 'Beşiktaş'},
            'bebek': {'lat': 41.0833, 'lng': 29.0439, 'district': 'Beşiktaş'},
            'arnavutkoy': {'lat': 41.0719, 'lng': 29.0408, 'district': 'Beşiktaş'},
            'etiler': {'lat': 41.0783, 'lng': 29.0161, 'district': 'Beşiktaş'},
            
            # Kadıköy District
            'kadikoy': {'lat': 41.0066, 'lng': 29.0297, 'district': 'Kadıköy'},
            'moda': {'lat': 40.9897, 'lng': 29.0297, 'district': 'Kadıköy'},
            'bagdat street': {'lat': 40.9666, 'lng': 29.1067, 'district': 'Kadıköy'},
            'fenerbahce': {'lat': 40.9648, 'lng': 29.0897, 'district': 'Kadıköy'},
            
            # Üsküdar District
            'uskudar': {'lat': 41.0214, 'lng': 29.0106, 'district': 'Üsküdar'},
            'camlica': {'lat': 41.0292, 'lng': 29.0611, 'district': 'Üsküdar'},
            'beylerbeyi': {'lat': 41.0419, 'lng': 29.0406, 'district': 'Üsküdar'},
            
            # Şişli District
            'sisli': {'lat': 41.0611, 'lng': 28.9869, 'district': 'Şişli'},
            'nisantasi': {'lat': 41.0472, 'lng': 28.9922, 'district': 'Şişli'},
            'mecidiyekoy': {'lat': 41.0661, 'lng': 28.9906, 'district': 'Şişli'},
        }
    
    def load_transportation_hubs(self):
        """Load major transportation hubs"""
        self.transport_hubs = {
            # Metro Stations
            'taksim metro': {'lat': 41.0369, 'lng': 28.9850, 'type': 'metro'},
            'sirkeci station': {'lat': 41.0147, 'lng': 28.9761, 'type': 'train'},
            'kadikoy ferry': {'lat': 41.0066, 'lng': 29.0297, 'type': 'ferry'},
            'eminonu ferry': {'lat': 41.0166, 'lng': 28.9707, 'type': 'ferry'},
            'besiktas ferry': {'lat': 41.0422, 'lng': 29.0064, 'type': 'ferry'},
            'kabatas ferry': {'lat': 41.0391, 'lng': 29.0000, 'type': 'ferry'},
            
            # Bus Stations
            'esenler bus station': {'lat': 41.0411, 'lng': 28.8822, 'type': 'bus'},
            'harem bus station': {'lat': 41.0167, 'lng': 29.0289, 'type': 'bus'},
        }

    async def detect_location_from_text(self, text: str, user_context: Optional[Dict] = None) -> DetectedLocation:
        """
        Detect location from user text input using various methods
        """
        text_lower = text.lower().strip()
        
        # Try different detection methods in order of confidence
        location = None
        
        # 1. GPS Coordinates (highest confidence)
        location = self._extract_gps_coordinates(text)
        if location.confidence != LocationConfidence.UNKNOWN:
            location.source = "gps_coordinates"
            location.raw_input = text
            return location
        
        # 2. Landmark detection (high confidence)
        location = self._detect_landmarks(text_lower)
        if location.confidence != LocationConfidence.UNKNOWN:
            location.source = "landmark_detection"
            location.raw_input = text
            return location
        
        # 3. Neighborhood detection (medium confidence)
        location = self._detect_neighborhoods(text_lower)
        if location.confidence != LocationConfidence.UNKNOWN:
            location.source = "neighborhood_detection"
            location.raw_input = text
            return location
        
        # 4. Transportation hub detection (medium confidence)
        location = self._detect_transport_hubs(text_lower)
        if location.confidence != LocationConfidence.UNKNOWN:
            location.source = "transport_hub_detection"
            location.raw_input = text
            return location
        
        # 5. Contextual location inference (low confidence)
        if user_context:
            location = self._infer_from_context(text_lower, user_context)
            if location.confidence != LocationConfidence.UNKNOWN:
                location.source = "context_inference"
                location.raw_input = text
                return location
        
        # 6. Generic location patterns (low confidence)
        location = self._detect_generic_patterns(text_lower)
        if location.confidence != LocationConfidence.UNKNOWN:
            location.source = "pattern_detection"
            location.raw_input = text
            return location
        
        # Return unknown location
        return DetectedLocation(
            confidence=LocationConfidence.UNKNOWN,
            source="none",
            raw_input=text
        )
    
    def _extract_gps_coordinates(self, text: str) -> DetectedLocation:
        """Extract GPS coordinates from text"""
        # Pattern for decimal coordinates
        coord_patterns = [
            r'(?:gps|coordinates?|location)[:\s]*([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)',
            r'([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)',
            r'lat[:\s]*([+-]?\d+\.?\d*)[,\s]*lng?[:\s]*([+-]?\d+\.?\d*)',
            r'latitude[:\s]*([+-]?\d+\.?\d*)[,\s]*longitude[:\s]*([+-]?\d+\.?\d*)',
        ]
        
        for pattern in coord_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    lat = float(match.group(1))
                    lng = float(match.group(2))
                    
                    # Validate coordinates against OpenStreetMap data
                    is_valid, district_name = self._validate_gps_against_osm(lat, lng)
                    
                    if is_valid:
                        location = DetectedLocation(
                            latitude=lat,
                            longitude=lng,
                            confidence=LocationConfidence.VERY_HIGH,
                            accuracy_meters=10.0
                        )
                        
                        # Add district information if found
                        if district_name:
                            location.district = district_name
                            self.logger.info(f"✅ GPS coordinates validated in OSM district: {district_name}")
                        
                        return location
                        
                except ValueError:
                    continue
        
        return DetectedLocation()
    
    def _detect_landmarks(self, text: str) -> DetectedLocation:
        """Detect known landmarks in text"""
        for landmark, info in self.landmarks.items():
            if landmark in text or any(word in text for word in landmark.split()):
                return DetectedLocation(
                    latitude=info['lat'],
                    longitude=info['lng'],
                    name=landmark.title(),
                    neighborhood=info['neighborhood'],
                    district=info['district'],
                    confidence=LocationConfidence.HIGH,
                    accuracy_meters=50.0
                )
        
        return DetectedLocation()
    
    def _detect_neighborhoods(self, text: str) -> DetectedLocation:
        """Detect neighborhoods in text"""
        for neighborhood, info in self.neighborhoods.items():
            if neighborhood in text:
                return DetectedLocation(
                    latitude=info['lat'],
                    longitude=info['lng'],
                    neighborhood=neighborhood.title(),
                    district=info['district'],
                    confidence=LocationConfidence.MEDIUM,
                    accuracy_meters=500.0
                )
        
        return DetectedLocation()
    
    def _detect_transport_hubs(self, text: str) -> DetectedLocation:
        """Detect transportation hubs in text"""
        for hub, info in self.transport_hubs.items():
            if hub in text or any(word in text for word in hub.split()):
                return DetectedLocation(
                    latitude=info['lat'],
                    longitude=info['lng'],
                    name=hub.title(),
                    confidence=LocationConfidence.MEDIUM,
                    accuracy_meters=100.0
                )
        
        return DetectedLocation()
    
    def _infer_from_context(self, text: str, context: Dict) -> DetectedLocation:
        """Infer location from user context and previous interactions"""
        # Check if user has previous location
        if 'last_location' in context:
            last_loc = context['last_location']
            if isinstance(last_loc, dict) and 'lat' in last_loc and 'lng' in last_loc:
                return DetectedLocation(
                    latitude=last_loc['lat'],
                    longitude=last_loc['lng'],
                    confidence=LocationConfidence.LOW,
                    accuracy_meters=1000.0
                )
        
        # Check for "here", "my location", "where I am" patterns
        location_patterns = [
            'here', 'my location', 'where i am', 'current location',
            'this place', 'this area', 'around here', 'nearby'
        ]
        
        if any(pattern in text for pattern in location_patterns):
            # Default to Sultanahmet if no other context
            return DetectedLocation(
                latitude=41.0082,
                longitude=28.9784,
                neighborhood='Sultanahmet',
                district='Fatih',
                confidence=LocationConfidence.LOW,
                accuracy_meters=2000.0
            )
        
        return DetectedLocation()
    
    def _detect_generic_patterns(self, text: str) -> DetectedLocation:
        """Detect generic location patterns"""
        # Hotel names, restaurant names, etc.
        hotel_patterns = [
            r'hotel\s+(\w+)', r'(\w+)\s+hotel',
            r'staying at\s+(\w+)', r'at the\s+(\w+)'
        ]
        
        for pattern in hotel_patterns:
            match = re.search(pattern, text)
            if match:
                # Default to central Istanbul for hotel mentions
                return DetectedLocation(
                    latitude=41.0082,
                    longitude=28.9784,
                    name=match.group(1).title(),
                    neighborhood='Sultanahmet',
                    district='Fatih',
                    confidence=LocationConfidence.LOW,
                    accuracy_meters=3000.0
                )
        
        return DetectedLocation()
    
    def _is_istanbul_coordinates(self, lat: float, lng: float) -> bool:
        """Check if coordinates are within Istanbul boundaries"""
        # Istanbul rough boundaries
        istanbul_bounds = {
            'lat_min': 40.8, 'lat_max': 41.5,
            'lng_min': 28.5, 'lng_max': 29.5
        }
        
        return (istanbul_bounds['lat_min'] <= lat <= istanbul_bounds['lat_max'] and
                istanbul_bounds['lng_min'] <= lng <= istanbul_bounds['lng_max'])
    
    def _validate_gps_against_osm(self, lat: float, lng: float) -> Tuple[bool, Optional[str]]:
        """
        Validate GPS coordinates against loaded OpenStreetMap districts
        Returns (is_valid, district_name)
        """
        if not ROUTE_MAKER_AVAILABLE:
            # Fallback to basic Istanbul bounds check
            return self._is_istanbul_coordinates(lat, lng), None
        
        try:
            route_maker = get_route_maker()
            
            # Check if coordinates are within any loaded district
            if hasattr(route_maker, 'available_districts'):
                for district_name, district_graph in route_maker.available_districts.items():
                    # Get the bounds of this district
                    if hasattr(district_graph, 'graph') and 'x' in district_graph.graph:
                        nodes_data = district_graph.graph['x'], district_graph.graph['y']
                    else:
                        # Get node coordinates from the graph
                        nodes = list(district_graph.nodes(data=True))
                        if not nodes:
                            continue
                        
                        lats = [node[1].get('y', 0) for node in nodes if 'y' in node[1]]
                        lngs = [node[1].get('x', 0) for node in nodes if 'x' in node[1]]
                        
                        if not lats or not lngs:
                            continue
                        
                        # Check if GPS coordinates are within district bounds
                        min_lat, max_lat = min(lats), max(lats)
                        min_lng, max_lng = min(lngs), max(lngs)
                        
                        # Add small buffer for edge cases
                        buffer = 0.005  # ~500m
                        if (min_lat - buffer <= lat <= max_lat + buffer and
                            min_lng - buffer <= lng <= max_lng + buffer):
                            return True, district_name
            
            # If not found in specific districts, check general Istanbul bounds
            return self._is_istanbul_coordinates(lat, lng), None
            
        except Exception as e:
            self.logger.warning(f"OSM GPS validation failed: {e}")
            # Fallback to basic bounds check
            return self._is_istanbul_coordinates(lat, lng), None
    
    async def get_location_from_ip(self, ip_address: str) -> DetectedLocation:
        """Get location from IP address (fallback method)"""
        try:
            # Use a free IP geolocation service
            response = requests.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    lat = float(data.get('lat', 0))
                    lng = float(data.get('lon', 0))
                    
                    if self._is_istanbul_coordinates(lat, lng):
                        return DetectedLocation(
                            latitude=lat,
                            longitude=lng,
                            name=data.get('city', 'Istanbul'),
                            confidence=LocationConfidence.MEDIUM,
                            accuracy_meters=5000.0
                        )
        except Exception as e:
            self.logger.warning(f"IP geolocation failed: {e}")
        
        return DetectedLocation()
    
    def enhance_location_with_nearby_info(self, location: DetectedLocation) -> DetectedLocation:
        """Enhance detected location with nearby landmarks and district info"""
        if not location.latitude or not location.longitude:
            return location
        
        # 1. First try to validate and enhance with OpenStreetMap data
        if ROUTE_MAKER_AVAILABLE:
            # Validate coordinates against OSM data
            is_valid, osm_district = self._validate_gps_against_osm(location.latitude, location.longitude)
            if is_valid and osm_district and not location.district:
                location.district = osm_district
                self.logger.info(f"📍 Enhanced location with OSM district: {osm_district}")
            
            # Find nearest OSM node for more precise location info
            nearest_node = self._find_nearest_osm_node(location.latitude, location.longitude)
            if nearest_node:
                # If we found a very close OSM node, this gives us confidence the location is accurate
                if nearest_node['distance_km'] < 0.1:  # Within 100m
                    location.confidence = LocationConfidence.VERY_HIGH
                    location.accuracy_meters = nearest_node['distance_km'] * 1000
                elif nearest_node['distance_km'] < 0.5:  # Within 500m
                    if location.confidence == LocationConfidence.UNKNOWN:
                        location.confidence = LocationConfidence.HIGH
                    location.accuracy_meters = nearest_node['distance_km'] * 1000
                
                # Update district info from OSM if not already set
                if not location.district and nearest_node['district']:
                    location.district = nearest_node['district']
        
        # 2. Find nearest landmark from our predefined list
        min_distance = float('inf')
        nearest_landmark = None
        
        for landmark, info in self.landmarks.items():
            distance = self._calculate_distance(
                location.latitude, location.longitude,
                info['lat'], info['lng']
            )
            if distance < min_distance:
                min_distance = distance
                nearest_landmark = (landmark, info)
        
        # If within 1km of a landmark, enhance the location
        if nearest_landmark and min_distance < 1.0:
            landmark_name, landmark_info = nearest_landmark
            if not location.neighborhood:
                location.neighborhood = landmark_info['neighborhood']
            if not location.district:
                location.district = landmark_info['district']
            if not location.name and min_distance < 0.1:  # Very close
                location.name = landmark_name.title()
                
            self.logger.info(f"🏛️ Found nearby landmark: {landmark_name} ({min_distance:.3f}km away)")
        
        return location
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        import math
        
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lng / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _find_nearest_osm_node(self, lat: float, lng: float) -> Optional[Dict[str, Any]]:
        """
        Find the nearest OpenStreetMap node to given GPS coordinates
        Returns node information if found
        """
        if not ROUTE_MAKER_AVAILABLE:
            return None
        
        try:
            route_maker = get_route_maker()
            
            if not hasattr(route_maker, 'available_districts'):
                return None
            
            nearest_node = None
            min_distance = float('inf')
            best_district = None
            
            # Search through all loaded districts
            for district_name, district_graph in route_maker.available_districts.items():
                nodes = list(district_graph.nodes(data=True))
                
                for node_id, node_data in nodes:
                    if 'y' not in node_data or 'x' not in node_data:
                        continue
                        
                    node_lat = node_data['y']
                    node_lng = node_data['x']
                    
                    # Calculate distance to the target GPS coordinate
                    distance = self._calculate_distance(lat, lng, node_lat, node_lng)
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_node = {
                            'node_id': node_id,
                            'lat': node_lat,
                            'lng': node_lng,
                            'distance_km': distance,
                            'district': district_name
                        }
                        best_district = district_name
            
            # Only return if reasonably close (within 2km)
            if nearest_node and min_distance < 2.0:
                self.logger.info(f"🎯 Found nearest OSM node in {best_district}: {min_distance:.3f}km away")
                return nearest_node
            
        except Exception as e:
            self.logger.warning(f"Error finding nearest OSM node: {e}")
        
        return None

# Global instance
intelligent_location_detector = IntelligentLocationDetector()

async def detect_user_location(text: str, user_context: Optional[Dict] = None, ip_address: Optional[str] = None) -> DetectedLocation:
    """
    Main function to detect user location from various sources
    """
    detector = intelligent_location_detector
    
    # Try text-based detection first
    location = await detector.detect_location_from_text(text, user_context)
    
    # If text detection failed and we have IP, try IP geolocation
    if location.confidence == LocationConfidence.UNKNOWN and ip_address:
        location = await detector.get_location_from_ip(ip_address)
    
    # Enhance location with nearby information
    if location.latitude and location.longitude:
        location = detector.enhance_location_with_nearby_info(location)
    
    return location
