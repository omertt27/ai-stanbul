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
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import aiohttp

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

class EventCategory(Enum):
    """Event categories in Istanbul"""
    MUSIC = "music"
    THEATER = "theater"
    DANCE = "dance"
    VISUAL_ARTS = "visual_arts"
    FILM = "film"
    LITERATURE = "literature"
    FESTIVAL = "festival"
    CULTURAL = "cultural"
    SPORTS = "sports"
    FOOD = "food"
    NIGHTLIFE = "nightlife"
    FAMILY = "family"
    EXHIBITION = "exhibition"
    CONFERENCE = "conference"

@dataclass
class IstanbulEvent:
    """Istanbul event information"""
    title: str
    category: EventCategory
    venue: str
    venue_lat: Optional[float] = None
    venue_lng: Optional[float] = None
    district: Optional[str] = None
    neighborhood: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    price: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    organizer: Optional[str] = None
    is_free: bool = False
    is_recurring: bool = False
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = []

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
    nearby_events: List[IstanbulEvent] = None
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()
        if self.nearby_events is None:
            self.nearby_events = []

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
        self.load_event_venues()
        self.events_cache = {}
        self.cache_expiry = None
        
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
    
    def load_event_venues(self):
        """Load major event venues in Istanbul with coordinates"""
        self.event_venues = {
            # İKSV Venues
            'akm': {'lat': 41.0369, 'lng': 28.9850, 'name': 'Atatürk Cultural Center', 'district': 'Beyoğlu'},
            'akbank sanat': {'lat': 41.0342, 'lng': 28.9784, 'name': 'Akbank Sanat', 'district': 'Beyoğlu'},
            'garaj istanbul': {'lat': 41.0256, 'lng': 28.9744, 'name': 'Garaj Istanbul', 'district': 'Beyoğlu'},
            'salon iksv': {'lat': 41.0369, 'lng': 28.9850, 'name': 'Salon İKSV', 'district': 'Beyoğlu'},
            
            # Major Cultural Centers
            'cemal resit rey': {'lat': 41.0783, 'lng': 29.0161, 'name': 'Cemal Reşit Rey Concert Hall', 'district': 'Şişli'},
            'zorlu psm': {'lat': 41.0783, 'lng': 29.0161, 'name': 'Zorlu PSM', 'district': 'Beşiktaş'},
            'turkcell kuruçeşme arena': {'lat': 41.0719, 'lng': 29.0408, 'name': 'Turkcell Kuruçeşme Arena', 'district': 'Beşiktaş'},
            'harbiye cemil topuzlu': {'lat': 41.0472, 'lng': 28.9922, 'name': 'Harbiye Cemil Topuzlu Open Air Theater', 'district': 'Şişli'},
            
            # Museums and Galleries
            'istanbul modern': {'lat': 41.0256, 'lng': 28.9744, 'name': 'Istanbul Modern', 'district': 'Beyoğlu'},
            'pera museum': {'lat': 41.0342, 'lng': 28.9784, 'name': 'Pera Museum', 'district': 'Beyoğlu'},
            'sakıp sabancı museum': {'lat': 41.1086, 'lng': 29.0586, 'name': 'Sakıp Sabancı Museum', 'district': 'Sarıyer'},
            
            # Traditional Venues
            'hagia irene': {'lat': 41.0115, 'lng': 28.9833, 'name': 'Hagia Irene', 'district': 'Fatih'},
            'basilica cistern': {'lat': 41.0084, 'lng': 28.9778, 'name': 'Basilica Cistern', 'district': 'Fatih'},
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
            response = requests.get(f'http://ip-api.com/json/{ip_address}')
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

    async def fetch_iksv_events(self) -> List[IstanbulEvent]:
        """Fetch events from İKSV using the MonthlyEventsScheduler"""
        events = []
        
        try:
            # Import and use the MonthlyEventsScheduler
            from monthly_events_scheduler import MonthlyEventsScheduler
            
            scheduler = MonthlyEventsScheduler()
            
            # Try to get cached events first
            cached_events = scheduler.load_cached_events()
            
            if cached_events and not scheduler.is_fetch_needed():
                self.logger.info(f"📚 Using {len(cached_events)} cached İKSV events")
                raw_events = cached_events
            else:
                # Fetch fresh events if no cache or cache is old
                self.logger.info("🌐 Fetching fresh İKSV events...")
                raw_events = await scheduler.fetch_iksv_events()
                
                # Cache the events for future use
                if raw_events:
                    await scheduler.save_events_to_cache(raw_events)
                    self.logger.info(f"💾 Cached {len(raw_events)} İKSV events")
            
            # Convert raw events to IstanbulEvent objects
            for raw_event in raw_events:
                try:
                    # Parse date from date_str if available
                    start_date = None
                    if 'date_str' in raw_event:
                        start_date = self._parse_event_date(raw_event['date_str'])
                    
                    # Get venue coordinates
                    venue_name = raw_event.get('venue', '')
                    venue_lat, venue_lng = self._get_venue_coordinates(venue_name)
                    
                    # Determine category
                    category = self._categorize_iksv_event(raw_event.get('category', ''))
                    
                    event = IstanbulEvent(
                        title=raw_event.get('title', 'İKSV Event'),
                        description=raw_event.get('description', f"An event at {venue_name}"),
                        start_date=start_date,
                        venue=venue_name,
                        district=raw_event.get('district', self._get_district_for_venue(venue_name)),
                        category=category,
                        organizer='İKSV',
                        url=raw_event.get('url', 'https://www.iksv.org'),
                        is_free=raw_event.get('is_free', False),
                        venue_lat=venue_lat,
                        venue_lng=venue_lng
                    )
                    
                    events.append(event)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing İKSV event: {e}")
                    continue
            
            self.logger.info(f"🎭 Processed {len(events)} İKSV events")
            
        except ImportError:
            self.logger.warning("MonthlyEventsScheduler not available, using fallback İKSV events")
            events = self._get_fallback_iksv_events()
        except Exception as e:
            self.logger.error(f"Error fetching İKSV events: {e}")
            events = self._get_fallback_iksv_events()
        
        return events

    def _parse_event_date(self, date_str: str) -> Optional[datetime]:
        """Parse event date from various formats"""
        try:
            # Handle formats like "20 October Monday 19.00"
            import re
            from datetime import datetime
            
            # Extract date and time components
            date_patterns = [
                r'(\d{1,2})\s+(\w+)\s+\w+\s+(\d{1,2})[:\.](\d{2})',  # "20 October Monday 19.00"
                r'(\d{1,2})\s+(\w+)\s+(\d{1,2})[:\.](\d{2})',        # "20 October 19.00"
                r'(\d{1,2})/(\d{1,2})/(\d{4})',                      # "20/10/2024"
                r'(\d{4})-(\d{2})-(\d{2})',                          # "2024-10-20"
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_str)
                if match:
                    try:
                        current_year = datetime.now().year
                        if len(match.groups()) == 4:
                            day, month_name, hour, minute = match.groups()
                            # Convert month name to number
                            month_names = {
                                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                'september': 9, 'october': 10, 'november': 11, 'december': 12,
                                'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4, 'mayıs': 5,
                                'haziran': 6, 'temmuz': 7, 'ağustos': 8, 'eylül': 9,
                                'ekim': 10, 'kasım': 11, 'aralık': 12
                            }
                            month = month_names.get(month_name.lower(), 1)
                            return datetime(current_year, month, int(day), int(hour), int(minute))
                    except ValueError:
                        continue
            
            return None
        except Exception:
            return None

    def _get_venue_coordinates(self, venue_name: str) -> tuple:
        """Get coordinates for a venue"""
        venue_coords = {
            # İKSV Venues
            'zorlu psm': (41.0677, 29.0197),
            'zorlu psm turkcell stage': (41.0677, 29.0197),
            'zorlu psm turkcell platinum stage': (41.0677, 29.0197),
            'salon iksv': (41.0369, 28.9850),
            'salon İKSV': (41.0369, 28.9850),
            'harbiye muhsin ertuğrul stage': (41.0458, 28.9881),
            'cemal reşit rey concert hall': (41.0458, 28.9881),
            'lütfi kırdar convention center': (41.0458, 28.9881),
            'atatürk cultural center': (41.0369, 28.9850),
            'akm': (41.0369, 28.9850),
        }
        
        venue_key = venue_name.lower().strip()
        return venue_coords.get(venue_key, (None, None))

    def _categorize_iksv_event(self, category_str: str) -> EventCategory:
        """Categorize İKSV events"""
        category_lower = category_str.lower()
        
        if any(word in category_lower for word in ['music', 'concert', 'jazz', 'salon']):
            return EventCategory.MUSIC
        elif any(word in category_lower for word in ['theater', 'theatre', 'drama', 'play']):
            return EventCategory.THEATER
        elif any(word in category_lower for word in ['dance', 'ballet', 'choreography']):
            return EventCategory.DANCE
        elif any(word in category_lower for word in ['art', 'exhibition', 'gallery', 'visual']):
            return EventCategory.ART
        elif any(word in category_lower for word in ['film', 'cinema', 'movie', 'screening']):
            return EventCategory.FILM
        else:
            return EventCategory.CULTURAL

    def _get_district_for_venue(self, venue_name: str) -> str:
        """Get district for venue"""
        venue_districts = {
            'zorlu psm': 'Beşiktaş',
            'salon iksv': 'Beyoğlu',
            'salon İKSV': 'Beyoğlu', 
            'harbiye muhsin ertuğrul stage': 'Şişli',
            'cemal reşit rey concert hall': 'Şişli',
            'atatürk cultural center': 'Beyoğlu',
            'akm': 'Beyoğlu',
        }
        
        venue_key = venue_name.lower().strip()
        return venue_districts.get(venue_key, 'İstanbul')

    def _get_fallback_iksv_events(self) -> List[IstanbulEvent]:
        """Provide fallback İKSV events when the main system is unavailable"""
        fallback_events = [
            IstanbulEvent(
                title="İstanbul Jazz Festival",
                description="Annual international jazz festival featuring world-class musicians",
                venue="Salon İKSV",
                district="Beyoğlu",
                category=EventCategory.MUSIC,
                organizer="İKSV",
                url="https://www.iksv.org/en/jazz",
                is_free=False,
                venue_lat=41.0369,
                venue_lng=28.9850
            ),
            IstanbulEvent(
                title="İstanbul Theatre Festival",
                description="International theatre festival showcasing contemporary performances",
                venue="Zorlu PSM",
                district="Beşiktaş", 
                category=EventCategory.THEATER,
                organizer="İKSV",
                url="https://www.iksv.org/en/theatre",
                is_free=False,
                venue_lat=41.0677,
                venue_lng=29.0197
            ),
            IstanbulEvent(
                title="İstanbul Biennial",
                description="Contemporary art biennial featuring international artists",
                venue="Various İKSV Venues",
                district="İstanbul",
                category=EventCategory.ART,
                organizer="İKSV",
                url="https://www.iksv.org/en/biennial",
                is_free=False,
                venue_lat=41.0369,
                venue_lng=28.9850
            )
        ]
        
        return fallback_events

    async def find_nearby_events(self, location: DetectedLocation, radius_km: float = 5.0) -> List[IstanbulEvent]:
        """Find events near the detected location"""
        if not location.latitude or not location.longitude:
            return []
        
        nearby_events = []
        
        try:
            # Fetch events from İKSV
            iksv_events = await self.fetch_iksv_events()
            
            for event in iksv_events:
                if event.venue_lat and event.venue_lng:
                    distance = self._calculate_distance(
                        location.latitude, location.longitude,
                        event.venue_lat, event.venue_lng
                    )
                    
                    if distance <= radius_km:
                        nearby_events.append(event)
                        
            # Sort by distance
            nearby_events.sort(key=lambda e: self._calculate_distance(
                location.latitude, location.longitude,
                e.venue_lat or 0, e.venue_lng or 0
            ))
            
            # Also add static events based on district/neighborhood
            static_events = self._get_static_events_for_location(location)
            nearby_events.extend(static_events)
            
            self.logger.info(f"🎭 Found {len(nearby_events)} events near location")
            
        except Exception as e:
            self.logger.warning(f"Error finding nearby events: {e}")
            
        return nearby_events[:10]  # Limit to 10 events
    
    def _get_static_events_for_location(self, location: DetectedLocation) -> List[IstanbulEvent]:
        """Get static/recurring events based on location"""
        static_events = []
        
        # Events by district
        district_events = {
            'Fatih': [
                IstanbulEvent(
                    title="Grand Bazaar Traditional Crafts",
                    category=EventCategory.CULTURAL,
                    venue="Grand Bazaar",
                    venue_lat=41.0106,
                    venue_lng=28.9681,
                    district="Fatih",
                    neighborhood="Beyazıt",
                    description="Traditional Turkish crafts and carpet weaving demonstrations",
                    is_recurring=True,
                    organizer="Grand Bazaar Artisans"
                ),
                IstanbulEvent(
                    title="Whirling Dervishes Ceremony",
                    category=EventCategory.CULTURAL,
                    venue="Hodjapasha Cultural Center",
                    venue_lat=41.0147,
                    venue_lng=28.9761,
                    district="Fatih",
                    neighborhood="Sirkeci",
                    description="Traditional Sufi whirling ceremony",
                    is_recurring=True,
                    organizer="Hodjapasha Cultural Center"
                )
            ],
            'Beyoğlu': [
                IstanbulEvent(
                    title="Istiklal Street Street Performances",
                    category=EventCategory.MUSIC,
                    venue="Istiklal Street",
                    venue_lat=41.0342,
                    venue_lng=28.9784,
                    district="Beyoğlu",
                    neighborhood="Beyoğlu",
                    description="Daily street musicians and performers",
                    is_recurring=True,
                    is_free=True,
                    organizer="Street Artists"
                ),
                IstanbulEvent(
                    title="Galata Tower Evening Views",
                    category=EventCategory.CULTURAL,
                    venue="Galata Tower",
                    venue_lat=41.0256,
                    venue_lng=28.9744,
                    district="Beyoğlu",
                    neighborhood="Galata",
                    description="Panoramic Istanbul views and photography",
                    is_recurring=True,
                    organizer="Galata Tower"
                )
            ],
            'Beşiktaş': [
                IstanbulEvent(
                    title="Ortaköy Weekend Market",
                    category=EventCategory.FOOD,
                    venue="Ortaköy Square",
                    venue_lat=41.0553,
                    venue_lng=29.0275,
                    district="Beşiktaş",
                    neighborhood="Ortaköy",
                    description="Traditional Turkish food and local crafts",
                    is_recurring=True,
                    is_free=True,
                    organizer="Ortaköy Municipality"
                )
            ],
            'Kadıköy': [
                IstanbulEvent(
                    title="Kadıköy Tuesday Market",
                    category=EventCategory.FOOD,
                    venue="Kadıköy Market Area",
                    venue_lat=41.0066,
                    venue_lng=29.0297,
                    district="Kadıköy",
                    neighborhood="Kadıköy",
                    description="Fresh produce and local food market",
                    is_recurring=True,
                    is_free=True,
                    organizer="Kadıköy Municipality"
                )
            ]
        }
        
        # Add events for user's district
        if location.district and location.district in district_events:
            static_events.extend(district_events[location.district])
            
        return static_events

    # Global instance
intelligent_location_detector = IntelligentLocationDetector()

async def detect_user_location(text: str, user_context: Optional[Dict] = None, ip_address: Optional[str] = None, include_events: bool = True) -> DetectedLocation:
    """
    Main function to detect user location from various sources and nearby events
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
        
        # Find nearby events if requested
        if include_events:
            try:
                nearby_events = await detector.find_nearby_events(location)
                location.nearby_events = nearby_events
                if nearby_events:
                    detector.logger.info(f"🎭 Added {len(nearby_events)} nearby events to location")
            except Exception as e:
                detector.logger.warning(f"Error finding events: {e}")
    
    return location

async def get_events_for_location(latitude: float, longitude: float, radius_km: float = 5.0) -> List[IstanbulEvent]:
    """
    Get events for a specific location
    """
    detector = intelligent_location_detector
    location = DetectedLocation(latitude=latitude, longitude=longitude)
    return await detector.find_nearby_events(location, radius_km)