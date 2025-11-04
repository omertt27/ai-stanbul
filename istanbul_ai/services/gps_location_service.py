#!/usr/bin/env python3
"""
GPS Location Service for Istanbul AI
Handles GPS coordinate processing, district mapping, and location accuracy
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class GPSAccuracy(Enum):
    """GPS accuracy levels"""
    HIGH = "high"      # < 10 meters
    MEDIUM = "medium"  # 10-50 meters  
    LOW = "low"        # > 50 meters
    UNKNOWN = "unknown"

@dataclass
class GPSLocation:
    """GPS location with metadata"""
    latitude: float
    longitude: float
    accuracy: Optional[float] = None  # meters
    timestamp: Optional[datetime] = None
    source: str = "user_provided"

class DistrictBoundary(NamedTuple):
    """District boundary information"""
    name: str
    center_lat: float
    center_lng: float
    radius: float  # approximate radius in degrees
    landmarks: List[str]

class GPSLocationService:
    """
    Advanced GPS location service for Istanbul
    Provides accurate district mapping and location services
    """
    
    def __init__(self):
        self.logger = logger
        
        # Detailed Istanbul district boundaries with landmarks
        self.district_boundaries = {
            'Sultanahmet': DistrictBoundary(
                name='Sultanahmet',
                center_lat=41.0086,
                center_lng=28.9802,
                radius=0.01,
                landmarks=['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace', 'Grand Bazaar']
            ),
            'Beyoğlu': DistrictBoundary(
                name='Beyoğlu',
                center_lat=41.0362,
                center_lng=28.9773,
                radius=0.015,
                landmarks=['Galata Tower', 'Istiklal Street', 'Pera Museum']
            ),
            'Taksim': DistrictBoundary(
                name='Taksim',
                center_lat=41.0370,
                center_lng=28.9850,
                radius=0.008,
                landmarks=['Taksim Square', 'Gezi Park', 'Istiklal Avenue']
            ),
            'Kadıköy': DistrictBoundary(
                name='Kadıköy',
                center_lat=40.9833,
                center_lng=29.0333,
                radius=0.02,
                landmarks=['Kadıköy Market', 'Moda Park', 'Bağdat Avenue']
            ),
            'Beşiktaş': DistrictBoundary(
                name='Beşiktaş',
                center_lat=41.0422,
                center_lng=29.0097,
                radius=0.015,
                landmarks=['Dolmabahçe Palace', 'Vodafone Park', 'Barbaros Boulevard']
            ),
            'Galata': DistrictBoundary(
                name='Galata',
                center_lat=41.0256,
                center_lng=28.9744,
                radius=0.008,
                landmarks=['Galata Tower', 'Galata Bridge', 'Istanbul Modern']
            ),
            'Karaköy': DistrictBoundary(
                name='Karaköy',
                center_lat=41.0257,
                center_lng=28.9739,
                radius=0.005,
                landmarks=['Galata Bridge', 'Salt Galata', 'Karaköy Port']
            ),
            'Levent': DistrictBoundary(
                name='Levent',
                center_lat=41.0766,
                center_lng=29.0092,
                radius=0.012,
                landmarks=['Kanyon Mall', 'Sapphire Tower', 'Metro City']
            ),
            'Şişli': DistrictBoundary(
                name='Şişli',
                center_lat=41.0608,
                center_lng=28.9866,
                radius=0.015,
                landmarks=['Cevahir Mall', 'Military Museum', 'Osmanbey']
            ),
            'Nişantaşı': DistrictBoundary(
                name='Nişantaşı',
                center_lat=41.0489,
                center_lng=28.9944,
                radius=0.008,
                landmarks=['City\'s Mall', 'Abdi İpekçi Street', 'Maçka Park']
            ),
            'Ortaköy': DistrictBoundary(
                name='Ortaköy',
                center_lat=41.0552,
                center_lng=29.0267,
                radius=0.006,
                landmarks=['Ortaköy Mosque', 'Bosphorus Bridge', 'Ortaköy Market']
            ),
            'Üsküdar': DistrictBoundary(
                name='Üsküdar',
                center_lat=41.0214,
                center_lng=29.0206,
                radius=0.02,
                landmarks=['Maiden\'s Tower', 'Çamlıca Hill', 'Mihrimah Sultan Mosque']
            ),
            'Eminönü': DistrictBoundary(
                name='Eminönü',
                center_lat=41.0172,
                center_lng=28.9709,
                radius=0.008,
                landmarks=['Spice Bazaar', 'New Mosque', 'Galata Bridge']
            ),
            'Cihangir': DistrictBoundary(
                name='Cihangir',
                center_lat=41.0315,
                center_lng=28.9794,
                radius=0.005,
                landmarks=['Cihangir Park', 'Firuzağa Mosque']
            ),
            'Arnavutköy': DistrictBoundary(
                name='Arnavutköy',
                center_lat=41.0706,
                center_lng=29.0424,
                radius=0.01,
                landmarks=['Arnavutköy Waterfront', 'Historic Houses']
            ),
            'Bebek': DistrictBoundary(
                name='Bebek',
                center_lat=41.0838,
                center_lng=29.0432,
                radius=0.008,
                landmarks=['Bebek Park', 'Bebek Bay', 'Boğaziçi University']
            ),
            'Bostancı': DistrictBoundary(
                name='Bostancı',
                center_lat=40.9658,
                center_lng=29.0906,
                radius=0.015,
                landmarks=['Bostancı Marina', 'Bostancı Beach']
            ),
            'Fenerbahçe': DistrictBoundary(
                name='Fenerbahçe',
                center_lat=40.9638,
                center_lng=29.0469,
                radius=0.01,
                landmarks=['Fenerbahçe Stadium', 'Fenerbahçe Park']
            ),
            'Moda': DistrictBoundary(
                name='Moda',
                center_lat=40.9826,
                center_lng=29.0252,
                radius=0.008,
                landmarks=['Moda Park', 'Moda Pier', 'Kurbağalıdere']
            ),
            'Balat': DistrictBoundary(
                name='Balat',
                center_lat=41.0289,
                center_lng=28.9487,
                radius=0.008,
                landmarks=['Balat Houses', 'Bulgarian St. Stephen Church']
            ),
            'Fener': DistrictBoundary(
                name='Fener',
                center_lat=41.0336,
                center_lng=28.9464,
                radius=0.006,
                landmarks=['Fener Greek Patriarchate', 'Historic Houses']
            )
        }
        
        # Areas that are close to each other and might cause confusion
        self.adjacent_districts = {
            'Sultanahmet': ['Eminönü', 'Beyoğlu'],
            'Beyoğlu': ['Galata', 'Taksim', 'Cihangir'],
            'Taksim': ['Beyoğlu', 'Şişli', 'Nişantaşı'],
            'Kadıköy': ['Moda', 'Fenerbahçe'],
            'Beşiktaş': ['Ortaköy', 'Şişli'],
            'Galata': ['Karaköy', 'Beyoğlu'],
            'Karaköy': ['Galata', 'Eminönü'],
            'Nişantaşı': ['Şişli', 'Taksim'],
            'Moda': ['Kadıköy'],
            'Cihangir': ['Beyoğlu', 'Galata'],
            'Balat': ['Fener'],
            'Fener': ['Balat']
        }

    def detect_district_from_gps(
        self, 
        gps_location: GPSLocation,
        include_confidence: bool = True
    ) -> Dict[str, any]:
        """
        Detect district from GPS coordinates with confidence scoring
        """
        
        lat, lng = gps_location.latitude, gps_location.longitude
        
        # Validate coordinates are in Istanbul area (rough bounds)
        if not self._is_in_istanbul_bounds(lat, lng):
            return {
                'district': None,
                'confidence': 0.0,
                'error': 'Coordinates appear to be outside Istanbul area',
                'suggestions': ['Sultanahmet', 'Taksim', 'Beyoğlu']  # Popular areas
            }
        
        # Find closest districts
        district_distances = []
        
        for district_name, boundary in self.district_boundaries.items():
            distance = self._calculate_distance(
                lat, lng, 
                boundary.center_lat, 
                boundary.center_lng
            )
            
            district_distances.append({
                'district': district_name,
                'distance': distance,
                'within_radius': distance <= boundary.radius,
                'boundary': boundary
            })
        
        # Sort by distance
        district_distances.sort(key=lambda x: x['distance'])
        
        closest = district_distances[0]
        second_closest = district_distances[1] if len(district_distances) > 1 else None
        
        # Calculate confidence
        confidence = self._calculate_confidence(closest, second_closest, gps_location)
        
        result = {
            'district': closest['district'],
            'confidence': confidence,
            'distance_km': self._degrees_to_km(closest['distance']),
            'within_district_bounds': closest['within_radius'],
            'landmarks': closest['boundary'].landmarks,
            'alternatives': [d['district'] for d in district_distances[1:4]]  # Top 3 alternatives
        }
        
        # Add accuracy warnings
        if gps_location.accuracy and gps_location.accuracy > 100:
            result['warning'] = 'Low GPS accuracy may affect district detection'
        
        # Check for adjacent districts if confidence is low
        if confidence < 0.7:
            adjacent = self.adjacent_districts.get(closest['district'], [])
            result['adjacent_districts'] = adjacent
            result['suggestion'] = f"You appear to be in or near {closest['district']}"
        
        return result

    def get_district_info(self, district_name: str) -> Optional[Dict[str, any]]:
        """Get detailed information about a district"""
        boundary = self.district_boundaries.get(district_name)
        if not boundary:
            return None
        
        return {
            'name': boundary.name,
            'center_coordinates': {
                'lat': boundary.center_lat,
                'lng': boundary.center_lng
            },
            'radius_km': self._degrees_to_km(boundary.radius),
            'landmarks': boundary.landmarks,
            'adjacent_districts': self.adjacent_districts.get(district_name, [])
        }

    def get_distance_between_districts(self, district1: str, district2: str) -> Optional[float]:
        """Get distance between two districts in kilometers"""
        boundary1 = self.district_boundaries.get(district1)
        boundary2 = self.district_boundaries.get(district2)
        
        if not boundary1 or not boundary2:
            return None
        
        distance_degrees = self._calculate_distance(
            boundary1.center_lat, boundary1.center_lng,
            boundary2.center_lat, boundary2.center_lng
        )
        
        return self._degrees_to_km(distance_degrees)

    def find_nearby_districts(self, district_name: str, max_distance_km: float = 5.0) -> List[Dict[str, any]]:
        """Find districts within a certain distance"""
        center_boundary = self.district_boundaries.get(district_name)
        if not center_boundary:
            return []
        
        nearby = []
        max_distance_degrees = self._km_to_degrees(max_distance_km)
        
        for name, boundary in self.district_boundaries.items():
            if name == district_name:
                continue
            
            distance = self._calculate_distance(
                center_boundary.center_lat, center_boundary.center_lng,
                boundary.center_lat, boundary.center_lng
            )
            
            if distance <= max_distance_degrees:
                nearby.append({
                    'district': name,
                    'distance_km': self._degrees_to_km(distance),
                    'landmarks': boundary.landmarks
                })
        
        return sorted(nearby, key=lambda x: x['distance_km'])

    def validate_gps_location(self, gps_data: Dict[str, any]) -> GPSLocation:
        """Validate and create GPSLocation from raw GPS data"""
        lat = gps_data.get('lat') or gps_data.get('latitude')
        lng = gps_data.get('lng') or gps_data.get('longitude')
        
        if lat is None or lng is None:
            raise ValueError("GPS data must contain latitude and longitude")
        
        # Basic validation
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lng <= 180):
            raise ValueError(f"Invalid longitude: {lng}")
        
        return GPSLocation(
            latitude=float(lat),
            longitude=float(lng),
            accuracy=gps_data.get('accuracy'),
            timestamp=datetime.now(),
            source=gps_data.get('source', 'user_provided')
        )

    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in degrees (simple Euclidean)"""
        return math.sqrt((lat2 - lat1)**2 + (lng2 - lng1)**2)

    def _calculate_haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance using Haversine formula (more accurate for larger distances)"""
        R = 6371  # Earth radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    def _degrees_to_km(self, degrees: float) -> float:
        """Convert degrees to kilometers (approximate)"""
        return degrees * 111.32  # Average km per degree at Istanbul's latitude

    def _km_to_degrees(self, km: float) -> float:
        """Convert kilometers to degrees (approximate)"""
        return km / 111.32

    def _is_in_istanbul_bounds(self, lat: float, lng: float) -> bool:
        """Check if coordinates are within Istanbul metropolitan area"""
        # Istanbul rough bounds
        min_lat, max_lat = 40.8, 41.35
        min_lng, max_lng = 28.7, 29.35
        
        return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng

    def _calculate_confidence(
        self, 
        closest: Dict[str, any], 
        second_closest: Optional[Dict[str, any]], 
        gps_location: GPSLocation
    ) -> float:
        """Calculate confidence score for district detection"""
        base_confidence = 0.8
        
        # Reduce confidence if outside district radius
        if not closest['within_radius']:
            base_confidence *= 0.7
        
        # Reduce confidence if GPS accuracy is poor
        if gps_location.accuracy:
            if gps_location.accuracy > 100:  # > 100 meters
                base_confidence *= 0.6
            elif gps_location.accuracy > 50:  # > 50 meters
                base_confidence *= 0.8
        
        # Reduce confidence if there's another district very close
        if second_closest and closest['distance'] > 0:
            distance_ratio = second_closest['distance'] / closest['distance']
            if distance_ratio < 1.5:  # Very close to another district
                base_confidence *= 0.7
        
        # Increase confidence if well within district bounds
        if closest['within_radius'] and closest['distance'] < closest['boundary'].radius * 0.5:
            base_confidence = min(0.95, base_confidence * 1.1)
        
        return round(base_confidence, 2)

    def get_all_districts(self) -> List[str]:
        """Get list of all supported districts"""
        return list(self.district_boundaries.keys())

    def get_popular_districts(self) -> List[str]:
        """Get list of popular/touristy districts"""
        return [
            'Sultanahmet', 'Taksim', 'Beyoğlu', 'Galata', 'Kadıköy', 
            'Beşiktaş', 'Nişantaşı', 'Ortaköy', 'Üsküdar'
        ]
