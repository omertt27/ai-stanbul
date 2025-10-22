"""
Location Coordinates Service
Provides GPS coordinates for Istanbul locations (restaurants, attractions, neighborhoods)
Phase 1 Map Integration Enhancement
"""

from typing import Dict, List, Optional, Tuple


class LocationCoordinatesService:
    """Service to provide GPS coordinates for Istanbul locations"""
    
    def __init__(self):
        self.initialize_coordinates()
    
    def initialize_coordinates(self):
        """Initialize coordinate databases for various location types"""
        
        # Restaurant coordinates
        self.restaurants = {
            'pandeli': {'lat': 41.0164, 'lng': 28.9707, 'address': 'Mısır Çarşısı No:1, Eminönü'},
            'hünkar': {'lat': 41.0156, 'lng': 28.9880, 'address': 'Mimar Kemalettin Cad, Fatih'},
            'balıkçı sabahattin': {'lat': 41.0067, 'lng': 28.9780, 'address': 'Seyit Hasan Kuyu Sok, Sultanahmet'},
            'tarihi eminönü balık ekmek': {'lat': 41.0171, 'lng': 28.9732, 'address': 'Eminönü İskelesi'},
            'nusr-et': {'lat': 41.0431, 'lng': 29.0098, 'address': 'Kuruçeşme, Ortaköy'},
            'mikla': {'lat': 41.0342, 'lng': 28.9785, 'address': 'The Marmara Pera, Beyoğlu'},
            'karaköy lokantası': {'lat': 41.0242, 'lng': 28.9742, 'address': 'Kemankeş Caddesi, Karaköy'},
            'çiya sofrası': {'lat': 40.9881, 'lng': 29.0287, 'address': 'Caferağa Mahallesi, Kadıköy'},
            'tarihi sultanahmet köftecisi': {'lat': 41.0059, 'lng': 28.9769, 'address': 'Divanyolu Caddesi, Sultanahmet'},
            'vefa bozacısı': {'lat': 41.0153, 'lng': 28.9680, 'address': 'Katip Çelebi Mahallesi, Vefa'},
        }
        
        # Attraction coordinates
        self.attractions = {
            'hagia sophia': {'lat': 41.0086, 'lng': 28.9802, 'address': 'Sultan Ahmet Mahallesi, Fatih'},
            'topkapi palace': {'lat': 41.0115, 'lng': 28.9833, 'address': 'Cankurtaran Mahallesi, Fatih'},
            'blue mosque': {'lat': 41.0054, 'lng': 28.9768, 'address': 'Sultan Ahmet Mahallesi, Fatih'},
            'grand bazaar': {'lat': 41.0106, 'lng': 28.9680, 'address': 'Beyazıt, Fatih'},
            'basilica cistern': {'lat': 41.0084, 'lng': 28.9778, 'address': 'Alemdar Mahallesi, Fatih'},
            'galata tower': {'lat': 41.0256, 'lng': 28.9744, 'address': 'Bereketzade Mahallesi, Beyoğlu'},
            'dolmabahçe palace': {'lat': 41.0392, 'lng': 29.0000, 'address': 'Vişnezade Mahallesi, Beşiktaş'},
            'spice bazaar': {'lat': 41.0165, 'lng': 28.9707, 'address': 'Rüstem Paşa Mahallesi, Fatih'},
            'istanbul archaeology museums': {'lat': 41.0117, 'lng': 28.9811, 'address': 'Cankurtaran Mahallesi, Fatih'},
            'süleymaniye mosque': {'lat': 41.0165, 'lng': 28.9640, 'address': 'Süleymaniye Mahallesi, Fatih'},
            'chora church': {'lat': 41.0308, 'lng': 28.9381, 'address': 'Dervişali Mahallesi, Fatih'},
            'taksim square': {'lat': 41.0369, 'lng': 28.9850, 'address': 'Taksim Meydanı, Beyoğlu'},
            'istanbul modern': {'lat': 41.0262, 'lng': 28.9742, 'address': 'Karaköy, Beyoğlu'},
            'pera museum': {'lat': 41.0316, 'lng': 28.9745, 'address': 'Asmalımescit Mahallesi, Beyoğlu'},
            'maiden\'s tower': {'lat': 41.0210, 'lng': 29.0042, 'address': 'Salacak Mahallesi, Üsküdar'},
        }
        
        # Neighborhood center coordinates
        self.neighborhoods = {
            'sultanahmet': {'lat': 41.0086, 'lng': 28.9780, 'address': 'Historic Peninsula, Fatih'},
            'taksim': {'lat': 41.0369, 'lng': 28.9850, 'address': 'Beyoğlu District'},
            'beyoğlu': {'lat': 41.0342, 'lng': 28.9785, 'address': 'Beyoğlu District'},
            'kadıköy': {'lat': 40.9881, 'lng': 29.0287, 'address': 'Asian Side, Kadıköy'},
            'beşiktaş': {'lat': 41.0420, 'lng': 29.0079, 'address': 'European Side, Beşiktaş'},
            'üsküdar': {'lat': 41.0220, 'lng': 29.0159, 'address': 'Asian Side, Üsküdar'},
            'ortaköy': {'lat': 41.0552, 'lng': 29.0274, 'address': 'Beşiktaş District'},
            'karaköy': {'lat': 41.0242, 'lng': 28.9742, 'address': 'Beyoğlu District'},
            'eminönü': {'lat': 41.0171, 'lng': 28.9732, 'address': 'Fatih District'},
            'galata': {'lat': 41.0256, 'lng': 28.9744, 'address': 'Beyoğlu District'},
            'balat': {'lat': 41.0289, 'lng': 28.9486, 'address': 'Fatih District'},
            'fener': {'lat': 41.0291, 'lng': 28.9495, 'address': 'Fatih District'},
            'cihangir': {'lat': 41.0331, 'lng': 28.9818, 'address': 'Beyoğlu District'},
            'nişantaşı': {'lat': 41.0456, 'lng': 28.9930, 'address': 'Şişli District'},
            'bebek': {'lat': 41.0790, 'lng': 29.0435, 'address': 'Beşiktaş District'},
        }
        
        # Transport hub coordinates
        self.transport_hubs = {
            'atatürk airport': {'lat': 40.9769, 'lng': 28.8146, 'address': 'Yeşilköy, Bakırköy'},
            'istanbul airport': {'lat': 41.2615, 'lng': 28.7419, 'address': 'Arnavutköy'},
            'sabiha gökçen airport': {'lat': 40.8989, 'lng': 29.3092, 'address': 'Pendik'},
            'sirkeci station': {'lat': 41.0164, 'lng': 28.9764, 'address': 'Fatih'},
            'haydarpaşa station': {'lat': 40.9990, 'lng': 29.0167, 'address': 'Kadıköy'},
            'kabataş ferry terminal': {'lat': 41.0311, 'lng': 29.0097, 'address': 'Beyoğlu'},
            'eminönü ferry terminal': {'lat': 41.0171, 'lng': 28.9732, 'address': 'Fatih'},
            'kadıköy ferry terminal': {'lat': 40.9881, 'lng': 29.0287, 'address': 'Kadıköy'},
        }
    
    def get_coordinates(self, location_name: str, location_type: str = 'auto') -> Optional[Dict]:
        """
        Get coordinates for a location by name
        
        Args:
            location_name: Name of the location (case-insensitive)
            location_type: Type of location ('restaurant', 'attraction', 'neighborhood', 'transport', 'auto')
        
        Returns:
            Dictionary with lat, lng, and address, or None if not found
        """
        location_lower = location_name.lower().strip()
        
        # Search based on location type
        if location_type == 'auto':
            # Search all types
            for db in [self.restaurants, self.attractions, self.neighborhoods, self.transport_hubs]:
                if location_lower in db:
                    return db[location_lower]
                # Try partial match
                for key in db.keys():
                    if location_lower in key or key in location_lower:
                        return db[key]
        elif location_type == 'restaurant':
            return self._search_in_db(location_lower, self.restaurants)
        elif location_type == 'attraction':
            return self._search_in_db(location_lower, self.attractions)
        elif location_type == 'neighborhood':
            return self._search_in_db(location_lower, self.neighborhoods)
        elif location_type == 'transport':
            return self._search_in_db(location_lower, self.transport_hubs)
        
        return None
    
    def _search_in_db(self, location_lower: str, db: Dict) -> Optional[Dict]:
        """Search for location in a specific database"""
        # Exact match
        if location_lower in db:
            return db[location_lower]
        
        # Partial match
        for key in db.keys():
            if location_lower in key or key in location_lower:
                return db[key]
        
        return None
    
    def get_multiple_coordinates(self, location_names: List[str], location_type: str = 'auto') -> List[Dict]:
        """
        Get coordinates for multiple locations
        
        Args:
            location_names: List of location names
            location_type: Type of locations
        
        Returns:
            List of dictionaries with name, lat, lng, and address
        """
        results = []
        for name in location_names:
            coords = self.get_coordinates(name, location_type)
            if coords:
                results.append({
                    'name': name,
                    **coords
                })
        return results
    
    def get_nearby_locations(self, lat: float, lng: float, radius_km: float = 2.0, 
                           location_type: str = 'auto') -> List[Dict]:
        """
        Get locations within a radius of a point
        
        Args:
            lat: Latitude of center point
            lng: Longitude of center point
            radius_km: Radius in kilometers
            location_type: Type of locations to search
        
        Returns:
            List of nearby locations with distance
        """
        nearby = []
        
        # Select databases to search
        databases = []
        if location_type == 'auto':
            databases = [
                ('restaurant', self.restaurants),
                ('attraction', self.attractions),
                ('neighborhood', self.neighborhoods),
                ('transport', self.transport_hubs)
            ]
        elif location_type == 'restaurant':
            databases = [('restaurant', self.restaurants)]
        elif location_type == 'attraction':
            databases = [('attraction', self.attractions)]
        elif location_type == 'neighborhood':
            databases = [('neighborhood', self.neighborhoods)]
        elif location_type == 'transport':
            databases = [('transport', self.transport_hubs)]
        
        # Search each database
        for loc_type, db in databases:
            for name, coords in db.items():
                distance = self._calculate_distance(lat, lng, coords['lat'], coords['lng'])
                if distance <= radius_km:
                    nearby.append({
                        'name': name,
                        'type': loc_type,
                        'lat': coords['lat'],
                        'lng': coords['lng'],
                        'address': coords['address'],
                        'distance_km': round(distance, 2)
                    })
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance_km'])
        return nearby
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        
        Args:
            lat1, lng1: First point coordinates
            lat2, lng2: Second point coordinates
        
        Returns:
            Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in kilometers
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lng = radians(lng2 - lng1)
        
        a = sin(delta_lat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def enrich_location_data(self, location_name: str, location_type: str = 'auto') -> Dict:
        """
        Get enriched location data including coordinates and metadata
        
        Args:
            location_name: Name of the location
            location_type: Type of location
        
        Returns:
            Dictionary with full location data
        """
        coords = self.get_coordinates(location_name, location_type)
        
        if coords:
            return {
                'name': location_name,
                'coordinates': {
                    'lat': coords['lat'],
                    'lng': coords['lng']
                },
                'address': coords['address'],
                'type': location_type if location_type != 'auto' else 'location',
                'map_ready': True
            }
        else:
            return {
                'name': location_name,
                'coordinates': None,
                'address': None,
                'type': location_type if location_type != 'auto' else 'location',
                'map_ready': False
            }


# Global instance
_location_service = None

def get_location_coordinates_service():
    """Get singleton instance of LocationCoordinatesService"""
    global _location_service
    if _location_service is None:
        _location_service = LocationCoordinatesService()
    return _location_service
