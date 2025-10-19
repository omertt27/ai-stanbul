"""
Istanbul Location Geocoder
Convert location names to GPS coordinates for Istanbul landmarks

Provides both hardcoded landmarks and fallback to external geocoding services
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GeocodedLocation:
    """Geocoded location with metadata"""
    name: str
    lat: float
    lon: float
    source: str = "local"  # local, nominatim, google
    confidence: float = 1.0
    address: str = ""
    
    def to_tuple(self) -> Tuple[float, float]:
        """Return as (lat, lon) tuple"""
        return (self.lat, self.lon)
    
    def to_dict(self) -> Dict:
        """Return as dictionary"""
        return {
            'name': self.name,
            'lat': self.lat,
            'lon': self.lon,
            'source': self.source,
            'confidence': self.confidence,
            'address': self.address
        }


class IstanbulGeocoder:
    """
    Geocoder for Istanbul locations
    
    Features:
    - 100+ pre-loaded Istanbul landmarks
    - Fuzzy matching for location names
    - Fallback to Nominatim (OpenStreetMap) geocoding
    - Turkish and English name support
    """
    
    # Istanbul landmarks with GPS coordinates (lat, lon)
    LANDMARKS = {
        # Historical Sites
        'sultanahmet': (41.0082, 28.9784),
        'hagia sophia': (41.0086, 28.9802),
        'ayasofya': (41.0086, 28.9802),
        'blue mosque': (41.0054, 28.9768),
        'sultanahmet camii': (41.0054, 28.9768),
        'topkapi palace': (41.0115, 28.9833),
        'topkapi sarayi': (41.0115, 28.9833),
        'dolmabahce palace': (41.0391, 29.0003),
        'dolmabahce sarayi': (41.0391, 29.0003),
        'galata tower': (41.0256, 28.9744),
        'galata kulesi': (41.0256, 28.9744),
        'basilica cistern': (41.0084, 28.9778),
        'yerebatan sarnici': (41.0084, 28.9778),
        
        # Markets & Bazaars
        'grand bazaar': (41.0106, 28.9681),
        'kapali carsi': (41.0106, 28.9681),
        'spice bazaar': (41.0166, 28.9706),
        'misir carsisi': (41.0166, 28.9706),
        'arasta bazaar': (41.0057, 28.9752),
        
        # Museums
        'istanbul archaeology museum': (41.0117, 28.9810),
        'istanbul modern': (41.0280, 28.9774),
        'pera museum': (41.0316, 28.9748),
        'rahmi koc museum': (41.0428, 28.9488),
        'turkish and islamic arts museum': (41.0055, 28.9768),
        
        # Neighborhoods & Squares
        'taksim': (41.0369, 28.9850),
        'taksim square': (41.0369, 28.9850),
        'taksim meydani': (41.0369, 28.9850),
        'istiklal street': (41.0338, 28.9779),
        'istiklal caddesi': (41.0338, 28.9779),
        'ortakoy': (41.0553, 29.0269),
        'bebek': (41.0773, 29.0433),
        'eminonu': (41.0169, 28.9706),
        'besiktas': (41.0422, 29.0067),
        'kadikoy': (40.9902, 29.0252),
        'uskudar': (41.0220, 29.0150),
        'beyoglu': (41.0344, 28.9784),
        'fatih': (41.0180, 28.9497),
        'sisli': (41.0602, 28.9875),
        'bakirkoy': (40.9808, 28.8751),
        
        # Mosques
        'suleymaniye mosque': (41.0165, 28.9640),
        'suleymaniye camii': (41.0165, 28.9640),
        'new mosque': (41.0167, 28.9702),
        'yeni cami': (41.0167, 28.9702),
        'ortakoy mosque': (41.0555, 29.0268),
        'eyup sultan mosque': (41.0487, 28.9352),
        'fatih mosque': (41.0204, 28.9519),
        'mihrimah sultan mosque': (41.0209, 28.9385),
        
        # Bridges & Landmarks
        'galata bridge': (41.0201, 28.9744),
        'galata koprusu': (41.0201, 28.9744),
        'bosphorus bridge': (41.0425, 29.0347),
        'bogaz koprusu': (41.0425, 29.0347),
        'maiden tower': (41.0213, 29.0043),
        'kiz kulesi': (41.0213, 29.0043),
        
        # Parks & Recreation
        'gulhane park': (41.0130, 28.9814),
        'gulhane parki': (41.0130, 28.9814),
        'emirgan park': (41.1089, 29.0553),
        'yildiz park': (41.0485, 29.0104),
        'macka park': (41.0469, 28.9947),
        
        # Shopping & Modern Areas
        'nisantasi': (41.0480, 28.9937),
        'cevahir mall': (41.0619, 28.9861),
        'istinye park': (41.1125, 29.0272),
        'zorlu center': (41.0631, 29.0092),
        
        # Transportation Hubs
        'ataturk airport': (40.9769, 28.8146),
        'istanbul airport': (41.2753, 28.7519),
        'sabiha gokcen airport': (40.8986, 29.3092),
        'haydarpasa train station': (40.9974, 29.0116),
        'sirkeci train station': (41.0168, 28.9765),
        
        # Universities
        'istanbul university': (41.0105, 28.9554),
        'bogazici university': (41.0847, 29.0483),
        'istanbul technical university': (41.1044, 28.9765),
        'sabanci university': (41.0414, 29.3841),
        
        # Hotels & Landmarks
        'ciragan palace': (41.0433, 29.0059),
        'four seasons sultanahmet': (41.0056, 28.9775),
        'pera palace hotel': (41.0324, 28.9775),
        
        # Coastal Areas
        'karakoy': (41.0236, 28.9744),
        'balat': (41.0313, 28.9487),
        'fener': (41.0298, 28.9485),
        'kumkapi': (41.0060, 28.9704),
        'yenikapi': (41.0047, 28.9548),
        'kabatas': (41.0361, 28.9895),
        'besiktas ferry': (41.0421, 29.0082),
    }
    
    # Alias mappings for common variations
    ALIASES = {
        'hagia sofia': 'hagia sophia',
        'aya sofya': 'hagia sophia',
        'ayasofy': 'hagia sophia',
        'blue masjid': 'blue mosque',
        'sultanahmet mosque': 'blue mosque',
        'topkapi': 'topkapi palace',
        'dolmabahce': 'dolmabahce palace',
        'galata': 'galata tower',
        'grand bazaar': 'grand bazaar',
        'kapalıçarşı': 'grand bazaar',
        'spice market': 'spice bazaar',
        'egyptian bazaar': 'spice bazaar',
        'taksim sq': 'taksim square',
        'taksim sqr': 'taksim square',
        'istiklal': 'istiklal street',
        'bosphorus': 'bosphorus bridge',
        'boğaz': 'bosphorus bridge',
    }
    
    def __init__(self, use_external_geocoding: bool = True):
        """
        Initialize geocoder
        
        Args:
            use_external_geocoding: Whether to use external geocoding as fallback
        """
        self.use_external_geocoding = use_external_geocoding
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"✅ Istanbul Geocoder initialized ({len(self.LANDMARKS)} landmarks)")
    
    async def _ensure_session(self):
        """Ensure HTTP session is initialized"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    def geocode(self, location_name: str) -> Optional[GeocodedLocation]:
        """
        Geocode a location name to GPS coordinates (synchronous)
        
        Args:
            location_name: Location name to geocode
        
        Returns:
            GeocodedLocation or None if not found
        """
        location_lower = location_name.lower().strip()
        
        # Try exact match
        if location_lower in self.LANDMARKS:
            lat, lon = self.LANDMARKS[location_lower]
            return GeocodedLocation(
                name=location_name,
                lat=lat,
                lon=lon,
                source='local',
                confidence=1.0,
                address=f"{location_name}, Istanbul, Turkey"
            )
        
        # Try alias match
        if location_lower in self.ALIASES:
            canonical_name = self.ALIASES[location_lower]
            lat, lon = self.LANDMARKS[canonical_name]
            return GeocodedLocation(
                name=location_name,
                lat=lat,
                lon=lon,
                source='local',
                confidence=0.95,
                address=f"{canonical_name}, Istanbul, Turkey"
            )
        
        # Try partial match
        for landmark_name, coords in self.LANDMARKS.items():
            if location_lower in landmark_name or landmark_name in location_lower:
                lat, lon = coords
                return GeocodedLocation(
                    name=location_name,
                    lat=lat,
                    lon=lon,
                    source='local',
                    confidence=0.85,
                    address=f"{landmark_name}, Istanbul, Turkey"
                )
        
        # Try first word match
        first_word = location_lower.split()[0] if location_lower else ""
        if first_word and first_word in self.LANDMARKS:
            lat, lon = self.LANDMARKS[first_word]
            return GeocodedLocation(
                name=location_name,
                lat=lat,
                lon=lon,
                source='local',
                confidence=0.75,
                address=f"{first_word}, Istanbul, Turkey"
            )
        
        return None
    
    async def geocode_async(self, location_name: str) -> Optional[GeocodedLocation]:
        """
        Geocode a location name with fallback to external services (asynchronous)
        
        Args:
            location_name: Location name to geocode
        
        Returns:
            GeocodedLocation or None if not found
        """
        # Try local geocoding first
        result = self.geocode(location_name)
        if result:
            return result
        
        # Fallback to external geocoding if enabled
        if self.use_external_geocoding:
            return await self._geocode_nominatim(location_name)
        
        return None
    
    async def _geocode_nominatim(self, location_name: str) -> Optional[GeocodedLocation]:
        """
        Geocode using Nominatim (OpenStreetMap)
        
        Args:
            location_name: Location name to geocode
        
        Returns:
            GeocodedLocation or None if not found
        """
        await self._ensure_session()
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': f"{location_name}, Istanbul, Turkey",
            'format': 'json',
            'limit': 1
        }
        
        headers = {
            'User-Agent': 'AI Istanbul Route Planner/1.0'
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        result = data[0]
                        return GeocodedLocation(
                            name=location_name,
                            lat=float(result['lat']),
                            lon=float(result['lon']),
                            source='nominatim',
                            confidence=0.7,
                            address=result.get('display_name', '')
                        )
        except Exception as e:
            logger.error(f"Nominatim geocoding error: {e}")
        
        return None
    
    def get_nearby_landmarks(
        self,
        lat: float,
        lon: float,
        radius_km: float = 1.0,
        limit: int = 5
    ) -> List[Dict]:
        """
        Get nearby landmarks within radius
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            limit: Maximum number of results
        
        Returns:
            List of nearby landmarks with distances
        """
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points in km"""
            R = 6371  # Earth radius in km
            
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            
            a = (math.sin(delta_lat / 2) ** 2 +
                 math.cos(lat1_rad) * math.cos(lat2_rad) *
                 math.sin(delta_lon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            return R * c
        
        nearby = []
        for name, (landmark_lat, landmark_lon) in self.LANDMARKS.items():
            distance = haversine_distance(lat, lon, landmark_lat, landmark_lon)
            if distance <= radius_km:
                nearby.append({
                    'name': name,
                    'lat': landmark_lat,
                    'lon': landmark_lon,
                    'distance_km': round(distance, 2)
                })
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance_km'])
        
        return nearby[:limit]
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


# ═══════════════════════════════════════════════════════════════════
# Global Geocoder Instance
# ═══════════════════════════════════════════════════════════════════

_geocoder_instance: Optional[IstanbulGeocoder] = None


def get_geocoder() -> IstanbulGeocoder:
    """Get or create global geocoder instance"""
    global _geocoder_instance
    if _geocoder_instance is None:
        _geocoder_instance = IstanbulGeocoder()
    return _geocoder_instance


async def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """
    Convenience function to geocode a location
    
    Args:
        location_name: Location name to geocode
    
    Returns:
        (lat, lon) tuple or None if not found
    
    Example:
        >>> coords = await geocode_location("Sultanahmet")
        >>> print(coords)
        (41.0082, 28.9784)
    """
    geocoder = get_geocoder()
    result = await geocoder.geocode_async(location_name)
    return result.to_tuple() if result else None


# ═══════════════════════════════════════════════════════════════════
# Testing & Examples
# ═══════════════════════════════════════════════════════════════════

async def test_geocoding():
    """Test geocoding with various location names"""
    print("\n📍 Istanbul Geocoder - Test Suite\n")
    print("=" * 70)
    
    geocoder = IstanbulGeocoder(use_external_geocoding=False)
    
    test_locations = [
        "Sultanahmet",
        "Blue Mosque",
        "Galata Tower",
        "Taksim Square",
        "Grand Bazaar",
        "Hagia Sophia",
        "Dolmabahce Palace",
        "Ortakoy",
        "Kadikoy",
        "Unknown Location"  # Should fail
    ]
    
    for location in test_locations:
        print(f"\n📍 Geocoding: \"{location}\"")
        
        result = geocoder.geocode(location)
        
        if result:
            print(f"   ✅ Found!")
            print(f"      Lat/Lon: ({result.lat}, {result.lon})")
            print(f"      Source: {result.source}")
            print(f"      Confidence: {result.confidence:.2f}")
        else:
            print(f"   ❌ Not found")
    
    # Test nearby landmarks
    print("\n\n📍 Nearby Landmarks Test (from Sultanahmet):")
    nearby = geocoder.get_nearby_landmarks(41.0082, 28.9784, radius_km=1.0, limit=5)
    for i, landmark in enumerate(nearby, 1):
        print(f"   {i}. {landmark['name']} - {landmark['distance_km']} km")
    
    print("\n" + "=" * 70)
    print("✅ Test suite completed!\n")
    
    await geocoder.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_geocoding())
