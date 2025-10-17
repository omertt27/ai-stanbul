"""
POI Database Service for Istanbul AI

Provides access to Points of Interest (museums, palaces, mosques, attractions)
with precomputed station connections, opening hours, ratings, and crowding patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, time
import json
import math
from pathlib import Path


@dataclass
class GeoCoordinate:
    """Geographic coordinate"""
    lat: float
    lon: float
    
    def distance_to(self, other: 'GeoCoordinate') -> float:
        """Calculate distance in km using Haversine formula"""
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(self.lat)
        lat2_rad = math.radians(other.lat)
        delta_lat = math.radians(other.lat - self.lat)
        delta_lon = math.radians(other.lon - self.lon)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


@dataclass
class POI:
    """Point of Interest data model"""
    poi_id: str
    name: str
    name_en: str
    category: str
    subcategory: str
    location: GeoCoordinate
    rating: float  # 0.0-5.0
    popularity_score: float  # 0.0-1.0
    visit_duration_min: int
    opening_hours: Dict[str, Tuple[str, str]]  # {"monday": ("09:00", "17:00")}
    ticket_price: float
    accessibility_score: float
    facilities: List[str] = field(default_factory=list)
    nearest_stations: List[Tuple[str, float, int]] = field(default_factory=list)  # [(station_id, distance_km, walk_time_min)]
    crowding_patterns: Dict[str, List[float]] = field(default_factory=dict)  # {"weekday": [0.3, 0.4, ...]}
    best_visit_times: List[Tuple[int, int]] = field(default_factory=list)  # [(hour_start, hour_end)]
    district: str = ""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    description_en: str = ""
    website: str = ""
    phone: str = ""
    
    def is_open(self, check_time: datetime) -> bool:
        """Check if POI is open at given time"""
        day_name = check_time.strftime('%A').lower()
        
        if day_name not in self.opening_hours:
            return False
        
        open_time_str, close_time_str = self.opening_hours[day_name]
        
        # Handle closed days
        if open_time_str == "closed" or close_time_str == "closed":
            return False
        
        try:
            open_time = datetime.strptime(open_time_str, "%H:%M").time()
            close_time = datetime.strptime(close_time_str, "%H:%M").time()
            current_time = check_time.time()
            
            return open_time <= current_time <= close_time
        except ValueError:
            return False
    
    def get_crowding_level(self, check_time: datetime) -> float:
        """Get predicted crowding level (0.0-1.0) for given time"""
        day_type = "weekend" if check_time.weekday() >= 5 else "weekday"
        hour = check_time.hour
        
        if day_type in self.crowding_patterns:
            patterns = self.crowding_patterns[day_type]
            if 0 <= hour < len(patterns):
                return patterns[hour]
        
        # Default moderate crowding
        return 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'poi_id': self.poi_id,
            'name': self.name,
            'name_en': self.name_en,
            'category': self.category,
            'subcategory': self.subcategory,
            'location': {'lat': self.location.lat, 'lon': self.location.lon},
            'rating': self.rating,
            'popularity_score': self.popularity_score,
            'visit_duration_min': self.visit_duration_min,
            'opening_hours': self.opening_hours,
            'ticket_price': self.ticket_price,
            'accessibility_score': self.accessibility_score,
            'facilities': self.facilities,
            'nearest_stations': self.nearest_stations,
            'crowding_patterns': self.crowding_patterns,
            'best_visit_times': self.best_visit_times,
            'district': self.district,
            'tags': self.tags,
            'description': self.description,
            'description_en': self.description_en,
            'website': self.website,
            'phone': self.phone
        }


class POIDatabaseService:
    """Service for managing and querying POI database"""
    
    # POI Categories
    CATEGORIES = {
        'museum': 'Museums and exhibitions',
        'palace': 'Ottoman palaces and pavilions',
        'mosque': 'Historic mosques and religious sites',
        'viewpoint': 'Towers, hills, observation points',
        'market': 'Bazaars and markets',
        'park': 'Public parks and gardens',
        'cultural': 'Galleries and cultural centers',
        'waterfront': 'Piers and coastal areas'
    }
    
    def __init__(self, data_file: Optional[str] = None):
        """Initialize POI database service"""
        self.pois: Dict[str, POI] = {}
        self.data_file = data_file or str(Path(__file__).parent.parent / 'data' / 'istanbul_pois.json')
        self._load_pois()
    
    def _load_pois(self):
        """Load POIs from JSON file"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for poi_data in data.get('pois', []):
                location = GeoCoordinate(
                    lat=poi_data['location']['lat'],
                    lon=poi_data['location']['lon']
                )
                
                poi = POI(
                    poi_id=poi_data['poi_id'],
                    name=poi_data['name'],
                    name_en=poi_data['name_en'],
                    category=poi_data['category'],
                    subcategory=poi_data['subcategory'],
                    location=location,
                    rating=poi_data['rating'],
                    popularity_score=poi_data['popularity_score'],
                    visit_duration_min=poi_data['visit_duration_min'],
                    opening_hours=poi_data['opening_hours'],
                    ticket_price=poi_data['ticket_price'],
                    accessibility_score=poi_data['accessibility_score'],
                    facilities=poi_data.get('facilities', []),
                    nearest_stations=poi_data.get('nearest_stations', []),
                    crowding_patterns=poi_data.get('crowding_patterns', {}),
                    best_visit_times=poi_data.get('best_visit_times', []),
                    district=poi_data.get('district', ''),
                    tags=poi_data.get('tags', []),
                    description=poi_data.get('description', ''),
                    description_en=poi_data.get('description_en', ''),
                    website=poi_data.get('website', ''),
                    phone=poi_data.get('phone', '')
                )
                
                self.pois[poi.poi_id] = poi
                
            print(f"✅ Loaded {len(self.pois)} POIs from database")
            
        except FileNotFoundError:
            print(f"⚠️ POI database file not found: {self.data_file}")
            print("   Creating empty database. Use add_poi() to populate.")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing POI database: {e}")
        except Exception as e:
            print(f"❌ Error loading POI database: {e}")
    
    def save_pois(self):
        """Save POIs to JSON file"""
        try:
            # Ensure directory exists
            Path(self.data_file).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'total_pois': len(self.pois),
                'pois': [poi.to_dict() for poi in self.pois.values()]
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved {len(self.pois)} POIs to database")
            
        except Exception as e:
            print(f"❌ Error saving POI database: {e}")
    
    def add_poi(self, poi: POI):
        """Add a POI to the database"""
        self.pois[poi.poi_id] = poi
    
    def get_poi(self, poi_id: str) -> Optional[POI]:
        """Get a POI by ID"""
        return self.pois.get(poi_id)
    
    def get_all_pois(self) -> List[POI]:
        """Get all POIs"""
        return list(self.pois.values())
    
    def get_pois_by_category(self, category: str) -> List[POI]:
        """Get all POIs in a category"""
        return [poi for poi in self.pois.values() if poi.category == category]
    
    def get_pois_by_district(self, district: str) -> List[POI]:
        """Get all POIs in a district"""
        return [poi for poi in self.pois.values() if poi.district.lower() == district.lower()]
    
    def find_pois_near_location(
        self,
        lat: float,
        lon: float,
        radius_km: float = 2.0,
        categories: Optional[List[str]] = None,
        min_rating: float = 0.0,
        open_at: Optional[datetime] = None
    ) -> List[Tuple[POI, float]]:
        """
        Find POIs near a location
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            categories: Filter by categories (None = all)
            min_rating: Minimum rating (0.0-5.0)
            open_at: Only return POIs open at this time (None = no filter)
        
        Returns:
            List of (POI, distance_km) tuples, sorted by distance
        """
        location = GeoCoordinate(lat, lon)
        results = []
        
        for poi in self.pois.values():
            distance = location.distance_to(poi.location)
            
            # Apply filters
            if distance > radius_km:
                continue
            if categories and poi.category not in categories:
                continue
            if poi.rating < min_rating:
                continue
            if open_at and not poi.is_open(open_at):
                continue
            
            results.append((poi, distance))
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results
    
    def find_pois_near_station(
        self,
        station_id: str,
        max_distance_km: float = 1.0,
        categories: Optional[List[str]] = None
    ) -> List[Tuple[POI, float]]:
        """
        Find POIs near a transit station
        
        Args:
            station_id: Station ID
            max_distance_km: Maximum walking distance
            categories: Filter by categories
        
        Returns:
            List of (POI, distance_km) tuples, sorted by distance
        """
        results = []
        
        for poi in self.pois.values():
            # Check if station is in poi's nearest_stations
            for station, distance, walk_time in poi.nearest_stations:
                if station == station_id and distance <= max_distance_km:
                    if not categories or poi.category in categories:
                        results.append((poi, distance))
                    break
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results
    
    def search_pois(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        min_rating: float = 0.0
    ) -> List[POI]:
        """
        Search POIs by name or tags
        
        Args:
            query: Search query
            categories: Filter by categories
            min_rating: Minimum rating
        
        Returns:
            List of matching POIs
        """
        query_lower = query.lower()
        results = []
        
        for poi in self.pois.values():
            # Check if query matches name or tags
            if (query_lower in poi.name.lower() or
                query_lower in poi.name_en.lower() or
                any(query_lower in tag.lower() for tag in poi.tags)):
                
                # Apply filters
                if categories and poi.category not in categories:
                    continue
                if poi.rating < min_rating:
                    continue
                
                results.append(poi)
        
        # Sort by rating
        results.sort(key=lambda x: x.rating, reverse=True)
        return results
    
    def get_popular_pois(
        self,
        limit: int = 10,
        categories: Optional[List[str]] = None,
        district: Optional[str] = None
    ) -> List[POI]:
        """
        Get most popular POIs
        
        Args:
            limit: Maximum number of results
            categories: Filter by categories
            district: Filter by district
        
        Returns:
            List of POIs sorted by popularity
        """
        pois = list(self.pois.values())
        
        # Apply filters
        if categories:
            pois = [p for p in pois if p.category in categories]
        if district:
            pois = [p for p in pois if p.district.lower() == district.lower()]
        
        # Sort by popularity score and rating
        pois.sort(key=lambda x: (x.popularity_score, x.rating), reverse=True)
        
        return pois[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        category_counts = {}
        district_counts = {}
        
        for poi in self.pois.values():
            category_counts[poi.category] = category_counts.get(poi.category, 0) + 1
            if poi.district:
                district_counts[poi.district] = district_counts.get(poi.district, 0) + 1
        
        return {
            'total_pois': len(self.pois),
            'categories': category_counts,
            'districts': district_counts,
            'avg_rating': sum(p.rating for p in self.pois.values()) / len(self.pois) if self.pois else 0.0,
            'avg_visit_duration': sum(p.visit_duration_min for p in self.pois.values()) / len(self.pois) if self.pois else 0
        }


def calculate_walking_time(distance_km: float, speed_kmh: float = 5.0) -> int:
    """
    Calculate walking time in minutes
    
    Args:
        distance_km: Distance in kilometers
        speed_kmh: Walking speed in km/h (default: 5 km/h)
    
    Returns:
        Walking time in minutes
    """
    return int((distance_km / speed_kmh) * 60)


# Singleton instance
_poi_service_instance = None


def get_poi_service(data_file: Optional[str] = None) -> POIDatabaseService:
    """Get singleton instance of POI database service"""
    global _poi_service_instance
    if _poi_service_instance is None:
        _poi_service_instance = POIDatabaseService(data_file)
    return _poi_service_instance
