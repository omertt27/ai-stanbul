# -*- coding: utf-8 -*-
"""
Location Database Service
Unified service for querying attractions and museums by GPS proximity.
Integrates the attractions database with the museum database.
Now enhanced with transportation routing recommendations!
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any

from ..utils.gps_utils import calculate_distance, estimate_walking_time, format_gps_coordinates

logger = logging.getLogger(__name__)


class LocationDatabaseService:
    """
    Unified location database service that queries both:
    - Attractions database (JSON with GPS)
    - Museums database (Python dataclass + GPS lookup)
    - Transportation system integration (route recommendations)
    
    Provides a single interface for GPS-based location queries with transport advice.
    """
    
    def __init__(self):
        """Initialize location database service with both data sources."""
        self.attractions = self._load_attractions_database()
        self.museum_gps = self._load_museum_gps_coordinates()
        self.museum_db = self._load_museum_database()
        self.transport_service = self._load_transportation_service()
        
        logger.info(
            f"✅ Location Database Service initialized: "
            f"{len(self.attractions)} attractions, "
            f"{len(self.museum_gps)} museums with GPS"
            f"{', with transportation routing' if self.transport_service else ''}"
        )
    
    def _load_attractions_database(self) -> List[Dict]:
        """Load attractions database from JSON file."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(current_dir), 'data')
            db_path = os.path.join(data_dir, 'attractions_database.json')
            
            if os.path.exists(db_path):
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"✅ Loaded {len(data['attractions'])} attractions")
                    return data['attractions']
            else:
                logger.warning(f"⚠️ Attractions database not found at {db_path}")
                return []
        except Exception as e:
            logger.error(f"❌ Error loading attractions database: {e}")
            return []
    
    def _load_museum_gps_coordinates(self) -> Dict[str, Dict]:
        """Load museum GPS coordinates from JSON file."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(current_dir), 'data')
            gps_path = os.path.join(data_dir, 'museum_gps_coordinates.json')
            
            if os.path.exists(gps_path):
                with open(gps_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"✅ Loaded GPS for {len(data['museums'])} museums")
                    return data['museums']
            else:
                logger.warning(f"⚠️ Museum GPS coordinates not found at {gps_path}")
                return {}
        except Exception as e:
            logger.error(f"❌ Error loading museum GPS coordinates: {e}")
            return {}
    
    def _load_museum_database(self):
        """Load the full museum database for detailed information."""
        try:
            from backend.accurate_museum_database import IstanbulMuseumDatabase
            db = IstanbulMuseumDatabase()
            logger.info(f"✅ Loaded detailed info for {len(db.museums)} museums")
            return db
        except ImportError as e:
            logger.warning(f"⚠️ Museum database not available: {e}")
            return None
    
    def _load_transportation_service(self):
        """Load the enhanced transportation service for route recommendations."""
        try:
            from backend.enhanced_transportation_service import EnhancedTransportationService
            service = EnhancedTransportationService()
            logger.info("✅ Transportation routing service loaded")
            return service
        except Exception as e:
            logger.warning(f"⚠️ Transportation service not available: {e}")
            return None
    
    def get_nearby_locations(
        self,
        user_lat: float,
        user_lon: float,
        radius_km: float = 2.0,
        categories: Optional[List[str]] = None,
        max_results: int = 10,
        location_types: Optional[List[str]] = None,  # ['attraction', 'museum'] or None for both
        include_transport: bool = True  # NEW: Include transport recommendations
    ) -> List[Dict]:
        """
        Get nearby locations (attractions + museums) sorted by distance.
        
        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            radius_km: Search radius in kilometers
            categories: Filter by categories (culture, food, shopping, etc.)
            max_results: Maximum number of results
            location_types: Filter by type ('attraction', 'museum', or both)
            include_transport: Include transport route recommendations for each location
            
        Returns:
            List of locations with distance information, sorted by distance
        """
        nearby_locations = []
        
        # Search attractions (if requested or no filter)
        if not location_types or 'attraction' in location_types:
            nearby_locations.extend(
                self._search_attractions(user_lat, user_lon, radius_km, categories)
            )
        
        # Search museums (if requested or no filter)
        if not location_types or 'museum' in location_types:
            nearby_locations.extend(
                self._search_museums(user_lat, user_lon, radius_km, categories)
            )
        
        # Sort by distance
        nearby_locations.sort(key=lambda x: x['distance_km'])
        
        # Limit results
        results = nearby_locations[:max_results]
        
        # Add transport recommendations if requested
        if include_transport and self.transport_service:
            for location in results:
                location['transport_route'] = self._get_basic_transport_recommendation(
                    (user_lat, user_lon),
                    tuple(location['gps']),
                    location['name']
                )
        
        return results
    
    def _search_attractions(
        self,
        user_lat: float,
        user_lon: float,
        radius_km: float,
        categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search attractions database by GPS proximity."""
        results = []
        
        for attraction in self.attractions:
            attr_lat, attr_lon = attraction['gps']
            distance_m = calculate_distance((user_lat, user_lon), (attr_lat, attr_lon))
            distance_km = distance_m / 1000
            
            # Check if within radius
            if distance_km <= radius_km:
                # Filter by categories if specified
                if categories:
                    if not any(cat in attraction['category'] for cat in categories):
                        continue
                
                # Build location dict
                location = {
                    'type': 'attraction',
                    'id': attraction['id'],
                    'name': attraction['name'],
                    'name_tr': attraction.get('name_tr'),
                    'category': attraction['category'],
                    'gps': attraction['gps'],
                    'distance_km': round(distance_km, 2),
                    'walking_time_min': estimate_walking_time(distance_m),
                    'description': attraction['description'],
                    'visit_duration_min': attraction['visit_duration_min'],
                    'entry_fee_tl': attraction['entry_fee_tl'],
                    'opening_hours': attraction['opening_hours'],
                    'rating': attraction.get('rating'),
                    'popularity_score': attraction.get('popularity_score'),
                    'best_time': attraction.get('best_time'),
                    'district': attraction['district'],
                    'nearby_transport': attraction.get('nearby_transport', [])
                }
                
                results.append(location)
        
        return results
    
    def _search_museums(
        self,
        user_lat: float,
        user_lon: float,
        radius_km: float,
        categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search museums database by GPS proximity."""
        results = []
        
        for museum_id, gps_info in self.museum_gps.items():
            museum_lat, museum_lon = gps_info['gps']
            distance_m = calculate_distance((user_lat, user_lon), (museum_lat, museum_lon))
            distance_km = distance_m / 1000
            
            # Check if within radius
            if distance_km <= radius_km:
                # Get detailed museum info if available
                museum_details = None
                if self.museum_db and museum_id in self.museum_db.museums:
                    museum_details = self.museum_db.museums[museum_id]
                
                # Extract categories from museum details
                museum_categories = ['museum', 'culture', 'history']
                if museum_details:
                    # Add more specific categories based on museum type
                    if 'palace' in museum_details.name.lower():
                        museum_categories.append('architecture')
                    if 'art' in museum_details.name.lower() or 'contemporary' in museum_details.name.lower():
                        museum_categories.append('art')
                    if 'mosque' in museum_details.name.lower() or 'church' in museum_details.name.lower():
                        museum_categories.append('religious')
                
                # Filter by categories if specified
                if categories:
                    if not any(cat in museum_categories for cat in categories):
                        continue
                
                # Build location dict
                location = {
                    'type': 'museum',
                    'id': museum_id,
                    'name': museum_details.name if museum_details else museum_id.replace('_', ' ').title(),
                    'category': museum_categories,
                    'gps': gps_info['gps'],
                    'distance_km': round(distance_km, 2),
                    'walking_time_min': estimate_walking_time(distance_m),
                    'district': gps_info['district'],
                    'address': gps_info['address'],
                    'nearby_transport': gps_info['nearby_transport'],
                }
                
                # Add detailed info if available
                if museum_details:
                    location.update({
                        'description': museum_details.historical_significance,
                        'opening_hours': museum_details.opening_hours,
                        'entry_fee': museum_details.entrance_fee,
                        'visit_duration': museum_details.visiting_duration,
                        'best_time': museum_details.best_time_to_visit,
                        'highlights': museum_details.must_see_highlights,
                        'historical_period': museum_details.historical_period,
                        'architectural_style': museum_details.architectural_style
                    })
                
                results.append(location)
        
        return results
    
    def get_location_details(self, location_id: str, location_type: str = None) -> Optional[Dict]:
        """
        Get detailed information about a specific location.
        
        Args:
            location_id: ID of the location
            location_type: 'attraction' or 'museum' (auto-detect if None)
            
        Returns:
            Detailed location information or None if not found
        """
        # Try attractions first
        if not location_type or location_type == 'attraction':
            for attraction in self.attractions:
                if attraction['id'] == location_id:
                    return {
                        'type': 'attraction',
                        **attraction
                    }
        
        # Try museums
        if not location_type or location_type == 'museum':
            if location_id in self.museum_gps:
                gps_info = self.museum_gps[location_id]
                result = {
                    'type': 'museum',
                    'id': location_id,
                    'gps': gps_info['gps'],
                    'district': gps_info['district'],
                    'address': gps_info['address'],
                    'nearby_transport': gps_info['nearby_transport']
                }
                
                # Add detailed museum info if available
                if self.museum_db and location_id in self.museum_db.museums:
                    museum = self.museum_db.museums[location_id]
                    result.update({
                        'name': museum.name,
                        'description': museum.historical_significance,
                        'historical_period': museum.historical_period,
                        'architect': museum.architect,
                        'key_features': museum.key_features,
                        'opening_hours': museum.opening_hours,
                        'entry_fee': museum.entrance_fee,
                        'visit_duration': museum.visiting_duration,
                        'best_time': museum.best_time_to_visit,
                        'highlights': museum.must_see_highlights,
                        'architectural_style': museum.architectural_style,
                        'photography_allowed': museum.photography_allowed,
                        'accessibility': museum.accessibility,
                        'closing_days': museum.closing_days
                    })
                
                return result
        
        return None
    
    def get_all_categories(self) -> List[str]:
        """Get all available location categories."""
        categories = set()
        
        # From attractions
        for attraction in self.attractions:
            categories.update(attraction['category'])
        
        # Common museum categories
        categories.update(['museum', 'culture', 'history', 'art', 'architecture', 'religious'])
        
        return sorted(list(categories))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'total_locations': len(self.attractions) + len(self.museum_gps),
            'attractions': len(self.attractions),
            'museums': len(self.museum_gps),
            'categories': len(self.get_all_categories()),
            'museum_db_available': self.museum_db is not None
        }
    
    def get_transport_route(
        self,
        user_gps: Tuple[float, float],
        destination_gps: Tuple[float, float],
        destination_name: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed transport route from user location to a destination.
        
        Args:
            user_gps: User's GPS coordinates (lat, lon)
            destination_gps: Destination GPS coordinates (lat, lon)
            destination_name: Name of the destination
            preferences: User preferences (fastest, cheapest, accessible, etc.)
            
        Returns:
            Transport route information with detailed instructions
        """
        if not self.transport_service:
            # Fallback to basic recommendations
            return self._get_basic_transport_recommendation(
                user_gps, destination_gps, destination_name
            )
        
        try:
            # Use the enhanced transportation service
            route = self.transport_service.get_route(
                origin_coords=user_gps,
                destination_coords=destination_gps,
                destination_name=destination_name,
                preferences=preferences or {}
            )
            return route
        except Exception as e:
            logger.error(f"Error getting transport route: {e}")
            return self._get_basic_transport_recommendation(
                user_gps, destination_gps, destination_name
            )
    
    def _get_basic_transport_recommendation(
        self,
        user_gps: Tuple[float, float],
        destination_gps: Tuple[float, float],
        destination_name: str
    ) -> Dict[str, Any]:
        """Fallback basic transport recommendation."""
        distance_m = calculate_distance(user_gps, destination_gps)
        distance_km = distance_m / 1000
        walking_time = estimate_walking_time(distance_m)
        
        recommendations = []
        
        # Walking option
        if distance_km <= 2.0:
            recommendations.append({
                'mode': 'walking',
                'duration_min': walking_time,
                'distance_km': round(distance_km, 2),
                'cost_tl': 0,
                'description': f'Walk directly to {destination_name}',
                'recommended': distance_km <= 1.0
            })
        
        # Public transport option
        if distance_km > 0.5:
            est_transit_time = max(15, int(distance_km * 15))
            recommendations.append({
                'mode': 'public_transport',
                'duration_min': est_transit_time,
                'distance_km': round(distance_km, 2),
                'cost_tl': 15,
                'description': f'Use metro/tram/bus to {destination_name}',
                'recommended': distance_km > 1.0 and distance_km < 10
            })
        
        # Taxi option
        if distance_km > 2.0:
            taxi_time = max(10, int(distance_km * 4))
            taxi_cost = max(45, 15 + (distance_km * 10))
            recommendations.append({
                'mode': 'taxi',
                'duration_min': taxi_time,
                'distance_km': round(distance_km, 2),
                'cost_tl': round(taxi_cost),
                'description': f'Take a taxi to {destination_name}',
                'recommended': distance_km > 10
            })
        
        return {
            'origin': user_gps,
            'destination': destination_gps,
            'destination_name': destination_name,
            'distance_km': round(distance_km, 2),
            'options': recommendations,
            'fallback_mode': True
        }


# Example usage and testing
if __name__ == '__main__':
    # Initialize service
    service = LocationDatabaseService()
    
    # Print statistics
    stats = service.get_statistics()
    print("\n📊 Location Database Statistics:")
    print(f"   Total locations: {stats['total_locations']}")
    print(f"   Attractions: {stats['attractions']}")
    print(f"   Museums: {stats['museums']}")
    print(f"   Categories: {stats['categories']}")
    print(f"   Museum details available: {stats['museum_db_available']}")
    
    # Test query: Find locations near Sultanahmet
    print("\n\n📍 Test Query: Locations near Sultanahmet (41.0086, 28.9802)")
    user_location = (41.0086, 28.9802)  # Hagia Sophia
    
    nearby = service.get_nearby_locations(
        user_location[0],
        user_location[1],
        radius_km=1.0,
        max_results=10
    )
    
    print(f"\n✅ Found {len(nearby)} locations within 1 km:\n")
    
    for i, location in enumerate(nearby, 1):
        print(f"{i}. {location['name']} ({location['type']})")
        print(f"   📏 {location['distance_km']} km • 🚶 ~{location['walking_time_min']} min walk")
        print(f"   📍 {location['district']}")
        if location.get('rating'):
            print(f"   ⭐ {location['rating']}/5.0")
        print()
    
    # Test transport route: From Sultanahmet to Taksim Square
    print("🚖 Test Transport Route: Sultanahmet to Taksim Square")
    sultanahmet_gps = (41.0086, 28.9802)
    taksim_square_gps = (41.0369, 28.9865)
    
    route = service.get_transport_route(
        user_gps=sultanahmet_gps,
        destination_gps=taksim_square_gps,
        destination_name="Taksim Square",
        preferences={'fastest': True}
    )
    
    print("Route options:")
    if route and 'options' in route:
        for i, option in enumerate(route['options'], 1):
            print(f"  {i}. {option['mode'].title()} - {option['duration_min']} min")
            print(f"     🚶 {option.get('walking_distance', 0)} km walk, "
                  f"🚍 {option.get('transit_lines', 'N/A')}, "
                  f"💰 {option.get('cost_tl', 0)} TL")
    else:
        print("  No route found, basic recommendation:")
        print(f"  🚶 Walk from Sultanahmet to Taksim Square (~{route['distance_km']} km)")
