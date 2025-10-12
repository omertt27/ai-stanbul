#!/usr/bin/env python3
"""
Geo Utilities Service - Fallback implementations for route caching and location services
"""

import math
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic


class GeoUtilities:
    """Basic geo utilities for route caching and location services"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        try:
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        except Exception:
            # Fallback to haversine formula
            return GeoUtilities.haversine_distance(lat1, lon1, lat2, lon2)
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    @staticmethod
    def is_within_bounds(lat: float, lon: float, bounds: Dict[str, float]) -> bool:
        """Check if coordinates are within bounding box"""
        return (bounds['south'] <= lat <= bounds['north'] and 
                bounds['west'] <= lon <= bounds['east'])
    
    @staticmethod
    def get_istanbul_bounds() -> Dict[str, float]:
        """Get Istanbul city bounds"""
        return {
            'north': 41.25,
            'south': 40.80,
            'east': 29.40,
            'west': 28.60
        }
    
    @staticmethod
    def generate_location_hash(lat: float, lon: float, precision: int = 4) -> str:
        """Generate a location hash for caching"""
        lat_rounded = round(lat, precision)
        lon_rounded = round(lon, precision)
        return f"{lat_rounded}_{lon_rounded}"
    
    @staticmethod
    def get_district_from_coordinates(lat: float, lon: float) -> str:
        """Simple district detection based on coordinates"""
        # Simplified district mapping for Istanbul
        districts = {
            'sultanahmet': {'lat_min': 41.000, 'lat_max': 41.015, 'lon_min': 28.975, 'lon_max': 28.985},
            'beyoglu': {'lat_min': 41.025, 'lat_max': 41.040, 'lon_min': 28.970, 'lon_max': 28.985},
            'taksim': {'lat_min': 41.035, 'lat_max': 41.042, 'lon_min': 28.985, 'lon_max': 28.995},
            'galata': {'lat_min': 41.020, 'lat_max': 41.028, 'lon_min': 28.970, 'lon_max': 28.980},
            'kadikoy': {'lat_min': 40.980, 'lat_max': 41.000, 'lon_min': 29.025, 'lon_max': 29.040},
            'besiktas': {'lat_min': 41.035, 'lat_max': 41.050, 'lon_min': 29.000, 'lon_max': 29.015},
            'ortakoy': {'lat_min': 41.045, 'lat_max': 41.055, 'lon_min': 29.020, 'lon_max': 29.030}
        }
        
        for district, bounds in districts.items():
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
                bounds['lon_min'] <= lon <= bounds['lon_max']):
                return district
        
        # Default based on side of Bosphorus
        return 'european_side' if lon < 29.0 else 'asian_side'


class RouteCacheUtilities:
    """Utilities for route caching"""
    
    @staticmethod
    def generate_route_key(start_lat: float, start_lon: float, 
                          end_lat: Optional[float] = None, end_lon: Optional[float] = None,
                          style: str = "balanced", max_distance: float = 5.0) -> str:
        """Generate a cache key for route requests"""
        start_hash = GeoUtilities.generate_location_hash(start_lat, start_lon)
        
        if end_lat and end_lon:
            end_hash = GeoUtilities.generate_location_hash(end_lat, end_lon)
            return f"route_{start_hash}_to_{end_hash}_{style}_{max_distance}"
        else:
            return f"route_{start_hash}_loop_{style}_{max_distance}"
    
    @staticmethod
    def is_similar_route(route1_key: str, route2_key: str, tolerance: float = 0.01) -> bool:
        """Check if two routes are similar enough to use cached result"""
        # Simple string comparison for now
        # In production, this would parse coordinates and compare with tolerance
        return route1_key == route2_key


# Global instances
geo_utils = GeoUtilities()
cache_utils = RouteCacheUtilities()
