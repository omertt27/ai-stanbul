"""
Walking Directions Service for Istanbul AI
==========================================
Adds first-mile/last-mile walking directions to transit routes.

This service:
1. Gets walking directions from user's GPS location to nearest station
2. Gets walking directions from last station to final destination (landmarks)
3. Integrates walking segments into route response

Uses OSRM for realistic walking directions on OpenStreetMap data.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Walking constants
WALKING_SPEED_M_PER_MIN = 80  # Average walking speed: 80m/minute (~5 km/h)
MAX_WALKING_DISTANCE_M = 2000  # Maximum walking distance to show (2km)


@dataclass
class WalkingSegment:
    """A walking segment with directions"""
    type: str  # 'first_mile' or 'last_mile'
    from_name: str
    to_name: str
    from_coords: Tuple[float, float]  # (lat, lon)
    to_coords: Tuple[float, float]  # (lat, lon)
    distance_m: float
    duration_min: float
    polyline: Optional[List[Tuple[float, float]]] = None  # Detailed path
    instructions: Optional[List[Dict[str, Any]]] = None  # Turn-by-turn
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'mode': 'walking',
            'from': self.from_name,
            'to': self.to_name,
            'from_coords': {'lat': self.from_coords[0], 'lon': self.from_coords[1]},
            'to_coords': {'lat': self.to_coords[0], 'lon': self.to_coords[1]},
            'distance_m': round(self.distance_m, 0),
            'distance_km': round(self.distance_m / 1000, 2),
            'duration_min': round(self.duration_min, 1),
            'polyline': self.polyline,
            'instructions': self.instructions,
            'emoji': '🚶'
        }


class WalkingDirectionsService:
    """
    Service to add walking directions to transit routes.
    
    Calculates and integrates:
    - First-mile: GPS location → nearest transit station
    - Last-mile: last transit station → final destination (landmark)
    """
    
    def __init__(self):
        """Initialize the walking directions service."""
        self.osrm_service = None
        self._init_osrm()
    
    def _init_osrm(self):
        """Initialize OSRM routing service."""
        try:
            from services.osrm_routing_service import OSRMRoutingService
            self.osrm_service = OSRMRoutingService(profile='foot')
            logger.info("✅ Walking directions service initialized with OSRM")
        except Exception as e:
            logger.warning(f"⚠️ OSRM not available, using simple calculations: {e}")
            self.osrm_service = None
    
    def _haversine_distance(
        self, 
        lat1: float, lon1: float, 
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in meters using Haversine formula."""
        import math
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def get_walking_segment(
        self,
        from_coords: Tuple[float, float],
        to_coords: Tuple[float, float],
        from_name: str,
        to_name: str,
        segment_type: str = 'first_mile'
    ) -> Optional[WalkingSegment]:
        """
        Get walking directions between two points.
        
        Args:
            from_coords: Starting point (lat, lon)
            to_coords: Ending point (lat, lon)
            from_name: Name of starting point
            to_name: Name of destination
            segment_type: 'first_mile' or 'last_mile'
            
        Returns:
            WalkingSegment with directions or None if too far
        """
        # Calculate straight-line distance
        distance_m = self._haversine_distance(
            from_coords[0], from_coords[1],
            to_coords[0], to_coords[1]
        )
        
        # Skip if too far to walk
        if distance_m > MAX_WALKING_DISTANCE_M:
            logger.info(f"🚶 Walking distance too far: {distance_m:.0f}m > {MAX_WALKING_DISTANCE_M}m")
            return None
        
        # Try to get detailed OSRM route
        polyline = None
        instructions = None
        actual_distance = distance_m
        
        if self.osrm_service:
            try:
                route = self.osrm_service.get_route(
                    start=from_coords,
                    end=to_coords,
                    steps=True
                )
                if route:
                    actual_distance = route.total_distance
                    polyline = route.waypoints
                    instructions = [
                        {
                            'instruction': step.instruction,
                            'distance_m': step.distance,
                            'duration_s': step.duration
                        }
                        for step in route.steps
                    ]
                    logger.debug(f"🚶 Got OSRM walking route: {actual_distance:.0f}m")
            except Exception as e:
                logger.warning(f"OSRM walking route failed: {e}")
        
        # Calculate walking time (actual distance or estimate)
        # OSRM returns seconds, we need minutes
        if self.osrm_service and polyline:
            duration_min = actual_distance / WALKING_SPEED_M_PER_MIN
        else:
            # Simple estimate: straight-line * 1.3 for urban walking
            actual_distance = distance_m * 1.3
            duration_min = actual_distance / WALKING_SPEED_M_PER_MIN
        
        # Create simple polyline if OSRM didn't provide one
        if not polyline:
            polyline = [from_coords, to_coords]
        
        return WalkingSegment(
            type=segment_type,
            from_name=from_name,
            to_name=to_name,
            from_coords=from_coords,
            to_coords=to_coords,
            distance_m=actual_distance,
            duration_min=duration_min,
            polyline=polyline,
            instructions=instructions
        )
    
    def add_walking_to_route(
        self,
        route_data: Dict[str, Any],
        user_gps: Optional[Dict[str, float]] = None,
        final_destination_coords: Optional[Tuple[float, float]] = None,
        final_destination_name: Optional[str] = None,
        first_station_coords: Optional[Tuple[float, float]] = None,
        first_station_name: Optional[str] = None,
        last_station_coords: Optional[Tuple[float, float]] = None,
        last_station_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add walking segments to a transit route.
        
        Args:
            route_data: The transit route data
            user_gps: User's GPS location {'lat': float, 'lon': float}
            final_destination_coords: GPS of final destination (landmark)
            final_destination_name: Name of final destination
            first_station_coords: GPS of first transit station
            first_station_name: Name of first transit station
            last_station_coords: GPS of last transit station
            last_station_name: Name of last transit station
            
        Returns:
            Enhanced route_data with walking segments
        """
        walking_segments = []
        additional_time = 0
        additional_distance = 0
        
        # First-mile: User GPS → First Station
        if user_gps and first_station_coords and first_station_name:
            user_coords = (user_gps['lat'], user_gps['lon'])
            
            first_mile = self.get_walking_segment(
                from_coords=user_coords,
                to_coords=first_station_coords,
                from_name="Your Location",
                to_name=first_station_name,
                segment_type='first_mile'
            )
            
            if first_mile:
                walking_segments.append(first_mile.to_dict())
                additional_time += first_mile.duration_min
                additional_distance += first_mile.distance_m
                logger.info(f"🚶 First-mile: {first_mile.distance_m:.0f}m, {first_mile.duration_min:.1f}min to {first_station_name}")
        
        # Last-mile: Last Station → Final Destination
        if final_destination_coords and last_station_coords and last_station_name and final_destination_name:
            last_mile = self.get_walking_segment(
                from_coords=last_station_coords,
                to_coords=final_destination_coords,
                from_name=last_station_name,
                to_name=final_destination_name,
                segment_type='last_mile'
            )
            
            if last_mile:
                walking_segments.append(last_mile.to_dict())
                additional_time += last_mile.duration_min
                additional_distance += last_mile.distance_m
                logger.info(f"🚶 Last-mile: {last_mile.distance_m:.0f}m, {last_mile.duration_min:.1f}min to {final_destination_name}")
        
        # Enhance route_data with walking segments
        if walking_segments:
            route_data['walking_segments'] = walking_segments
            route_data['has_walking'] = True
            
            # Update total time and distance
            if 'total_time' in route_data:
                route_data['total_time_with_walking'] = route_data['total_time'] + additional_time
            if 'total_distance' in route_data:
                route_data['total_distance_with_walking'] = route_data['total_distance'] + additional_distance / 1000
            
            route_data['walking_time_min'] = round(additional_time, 1)
            route_data['walking_distance_m'] = round(additional_distance, 0)
            
            logger.info(f"✅ Added {len(walking_segments)} walking segment(s): +{additional_time:.1f}min, +{additional_distance:.0f}m")
        
        return route_data


# Singleton instance
_walking_service: Optional[WalkingDirectionsService] = None


def get_walking_directions_service() -> WalkingDirectionsService:
    """Get the singleton walking directions service instance."""
    global _walking_service
    if _walking_service is None:
        _walking_service = WalkingDirectionsService()
    return _walking_service
