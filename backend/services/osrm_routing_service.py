"""
OSRM Routing Service for Istanbul AI
Provides realistic walking and transit routes using OpenStreetMap data
Uses the free public OSRM API (demo server)

FREE & OPEN-SOURCE:
- Uses OSRM (Open Source Routing Machine)
- Public OSRM demo server for development
- Can be self-hosted for production
- No API keys required for development
"""

import logging
import requests
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class RouteStep:
    """Individual step in a route"""
    distance: float  # meters
    duration: float  # seconds
    instruction: str
    location: Tuple[float, float]  # (lat, lon)
    maneuver_type: str  # 'turn', 'depart', 'arrive', etc.


@dataclass
class OSRMRoute:
    """Complete route from OSRM"""
    waypoints: List[Tuple[float, float]]  # List of (lat, lon) coordinates
    steps: List[RouteStep]
    total_distance: float  # meters
    total_duration: float  # seconds
    geometry: str  # Polyline-encoded geometry
    mode: str  # 'foot', 'car', 'bike'


class OSRMRoutingService:
    """
    OSRM Routing Service
    - Generates realistic walking routes using OpenStreetMap data
    - Uses free public OSRM demo server (can be self-hosted)
    - No API keys required
    - Supports multiple routing profiles (foot, car, bike)
    """
    
    # Public OSRM demo servers (free to use)
    # NOTE: For production, consider self-hosting OSRM
    OSRM_SERVERS = {
        'primary': 'http://router.project-osrm.org',
        'fallback': 'https://routing.openstreetmap.de/routed-foot'
    }
    
    # Routing profiles
    PROFILES = {
        'foot': 'foot',  # Walking
        'car': 'car',    # Driving
        'bike': 'bike'   # Cycling
    }
    
    def __init__(
        self,
        server: str = 'primary',
        profile: str = 'foot',
        timeout: int = 10,
        use_fallback: bool = True
    ):
        """
        Initialize OSRM Routing Service
        
        Args:
            server: Which OSRM server to use ('primary' or 'fallback')
            profile: Routing profile ('foot', 'car', 'bike')
            timeout: Request timeout in seconds
            use_fallback: Whether to use fallback server on failure
        """
        self.base_url = self.OSRM_SERVERS.get(server, self.OSRM_SERVERS['primary'])
        self.profile = self.PROFILES.get(profile, 'foot')
        self.timeout = timeout
        self.use_fallback = use_fallback
        
        logger.info(f"OSRM Routing Service initialized (server: {server}, profile: {profile})")
    
    def get_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]] = None,
        alternatives: int = 0,
        steps: bool = True,
        overview: str = 'full'
    ) -> Optional[OSRMRoute]:
        """
        Get route from OSRM
        
        Args:
            start: Start location (lat, lon)
            end: End location (lat, lon)
            waypoints: Optional intermediate waypoints
            alternatives: Number of alternative routes to request
            steps: Whether to include turn-by-turn instructions
            overview: Geometry overview level ('full', 'simplified', 'false')
            
        Returns:
            OSRMRoute object or None if request fails
        """
        try:
            # OSRM uses lon,lat format (opposite of standard lat,lon)
            coordinates = [
                f"{start[1]},{start[0]}"
            ]
            
            # Add intermediate waypoints if provided
            if waypoints:
                for wp in waypoints:
                    coordinates.append(f"{wp[1]},{wp[0]}")
            
            coordinates.append(f"{end[1]},{end[0]}")
            
            # Build URL
            coords_str = ';'.join(coordinates)
            url = f"{self.base_url}/route/v1/{self.profile}/{coords_str}"
            
            # Build parameters
            params = {
                'overview': overview,
                'steps': 'true' if steps else 'false',
                'alternatives': alternatives,
                'geometries': 'geojson'
            }
            
            # Make request
            logger.debug(f"Requesting route from OSRM: {url}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') != 'Ok':
                logger.error(f"OSRM returned error: {data.get('code')}")
                return None
            
            # Extract first route
            if not data.get('routes'):
                logger.error("No routes returned from OSRM")
                return None
            
            route_data = data['routes'][0]
            
            # Parse route
            return self._parse_route(route_data)
            
        except requests.exceptions.Timeout:
            logger.error("OSRM request timed out")
            if self.use_fallback:
                logger.info("Trying fallback server...")
                return self._try_fallback(start, end, waypoints, alternatives, steps, overview)
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OSRM request failed: {e}")
            if self.use_fallback:
                logger.info("Trying fallback server...")
                return self._try_fallback(start, end, waypoints, alternatives, steps, overview)
            return None
            
        except Exception as e:
            logger.error(f"Error getting route from OSRM: {e}")
            return None
    
    def _try_fallback(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]],
        alternatives: int,
        steps: bool,
        overview: str
    ) -> Optional[OSRMRoute]:
        """Try fallback server"""
        original_url = self.base_url
        self.base_url = self.OSRM_SERVERS['fallback']
        
        try:
            result = self.get_route(start, end, waypoints, alternatives, steps, overview)
            return result
        finally:
            self.base_url = original_url
    
    def _parse_route(self, route_data: Dict) -> OSRMRoute:
        """
        Parse OSRM route data
        
        Args:
            route_data: Raw route data from OSRM API
            
        Returns:
            OSRMRoute object
        """
        # Extract geometry (waypoints)
        geometry = route_data.get('geometry', {})
        coordinates = geometry.get('coordinates', [])
        
        # OSRM returns [lon, lat] - convert to [lat, lon]
        waypoints = [(coord[1], coord[0]) for coord in coordinates]
        
        # Extract steps
        steps = []
        for leg in route_data.get('legs', []):
            for step_data in leg.get('steps', []):
                maneuver = step_data.get('maneuver', {})
                location = maneuver.get('location', [0, 0])
                
                step = RouteStep(
                    distance=step_data.get('distance', 0),
                    duration=step_data.get('duration', 0),
                    instruction=maneuver.get('instruction', step_data.get('name', 'Continue')),
                    location=(location[1], location[0]),  # Convert to (lat, lon)
                    maneuver_type=maneuver.get('type', 'turn')
                )
                steps.append(step)
        
        # Extract totals
        total_distance = route_data.get('distance', 0)  # meters
        total_duration = route_data.get('duration', 0)  # seconds
        
        return OSRMRoute(
            waypoints=waypoints,
            steps=steps,
            total_distance=total_distance,
            total_duration=total_duration,
            geometry=str(coordinates),  # Store original geometry
            mode=self.profile
        )
    
    def get_walking_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]] = None
    ) -> Optional[OSRMRoute]:
        """
        Get walking route (convenience method)
        
        Args:
            start: Start location (lat, lon)
            end: End location (lat, lon)
            waypoints: Optional intermediate waypoints
            
        Returns:
            OSRMRoute object or None if request fails
        """
        # Ensure we're using foot profile
        original_profile = self.profile
        self.profile = 'foot'
        
        try:
            return self.get_route(start, end, waypoints)
        finally:
            self.profile = original_profile
    
    def get_multiple_routes(
        self,
        locations: List[Tuple[float, float]],
        profile: Optional[str] = None
    ) -> List[OSRMRoute]:
        """
        Get routes connecting multiple locations in sequence
        
        Args:
            locations: List of locations to connect (lat, lon)
            profile: Optional routing profile override
            
        Returns:
            List of OSRMRoute objects
        """
        if len(locations) < 2:
            logger.warning("Need at least 2 locations to create routes")
            return []
        
        if profile:
            original_profile = self.profile
            self.profile = profile
        
        routes = []
        try:
            for i in range(len(locations) - 1):
                route = self.get_route(locations[i], locations[i + 1])
                if route:
                    routes.append(route)
                # Be nice to the public OSRM server - rate limit
                time.sleep(0.1)
            
            return routes
            
        finally:
            if profile:
                self.profile = original_profile
    
    def format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted string (e.g., "15 min", "1h 30m")
        """
        if seconds < 60:
            return f"{int(seconds)} sec"
        elif seconds < 3600:
            return f"{int(seconds / 60)} min"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def format_distance(self, meters: float) -> str:
        """
        Format distance in human-readable format
        
        Args:
            meters: Distance in meters
            
        Returns:
            Formatted string (e.g., "500 m", "2.5 km")
        """
        if meters < 1000:
            return f"{int(meters)} m"
        else:
            return f"{meters / 1000:.1f} km"


def test_osrm_service():
    """Test OSRM routing service with Istanbul locations"""
    print("🧪 Testing OSRM Routing Service...\n")
    
    # Initialize service
    service = OSRMRoutingService(profile='foot')
    
    # Test locations in Istanbul
    # Sultanahmet Square to Hagia Sophia
    start = (41.0054, 28.9768)
    end = (41.0086, 28.9802)
    
    print(f"📍 Getting walking route from Sultanahmet to Hagia Sophia...")
    route = service.get_walking_route(start, end)
    
    if route:
        print(f"✅ Route found!")
        print(f"   Distance: {service.format_distance(route.total_distance)}")
        print(f"   Duration: {service.format_duration(route.total_duration)}")
        print(f"   Waypoints: {len(route.waypoints)} points")
        print(f"   Steps: {len(route.steps)}")
        
        print("\n📋 Turn-by-turn instructions:")
        for i, step in enumerate(route.steps[:5], 1):  # Show first 5 steps
            print(f"   {i}. {step.instruction} ({service.format_distance(step.distance)})")
        
        if len(route.steps) > 5:
            print(f"   ... and {len(route.steps) - 5} more steps")
        
        print(f"\n🗺️  First 5 waypoints:")
        for i, wp in enumerate(route.waypoints[:5], 1):
            print(f"   {i}. {wp}")
        
        return True
    else:
        print("❌ Failed to get route")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = test_osrm_service()
    
    if success:
        print("\n✅ OSRM Routing Service is working!")
        print("   Ready to integrate with Map Visualization Engine")
    else:
        print("\n❌ OSRM Routing Service test failed")
        print("   Check network connection and OSRM server availability")
