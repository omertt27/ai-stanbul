"""
OSRM Routing Service for Istanbul AI
Provides realistic walking and transit routes using OpenStreetMap data
Uses the free public OSRM API (demo server)

FREE & OPEN-SOURCE:
- Uses OSRM (Open Source Routing Machine)
- Public OSRM demo server for development
- Can be self-hosted for production
- No API keys required for development

WEEK 2 ENHANCEMENTS:
- Redis caching for route results
- Polyline encoding/decoding
- Multi-modal transport support
- Enhanced turn-by-turn directions
- API usage monitoring
"""

import logging
import requests
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
import time
import hashlib
import json
try:
    import polyline as polyline_lib
except ImportError:
    polyline_lib = None
    
from backend.services.redis_cache import RedisCache

logger = logging.getLogger(__name__)


@dataclass
class RouteStep:
    """Individual step in a route"""
    distance: float  # meters
    duration: float  # seconds
    instruction: str
    location: Tuple[float, float]  # (lat, lon)
    maneuver_type: str  # 'turn', 'depart', 'arrive', etc.
    bearing_before: Optional[float] = None
    bearing_after: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class OSRMRoute:
    """Complete route from OSRM"""
    waypoints: List[Tuple[float, float]]  # List of (lat, lon) coordinates
    steps: List[RouteStep]
    total_distance: float  # meters
    total_duration: float  # seconds
    geometry: str  # Polyline-encoded geometry
    mode: str  # 'foot', 'car', 'bike'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'waypoints': self.waypoints,
            'steps': [step.to_dict() for step in self.steps],
            'total_distance': self.total_distance,
            'total_duration': self.total_duration,
            'geometry': self.geometry,
            'mode': self.mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OSRMRoute':
        """Create from dictionary"""
        steps = [RouteStep(**step) for step in data.get('steps', [])]
        return cls(
            waypoints=data['waypoints'],
            steps=steps,
            total_distance=data['total_distance'],
            total_duration=data['total_duration'],
            geometry=data['geometry'],
            mode=data['mode']
        )


class OSRMRoutingService:
    """
    OSRM Routing Service with Week 2 Enhancements
    - Generates realistic walking routes using OpenStreetMap data
    - Uses free public OSRM demo server (can be self-hosted)
    - No API keys required
    - Supports multiple routing profiles (foot, car, bike)
    - Redis caching for performance
    - Polyline encoding for efficient geometry storage
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
    
    # Transport modes mapping (for Istanbul)
    TRANSPORT_MODES = {
        'walking': 'foot',
        'driving': 'car',
        'cycling': 'bike',
        'taxi': 'car',
        'uber': 'car'
    }
    
    def __init__(
        self,
        server: str = 'primary',
        profile: str = 'foot',
        timeout: int = 10,
        use_fallback: bool = True,
        use_cache: bool = True,
        cache_ttl: int = 3600  # 1 hour default
    ):
        """
        Initialize OSRM Routing Service
        
        Args:
            server: Which OSRM server to use ('primary' or 'fallback')
            profile: Routing profile ('foot', 'car', 'bike')
            timeout: Request timeout in seconds
            use_fallback: Whether to use fallback server on failure
            use_cache: Whether to use Redis caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.base_url = self.OSRM_SERVERS.get(server, self.OSRM_SERVERS['primary'])
        self.profile = self.PROFILES.get(profile, 'foot')
        self.timeout = timeout
        self.use_fallback = use_fallback
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        
        # Initialize Redis cache
        self.cache = None
        if use_cache:
            try:
                self.cache = RedisCache(db=3)  # Use DB 3 for routing cache
                if self.cache.ping():
                    logger.info("✅ OSRM routing cache enabled (Redis)")
                else:
                    self.cache = None
                    logger.warning("⚠️ Redis not available, caching disabled")
            except Exception as e:
                self.cache = None
                logger.warning(f"⚠️ Failed to initialize routing cache: {e}")
        
        # API usage tracking
        self.api_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"OSRM Routing Service initialized (server: {server}, profile: {profile}, cache: {self.cache is not None})")
    
    def _get_cache_key(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]],
        profile: str
    ) -> str:
        """Generate cache key for route request"""
        # Create deterministic key from coordinates
        coords_str = f"{start[0]:.6f},{start[1]:.6f}_{end[0]:.6f},{end[1]:.6f}"
        if waypoints:
            wp_str = "_".join([f"{wp[0]:.6f},{wp[1]:.6f}" for wp in waypoints])
            coords_str += f"_{wp_str}"
        
        # Add profile to key
        key_data = f"{profile}:{coords_str}"
        
        # Hash to keep key reasonably sized
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"osrm_route:{key_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[OSRMRoute]:
        """Get route from cache"""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.client.get(cache_key)
            if cached_data:
                self.cache_hits += 1
                logger.debug(f"✅ Cache hit: {cache_key}")
                route_dict = json.loads(cached_data)
                return OSRMRoute.from_dict(route_dict)
            
            self.cache_misses += 1
            logger.debug(f"❌ Cache miss: {cache_key}")
            return None
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, route: OSRMRoute):
        """Save route to cache"""
        if not self.cache:
            return
        
        try:
            route_json = json.dumps(route.to_dict())
            self.cache.client.setex(cache_key, self.cache_ttl, route_json)
            logger.debug(f"✅ Cached route: {cache_key}")
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
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
        Get route from OSRM with caching support
        
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
        # Check cache first
        if self.use_cache and steps:  # Only cache complete routes with steps
            cache_key = self._get_cache_key(start, end, waypoints, self.profile)
            cached_route = self._get_from_cache(cache_key)
            if cached_route:
                return cached_route
        
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
                'geometries': 'polyline'  # Use polyline encoding for efficiency
            }
            
            # Make request
            self.api_calls += 1
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
            route = self._parse_route(route_data)
            
            # Cache the route
            if self.use_cache and route and steps:
                self._save_to_cache(cache_key, route)
            
            return route
            
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
        Parse OSRM route data with enhanced step parsing
        
        Args:
            route_data: Raw route data from OSRM API
            
        Returns:
            OSRMRoute object
        """
        # Extract geometry (polyline encoded or GeoJSON)
        geometry_str = route_data.get('geometry', '')
        
        # Decode polyline if library is available
        waypoints = []
        if polyline_lib and isinstance(geometry_str, str):
            try:
                # Decode polyline to list of (lat, lon) tuples
                waypoints = polyline_lib.decode(geometry_str)
            except Exception as e:
                logger.warning(f"Failed to decode polyline: {e}")
                waypoints = []
        
        # Fallback: extract from legs if polyline decode failed
        if not waypoints:
            for leg in route_data.get('legs', []):
                for step_data in leg.get('steps', []):
                    maneuver = step_data.get('maneuver', {})
                    location = maneuver.get('location', [0, 0])
                    waypoints.append((location[1], location[0]))
        
        # Extract enhanced steps with bearings
        steps = []
        for leg in route_data.get('legs', []):
            for step_data in leg.get('steps', []):
                maneuver = step_data.get('maneuver', {})
                location = maneuver.get('location', [0, 0])
                
                # Enhanced instruction formatting
                name = step_data.get('name', '')
                mode = step_data.get('mode', 'walking')
                maneuver_type = maneuver.get('type', 'turn')
                modifier = maneuver.get('modifier', '')
                
                # Build human-readable instruction
                instruction = self._build_instruction(
                    maneuver_type, modifier, name, mode
                )
                
                step = RouteStep(
                    distance=step_data.get('distance', 0),
                    duration=step_data.get('duration', 0),
                    instruction=instruction,
                    location=(location[1], location[0]),  # Convert to (lat, lon)
                    maneuver_type=maneuver_type,
                    bearing_before=maneuver.get('bearing_before'),
                    bearing_after=maneuver.get('bearing_after')
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
            geometry=geometry_str,  # Store encoded polyline
            mode=self.profile
        )
    
    def _build_instruction(
        self,
        maneuver_type: str,
        modifier: str,
        street_name: str,
        mode: str
    ) -> str:
        """
        Build simplified turn-by-turn instruction
        Since we show visual directions on map, text can be concise
        
        Args:
            maneuver_type: Type of maneuver (turn, merge, etc.)
            modifier: Direction modifier (left, right, sharp, slight)
            street_name: Name of street
            mode: Transport mode
            
        Returns:
            Simple, concise instruction
        """
        # Simplified templates for map-based navigation
        if maneuver_type == 'depart':
            return f"Start" + (f" on {street_name}" if street_name else "")
        
        elif maneuver_type == 'arrive':
            return "Arrive at destination"
        
        elif maneuver_type == 'turn':
            direction = modifier.capitalize() if modifier else ""
            if street_name:
                return f"{direction} on {street_name}" if direction else f"Continue on {street_name}"
            return f"Turn {modifier}" if modifier else "Turn"
        
        elif maneuver_type in ['merge', 'fork']:
            return f"Keep {modifier}" if modifier else "Continue"
        
        elif maneuver_type == 'roundabout':
            return "Take roundabout"
        
        elif maneuver_type == 'continue':
            return f"Continue" + (f" on {street_name}" if street_name else "")
        
        elif maneuver_type == 'new name':
            return f"Continue on {street_name}" if street_name else "Continue"
        
        else:
            # Default: simple continue
            return "Continue"
    
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
    
    def get_driving_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]] = None
    ) -> Optional[OSRMRoute]:
        """
        Get driving route (convenience method)
        
        Args:
            start: Start location (lat, lon)
            end: End location (lat, lon)
            waypoints: Optional intermediate waypoints
            
        Returns:
            OSRMRoute object or None if request fails
        """
        original_profile = self.profile
        self.profile = 'car'
        
        try:
            return self.get_route(start, end, waypoints)
        finally:
            self.profile = original_profile
    
    def get_route_by_mode(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        transport_mode: str,
        waypoints: Optional[List[Tuple[float, float]]] = None
    ) -> Optional[OSRMRoute]:
        """
        Get route by transport mode (multi-modal support)
        
        Args:
            start: Start location (lat, lon)
            end: End location (lat, lon)
            transport_mode: Transport mode ('walking', 'driving', 'cycling', 'taxi', 'uber')
            waypoints: Optional intermediate waypoints
            
        Returns:
            OSRMRoute object or None if request fails
        """
        # Map transport mode to OSRM profile
        profile = self.TRANSPORT_MODES.get(transport_mode.lower(), 'foot')
        
        original_profile = self.profile
        self.profile = profile
        
        try:
            route = self.get_route(start, end, waypoints)
            if route:
                # Update mode to reflect actual transport mode requested
                route.mode = transport_mode
            return route
        finally:
            self.profile = original_profile
    
    def get_distance_matrix(
        self,
        locations: List[Tuple[float, float]],
        profile: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get distance and duration matrix for multiple locations
        Uses OSRM Table API for efficient matrix calculation
        
        Args:
            locations: List of locations (lat, lon)
            profile: Optional routing profile override
            
        Returns:
            Dictionary with distance and duration matrices, or None on error
        """
        if len(locations) < 2:
            logger.warning("Need at least 2 locations for distance matrix")
            return None
        
        # Check cache
        cache_key = None
        if self.use_cache:
            coords_str = "_".join([f"{loc[0]:.6f},{loc[1]:.6f}" for loc in locations])
            profile_str = profile or self.profile
            key_data = f"matrix:{profile_str}:{coords_str}"
            cache_key = f"osrm_matrix:{hashlib.md5(key_data.encode()).hexdigest()}"
            
            try:
                cached_data = self.cache.client.get(cache_key)
                if cached_data:
                    self.cache_hits += 1
                    logger.debug(f"✅ Matrix cache hit")
                    return json.loads(cached_data)
                self.cache_misses += 1
            except Exception:
                pass
        
        # Use provided profile or current
        current_profile = profile or self.profile
        
        # Build coordinates string (lon,lat format for OSRM)
        coords = ";".join([f"{loc[1]},{loc[0]}" for loc in locations])
        
        try:
            # OSRM Table API
            url = f"{self.base_url}/table/v1/{current_profile}/{coords}"
            params = {'annotations': 'distance,duration'}
            
            self.api_calls += 1
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') != 'Ok':
                logger.error(f"OSRM table request error: {data.get('code')}")
                return None
            
            result = {
                'distances': data['distances'],  # meters
                'durations': data['durations']   # seconds
            }
            
            # Cache the result
            if self.use_cache and cache_key:
                try:
                    self.cache.client.setex(cache_key, self.cache_ttl, json.dumps(result))
                except Exception as e:
                    logger.error(f"Failed to cache matrix: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting distance matrix: {e}")
            return None
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get API usage and cache statistics
        
        Returns:
            Dictionary with usage stats
        """
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'api_calls': self.api_calls,
            'cache_enabled': self.cache is not None,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'total_cached_requests': total_requests,
            'server': self.base_url,
            'profile': self.profile
        }
    
    def clear_cache(self, pattern: str = "osrm_*"):
        """
        Clear routing cache
        
        Args:
            pattern: Redis key pattern to delete (default: all OSRM cache)
        """
        if not self.cache:
            logger.warning("Cache not enabled")
            return 0
        
        try:
            keys = self.cache.client.keys(pattern)
            if keys:
                deleted = self.cache.client.delete(*keys)
                logger.info(f"✅ Cleared {deleted} routing cache entries")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0
    
    def _generate_cache_key(self, url: str, params: Dict) -> str:
        """
        Generate a unique cache key for the given URL and parameters
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            Cache key string
        """
        # Create a sorted tuple of parameters for consistent hashing
        param_tuple = tuple(sorted(params.items()))
        # Combine URL and parameters into a single string
        cache_key = f"{url}?{param_tuple}"
        # Hash the string to create a unique key
        return hashlib.md5(cache_key.encode()).hexdigest()
    
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
    
    def _cache_url_params(
        self,
        url: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key from URL and parameters"""
        # Sort params for consistent cache keys
        param_tuple = tuple(sorted(params.items()))
        # Combine URL and parameters into a single string
        cache_key = f"{url}?{param_tuple}"
        # Hash the string to create a unique key
        return hashlib.md5(cache_key.encode()).hexdigest()


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
