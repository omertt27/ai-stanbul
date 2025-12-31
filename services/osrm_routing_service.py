"""
OSRM Routing Service - Turn-by-turn navigation
Integrated with AI Istanbul Chat System

Provides Google Maps-like turn-by-turn navigation using OpenStreetMap data
"""

import aiohttp
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class RouteStep:
    """Individual step in a route with turn-by-turn instructions"""
    instruction: str
    distance_m: float
    duration_s: float
    geometry: List[Tuple[float, float]] = field(default_factory=list)
    street_name: str = ""
    maneuver_type: str = ""  # turn, arrive, depart, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'instruction': self.instruction,
            'distance_m': round(self.distance_m, 1),
            'duration_s': round(self.duration_s, 1),
            'street_name': self.street_name,
            'maneuver_type': self.maneuver_type,
            'geometry': self.geometry
        }


@dataclass
class OSRMRoute:
    """Complete route with turn-by-turn directions"""
    total_distance_m: float
    total_duration_s: float
    steps: List[RouteStep]
    geometry: List[Tuple[float, float]]
    alternatives: Optional[List['OSRMRoute']] = None
    
    @property
    def total_distance_km(self) -> float:
        """Distance in kilometers"""
        return self.total_distance_m / 1000
    
    @property
    def total_duration_min(self) -> float:
        """Duration in minutes"""
        return self.total_duration_s / 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'total_distance_m': round(self.total_distance_m, 1),
            'total_distance_km': round(self.total_distance_km, 2),
            'total_duration_s': round(self.total_duration_s, 1),
            'total_duration_min': round(self.total_duration_min, 1),
            'steps': [step.to_dict() for step in self.steps],
            'geometry': self.geometry,
            'alternatives_count': len(self.alternatives) if self.alternatives else 0
        }


class OSRMRoutingService:
    """
    Open Source Routing Machine integration
    
    Provides turn-by-turn navigation for:
    - Walking routes (default)
    - Driving routes
    - Cycling routes
    
    Features:
    - Multi-stop waypoint routing
    - Alternative route suggestions
    - Detailed turn-by-turn instructions
    - GeoJSON geometry for map display
    """
    
    def __init__(self, host: str = "http://router.project-osrm.org"):
        """
        Initialize OSRM routing service
        
        Args:
            host: OSRM server URL (default: public demo server)
                  For production, use self-hosted: http://localhost:5000
        """
        self.host = host
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"✅ OSRM Routing Service initialized (host={host})")
    
    async def _ensure_session(self):
        """Ensure HTTP session is initialized"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def get_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        mode: str = "walking",
        alternatives: bool = True,
        steps: bool = True
    ) -> Optional[OSRMRoute]:
        """
        Get route from OSRM API
        
        Args:
            start: Starting coordinates (lat, lon)
            end: Ending coordinates (lat, lon)
            mode: Transport mode - 'walking', 'driving', or 'cycling'
            alternatives: Whether to include alternative routes
            steps: Whether to include turn-by-turn steps
        
        Returns:
            OSRMRoute with turn-by-turn directions, or None if route not found
        
        Example:
            >>> osrm = OSRMRoutingService()
            >>> route = await osrm.get_route(
            ...     start=(41.0082, 28.9784),  # Sultanahmet
            ...     end=(41.0369, 28.9850),    # Taksim
            ...     mode="walking"
            ... )
            >>> print(f"Distance: {route.total_distance_km:.2f} km")
            >>> print(f"Duration: {route.total_duration_min:.0f} minutes")
        """
        await self._ensure_session()
        
        # OSRM expects lon,lat format (opposite of our lat,lon)
        coords = f"{start[1]},{start[0]};{end[1]},{end[0]}"
        url = f"{self.host}/route/v1/{mode}/{coords}"
        
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true" if steps else "false",
            "alternatives": "true" if alternatives else "false"
        }
        
        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    logger.error(f"OSRM API error: HTTP {response.status}")
                    return None
                
                data = await response.json()
                
                if data.get("code") != "Ok":
                    logger.error(f"OSRM error: {data.get('message')}")
                    return None
                
                return self._parse_osrm_response(data, parse_alternatives=alternatives)
        
        except asyncio.TimeoutError:
            logger.error("OSRM request timeout")
            return None
        except Exception as e:
            logger.error(f"OSRM request failed: {e}")
            return None
    
    async def get_multi_stop_route(
        self,
        waypoints: List[Tuple[float, float]],
        mode: str = "walking"
    ) -> Optional[OSRMRoute]:
        """
        Get route through multiple waypoints
        
        Args:
            waypoints: List of (lat, lon) coordinates to visit in order
            mode: Transport mode
        
        Returns:
            Optimized route through all waypoints
        
        Example:
            >>> waypoints = [
            ...     (41.0082, 28.9784),  # Sultanahmet
            ...     (41.0256, 28.9744),  # Galata Tower
            ...     (41.0369, 28.9850)   # Taksim
            ... ]
            >>> route = await osrm.get_multi_stop_route(waypoints, mode="walking")
        """
        await self._ensure_session()
        
        if len(waypoints) < 2:
            logger.error("Need at least 2 waypoints for multi-stop route")
            return None
        
        # Convert waypoints to OSRM format (lon,lat)
        coords = ";".join([f"{lon},{lat}" for lat, lon in waypoints])
        url = f"{self.host}/route/v1/{mode}/{coords}"
        
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true"
        }
        
        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    logger.error(f"OSRM API error: HTTP {response.status}")
                    return None
                
                data = await response.json()
                
                if data.get("code") != "Ok":
                    logger.error(f"OSRM error: {data.get('message')}")
                    return None
                
                return self._parse_osrm_response(data, parse_alternatives=False)
        
        except Exception as e:
            logger.error(f"Multi-stop OSRM request failed: {e}")
            return None
    
    def _parse_osrm_response(self, data: dict, parse_alternatives: bool = False) -> Optional[OSRMRoute]:
        """
        Parse OSRM API response into OSRMRoute
        
        Args:
            data: OSRM API response JSON
            parse_alternatives: Whether to parse alternative routes
        
        Returns:
            Parsed OSRMRoute object
        """
        if not data.get("routes"):
            return None
        
        route = data["routes"][0]
        
        # Parse steps with enhanced instructions
        steps = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                # Extract maneuver instruction
                maneuver = step.get("maneuver", {})
                instruction = maneuver.get("instruction", "Continue")
                maneuver_type = maneuver.get("type", "")
                
                # Get street name
                street_name = step.get("name", "")
                
                # Enhance instruction with street name if available
                if street_name and street_name not in instruction and street_name != "":
                    if "turn" in instruction.lower():
                        instruction = f"{instruction} onto {street_name}"
                    elif maneuver_type == "depart":
                        instruction = f"Start on {street_name}"
                    elif maneuver_type == "arrive":
                        instruction = f"Arrive at {street_name}"
                
                steps.append(RouteStep(
                    instruction=instruction,
                    distance_m=step.get("distance", 0),
                    duration_s=step.get("duration", 0),
                    geometry=self._decode_geometry(step.get("geometry", {})),
                    street_name=street_name,
                    maneuver_type=maneuver_type
                ))
        
        # Parse full route geometry
        geometry = self._decode_geometry(route.get("geometry", {}))
        
        # Parse alternative routes if requested
        alternatives = []
        if parse_alternatives:
            for alt_route in data.get("routes", [])[1:]:  # Skip first (main route)
                alt_steps = []
                for leg in alt_route.get("legs", []):
                    for step in leg.get("steps", []):
                        maneuver = step.get("maneuver", {})
                        alt_steps.append(RouteStep(
                            instruction=maneuver.get("instruction", "Continue"),
                            distance_m=step.get("distance", 0),
                            duration_s=step.get("duration", 0),
                            geometry=self._decode_geometry(step.get("geometry", {})),
                            street_name=step.get("name", ""),
                            maneuver_type=maneuver.get("type", "")
                        ))
                
                alternatives.append(OSRMRoute(
                    total_distance_m=alt_route.get("distance", 0),
                    total_duration_s=alt_route.get("duration", 0),
                    steps=alt_steps,
                    geometry=self._decode_geometry(alt_route.get("geometry", {}))
                ))
        
        return OSRMRoute(
            total_distance_m=route.get("distance", 0),
            total_duration_s=route.get("duration", 0),
            steps=steps,
            geometry=geometry,
            alternatives=alternatives if alternatives else None
        )
    
    def _decode_geometry(self, geometry_data: dict) -> List[Tuple[float, float]]:
        """
        Decode GeoJSON geometry to list of coordinates
        
        Args:
            geometry_data: GeoJSON geometry object
        
        Returns:
            List of (lat, lon) tuples
        """
        if geometry_data.get("type") == "LineString":
            # Convert [lon, lat] to (lat, lon)
            return [(coord[1], coord[0]) for coord in geometry_data.get("coordinates", [])]
        return []
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("OSRM session closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                asyncio.create_task(self.session.close())
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# Global Service Instance
# ═══════════════════════════════════════════════════════════════════

_osrm_service: Optional[OSRMRoutingService] = None


def get_osrm_service(host: str = "http://router.project-osrm.org") -> OSRMRoutingService:
    """
    Get or create global OSRM service instance
    
    Args:
        host: OSRM server URL
    
    Returns:
        Singleton OSRMRoutingService instance
    """
    global _osrm_service
    if _osrm_service is None:
        _osrm_service = OSRMRoutingService(host=host)
    return _osrm_service


# ═══════════════════════════════════════════════════════════════════
# CLI Testing & Demo
# ═══════════════════════════════════════════════════════════════════

async def test_osrm_basic():
    """Test basic OSRM routing"""
    print("\n🗺️  OSRM Routing Service - Basic Test\n")
    print("=" * 60)
    
    service = OSRMRoutingService()
    
    # Route from Sultanahmet to Taksim
    start = (41.0082, 28.9784)  # Sultanahmet
    end = (41.0369, 28.9850)    # Taksim Square
    
    print(f"\n📍 Route: Sultanahmet → Taksim Square")
    print(f"   Start: {start}")
    print(f"   End: {end}")
    print("\n🔄 Fetching route from OSRM...\n")
    
    route = await service.get_route(start, end, mode="walking")
    
    if route:
        print("✅ Route found!")
        print(f"\n📊 Summary:")
        print(f"   Distance: {route.total_distance_km:.2f} km")
        print(f"   Duration: {route.total_duration_min:.0f} minutes")
        print(f"   Steps: {len(route.steps)}")
        print(f"   Alternatives: {len(route.alternatives) if route.alternatives else 0}")
        
        print(f"\n🚶 Turn-by-turn directions:")
        for i, step in enumerate(route.steps[:10], 1):  # Show first 10 steps
            print(f"   {i}. {step.instruction}")
            print(f"      ({step.distance_m:.0f}m, {step.duration_s:.0f}s)")
        
        if len(route.steps) > 10:
            print(f"   ... and {len(route.steps) - 10} more steps")
        
        print(f"\n✅ Test completed successfully!")
    else:
        print("❌ Route not found")
    
    await service.close()


async def test_osrm_multi_stop():
    """Test multi-stop routing"""
    print("\n🗺️  OSRM Routing Service - Multi-Stop Test\n")
    print("=" * 60)
    
    service = OSRMRoutingService()
    
    waypoints = [
        (41.0082, 28.9784),  # Sultanahmet
        (41.0256, 28.9744),  # Galata Tower
        (41.0369, 28.9850)   # Taksim Square
    ]
    
    waypoint_names = ["Sultanahmet", "Galata Tower", "Taksim Square"]
    
    print(f"\n📍 Multi-stop route:")
    for i, (wp, name) in enumerate(zip(waypoints, waypoint_names), 1):
        print(f"   {i}. {name} {wp}")
    
    print("\n🔄 Fetching multi-stop route from OSRM...\n")
    
    route = await service.get_multi_stop_route(waypoints, mode="walking")
    
    if route:
        print("✅ Route found!")
        print(f"\n📊 Summary:")
        print(f"   Total Distance: {route.total_distance_km:.2f} km")
        print(f"   Total Duration: {route.total_duration_min:.0f} minutes")
        print(f"   Total Steps: {len(route.steps)}")
        
        print(f"\n✅ Multi-stop test completed successfully!")
    else:
        print("❌ Route not found")
    
    await service.close()


if __name__ == "__main__":
    import sys
    
    # Run tests
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        asyncio.run(test_osrm_multi_stop())
    else:
        asyncio.run(test_osrm_basic())
    
    print("\n" + "=" * 60)
    print("💡 Usage:")
    print("   python services/osrm_routing_service.py         # Basic test")
    print("   python services/osrm_routing_service.py multi   # Multi-stop test")
    print("=" * 60 + "\n")
