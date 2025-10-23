"""
Transportation Directions Service
==================================

Provides detailed, Google Maps-style directions for Istanbul public transportation.
Includes metro, tram, bus, ferry routes with step-by-step instructions.

Features:
- Multi-modal transportation (metro, tram, bus, ferry, walking)
- Detailed step-by-step directions
- Line-specific information
- Transfer instructions
- Real-time estimates
- Integration with OSRM for walking segments
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import OSRM for walking segments
try:
    from .osrm_routing_service import OSRMRoutingService
    OSRM_AVAILABLE = True
    logger.info("✅ Backend OSRM routing service imported successfully")
except ImportError as e:
    OSRM_AVAILABLE = False
    logger.warning(f"⚠️ OSRM not available - walking segments will be estimated. Error: {e}")
except Exception as e:
    OSRM_AVAILABLE = False
    logger.error(f"⚠️ Error importing OSRM service: {e}")


@dataclass
class TransportStep:
    """A single step in a transportation route"""
    mode: str  # 'walk', 'metro', 'tram', 'bus', 'ferry'
    instruction: str  # Human-readable instruction
    distance: float  # meters
    duration: int  # minutes
    start_location: Tuple[float, float]  # (lat, lng)
    end_location: Tuple[float, float]  # (lat, lng)
    line_name: Optional[str] = None  # e.g., "M2 Metro Line"
    stops_count: Optional[int] = None  # Number of stops
    waypoints: Optional[List[Tuple[float, float]]] = None  # Route polyline


@dataclass
class TransportRoute:
    """Complete transportation route"""
    steps: List[TransportStep]
    total_distance: float  # meters
    total_duration: int  # minutes
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    summary: str = ""
    modes_used: List[str] = None


class TransportationDirectionsService:
    """Service for generating detailed transportation directions in Istanbul"""
    
    def __init__(self):
        """Initialize the transportation directions service"""
        self.osrm = None
        if OSRM_AVAILABLE:
            try:
                # Initialize with backend OSRM service parameters
                self.osrm = OSRMRoutingService(
                    server='primary',
                    profile='foot',
                    timeout=10,
                    use_fallback=True
                )
                logger.info("✅ OSRM service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OSRM service: {e}")
                self.osrm = None
        
        # Initialize Istanbul metro/tram/bus lines
        self._initialize_transit_lines()
        logger.info("✅ Transportation Directions Service initialized")
    
    def _initialize_transit_lines(self):
        """Initialize Istanbul public transit lines with stations and routes"""
        
        # Metro lines with major stations
        self.metro_lines = {
            'M1': {
                'name': 'M1 Yenikapı - Atatürk Airport/Kirazlı',
                'color': 'red',
                'stations': [
                    {'name': 'Yenikapı', 'lat': 41.0035, 'lng': 28.9510},
                    {'name': 'Aksaray', 'lat': 41.0166, 'lng': 28.9548},
                    {'name': 'Emniyet-Fatih', 'lat': 41.0195, 'lng': 28.9419},
                    {'name': 'Topkapı-Ulubatlı', 'lat': 41.0115, 'lng': 28.9145},
                ]
            },
            'M2': {
                'name': 'M2 Yenikapı - Hacıosman',
                'color': 'green',
                'stations': [
                    {'name': 'Yenikapı', 'lat': 41.0035, 'lng': 28.9510},
                    {'name': 'Vezneciler', 'lat': 41.0130, 'lng': 28.9545},
                    {'name': 'Şişhane', 'lat': 41.0268, 'lng': 28.9737},
                    {'name': 'Taksim', 'lat': 41.0370, 'lng': 28.9850},
                    {'name': 'Osmanbey', 'lat': 41.0478, 'lng': 28.9885},
                    {'name': 'Levent', 'lat': 41.0788, 'lng': 29.0103},
                ]
            },
            'M3': {
                'name': 'M3 Kirazlı - Başakşehir/Olimpiyat',
                'color': 'blue',
                'stations': [
                    {'name': 'Kirazlı', 'lat': 41.0285, 'lng': 28.8264},
                    {'name': 'Başakşehir', 'lat': 41.0800, 'lng': 28.8100},
                ]
            },
            'M4': {
                'name': 'M4 Kadıköy - Tavşantepe',
                'color': 'pink',
                'stations': [
                    {'name': 'Kadıköy', 'lat': 40.9900, 'lng': 29.0250},
                    {'name': 'Ayrılık Çeşmesi', 'lat': 40.9850, 'lng': 29.0350},
                    {'name': 'Kartal', 'lat': 40.8956, 'lng': 29.1850},
                ]
            },
            'M5': {
                'name': 'M5 Üsküdar - Çekmeköy',
                'color': 'purple',
                'stations': [
                    {'name': 'Üsküdar', 'lat': 41.0226, 'lng': 29.0150},
                    {'name': 'Ümraniye', 'lat': 41.0200, 'lng': 29.1100},
                ]
            },
        }
        
        # Tram lines
        self.tram_lines = {
            'T1': {
                'name': 'T1 Kabataş - Bağcılar',
                'color': 'blue',
                'stations': [
                    {'name': 'Kabataş', 'lat': 41.0311, 'lng': 29.0097},
                    {'name': 'Karaköy', 'lat': 41.0242, 'lng': 28.9742},
                    {'name': 'Eminönü', 'lat': 41.0177, 'lng': 28.9742},
                    {'name': 'Sultanahmet', 'lat': 41.0059, 'lng': 28.9769},
                    {'name': 'Beyazıt-Kapalıçarşı', 'lat': 41.0106, 'lng': 28.9680},
                    {'name': 'Aksaray', 'lat': 41.0166, 'lng': 28.9548},
                ]
            },
        }
        
        # Major ferry routes
        self.ferry_routes = {
            'eminonu_kadikoy': {
                'name': 'Eminönü - Kadıköy Ferry',
                'color': 'cyan',
                'stops': [
                    {'name': 'Eminönü', 'lat': 41.0177, 'lng': 28.9742},
                    {'name': 'Kadıköy', 'lat': 40.9900, 'lng': 29.0250},
                ],
                'duration': 20,  # minutes
            },
            'kabatas_uskudar': {
                'name': 'Kabataş - Üsküdar Ferry',
                'color': 'cyan',
                'stops': [
                    {'name': 'Kabataş', 'lat': 41.0311, 'lng': 29.0097},
                    {'name': 'Üsküdar', 'lat': 41.0226, 'lng': 29.0150},
                ],
                'duration': 15,  # minutes
            },
            'besiktas_kadikoy': {
                'name': 'Beşiktaş - Kadıköy Ferry',
                'color': 'cyan',
                'stops': [
                    {'name': 'Beşiktaş', 'lat': 41.0426, 'lng': 29.0050},
                    {'name': 'Kadıköy', 'lat': 40.9900, 'lng': 29.0250},
                ],
                'duration': 25,  # minutes
            },
        }
        
        # Bus routes (sample major routes)
        self.bus_routes = {
            '500T': {
                'name': '500T Taksim - Sarıyer',
                'stops': [
                    {'name': 'Taksim', 'lat': 41.0370, 'lng': 28.9850},
                    {'name': 'Beşiktaş', 'lat': 41.0426, 'lng': 29.0050},
                    {'name': 'Ortaköy', 'lat': 41.0553, 'lng': 29.0275},
                ]
            },
        }
    
    def get_directions(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str = "Start",
        end_name: str = "Destination",
        preferred_modes: Optional[List[str]] = None
    ) -> Optional[TransportRoute]:
        """
        Get detailed transportation directions
        
        Args:
            start: Start coordinates (lat, lng)
            end: End coordinates (lat, lng)
            start_name: Name of start location
            end_name: Name of end location
            preferred_modes: Preferred transportation modes
            
        Returns:
            TransportRoute with detailed steps
        """
        logger.info(f"🚇 Getting directions from {start_name} to {end_name}")
        
        # Check if walking is feasible (< 2km)
        distance = self._calculate_distance(start, end)
        
        if distance < 2.0:  # Less than 2km, suggest walking
            return self._create_walking_route(start, end, start_name, end_name, distance)
        
        # Try to find transit route
        transit_route = self._find_transit_route(start, end, start_name, end_name, preferred_modes)
        
        if transit_route:
            return transit_route
        
        # Fallback to combined walking + transit
        return self._create_mixed_mode_route(start, end, start_name, end_name)
    
    def _create_walking_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str,
        distance: float
    ) -> TransportRoute:
        """Create a walking-only route"""
        
        # Get OSRM walking route if available
        waypoints = [start, end]
        if self.osrm:
            try:
                osrm_route = self.osrm.get_route(start, end)
                if osrm_route and osrm_route.waypoints:
                    waypoints = osrm_route.waypoints
                    distance = osrm_route.total_distance / 1000.0  # Convert to km
            except Exception as e:
                logger.warning(f"OSRM fallback failed: {e}")
        
        # Estimate duration (average walking speed 5 km/h)
        duration = int((distance / 5.0) * 60)  # minutes
        
        steps = [
            TransportStep(
                mode='walk',
                instruction=f"Walk from {start_name} to {end_name}",
                distance=distance * 1000,  # meters
                duration=duration,
                start_location=start,
                end_location=end,
                waypoints=waypoints
            )
        ]
        
        return TransportRoute(
            steps=steps,
            total_distance=distance * 1000,
            total_duration=duration,
            summary=f"Walk to {end_name} ({distance:.1f} km, {duration} min)",
            modes_used=['walk']
        )
    
    def _find_transit_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str,
        preferred_modes: Optional[List[str]] = None
    ) -> Optional[TransportRoute]:
        """Find transit route using metro/tram/ferry"""
        
        # Find nearest stations to start and end
        start_station = self._find_nearest_station(start)
        end_station = self._find_nearest_station(end)
        
        if not start_station or not end_station:
            return None
        
        # Check if they're on the same line
        route = self._find_direct_line(start_station, end_station)
        if route:
            # Add walking segments to/from stations
            return self._build_complete_route(
                start, end, start_name, end_name,
                start_station, end_station, route
            )
        
        # Check for transfer routes
        transfer_route = self._find_transfer_route(start_station, end_station)
        if transfer_route:
            return self._build_complete_route(
                start, end, start_name, end_name,
                start_station, end_station, transfer_route
            )
        
        return None
    
    def _create_mixed_mode_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str
    ) -> TransportRoute:
        """Create a route combining walking and basic transit info"""
        
        distance = self._calculate_distance(start, end)
        
        # Provide general guidance
        steps = []
        
        # Walking to nearest station
        nearest_station = self._find_nearest_station(start)
        if nearest_station:
            walk_dist = self._calculate_distance(start, (nearest_station['lat'], nearest_station['lng']))
            walk_time = int((walk_dist / 5.0) * 60)
            
            steps.append(
                TransportStep(
                    mode='walk',
                    instruction=f"Walk to {nearest_station['name']} station",
                    distance=walk_dist * 1000,
                    duration=walk_time,
                    start_location=start,
                    end_location=(nearest_station['lat'], nearest_station['lng'])
                )
            )
            
            # Transit suggestion
            steps.append(
                TransportStep(
                    mode='metro',
                    instruction=f"Take {nearest_station.get('line', 'metro')} towards your destination",
                    distance=distance * 1000 * 0.7,  # Estimate
                    duration=int(distance * 4),  # Rough estimate
                    start_location=(nearest_station['lat'], nearest_station['lng']),
                    end_location=end,
                    line_name=nearest_station.get('line', 'Metro')
                )
            )
        
        total_duration = sum(step.duration for step in steps)
        
        return TransportRoute(
            steps=steps,
            total_distance=distance * 1000,
            total_duration=total_duration,
            summary=f"Combined route: {total_duration} min",
            modes_used=['walk', 'metro']
        )
    
    def _find_nearest_station(self, location: Tuple[float, float]) -> Optional[Dict]:
        """Find nearest metro/tram station to a location"""
        nearest = None
        min_distance = float('inf')
        
        # Check metro stations
        for line_id, line_data in self.metro_lines.items():
            for station in line_data['stations']:
                dist = self._calculate_distance(
                    location,
                    (station['lat'], station['lng'])
                )
                if dist < min_distance:
                    min_distance = dist
                    nearest = {**station, 'line': f"{line_id} {line_data['name']}", 'type': 'metro'}
        
        # Check tram stations
        for line_id, line_data in self.tram_lines.items():
            for station in line_data['stations']:
                dist = self._calculate_distance(
                    location,
                    (station['lat'], station['lng'])
                )
                if dist < min_distance:
                    min_distance = dist
                    nearest = {**station, 'line': f"{line_id} {line_data['name']}", 'type': 'tram'}
        
        return nearest if min_distance < 2.0 else None  # Max 2km to station
    
    def _find_direct_line(self, start_station: Dict, end_station: Dict) -> Optional[Dict]:
        """Check if two stations are on the same line"""
        # Implementation would check if both stations exist on same line
        # For now, return None to use transfer logic
        return None
    
    def _find_transfer_route(self, start_station: Dict, end_station: Dict) -> Optional[Dict]:
        """Find route with transfers between lines"""
        # Simplified implementation
        # Real implementation would use graph search
        return None
    
    def _build_complete_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str,
        start_station: Dict,
        end_station: Dict,
        transit_info: Dict
    ) -> TransportRoute:
        """Build complete route with walking + transit segments"""
        steps = []
        
        # Walking to start station
        walk_to_station_dist = self._calculate_distance(start, (start_station['lat'], start_station['lng']))
        steps.append(
            TransportStep(
                mode='walk',
                instruction=f"Walk to {start_station['name']} station",
                distance=walk_to_station_dist * 1000,
                duration=int((walk_to_station_dist / 5.0) * 60),
                start_location=start,
                end_location=(start_station['lat'], start_station['lng'])
            )
        )
        
        # Transit segment
        transit_dist = self._calculate_distance(
            (start_station['lat'], start_station['lng']),
            (end_station['lat'], end_station['lng'])
        )
        steps.append(
            TransportStep(
                mode=start_station['type'],
                instruction=f"Take {start_station['line']} to {end_station['name']}",
                distance=transit_dist * 1000,
                duration=int(transit_dist * 3),  # Rough estimate
                start_location=(start_station['lat'], start_station['lng']),
                end_location=(end_station['lat'], end_station['lng']),
                line_name=start_station['line']
            )
        )
        
        # Walking from end station
        walk_from_station_dist = self._calculate_distance((end_station['lat'], end_station['lng']), end)
        steps.append(
            TransportStep(
                mode='walk',
                instruction=f"Walk to {end_name}",
                distance=walk_from_station_dist * 1000,
                duration=int((walk_from_station_dist / 5.0) * 60),
                start_location=(end_station['lat'], end_station['lng']),
                end_location=end
            )
        )
        
        total_duration = sum(step.duration for step in steps)
        
        return TransportRoute(
            steps=steps,
            total_distance=sum(step.distance for step in steps),
            total_duration=total_duration,
            summary=f"Via {start_station['line']}: {total_duration} min",
            modes_used=[step.mode for step in steps]
        )
    
    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in kilometers (Haversine formula)"""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1 = radians(loc1[0]), radians(loc1[1])
        lat2, lon2 = radians(loc2[0]), radians(loc2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return 6371.0 * c  # Earth radius in kilometers
    
    def format_directions_text(self, route: TransportRoute) -> str:
        """Format route as detailed text directions (Google Maps style)"""
        
        if not route or not route.steps:
            return "No route available"
        
        lines = []
        lines.append(f"📍 **{route.summary}**")
        lines.append(f"⏱️ Total time: {route.total_duration} min")
        lines.append(f"📏 Total distance: {route.total_distance/1000:.1f} km")
        lines.append("")
        
        for i, step in enumerate(route.steps, 1):
            icon = {
                'walk': '🚶',
                'metro': '🚇',
                'tram': '🚊',
                'bus': '🚌',
                'ferry': '⛴️'
            }.get(step.mode, '➡️')
            
            lines.append(f"{i}. {icon} **{step.instruction}**")
            lines.append(f"   📏 {step.distance/1000:.1f} km • ⏱️ {step.duration} min")
            
            if step.line_name:
                lines.append(f"   🚇 Line: {step.line_name}")
            if step.stops_count:
                lines.append(f"   🛑 {step.stops_count} stops")
            lines.append("")
        
        return "\n".join(lines)


# Singleton instance
_service_instance = None

def get_transportation_service() -> TransportationDirectionsService:
    """Get singleton instance of transportation service"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TransportationDirectionsService()
    return _service_instance
