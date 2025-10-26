"""
Walking Directions Generator
Generates turn-by-turn walking instructions from GPS location to transit stops
Production-grade implementation with accurate distance and time calculations
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class WalkingStep:
    """Single walking instruction step"""
    instruction: str
    distance_m: int
    duration_sec: int
    direction: str
    coordinates: Tuple[float, float]  # lat, lon
    
    def to_dict(self) -> Dict:
        return {
            'instruction': self.instruction,
            'distance_m': self.distance_m,
            'distance_text': self._format_distance(),
            'duration_sec': self.duration_sec,
            'duration_text': self._format_duration(),
            'direction': self.direction,
            'coordinates': {
                'lat': self.coordinates[0],
                'lon': self.coordinates[1]
            }
        }
    
    def _format_distance(self) -> str:
        """Format distance for human reading"""
        if self.distance_m < 1000:
            return f"{self.distance_m}m"
        else:
            return f"{self.distance_m / 1000:.1f}km"
    
    def _format_duration(self) -> str:
        """Format duration for human reading"""
        if self.duration_sec < 60:
            return f"{self.duration_sec}s"
        else:
            minutes = self.duration_sec // 60
            return f"{minutes}min"


class WalkingDirectionsGenerator:
    """
    Production-grade walking directions generator
    Generates turn-by-turn instructions for walking to transit stops
    """
    
    # Walking speed constants (meters per second)
    WALKING_SPEED_SLOW = 1.0  # 3.6 km/h
    WALKING_SPEED_NORMAL = 1.4  # 5.0 km/h
    WALKING_SPEED_FAST = 1.8  # 6.5 km/h
    
    def __init__(self, walking_speed: float = WALKING_SPEED_NORMAL):
        """
        Initialize walking directions generator
        
        Args:
            walking_speed: Walking speed in meters/second (default: normal speed)
        """
        self.walking_speed = walking_speed
        self.earth_radius_m = 6371000  # Earth radius in meters
    
    def generate_walking_directions(self,
                                   from_lat: float,
                                   from_lon: float,
                                   to_lat: float,
                                   to_lon: float,
                                   to_name: str,
                                   transport_type: Optional[str] = None) -> Dict:
        """
        Generate walking directions from GPS location to destination
        
        Args:
            from_lat: Starting latitude (user's GPS)
            from_lon: Starting longitude (user's GPS)
            to_lat: Destination latitude (stop)
            to_lon: Destination longitude (stop)
            to_name: Destination name (stop name)
            transport_type: Type of transport at destination (e.g., 'metro', 'bus')
            
        Returns:
            Walking directions dictionary with steps and summary
        """
        # Calculate total distance and bearing
        total_distance_m = self._haversine_distance(
            from_lat, from_lon, to_lat, to_lon
        )
        
        bearing = self._calculate_bearing(
            from_lat, from_lon, to_lat, to_lon
        )
        
        direction = self._bearing_to_direction(bearing)
        cardinal_direction = self._bearing_to_cardinal(bearing)
        
        # Calculate walking time
        duration_sec = int(total_distance_m / self.walking_speed)
        
        # Generate walking steps
        steps = self._generate_steps(
            from_lat, from_lon,
            to_lat, to_lon,
            to_name,
            transport_type,
            bearing,
            direction,
            cardinal_direction,
            total_distance_m
        )
        
        # Create summary
        summary = {
            'total_distance_m': int(total_distance_m),
            'total_distance_km': round(total_distance_m / 1000, 2),
            'total_duration_sec': duration_sec,
            'total_duration_min': max(1, duration_sec // 60),
            'walking_speed_mps': self.walking_speed,
            'bearing': round(bearing, 1),
            'direction': direction,
            'cardinal_direction': cardinal_direction,
            'steps': [step.to_dict() for step in steps],
            'step_count': len(steps),
            'destination_name': to_name,
            'transport_type': transport_type,
            'start_coordinates': {'lat': from_lat, 'lon': from_lon},
            'end_coordinates': {'lat': to_lat, 'lon': to_lon}
        }
        
        return summary
    
    def _generate_steps(self,
                       from_lat: float, from_lon: float,
                       to_lat: float, to_lon: float,
                       to_name: str,
                       transport_type: Optional[str],
                       bearing: float,
                       direction: str,
                       cardinal_direction: str,
                       total_distance_m: float) -> List[WalkingStep]:
        """
        Generate turn-by-turn walking steps
        For short walks, generates 1-2 steps
        For longer walks, generates intermediate waypoints
        """
        steps = []
        duration_sec = int(total_distance_m / self.walking_speed)
        
        # For very short walks (< 100m), single instruction
        if total_distance_m < 100:
            transport_emoji = self._get_transport_emoji(transport_type)
            instruction = (
                f"Walk {direction} to {to_name} {transport_emoji}"
            )
            
            step = WalkingStep(
                instruction=instruction,
                distance_m=int(total_distance_m),
                duration_sec=duration_sec,
                direction=direction,
                coordinates=(to_lat, to_lon)
            )
            steps.append(step)
            
            return steps
        
        # For longer walks, generate initial and final instructions
        
        # Step 1: Initial direction
        transport_emoji = self._get_transport_emoji(transport_type)
        initial_instruction = (
            f"Head {direction} ({cardinal_direction}) towards {to_name} {transport_emoji}"
        )
        
        # Allocate 30% of distance to initial step
        initial_distance = int(total_distance_m * 0.3)
        initial_duration = int(initial_distance / self.walking_speed)
        
        # Calculate intermediate waypoint
        mid_lat, mid_lon = self._interpolate_coordinates(
            from_lat, from_lon, to_lat, to_lon, 0.3
        )
        
        step1 = WalkingStep(
            instruction=initial_instruction,
            distance_m=initial_distance,
            duration_sec=initial_duration,
            direction=direction,
            coordinates=(mid_lat, mid_lon)
        )
        steps.append(step1)
        
        # Step 2: Continue walking (if distance > 300m)
        if total_distance_m > 300:
            continue_distance = int(total_distance_m * 0.5)
            continue_duration = int(continue_distance / self.walking_speed)
            
            mid2_lat, mid2_lon = self._interpolate_coordinates(
                from_lat, from_lon, to_lat, to_lon, 0.7
            )
            
            continue_instruction = f"Continue {direction} for {continue_distance}m"
            
            step2 = WalkingStep(
                instruction=continue_instruction,
                distance_m=continue_distance,
                duration_sec=continue_duration,
                direction=direction,
                coordinates=(mid2_lat, mid2_lon)
            )
            steps.append(step2)
        
        # Final step: Arrival
        final_distance = int(total_distance_m * 0.2)
        final_duration = int(final_distance / self.walking_speed)
        
        arrival_instruction = f"Arrive at {to_name} {transport_emoji}"
        
        final_step = WalkingStep(
            instruction=arrival_instruction,
            distance_m=final_distance,
            duration_sec=final_duration,
            direction=direction,
            coordinates=(to_lat, to_lon)
        )
        steps.append(final_step)
        
        return steps
    
    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance using Haversine formula
        Returns distance in meters
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon/2)**2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return self.earth_radius_m * c
    
    def _calculate_bearing(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """
        Calculate compass bearing between two points
        Returns bearing in degrees (0-360)
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def _bearing_to_direction(self, bearing: float) -> str:
        """
        Convert bearing to human-readable direction
        Returns: 'north', 'northeast', 'east', etc.
        """
        directions = [
            (0, 22.5, 'north'),
            (22.5, 67.5, 'northeast'),
            (67.5, 112.5, 'east'),
            (112.5, 157.5, 'southeast'),
            (157.5, 202.5, 'south'),
            (202.5, 247.5, 'southwest'),
            (247.5, 292.5, 'west'),
            (292.5, 337.5, 'northwest'),
            (337.5, 360, 'north')
        ]
        
        for min_deg, max_deg, direction in directions:
            if min_deg <= bearing < max_deg:
                return direction
        
        return 'north'
    
    def _bearing_to_cardinal(self, bearing: float) -> str:
        """
        Convert bearing to cardinal direction abbreviation
        Returns: 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'
        """
        cardinal_map = {
            'north': 'N',
            'northeast': 'NE',
            'east': 'E',
            'southeast': 'SE',
            'south': 'S',
            'southwest': 'SW',
            'west': 'W',
            'northwest': 'NW'
        }
        
        direction = self._bearing_to_direction(bearing)
        return cardinal_map.get(direction, 'N')
    
    def _interpolate_coordinates(self, lat1: float, lon1: float,
                                lat2: float, lon2: float,
                                fraction: float) -> Tuple[float, float]:
        """
        Calculate intermediate point between two coordinates
        
        Args:
            lat1, lon1: Start coordinates
            lat2, lon2: End coordinates
            fraction: Position along path (0.0 = start, 1.0 = end)
            
        Returns:
            (lat, lon) tuple of intermediate point
        """
        # Simple linear interpolation (good enough for short distances)
        lat = lat1 + (lat2 - lat1) * fraction
        lon = lon1 + (lon2 - lon1) * fraction
        
        return (lat, lon)
    
    def _get_transport_emoji(self, transport_type: Optional[str]) -> str:
        """Get emoji for transport type"""
        emoji_map = {
            'metro': '🚇',
            'metrobus': '🚌',
            'bus': '🚌',
            'tram': '🚊',
            'funicular': '🚡',
            'ferry': '⛴️',
            'marmaray': '🚆'
        }
        
        if transport_type:
            return emoji_map.get(transport_type.lower(), '🚏')
        return '🚏'
    
    def generate_summary_text(self, directions: Dict) -> str:
        """
        Generate human-readable summary of walking directions
        
        Args:
            directions: Walking directions dictionary from generate_walking_directions
            
        Returns:
            Formatted text summary
        """
        lines = []
        
        # Header
        transport_emoji = self._get_transport_emoji(directions.get('transport_type'))
        lines.append(
            f"🚶 Walking to {directions['destination_name']} {transport_emoji}"
        )
        lines.append(
            f"📏 {directions['total_distance_m']}m "
            f"({directions['total_distance_km']}km)"
        )
        lines.append(
            f"⏱️ {directions['total_duration_min']} minutes"
        )
        lines.append("")
        
        # Steps
        lines.append("Turn-by-turn directions:")
        for i, step in enumerate(directions['steps'], 1):
            lines.append(
                f"{i}. {step['instruction']} "
                f"({step['distance_text']}, ~{step['duration_text']})"
            )
        
        return '\n'.join(lines)
