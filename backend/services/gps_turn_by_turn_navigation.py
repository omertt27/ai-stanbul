#!/usr/bin/env python3
"""
GPS Turn-by-Turn Navigation System
===================================

Real-time GPS navigation with turn-by-turn directions for AI Istanbul.

Features:
- Real-time location tracking
- Turn-by-turn voice/text instructions
- Off-route detection and automatic rerouting
- Progress tracking and ETA updates
- Multi-language support (English, Turkish)
- Audio announcements at key points
- Speed-adaptive instructions

Usage:
    navigator = GPSTurnByTurnNavigator(route, mode='walking')
    navigator.start_navigation(current_location)
    state = navigator.update_location(new_location)
    print(state.current_instruction.text)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class NavigationMode(Enum):
    """Navigation transport modes"""
    WALKING = "walking"
    CYCLING = "cycling"
    DRIVING = "driving"
    TRANSIT = "transit"


class InstructionType(Enum):
    """Types of navigation instructions"""
    DEPART = "depart"
    TURN_RIGHT = "turn-right"
    TURN_LEFT = "turn-left"
    TURN_SLIGHT_RIGHT = "turn-slight-right"
    TURN_SLIGHT_LEFT = "turn-slight-left"
    TURN_SHARP_RIGHT = "turn-sharp-right"
    TURN_SHARP_LEFT = "turn-sharp-left"
    CONTINUE = "continue"
    ARRIVE = "arrive"
    ROUNDABOUT = "roundabout"
    EXIT_ROUNDABOUT = "exit-roundabout"
    UTURN = "uturn"
    MERGE = "merge"
    FORK_LEFT = "fork-left"
    FORK_RIGHT = "fork-right"
    END_OF_ROAD_LEFT = "end-of-road-left"
    END_OF_ROAD_RIGHT = "end-of-road-right"


class AnnouncementTiming(Enum):
    """When to announce instructions"""
    PREPARE = "prepare"      # "In 50 meters..."
    IMMINENT = "imminent"    # "Soon..."
    IMMEDIATE = "immediate"  # "Now"
    ONGOING = "ongoing"      # "Continue..."


@dataclass
class GPSLocation:
    """GPS location data"""
    latitude: float
    longitude: float
    accuracy: float = 10.0  # meters
    altitude: Optional[float] = None
    speed: Optional[float] = None  # m/s
    bearing: Optional[float] = None  # degrees
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        """Get (lat, lon) tuple"""
        return (self.latitude, self.longitude)


@dataclass
class RouteStep:
    """Single step in a route"""
    instruction_type: InstructionType
    distance: float  # meters
    duration: float  # seconds
    start_location: Tuple[float, float]  # (lat, lon)
    end_location: Tuple[float, float]  # (lat, lon)
    street_name: str = ""
    bearing_before: float = 0.0  # degrees
    bearing_after: float = 0.0  # degrees
    geometry: List[Tuple[float, float]] = field(default_factory=list)  # Path coordinates
    exit_number: Optional[int] = None  # For roundabouts
    
    def __post_init__(self):
        if not self.geometry:
            self.geometry = [self.start_location, self.end_location]


@dataclass
class NavigationInstruction:
    """Navigation instruction with timing"""
    text: str
    distance_to_maneuver: float  # meters
    timing: AnnouncementTiming
    instruction_type: InstructionType
    street_name: str = ""
    should_announce: bool = True
    icon: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'distance': self.distance_to_maneuver,
            'timing': self.timing.value,
            'type': self.instruction_type.value,
            'street': self.street_name,
            'announce': self.should_announce,
            'icon': self.icon
        }


@dataclass
class NavigationProgress:
    """Progress information"""
    distance_traveled: float  # meters
    distance_remaining: float  # meters
    total_distance: float  # meters
    time_elapsed: timedelta
    eta: datetime
    percent_complete: float
    current_speed: float = 0.0  # m/s
    average_speed: float = 0.0  # m/s


@dataclass
class NavigationState:
    """Complete navigation state"""
    current_location: GPSLocation
    current_instruction: NavigationInstruction
    next_instruction: Optional[NavigationInstruction]
    progress: NavigationProgress
    current_step_index: int
    off_route: bool = False
    rerouting: bool = False
    has_arrived: bool = False
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'location': {
                'lat': self.current_location.latitude,
                'lon': self.current_location.longitude,
                'accuracy': self.current_location.accuracy,
                'speed': self.current_location.speed,
                'bearing': self.current_location.bearing
            },
            'instruction': self.current_instruction.to_dict(),
            'next_instruction': self.next_instruction.to_dict() if self.next_instruction else None,
            'progress': {
                'distance_traveled': self.progress.distance_traveled,
                'distance_remaining': self.progress.distance_remaining,
                'total_distance': self.progress.total_distance,
                'percent_complete': self.progress.percent_complete,
                'time_elapsed_seconds': self.progress.time_elapsed.total_seconds(),
                'eta': self.progress.eta.isoformat(),
                'current_speed_kmh': self.progress.current_speed * 3.6,
                'average_speed_kmh': self.progress.average_speed * 3.6
            },
            'status': {
                'current_step': self.current_step_index,
                'off_route': self.off_route,
                'rerouting': self.rerouting,
                'arrived': self.has_arrived
            },
            'warnings': self.warnings
        }


class GPSTurnByTurnNavigator:
    """
    Real-time GPS navigation with turn-by-turn directions
    """
    
    # Announcement distances for different modes (meters)
    ANNOUNCEMENT_DISTANCES = {
        NavigationMode.WALKING: {
            'prepare': 50,    # "In 50 meters..."
            'imminent': 20,   # "Soon..."
            'immediate': 10   # "Now"
        },
        NavigationMode.CYCLING: {
            'prepare': 100,
            'imminent': 30,
            'immediate': 15
        },
        NavigationMode.DRIVING: {
            'prepare': 500,
            'imminent': 100,
            'immediate': 30
        },
        NavigationMode.TRANSIT: {
            'prepare': 100,
            'imminent': 50,
            'immediate': 10
        }
    }
    
    # Off-route detection thresholds
    OFF_ROUTE_THRESHOLD = {
        NavigationMode.WALKING: 25,   # meters
        NavigationMode.CYCLING: 40,
        NavigationMode.DRIVING: 50,
        NavigationMode.TRANSIT: 30
    }
    
    # Arrival detection threshold
    ARRIVAL_THRESHOLD = {
        NavigationMode.WALKING: 10,   # meters
        NavigationMode.CYCLING: 15,
        NavigationMode.DRIVING: 20,
        NavigationMode.TRANSIT: 15
    }
    
    def __init__(
        self,
        route_steps: List[RouteStep],
        mode: NavigationMode = NavigationMode.WALKING,
        language: str = "en"
    ):
        """
        Initialize GPS navigation
        
        Args:
            route_steps: List of route steps
            mode: Navigation mode
            language: Language for instructions (en, tr)
        """
        self.route_steps = route_steps
        self.mode = mode
        self.language = language
        
        self.current_step_index = 0
        self.start_time: Optional[datetime] = None
        self.start_location: Optional[GPSLocation] = None
        self.destination = route_steps[-1].end_location if route_steps else None
        
        self.is_navigating = False
        self.has_arrived = False
        self.off_route_count = 0
        
        self.location_history: List[GPSLocation] = []
        self.instruction_history: List[NavigationInstruction] = []
        self.last_announced_instruction: Optional[NavigationInstruction] = None
        
        # Calculate total route distance
        self.total_distance = sum(step.distance for step in route_steps)
        
        logger.info(f"✅ GPS Navigation initialized: {len(route_steps)} steps, {self.total_distance:.0f}m")
    
    def start_navigation(self, start_location: GPSLocation) -> NavigationState:
        """
        Start navigation from current location
        
        Args:
            start_location: Current GPS location
            
        Returns:
            Initial navigation state
        """
        self.is_navigating = True
        self.start_time = datetime.now()
        self.start_location = start_location
        self.current_step_index = 0
        self.has_arrived = False
        self.off_route_count = 0
        
        self.location_history.append(start_location)
        
        logger.info(f"🧭 Navigation started from {start_location.coordinates}")
        
        # Generate first instruction
        return self.update_location(start_location)
    
    def update_location(self, location: GPSLocation) -> NavigationState:
        """
        Update current location and get navigation state
        
        Args:
            location: Current GPS location
            
        Returns:
            Updated navigation state
        """
        if not self.is_navigating:
            raise ValueError("Navigation not started. Call start_navigation() first.")
        
        self.location_history.append(location)
        
        # Check if arrived
        if self._check_arrival(location):
            self.has_arrived = True
            self.is_navigating = False
            return self._create_arrival_state(location)
        
        # Check if off route
        off_route = self._check_off_route(location)
        
        if off_route:
            self.off_route_count += 1
            logger.warning(f"⚠️ Off route detected ({self.off_route_count} times)")
        else:
            self.off_route_count = 0
        
        # Update current step based on progress
        self._update_current_step(location)
        
        # Generate instructions
        current_instruction = self._generate_current_instruction(location)
        next_instruction = self._generate_next_instruction()
        
        # Check if we should announce
        if self._should_announce(current_instruction):
            self.last_announced_instruction = current_instruction
            logger.info(f"📢 Announce: {current_instruction.text}")
        
        # Calculate progress
        progress = self._calculate_progress(location)
        
        # Create warnings
        warnings = []
        if off_route and self.off_route_count >= 3:
            warnings.append("You are off route. Consider rerouting.")
        
        # Create state
        state = NavigationState(
            current_location=location,
            current_instruction=current_instruction,
            next_instruction=next_instruction,
            progress=progress,
            current_step_index=self.current_step_index,
            off_route=off_route,
            rerouting=False,
            has_arrived=False,
            warnings=warnings
        )
        
        return state
    
    def stop_navigation(self):
        """Stop navigation"""
        self.is_navigating = False
        logger.info("🛑 Navigation stopped")
    
    def _check_arrival(self, location: GPSLocation) -> bool:
        """Check if arrived at destination"""
        if not self.destination:
            return False
        
        distance = self._calculate_distance(
            location.coordinates,
            self.destination
        )
        
        threshold = self.ARRIVAL_THRESHOLD[self.mode]
        return distance < threshold
    
    def _check_off_route(self, location: GPSLocation) -> bool:
        """Check if location is off the planned route"""
        if not self.route_steps or self.current_step_index >= len(self.route_steps):
            return False
        
        current_step = self.route_steps[self.current_step_index]
        
        # Find closest point on current step's geometry
        min_distance = float('inf')
        
        for point in current_step.geometry:
            distance = self._calculate_distance(location.coordinates, point)
            min_distance = min(min_distance, distance)
        
        threshold = self.OFF_ROUTE_THRESHOLD[self.mode]
        return min_distance > threshold
    
    def _update_current_step(self, location: GPSLocation):
        """Update current step index based on location"""
        if self.current_step_index >= len(self.route_steps):
            return
        
        current_step = self.route_steps[self.current_step_index]
        
        # Check if we've reached the end of current step
        distance_to_end = self._calculate_distance(
            location.coordinates,
            current_step.end_location
        )
        
        # If very close to end of step, advance to next
        if distance_to_end < 15 and self.current_step_index < len(self.route_steps) - 1:
            self.current_step_index += 1
            logger.info(f"📍 Advanced to step {self.current_step_index + 1}/{len(self.route_steps)}")
    
    def _generate_current_instruction(self, location: GPSLocation) -> NavigationInstruction:
        """Generate current navigation instruction"""
        if self.current_step_index >= len(self.route_steps):
            return self._create_arrival_instruction(location)
        
        current_step = self.route_steps[self.current_step_index]
        
        # Calculate distance to maneuver
        distance_to_maneuver = self._calculate_distance(
            location.coordinates,
            current_step.end_location
        )
        
        # Determine timing
        timing = self._determine_timing(distance_to_maneuver)
        
        # Generate text
        text = self._generate_instruction_text(
            current_step,
            distance_to_maneuver,
            timing
        )
        
        # Get icon
        icon = self._get_instruction_icon(current_step.instruction_type)
        
        return NavigationInstruction(
            text=text,
            distance_to_maneuver=distance_to_maneuver,
            timing=timing,
            instruction_type=current_step.instruction_type,
            street_name=current_step.street_name,
            should_announce=True,
            icon=icon
        )
    
    def _generate_next_instruction(self) -> Optional[NavigationInstruction]:
        """Generate next instruction (preview)"""
        next_index = self.current_step_index + 1
        
        if next_index >= len(self.route_steps):
            return None
        
        next_step = self.route_steps[next_index]
        
        # Simple preview text
        text = self._generate_simple_instruction_text(next_step)
        
        return NavigationInstruction(
            text=text,
            distance_to_maneuver=next_step.distance,
            timing=AnnouncementTiming.PREPARE,
            instruction_type=next_step.instruction_type,
            street_name=next_step.street_name,
            should_announce=False,
            icon=self._get_instruction_icon(next_step.instruction_type)
        )
    
    def _determine_timing(self, distance: float) -> AnnouncementTiming:
        """Determine announcement timing based on distance"""
        distances = self.ANNOUNCEMENT_DISTANCES[self.mode]
        
        if distance <= distances['immediate']:
            return AnnouncementTiming.IMMEDIATE
        elif distance <= distances['imminent']:
            return AnnouncementTiming.IMMINENT
        elif distance <= distances['prepare']:
            return AnnouncementTiming.PREPARE
        else:
            return AnnouncementTiming.ONGOING
    
    def _generate_instruction_text(
        self,
        step: RouteStep,
        distance: float,
        timing: AnnouncementTiming
    ) -> str:
        """Generate instruction text in appropriate language"""
        
        # Get base instruction
        instruction = self._get_base_instruction(step.instruction_type, step.street_name)
        
        # Add distance prefix based on timing
        if timing == AnnouncementTiming.PREPARE:
            distance_text = self._format_distance(distance)
            if self.language == "tr":
                text = f"{distance_text} sonra {instruction}"
            else:
                text = f"In {distance_text}, {instruction}"
        
        elif timing == AnnouncementTiming.IMMINENT:
            if self.language == "tr":
                text = f"Yakında {instruction}"
            else:
                text = f"Soon, {instruction}"
        
        elif timing == AnnouncementTiming.IMMEDIATE:
            if self.language == "tr":
                text = f"Şimdi {instruction}"
            else:
                text = f"Now, {instruction}"
        
        else:  # ONGOING
            text = instruction
        
        return text
    
    def _generate_simple_instruction_text(self, step: RouteStep) -> str:
        """Generate simple instruction text for next step preview"""
        instruction = self._get_base_instruction(step.instruction_type, step.street_name)
        distance_text = self._format_distance(step.distance)
        
        if self.language == "tr":
            return f"Sonra: {instruction} ({distance_text})"
        else:
            return f"Then: {instruction} ({distance_text})"
    
    def _get_base_instruction(self, instruction_type: InstructionType, street_name: str = "") -> str:
        """Get base instruction text for instruction type"""
        
        street = f" onto {street_name}" if street_name else ""
        street_tr = f" {street_name}'e" if street_name else ""
        
        if self.language == "tr":
            instructions = {
                InstructionType.DEPART: f"Başlayın{street_tr}",
                InstructionType.TURN_RIGHT: f"Sağa dönün{street_tr}",
                InstructionType.TURN_LEFT: f"Sola dönün{street_tr}",
                InstructionType.TURN_SLIGHT_RIGHT: f"Hafif sağa dönün{street_tr}",
                InstructionType.TURN_SLIGHT_LEFT: f"Hafif sola dönün{street_tr}",
                InstructionType.TURN_SHARP_RIGHT: f"Keskin sağa dönün{street_tr}",
                InstructionType.TURN_SHARP_LEFT: f"Keskin sola dönün{street_tr}",
                InstructionType.CONTINUE: f"Devam edin{street_tr}",
                InstructionType.ARRIVE: "Varış noktasındasınız",
                InstructionType.ROUNDABOUT: "Dönel kavşağa girin",
                InstructionType.UTURN: "U dönüşü yapın",
                InstructionType.MERGE: f"Birleşin{street_tr}",
                InstructionType.FORK_LEFT: f"Sol şeridi takip edin{street_tr}",
                InstructionType.FORK_RIGHT: f"Sağ şeridi takip edin{street_tr}",
            }
        else:
            instructions = {
                InstructionType.DEPART: f"Depart{street}",
                InstructionType.TURN_RIGHT: f"Turn right{street}",
                InstructionType.TURN_LEFT: f"Turn left{street}",
                InstructionType.TURN_SLIGHT_RIGHT: f"Bear right{street}",
                InstructionType.TURN_SLIGHT_LEFT: f"Bear left{street}",
                InstructionType.TURN_SHARP_RIGHT: f"Sharp right{street}",
                InstructionType.TURN_SHARP_LEFT: f"Sharp left{street}",
                InstructionType.CONTINUE: f"Continue{street}",
                InstructionType.ARRIVE: "You have arrived",
                InstructionType.ROUNDABOUT: "Enter the roundabout",
                InstructionType.UTURN: "Make a U-turn",
                InstructionType.MERGE: f"Merge{street}",
                InstructionType.FORK_LEFT: f"Keep left{street}",
                InstructionType.FORK_RIGHT: f"Keep right{street}",
            }
        
        return instructions.get(instruction_type, "Continue")
    
    def _format_distance(self, distance: float) -> str:
        """Format distance for display"""
        if distance < 100:
            if self.language == "tr":
                return f"{int(distance)} metre"
            else:
                return f"{int(distance)} meters"
        elif distance < 1000:
            rounded = round(distance / 50) * 50
            if self.language == "tr":
                return f"{int(rounded)} metre"
            else:
                return f"{int(rounded)} meters"
        else:
            km = distance / 1000
            if self.language == "tr":
                return f"{km:.1f} kilometre"
            else:
                return f"{km:.1f} kilometers"
    
    def _get_instruction_icon(self, instruction_type: InstructionType) -> str:
        """Get icon name for instruction type"""
        icons = {
            InstructionType.DEPART: "🚶",
            InstructionType.TURN_RIGHT: "➡️",
            InstructionType.TURN_LEFT: "⬅️",
            InstructionType.TURN_SLIGHT_RIGHT: "↗️",
            InstructionType.TURN_SLIGHT_LEFT: "↖️",
            InstructionType.TURN_SHARP_RIGHT: "⤴️",
            InstructionType.TURN_SHARP_LEFT: "⤵️",
            InstructionType.CONTINUE: "⬆️",
            InstructionType.ARRIVE: "🎯",
            InstructionType.ROUNDABOUT: "🔄",
            InstructionType.UTURN: "↩️",
            InstructionType.MERGE: "🔀",
            InstructionType.FORK_LEFT: "↖️",
            InstructionType.FORK_RIGHT: "↗️",
        }
        return icons.get(instruction_type, "⬆️")
    
    def _should_announce(self, instruction: NavigationInstruction) -> bool:
        """Determine if instruction should be announced"""
        # Don't announce if same as last announced
        if self.last_announced_instruction:
            if (instruction.instruction_type == self.last_announced_instruction.instruction_type and
                instruction.timing == self.last_announced_instruction.timing and
                abs(instruction.distance_to_maneuver - self.last_announced_instruction.distance_to_maneuver) < 5):
                return False
        
        # Only announce at specific timings
        return instruction.timing in [
            AnnouncementTiming.PREPARE,
            AnnouncementTiming.IMMINENT,
            AnnouncementTiming.IMMEDIATE
        ]
    
    def _calculate_progress(self, location: GPSLocation) -> NavigationProgress:
        """Calculate navigation progress"""
        # Calculate distance traveled
        distance_traveled = 0.0
        for i in range(self.current_step_index):
            distance_traveled += self.route_steps[i].distance
        
        # Add distance within current step
        if self.current_step_index < len(self.route_steps):
            current_step = self.route_steps[self.current_step_index]
            distance_to_step_end = self._calculate_distance(
                location.coordinates,
                current_step.end_location
            )
            distance_traveled += max(0, current_step.distance - distance_to_step_end)
        
        # Calculate distance remaining
        distance_remaining = self.total_distance - distance_traveled
        
        # Calculate time elapsed
        time_elapsed = datetime.now() - self.start_time
        
        # Calculate average speed
        if time_elapsed.total_seconds() > 0:
            average_speed = distance_traveled / time_elapsed.total_seconds()
        else:
            average_speed = 0.0
        
        # Calculate ETA
        if average_speed > 0:
            remaining_seconds = distance_remaining / average_speed
            eta = datetime.now() + timedelta(seconds=remaining_seconds)
        else:
            # Use mode's default speed
            default_speed = self._get_default_speed()
            remaining_seconds = distance_remaining / default_speed
            eta = datetime.now() + timedelta(seconds=remaining_seconds)
        
        # Calculate percent complete
        percent_complete = (distance_traveled / self.total_distance * 100) if self.total_distance > 0 else 0
        
        return NavigationProgress(
            distance_traveled=distance_traveled,
            distance_remaining=distance_remaining,
            total_distance=self.total_distance,
            time_elapsed=time_elapsed,
            eta=eta,
            percent_complete=percent_complete,
            current_speed=location.speed or 0.0,
            average_speed=average_speed
        )
    
    def _create_arrival_state(self, location: GPSLocation) -> NavigationState:
        """Create state for arrival"""
        instruction = self._create_arrival_instruction(location)
        
        progress = NavigationProgress(
            distance_traveled=self.total_distance,
            distance_remaining=0,
            total_distance=self.total_distance,
            time_elapsed=datetime.now() - self.start_time,
            eta=datetime.now(),
            percent_complete=100.0,
            current_speed=0.0,
            average_speed=self.total_distance / (datetime.now() - self.start_time).total_seconds()
        )
        
        return NavigationState(
            current_location=location,
            current_instruction=instruction,
            next_instruction=None,
            progress=progress,
            current_step_index=len(self.route_steps),
            off_route=False,
            rerouting=False,
            has_arrived=True,
            warnings=[]
        )
    
    def _create_arrival_instruction(self, location: GPSLocation) -> NavigationInstruction:
        """Create arrival instruction"""
        if self.language == "tr":
            text = "🎯 Varış noktasına ulaştınız!"
        else:
            text = "🎯 You have arrived at your destination!"
        
        return NavigationInstruction(
            text=text,
            distance_to_maneuver=0,
            timing=AnnouncementTiming.IMMEDIATE,
            instruction_type=InstructionType.ARRIVE,
            should_announce=True,
            icon="🎯"
        )
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates (Haversine formula)"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371000  # Earth's radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _get_default_speed(self) -> float:
        """Get default speed for mode (m/s)"""
        speeds = {
            NavigationMode.WALKING: 1.4,   # 5 km/h
            NavigationMode.CYCLING: 4.2,   # 15 km/h
            NavigationMode.DRIVING: 8.3,   # 30 km/h
            NavigationMode.TRANSIT: 5.6    # 20 km/h
        }
        return speeds.get(self.mode, 1.4)
    
    def get_route_overview(self) -> Dict[str, Any]:
        """Get overview of entire route"""
        return {
            'total_distance': self.total_distance,
            'total_steps': len(self.route_steps),
            'estimated_duration': self.total_distance / self._get_default_speed(),
            'mode': self.mode.value,
            'steps': [
                {
                    'type': step.instruction_type.value,
                    'distance': step.distance,
                    'duration': step.duration,
                    'street': step.street_name,
                    'icon': self._get_instruction_icon(step.instruction_type)
                }
                for step in self.route_steps
            ]
        }


# Helper function to convert OSRM route to RouteSteps
def convert_osrm_to_steps(osrm_route: Dict[str, Any]) -> List[RouteStep]:
    """
    Convert OSRM route format to RouteStep list
    
    Args:
        osrm_route: Route from OSRM API
        
    Returns:
        List of RouteStep objects
    """
    steps = []
    
    if 'legs' not in osrm_route:
        return steps
    
    for leg in osrm_route['legs']:
        if 'steps' not in leg:
            continue
        
        for step_data in leg['steps']:
            # Extract maneuver type
            maneuver_type = step_data.get('maneuver', {}).get('type', 'continue')
            modifier = step_data.get('maneuver', {}).get('modifier', '')
            
            # Map OSRM types to our InstructionType
            instruction_type = _map_osrm_type(maneuver_type, modifier)
            
            # Extract locations
            start_coords = step_data.get('maneuver', {}).get('location', [])
            if start_coords:
                start_location = (start_coords[1], start_coords[0])  # OSRM uses lon,lat
            else:
                start_location = (0, 0)
            
            # Extract geometry for end location
            geometry = step_data.get('geometry', {})
            if geometry and 'coordinates' in geometry:
                coords = geometry['coordinates']
                if coords:
                    end_coords = coords[-1]
                    end_location = (end_coords[1], end_coords[0])
                    # Convert all geometry points
                    geometry_points = [(c[1], c[0]) for c in coords]
                else:
                    end_location = start_location
                    geometry_points = [start_location]
            else:
                end_location = start_location
                geometry_points = [start_location]
            
            # Create RouteStep
            step = RouteStep(
                instruction_type=instruction_type,
                distance=step_data.get('distance', 0),
                duration=step_data.get('duration', 0),
                start_location=start_location,
                end_location=end_location,
                street_name=step_data.get('name', ''),
                bearing_before=step_data.get('maneuver', {}).get('bearing_before', 0),
                bearing_after=step_data.get('maneuver', {}).get('bearing_after', 0),
                geometry=geometry_points,
                exit_number=step_data.get('maneuver', {}).get('exit', None)
            )
            
            steps.append(step)
    
    return steps


def _map_osrm_type(maneuver_type: str, modifier: str = '') -> InstructionType:
    """Map OSRM maneuver type to InstructionType"""
    type_map = {
        'depart': InstructionType.DEPART,
        'arrive': InstructionType.ARRIVE,
        'turn': {
            'right': InstructionType.TURN_RIGHT,
            'left': InstructionType.TURN_LEFT,
            'slight right': InstructionType.TURN_SLIGHT_RIGHT,
            'slight left': InstructionType.TURN_SLIGHT_LEFT,
            'sharp right': InstructionType.TURN_SHARP_RIGHT,
            'sharp left': InstructionType.TURN_SHARP_LEFT,
        },
        'continue': InstructionType.CONTINUE,
        'merge': InstructionType.MERGE,
        'fork': {
            'left': InstructionType.FORK_LEFT,
            'right': InstructionType.FORK_RIGHT,
        },
        'roundabout': InstructionType.ROUNDABOUT,
        'rotary': InstructionType.ROUNDABOUT,
        'uturn': InstructionType.UTURN,
        'end of road': {
            'left': InstructionType.END_OF_ROAD_LEFT,
            'right': InstructionType.END_OF_ROAD_RIGHT,
        }
    }
    
    if maneuver_type in type_map:
        mapping = type_map[maneuver_type]
        if isinstance(mapping, dict):
            return mapping.get(modifier, InstructionType.CONTINUE)
        return mapping
    
    return InstructionType.CONTINUE


# Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing GPS Turn-by-Turn Navigation...\n")
    
    # Create sample route
    steps = [
        RouteStep(
            instruction_type=InstructionType.DEPART,
            distance=100,
            duration=72,
            start_location=(41.0054, 28.9768),
            end_location=(41.0064, 28.9768),
            street_name="Sultanahmet Square"
        ),
        RouteStep(
            instruction_type=InstructionType.TURN_RIGHT,
            distance=200,
            duration=144,
            start_location=(41.0064, 28.9768),
            end_location=(41.0064, 28.9788),
            street_name="Divanyolu Street"
        ),
        RouteStep(
            instruction_type=InstructionType.TURN_LEFT,
            distance=150,
            duration=108,
            start_location=(41.0064, 28.9788),
            end_location=(41.0079, 28.9788),
            street_name="Yerebatan Street"
        ),
        RouteStep(
            instruction_type=InstructionType.ARRIVE,
            distance=50,
            duration=36,
            start_location=(41.0079, 28.9788),
            end_location=(41.0086, 28.9802),
            street_name="Hagia Sophia"
        )
    ]
    
    # Initialize navigator
    navigator = GPSTurnByTurnNavigator(steps, mode=NavigationMode.WALKING, language="en")
    
    print("📊 Route Overview:")
    overview = navigator.get_route_overview()
    print(f"  Total Distance: {overview['total_distance']:.0f}m")
    print(f"  Total Steps: {overview['total_steps']}")
    print(f"  Estimated Duration: {overview['estimated_duration']/60:.1f} minutes")
    print()
    
    # Simulate navigation
    print("🧭 Starting Navigation...\n")
    
    # Start location
    start_loc = GPSLocation(latitude=41.0054, longitude=28.9768, accuracy=10)
    state = navigator.start_navigation(start_loc)
    
    print(f"Step 1 - Current: {state.current_instruction.text}")
    print(f"  Distance to maneuver: {state.current_instruction.distance_to_maneuver:.0f}m")
    print(f"  Progress: {state.progress.percent_complete:.1f}%")
    print()
    
    # Simulate movement
    test_locations = [
        (41.0058, 28.9768, "Moving forward..."),
        (41.0062, 28.9768, "Approaching turn..."),
        (41.0064, 28.9770, "After turn..."),
        (41.0064, 28.9780, "On Divanyolu..."),
        (41.0070, 28.9788, "Approaching next turn..."),
        (41.0082, 28.9795, "Near destination..."),
        (41.0086, 28.9802, "At destination!"),
    ]
    
    for i, (lat, lon, desc) in enumerate(test_locations, 2):
        location = GPSLocation(latitude=lat, longitude=lon, accuracy=8, speed=1.4)
        state = navigator.update_location(location)
        
        print(f"Step {i} - {desc}")
        print(f"  Current: {state.current_instruction.text}")
        print(f"  Distance to maneuver: {state.current_instruction.distance_to_maneuver:.0f}m")
        print(f"  Progress: {state.progress.percent_complete:.1f}%")
        print(f"  ETA: {state.progress.eta.strftime('%H:%M:%S')}")
        
        if state.next_instruction:
            print(f"  Next: {state.next_instruction.text}")
        
        if state.has_arrived:
            print("\n🎉 Navigation completed!")
            break
        
        print()
    
    print("\n✅ Test completed!")
