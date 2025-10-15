"""
Multi-Modal Journey Planner
============================

Advanced journey planning system that combines multiple transportation modes
for optimal routing considering time, cost, comfort, and real-time conditions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from .real_time_schedule_service import RealTimeScheduleService


class TravelPreference(Enum):
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    MOST_COMFORTABLE = "most_comfortable"
    LEAST_WALKING = "least_walking"
    MOST_SCENIC = "most_scenic"
    ECO_FRIENDLY = "eco_friendly"


class TransportMode(Enum):
    METRO = "metro"
    TRAM = "tram"
    BUS = "bus"
    FERRY = "ferry"
    WALKING = "walking"
    TAXI = "taxi"
    RIDESHARE = "rideshare"


@dataclass
class JourneySegment:
    """Individual segment of a multi-modal journey"""
    mode: TransportMode
    line_id: str
    from_station: str
    to_station: str
    duration: int  # minutes
    cost: float  # TL
    walking_distance: int  # meters
    instructions: str
    real_time_delay: int = 0
    comfort_score: float = 0.8
    co2_impact: float = 0.0  # kg CO2


@dataclass
class CompleteJourney:
    """Complete multi-modal journey with all segments"""
    segments: List[JourneySegment]
    total_duration: int
    total_cost: float
    total_walking: int
    comfort_score: float
    eco_score: float
    preference_match: float
    real_time_reliability: float
    weather_impact: str
    accessibility_rating: str


class MultiModalJourneyPlanner:
    """Advanced multi-modal journey planning system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.schedule_service = RealTimeScheduleService()
        
        # Initialize transport network data
        self.transport_network = self._initialize_transport_network()
        self.station_connections = self._initialize_station_connections()
        self.cost_matrix = self._initialize_cost_matrix()
    
    def _initialize_transport_network(self) -> Dict[str, Dict[str, Any]]:
        """Initialize detailed transport network with connections"""
        return {
            # Major transport hubs with their connections
            'taksim': {
                'coordinates': (41.0369, 28.9857),
                'transport_modes': {
                    TransportMode.METRO: ['M2'],
                    TransportMode.BUS: ['500T', '25E', '28', '30D'],
                    TransportMode.TAXI: True,
                    TransportMode.WALKING: ['İstiklal Caddesi', 'Galata Tower area']
                },
                'nearby_stations': {
                    'Şişhane': {'distance': 800, 'walking_time': 10},
                    'Osmanbey': {'distance': 1200, 'walking_time': 15}
                },
                'poi_category': 'entertainment_shopping'
            },
            'sultanahmet': {
                'coordinates': (41.0056, 28.9769),
                'transport_modes': {
                    TransportMode.TRAM: ['T1'],
                    TransportMode.BUS: ['28', '30D', '99A'],
                    TransportMode.TAXI: True,
                    TransportMode.WALKING: ['Historic Peninsula']
                },
                'nearby_stations': {
                    'Gülhane': {'distance': 600, 'walking_time': 8},
                    'Eminönü': {'distance': 1000, 'walking_time': 12}
                },
                'poi_category': 'historic_cultural'
            },
            'kadikoy': {
                'coordinates': (40.9904, 29.0257),
                'transport_modes': {
                    TransportMode.METRO: ['M4'],
                    TransportMode.FERRY: ['KADIKOY_EMINONU'],
                    TransportMode.BUS: ['16', '20', '222'],
                    TransportMode.TAXI: True
                },
                'nearby_stations': {
                    'Haydarpaşa': {'distance': 1200, 'walking_time': 15}
                },
                'poi_category': 'entertainment_dining'
            },
            'airport': {
                'coordinates': (41.2753, 28.7519),
                'transport_modes': {
                    TransportMode.METRO: ['M11'],
                    TransportMode.BUS: ['HAVAIST'],
                    TransportMode.TAXI: True,
                    TransportMode.RIDESHARE: True
                },
                'nearby_stations': {},
                'poi_category': 'transport_hub'
            },
            'galata_tower': {
                'coordinates': (41.0256, 28.9744),
                'transport_modes': {
                    TransportMode.METRO: ['M2'],  # via Şişhane
                    TransportMode.TRAM: ['T1'],  # via Karaköy
                    TransportMode.BUS: ['28', '99A'],
                    TransportMode.WALKING: ['Karaköy', 'İstiklal Caddesi']
                },
                'nearby_stations': {
                    'Şişhane': {'distance': 400, 'walking_time': 5},
                    'Karaköy': {'distance': 800, 'walking_time': 10}
                },
                'poi_category': 'historic_scenic'
            },
            'besiktas': {
                'coordinates': (41.0422, 29.0067),
                'transport_modes': {
                    TransportMode.BUS: ['500T', '25E'],
                    TransportMode.FERRY: ['BOSPHORUS_TOUR'],
                    TransportMode.TAXI: True
                },
                'nearby_stations': {
                    'Kabataş': {'distance': 1000, 'walking_time': 12}
                },
                'poi_category': 'waterfront_entertainment'
            }
        }
    
    def _initialize_station_connections(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize inter-station connections for transfers"""
        return {
            'yenikapi': [
                {'to': 'M1A', 'walking_time': 2, 'escalator': True},
                {'to': 'M1B', 'walking_time': 2, 'escalator': True},
                {'to': 'M2', 'walking_time': 3, 'escalator': True}
            ],
            'gayrettepe': [
                {'to': 'M2', 'walking_time': 3, 'escalator': True},
                {'to': 'M11', 'walking_time': 2, 'same_level': True}
            ],
            'karakoy': [
                {'to': 'T1', 'walking_time': 1, 'same_level': True},
                {'to': 'ferry', 'walking_time': 3, 'outdoor': True}
            ],
            'sisli': [
                {'to': 'M2', 'walking_time': 0, 'same_platform': True}
            ]
        }
    
    def _initialize_cost_matrix(self) -> Dict[TransportMode, Dict[str, Any]]:
        """Initialize cost information for different transport modes"""
        return {
            TransportMode.METRO: {
                'cost_category': 'public_transport',
                'cost_level': 'low',  # Standard public transport fare
                'comfort_score': 0.9,
                'co2_per_km': 0.04,  # kg CO2 per km
                'payment_info': 'İstanbulkart recommended for best fares'
            },
            TransportMode.TRAM: {
                'cost_category': 'public_transport',
                'cost_level': 'low',
                'comfort_score': 0.8,
                'co2_per_km': 0.05,
                'payment_info': 'Same fare system as metro'
            },
            TransportMode.BUS: {
                'cost_category': 'public_transport',
                'cost_level': 'low',
                'comfort_score': 0.6,
                'co2_per_km': 0.08,
                'payment_info': 'Same fare system as metro/tram'
            },
            TransportMode.FERRY: {
                'cost_category': 'public_transport',
                'cost_level': 'low',
                'comfort_score': 0.95,
                'co2_per_km': 0.03,
                'payment_info': 'Same fare system, scenic route'
            },
            TransportMode.TAXI: {
                'cost_category': 'private_transport',
                'cost_level': 'high',
                'comfort_score': 0.95,
                'co2_per_km': 0.15,
                'payment_info': 'Metered fare, cash or card accepted'
            },
            TransportMode.RIDESHARE: {
                'cost_category': 'private_transport',
                'cost_level': 'medium-high',
                'comfort_score': 0.90,
                'co2_per_km': 0.12,
                'payment_info': 'App-based pricing, varies by demand'
            },
            TransportMode.WALKING: {
                'cost_category': 'free',
                'cost_level': 'free',
                'comfort_score': 0.7,
                'co2_per_km': 0,
                'payment_info': 'No cost'
            }
        }
    
    async def plan_journey(
        self,
        origin: str,
        destination: str,
        preferences: List[TravelPreference] = None,
        departure_time: datetime = None,
        user_profile: Dict[str, Any] = None
    ) -> List[CompleteJourney]:
        """Plan comprehensive multi-modal journeys"""
        
        if not departure_time:
            departure_time = datetime.now()
        
        if not preferences:
            preferences = [TravelPreference.FASTEST]
        
        # Get possible route combinations
        route_combinations = await self._generate_route_combinations(origin, destination)
        
        # Evaluate each combination
        evaluated_journeys = []
        for combination in route_combinations:
            journey = await self._evaluate_route_combination(
                combination, preferences, departure_time, user_profile
            )
            if journey:
                evaluated_journeys.append(journey)
        
        # Sort by preference match and return top options
        evaluated_journeys.sort(key=lambda j: j.preference_match, reverse=True)
        
        return evaluated_journeys[:5]  # Return top 5 options
    
    async def _generate_route_combinations(
        self, 
        origin: str, 
        destination: str
    ) -> List[List[Dict[str, Any]]]:
        """Generate possible route combinations between origin and destination"""
        
        combinations = []
        
        # Direct routes (single mode)
        direct_routes = self._find_direct_routes(origin, destination)
        combinations.extend(direct_routes)
        
        # Two-mode combinations
        hub_routes = self._find_hub_transfer_routes(origin, destination)
        combinations.extend(hub_routes)
        
        # Three-mode combinations (for complex routes)
        complex_routes = self._find_complex_routes(origin, destination)
        combinations.extend(complex_routes)
        
        return combinations
    
    def _find_direct_routes(self, origin: str, destination: str) -> List[List[Dict[str, Any]]]:
        """Find direct single-mode routes"""
        routes = []
        
        origin_data = self.transport_network.get(origin.lower())
        dest_data = self.transport_network.get(destination.lower())
        
        if not origin_data or not dest_data:
            return routes
        
        # Walking route (always available for reasonable distances)
        distance = self._calculate_distance(
            origin_data['coordinates'], 
            dest_data['coordinates']
        )
        
        if distance <= 5000:  # Max 5km walking
            walking_time = int(distance / 80)  # 80m/min walking speed
            routes.append([{
                'mode': TransportMode.WALKING,
                'from': origin,
                'to': destination,
                'duration': walking_time,
                'distance': distance
            }])
        
        # Taxi route (always available)
        taxi_time = max(10, int(distance / 500))  # ~30km/h in city
        routes.append([{
            'mode': TransportMode.TAXI,
            'from': origin,
            'to': destination,
            'duration': taxi_time,
            'distance': distance
        }])
        
        # Check for direct public transport
        origin_modes = origin_data.get('transport_modes', {})
        dest_modes = dest_data.get('transport_modes', {})
        
        for mode in origin_modes:
            if mode in dest_modes and mode != TransportMode.TAXI:
                # Find common lines
                origin_lines = origin_modes[mode] if isinstance(origin_modes[mode], list) else []
                dest_lines = dest_modes[mode] if isinstance(dest_modes[mode], list) else []
                
                common_lines = set(origin_lines) & set(dest_lines)
                
                for line in common_lines:
                    estimated_time = self._estimate_transit_time(mode, distance)
                    routes.append([{
                        'mode': mode,
                        'line': line,
                        'from': origin,
                        'to': destination,
                        'duration': estimated_time,
                        'distance': distance
                    }])
        
        return routes
    
    def _find_hub_transfer_routes(self, origin: str, destination: str) -> List[List[Dict[str, Any]]]:
        """Find routes with one transfer at major hubs"""
        routes = []
        
        # Major transfer hubs
        hubs = ['taksim', 'yenikapi', 'karakoy', 'besiktas', 'gayrettepe']
        
        origin_data = self.transport_network.get(origin.lower())
        dest_data = self.transport_network.get(destination.lower())
        
        if not origin_data or not dest_data:
            return routes
        
        for hub in hubs:
            if hub == origin.lower() or hub == destination.lower():
                continue
            
            hub_data = self.transport_network.get(hub)
            if not hub_data:
                continue
            
            # Find route from origin to hub
            origin_to_hub = self._find_best_connection(origin_data, hub_data, hub)
            
            # Find route from hub to destination
            hub_to_dest = self._find_best_connection(hub_data, dest_data, destination)
            
            if origin_to_hub and hub_to_dest:
                # Add transfer time
                transfer_segment = {
                    'mode': TransportMode.WALKING,
                    'from': f"{hub} (transfer)",
                    'to': f"{hub} (transfer)",
                    'duration': 5,  # 5 min transfer time
                    'distance': 200
                }
                
                routes.append([origin_to_hub, transfer_segment, hub_to_dest])
        
        return routes
    
    def _find_complex_routes(self, origin: str, destination: str) -> List[List[Dict[str, Any]]]:
        """Find complex multi-modal routes"""
        # For now, return empty - can be expanded for very complex routing
        return []
    
    def _find_best_connection(
        self, 
        from_data: Dict[str, Any], 
        to_data: Dict[str, Any], 
        to_location: str
    ) -> Optional[Dict[str, Any]]:
        """Find best connection between two locations"""
        
        from_modes = from_data.get('transport_modes', {})
        to_modes = to_data.get('transport_modes', {})
        
        # Check for common transport modes
        for mode in from_modes:
            if mode in to_modes and mode != TransportMode.TAXI:
                if isinstance(from_modes[mode], list) and isinstance(to_modes[mode], list):
                    common_lines = set(from_modes[mode]) & set(to_modes[mode])
                    if common_lines:
                        line = list(common_lines)[0]
                        distance = self._calculate_distance(
                            from_data['coordinates'], 
                            to_data['coordinates']
                        )
                        estimated_time = self._estimate_transit_time(mode, distance)
                        
                        return {
                            'mode': mode,
                            'line': line,
                            'from': from_data.get('name', 'origin'),
                            'to': to_location,
                            'duration': estimated_time,
                            'distance': distance
                        }
        
        return None
    
    async def _evaluate_route_combination(
        self,
        combination: List[Dict[str, Any]],
        preferences: List[TravelPreference],
        departure_time: datetime,
        user_profile: Dict[str, Any] = None
    ) -> Optional[CompleteJourney]:
        """Evaluate a route combination and create a complete journey"""
        
        segments = []
        total_duration = 0
        total_cost = 0
        total_walking = 0
        comfort_scores = []
        co2_impacts = []
        
        for segment_data in combination:
            segment = await self._create_journey_segment(segment_data, departure_time)
            if not segment:
                return None
            
            segments.append(segment)
            total_duration += segment.duration + segment.real_time_delay
            total_cost += segment.cost
            total_walking += segment.walking_distance
            comfort_scores.append(segment.comfort_score)
            co2_impacts.append(segment.co2_impact)
        
        # Calculate overall scores
        avg_comfort = sum(comfort_scores) / len(comfort_scores) if comfort_scores else 0.5
        total_co2 = sum(co2_impacts)
        eco_score = max(0, 1 - (total_co2 / 10))  # Normalize to 0-1 scale
        
        # Calculate preference match
        preference_match = self._calculate_preference_match(
            segments, preferences, total_duration, total_cost, total_walking, avg_comfort
        )
        
        # Calculate reliability based on real-time data
        reliability = self._calculate_reliability(segments)
        
        # Determine weather impact
        weather_impact = self._assess_weather_impact(segments)
        
        # Assess accessibility
        accessibility = self._assess_accessibility(segments)
        
        return CompleteJourney(
            segments=segments,
            total_duration=total_duration,
            total_cost=total_cost,
            total_walking=total_walking,
            comfort_score=avg_comfort,
            eco_score=eco_score,
            preference_match=preference_match,
            real_time_reliability=reliability,
            weather_impact=weather_impact,
            accessibility_rating=accessibility
        )
    
    async def _create_journey_segment(
        self, 
        segment_data: Dict[str, Any], 
        departure_time: datetime
    ) -> Optional[JourneySegment]:
        """Create a journey segment with real-time data"""
        
        mode = segment_data['mode']
        
        # Get real-time delay if applicable
        real_time_delay = 0
        if mode in [TransportMode.METRO, TransportMode.TRAM, TransportMode.BUS]:
            line_id = segment_data.get('line', '')
            if line_id:
                schedule_info = self.schedule_service.get_real_time_schedule(line_id)
                real_time_delay = schedule_info.estimated_delay
        
        # Get cost information (now using cost categories instead of specific amounts)
        cost_info = self.cost_matrix.get(mode, self.cost_matrix[TransportMode.WALKING])
        distance_km = segment_data.get('distance', 1000) / 1000
        
        # Assign cost category scores for internal calculations (0-10 scale)
        cost_category_scores = {
            'free': 0,
            'low': 2,
            'medium-high': 6,
            'high': 8
        }
        cost = cost_category_scores.get(cost_info['cost_level'], 2)
        
        # Calculate CO2 impact
        co2_impact = cost_info['co2_per_km'] * distance_km
        
        # Generate instructions
        instructions = self._generate_segment_instructions(segment_data)
        
        # Walking distance (for the segment itself if walking, or to/from stations)
        walking_distance = segment_data.get('distance', 0) if mode == TransportMode.WALKING else 100
        
        return JourneySegment(
            mode=mode,
            line_id=segment_data.get('line', ''),
            from_station=segment_data.get('from', ''),
            to_station=segment_data.get('to', ''),
            duration=segment_data.get('duration', 10),
            cost=cost,
            walking_distance=walking_distance,
            instructions=instructions,
            real_time_delay=real_time_delay,
            comfort_score=cost_info['comfort_score'],
            co2_impact=co2_impact
        )
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> int:
        """Calculate distance between two coordinates in meters"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Haversine formula
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return int(R * c)
    
    def _estimate_transit_time(self, mode: TransportMode, distance: int) -> int:
        """Estimate transit time based on mode and distance"""
        # Average speeds in km/h
        speeds = {
            TransportMode.METRO: 35,
            TransportMode.TRAM: 20,
            TransportMode.BUS: 15,
            TransportMode.FERRY: 25,
            TransportMode.TAXI: 25,
            TransportMode.RIDESHARE: 25,
            TransportMode.WALKING: 5
        }
        
        speed = speeds.get(mode, 20)
        time_hours = (distance / 1000) / speed
        return max(3, int(time_hours * 60))  # Minimum 3 minutes
    
    def _calculate_preference_match(
        self, 
        segments: List[JourneySegment], 
        preferences: List[TravelPreference],
        total_duration: int,
        total_cost: float,
        total_walking: int,
        avg_comfort: float
    ) -> float:
        """Calculate how well the journey matches user preferences"""
        
        score = 0.0
        
        for preference in preferences:
            if preference == TravelPreference.FASTEST:
                # Score based on duration (lower is better)
                score += max(0, 1 - (total_duration / 120))  # Normalized against 2 hours
            
            elif preference == TravelPreference.CHEAPEST:
                # Score based on cost level (lower is better, normalized 0-10 scale)
                score += max(0, 1 - (total_cost / 10))  # Normalized against max cost level of 10
            
            elif preference == TravelPreference.MOST_COMFORTABLE:
                score += avg_comfort
            
            elif preference == TravelPreference.LEAST_WALKING:
                # Score based on walking distance (lower is better)
                score += max(0, 1 - (total_walking / 2000))  # Normalized against 2km
            
            elif preference == TravelPreference.ECO_FRIENDLY:
                # Score based on public transport usage
                public_segments = sum(1 for s in segments if s.mode in [
                    TransportMode.METRO, TransportMode.TRAM, 
                    TransportMode.BUS, TransportMode.FERRY
                ])
                score += public_segments / len(segments)
        
        return score / len(preferences) if preferences else 0.5
    
    def _calculate_reliability(self, segments: List[JourneySegment]) -> float:
        """Calculate overall reliability score"""
        delays = [s.real_time_delay for s in segments]
        avg_delay = sum(delays) / len(delays) if delays else 0
        
        # Convert delay to reliability score (0-1)
        return max(0, 1 - (avg_delay / 20))  # 20 min delay = 0 reliability
    
    def _assess_weather_impact(self, segments: List[JourneySegment]) -> str:
        """Assess weather impact on the journey"""
        outdoor_segments = sum(1 for s in segments if s.mode in [
            TransportMode.WALKING, TransportMode.FERRY
        ])
        
        if outdoor_segments == 0:
            return "Weather-independent"
        elif outdoor_segments <= len(segments) / 2:
            return "Minimal weather impact"
        else:
            return "Weather-dependent"
    
    def _assess_accessibility(self, segments: List[JourneySegment]) -> str:
        """Assess accessibility of the journey"""
        # Simple assessment - can be made more sophisticated
        walking_segments = sum(1 for s in segments if s.mode == TransportMode.WALKING)
        
        if walking_segments <= 1 and all(s.mode in [TransportMode.METRO, TransportMode.TRAM] 
                                       for s in segments if s.mode != TransportMode.WALKING):
            return "Fully accessible"
        elif walking_segments <= 2:
            return "Mostly accessible"
        else:
            return "Limited accessibility"
    
    def _generate_segment_instructions(self, segment_data: Dict[str, Any]) -> str:
        """Generate human-readable instructions for a segment"""
        mode = segment_data['mode']
        
        if mode == TransportMode.METRO:
            return f"Take M{segment_data.get('line', '')} Metro from {segment_data.get('from', '')} to {segment_data.get('to', '')}"
        elif mode == TransportMode.TRAM:
            return f"Take T{segment_data.get('line', '')} Tram from {segment_data.get('from', '')} to {segment_data.get('to', '')}"
        elif mode == TransportMode.BUS:
            return f"Take Bus {segment_data.get('line', '')} from {segment_data.get('from', '')} to {segment_data.get('to', '')}"
        elif mode == TransportMode.FERRY:
            return f"Take Ferry from {segment_data.get('from', '')} to {segment_data.get('to', '')}"
        elif mode == TransportMode.WALKING:
            distance = segment_data.get('distance', 0)
            return f"Walk {distance}m from {segment_data.get('from', '')} to {segment_data.get('to', '')} ({segment_data.get('duration', 5)} min)"
        elif mode == TransportMode.TAXI:
            return f"Take Taxi from {segment_data.get('from', '')} to {segment_data.get('to', '')}"
        else:
            return f"Travel from {segment_data.get('from', '')} to {segment_data.get('to', '')}"
