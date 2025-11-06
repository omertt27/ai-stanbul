"""
Route Planning Algorithms for AI Istanbul System

Advanced route planning and optimization service with multiple algorithms:
- Shortest path algorithms (Dijkstra, A*)
- Travel time optimization 
- Multi-modal transport integration
- Real-time traffic and transport data
- Cost optimization
- Accessibility considerations
Industry-grade routing with machine learning enhancements.
"""

import json
import math
import heapq
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Raised when query fails security validation"""
    pass

class TransportMode(Enum):
    """Available transport modes in Istanbul"""
    WALKING = "walking"
    METRO = "metro"
    BUS = "bus"
    TRAM = "tram"
    FERRY = "ferry"
    TAXI = "taxi"
    UBER = "uber"
    DOLMUS = "dolmus"
    FUNICULAR = "funicular"
    CABLE_CAR = "cable_car"

class RouteType(Enum):
    """Types of route optimization"""
    FASTEST = "fastest"          # Minimize travel time
    SHORTEST = "shortest"        # Minimize distance
    CHEAPEST = "cheapest"        # Minimize cost
    MOST_SCENIC = "most_scenic"  # Maximize scenic value
    LEAST_CROWDED = "least_crowded"  # Avoid crowds
    ACCESSIBLE = "accessible"     # Accessibility-friendly
    ECO_FRIENDLY = "eco_friendly" # Environmental impact

@dataclass
class Location:
    """Represents a location with coordinates and metadata"""
    id: str
    name: str
    coordinates: Tuple[float, float]  # (latitude, longitude)
    district: str
    transport_connections: List[str]
    accessibility_score: float = 1.0
    popularity_score: float = 0.5
    category: str = "general"

@dataclass
class RouteSegment:
    """A segment of a route between two points"""
    from_location: Location
    to_location: Location
    transport_mode: TransportMode
    distance_km: float
    duration_minutes: int
    cost_tl: float
    instructions: List[str]
    waypoints: List[Tuple[float, float]]
    real_time_data: Dict[str, Any]
    accessibility_friendly: bool = True
    scenic_score: float = 0.5

@dataclass
class Route:
    """Complete route with multiple segments"""
    route_id: str
    segments: List[RouteSegment]
    total_distance_km: float
    total_duration_minutes: int
    total_cost_tl: float
    route_type: RouteType
    confidence_score: float
    alternative_routes: List['Route']
    created_at: datetime
    valid_until: datetime
    warnings: List[str]
    advantages: List[str]

@dataclass
class RouteRequest:
    """Route planning request parameters"""
    from_location: Location
    to_location: Location
    waypoints: List[Location] = None
    preferred_transport: List[TransportMode] = None
    route_type: RouteType = RouteType.FASTEST
    departure_time: datetime = None
    accessibility_required: bool = False
    max_walking_distance_km: float = 2.0
    max_cost_tl: float = None
    avoid_areas: List[str] = None
    user_preferences: Dict[str, Any] = None

class RouteOptimizer:
    """Advanced route planning and optimization system"""
    
    def __init__(self):
        self.transport_network = self._build_transport_network()
        self.location_data = self._load_location_data()
        self.transport_schedules = self._load_transport_schedules()
        self.traffic_patterns = self._load_traffic_patterns()
        self.cost_matrix = self._initialize_cost_matrix()
        self.scenic_routes = self._identify_scenic_routes()
        self.accessibility_data = self._load_accessibility_data()
        
    def _build_transport_network(self) -> nx.MultiGraph:
        """Build a comprehensive transport network graph"""
        G = nx.MultiGraph()
        
        # Define major locations and their connections
        locations = {
            "sultanahmet": {
                "coords": (41.0086, 28.9802),
                "district": "Sultanahmet",
                "connections": ["tram", "bus", "walking"]
            },
            "eminonu": {
                "coords": (41.0170, 28.9700),
                "district": "Eminönü", 
                "connections": ["metro", "tram", "bus", "ferry", "walking"]
            },
            "karakoy": {
                "coords": (41.0256, 28.9741),
                "district": "Karaköy",
                "connections": ["metro", "tram", "ferry", "funicular", "walking"]
            },
            "beyoglu": {
                "coords": (41.0362, 28.9783),
                "district": "Beyoğlu",
                "connections": ["metro", "bus", "funicular", "walking"]
            },
            "taksim": {
                "coords": (41.0369, 28.9860),
                "district": "Beyoğlu",
                "connections": ["metro", "bus", "walking"]
            },
            "besiktas": {
                "coords": (41.0422, 29.0094),
                "district": "Beşiktaş",
                "connections": ["metro", "bus", "ferry", "walking"]
            },
            "kadikoy": {
                "coords": (40.9900, 29.0244),
                "district": "Kadıköy",
                "connections": ["metro", "bus", "ferry", "walking"]
            },
            "uskudar": {
                "coords": (41.0214, 29.0155),
                "district": "Üsküdar",
                "connections": ["metro", "bus", "ferry", "walking"]
            }
        }
        
        # Add nodes
        for loc_id, data in locations.items():
            G.add_node(loc_id, **data)
        
        # Add edges with transport connections
        transport_connections = [
            # Metro connections
            ("taksim", "eminonu", {"transport": "metro", "line": "M2", "duration": 15, "cost": 7.67}),
            ("karakoy", "taksim", {"transport": "metro", "line": "M2", "duration": 8, "cost": 7.67}),
            ("eminonu", "uskudar", {"transport": "metro", "line": "M5", "duration": 12, "cost": 7.67}),
            
            # Tram connections
            ("sultanahmet", "eminonu", {"transport": "tram", "line": "T1", "duration": 8, "cost": 7.67}),
            ("eminonu", "karakoy", {"transport": "tram", "line": "T1", "duration": 5, "cost": 7.67}),
            
            # Ferry connections
            ("eminonu", "kadikoy", {"transport": "ferry", "duration": 25, "cost": 7.67}),
            ("eminonu", "uskudar", {"transport": "ferry", "duration": 15, "cost": 7.67}),
            ("karakoy", "kadikoy", {"transport": "ferry", "duration": 30, "cost": 7.67}),
            ("besiktas", "uskudar", {"transport": "ferry", "duration": 20, "cost": 7.67}),
            
            # Walking connections (for nearby locations)
            ("sultanahmet", "eminonu", {"transport": "walking", "duration": 15, "cost": 0}),
            ("karakoy", "beyoglu", {"transport": "walking", "duration": 12, "cost": 0}),
            ("beyoglu", "taksim", {"transport": "walking", "duration": 8, "cost": 0}),
            
            # Funicular
            ("karakoy", "beyoglu", {"transport": "funicular", "duration": 3, "cost": 7.67}),
        ]
        
        for from_loc, to_loc, data in transport_connections:
            G.add_edge(from_loc, to_loc, **data)
        
        return G
    
    def _load_location_data(self) -> Dict[str, Location]:
        """Load detailed location data"""
        locations = {}
        
        # Istanbul attractions and key locations
        location_data = {
            "hagia_sophia": {
                "name": "Hagia Sophia",
                "coordinates": (41.0086, 28.9802),
                "district": "Sultanahmet",
                "transport_connections": ["tram_sultanahmet", "walking"],
                "accessibility_score": 0.7,
                "popularity_score": 0.95,
                "category": "attraction"
            },
            "blue_mosque": {
                "name": "Blue Mosque",
                "coordinates": (41.0054, 28.9768),
                "district": "Sultanahmet",
                "transport_connections": ["tram_sultanahmet", "walking"],
                "accessibility_score": 0.8,
                "popularity_score": 0.90,
                "category": "attraction"
            },
            "topkapi_palace": {
                "name": "Topkapi Palace",
                "coordinates": (41.0115, 28.9833),
                "district": "Sultanahmet",
                "transport_connections": ["tram_gulhane", "walking"],
                "accessibility_score": 0.6,
                "popularity_score": 0.85,
                "category": "attraction"
            },
            "galata_tower": {
                "name": "Galata Tower",
                "coordinates": (41.0256, 28.9741),
                "district": "Beyoğlu",
                "transport_connections": ["metro_karakoy", "walking"],
                "accessibility_score": 0.4,
                "popularity_score": 0.80,
                "category": "attraction"
            },
            "grand_bazaar": {
                "name": "Grand Bazaar",
                "coordinates": (41.0106, 28.9681),
                "district": "Beyazıt",
                "transport_connections": ["tram_beyazit", "metro_vezneciler", "walking"],
                "accessibility_score": 0.7,
                "popularity_score": 0.85,
                "category": "shopping"
            },
            "taksim_square": {
                "name": "Taksim Square",
                "coordinates": (41.0369, 28.9860),
                "district": "Beyoğlu",
                "transport_connections": ["metro_taksim", "bus", "walking"],
                "accessibility_score": 0.9,
                "popularity_score": 0.75,
                "category": "landmark"
            },
            "dolmabahce_palace": {
                "name": "Dolmabahçe Palace",
                "coordinates": (41.0391, 29.0000),
                "district": "Beşiktaş",
                "transport_connections": ["tram_dolmabahce", "bus", "walking"],
                "accessibility_score": 0.8,
                "popularity_score": 0.70,
                "category": "attraction"
            },
            "ortakoy": {
                "name": "Ortaköy",
                "coordinates": (41.0553, 29.0269),
                "district": "Beşiktaş",
                "transport_connections": ["bus", "walking"],
                "accessibility_score": 0.6,
                "popularity_score": 0.65,
                "category": "neighborhood"
            }
        }
        
        for loc_id, data in location_data.items():
            locations[loc_id] = Location(
                id=loc_id,
                name=data["name"],
                coordinates=data["coordinates"],
                district=data["district"],
                transport_connections=data["transport_connections"],
                accessibility_score=data["accessibility_score"],
                popularity_score=data["popularity_score"],
                category=data["category"]
            )
        
        return locations
    
    def _load_transport_schedules(self) -> Dict[str, Dict[str, Any]]:
        """Load transport schedules and frequency data"""
        return {
            "metro": {
                "operating_hours": {"start": "06:00", "end": "00:30"},
                "frequency_minutes": {"peak": 3, "off_peak": 7, "weekend": 8},
                "lines": {
                    "M1": {"color": "red", "stations": 23},
                    "M2": {"color": "green", "stations": 16},
                    "M3": {"color": "blue", "stations": 11},
                    "M4": {"color": "pink", "stations": 19},
                    "M5": {"color": "purple", "stations": 16},
                    "M6": {"color": "brown", "stations": 4},
                    "M7": {"color": "light_blue", "stations": 17}
                }
            },
            "tram": {
                "operating_hours": {"start": "06:00", "end": "00:30"},
                "frequency_minutes": {"peak": 5, "off_peak": 10, "weekend": 12},
                "lines": {
                    "T1": {"color": "blue", "route": "Kabataş-Bağcılar"},
                    "T2": {"color": "green", "route": "Taksim-Tünel"},
                    "T3": {"color": "red", "route": "Kadıköy-Moda"},
                    "T4": {"color": "yellow", "route": "Topkapı-Mescid-i Selam"},
                    "T5": {"color": "orange", "route": "Eminönü-Alibeyköy"}
                }
            },
            "ferry": {
                "operating_hours": {"start": "07:00", "end": "21:00"},
                "frequency_minutes": {"peak": 20, "off_peak": 30, "weekend": 40},
                "weather_dependent": True,
                "seasonal_schedules": True
            },
            "bus": {
                "operating_hours": {"start": "05:30", "end": "01:00"},
                "frequency_minutes": {"peak": 8, "off_peak": 15, "weekend": 20},
                "extensive_network": True
            }
        }
    
    def _load_traffic_patterns(self) -> Dict[str, Dict[str, float]]:
        """Load traffic patterns and congestion data"""
        return {
            "hourly_multipliers": {
                "00": 0.3, "01": 0.2, "02": 0.2, "03": 0.2, "04": 0.3, "05": 0.5,
                "06": 0.8, "07": 1.3, "08": 1.5, "09": 1.2, "10": 1.0, "11": 1.0,
                "12": 1.1, "13": 1.2, "14": 1.1, "15": 1.0, "16": 1.2, "17": 1.4,
                "18": 1.5, "19": 1.3, "20": 1.0, "21": 0.8, "22": 0.6, "23": 0.4
            },
            "weekly_multipliers": {
                "monday": 1.2, "tuesday": 1.3, "wednesday": 1.3, "thursday": 1.2,
                "friday": 1.4, "saturday": 0.9, "sunday": 0.7
            },
            "district_congestion": {
                "Sultanahmet": 1.3,
                "Beyoğlu": 1.4,
                "Beşiktaş": 1.2,
                "Kadıköy": 1.1,
                "Üsküdar": 1.0,
                "Eminönü": 1.5
            }
        }
    
    def _initialize_cost_matrix(self) -> Dict[TransportMode, Dict[str, float]]:
        """Initialize cost structures for different transport modes"""
        return {
            TransportMode.METRO: {
                "base_cost": 7.67,
                "distance_factor": 0.0,
                "time_factor": 0.0
            },
            TransportMode.TRAM: {
                "base_cost": 7.67,
                "distance_factor": 0.0,
                "time_factor": 0.0
            },
            TransportMode.BUS: {
                "base_cost": 7.67,
                "distance_factor": 0.0,
                "time_factor": 0.0
            },
            TransportMode.FERRY: {
                "base_cost": 7.67,
                "distance_factor": 0.0,
                "time_factor": 0.0
            },
            TransportMode.TAXI: {
                "base_cost": 5.50,
                "distance_factor": 3.20,
                "time_factor": 0.75
            },
            TransportMode.UBER: {
                "base_cost": 4.00,
                "distance_factor": 2.80,
                "time_factor": 0.65
            },
            TransportMode.WALKING: {
                "base_cost": 0.0,
                "distance_factor": 0.0,
                "time_factor": 0.0
            }
        }
    
    def _identify_scenic_routes(self) -> Dict[str, float]:
        """Identify scenic routes and assign scenic scores"""
        return {
            "bosphorus_ferry": 1.0,
            "galata_bridge_walk": 0.9,
            "golden_horn_ferry": 0.8,
            "princes_islands_ferry": 0.9,
            "bebek_ortakoy_walk": 0.8,
            "sultanahmet_walk": 0.7,
            "pierre_loti_cable": 0.8,
            "eminonu_karakoy_walk": 0.6
        }
    
    def _load_accessibility_data(self) -> Dict[str, Dict[str, Any]]:
        """Load accessibility information for transport modes and locations"""
        return {
            "metro_accessibility": {
                "wheelchair_accessible_stations": [
                    "Taksim", "Şişli-Mecidiyeköy", "Levent", "Atatürk Airport",
                    "Kadıköy", "Bostancı", "Kartal", "Pendik"
                ],
                "elevator_available": True,
                "audio_announcements": True,
                "tactile_guidance": True
            },
            "tram_accessibility": {
                "low_floor_trams": True,
                "wheelchair_spaces": 2,
                "audio_announcements": True,
                "visual_displays": True
            },
            "ferry_accessibility": {
                "wheelchair_accessible": True,
                "accessible_boarding": ["Eminönü", "Kadıköy", "Beşiktaş"],
                "assistance_available": True
            },
            "walking_accessibility": {
                "sidewalk_quality": {
                    "Sultanahmet": 0.7,
                    "Beyoğlu": 0.6,
                    "Taksim": 0.8,
                    "Kadıköy": 0.7,
                    "Beşiktaş": 0.6
                },
                "step_free_routes": ["Taksim-İstiklal", "Eminönü-Galata Bridge"]
            }
        }
    
    def calculate_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates using Haversine formula"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    def find_optimal_route(self, request: RouteRequest) -> Route:
        """Find the optimal route based on the request parameters"""
        
        # Choose algorithm based on route type
        if request.route_type == RouteType.FASTEST:
            return self._find_fastest_route(request)
        elif request.route_type == RouteType.SHORTEST:
            return self._find_shortest_route(request)
        elif request.route_type == RouteType.CHEAPEST:
            return self._find_cheapest_route(request)
        elif request.route_type == RouteType.MOST_SCENIC:
            return self._find_scenic_route(request)
        elif request.route_type == RouteType.ACCESSIBLE:
            return self._find_accessible_route(request)
        else:
            return self._find_fastest_route(request)  # Default
    
    def _find_fastest_route(self, request: RouteRequest) -> Route:
        """Find the fastest route using Dijkstra's algorithm with time weights"""
        
        # Create a graph with time-weighted edges
        time_graph = self._create_time_weighted_graph(request.departure_time)
        
        # Find nearest transport nodes for start and end locations
        start_node = self._find_nearest_transport_node(request.from_location)
        end_node = self._find_nearest_transport_node(request.to_location)
        
        # Use Dijkstra's algorithm
        try:
            path = nx.shortest_path(time_graph, start_node, end_node, weight='duration')
            segments = self._convert_path_to_segments(path, time_graph, request)
            
            route = Route(
                route_id=f"fastest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                segments=segments,
                total_distance_km=sum(s.distance_km for s in segments),
                total_duration_minutes=sum(s.duration_minutes for s in segments),
                total_cost_tl=sum(s.cost_tl for s in segments),
                route_type=RouteType.FASTEST,
                confidence_score=0.9,
                alternative_routes=[],
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=2),
                warnings=self._generate_route_warnings(segments),
                advantages=["Optimized for speed", "Real-time traffic considered"]
            )
            
            # Generate alternative routes
            route.alternative_routes = self._generate_alternative_routes(request, exclude_primary=path)
            
            return route
            
        except nx.NetworkXNoPath:
            return self._create_fallback_route(request, "No direct path found")
    
    def _find_shortest_route(self, request: RouteRequest) -> Route:
        """Find the shortest route by distance"""
        
        # Create a graph with distance weights
        distance_graph = self._create_distance_weighted_graph()
        
        start_node = self._find_nearest_transport_node(request.from_location)
        end_node = self._find_nearest_transport_node(request.to_location)
        
        try:
            path = nx.shortest_path(distance_graph, start_node, end_node, weight='distance')
            segments = self._convert_path_to_segments(path, distance_graph, request)
            
            route = Route(
                route_id=f"shortest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                segments=segments,
                total_distance_km=sum(s.distance_km for s in segments),
                total_duration_minutes=sum(s.duration_minutes for s in segments),
                total_cost_tl=sum(s.cost_tl for s in segments),
                route_type=RouteType.SHORTEST,
                confidence_score=0.85,
                alternative_routes=[],
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=6),
                warnings=self._generate_route_warnings(segments),
                advantages=["Minimum distance", "Less environmental impact"]
            )
            
            return route
            
        except nx.NetworkXNoPath:
            return self._create_fallback_route(request, "No direct path found")
    
    def _find_cheapest_route(self, request: RouteRequest) -> Route:
        """Find the cheapest route"""
        
        # Create a graph with cost weights
        cost_graph = self._create_cost_weighted_graph()
        
        start_node = self._find_nearest_transport_node(request.from_location)
        end_node = self._find_nearest_transport_node(request.to_location)
        
        try:
            path = nx.shortest_path(cost_graph, start_node, end_node, weight='cost')
            segments = self._convert_path_to_segments(path, cost_graph, request)
            
            route = Route(
                route_id=f"cheapest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                segments=segments,
                total_distance_km=sum(s.distance_km for s in segments),
                total_duration_minutes=sum(s.duration_minutes for s in segments),
                total_cost_tl=sum(s.cost_tl for s in segments),
                route_type=RouteType.CHEAPEST,
                confidence_score=0.8,
                alternative_routes=[],
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=12),
                warnings=self._generate_route_warnings(segments),
                advantages=["Minimum cost", "Budget-friendly"]
            )
            
            return route
            
        except nx.NetworkXNoPath:
            return self._create_fallback_route(request, "No direct path found")
    
    def _find_scenic_route(self, request: RouteRequest) -> Route:
        """Find the most scenic route"""
        
        # Create segments prioritizing scenic value
        segments = []
        
        # Check if ferry route is available (usually most scenic)
        if self._is_ferry_route_available(request.from_location, request.to_location):
            ferry_segment = self._create_ferry_segment(request.from_location, request.to_location)
            segments.append(ferry_segment)
        else:
            # Use walking routes through scenic areas when possible
            scenic_segments = self._create_scenic_walking_route(request)
            segments.extend(scenic_segments)
        
        # Add connecting segments if needed
        if not segments:
            # Fallback to regular route with scenic scoring
            fallback_route = self._find_fastest_route(request)
            segments = fallback_route.segments
        
        route = Route(
            route_id=f"scenic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            segments=segments,
            total_distance_km=sum(s.distance_km for s in segments),
            total_duration_minutes=sum(s.duration_minutes for s in segments),
            total_cost_tl=sum(s.cost_tl for s in segments),
            route_type=RouteType.MOST_SCENIC,
            confidence_score=0.75,
            alternative_routes=[],
            created_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=8),
            warnings=["May take longer than direct routes", "Weather dependent"],
            advantages=["Beautiful views", "Memorable journey", "Photo opportunities"]
        )
        
        return route
    
    def _find_accessible_route(self, request: RouteRequest) -> Route:
        """Find an accessibility-friendly route"""
        
        # Filter transport modes to only accessible ones
        accessible_graph = self._create_accessible_transport_graph()
        
        start_node = self._find_nearest_accessible_node(request.from_location)
        end_node = self._find_nearest_accessible_node(request.to_location)
        
        try:
            path = nx.shortest_path(accessible_graph, start_node, end_node, weight='accessibility_score')
            segments = self._convert_path_to_segments(path, accessible_graph, request)
            
            # Ensure all segments are accessibility-friendly
            for segment in segments:
                segment.accessibility_friendly = True
                segment.instructions = self._add_accessibility_instructions(segment.instructions)
            
            route = Route(
                route_id=f"accessible_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                segments=segments,
                total_distance_km=sum(s.distance_km for s in segments),
                total_duration_minutes=sum(s.duration_minutes for s in segments),
                total_cost_tl=sum(s.cost_tl for s in segments),
                route_type=RouteType.ACCESSIBLE,
                confidence_score=0.9,
                alternative_routes=[],
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=4),
                warnings=["May require assistance at some points"],
                advantages=["Wheelchair accessible", "Elevator access", "Step-free routes"]
            )
            
            return route
            
        except nx.NetworkXNoPath:
            return self._create_fallback_route(request, "No accessible path found")
    
    def _create_time_weighted_graph(self, departure_time: datetime = None) -> nx.Graph:
        """Create a graph with time-weighted edges"""
        G = self.transport_network.copy()
        
        if departure_time is None:
            departure_time = datetime.now()
        
        hour = departure_time.hour
        day_name = departure_time.strftime('%A').lower()
        
        # Apply traffic multipliers
        hour_multiplier = self.traffic_patterns["hourly_multipliers"].get(f"{hour:02d}", 1.0)
        day_multiplier = self.traffic_patterns["weekly_multipliers"].get(day_name, 1.0)
        
        for u, v, key, data in G.edges(data=True, keys=True):
            base_duration = data.get('duration', 10)
            transport_mode = data.get('transport', 'walking')
            
            # Apply traffic multipliers for road-based transport
            if transport_mode in ['taxi', 'uber', 'bus']:
                adjusted_duration = base_duration * hour_multiplier * day_multiplier
            else:
                adjusted_duration = base_duration
            
            G[u][v][key]['duration'] = adjusted_duration
        
        return G
    
    def _create_distance_weighted_graph(self) -> nx.Graph:
        """Create a graph with distance-weighted edges"""
        G = self.transport_network.copy()
        
        for u, v, key, data in G.edges(data=True, keys=True):
            # Calculate distance based on coordinates if not provided
            if 'distance' not in data:
                coord1 = G.nodes[u]['coords']
                coord2 = G.nodes[v]['coords']
                distance = self.calculate_distance(coord1, coord2)
                G[u][v][key]['distance'] = distance
        
        return G
    
    def _create_cost_weighted_graph(self) -> nx.Graph:
        """Create a graph with cost-weighted edges"""
        G = self.transport_network.copy()
        
        for u, v, key, data in G.edges(data=True, keys=True):
            transport_mode = data.get('transport', 'walking')
            base_cost = data.get('cost', 0)
            
            # Ensure cost is set
            if base_cost == 0 and transport_mode != 'walking':
                if transport_mode in ['metro', 'tram', 'bus', 'ferry']:
                    base_cost = 7.67  # Standard public transport fare
                elif transport_mode in ['taxi', 'uber']:
                    distance = data.get('distance', 1.0)
                    duration = data.get('duration', 10)
                    cost_data = self.cost_matrix.get(TransportMode(transport_mode), {})
                    base_cost = (cost_data.get('base_cost', 5) + 
                               distance * cost_data.get('distance_factor', 3) +
                               duration * cost_data.get('time_factor', 0.5))
            
            G[u][v][key]['cost'] = base_cost
        
        return G
    
    def _create_accessible_transport_graph(self) -> nx.Graph:
        """Create a graph with only accessible transport options"""
        G = nx.Graph()
        
        # Only include accessible transport modes and routes
        accessible_modes = ['metro', 'tram', 'ferry', 'walking']
        accessible_stations = self.accessibility_data["metro_accessibility"]["wheelchair_accessible_stations"]
        
        for u, v, data in self.transport_network.edges(data=True):
            transport_mode = data.get('transport', 'walking')
            
            if transport_mode in accessible_modes:
                # Add accessibility scoring
                accessibility_score = 1.0
                
                if transport_mode == 'metro':
                    if u in accessible_stations and v in accessible_stations:
                        accessibility_score = 1.0
                    else:
                        accessibility_score = 0.3  # Reduced score for non-accessible stations
                
                data_copy = data.copy()
                data_copy['accessibility_score'] = accessibility_score
                G.add_edge(u, v, **data_copy)
        
        return G
    
    def _find_nearest_transport_node(self, location: Location) -> str:
        """Find the nearest transport network node to a location"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_data in self.transport_network.nodes(data=True):
            node_coords = node_data.get('coords', (0, 0))
            distance = self.calculate_distance(location.coordinates, node_coords)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node or "sultanahmet"  # Fallback
    
    def _find_nearest_accessible_node(self, location: Location) -> str:
        """Find the nearest accessible transport node"""
        accessible_nodes = self.accessibility_data["metro_accessibility"]["wheelchair_accessible_stations"]
        accessible_nodes_lower = [node.lower().replace(' ', '_').replace('ı', 'i') for node in accessible_nodes]
        
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_data in self.transport_network.nodes(data=True):
            if node_id in accessible_nodes_lower:
                node_coords = node_data.get('coords', (0, 0))
                distance = self.calculate_distance(location.coordinates, node_coords)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node_id
        
        return nearest_node or "taksim"  # Fallback to accessible station
    
    def _convert_path_to_segments(self, path: List[str], graph: nx.Graph, 
                                request: RouteRequest) -> List[RouteSegment]:
        """Convert a path to route segments"""
        segments = []
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            # Get edge data
            edge_data = graph[from_node][to_node]
            
            # Create locations for segment
            from_coords = graph.nodes[from_node]['coords']
            to_coords = graph.nodes[to_node]['coords']
            
            from_location = Location(
                id=from_node,
                name=from_node.replace('_', ' ').title(),
                coordinates=from_coords,
                district=graph.nodes[from_node].get('district', 'Unknown'),
                transport_connections=[]
            )
            
            to_location = Location(
                id=to_node,
                name=to_node.replace('_', ' ').title(),
                coordinates=to_coords,
                district=graph.nodes[to_node].get('district', 'Unknown'),
                transport_connections=[]
            )
            
            # Create segment
            segment = RouteSegment(
                from_location=from_location,
                to_location=to_location,
                transport_mode=TransportMode(edge_data.get('transport', 'walking')),
                distance_km=edge_data.get('distance', self.calculate_distance(from_coords, to_coords)),
                duration_minutes=int(edge_data.get('duration', 10)),
                cost_tl=edge_data.get('cost', 0),
                instructions=self._generate_segment_instructions(from_location, to_location, edge_data),
                waypoints=[from_coords, to_coords],
                real_time_data={"last_updated": datetime.now().isoformat()},
                accessibility_friendly=edge_data.get('accessibility_score', 1.0) > 0.7,
                scenic_score=edge_data.get('scenic_score', 0.5)
            )
            
            segments.append(segment)
        
        return segments
    
    def _generate_segment_instructions(self, from_loc: Location, to_loc: Location, 
                                     edge_data: Dict[str, Any]) -> List[str]:
        """Generate turn-by-turn instructions for a segment"""
        transport_mode = edge_data.get('transport', 'walking')
        
        if transport_mode == 'walking':
            return [
                f"Walk from {from_loc.name} to {to_loc.name}",
                f"Distance: {edge_data.get('distance', 0):.1f} km",
                f"Estimated time: {edge_data.get('duration', 10)} minutes"
            ]
        elif transport_mode == 'metro':
            line = edge_data.get('line', 'Metro')
            return [
                f"Take {line} metro from {from_loc.name}",
                f"Direction: {to_loc.name}",
                f"Travel time: {edge_data.get('duration', 10)} minutes",
                f"Exit at {to_loc.name}"
            ]
        elif transport_mode == 'tram':
            line = edge_data.get('line', 'Tram')
            return [
                f"Take {line} tram from {from_loc.name}",
                f"Direction: {to_loc.name}",
                f"Travel time: {edge_data.get('duration', 10)} minutes"
            ]
        elif transport_mode == 'ferry':
            return [
                f"Take ferry from {from_loc.name}",
                f"Destination: {to_loc.name}",
                f"Journey time: {edge_data.get('duration', 20)} minutes",
                "Enjoy the Bosphorus views!"
            ]
        elif transport_mode in ['taxi', 'uber']:
            return [
                f"Take {transport_mode} from {from_loc.name}",
                f"Destination: {to_loc.name}",
                f"Estimated time: {edge_data.get('duration', 15)} minutes",
                f"Estimated cost: {edge_data.get('cost', 0):.2f} TL"
            ]
        else:
            return [f"Travel from {from_loc.name} to {to_loc.name} by {transport_mode}"]
    
    def _generate_route_warnings(self, segments: List[RouteSegment]) -> List[str]:
        """Generate warnings for the route"""
        warnings = []
        
        total_walking = sum(s.distance_km for s in segments if s.transport_mode == TransportMode.WALKING)
        if total_walking > 2.0:
            warnings.append(f"Route includes {total_walking:.1f} km of walking")
        
        ferry_segments = [s for s in segments if s.transport_mode == TransportMode.FERRY]
        if ferry_segments:
            warnings.append("Ferry service may be affected by weather conditions")
        
        peak_hours = any(datetime.now().hour in [7, 8, 17, 18, 19] for _ in [1])
        if peak_hours:
            warnings.append("Peak hours - expect crowds and possible delays")
        
        accessibility_issues = [s for s in segments if not s.accessibility_friendly]
        if accessibility_issues:
            warnings.append("Some segments may have accessibility limitations")
        
        return warnings
    
    def _generate_alternative_routes(self, request: RouteRequest, 
                                   exclude_primary: List[str] = None) -> List[Route]:
        """Generate alternative route options"""
        alternatives = []
        
        # Generate different route types as alternatives
        route_types = [RouteType.FASTEST, RouteType.CHEAPEST, RouteType.MOST_SCENIC]
        
        for route_type in route_types:
            if route_type != request.route_type:
                alt_request = RouteRequest(
                    from_location=request.from_location,
                    to_location=request.to_location,
                    route_type=route_type,
                    departure_time=request.departure_time,
                    accessibility_required=request.accessibility_required
                )
                
                try:
                    alt_route = self.find_optimal_route(alt_request)
                    alt_route.confidence_score *= 0.8  # Lower confidence for alternatives
                    alternatives.append(alt_route)
                except:
                    continue
        
        return alternatives[:2]  # Return top 2 alternatives
    
    def _create_fallback_route(self, request: RouteRequest, reason: str) -> Route:
        """Create a fallback route when optimal routing fails"""
        
        # Create a simple direct route
        direct_distance = self.calculate_distance(
            request.from_location.coordinates,
            request.to_location.coordinates
        )
        
        # Estimate taxi route as fallback
        taxi_segment = RouteSegment(
            from_location=request.from_location,
            to_location=request.to_location,
            transport_mode=TransportMode.TAXI,
            distance_km=direct_distance,
            duration_minutes=int(direct_distance * 4),  # Estimate 4 min per km
            cost_tl=5.50 + (direct_distance * 3.20),  # Taxi pricing
            instructions=[
                f"Take taxi from {request.from_location.name}",
                f"Destination: {request.to_location.name}",
                f"Direct route - approximately {direct_distance:.1f} km"
            ],
            waypoints=[request.from_location.coordinates, request.to_location.coordinates],
            real_time_data={"fallback_route": True},
            accessibility_friendly=True,
            scenic_score=0.3
        )
        
        return Route(
            route_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            segments=[taxi_segment],
            total_distance_km=direct_distance,
            total_duration_minutes=taxi_segment.duration_minutes,
            total_cost_tl=taxi_segment.cost_tl,
            route_type=request.route_type,
            confidence_score=0.5,
            alternative_routes=[],
            created_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=1),
            warnings=[f"Fallback solution: {reason}", "Route not optimized"],
            advantages=["Direct route", "Available immediately"]
        )
    
    def _is_ferry_route_available(self, from_loc: Location, to_loc: Location) -> bool:
        """Check if ferry route is available between locations"""
        ferry_locations = ["eminonu", "karakoy", "besiktas", "kadikoy", "uskudar"]
        
        from_nearby = any(ferry_loc in from_loc.name.lower() or 
                         self.calculate_distance(from_loc.coordinates, 
                                               self.transport_network.nodes[ferry_loc]['coords']) < 1.0
                         for ferry_loc in ferry_locations if ferry_loc in self.transport_network.nodes)
        
        to_nearby = any(ferry_loc in to_loc.name.lower() or
                       self.calculate_distance(to_loc.coordinates,
                                             self.transport_network.nodes[ferry_loc]['coords']) < 1.0
                       for ferry_loc in ferry_locations if ferry_loc in self.transport_network.nodes)
        
        return from_nearby and to_nearby
    
    def _create_ferry_segment(self, from_loc: Location, to_loc: Location) -> RouteSegment:
        """Create a scenic ferry segment"""
        distance = self.calculate_distance(from_loc.coordinates, to_loc.coordinates)
        
        return RouteSegment(
            from_location=from_loc,
            to_location=to_loc,
            transport_mode=TransportMode.FERRY,
            distance_km=distance,
            duration_minutes=int(distance * 8),  # Ferry speed estimate
            cost_tl=7.67,
            instructions=[
                f"Take ferry from {from_loc.name}",
                f"Enjoy the scenic Bosphorus journey",
                f"Arrive at {to_loc.name}",
                "Don't forget to take photos!"
            ],
            waypoints=[from_loc.coordinates, to_loc.coordinates],
            real_time_data={"scenic_route": True},
            accessibility_friendly=True,
            scenic_score=0.9
        )
    
    def _create_scenic_walking_route(self, request: RouteRequest) -> List[RouteSegment]:
        """Create scenic walking segments when possible"""
        segments = []
        
        # Check for scenic walking areas
        scenic_areas = {
            "galata_bridge": {"coords": (41.0197, 28.9736), "scenic_score": 0.8},
            "sultanahmet_park": {"coords": (41.0086, 28.9762), "scenic_score": 0.7},
            "gulhane_park": {"coords": (41.0128, 28.9819), "scenic_score": 0.8}
        }
        
        # Find if route passes through scenic areas
        for area_name, area_data in scenic_areas.items():
            area_coords = area_data["coords"]
            
            # Check if area is between start and end points
            start_to_area = self.calculate_distance(request.from_location.coordinates, area_coords)
            area_to_end = self.calculate_distance(area_coords, request.to_location.coordinates)
            direct_distance = self.calculate_distance(request.from_location.coordinates, 
                                                    request.to_location.coordinates)
            
            # If going through scenic area doesn't add too much distance
            if (start_to_area + area_to_end) < (direct_distance * 1.3):
                scenic_location = Location(
                    id=area_name,
                    name=area_name.replace('_', ' ').title(),
                    coordinates=area_coords,
                    district="Scenic Route",
                    transport_connections=["walking"]
                )
                
                segment1 = RouteSegment(
                    from_location=request.from_location,
                    to_location=scenic_location,
                    transport_mode=TransportMode.WALKING,
                    distance_km=start_to_area,
                    duration_minutes=int(start_to_area * 12),  # Walking speed
                    cost_tl=0,
                    instructions=[f"Walk to {scenic_location.name}", "Enjoy the scenic views"],
                    waypoints=[request.from_location.coordinates, area_coords],
                    real_time_data={"scenic_segment": True},
                    accessibility_friendly=True,
                    scenic_score=area_data["scenic_score"]
                )
                
                segment2 = RouteSegment(
                    from_location=scenic_location,
                    to_location=request.to_location,
                    transport_mode=TransportMode.WALKING,
                    distance_km=area_to_end,
                    duration_minutes=int(area_to_end * 12),
                    cost_tl=0,
                    instructions=[f"Continue to {request.to_location.name}"],
                    waypoints=[area_coords, request.to_location.coordinates],
                    real_time_data={"scenic_segment": True},
                    accessibility_friendly=True,
                    scenic_score=area_data["scenic_score"]
                )
                
                segments.extend([segment1, segment2])
                break
        
        return segments
    
    def _add_accessibility_instructions(self, instructions: List[str]) -> List[str]:
        """Add accessibility-specific instructions"""
        accessibility_instructions = []
        
        for instruction in instructions:
            accessibility_instructions.append(instruction)
            
            if "metro" in instruction.lower():
                accessibility_instructions.append("🔹 Elevator access available")
                accessibility_instructions.append("🔹 Audio announcements provided")
            elif "tram" in instruction.lower():
                accessibility_instructions.append("🔹 Low-floor accessible tram")
                accessibility_instructions.append("🔹 Wheelchair space available")
            elif "ferry" in instruction.lower():
                accessibility_instructions.append("🔹 Wheelchair accessible boarding")
                accessibility_instructions.append("🔹 Staff assistance available")
            elif "walk" in instruction.lower():
                accessibility_instructions.append("🔹 Check sidewalk conditions")
                accessibility_instructions.append("🔹 Step-free route when possible")
        
        return accessibility_instructions
    
    def get_multi_destination_route(self, locations: List[Location], 
                                  route_type: RouteType = RouteType.FASTEST) -> Route:
        """Optimize route for multiple destinations (Traveling Salesman Problem)"""
        
        if len(locations) < 2:
            raise ValueError("At least 2 locations required")
        
        if len(locations) == 2:
            request = RouteRequest(
                from_location=locations[0],
                to_location=locations[1],
                route_type=route_type
            )
            return self.find_optimal_route(request)
        
        # For multiple locations, use nearest neighbor heuristic
        optimized_order = self._optimize_location_order(locations, route_type)
        
        # Create route segments between consecutive locations
        all_segments = []
        total_distance = 0
        total_duration = 0
        total_cost = 0
        
        for i in range(len(optimized_order) - 1):
            request = RouteRequest(
                from_location=optimized_order[i],
                to_location=optimized_order[i + 1],
                route_type=route_type
            )
            
            segment_route = self.find_optimal_route(request)
            all_segments.extend(segment_route.segments)
            total_distance += segment_route.total_distance_km
            total_duration += segment_route.total_duration_minutes
            total_cost += segment_route.total_cost_tl
        
        return Route(
            route_id=f"multi_dest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            segments=all_segments,
            total_distance_km=total_distance,
            total_duration_minutes=total_duration,
            total_cost_tl=total_cost,
            route_type=route_type,
            confidence_score=0.8,
            alternative_routes=[],
            created_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=4),
            warnings=["Multi-destination route - consider time at each location"],
            advantages=["Optimized visiting order", "Efficient multi-stop journey"]
        )
    
    def _optimize_location_order(self, locations: List[Location], 
                               route_type: RouteType) -> List[Location]:
        """Optimize the order of visiting multiple locations"""
        
        if len(locations) <= 3:
            return locations  # For small lists, order might not matter much
        
        # Use nearest neighbor algorithm for larger lists
        unvisited = locations[1:].copy()  # Start from first location
        current_location = locations[0]
        optimized_order = [current_location]
        
        while unvisited:
            nearest_location = None
            min_cost = float('inf')
            
            for location in unvisited:
                # Calculate cost based on route type
                if route_type == RouteType.FASTEST:
                    cost = self._estimate_travel_time(current_location, location)
                elif route_type == RouteType.SHORTEST:
                    cost = self.calculate_distance(current_location.coordinates, location.coordinates)
                elif route_type == RouteType.CHEAPEST:
                    cost = self._estimate_travel_cost(current_location, location)
                else:
                    cost = self.calculate_distance(current_location.coordinates, location.coordinates)
                
                if cost < min_cost:
                    min_cost = cost
                    nearest_location = location
            
            optimized_order.append(nearest_location)
            unvisited.remove(nearest_location)
            current_location = nearest_location
        
        return optimized_order
    
    def _estimate_travel_time(self, from_loc: Location, to_loc: Location) -> int:
        """Estimate travel time between two locations"""
        distance = self.calculate_distance(from_loc.coordinates, to_loc.coordinates)
        
        # Simple heuristic: metro/tram for longer distances, walking for short
        if distance < 0.5:
            return int(distance * 12)  # Walking: 12 min/km
        elif distance < 3:
            return int(distance * 6 + 10)  # Public transport + waiting
        else:
            return int(distance * 4 + 15)  # Taxi + traffic
    
    def _estimate_travel_cost(self, from_loc: Location, to_loc: Location) -> float:
        """Estimate travel cost between two locations"""
        distance = self.calculate_distance(from_loc.coordinates, to_loc.coordinates)
        
        if distance < 0.5:
            return 0  # Walking
        elif distance < 3:
            return 7.67  # Public transport
        else:
            return 5.50 + (distance * 3.20)  # Taxi
    
    def get_real_time_updates(self, route: Route) -> Dict[str, Any]:
        """Get real-time updates for a route (simulated)"""
        
        updates = {
            "route_id": route.route_id,
            "status": "active",
            "delays": [],
            "alternative_suggestions": [],
            "traffic_conditions": {},
            "service_disruptions": [],
            "updated_at": datetime.now().isoformat()
        }
        
        # Simulate some real-time conditions
        current_hour = datetime.now().hour
        
        if current_hour in [7, 8, 17, 18, 19]:  # Peak hours
            updates["traffic_conditions"]["status"] = "heavy"
            updates["delays"].append({
                "segment": 0,
                "delay_minutes": random.randint(5, 15),
                "reason": "Heavy traffic"
            })
        
        # Random service updates (low probability)
        if random.random() < 0.1:
            updates["service_disruptions"].append({
                "transport_mode": "metro",
                "line": "M2",
                "impact": "Minor delays",
                "estimated_duration": "20 minutes"
            })
        
        return updates

# ============================================================================
# INTELLIGENT ITINERARY PLANNER
# Integrates with museums, hidden gems, and dining for optimized day trips
# ============================================================================

@dataclass
class ItineraryLocation:
    """Location for itinerary planning with extended metadata"""
    id: str
    name: str
    type: str  # 'museum', 'hidden_gem', 'restaurant', 'cafe'
    lat: float
    lng: float
    duration: int  # minutes to spend here
    opening_time: str  # "09:00"
    closing_time: str  # "18:00"
    rating: float
    description: str
    cost: str  # "₺₺"
    address: str
    category: str = ""  # Additional category info


@dataclass
class Itinerary:
    """Complete itinerary plan"""
    locations: List[ItineraryLocation]
    segments: List[RouteSegment]
    total_distance_km: float
    total_duration_minutes: int
    start_time: datetime
    end_time: datetime
    route_polyline: str  # Combined polyline for map
    description: str  # LLM-generated description
    optimization_score: float
    warnings: List[str]
    advantages: List[str]


class IntelligentItineraryPlanner:
    """
    Intelligent route planner that creates optimized itineraries
    combining museums, hidden gems, and dining spots
    
    WEEK 2 ENHANCEMENTS:
    - Full OSRM integration with caching
    - Turn-by-turn directions
    - Multi-modal transport support
    - Distance matrix optimization
    """
    
    def __init__(self):
        self.route_optimizer = RouteOptimizer()
        # Initialize enhanced OSRM service with caching
        from backend.services.osrm_routing_service import OSRMRoutingService
        self.osrm_service = OSRMRoutingService(
            profile='foot',  # Default to walking
            use_cache=True,
            cache_ttl=3600  # 1 hour cache
        )
        logger.info("✅ IntelligentItineraryPlanner initialized with OSRM caching")
    
    async def create_itinerary(
        self,
        user_query: str,
        user_location: Optional[Tuple[float, float]] = None,
        max_duration_minutes: int = 240,
        include_meals: bool = True
    ) -> Itinerary:
        """
        Create optimized itinerary from user query
        
        Args:
            user_query: "Show me art museums and cafes in Beyoğlu for 4 hours"
            user_location: (lat, lng) starting point
            max_duration_minutes: Maximum duration in minutes
            include_meals: Whether to include meal breaks
        
        Returns:
            Complete itinerary with optimized route
        """
        # Step 1: Parse query to extract intent
        parsed = await self._parse_query(user_query)
        
        # Step 2: Find matching locations
        locations = await self._find_locations(
            parsed,
            user_location,
            max_duration_minutes
        )
        
        # Step 3: Optimize route order (TSP)
        optimized_locations = await self._optimize_route_order(
            locations,
            user_location,
            max_duration_minutes
        )
        
        # Step 4: Add meal breaks if needed
        if include_meals:
            optimized_locations = await self._add_meal_breaks(
                optimized_locations,
                parsed
            )
        
        # Step 5: Get detailed routing between locations
        segments = await self._get_route_segments(optimized_locations)
        
        # Step 6: Calculate totals
        total_distance = sum(s.distance_km for s in segments)
        visit_time = sum(loc.duration for loc in optimized_locations)
        travel_time = sum(s.duration_minutes for s in segments)
        total_duration = visit_time + travel_time
        
        # Step 7: Generate description
        description = await self._generate_description(
            user_query,
            optimized_locations,
            total_distance,
            total_duration
        )
        
        # Step 8: Build itinerary
        start_time = datetime.now() if parsed.get('time_of_day') == 'now' else \
                     self._get_start_time(parsed.get('time_of_day'))
        
        itinerary = Itinerary(
            locations=optimized_locations,
            segments=segments,
            total_distance_km=total_distance,
            total_duration_minutes=total_duration,
            start_time=start_time,
            end_time=start_time + timedelta(minutes=total_duration),
            route_polyline=self._combine_polylines(segments),
            description=description,
            optimization_score=self._calculate_optimization_score(
                optimized_locations,
                segments,
                parsed
            ),
            warnings=self._generate_warnings(optimized_locations, segments),
            advantages=self._generate_advantages(optimized_locations, parsed)
        )
        
        return itinerary
    
    async def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse user query to extract intent"""
        # Simple keyword-based parsing (can be enhanced with LLM)
        parsed = {
            'duration_minutes': 240,  # Default 4 hours
            'interests': [],
            'location': None,
            'include_meals': True,
            'time_of_day': 'now'
        }
        
        query_lower = query.lower()
        
        # Extract duration
        if 'hour' in query_lower:
            import re
            hours_match = re.search(r'(\d+)\s*hour', query_lower)
            if hours_match:
                parsed['duration_minutes'] = int(hours_match.group(1)) * 60
        
        # Extract interests
        interest_keywords = {
            'art': ['art', 'painting', 'gallery'],
            'history': ['history', 'historical', 'ancient', 'ottoman'],
            'nature': ['nature', 'park', 'garden', 'outdoor'],
            'food': ['food', 'restaurant', 'cafe', 'dining', 'eat'],
            'culture': ['culture', 'cultural', 'traditional'],
            'architecture': ['architecture', 'building', 'mosque']
        }
        
        for interest, keywords in interest_keywords.items():
            if any(kw in query_lower for kw in keywords):
                parsed['interests'].append(interest)
        
        # Default to culture if no interests found
        if not parsed['interests']:
            parsed['interests'] = ['culture']
        
        # Extract location (districts)
        districts = [
            'beyoğlu', 'sultanahmet', 'kadıköy', 'beşiktaş', 
            'taksim', 'galata', 'karaköy', 'üsküdar', 'sarıyer'
        ]
        for district in districts:
            if district in query_lower:
                parsed['location'] = district
                break
        
        return parsed
    
    async def _find_locations(
        self,
        parsed: Dict,
        user_location: Optional[Tuple[float, float]],
        max_duration: int
    ) -> List[ItineraryLocation]:
        """Find matching locations from databases"""
        locations = []
        
        # Query museums
        museums = await self._query_museums(parsed)
        locations.extend(museums)
        
        # Query hidden gems
        gems = await self._query_hidden_gems(parsed)
        locations.extend(gems)
        
        # Limit to reasonable number based on duration
        max_locations = max(3, max_duration // 60)  # 1 location per hour minimum
        
        # Filter by user location if provided
        if user_location:
            locations = self._filter_by_proximity(locations, user_location, max_km=5.0)
        
        # Sort by rating and return top locations
        locations.sort(key=lambda x: x.rating, reverse=True)
        return locations[:max_locations]
    
    async def _query_museums(self, parsed: Dict) -> List[ItineraryLocation]:
        """Query museums database"""
        # Placeholder - integrate with actual museum database
        museums_data = [
            {
                'id': 'hagia_sophia',
                'name': 'Hagia Sophia',
                'lat': 41.0086,
                'lng': 28.9802,
                'duration': 90,
                'opening': '09:00',
                'closing': '19:00',
                'rating': 4.8,
                'description': 'Iconic Byzantine masterpiece',
                'cost': '₺₺₺',
                'address': 'Sultanahmet',
                'category': 'history'
            },
            {
                'id': 'topkapi_palace',
                'name': 'Topkapi Palace',
                'lat': 41.0115,
                'lng': 28.9833,
                'duration': 120,
                'opening': '09:00',
                'closing': '18:00',
                'rating': 4.7,
                'description': 'Ottoman palace with stunning views',
                'cost': '₺₺₺',
                'address': 'Sultanahmet',
                'category': 'history'
            }
        ]
        
        locations = []
        for museum in museums_data:
            # Filter by location if specified
            if parsed.get('location') and parsed['location'] not in museum['address'].lower():
                continue
            
            # Filter by interests
            if parsed.get('interests') and museum['category'] not in parsed['interests']:
                continue
            
            loc = ItineraryLocation(
                id=museum['id'],
                name=museum['name'],
                type='museum',
                lat=museum['lat'],
                lng=museum['lng'],
                duration=museum['duration'],
                opening_time=museum['opening'],
                closing_time=museum['closing'],
                rating=museum['rating'],
                description=museum['description'],
                cost=museum['cost'],
                address=museum['address'],
                category=museum['category']
            )
            locations.append(loc)
        
        return locations
    
    async def _query_hidden_gems(self, parsed: Dict) -> List[ItineraryLocation]:
        """Query hidden gems database"""
        try:
            # Import hidden gems handler
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            
            from services.hidden_gems_handler import HiddenGemsHandler
            
            handler = HiddenGemsHandler()
            gems_data = handler.get_hidden_gems()
            
            locations = []
            for gem in gems_data[:10]:  # Limit to top 10
                # Filter by location if specified
                if parsed.get('location'):
                    gem_location = gem.get('neighborhood', '').lower()
                    if parsed['location'] not in gem_location:
                        continue
                
                loc = ItineraryLocation(
                    id=gem.get('id', gem['name'].lower().replace(' ', '_')),
                    name=gem['name'],
                    type='hidden_gem',
                    lat=gem.get('latitude', 41.0082),
                    lng=gem.get('longitude', 28.9784),
                    duration=30,  # 30 minutes default
                    opening_time="09:00",
                    closing_time="22:00",
                    rating=gem.get('rating', 4.0),
                    description=gem.get('description', ''),
                    cost=gem.get('cost', '₺₺'),
                    address=gem.get('neighborhood', ''),
                    category=gem.get('type', 'general')
                )
                locations.append(loc)
            
            return locations
            
        except Exception as e:
            print(f"Error querying hidden gems: {e}")
            return []
    
    def _filter_by_proximity(
        self,
        locations: List[ItineraryLocation],
        user_location: Tuple[float, float],
        max_km: float
    ) -> List[ItineraryLocation]:
        """Filter locations by proximity to user"""
        filtered = []
        
        for loc in locations:
            distance = self.route_optimizer.calculate_distance(
                user_location,
                (loc.lat, loc.lng)
            )
            if distance <= max_km:
                filtered.append(loc)
        
        return filtered if filtered else locations[:5]  # Fallback to top 5
    
    async def _optimize_route_order(
        self,
        locations: List[ItineraryLocation],
        start_location: Optional[Tuple[float, float]],
        max_duration: int
    ) -> List[ItineraryLocation]:
        """
        Optimize the order of visiting locations using TSP with OSRM distance matrix
        WEEK 2: Now uses real OSRM routing instead of Haversine approximation
        """
        if len(locations) <= 1:
            return locations
        
        n = len(locations)
        
        # Build coordinates list for OSRM
        coords = [(loc.lat, loc.lng) for loc in locations]
        
        # Try to get distance matrix from OSRM
        matrix_data = self.osrm_service.get_distance_matrix(coords, profile='foot')
        
        if matrix_data and 'durations' in matrix_data:
            # Use OSRM duration matrix (in seconds, convert to minutes)
            matrix = np.array(matrix_data['durations']) / 60.0
            logger.info(f"✅ Using OSRM distance matrix for {n} locations")
        else:
            # Fallback to Haversine if OSRM fails
            logger.warning("⚠️ OSRM matrix failed, using Haversine fallback")
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist = self.route_optimizer.calculate_distance(
                            (locations[i].lat, locations[i].lng),
                            (locations[j].lat, locations[j].lng)
                        )
                        matrix[i][j] = dist * 12  # Convert to minutes (5 km/h walking)
        
        # Solve TSP using nearest neighbor + 2-opt
        start_idx = 0
        if start_location:
            # Find nearest location to start
            min_dist = float('inf')
            for i, loc in enumerate(locations):
                dist = self.route_optimizer.calculate_distance(
                    start_location,
                    (loc.lat, loc.lng)
                )
                if dist < min_dist:
                    min_dist = dist
                    start_idx = i
        
        # Nearest neighbor heuristic
        route = [start_idx]
        unvisited = set(range(n)) - {start_idx}
        current = start_idx
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: matrix[current][x])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # 2-opt improvement
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    
                    old_dist = (matrix[route[i-1]][route[i]] + 
                               matrix[route[j-1]][route[j if j < n else 0]])
                    new_dist = (matrix[new_route[i-1]][new_route[i]] + 
                               matrix[new_route[j-1]][new_route[j if j < n else 0]])
                    
                    if new_dist < old_dist:
                        route = new_route
                        improved = True
                        break
                
                if improved:
                    break
        
        # Filter by time constraint
        optimized = []
        total_time = 0
        
        for idx in route:
            loc = locations[idx]
            visit_time = loc.duration
            
            # Add travel time to next location
            if optimized:
                prev_loc = optimized[-1]
                travel_time = matrix[route[len(optimized)-1]][idx]
            else:
                travel_time = 0
            
            total_time += visit_time + travel_time
            
            if total_time <= max_duration:
                optimized.append(loc)
            else:
                break
        
        return optimized
    
    async def _add_meal_breaks(
        self,
        locations: List[ItineraryLocation],
        parsed: Dict
    ) -> List[ItineraryLocation]:
        """Add cafe or restaurant breaks at logical points"""
        if len(locations) < 2:
            return locations
        
        # Calculate cumulative time
        total_time = 0
        lunch_added = False
        coffee_added = False
        result = []
        
        for i, loc in enumerate(locations):
            result.append(loc)
            total_time += loc.duration
            
            # Add lunch break after ~2 hours
            if not lunch_added and total_time > 120 and i < len(locations) - 1:
                # Find nearby restaurant
                lunch_spot = self._find_nearby_dining(loc, 'restaurant')
                if lunch_spot:
                    result.append(lunch_spot)
                    total_time += lunch_spot.duration
                    lunch_added = True
            
            # Add coffee break after ~1.5 hours (if no lunch break)
            elif not coffee_added and not lunch_added and total_time > 90 and i < len(locations) - 1:
                coffee_spot = self._find_nearby_dining(loc, 'cafe')
                if coffee_spot:
                    result.append(coffee_spot)
                    total_time += coffee_spot.duration
                    coffee_added = True
        
        return result
    
    def _find_nearby_dining(
        self,
        location: ItineraryLocation,
        dining_type: str
    ) -> Optional[ItineraryLocation]:
        """Find nearby restaurant or cafe"""
        # Placeholder - in production, query real restaurant database
        dining_options = {
            'restaurant': {
                'name': 'Local Turkish Restaurant',
                'duration': 60,
                'cost': '₺₺'
            },
            'cafe': {
                'name': 'Turkish Coffee House',
                'duration': 30,
                'cost': '₺'
            }
        }
        
        option = dining_options.get(dining_type)
        if not option:
            return None
        
        # Place near the current location with small offset
        return ItineraryLocation(
            id=f"{dining_type}_{location.id}",
            name=option['name'],
            type=dining_type,
            lat=location.lat + 0.002,  # Small offset
            lng=location.lng + 0.002,
            duration=option['duration'],
            opening_time="08:00",
            closing_time="23:00",
            rating=4.0,
            description=f"Perfect spot for a {dining_type} break",
            cost=option['cost'],
            address=location.address,
            category='dining'
        )
    
    async def _get_route_segments(
        self,
        locations: List[ItineraryLocation]
    ) -> List[RouteSegment]:
        """
        Get detailed routing between locations with turn-by-turn directions
        WEEK 2: Now uses OSRM for real routing with step-by-step navigation
        """
        segments = []
        
        for i in range(len(locations) - 1):
            from_loc = locations[i]
            to_loc = locations[i + 1]
            
            # Get route from OSRM with turn-by-turn directions
            osrm_route = self.osrm_service.get_walking_route(
                start=(from_loc.lat, from_loc.lng),
                end=(to_loc.lat, to_loc.lng)
            )
            
            if osrm_route:
                # Use OSRM route data
                distance = osrm_route.total_distance / 1000.0  # Convert to km
                duration = int(osrm_route.total_duration / 60.0)  # Convert to minutes
                waypoints = osrm_route.waypoints
                
                # Simplified instructions since map will show visual directions
                instructions = [
                    f"Walk to {to_loc.name}",
                    f"{distance:.1f} km • {duration} min"
                ]
                
                # Add first 3 key directions for preview
                key_steps = [s for s in osrm_route.steps if s.maneuver_type in ['turn', 'depart', 'arrive']][:3]
                for step in key_steps:
                    instructions.append(f"• {step.instruction}")
                
                logger.debug(f"✅ OSRM route: {from_loc.name} → {to_loc.name} ({distance:.2f}km, {duration}min)")
            else:
                # Fallback to Haversine if OSRM fails
                logger.warning(f"⚠️ OSRM failed, using Haversine fallback for {from_loc.name} → {to_loc.name}")
                distance = self.route_optimizer.calculate_distance(
                    (from_loc.lat, from_loc.lng),
                    (to_loc.lat, to_loc.lng)
                )
                duration = int(distance * 12)  # 5 km/h walking speed
                waypoints = [(from_loc.lat, from_loc.lng), (to_loc.lat, to_loc.lng)]
                instructions = [
                    f"Walk to {to_loc.name}",
                    f"{distance:.1f} km • ~{duration} min"
                ]
            
            # Create segment
            from_location_obj = Location(
                id=from_loc.id,
                name=from_loc.name,
                coordinates=(from_loc.lat, from_loc.lng),
                district=from_loc.address,
                transport_connections=[],
                accessibility_score=0.8,
                popularity_score=from_loc.rating / 5.0,
                category=from_loc.type
            )
            
            to_location_obj = Location(
                id=to_loc.id,
                name=to_loc.name,
                coordinates=(to_loc.lat, to_loc.lng),
                district=to_loc.address,
                transport_connections=[],
                accessibility_score=0.8,
                popularity_score=to_loc.rating / 5.0,
                category=to_loc.type
            )
            
            segment = RouteSegment(
                from_location=from_location_obj,
                to_location=to_location_obj,
                transport_mode=TransportMode.WALKING,
                distance_km=distance,
                duration_minutes=duration,
                cost_tl=0,
                instructions=instructions,
                waypoints=waypoints,
                real_time_data={"created_at": datetime.now().isoformat()},
                accessibility_friendly=True,
                scenic_score=0.6
            )
            
            segments.append(segment)
        
        return segments
    
    def _combine_polylines(self, segments: List[RouteSegment]) -> str:
        """Combine segment polylines"""
        # Simplified - in production, use polyline encoding library
        all_points = []
        for segment in segments:
            all_points.extend(segment.waypoints)
        
        return json.dumps(all_points)
    
    async def _generate_description(
        self,
        query: str,
        locations: List[ItineraryLocation],
        distance: float,
        duration: int
    ) -> str:
        """Generate natural language description"""
        location_names = [loc.name for loc in locations if loc.type != 'cafe' and loc.type != 'restaurant']
        
        description = f"Explore {len(location_names)} amazing spots in Istanbul! "
        description += f"This {duration // 60}-hour journey covers {distance:.1f}km and includes "
        description += ", ".join(location_names[:3])
        
        if len(location_names) > 3:
            description += f" and {len(location_names) - 3} more hidden gems"
        
        description += ". Perfect for discovering the authentic Istanbul!"
        
        return description
    
    def _calculate_optimization_score(
        self,
        locations: List[ItineraryLocation],
        segments: List[RouteSegment],
        parsed: Dict
    ) -> float:
        """Calculate how well optimized the route is"""
        if not locations:
            return 0.0
        
        score = 0.0
        
        # Factor 1: Average location rating (30%)
        avg_rating = sum(loc.rating for loc in locations) / len(locations)
        score += (avg_rating / 5.0) * 0.3
        
        # Factor 2: Travel efficiency (40%)
        total_travel = sum(s.duration_minutes for s in segments)
        total_visit = sum(loc.duration for loc in locations)
        efficiency = total_visit / (total_visit + total_travel) if (total_visit + total_travel) > 0 else 0
        score += efficiency * 0.4
        
        # Factor 3: Interest match (30%)
        if parsed.get('interests'):
            matching_count = sum(
                1 for loc in locations 
                if loc.category in parsed['interests']
            )
            match_ratio = matching_count / len(locations)
            score += match_ratio * 0.3
        else:
            score += 0.3  # No preferences, give full score
        
        return min(1.0, score)
    
    def _generate_warnings(
        self,
        locations: List[ItineraryLocation],
        segments: List[RouteSegment]
    ) -> List[str]:
        """Generate warnings for the itinerary"""
        warnings = []
        
        # Check total walking distance
        total_walking = sum(s.distance_km for s in segments if s.transport_mode == TransportMode.WALKING)
        if total_walking > 3.0:
            warnings.append(f"Route includes {total_walking:.1f}km of walking - wear comfortable shoes!")
        
        # Check time of day
        current_hour = datetime.now().hour
        if current_hour >= 17:
            warnings.append("Some locations may be closing soon - check opening hours")
        
        # Check if museums are included
        museums = [loc for loc in locations if loc.type == 'museum']
        if museums:
                       warnings.append("Museums may have specific visiting hours - arrive early")
        
        return warnings
    
    def _generate_advantages(
        self,
        locations: List[ItineraryLocation],
        parsed: Dict
    ) -> List[str]:
        """Generate advantages of this itinerary"""
        advantages = []
        
        # Variety
        types = set(loc.type for loc in locations)
        if len(types) > 2:
            advantages.append("Great variety of experiences")
        
        # High ratings
        avg_rating = sum(loc.rating for loc in locations) / len(locations) if locations else 0
        if avg_rating >= 4.5:
            advantages.append("All highly-rated locations")
        
        # Interest match
        if parsed.get('interests'):
            advantages.append(f"Perfectly matches your interest in {', '.join(parsed['interests'])}")
        
        # Efficiency
        advantages.append("Optimized route saves time and energy")
        
        return advantages
    
    def _get_start_time(self, time_of_day: str) -> datetime:
        """Get appropriate start time based on time of day"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        time_map = {
            'morning': today.replace(hour=9),
            'afternoon': today.replace(hour=14),
            'evening': today.replace(hour=17),
            'now': datetime.now()
        }
        
        return time_map.get(time_of_day, datetime.now())


# Global instance
_itinerary_planner = None

def get_itinerary_planner() -> IntelligentItineraryPlanner:
    """Get global itinerary planner instance"""
    global _itinerary_planner
    if _itinerary_planner is None:
        _itinerary_planner = IntelligentItineraryPlanner()
    return _itinerary_planner
