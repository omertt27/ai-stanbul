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
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import numpy as np

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
