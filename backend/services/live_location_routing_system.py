"""
Live Location-Based Routing and POI Recommendation System
Algorithmic approach for real-time location-based recommendations in Istanbul

Features:
- Real-time GPS coordinate handling (privacy-safe)
- Algorithmic route optimization (Dijkstra, A*, TSP)
- Dynamic POI recommendations (museums, restaurants, landmarks)
- Distance and time estimates between districts
- Smart filtering (cuisine, category, hours, dietary, accessibility)
- Dynamic updates as user moves
- Offline/minimal AI operation support
- Integration with mapping services

No GPT/LLM dependencies - Pure algorithmic approach
"""

from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time, timedelta
import math
import heapq
import json
import logging
from collections import defaultdict, deque
import asyncio
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class POICategory(Enum):
    RESTAURANT = "restaurant"
    MUSEUM = "museum"
    LANDMARK = "landmark"
    SHOPPING = "shopping"
    ENTERTAINMENT = "entertainment"
    TRANSPORT = "transport"
    HOTEL = "hotel"
    PARK = "park"
    RELIGIOUS = "religious"
    VIEWPOINT = "viewpoint"

class RoutingAlgorithm(Enum):
    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"
    TSP_NEAREST_NEIGHBOR = "tsp_nearest"
    TSP_GREEDY = "tsp_greedy"

class FilterCriteria(Enum):
    CUISINE_TYPE = "cuisine"
    PRICE_RANGE = "price_range"
    OPEN_HOURS = "open_hours"
    ACCESSIBILITY = "accessibility"
    DIETARY = "dietary"
    RATING = "rating"
    DISTANCE = "distance"

@dataclass
class Coordinates:
    """GPS coordinates with privacy-safe handling"""
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def privacy_hash(self) -> str:
        """Generate privacy-safe hash of coordinates"""
        coord_str = f"{self.latitude:.4f},{self.longitude:.4f}"
        return hashlib.sha256(coord_str.encode()).hexdigest()[:12]

@dataclass
class POI:
    """Point of Interest with comprehensive data"""
    id: str
    name: str
    category: POICategory
    coordinates: Coordinates
    address: str
    rating: Optional[float] = None
    price_range: Optional[str] = None
    open_hours: Dict[str, Tuple[time, time]] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    cuisine_type: Optional[str] = None
    accessibility_features: List[str] = field(default_factory=list)
    dietary_options: List[str] = field(default_factory=list)
    estimated_visit_duration: Optional[int] = None  # minutes
    description: str = ""
    
    @property
    def is_open_now(self) -> bool:
        """Check if POI is currently open"""
        now = datetime.now()
        today = now.strftime("%A").lower()
        current_time = now.time()
        
        if today in self.open_hours:
            open_time, close_time = self.open_hours[today]
            return open_time <= current_time <= close_time
        return True  # Assume open if hours not specified

@dataclass
class RouteSegment:
    """Individual segment of a route"""
    from_poi: str
    to_poi: str
    distance_km: float
    estimated_time_minutes: int
    transport_mode: str
    instructions: List[str] = field(default_factory=list)

@dataclass
class OptimizedRoute:
    """Complete optimized route with segments"""
    segments: List[RouteSegment]
    total_distance_km: float
    total_time_minutes: int
    total_cost_estimate: Optional[str] = None
    route_type: str = "mixed"  # walking, public_transport, mixed
    confidence_score: float = 0.0
    
    @property
    def waypoints(self) -> List[str]:
        """Get ordered list of POI IDs in route"""
        waypoints = []
        for segment in self.segments:
            if not waypoints:
                waypoints.append(segment.from_poi)
            waypoints.append(segment.to_poi)
        return waypoints

class IstanbulLocationData:
    """Comprehensive Istanbul location and routing data"""
    
    def __init__(self):
        self.pois: Dict[str, POI] = {}
        self.districts: Dict[str, Coordinates] = {}
        self.distance_matrix: Dict[Tuple[str, str], float] = {}
        self.transport_connections: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        self._load_istanbul_data()
    
    def _load_istanbul_data(self):
        """Load comprehensive Istanbul location data"""
        
        # Major Istanbul districts with coordinates
        self.districts = {
            "sultanahmet": Coordinates(41.0082, 28.9784),
            "beyoglu": Coordinates(41.0369, 28.9744),
            "galata": Coordinates(41.0256, 28.9744),
            "karakoy": Coordinates(41.0255, 28.9741),
            "eminonu": Coordinates(41.0176, 28.9706),
            "kadikoy": Coordinates(40.9833, 29.0333),
            "besiktas": Coordinates(41.0422, 29.0061),
            "ortakoy": Coordinates(41.0553, 29.0264),
            "uskudar": Coordinates(41.0214, 29.0144),
            "fatih": Coordinates(41.0209, 28.9497),
            "bakirkoy": Coordinates(40.9833, 28.8667),
            "sisli": Coordinates(41.0608, 28.9875),
            "taksim": Coordinates(41.0369, 28.9850),
            "levent": Coordinates(41.0775, 29.0061),
            "maslak": Coordinates(41.1064, 29.0264)
        }
        
        # Sample POIs with comprehensive data
        sample_pois = [
            # Museums
            POI(
                id="hagia_sophia",
                name="Hagia Sophia",
                category=POICategory.MUSEUM,
                coordinates=Coordinates(41.0086, 28.9802),
                address="Sultan Ahmet, Ayasofya Meydanı No:1, 34122 Fatih/İstanbul",
                rating=4.7,
                open_hours={
                    "monday": (time(9, 0), time(19, 0)),
                    "tuesday": (time(9, 0), time(19, 0)),
                    "wednesday": (time(9, 0), time(19, 0)),
                    "thursday": (time(9, 0), time(19, 0)),
                    "friday": (time(9, 0), time(19, 0)),
                    "saturday": (time(9, 0), time(19, 0)),
                    "sunday": (time(9, 0), time(19, 0))
                },
                estimated_visit_duration=90,
                accessibility_features=["wheelchair_accessible", "audio_guide"],
                description="Historic Byzantine church and Ottoman mosque, now a museum"
            ),
            
            POI(
                id="topkapi_palace",
                name="Topkapi Palace",
                category=POICategory.MUSEUM,
                coordinates=Coordinates(41.0115, 28.9833),
                address="Cankurtaran, 34122 Fatih/İstanbul",
                rating=4.6,
                open_hours={
                    "monday": (time(9, 0), time(18, 0)),
                    "wednesday": (time(9, 0), time(18, 0)),
                    "thursday": (time(9, 0), time(18, 0)),
                    "friday": (time(9, 0), time(18, 0)),
                    "saturday": (time(9, 0), time(18, 0)),
                    "sunday": (time(9, 0), time(18, 0))
                },
                estimated_visit_duration=120,
                accessibility_features=["partial_wheelchair_access"],
                description="Former Ottoman imperial palace with stunning Bosphorus views"
            ),
            
            # Restaurants
            POI(
                id="pandeli_restaurant",
                name="Pandeli Restaurant",
                category=POICategory.RESTAURANT,
                coordinates=Coordinates(41.0176, 28.9706),
                address="Eminönü, Mısır Çarşısı No:1, 34116 Fatih/İstanbul",
                rating=4.4,
                price_range="luxury",
                cuisine_type="ottoman",
                open_hours={
                    "monday": (time(12, 0), time(15, 0)),
                    "tuesday": (time(12, 0), time(15, 0)),
                    "wednesday": (time(12, 0), time(15, 0)),
                    "thursday": (time(12, 0), time(15, 0)),
                    "friday": (time(12, 0), time(15, 0)),
                    "saturday": (time(12, 0), time(15, 0))
                },
                estimated_visit_duration=90,
                dietary_options=["vegetarian_options"],
                accessibility_features=["ground_floor_access"],
                description="Historic Ottoman cuisine in the Spice Bazaar"
            ),
            
            POI(
                id="hamdi_restaurant",
                name="Hamdi Restaurant",
                category=POICategory.RESTAURANT,
                coordinates=Coordinates(41.0176, 28.9731),
                address="Kalçın Sk. No:17, 34116 Fatih/İstanbul",
                rating=4.5,
                price_range="mid-range",
                cuisine_type="turkish",
                open_hours={
                    "monday": (time(11, 30), time(23, 59)),
                    "tuesday": (time(11, 30), time(23, 59)),
                    "wednesday": (time(11, 30), time(23, 59)),
                    "thursday": (time(11, 30), time(23, 59)),
                    "friday": (time(11, 30), time(23, 59)),
                    "saturday": (time(11, 30), time(23, 59)),
                    "sunday": (time(11, 30), time(23, 59))
                },
                estimated_visit_duration=75,
                dietary_options=["halal", "vegetarian_options"],
                accessibility_features=["wheelchair_accessible"],
                description="Famous for kebabs with Golden Horn views"
            ),
            
            # Landmarks
            POI(
                id="galata_tower",
                name="Galata Tower",
                category=POICategory.LANDMARK,
                coordinates=Coordinates(41.0256, 28.9744),
                address="Bereketzade, Galata Kulesi Sk., 34421 Beyoğlu/İstanbul",
                rating=4.3,
                price_range="mid-range",
                open_hours={
                    "monday": (time(8, 30), time(22, 0)),
                    "tuesday": (time(8, 30), time(22, 0)),
                    "wednesday": (time(8, 30), time(22, 0)),
                    "thursday": (time(8, 30), time(22, 0)),
                    "friday": (time(8, 30), time(22, 0)),
                    "saturday": (time(8, 30), time(22, 0)),
                    "sunday": (time(8, 30), time(22, 0))
                },
                estimated_visit_duration=45,
                accessibility_features=["elevator_available"],
                description="Medieval tower with panoramic city views"
            ),
            
            POI(
                id="blue_mosque",
                name="Blue Mosque (Sultan Ahmed Mosque)",
                category=POICategory.RELIGIOUS,
                coordinates=Coordinates(41.0054, 28.9768),
                address="Sultan Ahmet, Atmeydanı Cd. No:7, 34122 Fatih/İstanbul",
                rating=4.6,
                open_hours={
                    "monday": (time(8, 30), time(11, 30)),
                    "tuesday": (time(8, 30), time(11, 30)),
                    "wednesday": (time(8, 30), time(11, 30)),
                    "thursday": (time(8, 30), time(11, 30)),
                    "friday": (time(14, 30), time(15, 45)),
                    "saturday": (time(8, 30), time(11, 30)),
                    "sunday": (time(8, 30), time(11, 30))
                },
                estimated_visit_duration=30,
                accessibility_features=["wheelchair_accessible"],
                features=["dress_code_required", "prayer_times_restricted"],
                description="Historic Ottoman mosque famous for blue tiles"
            )
        ]
        
        # Add POIs to the system
        for poi in sample_pois:
            self.pois[poi.id] = poi
        
        # Calculate distance matrix between all POIs and districts
        self._calculate_distance_matrix()
        self._setup_transport_connections()
    
    def _calculate_distance_matrix(self):
        """Calculate distances between all POIs and districts"""
        all_locations = {**{poi_id: poi.coordinates for poi_id, poi in self.pois.items()}, 
                        **self.districts}
        
        for loc1_id, coord1 in all_locations.items():
            for loc2_id, coord2 in all_locations.items():
                if loc1_id != loc2_id:
                    distance = self._haversine_distance(coord1, coord2)
                    self.distance_matrix[(loc1_id, loc2_id)] = distance
    
    def _haversine_distance(self, coord1: Coordinates, coord2: Coordinates) -> float:
        """Calculate distance between two coordinates using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = math.radians(coord1.latitude), math.radians(coord1.longitude)
        lat2, lon2 = math.radians(coord2.latitude), math.radians(coord2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _setup_transport_connections(self):
        """Setup transportation connections between locations"""
        # This would typically be loaded from a real transport API
        # For now, we'll create basic connections based on distance
        
        all_location_ids = list(self.pois.keys()) + list(self.districts.keys())
        for loc1_id in all_location_ids:
            for loc2_id in all_location_ids:
                if loc1_id != loc2_id:
                    distance = self.distance_matrix.get((loc1_id, loc2_id), 0)
                    if distance > 0:
                        # Estimate walking time (5 km/h average)
                        walking_time = int(distance * 12)  # minutes
                        # Estimate public transport time (15 km/h average)
                        transport_time = int(distance * 4)  # minutes
                        
                        self.transport_connections[loc1_id].append(
                            (loc2_id, distance, walking_time)
                        )

class LiveLocationRoutingSystem:
    """Main system for live location-based routing and POI recommendations"""
    
    def __init__(self):
        self.location_data = IstanbulLocationData()
        self.user_sessions: Dict[str, Dict] = {}
        self.route_cache: Dict[str, OptimizedRoute] = {}
        logger.info("Live Location Routing System initialized")
    
    def create_user_session(self, user_id: str, initial_location: Coordinates) -> str:
        """Create a new user session with privacy-safe location tracking"""
        session_id = f"{user_id}_{initial_location.privacy_hash}_{datetime.now().timestamp()}"
        
        self.user_sessions[session_id] = {
            "user_id": user_id,
            "current_location": initial_location,
            "visited_pois": [],
            "preferences": {},
            "active_route": None,
            "last_updated": datetime.now()
        }
        
        logger.info(f"Created user session: {session_id}")
        return session_id
    
    def update_user_location(self, session_id: str, new_location: Coordinates) -> bool:
        """Update user's current location (privacy-safe)"""
        if session_id not in self.user_sessions:
            return False
        
        old_location = self.user_sessions[session_id]["current_location"]
        distance_moved = self.location_data._haversine_distance(old_location, new_location)
        
        # Only update if user has moved significantly (>50m)
        if distance_moved > 0.05:
            self.user_sessions[session_id]["current_location"] = new_location
            self.user_sessions[session_id]["last_updated"] = datetime.now()
            
            # Check if route needs updating
            if self.user_sessions[session_id]["active_route"]:
                self._update_active_route(session_id)
            
            logger.info(f"Updated location for session {session_id}, moved {distance_moved:.3f}km")
            return True
        
        return False
    
    def find_nearby_pois(self, 
                         location: Coordinates, 
                         radius_km: float = 2.0,
                         categories: Optional[List[POICategory]] = None,
                         filters: Optional[Dict[FilterCriteria, Any]] = None,
                         limit: int = 20) -> List[Tuple[POI, float]]:
        """Find POIs near a location with smart filtering"""
        
        nearby_pois = []
        
        for poi_id, poi in self.location_data.pois.items():
            distance = self.location_data._haversine_distance(location, poi.coordinates)
            
            if distance <= radius_km:
                # Apply category filter
                if categories and poi.category not in categories:
                    continue
                
                # Apply smart filters
                if filters and not self._apply_poi_filters(poi, filters):
                    continue
                
                nearby_pois.append((poi, distance))
        
        # Sort by distance
        nearby_pois.sort(key=lambda x: x[1])
        
        return nearby_pois[:limit]
    
    def _apply_poi_filters(self, poi: POI, filters: Dict[FilterCriteria, Any]) -> bool:
        """Apply smart filters to POI"""
        
        for filter_type, filter_value in filters.items():
            if filter_type == FilterCriteria.OPEN_HOURS:
                if filter_value and not poi.is_open_now:
                    return False
            
            elif filter_type == FilterCriteria.CUISINE_TYPE:
                if isinstance(filter_value, list) and poi.cuisine_type not in filter_value:
                    return False
                elif isinstance(filter_value, str) and poi.cuisine_type != filter_value:
                    return False
            
            elif filter_type == FilterCriteria.PRICE_RANGE:
                if isinstance(filter_value, list) and poi.price_range not in filter_value:
                    return False
                elif isinstance(filter_value, str) and poi.price_range != filter_value:
                    return False
            
            elif filter_type == FilterCriteria.RATING:
                min_rating = filter_value if isinstance(filter_value, (int, float)) else filter_value.get("min", 0)
                if poi.rating and poi.rating < min_rating:
                    return False
            
            elif filter_type == FilterCriteria.ACCESSIBILITY:
                required_features = filter_value if isinstance(filter_value, list) else [filter_value]
                if not all(feature in poi.accessibility_features for feature in required_features):
                    return False
            
            elif filter_type == FilterCriteria.DIETARY:
                required_options = filter_value if isinstance(filter_value, list) else [filter_value]
                if not all(option in poi.dietary_options for option in required_options):
                    return False
        
        return True
    
    def calculate_optimal_route(self, 
                              start_location: Coordinates,
                              target_pois: List[str],
                              algorithm: RoutingAlgorithm = RoutingAlgorithm.TSP_NEAREST_NEIGHBOR,
                              transport_mode: str = "mixed") -> OptimizedRoute:
        """Calculate optimal route through multiple POIs"""
        
        if not target_pois:
            return OptimizedRoute([], 0, 0)
        
        # Generate cache key
        cache_key = self._generate_route_cache_key(start_location, target_pois, algorithm)
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        # Choose routing algorithm
        if algorithm == RoutingAlgorithm.DIJKSTRA:
            route = self._dijkstra_route(start_location, target_pois, transport_mode)
        elif algorithm == RoutingAlgorithm.A_STAR:
            route = self._a_star_route(start_location, target_pois, transport_mode)
        elif algorithm == RoutingAlgorithm.TSP_NEAREST_NEIGHBOR:
            route = self._tsp_nearest_neighbor_route(start_location, target_pois, transport_mode)
        elif algorithm == RoutingAlgorithm.TSP_GREEDY:
            route = self._tsp_greedy_route(start_location, target_pois, transport_mode)
        else:
            route = self._tsp_nearest_neighbor_route(start_location, target_pois, transport_mode)
        
        # Cache the result
        self.route_cache[cache_key] = route
        
        return route
    
    def _generate_route_cache_key(self, start_location: Coordinates, target_pois: List[str], algorithm: RoutingAlgorithm) -> str:
        """Generate cache key for route"""
        location_str = f"{start_location.latitude:.4f},{start_location.longitude:.4f}"
        pois_str = ",".join(sorted(target_pois))
        return hashlib.md5(f"{location_str}|{pois_str}|{algorithm.value}".encode()).hexdigest()
    
    def _tsp_nearest_neighbor_route(self, start_location: Coordinates, target_pois: List[str], transport_mode: str) -> OptimizedRoute:
        """Solve TSP using nearest neighbor heuristic"""
        if not target_pois:
            return OptimizedRoute([], 0, 0)
        
        # Start with current location
        current_pos = "start"
        unvisited = set(target_pois)
        route_order = []
        segments = []
        total_distance = 0
        total_time = 0
        
        # Create temporary location for start
        temp_locations = {"start": start_location}
        for poi_id in target_pois:
            temp_locations[poi_id] = self.location_data.pois[poi_id].coordinates
        
        while unvisited:
            nearest_poi = None
            min_distance = float('inf')
            
            # Find nearest unvisited POI
            for poi_id in unvisited:
                if current_pos in temp_locations and poi_id in temp_locations:
                    distance = self.location_data._haversine_distance(
                        temp_locations[current_pos], 
                        temp_locations[poi_id]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        nearest_poi = poi_id
            
            if nearest_poi:
                # Create route segment
                walking_time = int(min_distance * 12)  # 5 km/h walking speed
                transport_time = int(min_distance * 4)   # 15 km/h public transport
                
                estimated_time = walking_time if transport_mode == "walking" else min(walking_time, transport_time)
                
                segment = RouteSegment(
                    from_poi=current_pos,
                    to_poi=nearest_poi,
                    distance_km=min_distance,
                    estimated_time_minutes=estimated_time,
                    transport_mode=transport_mode,
                    instructions=[f"Travel from {current_pos} to {nearest_poi}"]
                )
                
                segments.append(segment)
                route_order.append(nearest_poi)
                unvisited.remove(nearest_poi)
                current_pos = nearest_poi
                total_distance += min_distance
                total_time += estimated_time
        
        confidence_score = 0.8  # Reasonable confidence for nearest neighbor
        
        return OptimizedRoute(
            segments=segments,
            total_distance_km=total_distance,
            total_time_minutes=total_time,
            route_type=transport_mode,
            confidence_score=confidence_score
        )
    
    def _dijkstra_route(self, start_location: Coordinates, target_pois: List[str], transport_mode: str) -> OptimizedRoute:
        """Use Dijkstra's algorithm for shortest path"""
        # For simplicity, we'll use nearest neighbor for now
        # In a full implementation, this would use proper graph algorithms
        return self._tsp_nearest_neighbor_route(start_location, target_pois, transport_mode)
    
    def _a_star_route(self, start_location: Coordinates, target_pois: List[str], transport_mode: str) -> OptimizedRoute:
        """Use A* algorithm for shortest path with heuristic"""
        # For simplicity, we'll use nearest neighbor for now
        # In a full implementation, this would use A* with proper heuristics
        return self._tsp_nearest_neighbor_route(start_location, target_pois, transport_mode)
    
    def _tsp_greedy_route(self, start_location: Coordinates, target_pois: List[str], transport_mode: str) -> OptimizedRoute:
        """Greedy TSP approach"""
        # For now, same as nearest neighbor
        return self._tsp_nearest_neighbor_route(start_location, target_pois, transport_mode)
    
    def get_district_estimates(self, from_location: Coordinates) -> Dict[str, Dict[str, Any]]:
        """Get distance and time estimates to all districts"""
        estimates = {}
        
        for district_name, district_coord in self.location_data.districts.items():
            distance = self.location_data._haversine_distance(from_location, district_coord)
            walking_time = int(distance * 12)  # 5 km/h
            transport_time = int(distance * 4)  # 15 km/h
            
            estimates[district_name] = {
                "distance_km": round(distance, 2),
                "walking_time_minutes": walking_time,
                "transport_time_minutes": transport_time,
                "coordinates": {
                    "latitude": district_coord.latitude,
                    "longitude": district_coord.longitude
                }
            }
        
        return estimates
    
    def get_smart_recommendations(self, 
                                session_id: str,
                                categories: Optional[List[POICategory]] = None,
                                filters: Optional[Dict[FilterCriteria, Any]] = None,
                                limit: int = 10) -> Dict[str, Any]:
        """Get smart POI recommendations based on user context"""
        
        if session_id not in self.user_sessions:
            return {"error": "Invalid session"}
        
        session = self.user_sessions[session_id]
        current_location = session["current_location"]
        visited_pois = session.get("visited_pois", [])
        
        # Find nearby POIs
        nearby_pois = self.find_nearby_pois(
            location=current_location,
            radius_km=3.0,
            categories=categories,
            filters=filters,
            limit=limit * 2  # Get more to filter out visited ones
        )
        
        # Filter out already visited POIs
        unvisited_pois = [(poi, distance) for poi, distance in nearby_pois if poi.id not in visited_pois]
        
        # Take top recommendations
        recommendations = unvisited_pois[:limit]
        
        # Get district estimates
        district_estimates = self.get_district_estimates(current_location)
        
        return {
            "session_id": session_id,
            "current_location": {
                "latitude": current_location.latitude,
                "longitude": current_location.longitude,
                "privacy_hash": current_location.privacy_hash
            },
            "recommendations": [
                {
                    "poi": {
                        "id": poi.id,
                        "name": poi.name,
                        "category": poi.category.value,
                        "rating": poi.rating,
                        "price_range": poi.price_range,
                        "is_open": poi.is_open_now,
                        "estimated_visit_duration": poi.estimated_visit_duration,
                        "description": poi.description,
                        "coordinates": {
                            "latitude": poi.coordinates.latitude,
                            "longitude": poi.coordinates.longitude
                        }
                    },
                    "distance_km": round(distance, 2),
                    "walking_time_minutes": int(distance * 12),
                    "transport_time_minutes": int(distance * 4)
                }
                for poi, distance in recommendations
            ],
            "district_estimates": district_estimates,
            "filter_summary": {
                "categories": [cat.value for cat in categories] if categories else "all",
                "filters_applied": len(filters) if filters else 0,
                "total_pois_in_area": len(nearby_pois),
                "recommendations_count": len(recommendations)
            }
        }
    
    def plan_dynamic_route(self, 
                          session_id: str, 
                          target_poi_ids: List[str],
                          algorithm: RoutingAlgorithm = RoutingAlgorithm.TSP_NEAREST_NEIGHBOR,
                          transport_mode: str = "mixed") -> Dict[str, Any]:
        """Plan a dynamic route that updates as user moves"""
        
        if session_id not in self.user_sessions:
            return {"error": "Invalid session"}
        
        session = self.user_sessions[session_id]
        current_location = session["current_location"]
        
        # Calculate optimal route
        optimal_route = self.calculate_optimal_route(
            start_location=current_location,
            target_pois=target_poi_ids,
            algorithm=algorithm,
            transport_mode=transport_mode
        )
        
        # Store route in session
        session["active_route"] = optimal_route
        
        # Prepare response
        return {
            "session_id": session_id,
            "route": {
                "total_distance_km": round(optimal_route.total_distance_km, 2),
                "total_time_minutes": optimal_route.total_time_minutes,
                "route_type": optimal_route.route_type,
                "confidence_score": optimal_route.confidence_score,
                "waypoints": optimal_route.waypoints,
                "segments": [
                    {
                        "from": segment.from_poi,
                        "to": segment.to_poi,
                        "distance_km": round(segment.distance_km, 2),
                        "time_minutes": segment.estimated_time_minutes,
                        "transport_mode": segment.transport_mode,
                        "instructions": segment.instructions
                    }
                    for segment in optimal_route.segments
                ]
            },
            "poi_details": [
                {
                    "id": poi_id,
                    "name": self.location_data.pois[poi_id].name,
                    "category": self.location_data.pois[poi_id].category.value,
                    "estimated_visit_duration": self.location_data.pois[poi_id].estimated_visit_duration,
                    "is_open": self.location_data.pois[poi_id].is_open_now
                }
                for poi_id in target_poi_ids if poi_id in self.location_data.pois
            ],
            "algorithm_used": algorithm.value,
            "created_at": datetime.now().isoformat()
        }
    
    def _update_active_route(self, session_id: str):
        """Update active route when user location changes"""
        session = self.user_sessions[session_id]
        active_route = session.get("active_route")
        
        if not active_route:
            return
        
        current_location = session["current_location"]
        
        # Check if user has reached any waypoints
        for segment in active_route.segments:
            if segment.to_poi in self.location_data.pois:
                poi_location = self.location_data.pois[segment.to_poi].coordinates
                distance_to_poi = self.location_data._haversine_distance(current_location, poi_location)
                
                # If within 100m of POI, mark as visited
                if distance_to_poi < 0.1:
                    if segment.to_poi not in session["visited_pois"]:
                        session["visited_pois"].append(segment.to_poi)
                        logger.info(f"User reached POI: {segment.to_poi}")
    
    def get_offline_recommendations(self, location: Coordinates, radius_km: float = 1.0) -> Dict[str, Any]:
        """Get offline recommendations for a location"""
        
        nearby_pois = self.find_nearby_pois(
            location=location,
            radius_km=radius_km,
            limit=10
        )
        
        # Simple categorized recommendations
        recommendations_by_category = defaultdict(list)
        
        for poi, distance in nearby_pois:
            recommendations_by_category[poi.category.value].append({
                "name": poi.name,
                "distance_km": round(distance, 2),
                "walking_minutes": int(distance * 12),
                "rating": poi.rating,
                "is_open": poi.is_open_now
            })
        
        return {
            "pois": filtered_pois,
            "districts": nearby_districts,
            "cached_routes": [],
            "transport_info": transport_info,
            "offline_timestamp": datetime.now().isoformat()
        }
    
    # New methods for FastAPI compatibility
    def plan_multi_stop_route(self, start_location: Dict[str, float], stops: List[Dict[str, Any]], preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plan a multi-stop route with optimization"""
        try:
            # Convert dict to Coordinates
            start_coords = Coordinates(start_location['lat'], start_location['lng'])
            
            # Extract POI names from stops
            target_pois = []
            for stop in stops:
                if 'name' in stop:
                    # Convert name to POI ID format
                    poi_id = stop['name'].lower().replace(' ', '_').replace('ğ', 'g').replace('ü', 'u').replace('ş', 's').replace('ı', 'i').replace('ö', 'o').replace('ç', 'c')
                    target_pois.append(poi_id)
            
            # Get preferences
            prefs = preferences or {}
            algorithm = RoutingAlgorithm.TSP_NEAREST_NEIGHBOR
            if prefs.get('optimize_for') == 'distance':
                algorithm = RoutingAlgorithm.TSP_GREEDY
            
            transport_mode = prefs.get('transport', 'walking')
            
            # Calculate optimal route
            route = self.calculate_optimal_route(
                start_location=start_coords,
                target_pois=target_pois,
                algorithm=algorithm,
                transport_mode=transport_mode
            )
            
            # Convert to dict format
            waypoints = []
            for poi_id in route.waypoints:
                if poi_id in self.location_data.pois:
                    poi = self.location_data.pois[poi_id]
                    waypoints.append({
                        'name': poi.name,
                        'location': {'lat': poi.coordinates.latitude, 'lng': poi.coordinates.longitude},
                        'type': poi.category.value,
                        'estimated_time': route.estimated_time_minutes
                    })
            
            return {
                'route_id': route.route_id,
                'waypoints': waypoints,
                'total_distance_km': route.total_distance_km,
                'estimated_time_minutes': route.estimated_time_minutes,
                'transport_mode': transport_mode,
                'algorithm': algorithm.value
            }
            
        except Exception as e:
            logger.error(f"Error planning multi-stop route: {e}")
            return {
                'route_id': None,
                'waypoints': [],
                'total_distance_km': 0,
                'estimated_time_minutes': 0,
                'error': str(e)
            }
    
    def get_pois_by_category(self, category: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get POIs by category with optional filters"""
        try:
            # Find POIs matching the category
            matching_pois = []
            for poi_id, poi in self.location_data.pois.items():
                # Check category match
                if poi.category.value == category:
                    # Apply filters
                    if filters:
                        if 'cuisine' in filters and poi.cuisine_type:
                            if filters['cuisine'].lower() not in poi.cuisine_type.lower():
                                continue
                        if 'rating' in filters and poi.rating:
                            if poi.rating < filters['rating']:
                                continue
                    
                    matching_pois.append({
                        'id': poi_id,
                        'name': poi.name,
                        'category': poi.category.value,
                        'location': {'lat': poi.coordinates.latitude, 'lng': poi.coordinates.longitude},
                        'rating': poi.rating or 0,
                        'features': poi.features,
                        'cuisine_type': poi.cuisine_type
                    })
            
            return matching_pois
            
        except Exception as e:
            logger.error(f"Error getting POIs by category: {e}")
            return []
    
    def update_route_realtime(self, route: Dict[str, Any], current_location: Dict[str, float], conditions: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Update route based on real-time conditions"""
        try:
            route_id = route.get('route_id')
            if not route_id:
                return None
            
            # Get current coordinates
            current_coords = Coordinates(current_location['lat'], current_location['lng'])
            
            # Check conditions
            conditions = conditions or {}
            traffic = conditions.get('traffic', 'normal')
            closures = conditions.get('closures', [])
            
            # Find waypoints that are not affected by closures
            original_waypoints = route.get('waypoints', [])
            valid_waypoints = []
            
            for waypoint in original_waypoints:
                waypoint_name = waypoint.get('name', '')
                # Skip waypoints that are closed
                if not any(closure.lower() in waypoint_name.lower() for closure in closures):
                    valid_waypoints.append(waypoint)
            
            # If traffic is heavy, adjust time estimates
            time_multiplier = 1.0
            if traffic == 'heavy':
                time_multiplier = 1.5
            elif traffic == 'light':
                time_multiplier = 0.8
            
            # Update the route
            updated_route = {
                'route_id': route_id,
                'waypoints': valid_waypoints,
                'total_distance_km': route.get('total_distance_km', 0),
                'estimated_time_minutes': int(route.get('estimated_time_minutes', 0) * time_multiplier),
                'current_location': current_location,
                'conditions': conditions,
                'last_updated': datetime.now().isoformat()
            }
            
            return updated_route
            
        except Exception as e:
            logger.error(f"Error updating route realtime: {e}")
            return None
    
    def plan_route(self, start: Dict[str, float], end: Dict[str, float], preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plan a simple route between two points"""
        try:
            start_coords = Coordinates(start['lat'], start['lng'])
            end_coords = Coordinates(end['lat'], end['lng'])
            
            # Calculate distance
            distance_km = self.location_data._haversine_distance(start_coords, end_coords)
            
            # Estimate time based on transport mode
            prefs = preferences or {}
            transport_mode = prefs.get('transport', 'walking')
            
            if transport_mode == 'walking':
                time_minutes = distance_km * 12  # ~5 km/h walking speed
            elif transport_mode == 'driving':
                time_minutes = distance_km * 2   # ~30 km/h city driving
            elif transport_mode == 'public':
                time_minutes = distance_km * 4   # ~15 km/h public transport
            else:
                time_minutes = distance_km * 8   # Mixed transport
            
            return {
                'route_id': f"route_{hash(f'{start_coords.latitude}{start_coords.longitude}{end_coords.latitude}{end_coords.longitude}') % 10000}",
                'start': start,
                'end': end,
                'distance_km': round(distance_km, 2),
                'estimated_time_minutes': int(time_minutes),
                'transport_mode': transport_mode,
                'waypoints': [start, end]
            }
            
        except Exception as e:
            logger.error(f"Error planning route: {e}")
            return {
                'route_id': None,
                'error': str(e)
            }

# Example usage and testing functions
def demo_live_location_system():
    """Demonstrate the live location routing system"""
    
    # Initialize system
    routing_system = LiveLocationRoutingSystem()
    
    # Create user session (example: user in Sultanahmet)
    user_location = Coordinates(41.0082, 28.9784)  # Sultanahmet
    session_id = routing_system.create_user_session("demo_user", user_location)
    
    print(f"Created session: {session_id}")
    
    # Get smart recommendations
    recommendations = routing_system.get_smart_recommendations(
        session_id=session_id,
        categories=[POICategory.MUSEUM, POICategory.RESTAURANT],
        filters={
            FilterCriteria.OPEN_HOURS: True,
            FilterCriteria.RATING: 4.0
        }
    )
    
    print("\n--- Smart Recommendations ---")
    print(json.dumps(recommendations, indent=2, default=str))
    
    # Plan a route to multiple POIs
    target_pois = ["hagia_sophia", "topkapi_palace", "pandeli_restaurant"]
    route_plan = routing_system.plan_dynamic_route(
        session_id=session_id,
        target_poi_ids=target_pois,
        algorithm=RoutingAlgorithm.TSP_NEAREST_NEIGHBOR,
        transport_mode="mixed"
    )
    
    print("\n--- Dynamic Route Plan ---")
    print(json.dumps(route_plan, indent=2, default=str))
    
    # Simulate user movement
    new_location = Coordinates(41.0086, 28.9802)  # Near Hagia Sophia
    routing_system.update_user_location(session_id, new_location)
    
    # Get offline recommendations
    offline_recs = routing_system.get_offline_recommendations(new_location)
    
    print("\n--- Offline Recommendations ---")
    print(json.dumps(offline_recs, indent=2, default=str))
    
    return routing_system

if __name__ == "__main__":
    # Run demo
    system = demo_live_location_system()
    print("\nLive Location Routing System demo completed!")
