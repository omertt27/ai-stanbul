"""
Production Route Maker Service - LLM-Free, OSM-Based
Istanbul Travel Assistant Route Generator using OSMNX and NetworkX

Featur                    # Instead of merging, use the largest/most central district for primary routing
            print("🎯 Selecting optimal district for routing...")
            
            # Find the district with most nodes (usually most comprehensive)
            largest_graph = max(graphs, key=lambda g: len(g.nodes))
            largest_district_index = graphs.index(largest_graph)
            selected_district = districts[largest_district_index].split(',')[0]
            
            self.graph = largest_graph
            self.available_districts = {d.split(',')[0]: g for d, g in zip(districts, graphs)}
            
            print(f"✅ Selected {selected_district} as primary routing district")
            print(f"📊 Primary graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            print(f"🗺️ Available districts: {list(self.available_districts.keys())}")e all district graphs into one comprehensive graph
            print("🔄 Merging all district graphs...")
            
            if len(graphs) == 1:
                self.graph = graphs[0]
            else:
                # Combine all graphs using NetworkX compose
                self.graph = graphs[0].copy()
                for additional_graph in graphs[1:]:
                    self.graph = nx.compose(self.graph, additional_graph)based street network pathfinding
- Multi-stop route optimization
- Attraction scoring and selection
- Distance and time estimation
- User preference integration
- No LLM/GPT dependencies - purely algorithmic
"""

import osmnx as ox
import networkx as nx
import geopandas as gpd
import folium
from geopy.distance import geodesic

# Fix SSL certificate issues for geopy
import ssl
import certifi
import geopy.geocoders
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
from typing import List, Dict, Tuple, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import json
from datetime import datetime, timedelta, time
import math
import random
from dataclasses import dataclass
from enum import Enum
import requests
import pytz

# Istanbul timezone
ISTANBUL_TIMEZONE = pytz.timezone('Europe/Istanbul')
import pytz

# Import local modules - with fallback for missing database
try:
    from database import get_db
except ImportError:
    # Fallback when database module is not available
    def get_db():
        return None

try:
    from services.performance_monitor import monitor_performance, profile_route_generation, performance_monitor
except ImportError:
    # Fallback when performance monitor is not available
    def monitor_performance(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    def profile_route_generation(func):
        return func
    class MockPerformanceMonitor:
        def log_route_generation(self, *args, **kwargs):
            pass
    performance_monitor = MockPerformanceMonitor()


class IstanbulOptimizations:
    """Istanbul-specific optimizations for routing and attraction selection"""
    
    @staticmethod
    def is_prayer_time(current_time: time) -> bool:
        """Check if current time falls within typical prayer times in Istanbul"""
        # Approximate prayer times for Istanbul (these vary by season but this gives a general idea)
        prayer_times = [
            (time(5, 30), time(6, 30)),  # Fajr
            (time(12, 0), time(13, 0)),  # Dhuhr  
            (time(15, 30), time(16, 30)), # Asr
            (time(18, 0), time(19, 0)),  # Maghrib
            (time(19, 30), time(20, 30)) # Isha
        ]
        
        for start_time, end_time in prayer_times:
            if start_time <= current_time <= end_time:
                return True
        return False
    
    @staticmethod
    def is_rush_hour(current_time: time) -> bool:
        """Check if current time is during Istanbul rush hours"""
        # Morning rush: 7:30-9:30, Evening rush: 17:00-19:30
        morning_rush = time(7, 30) <= current_time <= time(9, 30)
        evening_rush = time(17, 0) <= current_time <= time(19, 30)
        return morning_rush or evening_rush
    
    @staticmethod
    def get_district_clusters() -> Dict[str, Any]:
        """Get Istanbul district clusters for efficient routing"""
        return {
            "historic_core": {
                "districts": ["Fatih", "Sultanahmet", "Eminönü"],
                "center": [41.0082, 28.9784],
                "characteristics": ["historic", "walking_friendly", "tourist_dense"],
                "recommended_time": 3.0  # hours
            },
            "modern_center": {
                "districts": ["Beyoğlu", "Taksim", "Galata"],
                "center": [41.0369, 28.9853],
                "characteristics": ["modern", "nightlife", "shopping"],
                "recommended_time": 2.5
            },
            "bosphorus_european": {
                "districts": ["Beşiktaş", "Ortaköy", "Arnavutköy"],
                "center": [41.0473, 29.0061],
                "characteristics": ["scenic", "waterfront", "upscale"],
                "recommended_time": 2.0
            },
            "bosphorus_asian": {
                "districts": ["Üsküdar", "Kadıköy", "Çengelköy"],
                "center": [41.0214, 29.0069],
                "characteristics": ["local", "authentic", "ferry_access"],
                "recommended_time": 2.5
            },
            "golden_horn": {
                "districts": ["Eyüp", "Balat", "Fener"],
                "center": [41.0539, 28.9468],
                "characteristics": ["authentic", "religious", "colorful"],
                "recommended_time": 2.0
            }
        }
    
    @staticmethod
    def get_ferry_routes() -> Dict[str, Any]:
        """Get major ferry routes in Istanbul"""
        return {
            "bosphorus_tour": {
                "route": "Eminönü - Üsküdar - Beşiktaş - Ortaköy - Arnavutköy - Sarıyer",
                "duration_minutes": 90,
                "frequency_minutes": 30,
                "scenic_value": 1.0,
                "price_tl": 15.0
            },
            "kadikoy_eminonu": {
                "route": "Kadıköy - Eminönü",
                "duration_minutes": 20,
                "frequency_minutes": 15,
                "scenic_value": 0.7,
                "price_tl": 4.0
            },
            "uskudar_eminonu": {
                "route": "Üsküdar - Eminönü",
                "duration_minutes": 15,
                "frequency_minutes": 10,
                "scenic_value": 0.8,
                "price_tl": 4.0
            },
            "golden_horn": {
                "route": "Eminönü - Eyüp",
                "duration_minutes": 25,
                "frequency_minutes": 20,
                "scenic_value": 0.9,
                "price_tl": 4.0
            }
        }
    
    @staticmethod
    def get_weather_alternatives() -> Dict[str, List[str]]:
        """Get indoor alternatives for different weather conditions"""
        return {
            "rain": [
                "museum", "gallery", "covered_bazaar", "shopping_center", 
                "restaurant", "cafe", "hammam", "cistern"
            ],
            "extreme_heat": [
                "museum", "gallery", "underground", "air_conditioned_venue",
                "shaded_park", "waterfront_cafe"
            ],
            "cold": [
                "museum", "gallery", "heated_venue", "hammam", 
                "covered_market", "indoor_attraction"
            ],
            "snow": [
                "museum", "gallery", "covered_venue", "cafe", 
                "heated_indoor_space", "scenic_winter_view"
            ]
        }
    
    @staticmethod
    def get_prayer_times(current_date: datetime) -> Dict[str, time]:
        """Get prayer times for given date (simplified approximation)"""
        # This is a simplified version - in production you'd use an Islamic calendar API
        return {
            "fajr": time(5, 45),
            "dhuhr": time(12, 15),
            "asr": time(15, 45),
            "maghrib": time(18, 15),
            "isha": time(19, 45)
        }


class RouteStyle(Enum):
    EFFICIENT = "efficient"      # Minimize total distance
    SCENIC = "scenic"           # Prioritize beautiful routes
    CULTURAL = "cultural"       # Focus on cultural attractions
    BALANCED = "balanced"       # Balance of all factors

class TransportMode(Enum):
    WALKING = "walking"
    DRIVING = "driving"
    PUBLIC_TRANSPORT = "public_transport"

@dataclass
class RouteRequest:
    """User request for route generation"""
    start_lat: float
    start_lng: float
    end_lat: Optional[float] = None
    end_lng: Optional[float] = None
    max_distance_km: float = 5.0
    available_time_hours: float = 4.0
    preferred_categories: List[str] = None
    route_style: RouteStyle = RouteStyle.BALANCED
    transport_mode: TransportMode = TransportMode.WALKING
    include_food: bool = True
    max_attractions: int = 6

@dataclass
class RoutePoint:
    """A point in the route with metadata"""
    lat: float
    lng: float
    attraction_id: Optional[int] = None
    name: str = ""
    category: str = ""
    estimated_duration_minutes: int = 60
    arrival_time: Optional[str] = None
    score: float = 0.0
    notes: str = ""

@dataclass
class GeneratedRoute:
    """Complete generated route with all metadata"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    points: List[RoutePoint] = None
    total_distance_km: float = 0.0
    estimated_duration_hours: float = 0.0
    overall_score: float = 0.0
    diversity_score: float = 0.0
    efficiency_score: float = 0.0
    created_at: Optional[datetime] = None

class IstanbulRoutemaker:
    """
    Production-ready route maker for Istanbul using OSM data
    LLM-free, algorithmic approach with intelligent attraction selection
    """
    
    # Class-level cache to share graph data across instances
    _shared_graph = None
    _shared_districts = None
    _shared_available_districts = None
    _shared_primary_district = None
    _class_graph_loaded = False
    
    def __init__(self):
        # Use shared graph data if available
        if IstanbulRoutemaker._class_graph_loaded:
            print("🗺️ Reusing cached Istanbul graph data")
            self.graph = IstanbulRoutemaker._shared_graph
            self.covered_districts = IstanbulRoutemaker._shared_districts or ["Cached"]
            self.available_districts = IstanbulRoutemaker._shared_available_districts or {}
            self.primary_district = IstanbulRoutemaker._shared_primary_district or "Unknown"
        else:
            self.graph = None
            self.covered_districts = []
            self.available_districts = {}
            self.primary_district = None
            
        self.istanbul_bounds = {
            'north': 41.25,
            'south': 40.80,
            'east': 29.40,
            'west': 28.60
        }
        self._graph_loaded = False
        self._load_istanbul_graph()
    
    def _load_istanbul_graph(self):
        """Load Istanbul street network from OSM - using district selection instead of merging"""
        # Check if class-level cache is available
        if IstanbulRoutemaker._class_graph_loaded:
            print("🗺️ Using cached Istanbul graph data from class cache")
            self.graph = IstanbulRoutemaker._shared_graph
            self.covered_districts = IstanbulRoutemaker._shared_districts or ["Cached"]
            self.available_districts = IstanbulRoutemaker._shared_available_districts or {}
            self.primary_district = IstanbulRoutemaker._shared_primary_district or "Unknown"
            self._graph_loaded = True
            return
            
        # Check if already loaded to prevent multiple downloads
        if self._graph_loaded and self.graph is not None:
            print("🗺️ Istanbul graph already loaded, skipping OpenStreetMap download")
            return
            
        try:
            print("🗺️ Loading Istanbul tourist districts from OpenStreetMap...")
            
            # Define all major tourist districts by name (more accurate than bounding boxes)
            districts = [
                "Kadıköy, Istanbul, Turkey",
                "Şişli, Istanbul, Turkey", 
                "Beşiktaş, Istanbul, Turkey",
                "Sarıyer, Istanbul, Turkey",
                "Fatih, Istanbul, Turkey",
                "Beyoğlu, Istanbul, Turkey",
                "Üsküdar, Istanbul, Turkey"
            ]
            
            print(f"📍 Loading {len(districts)} districts individually...")
            graphs = []
            
            for district in districts:
                try:
                    print(f"   🏘️ Downloading {district}...")
                    G = ox.graph_from_place(district, network_type="walk")
                    graphs.append(G)
                    print(f"   ✅ {district}: {len(G.nodes)} nodes, {len(G.edges)} edges")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {district}: {e}")
                    continue
            
            if not graphs:
                raise Exception("No districts loaded successfully")
            
            # Instead of merging, use the largest/most central district for primary routing
            print("🎯 Selecting optimal district for routing...")
            
            # Find the district with most nodes (usually most comprehensive)
            largest_graph = max(graphs, key=lambda g: len(g.nodes))
            largest_district_index = graphs.index(largest_graph)
            selected_district = districts[largest_district_index].split(',')[0]
            
            self.graph = largest_graph
            self.available_districts = {d.split(',')[0]: g for d, g in zip(districts, graphs)}
            
            print(f"✅ Selected {selected_district} as primary routing district")
            print(f"📊 Primary graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            print(f"🗺️ Available districts: {list(self.available_districts.keys())}")
            
            # Store district info for reference
            self.covered_districts = [selected_district]
            self.primary_district = selected_district
            
        except Exception as e:
            print(f"❌ Error loading full Istanbul graph: {e}")
            print("🔄 Trying fallback approach with smaller area...")
            try:
                # Fallback to just the core historical area (Fatih only)
                print("📍 Loading Fatih district only as fallback...")
                self.graph = ox.graph_from_place("Fatih, Istanbul, Turkey", network_type="walk")
                print(f"⚠️ Loaded fallback graph (Fatih only): {len(self.graph.nodes)} nodes")
                self.covered_districts = ["Fatih"]
            except Exception as fallback_error:
                print(f"❌ District fallback also failed: {fallback_error}")
                print("🔄 Trying bounding box fallback...")
                try:
                    # Final fallback to small bounding box
                    core_bbox = (41.025, 40.990, 28.995, 28.945)  # Small Sultanahmet area
                    self.graph = ox.graph_from_bbox(
                        bbox=core_bbox,
                        network_type='walk',
                        simplify=True
                    )
                    print(f"⚠️ Loaded minimal fallback graph: {len(self.graph.nodes)} nodes")
                    self.covered_districts = ["Sultanahmet Core"]
                except Exception as final_error:
                    print(f"❌ All fallbacks failed: {final_error}")
                    print("🆘 Using minimal test graph...")
                    self.graph = self._create_fallback_graph()
                    self.covered_districts = ["Test Graph"]
        
        # Mark graph as loaded to prevent future downloads
        self._graph_loaded = True
        
        # Store in class cache for other instances
        IstanbulRoutemaker._shared_graph = self.graph
        IstanbulRoutemaker._shared_districts = getattr(self, 'covered_districts', ["Unknown"])
        IstanbulRoutemaker._shared_available_districts = getattr(self, 'available_districts', {})
        IstanbulRoutemaker._shared_primary_district = getattr(self, 'primary_district', "Unknown")
        IstanbulRoutemaker._class_graph_loaded = True
        
        print("✅ Graph loading completed - cached for future instances")
    
    def _create_fallback_graph(self):
        """Create a simple fallback graph for testing"""
        G = nx.Graph()
        # Add some basic Istanbul nodes (simplified)
        nodes = [
            (41.0083, 28.9784),  # Sultanahmet
            (41.0270, 28.9744),  # Galata
            (41.0285, 28.9770),  # Karakoy
            (41.0351, 29.0089),  # Besiktas
            (40.9983, 29.0270),  # Kadikoy
        ]
        
        for i, (lat, lon) in enumerate(nodes):
            G.add_node(i, y=lat, x=lon)
        
        # Add edges between nearby nodes
        for i in range(len(nodes)-1):
            G.add_edge(i, i+1, length=1000)  # 1km default
        
        return G
    
    def find_nearest_node(self, lat: float, lng: float) -> int:
        """Find the nearest graph node to given coordinates"""
        try:
            return ox.nearest_nodes(self.graph, lng, lat)
        except:
            # Fallback for simple graphs
            min_dist = float('inf')
            nearest = None
            for node, data in self.graph.nodes(data=True):
                node_lat = data.get('y', 0)
                node_lng = data.get('x', 0)
                dist = geodesic((lat, lng), (node_lat, node_lng)).kilometers
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
            return nearest or 0
    
    def calculate_path_distance(self, start_node: int, end_node: int) -> float:
        """Calculate shortest path distance between two nodes"""
        try:
            path = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            total_length = 0
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    total_length += edge_data.get('length', 1000)  # Default 1km
            return total_length / 1000  # Convert to km
        except:
            # Fallback: calculate straight-line distance
            start_data = self.graph.nodes[start_node]
            end_data = self.graph.nodes[end_node]
            return geodesic(
                (start_data.get('y', 0), start_data.get('x', 0)),
                (end_data.get('y', 0), end_data.get('x', 0))
            ).kilometers
    
    def get_attractions_near_point(self, lat: float, lng: float, radius_km: float, db: Session) -> List:
        """Get attractions within radius of a point"""
        from models import EnhancedAttraction
        
        # Simple bounding box query (more efficient than complex spatial queries)
        lat_delta = radius_km / 111  # Approximate degrees per km
        lng_delta = radius_km / (111 * math.cos(math.radians(lat)))
        
        attractions = db.query(EnhancedAttraction).filter(
            and_(
                EnhancedAttraction.coordinates_lat.between(lat - lat_delta, lat + lat_delta),
                EnhancedAttraction.coordinates_lng.between(lng - lng_delta, lng + lng_delta),
                EnhancedAttraction.is_active == True
            )
        ).all()
        
        # Filter by actual distance and return sorted by distance
        nearby_attractions = []
        for attraction in attractions:
            distance = geodesic((lat, lng), (attraction.coordinates_lat, attraction.coordinates_lng)).kilometers
            if distance <= radius_km:
                attraction.distance_from_point = distance
                nearby_attractions.append(attraction)
        
        return sorted(nearby_attractions, key=lambda x: x.distance_from_point)
    
    def score_attraction(self, attraction, request: RouteRequest, current_time: str = "10:00", visited_categories: List[str] = None) -> float:
        """Enhanced attraction scoring based on user preferences and route context"""
        if visited_categories is None:
            visited_categories = []
        
        # Base score from popularity (1-5)
        score = float(attraction.popularity_score)
        
        # Category preference bonus (stronger weighting)
        if request.preferred_categories and attraction.category in request.preferred_categories:
            score += 1.5
        
        # Route style adjustments (enhanced)
        style_bonuses = {
            RouteStyle.CULTURAL: {
                'mosque': 1.0, 'museum': 1.0, 'cultural_site': 0.8, 
                'historical_site': 0.8, 'religious_site': 0.6
            },
            RouteStyle.SCENIC: {
                'viewpoint': 1.0, 'park': 0.8, 'waterfront': 1.0,
                'bridge': 0.6, 'garden': 0.7, 'palace': 0.5
            },
            RouteStyle.EFFICIENT: {
                # Bonus for attractions with shorter visit times
            },
            RouteStyle.BALANCED: {
                # Small bonus for all categories
            }
        }
        
        if request.route_style in style_bonuses:
            category_bonus = style_bonuses[request.route_style].get(attraction.category, 0)
            score += category_bonus
        
        # Time of day optimization (enhanced)
        hour = int(current_time.split(':')[0])
        if hasattr(attraction, 'best_time_of_day') and attraction.best_time_of_day:
            time_bonus = 0
            if attraction.best_time_of_day == 'morning' and 6 <= hour <= 11:
                time_bonus = 0.5
            elif attraction.best_time_of_day == 'afternoon' and 12 <= hour <= 17:
                time_bonus = 0.5
            elif attraction.best_time_of_day == 'evening' and 18 <= hour <= 22:
                time_bonus = 0.5
            elif attraction.best_time_of_day == 'any':
                time_bonus = 0.2
            score += time_bonus
        
        # Crowd level preferences (enhanced)
        if hasattr(attraction, 'crowd_level'):
            if attraction.crowd_level == 'low':
                score += 0.4
            elif attraction.crowd_level == 'medium':
                score += 0.1
            elif attraction.crowd_level == 'high':
                score -= 0.3
        
        # Diversity bonus (encourage variety)
        if attraction.category not in visited_categories:
            score += 0.3
        elif visited_categories.count(attraction.category) >= 2:
            score -= 0.4  # Penalty for too much repetition
        
        # Visit duration consideration for efficiency
        if request.route_style == RouteStyle.EFFICIENT:
            visit_time = getattr(attraction, 'estimated_visit_time_minutes', 60)
            if visit_time <= 30:
                score += 0.3
            elif visit_time >= 120:
                score -= 0.2
        
        # Food preference integration
        if request.include_food and attraction.category in ['restaurant', 'cafe', 'food_market']:
            score += 0.4
        elif not request.include_food and attraction.category in ['restaurant', 'cafe', 'food_market']:
            score -= 0.5
        
        # Distance penalty for far attractions (encourage compact routes)
        if hasattr(attraction, 'distance_from_point'):
            distance_km = attraction.distance_from_point
            if distance_km > request.max_distance_km * 0.8:
                score -= 0.3
            elif distance_km <= request.max_distance_km * 0.3:
                score += 0.2
        
        return max(0.1, min(5.0, score))  # Clamp between 0.1 and 5.0
    
    def optimize_route_order(self, attractions: List, start_lat: float, start_lng: float, method: str = "tsp") -> List:
        """Optimize the order of attractions using various algorithms"""
        if not attractions:
            return []
        
        if len(attractions) <= 2:
            return attractions
        
        if method == "tsp" and len(attractions) <= 10:
            return self._tsp_optimize(attractions, start_lat, start_lng)
        elif method == "tsp_heuristic" and len(attractions) <= 15:
            return self._tsp_heuristic_optimize(attractions, start_lat, start_lng)
        else:
            # Fallback to nearest neighbor for large sets
            return self._nearest_neighbor_optimize(attractions, start_lat, start_lng)
    
    def _nearest_neighbor_optimize(self, attractions: List, start_lat: float, start_lng: float) -> List:
        """Simple nearest-neighbor optimization"""
        optimized = []
        remaining = attractions.copy()
        current_lat, current_lng = start_lat, start_lng
        
        while remaining:
            nearest = min(remaining, key=lambda a: geodesic((current_lat, current_lng), (a.coordinates_lat, a.coordinates_lng)).kilometers)
            optimized.append(nearest)
            remaining.remove(nearest)
            current_lat, current_lng = nearest.coordinates_lat, nearest.coordinates_lng
        
        return optimized
    
    @monitor_performance(operation_name="tsp_exact_optimization")
    def _tsp_optimize(self, attractions: List, start_lat: float, start_lng: float) -> List:
        """Traveling Salesman Problem optimization using brute force for small sets"""
        from itertools import permutations
        
        if len(attractions) > 8:  # Brute force limit
            return self._tsp_heuristic_optimize(attractions, start_lat, start_lng)
        
        best_distance = float('inf')
        best_route = attractions
        
        # Try all permutations and find the shortest
        for perm in permutations(attractions):
            total_distance = 0
            current_lat, current_lng = start_lat, start_lng
            
            for attraction in perm:
                distance = geodesic((current_lat, current_lng), (attraction.coordinates_lat, attraction.coordinates_lng)).kilometers
                total_distance += distance
                current_lat, current_lng = attraction.coordinates_lat, attraction.coordinates_lng
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_route = list(perm)
        
        return best_route
    
    @monitor_performance(operation_name="tsp_heuristic_optimization")
    def _tsp_heuristic_optimize(self, attractions: List, start_lat: float, start_lng: float) -> List:
        """TSP optimization using 2-opt heuristic for larger sets"""
        # Start with nearest neighbor
        route = self._nearest_neighbor_optimize(attractions, start_lat, start_lng)
        
        # Apply 2-opt improvement
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Calculate current distance
                    current_distance = self._calculate_route_distance(route, start_lat, start_lng)
                    
                    # Swap segments
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_distance = self._calculate_route_distance(new_route, start_lat, start_lng)
                    
                    if new_distance < current_distance:
                        route = new_route
                        improved = True
                        break
                
                if improved:
                    break
        
        print(f"🔧 TSP optimization completed in {iterations} iterations")
        return route
    
    def _calculate_route_distance(self, attractions: List, start_lat: float, start_lng: float) -> float:
        """Calculate total distance for a route"""
        total_distance = 0
        current_lat, current_lng = start_lat, start_lng
        
        for attraction in attractions:
            distance = geodesic((current_lat, current_lng), (attraction.coordinates_lat, attraction.coordinates_lng)).kilometers
            total_distance += distance
            current_lat, current_lng = attraction.coordinates_lat, attraction.coordinates_lng
        
        return total_distance
    
    @profile_route_generation
    @monitor_performance(operation_name="route_generation", track_cache=True)
    def generate_route(self, request: RouteRequest, db: Session) -> GeneratedRoute:
        """Generate a complete multi-stop route with TSP optimization and caching"""
        print(f"🚀 Generating route: {request.max_distance_km}km, {request.available_time_hours}h, style: {request.route_style.value}")
        
        # Phase 3: Check cache first
        try:
            from services.route_cache import route_cache
            cached_route = route_cache.get_cached_route(request)
            if cached_route:
                print("⚡ Returning cached route")
                return cached_route
        except ImportError:
            print("⚠️ Route cache not available")
        
        # Auto-select best district for starting location
        if hasattr(self, 'available_districts'):
            best_district = self.get_best_district_for_location(request.start_lat, request.start_lng)
            if best_district != self.primary_district:
                print(f"🎯 Switching to {best_district} district for better coverage")
                self.switch_to_district(best_district)
        
        # Get candidate attractions
        attractions = self.get_attractions_near_point(
            request.start_lat, 
            request.start_lng, 
            request.max_distance_km, 
            db
        )
        
        print(f"🎯 Found {len(attractions)} candidate attractions")
        
        if not attractions:
            print("⚠️ No attractions found in the specified area")
            return self._create_empty_route(request)
        
        # Apply Istanbul-specific optimizations
        istanbul_optimized_attractions = self.optimize_for_istanbul_context(attractions, request)
        
        # Enhanced attraction selection with iterative scoring
        selected_attractions = self._select_optimal_attractions(istanbul_optimized_attractions, request)
        
        print(f"⭐ Selected {len(selected_attractions)} attractions with Istanbul optimizations")
        
        # Multi-stop TSP optimization
        optimization_method = "tsp" if len(selected_attractions) <= 8 else "tsp_heuristic"
        optimized_attractions = self.optimize_route_order(
            selected_attractions, 
            request.start_lat, 
            request.start_lng,
            method=optimization_method
        )
        
        # Build route points
        route_points = []
        
        # Add start point
        route_points.append(RoutePoint(
            lat=request.start_lat,
            lng=request.start_lng,
            name="Starting Point",
            category="start",
            estimated_duration_minutes=0,
            arrival_time="09:00"
        ))
        
        # Add attraction points
        current_time = datetime.strptime("09:00", "%H:%M")
        total_distance = 0.0
        
        for i, attraction in enumerate(optimized_attractions):
            # Calculate distance from previous point
            prev_point = route_points[-1]
            distance = geodesic((prev_point.lat, prev_point.lng), (attraction.coordinates_lat, attraction.coordinates_lng)).kilometers
            total_distance += distance
            
            # Calculate travel time (walking speed ~4 km/h)
            travel_time_minutes = int((distance / 4.0) * 60)
            current_time += timedelta(minutes=travel_time_minutes)
            
            # Add visit time
            visit_time = attraction.estimated_visit_time_minutes or 60
            
            route_points.append(RoutePoint(
                lat=attraction.coordinates_lat,
                lng=attraction.coordinates_lng,
                attraction_id=attraction.id,
                name=attraction.name,
                category=attraction.category,
                estimated_duration_minutes=visit_time,
                arrival_time=current_time.strftime("%H:%M"),
                score=attraction.route_score,
                notes=f"Distance from previous: {distance:.1f}km"
            ))
            
            current_time += timedelta(minutes=visit_time)
        
        # Add end point if specified
        if request.end_lat and request.end_lng:
            prev_point = route_points[-1]
            distance = geodesic((prev_point.lat, prev_point.lng), (request.end_lat, request.end_lng)).kilometers
            total_distance += distance
            
            route_points.append(RoutePoint(
                lat=request.end_lat,
                lng=request.end_lng,
                name="End Point",
                category="end",
                estimated_duration_minutes=0,
                arrival_time=current_time.strftime("%H:%M")
            ))
        
        # Calculate scores
        overall_score = sum(p.score for p in route_points if p.score > 0) / max(1, len([p for p in route_points if p.score > 0]))
        diversity_score = len(set(p.category for p in route_points)) / len(route_points) if route_points else 0
        efficiency_score = max(0, 5.0 - (total_distance / request.max_distance_km * 2))  # Penalty for long routes
        
        total_duration_hours = (current_time - datetime.strptime("09:00", "%H:%M")).total_seconds() / 3600
        
        generated_route = GeneratedRoute(
            name=f"Istanbul Route - {len(optimized_attractions)} Stops",
            description=f"Optimized {request.route_style.value} route covering {len(optimized_attractions)} attractions",
            points=route_points,
            total_distance_km=round(total_distance, 2),
            estimated_duration_hours=round(total_duration_hours, 1),
            overall_score=round(overall_score, 2),
            diversity_score=round(diversity_score, 2),
            efficiency_score=round(efficiency_score, 2),
            created_at=datetime.utcnow()
        )
        
        # Phase 3: Cache the generated route
        try:
            from services.route_cache import route_cache
            route_cache.cache_route(request, generated_route)
        except ImportError:
            pass  # Cache not available
        
        return generated_route
    
    def save_route_to_db(self, generated_route: GeneratedRoute, db: Session):
        """Save generated route to database"""
        from models import Route, RouteWaypoint
        
        # Create route record
        route = Route(
            name=generated_route.name,
            description=generated_route.description,
            start_lat=generated_route.points[0].lat if generated_route.points else 0,
            start_lng=generated_route.points[0].lng if generated_route.points else 0,
            end_lat=generated_route.points[-1].lat if len(generated_route.points) > 1 else None,
            end_lng=generated_route.points[-1].lng if len(generated_route.points) > 1 else None,
            total_distance_km=generated_route.total_distance_km,
            estimated_duration_hours=generated_route.estimated_duration_hours,
            overall_score=generated_route.overall_score,
            diversity_score=generated_route.diversity_score,
            efficiency_score=generated_route.efficiency_score,
            transportation_mode=TransportMode.WALKING.value
        )
        
        db.add(route)
        db.flush()  # Get the ID
        
        # Create waypoints
        for order, point in enumerate(generated_route.points):
            if point.attraction_id:  # Skip start/end points without attractions
                waypoint = RouteWaypoint(
                    route_id=route.id,
                    attraction_id=point.attraction_id,
                    waypoint_order=order,
                    estimated_arrival_time=point.arrival_time,
                    suggested_duration_minutes=point.estimated_duration_minutes,
                    attraction_score=point.score,
                    notes=point.notes
                )
                db.add(waypoint)
        
        db.commit()
        return route
    
    def generate_map_html(self, generated_route: GeneratedRoute) -> str:
        """Generate an interactive HTML map of the route"""
        if not generated_route.points:
            return "<p>No route points to display</p>"
        
        # Create map centered on route
        center_lat = sum(p.lat for p in generated_route.points) / len(generated_route.points)
        center_lng = sum(p.lng for p in generated_route.points) / len(generated_route.points)
        
        m = folium.Map(location=[center_lat, center_lng], zoom_start=13)
        
        # Add route points
        for i, point in enumerate(generated_route.points):
            color = 'green' if i == 0 else 'red' if i == len(generated_route.points) - 1 else 'blue'
            
            folium.Marker(
                [point.lat, point.lng],
                popup=f"{point.name}<br>Category: {point.category}<br>Arrival: {point.arrival_time}",
                tooltip=f"{i+1}. {point.name}",
                icon=folium.Icon(color=color)
            ).add_to(m)
        
        # Add route line
        route_coords = [[p.lat, p.lng] for p in generated_route.points]
        folium.PolyLine(route_coords, color='red', weight=3, opacity=0.7).add_to(m)
        
        return m._repr_html_()
    
    def switch_to_district(self, district_name: str) -> bool:
        """Switch to a different district for routing if available"""
        if hasattr(self, 'available_districts') and district_name in self.available_districts:
            self.graph = self.available_districts[district_name]
            self.primary_district = district_name
            self.covered_districts = [district_name]
            print(f"🔄 Switched to {district_name} district: {len(self.graph.nodes)} nodes")
            return True
        else:
            print(f"⚠️ District {district_name} not available. Available: {getattr(self, 'available_districts', {}).keys()}")
            return False
    
    def get_best_district_for_location(self, lat: float, lng: float) -> str:
        """Determine which district would be best for a given location"""
        # Simple heuristic based on coordinates
        if hasattr(self, 'available_districts'):
            # This is a simplified approach - in production you'd use proper spatial checks
            district_centers = {
                'Fatih': (41.008, 28.978),        # Sultanahmet area
                'Beyoğlu': (41.036, 28.977),      # Galata/Taksim
                'Beşiktaş': (41.043, 29.000),     # Beşiktaş center
                'Kadıköy': (40.980, 29.030),      # Kadıköy center
                'Üsküdar': (41.025, 29.020),      # Üsküdar center
                'Şişli': (41.055, 28.980),        # Şişli center
                'Sarıyer': (41.100, 29.050)       # Sarıyer center
            }
            
            best_district = min(
                district_centers.keys(),
                key=lambda d: geodesic((lat, lng), district_centers[d]).kilometers
                if d in self.available_districts else float('inf')
            )
            
            return best_district if best_district in self.available_districts else self.primary_district
        
        return self.primary_district

    def _select_optimal_attractions(self, attractions: List, request: RouteRequest) -> List:
        """Select optimal attractions based on scoring and constraints"""
        if not attractions:
            return []
        
        # Score all attractions
        visited_categories = []
        scored_attractions = []
        
        for attraction in attractions:
            score = self.score_attraction(attraction, request, visited_categories=visited_categories)
            attraction.route_score = score
            scored_attractions.append(attraction)
        
        # Sort by score (highest first)
        scored_attractions.sort(key=lambda x: x.route_score, reverse=True)
        
        # Select attractions considering constraints
        selected = []
        total_time = 0
        visited_categories = []
        
        for attraction in scored_attractions:
            if len(selected) >= request.max_attractions:
                break
            
            # Check time constraint
            visit_time = getattr(attraction, 'estimated_visit_time_minutes', 60)
            if total_time + visit_time > request.available_time_hours * 60:
                continue
            
            # Check distance constraint (already filtered in get_attractions_near_point)
            
            # Add to selection
            selected.append(attraction)
            total_time += visit_time
            visited_categories.append(attraction.category)
            
            # Add travel time between attractions (estimate)
            if len(selected) > 1:
                total_time += 15  # 15 minutes average travel time between attractions
        
        print(f"🎯 Selected {len(selected)} attractions from {len(attractions)} candidates")
        return selected
    
    def _create_empty_route(self, request: RouteRequest) -> GeneratedRoute:
        """Create an empty route when no attractions are found"""
        route_points = [
            RoutePoint(
                lat=request.start_lat,
                lng=request.start_lng,
                name="Starting Point",
                category="start",
                estimated_duration_minutes=0,
                arrival_time="09:00",
                notes="No attractions found in the specified area"
            )
        ]
        
        # Add end point if specified
        if request.end_lat and request.end_lng:
            distance = geodesic((request.start_lat, request.start_lng), (request.end_lat, request.end_lng)).kilometers
            route_points.append(RoutePoint(
                lat=request.end_lat,
                lng=request.end_lng,
                name="End Point",
                category="end",
                estimated_duration_minutes=0,
                arrival_time="09:00",
                notes=f"Direct route: {distance:.1f}km"
            ))
        
        return GeneratedRoute(
            name="Empty Route",
            description="No attractions found in the specified area",
            points=route_points,
            total_distance_km=geodesic((request.start_lat, request.start_lng), (request.end_lat or request.start_lat, request.end_lng or request.start_lng)).kilometers if request.end_lat and request.end_lng else 0.0,
            estimated_duration_hours=0.0,
            overall_score=0.0,
            diversity_score=0.0,
            efficiency_score=0.0,
            created_at=datetime.utcnow()
        )

    def get_district_status(self) -> Dict[str, Any]:
        """Get status of loaded districts"""
        return {
            "primary_district": getattr(self, 'primary_district', 'Unknown'),
            "covered_districts": getattr(self, 'covered_districts', []),
            "available_districts": list(getattr(self, 'available_districts', {}).keys()),
            "graph_stats": {
                "nodes": len(self.graph.nodes) if self.graph else 0,
                "edges": len(self.graph.edges) if self.graph else 0
            }
        }
    
    def optimize_for_istanbul_context(self, attractions: List, request: RouteRequest, current_time: datetime = None) -> List:
        """Apply Istanbul-specific optimizations to attraction selection and routing"""
        if not current_time:
            current_time = datetime.now(ISTANBUL_TIMEZONE)
        
        optimized_attractions = attractions.copy()
        
        # 1. Time-of-day optimization
        optimized_attractions = self._apply_time_optimization(optimized_attractions, current_time)
        
        # 2. District-based clustering
        optimized_attractions = self._apply_district_clustering(optimized_attractions, request)
        
        # 3. Ferry route integration
        optimized_attractions = self._consider_ferry_routes(optimized_attractions, request)
        
        # 4. Weather-aware filtering (if weather data available)
        optimized_attractions = self._apply_weather_filtering(optimized_attractions, request)
        
        return optimized_attractions
    
    def _apply_time_optimization(self, attractions: List, current_time: datetime) -> List:
        """Optimize attractions based on prayer times and rush hours"""
        optimized = []
        
        for attraction in attractions:
            time_score = 1.0
            
            # Check for mosque visits during prayer times
            if attraction.category == 'mosque':
                if IstanbulOptimizations.is_prayer_time(current_time.time()):
                    # Reduce score for mosque visits during prayer times (tourists should be respectful)
                    time_score = 0.3
                    attraction.time_warning = "Prayer time - limited tourist access"
                else:
                    # Bonus for visiting mosques outside prayer times
                    time_score = 1.2
            
            # Check for transport hubs during rush hour
            if attraction.category in ['transportation', 'ferry_terminal']:
                if IstanbulOptimizations.is_rush_hour(current_time.time()):
                    time_score = 0.7
                    attraction.time_warning = "Rush hour - expect crowds"
                else:
                    time_score = 1.1
            
            # Apply time optimization to existing score
            if hasattr(attraction, 'route_score'):
                attraction.route_score *= time_score
            
            attraction.time_optimization_score = time_score
            optimized.append(attraction)
        
        return optimized
    
    def _apply_district_clustering(self, attractions: List, request: RouteRequest) -> List:
        """Group attractions by district clusters for efficient routing"""
        clusters = IstanbulOptimizations.get_district_clusters()
        clustered_attractions = {}
        
        # Group attractions by cluster
        for attraction in attractions:
            district = getattr(attraction, 'district', 'unknown')
            cluster_found = False
            
            for cluster_name, cluster_info in clusters.items():
                if district in cluster_info['districts']:
                    if cluster_name not in clustered_attractions:
                        clustered_attractions[cluster_name] = []
                    clustered_attractions[cluster_name].append(attraction)
                    attraction.cluster = cluster_name
                    cluster_found = True
                    break
            
            if not cluster_found:
                if 'other' not in clustered_attractions:
                    clustered_attractions['other'] = []
                clustered_attractions['other'].append(attraction)
                attraction.cluster = 'other'
        
        # Prioritize clusters based on route style and available time
        cluster_priorities = self._calculate_cluster_priorities(clustered_attractions, request)
        
        # Reorder attractions based on cluster priorities
        optimized = []
        for cluster_name in sorted(cluster_priorities.keys(), key=lambda x: cluster_priorities[x], reverse=True):
            if cluster_name in clustered_attractions:
                cluster_attractions = clustered_attractions[cluster_name]
                
                # Add cluster bonus to attractions
                cluster_bonus = cluster_priorities[cluster_name] * 0.1
                for attraction in cluster_attractions:
                    if hasattr(attraction, 'route_score'):
                        attraction.route_score += cluster_bonus
                
                optimized.extend(cluster_attractions)
        
        return optimized
    
    def _calculate_cluster_priorities(self, clustered_attractions: Dict, request: RouteRequest) -> Dict[str, float]:
        """Calculate priority scores for district clusters"""
        clusters = IstanbulOptimizations.get_district_clusters()
        priorities = {}
        
        for cluster_name, attractions in clustered_attractions.items():
            if cluster_name == 'other':
                priorities[cluster_name] = 1.0
                continue
            
            cluster_info = clusters.get(cluster_name, {})
            priority = 3.0  # Base priority
            
            # Route style bonuses
            if request.route_style == RouteStyle.CULTURAL:
                if cluster_name == 'historical_peninsula':
                    priority += 2.0
                elif cluster_name == 'galata_beyoglu':
                    priority += 1.0
            elif request.route_style == RouteStyle.SCENIC:
                if cluster_name == 'bosphorus_north':
                    priority += 2.0
                elif cluster_name == 'asian_side':
                    priority += 1.5
            
            # Time availability consideration
            recommended_time = cluster_info.get('recommended_time', 'half_day')
            if request.available_time_hours >= 6 and recommended_time == 'full_day':
                priority += 1.0
            elif request.available_time_hours < 4 and recommended_time == 'full_day':
                priority -= 1.0
            
            # Walking preference
            if cluster_info.get('walking_friendly', False):
                priority += 0.5
            
            # Number of attractions in cluster
            priority += len(attractions) * 0.1
            
            priorities[cluster_name] = priority
        
        return priorities
    
    def _consider_ferry_routes(self, attractions: List, request: RouteRequest) -> List:
        """Integrate ferry routes for cross-Bosphorus travel"""
        ferry_routes = IstanbulOptimizations.get_ferry_routes()
        
        # Check if route crosses Bosphorus (European to Asian side or vice versa)
        european_attractions = []
        asian_attractions = []
        
        for attraction in attractions:
            # Simple longitude check (Bosphorus is roughly at 29.0)
            if attraction.coordinates_lng < 29.0:
                european_attractions.append(attraction)
            else:
                asian_attractions.append(attraction)
        
        # If attractions on both sides, add ferry route bonuses
        if european_attractions and asian_attractions:
            # Add ferry integration bonus to all attractions
            for attraction in attractions:
                ferry_bonus = 0.2  # Small bonus for cross-Bosphorus routes
                if hasattr(attraction, 'route_score'):
                    attraction.route_score += ferry_bonus
                
                attraction.ferry_accessible = True
                attraction.cross_bosphorus_route = True
                
                # Add ferry route suggestions
                best_ferry_route = max(ferry_routes.items(), key=lambda x: x[1]['scenic_value'])
                attraction.recommended_ferry = {
                    "name": best_ferry_route[0],
                    "info": best_ferry_route[1]
                }
        
        return attractions
    
    def _find_optimal_ferry_route(self, european_attractions: List, asian_attractions: List, ferry_routes: Dict) -> Optional[Dict]:
        """Find the best ferry route for cross-Bosphorus travel (simplified)"""
        # Return the most scenic ferry route for cross-Bosphorus travel
        if not ferry_routes:
            return None
        
        best_route = max(ferry_routes.items(), key=lambda x: x[1]['scenic_value'])
        return {
            "name": best_route[0],
            "info": best_route[1]
        }
    
    def _apply_weather_filtering(self, attractions: List, request: RouteRequest, weather_condition: str = "good") -> List:
        """Apply weather-aware filtering to attractions"""
        # In production, this would integrate with a weather API
        # For now, using a parameter to simulate weather conditions
        
        if weather_condition in ["rain", "snow", "storm"]:
            weather_alternatives = IstanbulOptimizations.get_weather_alternatives()
            
            for attraction in attractions:
                # Boost indoor attractions during bad weather
                is_indoor = any(
                    attraction.name in alternatives 
                    for alternatives in weather_alternatives.values()
                )
                
                if is_indoor:
                    weather_bonus = 0.5
                    if hasattr(attraction, 'route_score'):
                        attraction.route_score += weather_bonus
                    attraction.weather_suitable = True
                else:
                    # Reduce score for outdoor attractions
                    weather_penalty = -0.3
                    if hasattr(attraction, 'route_score'):
                        attraction.route_score += weather_penalty
                    attraction.weather_suitable = False
                    attraction.weather_warning = f"Outdoor attraction - not ideal in {weather_condition}"
        
        return attractions
    
    def get_istanbul_route_recommendations(self, request: RouteRequest) -> Dict[str, Any]:
        """Get Istanbul-specific route recommendations"""
        current_time = datetime.now(ISTANBUL_TIMEZONE)
        clusters = IstanbulOptimizations.get_district_clusters()
        
        recommendations = {
            "time_considerations": {
                "current_time": current_time.strftime("%H:%M"),
                "prayer_time_active": IstanbulOptimizations.is_prayer_time(current_time.time()),
                "rush_hour_active": IstanbulOptimizations.is_rush_hour(current_time.time()),
                "next_prayer_time": self._get_next_prayer_time(current_time)
            },
            "recommended_clusters": {},
            "ferry_opportunities": [],
            "cultural_tips": []
        }
        
        # Cluster recommendations
        for cluster_name, cluster_info in clusters.items():
            distance_from_start = geodesic(
                (request.start_lat, request.start_lng), 
                cluster_info['center']
            ).kilometers
            
            if distance_from_start <= request.max_distance_km:
                recommendations["recommended_clusters"][cluster_name] = {
                    "distance_km": round(distance_from_start, 1),
                    "recommended_time": cluster_info['recommended_time'],
                    "walking_friendly": cluster_info['walking_friendly'],
                    "main_attractions": cluster_info['main_attractions']
                }
        
        # Ferry recommendations
        ferry_routes = IstanbulOptimizations.get_ferry_routes()
        for route_name, route_info in ferry_routes.items():
            start_distance = geodesic(
                (request.start_lat, request.start_lng),
                (route_info['start']['lat'], route_info['start']['lng'])
            ).kilometers
            
            if start_distance <= request.max_distance_km:
                recommendations["ferry_opportunities"].append({
                    "route": route_name,
                    "start": route_info['start']['name'],
                    "end": route_info['end']['name'],
                    "duration_minutes": route_info['duration_minutes'],
                    "scenic_score": route_info['scenic_score'],
                    "distance_from_start": round(start_distance, 1)
                })
        
        # Cultural tips
        recommendations["cultural_tips"] = [
            "Remove shoes when entering mosques",
            "Dress modestly when visiting religious sites",
            "Try Turkish tea or coffee at local cafes",
            "Bargain is expected at bazaars and markets",
            "Ferries offer spectacular Bosphorus views",
            "Avoid visiting mosques during prayer times",
            "Tipping 10-15% is customary at restaurants"
        ]
        
        return recommendations
    
    def _get_next_prayer_time(self, current_time: datetime) -> str:
        """Get the next prayer time"""
        prayer_times = IstanbulOptimizations.get_prayer_times(current_time)
        current_time_only = current_time.time()
        
        for prayer_name, prayer_time in prayer_times.items():
            if prayer_time > current_time_only:
                return f"{prayer_name}: {prayer_time.strftime('%H:%M')}"
        
        # If no prayer time remaining today, return first prayer of next day
        return f"fajr (tomorrow): {prayer_times['fajr'].strftime('%H:%M')}"
    
    def get_istanbul_optimization_status(self, current_time: datetime = None) -> Dict[str, Any]:
        """Get comprehensive status of Istanbul-specific optimizations"""
        if not current_time:
            current_time = datetime.now(ISTANBUL_TIMEZONE)
        
        clusters = IstanbulOptimizations.get_district_clusters()
        ferry_routes = IstanbulOptimizations.get_ferry_routes()
        prayer_times = IstanbulOptimizations.get_prayer_times(current_time)
        
        return {
            "time_context": {
                "istanbul_time": current_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "prayer_time_active": IstanbulOptimizations.is_prayer_time(current_time.time()),
                "rush_hour_active": IstanbulOptimizations.is_rush_hour(current_time.time()),
                "prayer_times_today": {k: v.strftime("%H:%M") for k, v in prayer_times.items()}
            },
            "district_clusters": {
                "total_clusters": len(clusters),
                "cluster_names": list(clusters.keys()),
                "districts_covered": sum(len(cluster['districts']) for cluster in clusters.values())
            },
            "ferry_integration": {
                "total_routes": len(ferry_routes),
                "route_names": list(ferry_routes.keys()),
                "average_duration": sum(route['duration_minutes'] for route in ferry_routes.values()) / len(ferry_routes)
            },
            "weather_optimization": {
                "supported_conditions": list(IstanbulOptimizations.get_weather_alternatives().keys()),
                "total_alternative_categories": sum(len(alts) for alts in IstanbulOptimizations.get_weather_alternatives().values())
            },
            "graph_status": self.get_district_status()
        }

# Singleton instance - lazy loaded to prevent multiple OSM downloads
_route_maker_instance = None

def get_route_maker():
    """Get singleton instance of IstanbulRoutemaker to prevent multiple OSM downloads"""
    global _route_maker_instance
    if _route_maker_instance is None:
        print("🗺️ Initializing IstanbulRoutemaker singleton...")
        _route_maker_instance = IstanbulRoutemaker()
    return _route_maker_instance

# For backward compatibility
route_maker = None  # Will be lazy-loaded when accessed
