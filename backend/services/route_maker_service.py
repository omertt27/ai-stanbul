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
from typing import List, Dict, Tuple, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import json
from datetime import datetime, timedelta
import math
import random
from dataclasses import dataclass
from enum import Enum

from database import get_db
from services.performance_monitor import monitor_performance, profile_route_generation, performance_monitor

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
    
    def __init__(self):
        self.graph = None
        self.istanbul_bounds = {
            'north': 41.25,
            'south': 40.80,
            'east': 29.40,
            'west': 28.60
        }
        self._load_istanbul_graph()
    
    def _load_istanbul_graph(self):
        """Load Istanbul street network from OSM - using district selection instead of merging"""
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
        
        # Enhanced attraction selection with iterative scoring
        selected_attractions = self._select_optimal_attractions(attractions, request)
        
        print(f"⭐ Selected {len(selected_attractions)} attractions")
        
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

# Global instance
route_maker = IstanbulRoutemaker()
