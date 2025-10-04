"""
Advanced Route Planning Service
Implements Dijkstra and A* algorithms for optimal route planning in Istanbul
without using GPT. Uses real geographic data and transport networks.
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class Node:
    """Graph node representing a location or transport stop"""
    id: str
    name: str
    lat: float
    lon: float
    node_type: str  # 'attraction', 'metro_station', 'bus_stop', 'ferry_terminal'
    connections: List[str]

@dataclass
class Edge:
    """Graph edge representing a connection between nodes"""
    from_node: str
    to_node: str
    transport_type: str
    distance_km: float
    duration_minutes: int
    cost: float
    frequency_minutes: int
    operating_hours: Tuple[str, str]

@dataclass
class RouteSegment:
    """A segment of a route"""
    from_location: str
    to_location: str
    transport_type: str
    duration_minutes: int
    distance_km: float
    cost: float
    instructions: List[str]
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None

@dataclass
class OptimalRoute:
    """Complete optimal route with multiple segments"""
    origin: str
    destination: str
    segments: List[RouteSegment]
    total_duration: int
    total_distance: float
    total_cost: float
    total_walking_distance: float
    route_score: float

class AdvancedRoutePlanner:
    """
    Advanced route planning using Dijkstra and A* algorithms
    Handles multi-modal transport optimization in Istanbul
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency_list: Dict[str, List[Edge]] = {}
        
        # Load Istanbul transport network
        self._build_transport_network()
        
    def _build_transport_network(self):
        """Build comprehensive Istanbul transport network graph"""
        
        # Major locations and transport hubs
        locations = {
            # Major attractions
            "sultanahmet": Node("sultanahmet", "Sultanahmet", 41.0082, 28.9784, "attraction", []),
            "hagia_sophia": Node("hagia_sophia", "Hagia Sophia", 41.0086, 28.9802, "attraction", []),
            "topkapi": Node("topkapi", "Topkapi Palace", 41.0115, 28.9833, "attraction", []),
            "galata_tower": Node("galata_tower", "Galata Tower", 41.0256, 28.9741, "attraction", []),
            "taksim": Node("taksim", "Taksim Square", 41.0369, 28.9850, "attraction", []),
            
            # Metro stations
            "vezneciler_metro": Node("vezneciler_metro", "Vezneciler Metro", 41.0100, 28.9750, "metro_station", []),
            "taksim_metro": Node("taksim_metro", "Taksim Metro", 41.0369, 28.9850, "metro_station", []),
            "sisli_metro": Node("sisli_metro", "Şişli-Mecidiyeköy Metro", 41.0608, 29.0011, "metro_station", []),
            "levent_metro": Node("levent_metro", "Levent Metro", 41.0775, 29.0106, "metro_station", []),
            "besiktas_metro": Node("besiktas_metro", "Beşiktaş Metro", 41.0422, 29.0067, "metro_station", []),
            
            # Ferry terminals
            "eminonu_ferry": Node("eminonu_ferry", "Eminönü Ferry Terminal", 41.0175, 28.9700, "ferry_terminal", []),
            "karakoy_ferry": Node("karakoy_ferry", "Karaköy Ferry Terminal", 41.0256, 28.9741, "ferry_terminal", []),
            "besiktas_ferry": Node("besiktas_ferry", "Beşiktaş Ferry Terminal", 41.0422, 29.0067, "ferry_terminal", []),
            "uskudar_ferry": Node("uskudar_ferry", "Üsküdar Ferry Terminal", 40.9923, 29.0243, "ferry_terminal", []),
            "kadikoy_ferry": Node("kadikoy_ferry", "Kadıköy Ferry Terminal", 40.9918, 29.0253, "ferry_terminal", []),
            
            # Bus stops and major transport nodes
            "sultanahmet_bus": Node("sultanahmet_bus", "Sultanahmet Bus Stop", 41.0070, 28.9780, "bus_stop", []),
            "taksim_bus": Node("taksim_bus", "Taksim Bus Terminal", 41.0365, 28.9845, "bus_stop", []),
            "eminonu_bus": Node("eminonu_bus", "Eminönü Bus Station", 41.0170, 28.9705, "bus_stop", []),
        }
        
        self.nodes = locations
        
        # Build transport connections
        self._add_metro_connections()
        self._add_ferry_connections()
        self._add_bus_connections()
        self._add_walking_connections()
        
        # Build adjacency list for faster pathfinding
        self._build_adjacency_list()
    
    def _add_metro_connections(self):
        """Add metro line connections"""
        metro_edges = [
            # M2 Line (Green Line)
            Edge("vezneciler_metro", "taksim_metro", "metro", 5.2, 12, 15, 3, ("06:00", "24:00")),
            Edge("taksim_metro", "sisli_metro", "metro", 3.8, 8, 15, 3, ("06:00", "24:00")),
            Edge("sisli_metro", "levent_metro", "metro", 4.1, 9, 15, 3, ("06:00", "24:00")),
            Edge("levent_metro", "besiktas_metro", "metro", 2.7, 6, 15, 4, ("06:00", "24:00")),
            
            # Reverse connections
            Edge("taksim_metro", "vezneciler_metro", "metro", 5.2, 12, 15, 3, ("06:00", "24:00")),
            Edge("sisli_metro", "taksim_metro", "metro", 3.8, 8, 15, 3, ("06:00", "24:00")),
            Edge("levent_metro", "sisli_metro", "metro", 4.1, 9, 15, 3, ("06:00", "24:00")),
            Edge("besiktas_metro", "levent_metro", "metro", 2.7, 6, 15, 4, ("06:00", "24:00")),
        ]
        
        self.edges.extend(metro_edges)
    
    def _add_ferry_connections(self):
        """Add ferry route connections"""
        ferry_edges = [
            # Bosphorus ferries
            Edge("eminonu_ferry", "karakoy_ferry", "ferry", 1.2, 8, 15, 20, ("07:00", "21:00")),
            Edge("karakoy_ferry", "besiktas_ferry", "ferry", 2.1, 12, 15, 20, ("07:00", "21:00")),
            Edge("besiktas_ferry", "uskudar_ferry", "ferry", 1.8, 10, 15, 20, ("07:00", "21:00")),
            
            # Cross-Bosphorus connections
            Edge("eminonu_ferry", "uskudar_ferry", "ferry", 2.8, 15, 15, 30, ("07:00", "21:00")),
            Edge("eminonu_ferry", "kadikoy_ferry", "ferry", 3.2, 20, 15, 30, ("07:00", "21:00")),
            Edge("besiktas_ferry", "uskudar_ferry", "ferry", 1.5, 8, 15, 20, ("07:00", "21:00")),
            
            # Reverse connections
            Edge("karakoy_ferry", "eminonu_ferry", "ferry", 1.2, 8, 15, 20, ("07:00", "21:00")),
            Edge("besiktas_ferry", "karakoy_ferry", "ferry", 2.1, 12, 15, 20, ("07:00", "21:00")),
            Edge("uskudar_ferry", "besiktas_ferry", "ferry", 1.8, 10, 15, 20, ("07:00", "21:00")),
            Edge("uskudar_ferry", "eminonu_ferry", "ferry", 2.8, 15, 15, 30, ("07:00", "21:00")),
            Edge("kadikoy_ferry", "eminonu_ferry", "ferry", 3.2, 20, 15, 30, ("07:00", "21:00")),
            Edge("uskudar_ferry", "besiktas_ferry", "ferry", 1.5, 8, 15, 20, ("07:00", "21:00")),
        ]
        
        self.edges.extend(ferry_edges)
    
    def _add_bus_connections(self):
        """Add bus route connections"""
        bus_edges = [
            # Major bus routes
            Edge("sultanahmet_bus", "eminonu_bus", "bus", 1.5, 8, 10, 10, ("06:00", "23:00")),
            Edge("eminonu_bus", "taksim_bus", "bus", 3.2, 15, 10, 12, ("06:00", "23:00")),
            Edge("taksim_bus", "besiktas_ferry", "bus", 2.8, 12, 10, 10, ("06:00", "23:00")),
            
            # Reverse connections
            Edge("eminonu_bus", "sultanahmet_bus", "bus", 1.5, 8, 10, 10, ("06:00", "23:00")),
            Edge("taksim_bus", "eminonu_bus", "bus", 3.2, 15, 10, 12, ("06:00", "23:00")),
            Edge("besiktas_ferry", "taksim_bus", "bus", 2.8, 12, 10, 10, ("06:00", "23:00")),
        ]
        
        self.edges.extend(bus_edges)
    
    def _add_walking_connections(self):
        """Add walking connections between nearby locations"""
        walking_edges = [
            # Sultanahmet area walking
            Edge("sultanahmet", "hagia_sophia", "walking", 0.3, 4, 0, 0, ("00:00", "23:59")),
            Edge("hagia_sophia", "topkapi", "walking", 0.8, 10, 0, 0, ("00:00", "23:59")),
            Edge("sultanahmet", "sultanahmet_bus", "walking", 0.2, 3, 0, 0, ("00:00", "23:59")),
            Edge("sultanahmet", "vezneciler_metro", "walking", 1.2, 15, 0, 0, ("00:00", "23:59")),
            
            # Taksim area walking
            Edge("taksim", "taksim_metro", "walking", 0.1, 2, 0, 0, ("00:00", "23:59")),
            Edge("taksim", "taksim_bus", "walking", 0.15, 2, 0, 0, ("00:00", "23:59")),
            
            # Galata area walking
            Edge("galata_tower", "karakoy_ferry", "walking", 0.4, 5, 0, 0, ("00:00", "23:59")),
            
            # Ferry terminal connections
            Edge("eminonu_ferry", "eminonu_bus", "walking", 0.1, 2, 0, 0, ("00:00", "23:59")),
            Edge("besiktas_ferry", "besiktas_metro", "walking", 0.3, 4, 0, 0, ("00:00", "23:59")),
            
            # Reverse walking connections
            Edge("hagia_sophia", "sultanahmet", "walking", 0.3, 4, 0, 0, ("00:00", "23:59")),
            Edge("topkapi", "hagia_sophia", "walking", 0.8, 10, 0, 0, ("00:00", "23:59")),
            Edge("sultanahmet_bus", "sultanahmet", "walking", 0.2, 3, 0, 0, ("00:00", "23:59")),
            Edge("vezneciler_metro", "sultanahmet", "walking", 1.2, 15, 0, 0, ("00:00", "23:59")),
            Edge("taksim_metro", "taksim", "walking", 0.1, 2, 0, 0, ("00:00", "23:59")),
            Edge("taksim_bus", "taksim", "walking", 0.15, 2, 0, 0, ("00:00", "23:59")),
            Edge("karakoy_ferry", "galata_tower", "walking", 0.4, 5, 0, 0, ("00:00", "23:59")),
            Edge("eminonu_bus", "eminonu_ferry", "walking", 0.1, 2, 0, 0, ("00:00", "23:59")),
            Edge("besiktas_metro", "besiktas_ferry", "walking", 0.3, 4, 0, 0, ("00:00", "23:59")),
        ]
        
        self.edges.extend(walking_edges)
    
    def _build_adjacency_list(self):
        """Build adjacency list for efficient graph traversal"""
        self.adjacency_list = {node_id: [] for node_id in self.nodes.keys()}
        
        for edge in self.edges:
            self.adjacency_list[edge.from_node].append(edge)
    
    def dijkstra_shortest_path(self, start: str, end: str, 
                              current_time: Optional[datetime] = None) -> Optional[OptimalRoute]:
        """
        Find shortest path using Dijkstra's algorithm
        Optimizes for total travel time including waiting
        """
        if start not in self.nodes or end not in self.nodes:
            return None
        
        if current_time is None:
            current_time = datetime.now()
        
        # Priority queue: (total_time, current_node, path, total_cost, total_distance)
        pq = [(0, start, [], 0, 0)]
        visited = set()
        
        while pq:
            current_time_cost, current_node, path, total_cost, total_distance = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            if current_node == end:
                return self._construct_route(start, end, path, total_cost, total_distance, current_time_cost)
            
            # Explore neighbors
            for edge in self.adjacency_list.get(current_node, []):
                if edge.to_node not in visited:
                    # Calculate wait time and total time for this edge
                    wait_time = self._calculate_wait_time(edge, current_time)
                    edge_total_time = wait_time + edge.duration_minutes
                    
                    new_total_time = current_time_cost + edge_total_time
                    new_total_cost = total_cost + edge.cost
                    new_total_distance = total_distance + edge.distance_km
                    new_path = path + [edge]
                    
                    heapq.heappush(pq, (new_total_time, edge.to_node, new_path, new_total_cost, new_total_distance))
        
        return None
    
    def a_star_shortest_path(self, start: str, end: str, 
                           current_time: Optional[datetime] = None) -> Optional[OptimalRoute]:
        """
        Find shortest path using A* algorithm
        Uses haversine distance as heuristic function
        """
        if start not in self.nodes or end not in self.nodes:
            return None
        
        if current_time is None:
            current_time = datetime.now()
        
        start_node = self.nodes[start]
        end_node = self.nodes[end]
        
        # Priority queue: (f_score, g_score, current_node, path, total_cost, total_distance)
        pq = [(0, 0, start, [], 0, 0)]
        visited = set()
        g_scores = {start: 0}
        
        while pq:
            f_score, g_score, current_node, path, total_cost, total_distance = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            if current_node == end:
                return self._construct_route(start, end, path, total_cost, total_distance, g_score)
            
            current_node_obj = self.nodes[current_node]
            
            # Explore neighbors
            for edge in self.adjacency_list.get(current_node, []):
                if edge.to_node not in visited:
                    # Calculate wait time and total time for this edge
                    wait_time = self._calculate_wait_time(edge, current_time)
                    edge_total_time = wait_time + edge.duration_minutes
                    
                    tentative_g_score = g_score + edge_total_time
                    
                    if edge.to_node not in g_scores or tentative_g_score < g_scores[edge.to_node]:
                        g_scores[edge.to_node] = tentative_g_score
                        
                        # Calculate heuristic (straight-line distance time estimate)
                        to_node_obj = self.nodes[edge.to_node]
                        h_score = self._heuristic_time_estimate(to_node_obj, end_node)
                        f_score = tentative_g_score + h_score
                        
                        new_total_cost = total_cost + edge.cost
                        new_total_distance = total_distance + edge.distance_km
                        new_path = path + [edge]
                        
                        heapq.heappush(pq, (f_score, tentative_g_score, edge.to_node, 
                                          new_path, new_total_cost, new_total_distance))
        
        return None
    
    def _calculate_wait_time(self, edge: Edge, current_time: datetime) -> int:
        """Calculate waiting time for public transport"""
        if edge.transport_type == "walking":
            return 0
        
        # Check if transport is operating
        start_hour, start_min = map(int, edge.operating_hours[0].split(':'))
        end_hour, end_min = map(int, edge.operating_hours[1].split(':'))
        
        current_hour = current_time.hour
        current_min = current_time.minute
        
        # If outside operating hours, return high penalty
        if current_hour < start_hour or (current_hour == start_hour and current_min < start_min):
            return 60  # 1 hour penalty
        if current_hour > end_hour or (current_hour == end_hour and current_min > end_min):
            return 60  # 1 hour penalty
        
        # Average wait time is half the frequency
        return edge.frequency_minutes // 2
    
    def _heuristic_time_estimate(self, from_node: Node, to_node: Node) -> float:
        """Estimate travel time using haversine distance"""
        distance_km = self._haversine_distance(from_node.lat, from_node.lon, to_node.lat, to_node.lon)
        
        # Assume average speed of 30 km/h for transport, 5 km/h for walking
        estimated_time_minutes = (distance_km / 30) * 60
        return estimated_time_minutes
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _construct_route(self, start: str, end: str, path: List[Edge], 
                        total_cost: float, total_distance: float, total_time: float) -> OptimalRoute:
        """Construct OptimalRoute object from path"""
        segments = []
        current_time = datetime.now()
        walking_distance = 0
        
        for edge in path:
            from_name = self.nodes[edge.from_node].name
            to_name = self.nodes[edge.to_node].name
            
            # Calculate departure and arrival times
            wait_time = self._calculate_wait_time(edge, current_time)
            departure_time = current_time + timedelta(minutes=wait_time)
            arrival_time = departure_time + timedelta(minutes=edge.duration_minutes)
            
            # Create instructions
            instructions = self._generate_segment_instructions(edge, from_name, to_name)
            
            segment = RouteSegment(
                from_location=from_name,
                to_location=to_name,
                transport_type=edge.transport_type,
                duration_minutes=edge.duration_minutes + wait_time,
                distance_km=edge.distance_km,
                cost=edge.cost,
                instructions=instructions,
                departure_time=departure_time.strftime("%H:%M"),
                arrival_time=arrival_time.strftime("%H:%M")
            )
            
            segments.append(segment)
            current_time = arrival_time
            
            if edge.transport_type == "walking":
                walking_distance += edge.distance_km
        
        # Calculate route score (lower is better)
        route_score = self._calculate_route_score(total_time, total_cost, walking_distance, len(segments))
        
        return OptimalRoute(
            origin=self.nodes[start].name,
            destination=self.nodes[end].name,
            segments=segments,
            total_duration=int(total_time),
            total_distance=round(total_distance, 2),
            total_cost=round(total_cost, 2),
            total_walking_distance=round(walking_distance, 2),
            route_score=route_score
        )
    
    def _generate_segment_instructions(self, edge: Edge, from_name: str, to_name: str) -> List[str]:
        """Generate human-readable instructions for a route segment"""
        instructions = []
        
        if edge.transport_type == "walking":
            instructions.append(f"🚶 Walk from {from_name} to {to_name}")
            instructions.append(f"   Distance: {edge.distance_km:.1f} km (~{edge.duration_minutes} minutes)")
        elif edge.transport_type == "metro":
            instructions.append(f"🚇 Take Metro from {from_name} to {to_name}")
            instructions.append(f"   Journey time: {edge.duration_minutes} minutes")
            instructions.append(f"   Frequency: Every {edge.frequency_minutes} minutes")
        elif edge.transport_type == "ferry":
            instructions.append(f"⛴️ Take Ferry from {from_name} to {to_name}")
            instructions.append(f"   Journey time: {edge.duration_minutes} minutes")
            instructions.append(f"   Frequency: Every {edge.frequency_minutes} minutes")
        elif edge.transport_type == "bus":
            instructions.append(f"🚌 Take Bus from {from_name} to {to_name}")
            instructions.append(f"   Journey time: {edge.duration_minutes} minutes")
            instructions.append(f"   Frequency: Every {edge.frequency_minutes} minutes")
        
        return instructions
    
    def _calculate_route_score(self, total_time: float, total_cost: float, 
                             walking_distance: float, num_transfers: int) -> float:
        """Calculate route quality score (lower is better)"""
        # Weighted scoring factors
        time_weight = 1.0
        cost_weight = 0.5
        walking_weight = 2.0
        transfer_penalty = 10.0
        
        score = (total_time * time_weight + 
                total_cost * cost_weight + 
                walking_distance * 1000 * walking_weight +  # Convert km to m
                num_transfers * transfer_penalty)
        
        return round(score, 2)
    
    def find_optimal_route(self, start: str, end: str, 
                          algorithm: str = "a_star",
                          current_time: Optional[datetime] = None) -> Optional[OptimalRoute]:
        """
        Find optimal route using specified algorithm
        
        Args:
            start: Starting location name or node ID
            end: Destination location name or node ID
            algorithm: "dijkstra" or "a_star"
            current_time: Current time for scheduling
            
        Returns:
            OptimalRoute object or None if no route found
        """
        # Map location names to node IDs if needed
        start_node = self._find_node_by_name(start)
        end_node = self._find_node_by_name(end)
        
        if not start_node or not end_node:
            return None
        
        if algorithm == "dijkstra":
            return self.dijkstra_shortest_path(start_node, end_node, current_time)
        elif algorithm == "a_star":
            return self.a_star_shortest_path(start_node, end_node, current_time)
        else:
            # Default to A*
            return self.a_star_shortest_path(start_node, end_node, current_time)
    
    def _find_node_by_name(self, location_name: str) -> Optional[str]:
        """Find node ID by location name (fuzzy matching)"""
        location_lower = location_name.lower()
        
        # Direct node ID match
        if location_lower in self.nodes:
            return location_lower
        
        # Name matching
        for node_id, node in self.nodes.items():
            if location_lower in node.name.lower() or node.name.lower() in location_lower:
                return node_id
        
        # Fuzzy matching for common names
        name_mappings = {
            "sultanahmet": "sultanahmet",
            "blue mosque": "sultanahmet",
            "hagia sophia": "hagia_sophia",
            "ayasofya": "hagia_sophia",
            "topkapi": "topkapi",
            "topkapi palace": "topkapi",
            "galata tower": "galata_tower",
            "galata": "galata_tower",
            "taksim": "taksim",
            "taksim square": "taksim",
            "eminonu": "eminonu_ferry",
            "eminönü": "eminonu_ferry",
            "karakoy": "karakoy_ferry",
            "karaköy": "karakoy_ferry",
            "besiktas": "besiktas_ferry",
            "beşiktaş": "besiktas_ferry",
            "uskudar": "uskudar_ferry",
            "üsküdar": "uskudar_ferry",
            "kadikoy": "kadikoy_ferry",
            "kadıköy": "kadikoy_ferry"
        }
        
        return name_mappings.get(location_lower)
    
    def get_route_alternatives(self, start: str, end: str, 
                             current_time: Optional[datetime] = None) -> List[OptimalRoute]:
        """Get multiple route alternatives using different algorithms and preferences"""
        routes = []
        
        # Try both algorithms
        dijkstra_route = self.find_optimal_route(start, end, "dijkstra", current_time)
        if dijkstra_route:
            routes.append(dijkstra_route)
        
        a_star_route = self.find_optimal_route(start, end, "a_star", current_time)
        if a_star_route and a_star_route.route_score != dijkstra_route.route_score:
            routes.append(a_star_route)
        
        # Sort by route score (best first)
        routes.sort(key=lambda r: r.route_score)
        
        return routes[:3]  # Return top 3 alternatives

# Example usage and testing
if __name__ == "__main__":
    planner = AdvancedRoutePlanner()
    
    # Test route planning
    route = planner.find_optimal_route("sultanahmet", "taksim", "a_star")
    if route:
        print(f"Route from {route.origin} to {route.destination}")
        print(f"Total time: {route.total_duration} minutes")
        print(f"Total cost: ₺{route.total_cost}")
        print(f"Walking distance: {route.total_walking_distance} km")
        print(f"Route score: {route.route_score}")
        print("\nSegments:")
        for i, segment in enumerate(route.segments, 1):
            print(f"{i}. {segment.transport_type.title()}: {segment.from_location} → {segment.to_location}")
            print(f"   Time: {segment.duration_minutes} min, Cost: ₺{segment.cost}")
