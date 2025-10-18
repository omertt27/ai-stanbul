#!/usr/bin/env python3
"""
Transport Graph Service for Istanbul AI
========================================

Unified transport graph builder that integrates:
- Transit network (Metro, Tram, Bus, Ferry)
- POI nodes (Museums, Attractions, Landmarks)
- Walking connections
- Multi-objective optimization (time, cost, scenic value, crowding)

Phase 2.1: Build Unified Transport Graph (Week 2, 3 days)
"""

import json
import math
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import heapq

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import POI Database Service
try:
    from services.poi_database_service import POIDatabaseService, POI, GeoCoordinate
    POI_DATABASE_AVAILABLE = True
except ImportError:
    POI_DATABASE_AVAILABLE = False
    logging.warning("POI Database Service not available")

# Import ML-Enhanced Transportation System
try:
    from ml_enhanced_transportation_system import (
        MLEnhancedTransportationSystem,
        TransportMode,
        GPSLocation
    )
    ML_TRANSPORT_AVAILABLE = True
except ImportError:
    ML_TRANSPORT_AVAILABLE = False
    logging.warning("ML-Enhanced Transportation System not available")

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the transport graph"""
    STATION = "station"
    POI = "poi"
    TRANSFER = "transfer"
    LANDMARK = "landmark"


class EdgeType(Enum):
    """Types of edges in the transport graph"""
    METRO = "metro"
    TRAM = "tram"
    FERRY = "ferry"
    BUS = "bus"
    WALK = "walk"
    FUNICULAR = "funicular"
    CABLE_CAR = "cable_car"
    MARMARAY = "marmaray"


@dataclass
class GraphNode:
    """
    Node in the transport graph
    Represents stations, POIs, transfer points, or landmarks
    """
    node_id: str
    node_type: NodeType
    location: GeoCoordinate
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.node_id == other.node_id


@dataclass
class GraphEdge:
    """
    Edge in the transport graph
    Represents connections between nodes (transit or walking)
    """
    from_node: str
    to_node: str
    edge_type: EdgeType
    distance_km: float
    time_minutes: float
    cost: float
    scenic_score: float = 0.5
    crowding_factor: float = 1.0  # 1.0 = normal, 1.5 = crowded, 2.0 = very crowded
    line_id: Optional[str] = None  # e.g., "M2", "T1", "F1"
    accessibility_score: float = 1.0
    
    def __hash__(self):
        return hash((self.from_node, self.to_node, self.edge_type.value))


@dataclass
class PathConstraints:
    """Constraints for pathfinding"""
    max_time_minutes: Optional[int] = None
    max_cost: Optional[float] = None
    max_transfers: Optional[int] = None
    min_accessibility: float = 0.0
    avoid_crowded: bool = False
    prefer_scenic: bool = False
    allowed_transport_modes: Optional[Set[EdgeType]] = None


@dataclass
class PathResult:
    """Result from pathfinding algorithm"""
    path: List[GraphNode]
    edges: List[GraphEdge]
    total_distance_km: float
    total_time_minutes: float
    total_cost: float
    scenic_score: float
    crowding_score: float
    num_transfers: int
    optimization_score: float


class TransportGraph:
    """
    Unified transport graph for Istanbul
    
    Features:
    - Graph representation of transit network
    - Integration with POI database
    - A* pathfinding with multi-objective optimization
    - Constraint-based routing
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[GraphEdge]] = {}  # adjacency list: node_id -> [edges]
        self.reverse_edges: Dict[str, List[GraphEdge]] = {}  # for bidirectional search
        
        # Services
        self.poi_service: Optional[POIDatabaseService] = None
        self.ml_transport: Optional[Any] = None
        
        logger.info("🗺️ Transport Graph initialized")
    
    def build_from_services(
        self,
        poi_service: Optional[POIDatabaseService] = None,
        ml_transport_system: Optional[Any] = None
    ):
        """
        Build graph from POI database and transport system
        
        Args:
            poi_service: POI database service for attraction nodes
            ml_transport_system: ML-Enhanced transportation system for transit data
        """
        logger.info("🏗️ Building transport graph from services...")
        
        self.poi_service = poi_service
        self.ml_transport = ml_transport_system
        
        # Build station nodes
        self._add_station_nodes()
        
        # Build POI nodes
        if self.poi_service:
            self._add_poi_nodes()
        
        # Build transit edges
        self._add_transit_edges()
        
        # Build walking edges (station <-> POI, POI <-> POI, station <-> station)
        self._add_walking_edges()
        
        logger.info(f"✅ Graph built: {len(self.nodes)} nodes, {sum(len(e) for e in self.edges.values())} edges")
    
    def _add_station_nodes(self):
        """Add transit station nodes from ML transport system or static data"""
        # Istanbul Metro stations (M1-M11 lines)
        metro_stations = [
            # M2 Line (Green Line)
            ("M2_Yenikapi", "Yenikapı", 41.0080, 28.9514, ["M2", "Marmaray"]),
            ("M2_Vezneciler", "Vezneciler", 41.0133, 28.9597, ["M2"]),
            ("M2_Haliç", "Haliç", 41.0207, 28.9628, ["M2"]),
            ("M2_Şişhane", "Şişhane", 41.0286, 28.9726, ["M2"]),
            ("M2_Taksim", "Taksim", 41.0370, 28.9857, ["M2"]),
            
            # M1 Line (Red Line)
            ("M1_Aksaray", "Aksaray", 41.0152, 28.9495, ["M1"]),
            ("M1_Emniyet-Fatih", "Emniyet-Fatih", 41.0195, 28.9442, ["M1"]),
            
            # Tram T1 (Historic Peninsula)
            ("T1_Sultanahmet", "Sultanahmet", 41.0056, 28.9769, ["T1"]),
            ("T1_Gülhane", "Gülhane", 41.0125, 28.9813, ["T1"]),
            ("T1_Eminönü", "Eminönü", 41.0175, 28.9700, ["T1"]),
            ("T1_Karaköy", "Karaköy", 41.0243, 28.9748, ["T1"]),
            ("T1_Kabataş", "Kabataş", 41.0389, 29.0073, ["T1", "Funicular"]),
            
            # Ferry terminals
            ("F_Eminönü", "Eminönü Ferry", 41.0175, 28.9700, ["Ferry"]),
            ("F_Karaköy", "Karaköy Ferry", 41.0243, 28.9748, ["Ferry"]),
            ("F_Beşiktaş", "Beşiktaş Ferry", 41.0422, 29.0084, ["Ferry"]),
            ("F_Kadıköy", "Kadıköy Ferry", 40.9907, 29.0205, ["Ferry"]),
            ("F_Üsküdar", "Üsküdar Ferry", 41.0215, 29.0074, ["Ferry"]),
            
            # Funicular
            ("FUN_Kabataş", "Kabataş Funicular", 41.0389, 29.0073, ["Funicular"]),
            ("FUN_Taksim", "Taksim Funicular", 41.0370, 28.9857, ["Funicular"]),
        ]
        
        for station_id, name, lat, lon, lines in metro_stations:
            node = GraphNode(
                node_id=station_id,
                node_type=NodeType.STATION,
                location=GeoCoordinate(lat=lat, lon=lon),
                name=name,
                properties={
                    "lines": lines,
                    "station_type": lines[0] if lines else "unknown",
                    "accessibility": True
                }
            )
            self.add_node(node)
        
        logger.info(f"Added {len(metro_stations)} station nodes")
    
    def _add_poi_nodes(self):
        """Add POI nodes from POI database service"""
        if not self.poi_service:
            logger.warning("POI service not available, skipping POI nodes")
            return
        
        try:
            # Get all POIs from database - pois is a dictionary, not a list
            all_pois = self.poi_service.pois.values()  # Fixed: get values from dict
            
            added_count = 0
            for poi in all_pois:
                # POI model uses different attribute names
                categories = [poi.category]
                if hasattr(poi, 'subcategory') and poi.subcategory:
                    categories.append(poi.subcategory)
                
                node = GraphNode(
                    node_id=f"POI_{poi.poi_id}",
                    node_type=NodeType.POI,
                    location=poi.location,
                    name=poi.name,
                    properties={
                        "poi": poi,
                        "visit_duration": poi.visit_duration_min,  # Fixed attribute name
                        "categories": categories,
                        "popularity": poi.popularity_score,
                        "accessibility": poi.accessibility_score,  # Fixed attribute name
                        "district": poi.district
                    }
                )
                self.add_node(node)
                added_count += 1
            
            logger.info(f"Added {added_count} POI nodes")
            
        except Exception as e:
            logger.error(f"Failed to add POI nodes: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _add_transit_edges(self):
        """Add transit edges (metro, tram, ferry, bus connections)"""
        # Metro M2 connections (complete line)
        m2_connections = [
            ("M2_Yenikapi", "M2_Vezneciler", 3, 1.5, 7.67, "M2"),
            ("M2_Vezneciler", "M2_Haliç", 2, 1.0, 7.67, "M2"),
            ("M2_Haliç", "M2_Şişhane", 2, 0.8, 7.67, "M2"),
            ("M2_Şişhane", "M2_Taksim", 3, 1.2, 7.67, "M2"),
        ]
        
        for from_id, to_id, time_min, dist_km, cost, line in m2_connections:
            # Add bidirectional edges
            self._add_transit_edge(from_id, to_id, EdgeType.METRO, time_min, dist_km, cost, line, scenic=0.1)
            self._add_transit_edge(to_id, from_id, EdgeType.METRO, time_min, dist_km, cost, line, scenic=0.1)
        
        # Tram T1 connections (complete line)
        t1_connections = [
            ("T1_Sultanahmet", "T1_Gülhane", 2, 0.5, 7.67, "T1"),
            ("T1_Gülhane", "T1_Eminönü", 3, 1.0, 7.67, "T1"),
            ("T1_Eminönü", "T1_Karaköy", 5, 1.5, 7.67, "T1"),
            ("T1_Karaköy", "T1_Kabataş", 8, 2.5, 7.67, "T1"),
        ]
        
        for from_id, to_id, time_min, dist_km, cost, line in t1_connections:
            # Tram is more scenic (0.6)
            self._add_transit_edge(from_id, to_id, EdgeType.TRAM, time_min, dist_km, cost, line, scenic=0.6)
            self._add_transit_edge(to_id, from_id, EdgeType.TRAM, time_min, dist_km, cost, line, scenic=0.6)
        
        # Ferry connections
        ferry_connections = [
            ("F_Eminönü", "F_Kadıköy", 25, 3.5, 7.67, "F1"),
            ("F_Eminönü", "F_Üsküdar", 20, 2.8, 7.67, "F2"),
            ("F_Karaköy", "F_Kadıköy", 20, 3.2, 7.67, "F3"),
            ("F_Beşiktaş", "F_Kadıköy", 30, 4.0, 7.67, "F4"),
        ]
        
        for from_id, to_id, time_min, dist_km, cost, line in ferry_connections:
            # Ferry is very scenic (1.0)
            self._add_transit_edge(from_id, to_id, EdgeType.FERRY, time_min, dist_km, cost, line, scenic=1.0, crowding=0.3)
            self._add_transit_edge(to_id, from_id, EdgeType.FERRY, time_min, dist_km, cost, line, scenic=1.0, crowding=0.3)
        
        # Funicular connection (connects Kabataş to Taksim)
        self._add_transit_edge("FUN_Kabataş", "FUN_Taksim", EdgeType.FUNICULAR, 3, 0.6, 7.67, "F1", scenic=0.7)
        self._add_transit_edge("FUN_Taksim", "FUN_Kabataş", EdgeType.FUNICULAR, 3, 0.6, 7.67, "F1", scenic=0.7)
        
        # Transfer connections (same location, different lines)
        # Taksim: M2 <-> Funicular
        self._add_transit_edge("M2_Taksim", "FUN_Taksim", EdgeType.WALK, 2, 0.05, 0.0, "Transfer", scenic=0.5)
        self._add_transit_edge("FUN_Taksim", "M2_Taksim", EdgeType.WALK, 2, 0.05, 0.0, "Transfer", scenic=0.5)
        
        # Kabataş: T1 <-> Funicular
        self._add_transit_edge("T1_Kabataş", "FUN_Kabataş", EdgeType.WALK, 2, 0.05, 0.0, "Transfer", scenic=0.5)
        self._add_transit_edge("FUN_Kabataş", "T1_Kabataş", EdgeType.WALK, 2, 0.05, 0.0, "Transfer", scenic=0.5)
        
        # Eminönü: T1 <-> Ferry
        self._add_transit_edge("T1_Eminönü", "F_Eminönü", EdgeType.WALK, 3, 0.1, 0.0, "Transfer", scenic=0.5)
        self._add_transit_edge("F_Eminönü", "T1_Eminönü", EdgeType.WALK, 3, 0.1, 0.0, "Transfer", scenic=0.5)
        
        # Karaköy: T1 <-> Ferry
        self._add_transit_edge("T1_Karaköy", "F_Karaköy", EdgeType.WALK, 3, 0.1, 0.0, "Transfer", scenic=0.5)
        self._add_transit_edge("F_Karaköy", "T1_Karaköy", EdgeType.WALK, 3, 0.1, 0.0, "Transfer", scenic=0.5)
        
        logger.info(f"Added transit edges with transfer connections")
    
    def _add_transit_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        time_min: float,
        dist_km: float,
        cost: float,
        line: str,
        scenic: float = 0.5,
        crowding: float = 1.0
    ):
        """Helper to add a transit edge"""
        edge = GraphEdge(
            from_node=from_id,
            to_node=to_id,
            edge_type=edge_type,
            distance_km=dist_km,
            time_minutes=time_min,
            cost=cost,
            scenic_score=scenic,
            crowding_factor=crowding,
            line_id=line
        )
        self.add_edge(edge)
    
    def _add_walking_edges(self):
        """
        Add walking edges between:
        1. Stations and nearby POIs
        2. Nearby POIs
        3. Nearby stations (for transfers)
        """
        max_walking_distance_km = 1.5  # Maximum walking distance
        walking_speed_kmh = 4.5
        
        # 1. Station <-> POI connections
        station_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.STATION]
        poi_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.POI]
        
        for station in station_nodes:
            for poi in poi_nodes:
                distance_km = self._calculate_distance(station.location, poi.location)
                
                if distance_km <= max_walking_distance_km:
                    time_minutes = (distance_km / walking_speed_kmh) * 60
                    scenic = self._calculate_scenic_score(station.location, poi.location)
                    
                    # Bidirectional walking edges
                    self.add_edge(GraphEdge(
                        from_node=station.node_id,
                        to_node=poi.node_id,
                        edge_type=EdgeType.WALK,
                        distance_km=distance_km,
                        time_minutes=time_minutes,
                        cost=0.0,
                        scenic_score=scenic,
                        crowding_factor=0.0
                    ))
                    
                    self.add_edge(GraphEdge(
                        from_node=poi.node_id,
                        to_node=station.node_id,
                        edge_type=EdgeType.WALK,
                        distance_km=distance_km,
                        time_minutes=time_minutes,
                        cost=0.0,
                        scenic_score=scenic,
                        crowding_factor=0.0
                    ))
        
        # 2. POI <-> POI connections (for walking between attractions)
        for i, poi1 in enumerate(poi_nodes):
            for poi2 in poi_nodes[i+1:]:
                distance_km = self._calculate_distance(poi1.location, poi2.location)
                
                if distance_km <= max_walking_distance_km:
                    time_minutes = (distance_km / walking_speed_kmh) * 60
                    scenic = self._calculate_scenic_score(poi1.location, poi2.location)
                    
                    # Bidirectional walking edges
                    self.add_edge(GraphEdge(
                        from_node=poi1.node_id,
                        to_node=poi2.node_id,
                        edge_type=EdgeType.WALK,
                        distance_km=distance_km,
                        time_minutes=time_minutes,
                        cost=0.0,
                        scenic_score=scenic
                    ))
                    
                    self.add_edge(GraphEdge(
                        from_node=poi2.node_id,
                        to_node=poi1.node_id,
                        edge_type=EdgeType.WALK,
                        distance_km=distance_km,
                        time_minutes=time_minutes,
                        cost=0.0,
                        scenic_score=scenic
                    ))
        
        logger.info(f"Added walking edges")
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        if node.node_id not in self.edges:
            self.edges[node.node_id] = []
        if node.node_id not in self.reverse_edges:
            self.reverse_edges[node.node_id] = []
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph"""
        if edge.from_node not in self.edges:
            self.edges[edge.from_node] = []
        self.edges[edge.from_node].append(edge)
        
        # Also add to reverse edges for bidirectional search
        if edge.to_node not in self.reverse_edges:
            self.reverse_edges[edge.to_node] = []
        self.reverse_edges[edge.to_node].append(edge)
    
    def find_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        constraints: Optional[PathConstraints] = None
    ) -> Optional[PathResult]:
        """
        Find shortest path using A* algorithm with multi-objective optimization
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Ending node ID
            constraints: Optional path constraints
            
        Returns:
            PathResult with optimal path, or None if no path found
        """
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            logger.error(f"Start or end node not found in graph")
            return None
        
        if constraints is None:
            constraints = PathConstraints()
        
        logger.info(f"🔍 Finding path from {start_node_id} to {end_node_id}")
        
        # A* algorithm
        start_node = self.nodes[start_node_id]
        end_node = self.nodes[end_node_id]
        
        # Priority queue: (f_score, node_id, g_score, path, edges, transfers)
        pq = [(0, start_node_id, 0, [start_node], [], 0)]
        visited = set()
        
        best_path = None
        best_score = float('inf')
        
        while pq:
            f_score, current_id, g_score, path, edges_taken, transfers = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Goal reached
            if current_id == end_node_id:
                result = self._create_path_result(path, edges_taken, transfers, constraints)
                if result.optimization_score < best_score:
                    best_score = result.optimization_score
                    best_path = result
                continue
            
            # Explore neighbors
            for edge in self.edges.get(current_id, []):
                if edge.to_node in visited:
                    continue
                
                # Check constraints
                if not self._satisfies_constraints(edge, g_score, edges_taken, transfers, constraints):
                    continue
                
                # Calculate new scores
                new_g = g_score + self._calculate_edge_weight(edge, constraints)
                new_transfers = transfers + (1 if edges_taken and edges_taken[-1].edge_type != edge.edge_type else 0)
                
                # Heuristic: straight-line distance to goal
                h_score = self._heuristic(self.nodes[edge.to_node].location, end_node.location)
                new_f = new_g + h_score
                
                # Add to priority queue
                new_path = path + [self.nodes[edge.to_node]]
                new_edges = edges_taken + [edge]
                
                heapq.heappush(pq, (new_f, edge.to_node, new_g, new_path, new_edges, new_transfers))
        
        if best_path:
            logger.info(f"✅ Path found: {best_path.total_time_minutes:.1f} min, {best_path.total_cost:.2f} TL")
        else:
            logger.warning(f"❌ No path found from {start_node_id} to {end_node_id}")
        
        return best_path
    
    def _calculate_edge_weight(self, edge: GraphEdge, constraints: PathConstraints) -> float:
        """
        Calculate edge weight for multi-objective optimization
        
        Balances:
        - Time
        - Cost
        - Scenic value
        - Crowding
        """
        time_weight = 1.0
        cost_weight = 0.3
        scenic_weight = 0.2 if constraints.prefer_scenic else 0.0
        crowding_weight = 0.5 if constraints.avoid_crowded else 0.0
        
        weight = (
            edge.time_minutes * time_weight +
            edge.cost * cost_weight +
            (1.0 - edge.scenic_score) * scenic_weight * 10 +  # Inverse: higher scenic = lower weight
            edge.crowding_factor * crowding_weight * 5
        )
        
        return weight
    
    def _satisfies_constraints(
        self,
        edge: GraphEdge,
        current_g_score: float,
        edges_taken: List[GraphEdge],
        transfers: int,
        constraints: PathConstraints
    ) -> bool:
        """Check if edge satisfies path constraints"""
        # Check allowed transport modes
        if constraints.allowed_transport_modes:
            if edge.edge_type not in constraints.allowed_transport_modes:
                return False
        
        # Check max time
        if constraints.max_time_minutes:
            total_time = sum(e.time_minutes for e in edges_taken) + edge.time_minutes
            if total_time > constraints.max_time_minutes:
                return False
        
        # Check max cost
        if constraints.max_cost:
            total_cost = sum(e.cost for e in edges_taken) + edge.cost
            if total_cost > constraints.max_cost:
                return False
        
        # Check max transfers
        if constraints.max_transfers:
            new_transfers = transfers + (1 if edges_taken and edges_taken[-1].edge_type != edge.edge_type else 0)
            if new_transfers > constraints.max_transfers:
                return False
        
        # Check accessibility
        if edge.accessibility_score < constraints.min_accessibility:
            return False
        
        return True
    
    def _create_path_result(
        self,
        path: List[GraphNode],
        edges: List[GraphEdge],
        transfers: int,
        constraints: PathConstraints
    ) -> PathResult:
        """Create PathResult from path and edges"""
        total_distance = sum(e.distance_km for e in edges)
        total_time = sum(e.time_minutes for e in edges)
        total_cost = sum(e.cost for e in edges)
        scenic_score = sum(e.scenic_score for e in edges) / len(edges) if edges else 0.0
        crowding_score = sum(e.crowding_factor for e in edges) / len(edges) if edges else 1.0
        
        # Optimization score (lower is better)
        optimization_score = self._calculate_edge_weight(
            GraphEdge(
                from_node="", to_node="",
                edge_type=EdgeType.WALK,
                distance_km=total_distance,
                time_minutes=total_time,
                cost=total_cost,
                scenic_score=scenic_score,
                crowding_factor=crowding_score
            ),
            constraints
        )
        
        return PathResult(
            path=path,
            edges=edges,
            total_distance_km=total_distance,
            total_time_minutes=total_time,
            total_cost=total_cost,
            scenic_score=scenic_score,
            crowding_score=crowding_score,
            num_transfers=transfers,
            optimization_score=optimization_score
        )
    
    def _heuristic(self, loc1: GeoCoordinate, loc2: GeoCoordinate) -> float:
        """Heuristic for A*: straight-line distance converted to time"""
        distance_km = self._calculate_distance(loc1, loc2)
        # Assume average speed of 20 km/h for heuristic
        time_estimate = (distance_km / 20.0) * 60
        return time_estimate
    
    def _calculate_distance(self, loc1: GeoCoordinate, loc2: GeoCoordinate) -> float:
        """Calculate Haversine distance between two coordinates"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(loc1.lat)
        lat2_rad = math.radians(loc2.lat)
        dlat = math.radians(loc2.lat - loc1.lat)
        dlon = math.radians(loc2.lon - loc1.lon)
        
        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _calculate_scenic_score(self, loc1: GeoCoordinate, loc2: GeoCoordinate) -> float:
        """Calculate scenic score based on location"""
        avg_lat = (loc1.lat + loc2.lat) / 2
        avg_lon = (loc1.lon + loc2.lon) / 2
        
        scenic = 0.5  # Base scenic score
        
        # Bosphorus area
        if 28.95 <= avg_lon <= 29.05 and 41.0 <= avg_lat <= 41.1:
            scenic += 0.3
        
        # Historic peninsula
        if 28.92 <= avg_lon <= 28.99 and 41.0 <= avg_lat <= 41.02:
            scenic += 0.2
        
        # Golden Horn
        if 28.94 <= avg_lon <= 28.98 and 41.01 <= avg_lat <= 41.04:
            scenic += 0.2
        
        return min(scenic, 1.0)
    
    def get_node_by_location(self, location: GeoCoordinate, max_distance_km: float = 0.5) -> Optional[GraphNode]:
        """Find nearest node to a location"""
        nearest = None
        min_distance = float('inf')
        
        for node in self.nodes.values():
            distance = self._calculate_distance(location, node.location)
            if distance < min_distance and distance <= max_distance_km:
                min_distance = distance
                nearest = node
        
        return nearest
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary"""
        return {
            "nodes": {
                node_id: {
                    "node_type": node.node_type.value,
                    "name": node.name,
                    "location": {"lat": node.location.lat, "lon": node.location.lon},
                    "properties": node.properties
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                from_id: [
                    {
                        "to": edge.to_node,
                        "type": edge.edge_type.value,
                        "distance_km": edge.distance_km,
                        "time_minutes": edge.time_minutes,
                        "cost": edge.cost,
                        "scenic_score": edge.scenic_score,
                        "line_id": edge.line_id
                    }
                    for edge in edges
                ]
                for from_id, edges in self.edges.items()
            },
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": sum(len(e) for e in self.edges.values()),
                "station_nodes": sum(1 for n in self.nodes.values() if n.node_type == NodeType.STATION),
                "poi_nodes": sum(1 for n in self.nodes.values() if n.node_type == NodeType.POI)
            }
        }


def create_transport_graph(
    poi_service: Optional[POIDatabaseService] = None,
    ml_transport: Optional[Any] = None
) -> TransportGraph:
    """
    Factory function to create and build transport graph
    
    Args:
        poi_service: Optional POI database service
        ml_transport: Optional ML-Enhanced transportation system
        
    Returns:
        Fully built TransportGraph
    """
    graph = TransportGraph()
    graph.build_from_services(poi_service, ml_transport)
    return graph


# Singleton instance
_transport_graph_instance: Optional[TransportGraph] = None


def get_transport_graph() -> TransportGraph:
    """Get or create singleton transport graph instance"""
    global _transport_graph_instance
    
    if _transport_graph_instance is None:
        # Try to load POI service
        poi_service = None
        if POI_DATABASE_AVAILABLE:
            try:
                poi_service = POIDatabaseService()
            except Exception as e:
                logger.warning(f"Could not initialize POI service: {e}")
        
        # Try to load ML transport
        ml_transport = None
        if ML_TRANSPORT_AVAILABLE:
            try:
                ml_transport = MLEnhancedTransportationSystem()
            except Exception as e:
                logger.warning(f"Could not initialize ML transport: {e}")
        
        _transport_graph_instance = create_transport_graph(poi_service, ml_transport)
    
    return _transport_graph_instance


if __name__ == "__main__":
    # Test the transport graph
    logging.basicConfig(level=logging.INFO)
    
    print("🗺️ Testing Transport Graph Service")
    print("=" * 60)
    
    # Create graph
    graph = get_transport_graph()
    
    print(f"\n📊 Graph Statistics:")
    stats = graph.to_dict()['stats']
    print(f"   Total Nodes: {stats['total_nodes']}")
    print(f"   Total Edges: {stats['total_edges']}")
    print(f"   Station Nodes: {stats['station_nodes']}")
    print(f"   POI Nodes: {stats['poi_nodes']}")
    
    # Test pathfinding
    print(f"\n🔍 Testing Pathfinding:")
    print(f"   Route: Sultanahmet → Taksim")
    
    result = graph.find_shortest_path("T1_Sultanahmet", "M2_Taksim")
    
    if result:
        print(f"\n✅ Path Found!")
        print(f"   Time: {result.total_time_minutes:.1f} minutes")
        print(f"   Cost: {result.total_cost:.2f} TL")
        print(f"   Distance: {result.total_distance_km:.2f} km")
        print(f"   Transfers: {result.num_transfers}")
        print(f"   Scenic Score: {result.scenic_score:.2f}")
        print(f"\n   Route:")
        for i, node in enumerate(result.path):
            print(f"   {i+1}. {node.name} ({node.node_type.value})")
            if i < len(result.edges):
                edge = result.edges[i]
                print(f"      → {edge.edge_type.value} ({edge.time_minutes:.0f} min)")
    else:
        print("❌ No path found")
    
    print("\n✅ Transport Graph Test Complete!")
