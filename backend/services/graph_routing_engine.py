"""
Graph-Based Multi-Modal Routing Engine for Istanbul
====================================================

Google Maps-level routing engine using graph search algorithms (Dijkstra/A*).
Handles metro, tram, bus, ferry, funicular, Marmaray, and walking segments
with intelligent transfer logic and multi-leg journey planning.

Features:
- Graph-based routing with Dijkstra's algorithm
- Multi-modal transportation (metro, tram, bus, ferry, funicular, Marmaray, walking)
- Intelligent transfer handling with time penalties
- Time-aware pathfinding
- Support for multiple route alternatives
- Real transfer points (e.g., Yenikapı, Ayrılık Çeşmesi, Kabataş)
"""

import logging
import heapq
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the transportation graph (station/stop)"""
    id: str
    name: str
    lat: float
    lng: float
    node_type: str  # 'metro', 'tram', 'bus', 'ferry', 'funicular', 'marmaray', 'walking_point'
    line_id: Optional[str] = None  # Which line this station belongs to
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.id == other.id
    
    def __lt__(self, other):
        """Less than comparison for heap operations"""
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id < other.id
    
    def __le__(self, other):
        """Less than or equal comparison"""
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id <= other.id
    
    def __gt__(self, other):
        """Greater than comparison"""
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id > other.id
    
    def __ge__(self, other):
        """Greater than or equal comparison"""
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id >= other.id


@dataclass
class GraphEdge:
    """An edge in the transportation graph (connection between stops)"""
    from_node: GraphNode
    to_node: GraphNode
    edge_type: str  # 'transit', 'transfer', 'walking'
    mode: str  # 'metro', 'tram', 'bus', 'ferry', 'funicular', 'marmaray', 'walk'
    duration: int  # minutes
    distance: float  # meters
    line_id: Optional[str] = None
    cost: float = 0.0  # Combined cost for routing (duration + penalties)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutePath:
    """A complete route path from start to destination"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_duration: int  # minutes
    total_distance: float  # meters
    total_cost: float
    transfers: int
    modes_used: List[str]
    summary: str = ""


class TransportationGraph:
    """Graph representation of Istanbul's transportation network"""
    
    def __init__(self):
        """Initialize empty transportation graph"""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[GraphEdge]] = defaultdict(list)  # from_node_id -> [edges]
        self.reverse_edges: Dict[str, List[GraphEdge]] = defaultdict(list)  # to_node_id -> [edges]
        
        # Transfer penalties (minutes)
        self.TRANSFER_PENALTY = 5  # Time penalty for transfers
        self.WALKING_TRANSFER_PENALTY = 3  # Additional penalty for walking transfers
        self.INTER_LINE_TRANSFER_PENALTY = 2  # Penalty for changing lines
        
        # Known major transfer hubs
        self.TRANSFER_HUBS = {
            'yenikapi': ['M1', 'M2', 'Marmaray'],
            'ayrilik_cesmesi': ['M4', 'Marmaray'],
            'kabatas': ['T1', 'F1', 'ferry'],
            'karakoy': ['T1', 'F2', 'ferry'],
            'taksim': ['M2', 'F1'],
            'sisli': ['M2', 'M7'],
            'aksaray': ['M1', 'T1'],
        }
        
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        
    def add_edge(self, edge: GraphEdge) -> None:
        """Add a directed edge to the graph"""
        # Calculate cost (duration + penalties)
        cost = edge.duration
        if edge.edge_type == 'transfer':
            cost += self.TRANSFER_PENALTY
            if edge.mode == 'walk':
                cost += self.WALKING_TRANSFER_PENALTY
        
        edge.cost = cost
        
        # Add forward edge
        self.edges[edge.from_node.id].append(edge)
        # Add reverse edge for lookup
        self.reverse_edges[edge.to_node.id].append(edge)
        
    def add_bidirectional_edge(
        self, 
        node1: GraphNode, 
        node2: GraphNode,
        edge_type: str,
        mode: str,
        duration: int,
        distance: float,
        line_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a bidirectional edge (for metro/tram lines)"""
        metadata = metadata or {}
        
        # Forward edge
        forward_edge = GraphEdge(
            from_node=node1,
            to_node=node2,
            edge_type=edge_type,
            mode=mode,
            duration=duration,
            distance=distance,
            line_id=line_id,
            metadata=metadata
        )
        self.add_edge(forward_edge)
        
        # Reverse edge
        reverse_edge = GraphEdge(
            from_node=node2,
            to_node=node1,
            edge_type=edge_type,
            mode=mode,
            duration=duration,
            distance=distance,
            line_id=line_id,
            metadata=metadata
        )
        self.add_edge(reverse_edge)
        
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
        
    def get_neighbors(self, node_id: str) -> List[GraphEdge]:
        """Get all outgoing edges from a node"""
        return self.edges.get(node_id, [])
        
    def find_nearest_node(self, lat: float, lng: float, max_distance_km: float = 1.0) -> Optional[GraphNode]:
        """Find nearest node to a location"""
        nearest = None
        min_distance = float('inf')
        
        for node in self.nodes.values():
            distance = self._haversine_distance(lat, lng, node.lat, node.lng)
            if distance < min_distance and distance <= max_distance_km:
                min_distance = distance
                nearest = node
                
        return nearest
        
    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in kilometers using Haversine formula"""
        R = 6371  # Earth radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


class GraphRoutingEngine:
    """Main routing engine using Dijkstra's algorithm"""
    
    def __init__(self, graph: TransportationGraph):
        """Initialize routing engine with a transportation graph"""
        self.graph = graph
        
    def find_route(
        self,
        start_lat: float,
        start_lng: float,
        end_lat: float,
        end_lng: float,
        max_transfers: int = 3,
        max_walking_distance: float = 1.0  # km
    ) -> Optional[RoutePath]:
        """
        Find optimal route using Dijkstra's algorithm
        
        Args:
            start_lat, start_lng: Start coordinates
            end_lat, end_lng: End coordinates
            max_transfers: Maximum number of transfers allowed
            max_walking_distance: Maximum walking distance to/from stations (km)
            
        Returns:
            RoutePath with optimal route, or None if no route found
        """
        logger.info(f"🔍 Finding route from ({start_lat}, {start_lng}) to ({end_lat}, {end_lng})")
        
        # Find nearest nodes to start and end
        start_node = self.graph.find_nearest_node(start_lat, start_lng, max_walking_distance)
        end_node = self.graph.find_nearest_node(end_lat, end_lng, max_walking_distance)
        
        if not start_node:
            logger.warning(f"No station found near start location within {max_walking_distance}km")
            return None
            
        if not end_node:
            logger.warning(f"No station found near end location within {max_walking_distance}km")
            return None
            
        logger.info(f"🚉 Start station: {start_node.name}, End station: {end_node.name}")
        
        # Run Dijkstra's algorithm
        path = self._dijkstra(start_node, end_node, max_transfers)
        
        if not path:
            logger.warning("No route found between stations")
            return None
            
        return path
        
    def _dijkstra(
        self,
        start_node: GraphNode,
        end_node: GraphNode,
        max_transfers: int = 3
    ) -> Optional[RoutePath]:
        """
        Dijkstra's algorithm for shortest path
        
        Returns path with minimum cost (time + transfer penalties)
        """
        # Priority queue: (cost, transfer_count, current_node, path_nodes, path_edges)
        queue = [(0, 0, start_node, [start_node], [])]
        
        # Best cost to reach each node
        best_cost: Dict[str, float] = {start_node.id: 0}
        
        # Track visited to avoid cycles
        visited: Set[str] = set()
        
        while queue:
            current_cost, transfers, current_node, path_nodes, path_edges = heapq.heappop(queue)
            
            # Found destination
            if current_node.id == end_node.id:
                return self._build_route_path(path_nodes, path_edges)
                
            # Skip if already visited with better cost
            if current_node.id in visited:
                continue
                
            visited.add(current_node.id)
            
            # Too many transfers
            if transfers > max_transfers:
                continue
                
            # Explore neighbors
            for edge in self.graph.get_neighbors(current_node.id):
                next_node = edge.to_node
                
                # Skip temporarily removed nodes (for Yen's algorithm)
                if hasattr(next_node, '_temp_removed') and next_node._temp_removed:
                    continue
                
                # Skip if already visited
                if next_node.id in visited:
                    continue
                    
                # Calculate new cost
                new_cost = current_cost + edge.cost
                
                # Count transfers
                new_transfers = transfers
                if path_edges and edge.edge_type == 'transfer':
                    new_transfers += 1
                elif path_edges and edge.line_id != path_edges[-1].line_id:
                    # Different line = transfer
                    new_transfers += 1
                    # Add inter-line penalty
                    new_cost += self.graph.INTER_LINE_TRANSFER_PENALTY
                
                # Skip if we've found a better path to this node
                if next_node.id in best_cost and new_cost >= best_cost[next_node.id]:
                    continue
                    
                best_cost[next_node.id] = new_cost
                
                # Add to queue
                new_path_nodes = path_nodes + [next_node]
                new_path_edges = path_edges + [edge]
                
                heapq.heappush(queue, (new_cost, new_transfers, next_node, new_path_nodes, new_path_edges))
                
        # No path found
        return None
        
    def _build_route_path(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> RoutePath:
        """Build a RoutePath object from nodes and edges"""
        total_duration = sum(edge.duration for edge in edges)
        total_distance = sum(edge.distance for edge in edges)
        total_cost = sum(edge.cost for edge in edges)
        
        # Count transfers
        transfers = 0
        current_line = None
        for edge in edges:
            if edge.edge_type == 'transfer':
                transfers += 1
            elif current_line and edge.line_id != current_line:
                transfers += 1
            current_line = edge.line_id
            
        # Get unique modes
        modes_used = list(dict.fromkeys([edge.mode for edge in edges]))
        
        # Build summary
        summary = f"{total_duration} min, {transfers} transfer(s), {len(modes_used)} mode(s)"
        
        return RoutePath(
            nodes=nodes,
            edges=edges,
            total_duration=total_duration,
            total_distance=total_distance,
            total_cost=total_cost,
            transfers=transfers,
            modes_used=modes_used,
            summary=summary
        )
        
    def find_alternative_routes(
        self,
        start_lat: float,
        start_lng: float,
        end_lat: float,
        end_lng: float,
        num_alternatives: int = 3,
        max_transfers: int = 3
    ) -> List[RoutePath]:
        """
        Find multiple alternative routes using Yen's k-shortest paths algorithm
        
        Yen's algorithm finds k-shortest loopless paths by:
        1. Finding the shortest path
        2. For each path found, systematically removing edges to find deviations
        3. Returns diverse alternative routes
        
        Args:
            start_lat, start_lng: Starting coordinates
            end_lat, end_lng: Destination coordinates
            num_alternatives: Number of alternative routes to find
            max_transfers: Maximum number of transfers allowed
            
        Returns:
            List of RoutePath objects, sorted by cost
        """
        # Find start and end nodes
        start_node = self.graph.find_nearest_node(start_lat, start_lng)
        end_node = self.graph.find_nearest_node(end_lat, end_lng)
        
        if not start_node or not end_node:
            logger.warning("Could not find start or end nodes for alternative routes")
            return []
        
        # Use Yen's algorithm to find k-shortest paths
        k_paths = self._yens_algorithm(start_node, end_node, num_alternatives, max_transfers)
        
        logger.info(f"🔀 Found {len(k_paths)} alternative routes")
        return k_paths
    
    def _yens_algorithm(
        self,
        start_node: GraphNode,
        end_node: GraphNode,
        k: int,
        max_transfers: int = 3
    ) -> List[RoutePath]:
        """
        Diversity-Aware Yen's k-shortest paths algorithm
        
        Finds k loopless paths from start to end with emphasis on genuine diversity.
        Filters out paths that differ only in transfer variations but use the same
        transit line combinations.
        """
        # Store the k shortest paths
        A = []
        
        # Potential k-th shortest paths
        B = []
        
        # Track line combinations we've seen to encourage diversity
        seen_line_combinations = set()
        
        # Find the shortest path
        shortest_path = self._dijkstra(start_node, end_node, max_transfers)
        if not shortest_path:
            return []
        
        A.append(shortest_path)
        line_combo = self._get_line_combination(shortest_path)
        seen_line_combinations.add(line_combo)
        logger.debug(f"🔀 Path 1 line combo: {line_combo}")
        
        # Find k-1 more paths
        for k_iter in range(1, k):
            # Get the (k-1)th path
            prev_path = A[k_iter - 1]
            
            # Find deviation paths
            for i in range(len(prev_path.nodes) - 1):
                # Spur node: node where the new path will deviate
                spur_node = prev_path.nodes[i]
                # Root path: portion of previous path from start to spur node
                root_path_nodes = prev_path.nodes[:i+1]
                root_path_edges = prev_path.edges[:i]
                
                # Temporarily store removed edges
                removed_edges = []
                
                # Remove edges that are part of previous paths sharing the same root
                for path in A:
                    if len(path.nodes) > i and path.nodes[:i+1] == root_path_nodes:
                        # Remove the edge after the spur node in this path
                        if i < len(path.edges):
                            edge_to_remove = path.edges[i]
                            removed_edges.append(edge_to_remove)
                            self._remove_edge_temporarily(edge_to_remove)
                
                # Also remove root path nodes (except spur node) to avoid loops
                removed_nodes = []
                for node in root_path_nodes[:-1]:  # All except spur node
                    removed_nodes.append(node)
                    self._remove_node_temporarily(node)
                
                # Calculate spur path from spur node to end
                spur_path = self._dijkstra(spur_node, end_node, max_transfers)
                
                # Restore removed edges and nodes
                for edge in removed_edges:
                    self._restore_edge(edge)
                for node in removed_nodes:
                    self._restore_node(node)
                
                if spur_path:
                    # Combine root path and spur path
                    total_path = self._combine_paths(root_path_nodes, root_path_edges, spur_path)
                    
                    # Early duplicate filtering: Check line combination
                    path_line_combo = self._get_line_combination(total_path)
                    
                    # Skip if this is just a transfer variation of a path we already have
                    if self._is_equivalent_path(total_path, path_line_combo, seen_line_combinations):
                        logger.debug(f"⏭️  Skipping equivalent path with combo: {path_line_combo}")
                        continue
                    
                    # Add to potential paths if not already present
                    if not self._path_exists_in_list(total_path, B) and not self._path_exists_in_list(total_path, A):
                        B.append(total_path)
                        logger.debug(f"➕ Added diverse candidate with combo: {path_line_combo}")
            
            if not B:
                # No more paths found
                break
            
            # Sort B by diversity score (prefer different line combinations) then by cost
            B.sort(key=lambda p: (
                self._get_diversity_penalty(p, seen_line_combinations),
                p.total_cost
            ))
            
            # Add the best diverse path to A
            best_path = B.pop(0)
            A.append(best_path)
            
            # Track this line combination
            line_combo = self._get_line_combination(best_path)
            seen_line_combinations.add(line_combo)
            logger.debug(f"🔀 Path {len(A)} line combo: {line_combo}")
        
        return A
    
    def _get_line_combination(self, path: RoutePath) -> str:
        """
        Get the unique line combination for a path
        
        Returns a string representing the transit lines used in order,
        ignoring transfer-only variations.
        """
        lines = []
        for edge in path.edges:
            if edge.edge_type == 'transit' and edge.line_id:
                # Only include actual transit segments
                lines.append(f"{edge.mode}:{edge.line_id}")
        
        # Deduplicate consecutive same lines (happens with multiple segments on same line)
        unique_lines = []
        prev_line = None
        for line in lines:
            if line != prev_line:
                unique_lines.append(line)
                prev_line = line
        
        return "_".join(unique_lines) if unique_lines else "walk_only"
    
    def _is_equivalent_path(
        self, 
        path: RoutePath, 
        line_combo: str, 
        seen_combinations: Set[str]
    ) -> bool:
        """
        Check if this path is equivalent to one we've already seen
        
        A path is equivalent if it uses the same line combination,
        even if transfer details differ.
        """
        # Check if we've seen this exact line combination
        if line_combo in seen_combinations:
            # Count transfer edges
            transfer_count = sum(1 for e in path.edges if e.edge_type == 'transfer')
            
            # If this path has many transfers (likely just transfer variations), skip it
            # Allow if it has 0-1 transfers per line segment as those are necessary
            transit_segments = sum(1 for e in path.edges if e.edge_type == 'transit')
            if transfer_count > transit_segments + 1:  # More than 1 transfer per segment
                return True
        
        return False
    
    def _get_diversity_penalty(self, path: RoutePath, seen_combinations: Set[str]) -> int:
        """
        Calculate diversity penalty for path sorting
        
        Returns:
            0 if path has unique line combination (most diverse)
            1 if path has seen line combination but different characteristics
        """
        line_combo = self._get_line_combination(path)
        
        if line_combo not in seen_combinations:
            return 0  # Most diverse - completely new line combination
        
        return 1  # Less diverse - similar to existing path
    
    def _remove_edge_temporarily(self, edge: GraphEdge) -> None:
        """Temporarily remove an edge from the graph"""
        if edge.from_node.id in self.graph.edges:
            edges_list = self.graph.edges[edge.from_node.id]
            # Store original list and filter out the edge
            self.graph.edges[edge.from_node.id] = [e for e in edges_list if not self._edges_equal(e, edge)]
    
    def _restore_edge(self, edge: GraphEdge) -> None:
        """Restore a temporarily removed edge"""
        if edge.from_node.id not in self.graph.edges:
            self.graph.edges[edge.from_node.id] = []
        # Only add if not already present
        if not any(self._edges_equal(e, edge) for e in self.graph.edges[edge.from_node.id]):
            self.graph.edges[edge.from_node.id].append(edge)
    
    def _remove_node_temporarily(self, node: GraphNode) -> None:
        """Temporarily remove a node from the graph by removing all its edges"""
        # Mark node as temporarily removed
        if not hasattr(node, '_temp_removed'):
            node._temp_removed = False
        node._temp_removed = True
    
    def _restore_node(self, node: GraphNode) -> None:
        """Restore a temporarily removed node"""
        if hasattr(node, '_temp_removed'):
            node._temp_removed = False
    
    def _edges_equal(self, edge1: GraphEdge, edge2: GraphEdge) -> bool:
        """Check if two edges are equal"""
        return (edge1.from_node.id == edge2.from_node.id and 
                edge1.to_node.id == edge2.to_node.id and
                edge1.line_id == edge2.line_id)
    
    def _combine_paths(
        self, 
        root_nodes: List[GraphNode], 
        root_edges: List[GraphEdge], 
        spur_path: RoutePath
    ) -> RoutePath:
        """Combine root path and spur path into a single path"""
        # Combine nodes (skip first node of spur path as it's the same as last root node)
        combined_nodes = root_nodes + spur_path.nodes[1:]
        combined_edges = root_edges + spur_path.edges
        
        # Build the combined route
        return self._build_route_path(combined_nodes, combined_edges)
    
    def _path_exists_in_list(self, path: RoutePath, path_list: List[RoutePath]) -> bool:
        """Check if a path already exists in a list of paths"""
        path_node_ids = [node.id for node in path.nodes]
        for existing_path in path_list:
            existing_node_ids = [node.id for node in existing_path.nodes]
            if path_node_ids == existing_node_ids:
                return True
        return False


def create_istanbul_graph(transit_data: Dict) -> TransportationGraph:
    """
    Create a complete transportation graph from transit data
    
    Args:
        transit_data: Dictionary containing metro_lines, tram_lines, ferry_routes, 
                     funicular_lines, bus_routes, etc.
                     
    Returns:
        Fully populated TransportationGraph
    """
    logger.info("🏗️ Building Istanbul transportation graph...")
    
    graph = TransportationGraph()
    
    # Add metro lines
    if 'metro_lines' in transit_data:
        _add_metro_lines_to_graph(graph, transit_data['metro_lines'])
        
    # Add tram lines
    if 'tram_lines' in transit_data:
        _add_tram_lines_to_graph(graph, transit_data['tram_lines'])
        
    # Add funicular lines
    if 'funicular_lines' in transit_data:
        _add_funicular_lines_to_graph(graph, transit_data['funicular_lines'])
        
    # Add ferry routes
    if 'ferry_routes' in transit_data:
        _add_ferry_routes_to_graph(graph, transit_data['ferry_routes'])
        
    # Add bus routes
    if 'bus_routes' in transit_data:
        _add_bus_routes_to_graph(graph, transit_data['bus_routes'])
        
    # Add transfer connections
    _add_transfer_connections(graph)
    
    logger.info(f"✅ Graph built: {len(graph.nodes)} nodes, {sum(len(edges) for edges in graph.edges.values())} edges")
    
    return graph


def _add_metro_lines_to_graph(graph: TransportationGraph, metro_lines: Dict) -> None:
    """Add metro lines to the graph"""
    for line_id, line_data in metro_lines.items():
        stations = line_data.get('stations', [])
        
        # Determine if this is Marmaray or regular metro
        node_type = 'marmaray' if line_id == 'Marmaray' else 'metro'
        mode = 'marmaray' if line_id == 'Marmaray' else 'metro'
        
        for i, station_data in enumerate(stations):
            # Create node
            node_id = f"{mode}_{line_id}_{station_data['name'].lower().replace(' ', '_').replace('-', '_')}"
            node = GraphNode(
                id=node_id,
                name=station_data['name'],
                lat=station_data['lat'],
                lng=station_data['lng'],
                node_type=node_type,
                line_id=line_id,
                metadata={'line_name': line_data.get('name', line_id)}
            )
            graph.add_node(node)
            
            # Connect to previous station
            if i > 0:
                prev_station = stations[i - 1]
                prev_node_id = f"{mode}_{line_id}_{prev_station['name'].lower().replace(' ', '_').replace('-', '_')}"
                prev_node = graph.get_node(prev_node_id)
                
                if prev_node:
                    # Estimate distance and duration
                    distance = graph._haversine_distance(
                        prev_station['lat'], prev_station['lng'],
                        station_data['lat'], station_data['lng']
                    ) * 1000  # meters
                    
                    # Metro average speed ~40 km/h (Marmaray similar speed)
                    duration = max(2, int((distance / 1000) / 40 * 60))  # minutes, minimum 2 min
                    
                    graph.add_bidirectional_edge(
                        prev_node, node,
                        edge_type='transit',
                        mode=mode,
                        duration=duration,
                        distance=distance,
                        line_id=line_id
                    )


def _add_tram_lines_to_graph(graph: TransportationGraph, tram_lines: Dict) -> None:
    """Add tram lines to the graph"""
    for line_id, line_data in tram_lines.items():
        stations = line_data.get('stations', [])
        
        for i, station_data in enumerate(stations):
            node_id = f"tram_{line_id}_{station_data['name'].lower().replace(' ', '_').replace('-', '_')}"
            node = GraphNode(
                id=node_id,
                name=station_data['name'],
                lat=station_data['lat'],
                lng=station_data['lng'],
                node_type='tram',
                line_id=line_id,
                metadata={'line_name': line_data.get('name', line_id)}
            )
            graph.add_node(node)
            
            if i > 0:
                prev_station = stations[i - 1]
                prev_node_id = f"tram_{line_id}_{prev_station['name'].lower().replace(' ', '_').replace('-', '_')}"
                prev_node = graph.get_node(prev_node_id)
                
                if prev_node:
                    distance = graph._haversine_distance(
                        prev_station['lat'], prev_station['lng'],
                        station_data['lat'], station_data['lng']
                    ) * 1000
                    
                    # Tram average speed ~20 km/h
                    duration = max(2, int((distance / 1000) / 20 * 60))
                    
                    graph.add_bidirectional_edge(
                        prev_node, node,
                        edge_type='transit',
                        mode='tram',
                        duration=duration,
                        distance=distance,
                        line_id=line_id
                    )


def _add_funicular_lines_to_graph(graph: TransportationGraph, funicular_lines: Dict) -> None:
    """Add funicular lines to the graph"""
    for line_id, line_data in funicular_lines.items():
        stations = line_data.get('stations', [])
        duration = line_data.get('duration', 3)  # Default 3 minutes
        
        for i, station_data in enumerate(stations):
            node_id = f"funicular_{line_id}_{station_data['name'].lower().replace(' ', '_').replace('-', '_')}"
            node = GraphNode(
                id=node_id,
                name=station_data['name'],
                lat=station_data['lat'],
                lng=station_data['lng'],
                node_type='funicular',
                line_id=line_id,
                metadata={'line_name': line_data.get('name', line_id)}
            )
            graph.add_node(node)
            
            if i > 0:
                prev_station = stations[i - 1]
                prev_node_id = f"funicular_{line_id}_{prev_station['name'].lower().replace(' ', '_').replace('-', '_')}"
                prev_node = graph.get_node(prev_node_id)
                
                if prev_node:
                    distance = graph._haversine_distance(
                        prev_station['lat'], prev_station['lng'],
                        station_data['lat'], station_data['lng']
                    ) * 1000
                    
                    graph.add_bidirectional_edge(
                        prev_node, node,
                        edge_type='transit',
                        mode='funicular',
                        duration=duration,
                        distance=distance,
                        line_id=line_id
                    )


def _add_ferry_routes_to_graph(graph: TransportationGraph, ferry_routes: Dict) -> None:
    """Add ferry routes to the graph"""
    for route_id, route_data in ferry_routes.items():
        stops = route_data.get('stops', [])
        duration = route_data.get('duration', 20)
        
        for i, stop_data in enumerate(stops):
            node_id = f"ferry_{route_id}_{stop_data['name'].lower().replace(' ', '_').replace('-', '_')}"
            node = GraphNode(
                id=node_id,
                name=stop_data['name'],
                lat=stop_data['lat'],
                lng=stop_data['lng'],
                node_type='ferry',
                line_id=route_id,
                metadata={'route_name': route_data.get('name', route_id)}
            )
            graph.add_node(node)
            
            if i > 0:
                prev_stop = stops[i - 1]
                prev_node_id = f"ferry_{route_id}_{prev_stop['name'].lower().replace(' ', '_').replace('-', '_')}"
                prev_node = graph.get_node(prev_node_id)
                
                if prev_node:
                    distance = graph._haversine_distance(
                        prev_stop['lat'], prev_stop['lng'],
                        stop_data['lat'], stop_data['lng']
                    ) * 1000
                    
                    # Divide duration evenly between stops
                    segment_duration = duration // len(stops) if len(stops) > 1 else duration
                    
                    graph.add_bidirectional_edge(
                        prev_node, node,
                        edge_type='transit',
                        mode='ferry',
                        duration=segment_duration,
                        distance=distance,
                        line_id=route_id
                    )


def _add_bus_routes_to_graph(graph: TransportationGraph, bus_routes: Dict) -> None:
    """Add bus routes to the graph"""
    for route_id, route_data in bus_routes.items():
        stops = route_data.get('stops', [])
        
        for i, stop_data in enumerate(stops):
            node_id = f"bus_{route_id}_{stop_data['name'].lower().replace(' ', '_').replace('-', '_')}"
            node = GraphNode(
                id=node_id,
                name=stop_data['name'],
                lat=stop_data['lat'],
                lng=stop_data['lng'],
                node_type='bus',
                line_id=route_id,
                metadata={'route_name': route_data.get('name', route_id)}
            )
            graph.add_node(node)
            
            if i > 0:
                prev_stop = stops[i - 1]
                prev_node_id = f"bus_{route_id}_{prev_stop['name'].lower().replace(' ', '_').replace('-', '_')}"
                prev_node = graph.get_node(prev_node_id)
                
                if prev_node:
                    distance = graph._haversine_distance(
                        prev_stop['lat'], prev_stop['lng'],
                        stop_data['lat'], stop_data['lng']
                    ) * 1000
                    
                    # Bus average speed ~25 km/h in Istanbul traffic
                    duration = max(3, int((distance / 1000) / 25 * 60))
                    
                    graph.add_bidirectional_edge(
                        prev_node, node,
                        edge_type='transit',
                        mode='bus',
                        duration=duration,
                        distance=distance,
                        line_id=route_id
                    )


def _add_transfer_connections(graph: TransportationGraph) -> None:
    """
    Add transfer connections between different lines at shared stations
    
    Detects stations with the same name on different lines and adds transfer edges
    """
    logger.info("🔄 Adding transfer connections...")
    
    # Group nodes by normalized station name
    stations_by_name: Dict[str, List[GraphNode]] = defaultdict(list)
    
    for node in graph.nodes.values():
        # Normalize station name
        normalized_name = node.name.lower().replace(' ', '').replace('-', '').replace('ı', 'i')
        stations_by_name[normalized_name].append(node)
        
    # Add transfer edges between nodes with same name
    transfer_count = 0
    for station_name, nodes in stations_by_name.items():
        if len(nodes) < 2:
            continue
            
        # Add transfer edges between all combinations
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                # Transfer time (5 minutes base + distance penalty)
                distance = graph._haversine_distance(node1.lat, node1.lng, node2.lat, node2.lng) * 1000
                transfer_duration = 5
                
                # If stations are far apart (>100m), add walking time
                if distance > 100:
                    walking_time = int((distance / 1000) / 5 * 60)  # 5 km/h walking
                    transfer_duration += walking_time
                
                # Add bidirectional transfer edges
                transfer_edge_1 = GraphEdge(
                    from_node=node1,
                    to_node=node2,
                    edge_type='transfer',
                    mode='walk',
                    duration=transfer_duration,
                    distance=distance,
                    line_id=None,
                    metadata={'transfer_type': 'station_transfer'}
                )
                graph.add_edge(transfer_edge_1)
                
                transfer_edge_2 = GraphEdge(
                    from_node=node2,
                    to_node=node1,
                    edge_type='transfer',
                    mode='walk',
                    duration=transfer_duration,
                    distance=distance,
                    line_id=None,
                    metadata={'transfer_type': 'station_transfer'}
                )
                graph.add_edge(transfer_edge_2)
                
                transfer_count += 2
                logger.debug(f"  Transfer: {node1.name} ({node1.line_id}) ↔ {node2.name} ({node2.line_id})")
                
    logger.info(f"✅ Added {transfer_count} transfer connections")
