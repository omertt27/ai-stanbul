"""
Intelligent Route Finder - Industry-Level Graph-Based Routing
Uses Dijkstra and A* algorithms for optimal multi-modal journey planning
"""

import heapq
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

@dataclass
class RouteSegment:
    """Represents a segment of a journey"""
    line_id: str
    line_name: str
    transport_type: str  # metro, bus, tram, ferry, etc.
    from_stop: str
    to_stop: str
    from_stop_name: str
    to_stop_name: str
    stops_count: int
    duration_minutes: int
    distance_km: float
    coordinates: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class TransferSegment:
    """Represents a transfer between transport lines"""
    from_stop: str
    to_stop: str
    from_stop_name: str
    to_stop_name: str
    transfer_type: str  # "same_station", "walking", "nearby"
    duration_minutes: int
    distance_meters: int
    walking_required: bool = True

@dataclass
class Journey:
    """Complete journey with multiple segments"""
    origin: str
    destination: str
    origin_name: str
    destination_name: str
    segments: List[RouteSegment]
    transfers: List[TransferSegment]
    total_duration_minutes: int
    total_distance_km: float
    total_transfers: int
    total_walking_meters: int
    estimated_cost_tl: float
    transport_types_used: Set[str]
    quality_score: float  # Overall route quality (0-1)
    
    def to_dict(self) -> Dict:
        """Convert journey to dictionary for JSON serialization"""
        return {
            'origin': self.origin_name,
            'destination': self.destination_name,
            'total_duration': self.total_duration_minutes,
            'total_distance_km': round(self.total_distance_km, 2),
            'transfers': self.total_transfers,
            'walking_distance_m': self.total_walking_meters,
            'cost_tl': self.estimated_cost_tl,
            'transport_types': list(self.transport_types_used),
            'quality_score': round(self.quality_score, 2),
            'segments': [
                {
                    'type': 'transport',
                    'line': seg.line_name,
                    'transport': seg.transport_type,
                    'from': seg.from_stop_name,
                    'to': seg.to_stop_name,
                    'stops': seg.stops_count,
                    'duration': seg.duration_minutes
                }
                for seg in self.segments
            ],
            'transfer_details': [
                {
                    'type': 'transfer',
                    'from': t.from_stop_name,
                    'to': t.to_stop_name,
                    'transfer_type': t.transfer_type,
                    'duration': t.duration_minutes,
                    'walking': t.walking_required
                }
                for t in self.transfers
            ]
        }

@dataclass
class PathNode:
    """Node in the pathfinding graph"""
    stop_id: str
    cost: float  # Total cost to reach this node
    heuristic: float  # Estimated cost to destination (for A*)
    parent: Optional['PathNode'] = None
    line_id: Optional[str] = None
    
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

@dataclass
class RoutePreferences:
    """User preferences for route optimization"""
    minimize_transfers: bool = True
    minimize_walking: bool = False
    minimize_time: bool = True
    minimize_cost: bool = False
    prefer_metro: bool = True
    prefer_direct_routes: bool = True
    max_transfers: int = 3
    max_walking_meters: int = 1000
    avoid_transport_types: Set[str] = field(default_factory=set)
    wheelchair_accessible: bool = False
    
    def get_weight_factors(self) -> Dict[str, float]:
        """Get weighting factors for route optimization"""
        return {
            'time_weight': 1.0 if self.minimize_time else 0.5,
            'transfer_weight': 3.0 if self.minimize_transfers else 1.0,
            'walking_weight': 2.0 if self.minimize_walking else 0.5,
            'cost_weight': 1.5 if self.minimize_cost else 0.3,
            'metro_bonus': 0.8 if self.prefer_metro else 1.0
        }

class IntelligentRouteFinder:
    """
    Industry-level route finding using graph algorithms
    Supports Dijkstra and A* for optimal pathfinding
    """
    
    def __init__(self, network_graph):
        """
        Initialize with transportation network graph
        
        Args:
            network_graph: TransportationNetwork from route_network_builder
        """
        self.network = network_graph
        self.earth_radius_km = 6371.0
    
    def calculate_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """
        Calculate great-circle distance between two coordinates (Haversine formula)
        
        Args:
            coord1: (latitude, longitude) tuple
            coord2: (latitude, longitude) tuple
            
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return self.earth_radius_km * c
    
    def heuristic(self, stop_id: str, destination_id: str) -> float:
        """
        A* heuristic function - straight-line distance to destination
        
        Args:
            stop_id: Current stop ID
            destination_id: Destination stop ID
            
        Returns:
            Estimated cost (time in minutes)
        """
        if stop_id not in self.network.stops or destination_id not in self.network.stops:
            return 0.0
        
        stop1 = self.network.stops[stop_id]
        stop2 = self.network.stops[destination_id]
        
        # Calculate straight-line distance
        distance_km = self.calculate_distance(
            (stop1.lat, stop1.lon),
            (stop2.lat, stop2.lon)
        )
        
        # Assume average speed of 30 km/h for public transport
        estimated_time_minutes = (distance_km / 30.0) * 60.0
        
        return estimated_time_minutes
    
    def find_optimal_route(self, 
                          origin_id: str, 
                          destination_id: str,
                          preferences: RoutePreferences = None,
                          use_astar: bool = True) -> Optional[Journey]:
        """
        Find optimal route using A* or Dijkstra algorithm
        
        Args:
            origin_id: Starting stop ID
            destination_id: Ending stop ID
            preferences: Route optimization preferences
            use_astar: Use A* (True) or Dijkstra (False)
            
        Returns:
            Journey object or None if no route found
        """
        if preferences is None:
            preferences = RoutePreferences()
        
        if origin_id not in self.network.stops or destination_id not in self.network.stops:
            return None
        
        # Priority queue: (total_cost, stop_id, node)
        open_set = []
        start_node = PathNode(
            stop_id=origin_id,
            cost=0.0,
            heuristic=self.heuristic(origin_id, destination_id) if use_astar else 0.0
        )
        heapq.heappush(open_set, (start_node.cost + start_node.heuristic, origin_id, start_node))
        
        # Track visited stops and best costs
        visited: Set[str] = set()
        best_cost: Dict[str, float] = {origin_id: 0.0}
        
        weights = preferences.get_weight_factors()
        
        while open_set:
            current_f, current_id, current_node = heapq.heappop(open_set)
            
            # Found destination
            if current_id == destination_id:
                return self._reconstruct_journey(current_node, preferences)
            
            # Skip if already visited
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Explore neighbors
            if current_id not in self.network.graph:
                continue
            
            for neighbor_id, edge_data in self.network.graph[current_id].items():
                if neighbor_id in visited:
                    continue
                
                # Calculate edge cost based on preferences
                edge_cost = self._calculate_edge_cost(edge_data, current_node, weights)
                new_cost = current_node.cost + edge_cost
                
                # Skip if not an improvement
                if neighbor_id in best_cost and new_cost >= best_cost[neighbor_id]:
                    continue
                
                best_cost[neighbor_id] = new_cost
                
                # Create neighbor node
                neighbor_node = PathNode(
                    stop_id=neighbor_id,
                    cost=new_cost,
                    heuristic=self.heuristic(neighbor_id, destination_id) if use_astar else 0.0,
                    parent=current_node,
                    line_id=edge_data.get('line_id')
                )
                
                heapq.heappush(
                    open_set, 
                    (neighbor_node.cost + neighbor_node.heuristic, neighbor_id, neighbor_node)
                )
        
        # No route found
        return None
    
    def _calculate_edge_cost(self, edge_data: Dict, 
                            current_node: PathNode,
                            weights: Dict[str, float]) -> float:
        """
        Calculate cost of traversing an edge based on preferences
        
        Args:
            edge_data: Edge information from network graph
            current_node: Current path node
            weights: Weight factors from preferences
            
        Returns:
            Edge cost (weighted time)
        """
        base_time = edge_data.get('duration_minutes', 5.0)
        transport_type = edge_data.get('transport_type', 'unknown')
        is_transfer = edge_data.get('is_transfer', False)
        walking_distance = edge_data.get('walking_meters', 0)
        
        # Start with base time cost
        cost = base_time * weights['time_weight']
        
        # Add transfer penalty
        if is_transfer:
            cost += 5.0 * weights['transfer_weight']  # 5 min base transfer penalty
        
        # Add walking penalty
        if walking_distance > 0:
            walking_time = walking_distance / 80.0  # 80 m/min walking speed
            cost += walking_time * weights['walking_weight']
        
        # Apply transport type bonuses/penalties
        if transport_type == 'metro':
            cost *= weights['metro_bonus']  # Metro is usually faster/more reliable
        elif transport_type == 'bus':
            cost *= 1.1  # Bus slightly slower due to traffic
        elif transport_type == 'ferry':
            cost *= 1.2  # Ferry has longer wait times
        
        # Penalty for line changes (different from transfers)
        if current_node.line_id and edge_data.get('line_id') != current_node.line_id:
            if not is_transfer:
                cost += 2.0 * weights['transfer_weight']
        
        return cost
    
    def _reconstruct_journey(self, end_node: PathNode, 
                            preferences: RoutePreferences) -> Journey:
        """
        Reconstruct journey from path nodes
        
        Args:
            end_node: Final path node at destination
            preferences: Route preferences
            
        Returns:
            Complete Journey object
        """
        # Trace back through parent nodes
        path_nodes = []
        current = end_node
        while current:
            path_nodes.append(current)
            current = current.parent
        path_nodes.reverse()
        
        # Build journey segments
        segments = []
        transfers = []
        current_line_id = None
        segment_start_idx = 0
        
        for i in range(1, len(path_nodes)):
            prev_node = path_nodes[i - 1]
            curr_node = path_nodes[i]
            
            # Get edge data
            edge_data = self.network.graph.get(prev_node.stop_id, {}).get(curr_node.stop_id, {})
            is_transfer = edge_data.get('is_transfer', False)
            
            # Handle transfers
            if is_transfer:
                transfer = TransferSegment(
                    from_stop=prev_node.stop_id,
                    to_stop=curr_node.stop_id,
                    from_stop_name=self.network.stops[prev_node.stop_id].name,
                    to_stop_name=self.network.stops[curr_node.stop_id].name,
                    transfer_type=edge_data.get('transfer_type', 'walking'),
                    duration_minutes=edge_data.get('duration_minutes', 3),
                    distance_meters=edge_data.get('walking_meters', 100)
                )
                transfers.append(transfer)
                
                # Close current segment if exists
                if current_line_id and segment_start_idx < i - 1:
                    segment = self._create_segment(path_nodes, segment_start_idx, i - 1, current_line_id)
                    if segment:
                        segments.append(segment)
                
                segment_start_idx = i
                current_line_id = None
            else:
                # Regular stop on a line
                new_line_id = edge_data.get('line_id')
                
                # Line change without transfer
                if current_line_id and new_line_id != current_line_id:
                    segment = self._create_segment(path_nodes, segment_start_idx, i - 1, current_line_id)
                    if segment:
                        segments.append(segment)
                    segment_start_idx = i - 1
                
                current_line_id = new_line_id
        
        # Add final segment
        if current_line_id and segment_start_idx < len(path_nodes) - 1:
            segment = self._create_segment(path_nodes, segment_start_idx, len(path_nodes) - 1, current_line_id)
            if segment:
                segments.append(segment)
        
        # Calculate journey metrics
        total_duration = sum(s.duration_minutes for s in segments) + sum(t.duration_minutes for t in transfers)
        total_distance = sum(s.distance_km for s in segments)
        total_walking = sum(t.distance_meters for t in transfers)
        transport_types = {s.transport_type for s in segments}
        
        # Calculate cost (Istanbul public transport: ~15 TL per trip)
        base_cost = 15.0
        additional_transfers_cost = max(0, len(transfers) - 1) * 3.0  # Extra cost for multiple transfers
        estimated_cost = base_cost + additional_transfers_cost
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            segments, transfers, total_duration, total_walking, preferences
        )
        
        origin_stop = self.network.stops[path_nodes[0].stop_id]
        dest_stop = self.network.stops[path_nodes[-1].stop_id]
        
        return Journey(
            origin=path_nodes[0].stop_id,
            destination=path_nodes[-1].stop_id,
            origin_name=origin_stop.name,
            destination_name=dest_stop.name,
            segments=segments,
            transfers=transfers,
            total_duration_minutes=total_duration,
            total_distance_km=total_distance,
            total_transfers=len(transfers),
            total_walking_meters=total_walking,
            estimated_cost_tl=estimated_cost,
            transport_types_used=transport_types,
            quality_score=quality_score
        )
    
    def _create_segment(self, path_nodes: List[PathNode], 
                       start_idx: int, end_idx: int,
                       line_id: str) -> Optional[RouteSegment]:
        """Create a route segment from path nodes"""
        if start_idx >= end_idx:
            return None
        
        start_stop = self.network.stops[path_nodes[start_idx].stop_id]
        end_stop = self.network.stops[path_nodes[end_idx].stop_id]
        
        # Get line information
        line = self.network.lines.get(line_id)
        
        # Calculate segment metrics
        stops_count = end_idx - start_idx
        duration = stops_count * 3  # Assume 3 min between stops
        distance = self.calculate_distance(
            (start_stop.lat, start_stop.lon),
            (end_stop.lat, end_stop.lon)
        )
        
        # Extract line name and transport type
        if line:
            line_name = line.name if hasattr(line, 'name') else f'Line {line_id}'
            transport_type = line.transport_type if hasattr(line, 'transport_type') else 'unknown'
        else:
            line_name = f'Line {line_id}'
            transport_type = 'unknown'
        
        return RouteSegment(
            line_id=line_id,
            line_name=line_name,
            transport_type=transport_type,
            from_stop=start_stop.stop_id,
            to_stop=end_stop.stop_id,
            from_stop_name=start_stop.name,
            to_stop_name=end_stop.name,
            stops_count=stops_count,
            duration_minutes=duration,
            distance_km=distance
        )
    
    def _calculate_quality_score(self, segments: List[RouteSegment],
                                 transfers: List[TransferSegment],
                                 total_duration: int,
                                 total_walking: int,
                                 preferences: RoutePreferences) -> float:
        """
        Calculate overall route quality score (0-1, higher is better)
        """
        score = 1.0
        
        # Penalize for excessive duration (over 60 minutes)
        if total_duration > 60:
            score -= min(0.3, (total_duration - 60) / 200.0)
        
        # Penalize for transfers
        score -= len(transfers) * 0.1
        
        # Penalize for excessive walking
        if total_walking > 500:
            score -= min(0.2, (total_walking - 500) / 2000.0)
        
        # Bonus for using preferred transport (metro)
        if preferences.prefer_metro:
            metro_count = sum(1 for s in segments if s.transport_type == 'metro')
            score += metro_count * 0.05
        
        # Bonus for direct routes (fewer segments)
        if len(segments) == 1:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def find_alternative_routes(self, 
                               origin_id: str,
                               destination_id: str,
                               primary_journey: Journey = None,
                               preferences: RoutePreferences = None,
                               max_alternatives: int = 3) -> List[Journey]:
        """
        Find alternative routes using k-shortest paths approach
        
        Args:
            origin_id: Starting stop ID
            destination_id: Ending stop ID
            primary_journey: Primary journey to compare against
            preferences: Route preferences
            max_alternatives: Maximum number of alternatives
            
        Returns:
            List of alternative Journey objects
        """
        if preferences is None:
            preferences = RoutePreferences()
        
        alternatives = []
        
        # Try different preference variations
        preference_variations = [
            # Minimize transfers
            RoutePreferences(
                minimize_transfers=True,
                minimize_time=False,
                max_transfers=2
            ),
            # Minimize walking
            RoutePreferences(
                minimize_walking=True,
                minimize_transfers=False,
                max_walking_meters=500
            ),
            # Prefer buses over metro
            RoutePreferences(
                prefer_metro=False,
                minimize_time=True
            ),
        ]
        
        for pref in preference_variations:
            if len(alternatives) >= max_alternatives:
                break
            
            journey = self.find_optimal_route(origin_id, destination_id, pref)
            
            if journey and journey not in alternatives:
                # Check if significantly different from primary
                if primary_journey is None or self._is_significantly_different(journey, primary_journey):
                    alternatives.append(journey)
        
        # Sort by quality score
        alternatives.sort(key=lambda j: j.quality_score, reverse=True)
        
        return alternatives[:max_alternatives]
    
    def _is_significantly_different(self, journey1: Journey, journey2: Journey) -> bool:
        """Check if two journeys are significantly different"""
        # Different if using different transport types
        if journey1.transport_types_used != journey2.transport_types_used:
            return True
        
        # Different if duration differs by more than 20%
        duration_diff = abs(journey1.total_duration_minutes - journey2.total_duration_minutes)
        if duration_diff > journey2.total_duration_minutes * 0.2:
            return True
        
        # Different if different number of transfers
        if journey1.total_transfers != journey2.total_transfers:
            return True
        
        return False
