"""
Pathfinding - Dijkstra Routing Algorithm
========================================

Weighted Dijkstra pathfinding for Istanbul transit network:
- Multi-modal routing (metro, tram, ferry, bus, funicular, Marmaray)
- Transfer optimization with penalties
- Alternative route discovery
- Route ranking by time/transfers

Author: AI Istanbul Team
Date: December 2024
"""

import heapq
import logging
from typing import Dict, List, Optional, Tuple, Any

from .route_builder import TransitStation, TransitRoute, RouteBuilder

logger = logging.getLogger(__name__)


class Pathfinder:
    """Dijkstra-based pathfinding for Istanbul transit network."""
    
    def __init__(
        self,
        stations: Dict[str, TransitStation],
        travel_time_db,
        station_normalizer,
        route_builder: RouteBuilder
    ):
        self.stations = stations
        self.travel_time_db = travel_time_db
        self.station_normalizer = station_normalizer
        self.route_builder = route_builder
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_transfers: int = 3
    ) -> Optional[TransitRoute]:
        """
        Find the best route using weighted Dijkstra's algorithm.
        
        Uses ACTUAL travel times from travel_time_db instead of uniform estimates.
        """
        if start_id not in self.stations or end_id not in self.stations:
            logger.warning(f"Station not found: {start_id} or {end_id}")
            return None
        
        if start_id == end_id:
            return self.route_builder.create_arrival_route(self.stations[start_id])
        
        # Priority queue: (total_time, station_id, path, lines_used, transfers, confidences)
        heap = [(0, start_id, [start_id], [self.stations[start_id].line], 0, [])]
        best_time_to_station = {start_id: 0}
        best_route = None
        
        while heap:
            current_time, current_id, path, lines_used, transfers, confidences = heapq.heappop(heap)
            
            if current_id == end_id:
                route = self.route_builder.build_route_from_path_weighted(
                    path, lines_used, transfers, current_time, confidences
                )
                if route and (best_route is None or route.total_time < best_route.total_time):
                    best_route = route
                continue
            
            if current_time > best_time_to_station.get(current_id, float('inf')) + 5:
                continue
            
            current_line = self.stations[current_id].line
            
            # 1. Same-line neighbors
            same_line_neighbors = self._get_same_line_neighbors(current_id)
            for neighbor_id in same_line_neighbors:
                if neighbor_id not in path:
                    travel_time, confidence = self.travel_time_db.get_travel_time(
                        current_id, neighbor_id
                    )
                    new_time = current_time + travel_time
                    new_path = path + [neighbor_id]
                    new_confidences = confidences + [confidence]
                    
                    if neighbor_id not in best_time_to_station or new_time < best_time_to_station[neighbor_id]:
                        best_time_to_station[neighbor_id] = new_time
                        heapq.heappush(heap, (
                            new_time,
                            neighbor_id,
                            new_path,
                            lines_used,
                            transfers,
                            new_confidences
                        ))
            
            # 2. Transfer neighbors
            if transfers < max_transfers:
                transfer_neighbors = self._get_transfer_neighbors(current_id)
                for neighbor_id, transfer_line in transfer_neighbors:
                    if neighbor_id not in path:
                        transfer_penalty = self.travel_time_db.get_transfer_penalty(
                            current_line, transfer_line
                        )
                        
                        new_time = current_time + transfer_penalty
                        new_lines = lines_used + [transfer_line]
                        new_path = path + [neighbor_id]
                        new_confidences = confidences + ["high"]
                        
                        if neighbor_id not in best_time_to_station or new_time < best_time_to_station[neighbor_id]:
                            best_time_to_station[neighbor_id] = new_time
                            heapq.heappush(heap, (
                                new_time,
                                neighbor_id,
                                new_path,
                                new_lines,
                                transfers + 1,
                                new_confidences
                            ))
        
        return best_route
    
    def _get_same_line_neighbors(self, station_id: str) -> List[str]:
        """Get only physically adjacent stations on the same line."""
        if station_id not in self.stations:
            return []
        
        current_line = self.stations[station_id].line
        neighbors = []
        
        # Special case: Ferry connections (point-to-point)
        if current_line.upper() == "FERRY":
            for other_id, other_station in self.stations.items():
                if other_station.line.upper() == "FERRY" and other_id != station_id:
                    neighbors.append(other_id)
            return neighbors
        
        try:
            line_station_ids = self.station_normalizer.get_stations_on_line_in_order(current_line)
            
            if not line_station_ids:
                return []
            
            try:
                current_idx = line_station_ids.index(station_id)
                
                if current_idx > 0:
                    neighbors.append(line_station_ids[current_idx - 1])
                
                if current_idx < len(line_station_ids) - 1:
                    neighbors.append(line_station_ids[current_idx + 1])
                    
            except ValueError:
                logger.warning(f"Station {station_id} not found in line {current_line}")
                
        except Exception as e:
            logger.error(f"Error getting same-line neighbors for {station_id}: {e}")
        
        return neighbors
    
    def _get_transfer_neighbors(self, station_id: str) -> List[Tuple[str, str]]:
        """Get all stations reachable by transfer."""
        if station_id not in self.stations:
            return []
        
        current_station = self.stations[station_id]
        neighbors = []
        
        for transfer_line in current_station.transfers:
            for other_id, other_station in self.stations.items():
                if (other_station.line == transfer_line and 
                    other_station.name == current_station.name and
                    other_id != station_id):
                    neighbors.append((other_id, transfer_line))
        
        return neighbors
    
    def find_path_with_penalty(
        self, 
        start_id: str, 
        end_id: str, 
        max_transfers: int,
        transfer_penalty_multiplier: float = 0.5
    ) -> Optional[TransitRoute]:
        """Find alternative path with reduced transfer penalties."""
        original_penalty = getattr(self.travel_time_db, '_transfer_penalty_override', None)
        
        try:
            self.travel_time_db._transfer_penalty_override = transfer_penalty_multiplier
            route = self.find_path(start_id, end_id, max_transfers)
            return route
        except Exception as e:
            logger.debug(f"Alternative path search failed: {e}")
            return None
        finally:
            if original_penalty is not None:
                self.travel_time_db._transfer_penalty_override = original_penalty
            elif hasattr(self.travel_time_db, '_transfer_penalty_override'):
                delattr(self.travel_time_db, '_transfer_penalty_override')
    
    def find_ferry_alternative(self, start_id: str, end_id: str) -> Optional[TransitRoute]:
        """Try to find a route that includes a ferry."""
        try:
            ferry_stations = [sid for sid, st in self.stations.items() if st.line.upper() == "FERRY"]
            
            if not ferry_stations:
                return None
            
            best_ferry_route = None
            best_time = float('inf')
            
            for ferry_id in ferry_stations[:3]:
                route1 = self.find_path(start_id, ferry_id, 2)
                route2 = self.find_path(ferry_id, end_id, 2)
                
                if route1 and route2:
                    total_time = route1.total_time + route2.total_time
                    if total_time < best_time:
                        best_time = total_time
                        combined_steps = route1.steps + route2.steps
                        combined_lines = list(set(route1.lines_used + route2.lines_used))
                        
                        best_ferry_route = TransitRoute(
                            origin=route1.origin,
                            destination=route2.destination,
                            total_time=total_time,
                            total_distance=route1.total_distance + route2.total_distance,
                            steps=combined_steps,
                            transfers=route1.transfers + route2.transfers + 1,
                            lines_used=combined_lines,
                            alternatives=[],
                            time_confidence='medium'
                        )
            
            return best_ferry_route
        except Exception as e:
            logger.debug(f"Ferry alternative search failed: {e}")
            return None
    
    def rank_routes(
        self, 
        routes: List[TransitRoute], 
        origin_gps: Optional[Dict[str, float]] = None
    ) -> List[TransitRoute]:
        """Rank routes by time and transfers."""
        if not routes:
            return []
        
        for route in routes:
            time_score = route.total_time
            transfer_penalty = route.transfers * 10
            
            route.ranking_scores = {
                'time': time_score,
                'transfers': route.transfers,
                'combined': time_score + transfer_penalty
            }
        
        return sorted(routes, key=lambda r: r.ranking_scores.get('combined', float('inf')))
    
    def is_duplicate_route(self, new_route: TransitRoute, existing_routes: List[TransitRoute]) -> bool:
        """Check if a route is essentially a duplicate."""
        if not new_route or not existing_routes:
            return False
        
        new_lines = tuple(new_route.lines_used)
        
        for route in existing_routes:
            if tuple(route.lines_used) == new_lines:
                return True
        
        return False
