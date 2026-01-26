"""
Route Builder - Route Creation and Formatting
=============================================

Creates TransitRoute objects with:
- Step-by-step directions
- Time and distance calculations
- Transfer handling
- Walking segments

Author: AI Istanbul Team
Date: December 2024
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .destination_types import haversine_distance

logger = logging.getLogger(__name__)


@dataclass
class TransitStation:
    """A single transit station"""
    name: str
    line: str
    lat: float
    lon: float
    transfers: List[str]


@dataclass
class TransitRoute:
    """A complete route between two points"""
    origin: str
    destination: str
    total_time: int  # minutes
    total_distance: float  # km
    steps: List[Dict[str, Any]]
    transfers: int
    lines_used: List[str]
    alternatives: List['TransitRoute']
    time_confidence: str = "medium"
    ranking_scores: Dict[str, float] = None
    origin_gps: Optional[Dict[str, float]] = None
    destination_gps: Optional[Dict[str, float]] = None
    walking_segments: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        if self.ranking_scores is None:
            self.ranking_scores = {}


class RouteBuilder:
    """Builds formatted routes from pathfinding results."""
    
    def __init__(self, stations: Dict[str, TransitStation], travel_time_db, station_normalizer):
        self.stations = stations
        self.travel_time_db = travel_time_db
        self.station_normalizer = station_normalizer
    
    def build_route_from_path_weighted(
        self,
        path: List[str],
        lines_used: List[str],
        transfers: int,
        total_time: float,
        confidences: List[str]
    ) -> Optional[TransitRoute]:
        """Build a TransitRoute from a Dijkstra path with REAL travel times."""
        if not path or len(path) < 2:
            return None
        
        origin_station = self.stations[path[0]]
        dest_station = self.stations[path[-1]]
        
        if any(self.stations[sid].line.upper() == "FERRY" for sid in path):
            logger.info(f"🛳️ Building ferry route: {origin_station.name} → {dest_station.name}")
        
        steps = []
        current_line = self.stations[path[0]].line
        segment_start = 0
        segment_time = 0.0
        overall_confidences = []
        
        for i in range(1, len(path)):
            station_id = path[i]
            station = self.stations[station_id]
            prev_station_id = path[i-1]
            
            if i < len(confidences) + 1:
                travel_time, confidence = self.travel_time_db.get_travel_time(
                    prev_station_id, station_id
                )
                overall_confidences.append(confidence)
            else:
                travel_time = 2.5
                confidence = "low"
            
            if station.line != current_line:
                start_station = self.stations[path[segment_start]]
                end_station = self.stations[path[i-1]]
                
                if start_station.name != end_station.name:
                    stops_count = i - segment_start - 1
                    is_ferry = current_line.upper() == "FERRY"
                    
                    steps.append({
                        "instruction": f"Take {current_line} from {start_station.name} to {end_station.name}",
                        "line": current_line,
                        "from": start_station.name,
                        "to": end_station.name,
                        "duration": round(segment_time, 1),
                        "type": "transit",
                        "stops": None if is_ferry else stops_count,
                        "ferry_crossing": is_ferry
                    })
                
                transfer_penalty = self.travel_time_db.get_transfer_penalty(
                    current_line, station.line
                )
                
                if transfer_penalty > 0 and current_line != station.line:
                    steps.append({
                        "instruction": f"Transfer to {station.line} at {end_station.name}",
                        "line": station.line,
                        "from": end_station.name,
                        "to": end_station.name,
                        "duration": round(transfer_penalty, 1),
                        "type": "transfer"
                    })
                
                current_line = station.line
                segment_start = i
                segment_time = 0.0
            else:
                segment_time += travel_time
        
        # Final segment
        start_station = self.stations[path[segment_start]]
        end_station = self.stations[path[-1]]
        
        if start_station.name != end_station.name:
            final_stops_count = len(path) - segment_start - 1
            is_ferry_final = current_line.upper() == "FERRY"
            
            steps.append({
                "instruction": f"Take {current_line} from {start_station.name} to {end_station.name}",
                "line": current_line,
                "from": start_station.name,
                "to": end_station.name,
                "duration": round(segment_time, 1),
                "type": "transit",
                "stops": None if is_ferry_final else final_stops_count,
                "ferry_crossing": is_ferry_final
            })
        
        # Remove unnecessary transfers
        steps = [s for s in steps if not (s.get('type') == 'transfer' and s.get('from') == s.get('to') and steps.index(s) in [0, len(steps)-1])]
        
        # Calculate distance
        total_distance = self._calculate_total_distance(steps)
        
        # Determine confidence
        if not overall_confidences:
            time_confidence = "medium"
        elif all(c == "high" for c in overall_confidences):
            time_confidence = "high"
        elif any(c == "low" for c in overall_confidences):
            time_confidence = "low"
        else:
            time_confidence = "medium"
        
        actual_transfers = sum(1 for step in steps if step.get('type') == 'transfer')
        
        return TransitRoute(
            origin=self.stations[path[0]].name,
            destination=self.stations[path[-1]].name,
            total_time=round(total_time),
            total_distance=round(total_distance, 2),
            steps=steps,
            transfers=actual_transfers,
            lines_used=list(set(lines_used)),
            alternatives=[],
            time_confidence=time_confidence
        )
    
    def _calculate_total_distance(self, steps: List[Dict]) -> float:
        """Calculate total distance from steps."""
        total_distance = 0.0
        
        for step in steps:
            if step.get('type') != 'transit':
                continue
            
            from_name = step.get('from')
            to_name = step.get('to')
            line = step.get('line')
            
            from_id = None
            to_id = None
            for sid, st in self.stations.items():
                if st.line == line and st.name == from_name:
                    from_id = sid
                if st.line == line and st.name == to_name:
                    to_id = sid
            
            if not from_id or not to_id:
                total_distance += (step.get('duration', 0) / 10.0) * 1.5
                continue
            
            from_st = self.stations[from_id]
            to_st = self.stations[to_id]
            
            if step.get('ferry_crossing'):
                seg_dist = haversine_distance(
                    from_st.lat, from_st.lon, to_st.lat, to_st.lon
                )
                total_distance += seg_dist
            else:
                line_stations = self.station_normalizer.get_stations_on_line_in_order(line)
                try:
                    from_idx = line_stations.index(from_id)
                    to_idx = line_stations.index(to_id)
                    start_idx = min(from_idx, to_idx)
                    end_idx = max(from_idx, to_idx)
                    
                    seg_dist = 0.0
                    for i in range(start_idx, end_idx):
                        s1 = self.stations[line_stations[i]]
                        s2 = self.stations[line_stations[i + 1]]
                        seg_dist += haversine_distance(s1.lat, s1.lon, s2.lat, s2.lon)
                    
                    total_distance += seg_dist
                except (ValueError, IndexError):
                    seg_dist = haversine_distance(
                        from_st.lat, from_st.lon, to_st.lat, to_st.lon
                    )
                    total_distance += seg_dist
        
        if total_distance == 0:
            total_distance = 5.0  # Default fallback
        
        return total_distance
    
    def create_walking_route(self, origin: str, destination: str, walk_time: int) -> TransitRoute:
        """Create a walking-only route for nearby destinations."""
        return TransitRoute(
            origin=origin,
            destination=destination,
            total_time=walk_time,
            total_distance=walk_time * 0.08,
            steps=[{
                'type': 'walk',
                'instruction': f"Walk to {destination}",
                'duration': walk_time,
                'distance': walk_time * 80,
                'details': f"The destination is within walking distance ({walk_time} minutes)."
            }],
            transfers=0,
            lines_used=['WALK'],
            alternatives=[],
            time_confidence='high'
        )
    
    def create_deprecation_route(self, origin: str, destination: str, message: str) -> TransitRoute:
        """Create a route with deprecation warning."""
        return TransitRoute(
            origin=origin,
            destination=destination,
            total_time=0,
            total_distance=0.0,
            steps=[{
                'type': 'warning',
                'instruction': message,
                'duration': 0,
                'details': message
            }],
            transfers=0,
            lines_used=[],
            alternatives=[],
            time_confidence='low'
        )
    
    def create_arrival_route(self, station: TransitStation) -> TransitRoute:
        """Create route for when origin and destination are the same."""
        return TransitRoute(
            origin=station.name,
            destination=station.name,
            total_time=0,
            total_distance=0.0,
            steps=[{
                "instruction": f"You are already at {station.name}",
                "line": station.line,
                "from": station.name,
                "to": station.name,
                "duration": 0,
                "type": "arrival"
            }],
            transfers=0,
            lines_used=[station.line],
            alternatives=[]
        )
    
    def format_directions_english(self, route: TransitRoute) -> str:
        """Format directions in English."""
        lines = [
            f"🚇 **{route.origin} → {route.destination}**",
            f"⏱️ {route.total_time} min | 🔄 {route.transfers} transfer(s)",
            ""
        ]
        
        for i, step in enumerate(route.steps, 1):
            if step['type'] == 'transit':
                from_station = step.get('from', 'Start')
                to_station = step.get('to', 'End')
                line = step.get('line', '')
                duration = step.get('duration', 0)
                lines.append(f"{i}. **{from_station}** → **{to_station}** ({line}, {duration} min)")
            elif step['type'] == 'transfer':
                from_station = step.get('from', '')
                to_line = step.get('line', '')
                duration = step.get('duration', 0)
                lines.append(f"{i}. 🔄 Transfer at **{from_station}** to {to_line} ({duration} min)")
        
        return "\n".join(lines)
    
    def format_directions_turkish(self, route: TransitRoute) -> str:
        """Format directions in Turkish."""
        lines = [
            f"🚇 **{route.origin} → {route.destination}**",
            f"⏱️ {route.total_time} dk | 🔄 {route.transfers} aktarma",
            ""
        ]
        
        for i, step in enumerate(route.steps, 1):
            if step['type'] == 'transit':
                from_station = step.get('from', 'Başlangıç')
                to_station = step.get('to', 'Varış')
                line = step.get('line', '')
                duration = step.get('duration', 0)
                lines.append(f"{i}. **{from_station}** → **{to_station}** ({line}, {duration} dk)")
            elif step['type'] == 'transfer':
                from_station = step.get('from', '')
                to_line = step.get('line', '')
                duration = step.get('duration', 0)
                lines.append(f"{i}. 🔄 **{from_station}**'da {to_line} hattına aktarma ({duration} dk)")
        
        return "\n".join(lines)
    
    def route_to_dict(self, route: TransitRoute) -> Dict[str, Any]:
        """Convert TransitRoute to dictionary for caching."""
        return {
            'origin': route.origin,
            'destination': route.destination,
            'total_time': route.total_time,
            'total_distance': route.total_distance,
            'steps': route.steps,
            'transfers': route.transfers,
            'lines_used': route.lines_used,
            'alternatives': [self.route_to_dict(alt) for alt in route.alternatives] if route.alternatives else [],
            'time_confidence': route.time_confidence,
            'ranking_scores': route.ranking_scores
        }
    
    def dict_to_route(self, data: Dict[str, Any]) -> TransitRoute:
        """Convert dictionary to TransitRoute (from cache)."""
        alternatives = []
        if data.get('alternatives'):
            alternatives = [self.dict_to_route(alt) for alt in data['alternatives']]
        
        return TransitRoute(
            origin=data['origin'],
            destination=data['destination'],
            total_time=data['total_time'],
            total_distance=data['total_distance'],
            steps=data['steps'],
            transfers=data['transfers'],
            lines_used=data['lines_used'],
            alternatives=alternatives,
            time_confidence=data.get('time_confidence', 'medium'),
            ranking_scores=data.get('ranking_scores')
        )
