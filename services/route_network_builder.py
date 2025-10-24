"""
Route Network Builder - Build comprehensive transportation network graph from İBB data
Part of Industry-Level Routing Enhancement
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import math
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class TransportStop:
    """Universal stop model for all transport types"""
    stop_id: str
    name: str
    lat: float
    lon: float
    transport_type: str  # metro, bus, tram, ferry, funicular, metrobus
    lines: List[str] = field(default_factory=list)
    accessibility: bool = False
    facilities: List[str] = field(default_factory=list)
    
    def distance_to(self, other: 'TransportStop') -> float:
        """Calculate distance in meters using Haversine formula"""
        R = 6371000  # Earth's radius in meters
        
        lat1, lon1 = math.radians(self.lat), math.radians(self.lon)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


@dataclass
class TransportLine:
    """Universal line model for all transport types"""
    line_id: str
    name: str
    transport_type: str
    stops: List[str] = field(default_factory=list)  # Ordered list of stop IDs
    circular: bool = False
    operational_hours: Optional[Dict] = None
    color: Optional[str] = None


@dataclass
class TransferConnection:
    """Transfer point between stops/lines"""
    from_stop_id: str
    to_stop_id: str
    walking_time: float  # minutes
    walking_distance: float  # meters
    has_elevator: bool = False
    has_escalator: bool = False
    is_step_free: bool = False


@dataclass
class NetworkEdge:
    """Edge in transportation network graph"""
    from_stop_id: str
    to_stop_id: str
    line_id: str
    transport_type: str
    travel_time: float  # minutes
    distance: float  # meters
    edge_type: str  # 'direct', 'transfer', 'walking'


class TransportationNetwork:
    """Comprehensive transportation network graph"""
    
    def __init__(self):
        self.stops: Dict[str, TransportStop] = {}
        self.lines: Dict[str, TransportLine] = {}
        self.transfers: List[TransferConnection] = []
        self.edges: List[NetworkEdge] = []
        
        # Adjacency list for efficient pathfinding
        self.adjacency: Dict[str, List[NetworkEdge]] = {}
        
        # Graph representation for route finding (dict of dict)
        self._graph: Optional[Dict[str, Dict[str, Dict]]] = None
        
        self.last_update: Optional[datetime] = None
    
    def add_stop(self, stop: TransportStop):
        """Add a stop to the network"""
        self.stops[stop.stop_id] = stop
        if stop.stop_id not in self.adjacency:
            self.adjacency[stop.stop_id] = []
    
    def add_line(self, line: TransportLine):
        """Add a line to the network"""
        self.lines[line.line_id] = line
        
    def add_edge(self, edge: NetworkEdge):
        """Add an edge to the network"""
        self.edges.append(edge)
        if edge.from_stop_id not in self.adjacency:
            self.adjacency[edge.from_stop_id] = []
        self.adjacency[edge.from_stop_id].append(edge)
        
        # Clear cached graph when adding edges
        self._graph = None
    
    @property
    def graph(self) -> Dict[str, Dict[str, Dict]]:
        """
        Get graph representation for pathfinding
        Returns dict[from_stop_id][to_stop_id] -> edge data
        """
        if self._graph is None:
            self._build_graph_representation()
        return self._graph
    
    def _build_graph_representation(self):
        """Build graph dict for efficient pathfinding"""
        self._graph = {}
        for edge in self.edges:
            if edge.from_stop_id not in self._graph:
                self._graph[edge.from_stop_id] = {}
            
            self._graph[edge.from_stop_id][edge.to_stop_id] = {
                'line_id': edge.line_id,
                'transport_type': edge.transport_type,
                'duration_minutes': edge.travel_time,
                'distance_meters': edge.distance,
                'is_transfer': edge.edge_type == 'transfer',
                'transfer_type': 'walking' if edge.edge_type == 'transfer' else None,
                'walking_meters': edge.distance if edge.edge_type == 'transfer' else 0
            }
    
    def build_network(self):
        """Build network connections from lines"""
        logger.info("Building network connections...")
        
        # Build edges for each line
        for line_id, line in self.lines.items():
            if not line.stops or len(line.stops) < 2:
                continue
            
            # Create edges between consecutive stops
            for i in range(len(line.stops) - 1):
                from_stop_id = line.stops[i]
                to_stop_id = line.stops[i + 1]
                
                if from_stop_id not in self.stops or to_stop_id not in self.stops:
                    continue
                
                from_stop = self.stops[from_stop_id]
                to_stop = self.stops[to_stop_id]
                
                # Calculate distance and estimated travel time
                distance = from_stop.distance_to(to_stop)
                # Assume 30 km/h average speed, convert to minutes
                travel_time = (distance / 1000.0) / 30.0 * 60.0
                travel_time = max(2.0, travel_time)  # Minimum 2 minutes
                
                # Create forward edge
                edge = NetworkEdge(
                    from_stop_id=from_stop_id,
                    to_stop_id=to_stop_id,
                    line_id=line_id,
                    transport_type=line.transport_type,
                    travel_time=travel_time,
                    distance=distance,
                    edge_type='direct'
                )
                self.add_edge(edge)
                
                # Create reverse edge (bidirectional)
                reverse_edge = NetworkEdge(
                    from_stop_id=to_stop_id,
                    to_stop_id=from_stop_id,
                    line_id=line_id,
                    transport_type=line.transport_type,
                    travel_time=travel_time,
                    distance=distance,
                    edge_type='direct'
                )
                self.add_edge(reverse_edge)
        
        # Clear cached graph to force rebuild
        self._graph = None
        
        logger.info(f"Network built: {len(self.edges)} edges created")
    
    def add_transfer(self, from_stop_id: str, to_stop_id: str, 
                    transfer_type: str = "same_station",
                    walking_meters: int = 100, 
                    duration_minutes: int = 3):
        """
        Simplified add_transfer method that accepts direct parameters
        """
        transfer = TransferConnection(
            from_stop_id=from_stop_id,
            to_stop_id=to_stop_id,
            walking_time=duration_minutes,
            walking_distance=walking_meters,
            is_step_free=(transfer_type == "same_station")
        )
        self.transfers.append(transfer)
        
        # Create transfer edges
        transfer_edge = NetworkEdge(
            from_stop_id=transfer.from_stop_id,
            to_stop_id=transfer.to_stop_id,
            line_id="TRANSFER",
            transport_type="walking",
            travel_time=transfer.walking_time,
            distance=transfer.walking_distance,
            edge_type="transfer"
        )
        self.add_edge(transfer_edge)
        
        # Add reverse direction
        reverse_edge = NetworkEdge(
            from_stop_id=transfer.to_stop_id,
            to_stop_id=transfer.from_stop_id,
            line_id="TRANSFER",
            transport_type="walking",
            travel_time=transfer.walking_time,
            distance=transfer.walking_distance,
            edge_type="transfer"
        )
        self.add_edge(reverse_edge)
        
        # Clear cached graph
        self._graph = None
    
    def get_neighbors(self, stop_id: str) -> List[NetworkEdge]:
        """Get all edges from a stop"""
        return self.adjacency.get(stop_id, [])
    
    def get_stop(self, stop_id: str) -> Optional[TransportStop]:
        """Get stop by ID"""
        return self.stops.get(stop_id)
    
    def find_stops_near(self, lat: float, lon: float, max_distance: float = 500) -> List[Tuple[TransportStop, float]]:
        """Find stops within max_distance meters"""
        temp_stop = TransportStop(stop_id="temp", name="temp", lat=lat, lon=lon, transport_type="temp")
        
        nearby = []
        for stop in self.stops.values():
            distance = temp_stop.distance_to(stop)
            if distance <= max_distance:
                nearby.append((stop, distance))
        
        return sorted(nearby, key=lambda x: x[1])
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        return {
            "total_stops": len(self.stops),
            "total_lines": len(self.lines),
            "total_edges": len(self.edges),
            "total_transfers": len(self.transfers),
            "transport_types": {
                transport_type: len([s for s in self.stops.values() if s.transport_type == transport_type])
                for transport_type in set(s.transport_type for s in self.stops.values())
            },
            "last_update": self.last_update.isoformat() if self.last_update else None
        }


class RouteNetworkBuilder:
    """Builds a complete transportation network from İBB data"""
    
    def __init__(self, ibb_api):
        self.ibb_api = ibb_api
        self.network = TransportationNetwork()
        self.cache_file = "cache/transportation_network.json"
        
    async def build_network(self, force_rebuild: bool = False) -> TransportationNetwork:
        """Build complete transportation network"""
        logger.info("Building transportation network from İBB data...")
        
        # Try to load from cache first
        if not force_rebuild and os.path.exists(self.cache_file):
            logger.info("Loading network from cache...")
            if self._load_from_cache():
                return self.network
        
        try:
            # Load all transport data in parallel
            await asyncio.gather(
                self._load_metro_lines(),
                self._load_bus_routes(),
                self._load_tram_lines(),
                self._load_ferry_routes(),
                self._load_metrobus_routes(),
                self._load_funicular_routes(),
            )
            
            # Create transfer connections
            await self._create_transfer_connections()
            
            # Calculate walking connections between nearby stops
            await self._create_walking_connections()
            
            self.network.last_update = datetime.now()
            
            # Cache the network
            self._save_to_cache()
            
            # Log statistics
            stats = self.network.get_statistics()
            logger.info(f"Network built successfully: {stats}")
            
            return self.network
            
        except Exception as e:
            logger.error(f"Error building network: {e}", exc_info=True)
            raise
    
    async def _load_metro_lines(self):
        """Load all metro lines and stations"""
        logger.info("Loading metro lines...")
        
        try:
            # Get metro lines
            lines_data = await self.ibb_api.fetch_dataset('metro-hatlari')
            stations_data = await self.ibb_api.fetch_dataset('metro-istasyonlari')
            
            if not lines_data or not stations_data:
                logger.warning("Metro data not available")
                return
            
            # Process stations
            for station in stations_data:
                stop = TransportStop(
                    stop_id=f"metro_{station.get('id', station.get('hat_kodu', ''))}__{station.get('istasyon_adi', '')}",
                    name=station.get('istasyon_adi', ''),
                    lat=float(station.get('lat', station.get('enlem', 0))),
                    lon=float(station.get('lon', station.get('boylam', 0))),
                    transport_type='metro',
                    lines=[station.get('hat_kodu', '')],
                    accessibility=station.get('engelli_erisimi', False)
                )
                self.network.add_stop(stop)
            
            # Process lines and create edges
            for line in lines_data:
                line_id = line.get('hat_kodu', '')
                line_obj = TransportLine(
                    line_id=f"metro_{line_id}",
                    name=line.get('hat_adi', ''),
                    transport_type='metro',
                    color=line.get('renk')
                )
                
                # Get stations for this line (ordered)
                line_stations = sorted(
                    [s for s in stations_data if s.get('hat_kodu') == line_id],
                    key=lambda x: x.get('sira', 0)
                )
                
                # Create edges between consecutive stations
                for i in range(len(line_stations) - 1):
                    from_station = line_stations[i]
                    to_station = line_stations[i + 1]
                    
                    from_stop_id = f"metro_{from_station.get('id', from_station.get('hat_kodu', ''))}__{from_station.get('istasyon_adi', '')}"
                    to_stop_id = f"metro_{to_station.get('id', to_station.get('hat_kodu', ''))}__{to_station.get('istasyon_adi', '')}"
                    
                    # Get stops
                    from_stop = self.network.get_stop(from_stop_id)
                    to_stop = self.network.get_stop(to_stop_id)
                    
                    if from_stop and to_stop:
                        distance = from_stop.distance_to(to_stop)
                        # Estimate travel time (average metro speed ~40 km/h = 666 m/min)
                        travel_time = distance / 666
                        
                        # Create bidirectional edges
                        self.network.add_edge(NetworkEdge(
                            from_stop_id=from_stop_id,
                            to_stop_id=to_stop_id,
                            line_id=f"metro_{line_id}",
                            transport_type='metro',
                            travel_time=travel_time,
                            distance=distance,
                            edge_type='direct'
                        ))
                        
                        self.network.add_edge(NetworkEdge(
                            from_stop_id=to_stop_id,
                            to_stop_id=from_stop_id,
                            line_id=f"metro_{line_id}",
                            transport_type='metro',
                            travel_time=travel_time,
                            distance=distance,
                            edge_type='direct'
                        ))
                
                line_obj.stops = [f"metro_{s.get('id', s.get('hat_kodu', ''))}__{s.get('istasyon_adi', '')}" for s in line_stations]
                self.network.add_line(line_obj)
            
            logger.info(f"Loaded {len(lines_data)} metro lines with {len(stations_data)} stations")
            
        except Exception as e:
            logger.error(f"Error loading metro lines: {e}", exc_info=True)
    
    async def _load_bus_routes(self):
        """Load all bus routes and stops"""
        logger.info("Loading bus routes...")
        
        try:
            # Get bus routes and stops
            routes_data = await self.ibb_api.fetch_dataset('otobus-hatlari')
            stops_data = await self.ibb_api.fetch_dataset('otobus-duragi')
            
            if not routes_data or not stops_data:
                logger.warning("Bus data not available")
                return
            
            # Process stops
            for stop in stops_data:
                stop_id = f"bus_{stop.get('OBJECTID', stop.get('durak_kodu', ''))}"
                
                # Skip if already exists
                if stop_id in self.network.stops:
                    continue
                
                stop_obj = TransportStop(
                    stop_id=stop_id,
                    name=stop.get('durak_adi', stop.get('ADI', '')),
                    lat=float(stop.get('lat', stop.get('ENLEM', 0))),
                    lon=float(stop.get('lon', stop.get('BOYLAM', 0))),
                    transport_type='bus'
                )
                self.network.add_stop(stop_obj)
            
            # Process routes (limit to first 100 for performance, expand later)
            for route in routes_data[:100]:
                line_id = f"bus_{route.get('hat_kodu', route.get('HAT_KODU', ''))}"
                line_obj = TransportLine(
                    line_id=line_id,
                    name=route.get('hat_adi', route.get('HAT_ADI', '')),
                    transport_type='bus'
                )
                self.network.add_line(line_obj)
            
            logger.info(f"Loaded {min(100, len(routes_data))} bus routes with {len(stops_data)} stops")
            
        except Exception as e:
            logger.error(f"Error loading bus routes: {e}", exc_info=True)
    
    async def _load_tram_lines(self):
        """Load all tram lines and stations"""
        logger.info("Loading tram lines...")
        
        try:
            tram_data = await self.ibb_api.fetch_dataset('tramvay-hatlari')
            
            if not tram_data:
                logger.warning("Tram data not available")
                return
            
            # Process tram lines and stations
            for line in tram_data:
                line_id = f"tram_{line.get('hat_kodu', '')}"
                line_obj = TransportLine(
                    line_id=line_id,
                    name=line.get('hat_adi', ''),
                    transport_type='tram'
                )
                self.network.add_line(line_obj)
            
            logger.info(f"Loaded {len(tram_data)} tram lines")
            
        except Exception as e:
            logger.error(f"Error loading tram lines: {e}", exc_info=True)
    
    async def _load_ferry_routes(self):
        """Load all ferry routes and terminals"""
        logger.info("Loading ferry routes...")
        
        try:
            ferry_data = await self.ibb_api.fetch_dataset('vapur-hatlari')
            schedules_data = await self.ibb_api.fetch_dataset('vapur-seferleri')
            
            if not ferry_data:
                logger.warning("Ferry data not available")
                return
            
            # Process ferry routes
            for route in ferry_data:
                line_id = f"ferry_{route.get('hat_kodu', route.get('HAT_KODU', ''))}"
                line_obj = TransportLine(
                    line_id=line_id,
                    name=route.get('hat_adi', route.get('HAT_ADI', '')),
                    transport_type='ferry'
                )
                self.network.add_line(line_obj)
            
            logger.info(f"Loaded {len(ferry_data)} ferry routes")
            
        except Exception as e:
            logger.error(f"Error loading ferry routes: {e}", exc_info=True)
    
    async def _load_metrobus_routes(self):
        """Load Metrobus routes and stops"""
        logger.info("Loading Metrobus routes...")
        
        try:
            metrobus_data = await self.ibb_api.fetch_dataset('metrobus-hatlari')
            stops_data = await self.ibb_api.fetch_dataset('metrobus-durakları')
            
            if not metrobus_data:
                logger.warning("Metrobus data not available")
                return
            
            # Process Metrobus stops
            if stops_data:
                for stop in stops_data:
                    stop_id = f"metrobus_{stop.get('durak_kodu', '')}"
                    stop_obj = TransportStop(
                        stop_id=stop_id,
                        name=stop.get('durak_adi', ''),
                        lat=float(stop.get('lat', 0)),
                        lon=float(stop.get('lon', 0)),
                        transport_type='metrobus'
                    )
                    self.network.add_stop(stop_obj)
            
            # Process Metrobus line
            for route in metrobus_data:
                line_id = f"metrobus_{route.get('hat_kodu', '34')}"
                line_obj = TransportLine(
                    line_id=line_id,
                    name=route.get('hat_adi', 'Metrobüs'),
                    transport_type='metrobus'
                )
                self.network.add_line(line_obj)
            
            logger.info(f"Loaded Metrobus with {len(stops_data) if stops_data else 0} stops")
            
        except Exception as e:
            logger.error(f"Error loading Metrobus routes: {e}", exc_info=True)
    
    async def _load_funicular_routes(self):
        """Load funicular routes (F1, F2)"""
        logger.info("Loading funicular routes...")
        
        try:
            # Funicular data might be in metro data or separate
            # For now, add known funiculars manually
            funiculars = [
                {
                    'line_id': 'funicular_F1',
                    'name': 'Kabataş-Taksim Füniküler',
                    'stops': [
                        {'name': 'Kabataş', 'lat': 41.0383, 'lon': 28.9944},
                        {'name': 'Taksim', 'lat': 41.0369, 'lon': 29.0096}
                    ]
                },
                {
                    'line_id': 'funicular_F2',
                    'name': 'Karaköy-Tünel Füniküler',
                    'stops': [
                        {'name': 'Karaköy', 'lat': 41.0236, 'lon': 28.9744},
                        {'name': 'Tünel', 'lat': 41.0257, 'lon': 28.9742}
                    ]
                }
            ]
            
            for funicular in funiculars:
                line_obj = TransportLine(
                    line_id=funicular['line_id'],
                    name=funicular['name'],
                    transport_type='funicular'
                )
                
                # Add stops
                for stop_data in funicular['stops']:
                    stop_id = f"{funicular['line_id']}_{stop_data['name']}"
                    stop_obj = TransportStop(
                        stop_id=stop_id,
                        name=stop_data['name'],
                        lat=stop_data['lat'],
                        lon=stop_data['lon'],
                        transport_type='funicular',
                        lines=[funicular['line_id']]
                    )
                    self.network.add_stop(stop_obj)
                    line_obj.stops.append(stop_id)
                
                # Create edges
                if len(line_obj.stops) == 2:
                    from_stop = self.network.get_stop(line_obj.stops[0])
                    to_stop = self.network.get_stop(line_obj.stops[1])
                    
                    if from_stop and to_stop:
                        distance = from_stop.distance_to(to_stop)
                        travel_time = 3  # Funiculars are fast, ~3 minutes
                        
                        # Bidirectional
                        self.network.add_edge(NetworkEdge(
                            from_stop_id=from_stop.stop_id,
                            to_stop_id=to_stop.stop_id,
                            line_id=funicular['line_id'],
                            transport_type='funicular',
                            travel_time=travel_time,
                            distance=distance,
                            edge_type='direct'
                        ))
                        
                        self.network.add_edge(NetworkEdge(
                            from_stop_id=to_stop.stop_id,
                            to_stop_id=from_stop.stop_id,
                            line_id=funicular['line_id'],
                            transport_type='funicular',
                            travel_time=travel_time,
                            distance=distance,
                            edge_type='direct'
                        ))
                
                self.network.add_line(line_obj)
            
            logger.info(f"Loaded {len(funiculars)} funicular routes")
            
        except Exception as e:
            logger.error(f"Error loading funicular routes: {e}", exc_info=True)
    
    async def _create_transfer_connections(self):
        """Create transfer connections between lines at interchange stations"""
        logger.info("Creating transfer connections...")
        
        try:
            # Find stops with same or very similar names (potential transfer points)
            stop_groups: Dict[str, List[TransportStop]] = {}
            
            for stop in self.network.stops.values():
                # Normalize name for grouping
                normalized_name = stop.name.lower().strip()
                if normalized_name not in stop_groups:
                    stop_groups[normalized_name] = []
                stop_groups[normalized_name].append(stop)
            
            # Create transfers for grouped stops
            transfer_count = 0
            for name, stops in stop_groups.items():
                if len(stops) < 2:
                    continue
                
                # Create transfers between all combinations
                for i, stop1 in enumerate(stops):
                    for stop2 in stops[i+1:]:
                        # Calculate walking distance
                        distance = stop1.distance_to(stop2)
                        
                        # Only create transfer if within reasonable walking distance (< 500m)
                        if distance < 500:
                            # Estimate walking time (average walking speed ~80 m/min)
                            walking_time = max(2, distance / 80)  # Minimum 2 minutes
                            
                            transfer = TransferConnection(
                                from_stop_id=stop1.stop_id,
                                to_stop_id=stop2.stop_id,
                                walking_time=walking_time,
                                walking_distance=distance,
                                is_step_free=stop1.accessibility and stop2.accessibility
                            )
                            self.network.add_transfer(transfer)
                            transfer_count += 1
            
            logger.info(f"Created {transfer_count} transfer connections")
            
        except Exception as e:
            logger.error(f"Error creating transfer connections: {e}", exc_info=True)
    
    async def _create_walking_connections(self):
        """Create walking connections between nearby stops"""
        logger.info("Creating walking connections...")
        
        try:
            # For each stop, find nearby stops within walking distance
            walking_distance_threshold = 300  # 300 meters
            connection_count = 0
            
            stops_list = list(self.network.stops.values())
            
            for i, stop1 in enumerate(stops_list):
                # Only check stops not yet processed to avoid duplicates
                for stop2 in stops_list[i+1:]:
                    distance = stop1.distance_to(stop2)
                    
                    # Create walking connection if within threshold
                    if distance <= walking_distance_threshold:
                        walking_time = distance / 80  # 80 m/min walking speed
                        
                        # Create bidirectional edges
                        self.network.add_edge(NetworkEdge(
                            from_stop_id=stop1.stop_id,
                            to_stop_id=stop2.stop_id,
                            line_id="WALKING",
                            transport_type='walking',
                            travel_time=walking_time,
                            distance=distance,
                            edge_type='walking'
                        ))
                        
                        self.network.add_edge(NetworkEdge(
                            from_stop_id=stop2.stop_id,
                            to_stop_id=stop1.stop_id,
                            line_id="WALKING",
                            transport_type='walking',
                            travel_time=walking_time,
                            distance=distance,
                            edge_type='walking'
                        ))
                        
                        connection_count += 1
            
            logger.info(f"Created {connection_count} walking connections")
            
        except Exception as e:
            logger.error(f"Error creating walking connections: {e}", exc_info=True)
    
    def _save_to_cache(self):
        """Save network to cache file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                'stops': {k: v.__dict__ for k, v in self.network.stops.items()},
                'lines': {k: v.__dict__ for k, v in self.network.lines.items()},
                'transfers': [t.__dict__ for t in self.network.transfers],
                'edges': [e.__dict__ for e in self.network.edges],
                'last_update': self.network.last_update.isoformat() if self.network.last_update else None
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Network cached to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}", exc_info=True)
    
    def _load_from_cache(self) -> bool:
        """Load network from cache file"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Reconstruct network
            self.network = TransportationNetwork()
            
            # Load stops
            for stop_id, stop_data in cache_data.get('stops', {}).items():
                stop = TransportStop(**stop_data)
                self.network.add_stop(stop)
            
            # Load lines
            for line_id, line_data in cache_data.get('lines', {}).items():
                line = TransportLine(**line_data)
                self.network.add_line(line)
            
            # Load edges
            for edge_data in cache_data.get('edges', []):
                edge = NetworkEdge(**edge_data)
                self.network.add_edge(edge)
            
            # Load transfers
            for transfer_data in cache_data.get('transfers', []):
                transfer = TransferConnection(**transfer_data)
                self.network.transfers.append(transfer)
            
            # Load timestamp
            if cache_data.get('last_update'):
                self.network.last_update = datetime.fromisoformat(cache_data['last_update'])
            
            logger.info("Network loaded from cache successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}", exc_info=True)
            return False


# Example usage
async def main():
    """Example usage of RouteNetworkBuilder"""
    from ibb_real_time_api import IBBRealTimeAPI
    
    # Initialize API
    ibb_api = IBBRealTimeAPI(use_mock_data=False)
    
    # Build network
    builder = RouteNetworkBuilder(ibb_api)
    network = await builder.build_network(force_rebuild=True)
    
    # Print statistics
    stats = network.get_statistics()
    print("\n=== Transportation Network Statistics ===")
    print(f"Total Stops: {stats['total_stops']}")
    print(f"Total Lines: {stats['total_lines']}")
    print(f"Total Edges: {stats['total_edges']}")
    print(f"Total Transfers: {stats['total_transfers']}")
    print("\nStops by Transport Type:")
    for transport_type, count in stats['transport_types'].items():
        print(f"  {transport_type}: {count}")
    print(f"\nLast Updated: {stats['last_update']}")
    
    # Test finding nearby stops
    # Example: Find stops near Taksim Square
    taksim_lat, taksim_lon = 41.0369, 29.0096
    nearby_stops = network.find_stops_near(taksim_lat, taksim_lon, max_distance=500)
    
    print(f"\n=== Stops within 500m of Taksim Square ===")
    for stop, distance in nearby_stops[:5]:
        print(f"{stop.name} ({stop.transport_type}) - {distance:.0f}m")


if __name__ == "__main__":
    asyncio.run(main())
