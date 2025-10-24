"""
Full Istanbul Network Loader - Phase 4 Implementation
Loads complete Istanbul transportation network from İBB Open Data
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

from services.route_network_builder import (
    TransportationNetwork, TransportStop, TransportLine, TransferConnection
)

logger = logging.getLogger(__name__)

class IstanbulNetworkLoader:
    """
    Comprehensive Istanbul transportation network loader
    Integrates all İBB Open Data sources for complete city coverage
    """
    
    def __init__(self, ibb_api, use_mock_data=False):
        """
        Initialize network loader
        
        Args:
            ibb_api: IBBOpenDataAPI instance or IBBRealTimeAPI instance
            use_mock_data: Use simulated data if true
        """
        self.ibb_api = ibb_api
        self.use_mock_data = use_mock_data
        self.network = TransportationNetwork()
        self.cache_dir = "cache"
        self.cache_file = os.path.join(self.cache_dir, "istanbul_full_network.json")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Statistics
        self.load_stats = {
            'metro_stops': 0,
            'bus_stops': 0,
            'tram_stops': 0,
            'ferry_stops': 0,
            'metrobus_stops': 0,
            'funicular_stops': 0,
            'total_lines': 0,
            'total_transfers': 0,
            'load_time': 0.0
        }
    
    async def load_full_network(self, force_rebuild: bool = False) -> TransportationNetwork:
        """
        Load complete Istanbul transportation network
        
        Args:
            force_rebuild: Force rebuild even if cache exists
            
        Returns:
            Complete TransportationNetwork
        """
        start_time = datetime.now()
        logger.info("="*70)
        logger.info("LOADING FULL ISTANBUL TRANSPORTATION NETWORK")
        logger.info("="*70)
        
        # Try cache first
        if not force_rebuild and self._load_from_cache():
            logger.info(f"✓ Loaded network from cache: {self._get_stats_summary()}")
            return self.network
        
        logger.info("Building network from İBB Open Data...")
        
        try:
            # Load all transport types in parallel
            await asyncio.gather(
                self._load_metro_system(),
                self._load_bus_system(),
                self._load_tram_system(),
                self._load_ferry_system(),
                self._load_metrobus_system(),
                self._load_funicular_and_cable_cars(),
                self._load_marmaray(),
                return_exceptions=True
            )
            
            # Build network connections
            logger.info("Building network graph...")
            self.network.build_network()
            
            # Create transfer connections
            logger.info("Creating transfer connections...")
            await self._create_major_transfer_hubs()
            await self._create_proximity_transfers()
            
            # Update metadata
            self.network.last_update = datetime.now()
            
            # Calculate statistics
            self.load_stats['load_time'] = (datetime.now() - start_time).total_seconds()
            
            # Save to cache
            self._save_to_cache()
            
            # Log results
            stats = self.network.get_statistics()
            logger.info("="*70)
            logger.info("NETWORK BUILD COMPLETE")
            logger.info("="*70)
            logger.info(f"Total Stops: {stats['total_stops']}")
            logger.info(f"Total Lines: {stats['total_lines']}")
            logger.info(f"Total Edges: {stats['total_edges']}")
            logger.info(f"Total Transfers: {stats['total_transfers']}")
            logger.info(f"Transport Types: {stats['transport_types']}")
            logger.info(f"Build Time: {self.load_stats['load_time']:.2f}s")
            logger.info("="*70)
            
            return self.network
            
        except Exception as e:
            logger.error(f"Error loading full network: {e}", exc_info=True)
            # Return partial network if available
            return self.network
    
    async def _load_metro_system(self):
        """Load complete metro system (M1-M11+)"""
        logger.info("Loading metro system...")
        
        try:
            # Fetch metro data
            metro_data = await self.ibb_api.get_metro_real_time_data()
            
            if not metro_data or 'metro_lines' not in metro_data:
                logger.warning("Metro data not available, using fallback")
                self._load_fallback_metro()
                return
            
            # Process metro lines and stations
            lines_data = metro_data.get('metro_lines', {})
            
            for line_id, line_info in lines_data.items():
                stops = []
                
                # Process stations for this line
                if 'stations' in line_info:
                    for station in line_info['stations']:
                        stop_id = f"metro_{line_id}_{station.get('station_id', station.get('name', ''))}"
                        
                        # Add stop
                        stop = TransportStop(
                            stop_id=stop_id,
                            name=station.get('name', station.get('station_name', '')),
                            lat=float(station.get('lat', station.get('latitude', 41.0))),
                            lon=float(station.get('lon', station.get('longitude', 29.0))),
                            transport_type='metro',
                            lines=[line_id],
                            accessibility=station.get('wheelchair_accessible', False)
                        )
                        self.network.add_stop(stop)
                        stops.append(stop_id)
                        self.load_stats['metro_stops'] += 1
                
                # Add line
                if stops:
                    line = TransportLine(
                        line_id=line_id,
                        name=line_info.get('name', f'Metro {line_id}'),
                        transport_type='metro',
                        stops=stops,
                        color=line_info.get('color')
                    )
                    self.network.add_line(line)
                    self.load_stats['total_lines'] += 1
            
            logger.info(f"✓ Metro: {self.load_stats['metro_stops']} stops loaded")
            
        except Exception as e:
            logger.error(f"Error loading metro system: {e}")
            self._load_fallback_metro()
    
    async def _load_bus_system(self):
        """Load complete bus system (500+ routes)"""
        logger.info("Loading bus system...")
        
        try:
            bus_data = await self.ibb_api.get_bus_real_time_data()
            
            if not bus_data or 'bus_routes' not in bus_data:
                logger.warning("Bus data not available, using fallback")
                self._load_fallback_buses()
                return
            
            routes = bus_data.get('bus_routes', {})
            bus_stops = bus_data.get('bus_stops', {})
            
            # Add all bus stops first
            for stop_id, stop_info in bus_stops.items():
                stop = TransportStop(
                    stop_id=f"bus_{stop_id}",
                    name=stop_info.get('name', stop_info.get('stop_name', '')),
                    lat=float(stop_info.get('lat', 41.0)),
                    lon=float(stop_info.get('lon', 29.0)),
                    transport_type='bus'
                )
                self.network.add_stop(stop)
                self.load_stats['bus_stops'] += 1
            
            # Add bus routes
            for route_id, route_info in routes.items():
                if 'stops' in route_info:
                    stops = [f"bus_{sid}" for sid in route_info['stops']]
                    
                    line = TransportLine(
                        line_id=f"bus_{route_id}",
                        name=route_info.get('name', f'Bus {route_id}'),
                        transport_type='bus',
                        stops=stops
                    )
                    self.network.add_line(line)
                    self.load_stats['total_lines'] += 1
            
            logger.info(f"✓ Bus: {self.load_stats['bus_stops']} stops, {len(routes)} routes loaded")
            
        except Exception as e:
            logger.error(f"Error loading bus system: {e}")
            self._load_fallback_buses()
    
    async def _load_tram_system(self):
        """Load tram system (T1-T5)"""
        logger.info("Loading tram system...")
        
        try:
            # Fetch tram data from İBB API
            # Note: This would use a real İBB endpoint
            tram_lines = await self._fetch_tram_data()
            
            for line_id, line_data in tram_lines.items():
                stops = []
                
                for station in line_data.get('stations', []):
                    stop_id = f"tram_{line_id}_{station['id']}"
                    
                    stop = TransportStop(
                        stop_id=stop_id,
                        name=station['name'],
                        lat=station['lat'],
                        lon=station['lon'],
                        transport_type='tram',
                        lines=[line_id]
                    )
                    self.network.add_stop(stop)
                    stops.append(stop_id)
                    self.load_stats['tram_stops'] += 1
                
                if stops:
                    line = TransportLine(
                        line_id=line_id,
                        name=line_data['name'],
                        transport_type='tram',
                        stops=stops
                    )
                    self.network.add_line(line)
                    self.load_stats['total_lines'] += 1
            
            logger.info(f"✓ Tram: {self.load_stats['tram_stops']} stops loaded")
            
        except Exception as e:
            logger.error(f"Error loading tram system: {e}")
            self._load_fallback_trams()
    
    async def _load_ferry_system(self):
        """Load ferry system (İDO & Şehir Hatları)"""
        logger.info("Loading ferry system...")
        
        try:
            ferry_data = await self.ibb_api.get_ferry_real_time_data()
            
            if not ferry_data or 'ferry_routes' not in ferry_data:
                logger.warning("Ferry data not available, using fallback")
                self._load_fallback_ferries()
                return
            
            routes = ferry_data.get('ferry_routes', {})
            
            for route_id, route_info in routes.items():
                stops = []
                
                for pier in route_info.get('piers', []):
                    stop_id = f"ferry_{pier.get('id', pier.get('name', ''))}"
                    
                    stop = TransportStop(
                        stop_id=stop_id,
                        name=pier.get('name', ''),
                        lat=float(pier.get('lat', 41.0)),
                        lon=float(pier.get('lon', 29.0)),
                        transport_type='ferry'
                    )
                    self.network.add_stop(stop)
                    stops.append(stop_id)
                    self.load_stats['ferry_stops'] += 1
                
                if stops:
                    line = TransportLine(
                        line_id=f"ferry_{route_id}",
                        name=route_info.get('name', f'Ferry {route_id}'),
                        transport_type='ferry',
                        stops=stops
                    )
                    self.network.add_line(line)
                    self.load_stats['total_lines'] += 1
            
            logger.info(f"✓ Ferry: {self.load_stats['ferry_stops']} piers loaded")
            
        except Exception as e:
            logger.error(f"Error loading ferry system: {e}")
            self._load_fallback_ferries()
    
    async def _load_metrobus_system(self):
        """Load Metrobus system"""
        logger.info("Loading Metrobus system...")
        
        try:
            # Metrobus is a specialized BRT system
            metrobus_data = await self._fetch_metrobus_data()
            
            stops = []
            for stop_info in metrobus_data.get('stops', []):
                stop_id = f"metrobus_{stop_info['id']}"
                
                stop = TransportStop(
                    stop_id=stop_id,
                    name=stop_info['name'],
                    lat=stop_info['lat'],
                    lon=stop_info['lon'],
                    transport_type='metrobus'
                )
                self.network.add_stop(stop)
                stops.append(stop_id)
                self.load_stats['metrobus_stops'] += 1
            
            if stops:
                line = TransportLine(
                    line_id='metrobus_34',
                    name='34 Metrobüs',
                    transport_type='metrobus',
                    stops=stops
                )
                self.network.add_line(line)
                self.load_stats['total_lines'] += 1
            
            logger.info(f"✓ Metrobus: {self.load_stats['metrobus_stops']} stops loaded")
            
        except Exception as e:
            logger.error(f"Error loading Metrobus: {e}")
            self._load_fallback_metrobus()
    
    async def _load_funicular_and_cable_cars(self):
        """Load funicular (F1, F2) and cable car systems"""
        logger.info("Loading funicular and cable car systems...")
        
        try:
            # Funicular F1: Kabataş-Taksim
            f1_stops = [
                TransportStop("funicular_f1_kabatas", "Kabataş", lat=41.0387, lon=29.0074, transport_type='funicular'),
                TransportStop("funicular_f1_taksim", "Taksim", lat=41.0370, lon=28.9857, transport_type='funicular')
            ]
            
            for stop in f1_stops:
                self.network.add_stop(stop)
                self.load_stats['funicular_stops'] += 1
            
            self.network.add_line(TransportLine(
                line_id='F1',
                name='F1 Kabataş-Taksim',
                transport_type='funicular',
                stops=['funicular_f1_kabatas', 'funicular_f1_taksim']
            ))
            
            # Funicular F2: Tünel
            f2_stops = [
                TransportStop("funicular_f2_karakoy", "Karaköy", lat=41.0251, lon=28.9741, transport_type='funicular'),
                TransportStop("funicular_f2_beyoglu", "Beyoğlu", lat=41.0329, lon=28.9779, transport_type='funicular')
            ]
            
            for stop in f2_stops:
                self.network.add_stop(stop)
                self.load_stats['funicular_stops'] += 1
            
            self.network.add_line(TransportLine(
                line_id='F2',
                name='F2 Tünel',
                transport_type='funicular',
                stops=['funicular_f2_karakoy', 'funicular_f2_beyoglu']
            ))
            
            self.load_stats['total_lines'] += 2
            logger.info(f"✓ Funicular: {self.load_stats['funicular_stops']} stops loaded")
            
        except Exception as e:
            logger.error(f"Error loading funiculars: {e}")
    
    async def _load_marmaray(self):
        """Load Marmaray cross-continental rail system"""
        logger.info("Loading Marmaray system...")
        
        try:
            # Marmaray is part of the commuter rail system
            # For now, treat main stations as part of metro network
            marmaray_stations = [
                ("Kazlıçeşme", 40.9892, 28.9186),
                ("Yenikapı", 41.0054, 28.9518),
                ("Sirkeci", 41.0175, 28.9764),
                ("Üsküdar", 41.0226, 29.0078),
                ("Ayrılık Çeşmesi", 41.0192, 29.0339),
            ]
            
            stops = []
            for name, lat, lon in marmaray_stations:
                stop_id = f"marmaray_{name.lower().replace(' ', '_')}"
                stop = TransportStop(
                    stop_id=stop_id,
                    name=name,
                    lat=lat,
                    lon=lon,
                    transport_type='rail'
                )
                self.network.add_stop(stop)
                stops.append(stop_id)
            
            self.network.add_line(TransportLine(
                line_id='marmaray',
                name='Marmaray',
                transport_type='rail',
                stops=stops
            ))
            
            self.load_stats['total_lines'] += 1
            logger.info("✓ Marmaray loaded")
            
        except Exception as e:
            logger.error(f"Error loading Marmaray: {e}")
    
    async def _create_major_transfer_hubs(self):
        """Create transfer connections at major hubs"""
        logger.info("Creating major transfer hub connections...")
        
        # Define major transfer hubs in Istanbul
        major_hubs = [
            {
                'name': 'Taksim',
                'stops': ['metro_m2_taksim', 'funicular_f1_taksim', 'bus_taksim'],
                'transfer_time': 3,
                'walking_distance': 100
            },
            {
                'name': 'Yenikapı',
                'stops': ['metro_m1_yenikapi', 'metro_m2_yenikapi', 'marmaray_yenikapi', 'ferry_yenikapi'],
                'transfer_time': 5,
                'walking_distance': 200
            },
            {
                'name': 'Kabataş',
                'stops': ['tram_t1_kabatas', 'funicular_f1_kabatas', 'ferry_kabatas'],
                'transfer_time': 4,
                'walking_distance': 150
            },
            {
                'name': 'Kadıköy',
                'stops': ['ferry_kadikoy', 'bus_kadikoy', 'metro_m4_kadikoy'],
                'transfer_time': 5,
                'walking_distance': 250
            },
        ]
        
        for hub in major_hubs:
            stops = hub['stops']
            # Create transfers between all stop combinations in the hub
            for i in range(len(stops)):
                for j in range(i + 1, len(stops)):
                    if stops[i] in self.network.stops and stops[j] in self.network.stops:
                        self.network.add_transfer(
                            from_stop_id=stops[i],
                            to_stop_id=stops[j],
                            transfer_type='same_station',
                            walking_meters=hub['walking_distance'],
                            duration_minutes=hub['transfer_time']
                        )
                        self.load_stats['total_transfers'] += 1
        
        logger.info(f"✓ Created {self.load_stats['total_transfers']} major hub transfers")
    
    async def _create_proximity_transfers(self, max_distance_m: int = 300):
        """Create walking transfers between nearby stops of different types"""
        logger.info(f"Creating proximity transfers (max {max_distance_m}m)...")
        
        # Get all stops by type
        stops_by_type = {}
        for stop_id, stop in self.network.stops.items():
            if stop.transport_type not in stops_by_type:
                stops_by_type[stop.transport_type] = []
            stops_by_type[stop.transport_type].append(stop)
        
        transfer_count = 0
        
        # Create transfers between different transport types
        for type1, stops1 in stops_by_type.items():
            for type2, stops2 in stops_by_type.items():
                if type1 >= type2:  # Avoid duplicates
                    continue
                
                for stop1 in stops1:
                    for stop2 in stops2:
                        distance = stop1.distance_to(stop2)
                        
                        if distance <= max_distance_m:
                            # Calculate walking time (80m/min average)
                            walking_time = max(2, int(distance / 80))
                            
                            self.network.add_transfer(
                                from_stop_id=stop1.stop_id,
                                to_stop_id=stop2.stop_id,
                                transfer_type='walking',
                                walking_meters=int(distance),
                                duration_minutes=walking_time
                            )
                            transfer_count += 1
        
        logger.info(f"✓ Created {transfer_count} proximity transfers")
        self.load_stats['total_transfers'] += transfer_count
    
    # Fallback methods for when İBB data is unavailable
    
    def _load_fallback_metro(self):
        """Load basic metro data as fallback"""
        logger.info("Loading fallback metro data...")
        # Simplified metro data would go here
        pass
    
    def _load_fallback_buses(self):
        """Load basic bus data as fallback"""
        logger.info("Loading fallback bus data...")
        pass
    
    def _load_fallback_trams(self):
        """Load basic tram data as fallback"""
        logger.info("Loading fallback tram data...")
        pass
    
    def _load_fallback_ferries(self):
        """Load basic ferry data as fallback"""
        logger.info("Loading fallback ferry data...")
        pass
    
    def _load_fallback_metrobus(self):
        """Load basic metrobus data as fallback"""
        logger.info("Loading fallback metrobus data...")
        pass
    
    # Helper methods
    
    async def _fetch_tram_data(self) -> Dict:
        """Fetch tram data from İBB API"""
        # Placeholder - would call real İBB API
        return {}
    
    async def _fetch_metrobus_data(self) -> Dict:
        """Fetch metrobus data from İBB API"""
        # Placeholder - would call real İBB API
        return {}
    
    def _get_stats_summary(self) -> str:
        """Get statistics summary string"""
        return (f"{len(self.network.stops)} stops, "
                f"{len(self.network.lines)} lines, "
                f"{len(self.network.transfers)} transfers")
    
    def _load_from_cache(self) -> bool:
        """Load network from cache file"""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct network from cached data
            # (Simplified - full implementation would deserialize properly)
            logger.info(f"Loading from cache: {self.cache_file}")
            return False  # For now, always rebuild
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False
    
    def _save_to_cache(self):
        """Save network to cache file"""
        try:
            # Save basic statistics for now
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.load_stats,
                'network_stats': self.network.get_statistics()
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Network cached to: {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
