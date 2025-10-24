"""
Phase 4: İBB Data Integration - Real Istanbul Transportation Network Loader
Loads complete Istanbul transportation data from İBB Open Data Portal
"""

import asyncio
import aiohttp
import ssl
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

from ibb_real_time_api import IBBRealTimeAPI
from services.route_network_builder import (
    TransportationNetwork, TransportStop, TransportLine,
    RouteNetworkBuilder
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RealIBBNetworkLoader:
    """Loads real Istanbul transportation network from İBB Open Data"""
    
    def __init__(self):
        self.ibb_api = IBBRealTimeAPI()
        self.network = TransportationNetwork()
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        self.stats = {
            'stops_loaded': 0,
            'lines_loaded': 0,
            'transfers_created': 0,
            'data_sources': []
        }
    
    async def load_complete_network(self) -> TransportationNetwork:
        """Load complete Istanbul transportation network from İBB"""
        
        logger.info("="*70)
        logger.info("🌍 LOADING REAL ISTANBUL TRANSPORTATION NETWORK")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load stops first (needed for route matching)
            await asyncio.gather(
                self.load_iett_bus_stops(),
                self.load_ferry_stations(),
                self.load_metro_data(),
                return_exceptions=True
            )
            
            logger.info(f"\n✓ Stops loaded: {len(self.network.stops)}")
            
            # Step 2: Load routes (must be after stops for matching)
            logger.info("\n🔗 Loading route/line data...")
            await self.load_iett_bus_routes()
            
            logger.info(f"✓ Lines loaded: {len(self.network.lines)}")
            
            # Step 3: Build network connections (edges between stops)
            self.network.build_network()
            
            # Create transfer points
            await self.create_major_transfer_hubs()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info("="*70)
            logger.info(f"✓ Network Loading Complete! ({duration:.2f}s)")
            logger.info(f"  Stops: {len(self.network.stops)}")
            logger.info(f"  Lines: {len(self.network.lines)}")
            logger.info(f"  Transfers: {len(self.network.transfers)}")
            logger.info(f"  Edges: {len(self.network.edges)}")
            logger.info("="*70)
            
            return self.network
            
        except Exception as e:
            logger.error(f"Error loading network: {e}")
            raise
    
    async def load_iett_bus_stops(self):
        """Load IETT bus stops from real İBB GeoJSON data"""
        
        logger.info("\\n🚌 Loading İETT Bus Stops...")
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Fetch dataset metadata
                dataset = await self.ibb_api._fetch_dataset('iett_bus_stops')
                
                if not dataset.get('success'):
                    logger.warning(f"Could not fetch İETT bus stops metadata (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    return
                
                pkg = dataset.get('result', {})
                resources = pkg.get('resources', [])
                
                if not resources:
                    logger.warning("No resources found for İETT bus stops")
                    return
                
                # Get the GeoJSON resource
                geojson_resource = resources[0]
                data_url = geojson_resource.get('url')
                
                if not data_url:
                    logger.warning("No data URL found")
                    return
                
                logger.info(f"   Downloading from: {data_url[:60]}... (attempt {attempt + 1})")
                
                # Download and parse GeoJSON with retry
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(data_url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            geojson_data = await response.json()
                            
                            features = geojson_data.get('features', [])
                            logger.info(f"   Found {len(features)} bus stops")
                            
                            # Process each bus stop
                            for feature in features:
                                try:
                                    props = feature.get('properties', {})
                                    geom = feature.get('geometry', {})
                                    coords = geom.get('coordinates', [])
                                    
                                    if len(coords) >= 2:
                                        stop_id = f"BUS_{props.get('OBJECTID', props.get('DURAK_KODU', 'UNKNOWN'))}"
                                        stop_name = props.get('DURAK_ADI', props.get('ADI', 'Unknown Stop'))
                                        
                                        # Coordinates in GeoJSON are [lon, lat]
                                        lon, lat = coords[0], coords[1]
                                        
                                        stop = TransportStop(
                                            stop_id=stop_id,
                                            name=stop_name,
                                            lat=lat,
                                            lon=lon,
                                            transport_type='bus'
                                        )
                                        
                                        self.network.add_stop(stop)
                                        self.stats['stops_loaded'] += 1
                                
                                except Exception as e:
                                    logger.debug(f"Error processing bus stop: {e}")
                                    continue
                            
                            logger.info(f"   ✓ Loaded {self.stats['stops_loaded']} bus stops")
                            self.stats['data_sources'].append('iett_bus_stops')
                            return  # Success, exit retry loop
                        
                        elif response.status == 500 and attempt < max_retries - 1:
                            logger.warning(f"   ⚠️ Server error (HTTP 500), retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            logger.warning(f"Failed to download bus stops: HTTP {response.status}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            return
            
            except Exception as e:
                logger.error(f"Error loading bus stops (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return
    
    async def load_ferry_stations(self):
        """Load ferry/maritime stations from İBB GeoJSON data"""
        
        logger.info("\\n⛴️ Loading Ferry Stations...")
        
        try:
            dataset = await self.ibb_api._fetch_dataset('ferry_stations_vector')
            
            if not dataset.get('success'):
                logger.warning("Could not fetch ferry stations metadata")
                return
            
            pkg = dataset.get('result', {})
            resources = pkg.get('resources', [])
            
            if not resources:
                return
            
            geojson_resource = resources[0]
            data_url = geojson_resource.get('url')
            
            if data_url:
                logger.info(f"   Downloading from: {data_url[:60]}...")
                
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(data_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            geojson_data = await response.json()
                            
                            features = geojson_data.get('features', [])
                            logger.info(f"   Found {len(features)} ferry stations")
                            
                            for feature in features:
                                try:
                                    props = feature.get('properties', {})
                                    geom = feature.get('geometry', {})
                                    coords = geom.get('coordinates', [])
                                    
                                    if len(coords) >= 2:
                                        stop_id = f"FERRY_{props.get('OBJECTID', props.get('ID', 'UNKNOWN'))}"
                                        stop_name = props.get('ISKELE_ADI', props.get('ADI', 'Unknown Pier'))
                                        
                                        lon, lat = coords[0], coords[1]
                                        
                                        stop = TransportStop(
                                            stop_id=stop_id,
                                            name=stop_name,
                                            lat=lat,
                                            lon=lon,
                                            transport_type='ferry'
                                        )
                                        
                                        self.network.add_stop(stop)
                                        self.stats['stops_loaded'] += 1
                                
                                except Exception as e:
                                    continue
                            
                            logger.info(f"   ✓ Loaded {len(features)} ferry stations")
                            self.stats['data_sources'].append('ferry_stations')
        
        except Exception as e:
            logger.warning(f"Error loading ferry stations: {e}")
    
    async def load_metro_data(self):
        """Load metro stations and lines (placeholder - will use İBB data when available)"""
        
        logger.info("\\n🚇 Loading Metro Data...")
        
        # For now, add major metro stations manually
        # In production, this would fetch from İBB metro datasets
        major_metro_stations = [
            ("M1_TAK", "Taksim", 41.0370, 28.9857),
            ("M1_SIS", "Şişli-Mecidiyeköy", 41.0602, 28.9879),
            ("M1_OSM", "Osmanbey", 41.0486, 28.9868),
            ("M1_YEN", "Yenikapı", 41.0054, 28.9518),
            ("M1_KAD", "Kadıköy", 40.9905, 29.0250),
            ("M2_HAC", "Hacıosman", 41.1095, 29.0272),
            ("M2_LEV", "Levent", 41.0782, 29.0070),
            ("M3_KIR", "Kirazlı", 41.0243, 28.8328),
            ("M4_TAV", "Tavşantepe", 40.9898, 29.3254),
            ("M5_USK", "Üsküdar", 41.0226, 29.0078),
            ("M6_LEV", "Levent", 41.0782, 29.0070),
            ("M7_MEC", "Mecidiyeköy", 41.0602, 28.9879),
        ]
        
        for stop_id, name, lat, lon in major_metro_stations:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type='metro'
            )
            self.network.add_stop(stop)
            self.stats['stops_loaded'] += 1
        
        # Add metro lines
        metro_lines = [
            ("M1", "M1 Yenikapı-Hacıosman", ["M1_YEN", "M1_OSM", "M1_SIS", "M1_TAK"]),
            ("M2", "M2 Yenikapı-Hacıosman", ["M1_YEN", "M2_LEV", "M2_HAC"]),
            ("M3", "M3 Kirazlı-Olimpiyat", ["M3_KIR"]),
            ("M4", "M4 Kadıköy-Tavşantepe", ["M1_KAD", "M4_TAV"]),
            ("M5", "M5 Üsküdar-Çekmeköy", ["M5_USK"]),
        ]
        
        for line_id, line_name, stops in metro_lines:
            line = TransportLine(
                line_id=line_id,
                name=line_name,
                transport_type='metro',
                stops=stops
            )
            self.network.add_line(line)
            self.stats['lines_loaded'] += 1
        
        logger.info(f"   ✓ Loaded {len(major_metro_stations)} metro stations")
        logger.info(f"   ✓ Loaded {len(metro_lines)} metro lines")
        self.stats['data_sources'].append('metro_manual')
    
    async def create_major_transfer_hubs(self):
        """Create transfer connections at major hubs"""
        
        logger.info("\\n🔄 Creating Transfer Connections...")
        
        # Major transfer hubs in Istanbul
        transfer_hubs = [
            ("M1_TAK", "BUS_TAK", "Taksim Metro-Bus Transfer"),
            ("M1_YEN", "FERRY_YEN", "Yenikapı Metro-Ferry Transfer"),
            ("M1_KAD", "FERRY_KAD", "Kadıköy Metro-Ferry Transfer"),
            ("M5_USK", "FERRY_USK", "Üsküdar Metro-Ferry Transfer"),
        ]
        
        for stop1_id, stop2_id, description in transfer_hubs:
            if stop1_id in self.network.stops and stop2_id in self.network.stops:
                self.network.add_transfer(
                    from_stop_id=stop1_id,
                    to_stop_id=stop2_id,
                    transfer_type="walking",
                    walking_meters=200,
                    duration_minutes=3
                )
                self.stats['transfers_created'] += 1
        
        logger.info(f"   ✓ Created {self.stats['transfers_created']} transfer connections")
    
    async def load_iett_bus_routes(self):
        """Load İETT bus routes from İBB GeoJSON data"""
        
        logger.info("\\n🚌 Loading İETT Bus Routes (Lines)...")
        
        max_retries = 2
        retry_delay = 3
        
        for attempt in range(max_retries):
            try:
                # Fetch dataset metadata
                dataset = await self.ibb_api._fetch_dataset('iett_bus_routes')
                
                if not dataset.get('success'):
                    logger.warning(f"Could not fetch bus routes metadata (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    return
                
                pkg = dataset.get('result', {})
                resources = pkg.get('resources', [])
                
                if not resources:
                    logger.warning("No resources found for bus routes")
                    return
                
                geojson_resource = resources[0]
                data_url = geojson_resource.get('url')
                
                if not data_url:
                    logger.warning("No data URL found for bus routes")
                    return
                
                logger.info(f"   Downloading routes (this may take a minute)...")
                logger.info(f"   URL: {data_url[:60]}...")
                
                # Download with longer timeout (bus routes file is large)
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(data_url, timeout=aiohttp.ClientTimeout(total=180)) as response:
                        if response.status == 200:
                            logger.info(f"   Parsing route data...")
                            geojson_data = await response.json()
                            
                            features = geojson_data.get('features', [])
                            logger.info(f"   Found {len(features)} bus routes")
                            
                            routes_processed = 0
                            edges_created = 0
                            
                            # Process each route
                            for feature in features:
                                try:
                                    props = feature.get('properties', {})
                                    geom = feature.get('geometry', {})
                                    
                                    route_code = props.get('HAT_KODU', props.get('OBJECTID', f'ROUTE_{routes_processed}'))
                                    route_name = props.get('HAT_ADI', f'Route {route_code}')
                                    
                                    # Create transport line
                                    line = TransportLine(
                                        line_id=f"BUS_{route_code}",
                                        name=route_name,
                                        transport_type='bus'
                                    )
                                    
                                    # Parse route geometry (LineString with coordinates)
                                    if geom.get('type') == 'LineString':
                                        coords = geom.get('coordinates', [])
                                        
                                        # Match route coordinates to nearby stops
                                        # For each point in the route, find nearest stop within 50m
                                        route_stops = []
                                        for coord in coords[::5]:  # Sample every 5th point to reduce processing
                                            lon, lat = coord[0], coord[1]
                                            
                                            # Find nearest bus stop
                                            nearest_stop = self._find_nearest_stop(lat, lon, 'bus', max_distance_m=100)
                                            if nearest_stop and nearest_stop.stop_id not in route_stops:
                                                route_stops.append(nearest_stop.stop_id)
                                        
                                        # Add stops to line
                                        line.stops = route_stops
                                        
                                        # Only add line if it has at least 2 stops
                                        if len(route_stops) >= 2:
                                            self.network.add_line(line)
                                            self.stats['lines_loaded'] += 1
                                            routes_processed += 1
                                            
                                            # Log progress every 50 routes
                                            if routes_processed % 50 == 0:
                                                logger.info(f"   Processed {routes_processed} routes...")
                                
                                except Exception as e:
                                    logger.debug(f"Error processing route: {e}")
                                    continue
                            
                            logger.info(f"   ✓ Loaded {routes_processed} bus routes")
                            self.stats['data_sources'].append('iett_bus_routes')
                            return  # Success
                        
                        elif response.status == 500 and attempt < max_retries - 1:
                            logger.warning(f"   ⚠️ Server error (HTTP 500), retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            logger.warning(f"Failed to download bus routes: HTTP {response.status}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            return
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout downloading bus routes (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"   Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.warning("   Using stops-only mode (routes will be limited)")
                    return
            
            except Exception as e:
                logger.error(f"Error loading bus routes (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return
    
    def _find_nearest_stop(self, lat: float, lon: float, transport_type: str, max_distance_m: float = 100) -> Optional[TransportStop]:
        """Find nearest stop of given type within max distance"""
        
        nearest_stop = None
        min_distance = float('inf')
        
        for stop_id, stop in self.network.stops.items():
            if stop.transport_type != transport_type:
                continue
            
            # Calculate distance
            distance = self._haversine_distance(lat, lon, stop.lat, stop.lon)
            
            if distance < min_distance and distance <= max_distance_m:
                min_distance = distance
                nearest_stop = stop
        
        return nearest_stop
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters between two coordinates"""
        from math import radians, sin, cos, sqrt, asin
        
        R = 6371000  # Earth radius in meters
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c

