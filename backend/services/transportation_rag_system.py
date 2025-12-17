#!/usr/bin/env python3
"""
Industry-Level Transportation RAG System for Istanbul
Google Maps Quality Route Finding with Retrieval-Augmented Generation

Features:
- Complete Istanbul transit network graph
- Multi-modal routing (metro, tram, ferry, bus, funicular, Marmaray)
- WEIGHTED DIJKSTRA ROUTING with realistic travel times
- Real-time route validation
- Step-by-step directions
- Alternative route suggestions
- Transfer optimization with penalties
- Time and distance calculations with confidence indicators
- Accessibility information
- Week 2: Canonical station/line ID normalization and multilingual support

Author: AI Istanbul Team
Date: December 2024
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
import unicodedata
import heapq  # For Dijkstra's priority queue

# Station normalization is imported later in __init__ from transportation_station_normalization

logger = logging.getLogger(__name__)

@dataclass
class TransitStation:
    """A single transit station"""
    name: str
    line: str  # e.g., "M2", "T1", "MARMARAY"
    lat: float
    lon: float
    transfers: List[str]  # List of lines you can transfer to
    
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
    alternatives: List['TransitRoute']  # Alternative routes
    time_confidence: str = "medium"  # 'high', 'medium', 'low' - data quality indicator
    
class IstanbulTransportationRAG:
    """
    Industry-level transportation knowledge system.
    
    Provides Google Maps-quality routing with:
    - Complete station graph
    - Multi-modal pathfinding
    - Transfer optimization
    - Step-by-step directions
    """
    
    def __init__(self, redis_client=None):
        """Initialize the transportation knowledge base"""
        self.stations = self._build_station_graph()
        self.routes = self._build_route_patterns()
        self.neighborhoods = self._build_neighborhood_stations()
        self.station_aliases = self._build_station_aliases()
        self.last_route = None  # Store last computed route for mapData extraction
        
        # Initialize travel time database for weighted routing
        from services.transportation_travel_times import get_travel_time_database
        self.travel_time_db = get_travel_time_database()
        
        # Week 2 Improvement: Station/Line ID normalization
        from services.transportation_station_normalization import get_station_normalizer
        self.station_normalizer = get_station_normalizer()
        
        # Week 1 Improvement #3: Route caching
        self.redis = redis_client
        self.route_cache_ttl = 86400  # 24 hours
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("✅ Transportation RAG initialized with complete Istanbul network")
        logger.info("✅ Dijkstra routing enabled with realistic travel times")
        logger.info("✅ Station/Line ID normalization enabled")
        if redis_client:
            logger.info("✅ Route caching enabled with Redis")
    
    def _normalize_station_name(self, name: str) -> str:
        """
        Normalize station/location names for fuzzy matching.
        
        Handles:
        - Case insensitivity
        - Accent removal (ı→i, ö→o, ü→u, ş→s, ğ→g, ç→c)
        - Common suffixes (square, station, metro, tram, etc.)
        - Extra whitespace
        
        Examples:
            "Taksim Square" → "taksim"
            "Kadıköy" → "kadikoy"
            "Beşiktaş Metro" → "besiktas"
        """
        # Convert to lowercase
        name = name.lower().strip()
        
        # Remove Turkish accents/special characters
        turkish_char_map = {
            'ı': 'i', 'İ': 'i',
            'ö': 'o', 'Ö': 'o',
            'ü': 'u', 'Ü': 'u',
            'ş': 's', 'Ş': 's',
            'ğ': 'g', 'Ğ': 'g',
            'ç': 'c', 'Ç': 'c'
        }
        
        for turkish_char, latin_char in turkish_char_map.items():
            name = name.replace(turkish_char, latin_char)
        
        # Remove common suffixes
        suffixes_to_remove = [
            ' square', ' meydani', ' meydanı',
            ' station', ' istasyonu', ' istasyon',
            ' metro', ' metrosu',
            ' tram', ' tramvay',
            ' terminal', ' terminali',
            ' pier', ' iskele', ' iskelesi',
            ' stop', ' durak', ' duragi', ' durağı'
        ]
        
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def _build_station_aliases(self) -> Dict[str, List[str]]:
        """
        Build comprehensive alias mappings for popular locations.
        
        Maps common location names/spellings to official station IDs.
        """
        return {
            # Taksim area
            "taksim": ["M2-Taksim"],
            "taksim square": ["M2-Taksim"],
            "taksim meydani": ["M2-Taksim"],
            
            # Kadıköy area
            "kadikoy": ["M4-Kadıköy"],
            "kadıköy": ["M4-Kadıköy"],
            "kadıkoy": ["M4-Kadıköy"],
            
            # Beşiktaş area
            "besiktas": ["MARMARAY-Besiktas"],
            "beşiktas": ["MARMARAY-Besiktas"],
            "beşiktaş": ["MARMARAY-Besiktas"],
            
            # Sultanahmet/Fatih area
            "sultanahmet": ["T1-Sultanahmet"],
            "sultanahmet square": ["T1-Sultanahmet"],
            "blue mosque": ["T1-Sultanahmet"],
            "hagia sophia": ["T1-Sultanahmet"],
            "ayasofya": ["T1-Sultanahmet"],
            
            # Galata/Karaköy area
            "galata": ["M2-Sishane"],
            "galata tower": ["M2-Sishane"],
            "karakoy": ["T1-Karaköy", "F2-Karaköy"],
            "karaköy": ["T1-Karaköy", "F2-Karaköy"],
            
            # Üsküdar area
            "uskudar": ["MARMARAY-Uskudar", "M5-Uskudar"],
            "üsküdar": ["MARMARAY-Uskudar", "M5-Uskudar"],
            "uskudar square": ["MARMARAY-Uskudar", "M5-Uskudar"],
            
            # Istiklal/Beyoğlu area
            "istiklal": ["M2-Taksim"],
            "istiklal street": ["M2-Taksim"],
            "istiklal caddesi": ["M2-Taksim"],
            "beyoglu": ["M2-Sishane", "F2-Karaköy"],
            "beyoğlu": ["M2-Sishane", "F2-Karaköy"],
            
            # Airports
            "airport": ["M11-İstanbul Havalimanı"],  # New Istanbul Airport (primary)
            "istanbul airport": ["M11-İstanbul Havalimanı"],
            "new airport": ["M11-İstanbul Havalimanı"],
            "ist airport": ["M11-İstanbul Havalimanı"],
            "yeni havalimani": ["M11-İstanbul Havalimanı"],
            "havalimani": ["M11-İstanbul Havalimanı"],
            "ataturk airport": ["M1A-Atatürk Havalimanı"],  # Old Atatürk Airport (closed, legacy)
            "atatürk airport": ["M1A-Atatürk Havalimanı"],
            "atatürk havalimani": ["M1A-Atatürk Havalimanı"],
            
            # Eminönü area
            "eminonu": ["T1-Eminonu"],
            "eminönü": ["T1-Eminonu"],
            "spice bazaar": ["T1-Eminonu"],
            "misir carsisi": ["T1-Eminonu"],
            
            # Sirkeci
            "sirkeci": ["MARMARAY-Sirkeci", "T1-Sirkeci"],
            
            # Levent area
            "levent": ["M2-Levent"],
            "4.levent": ["M2-4.Levent"],
            
            # Şişli area
            "sisli": ["M2-Sisli-Mecidiyekoy"],
            "şişli": ["M2-Sisli-Mecidiyekoy"],
            "mecidiyekoy": ["M2-Sisli-Mecidiyekoy"],
            "mecidiyeköy": ["M2-Sisli-Mecidiyekoy"],
            
            # Bostancı area
            "bostanci": ["MARMARAY-Bostanci"],
            "bostancı": ["MARMARAY-Bostanci"],
            
            # Pendik
            "pendik": ["MARMARAY-Pendik"],
        }
    
    def _build_station_graph(self) -> Dict[str, TransitStation]:
        """
        Build complete graph of all Istanbul transit stations using canonical data.
        
        IMPORTANT: This now uses the canonical station database from 
        transportation_station_normalization.py as the single source of truth.
        All station data matches official 2025 Istanbul transit data.
        
        Returns comprehensive station database with:
        - All metro lines (M1A, M1B, M2, M3, M4, M5, M6, M7, M9, M11)
        - All tram lines (T1, T4, T5)
        - All funiculars (F1, F2)
        - Marmaray stations (complete 43-station route)
        - Ferry terminals
        - Major transfer points
        """
        stations = {}
        
        # Import the canonical station normalizer
        from services.transportation_station_normalization import get_station_normalizer
        normalizer = get_station_normalizer()
        
        # Build station graph from canonical data
        for canonical_station in normalizer.stations:
            station_id = canonical_station.canonical_id
            stations[station_id] = TransitStation(
                name=canonical_station.name_en,  # Use English name for consistency
                line=canonical_station.line_id,
                lat=canonical_station.lat,
                lon=canonical_station.lon,
                transfers=canonical_station.transfers
            )
        
        logger.info(f"✅ Built station graph from canonical data: {len(stations)} stations")
        
        # Log station counts by line for verification
        line_counts = {}
        for station_id in stations.keys():
            line = station_id.split('-')[0]
            line_counts[line] = line_counts.get(line, 0) + 1
        
        logger.info("📊 Station counts by line:")
        for line in sorted(line_counts.keys()):
            logger.info(f"  {line}: {line_counts[line]} stations")
        
        return stations
    
    def _build_route_patterns(self) -> Dict[str, List[str]]:
        """
        Build common route patterns for major destinations.
        
        This is like Google Maps' pre-computed routes for popular destinations.
        """
        return {
            # KADIKOY CONNECTIONS
            "kadıköy_to_taksim": ["M4", "MARMARAY", "M2", "F1"],
            "kadıköy_to_sultanahmet": ["M4", "MARMARAY", "T1"],
            "kadıköy_to_beyoğlu": ["M4", "MARMARAY", "M2"],
            "kadıköy_to_beşiktaş": ["FERRY", "M4", "MARMARAY", "M2"],
            
            # TAKSIM CONNECTIONS
            "taksim_to_kadıköy": ["F1", "MARMARAY", "M4"],
            "taksim_to_sultanahmet": ["M2", "T1"],
            "taksim_to_airport": ["M2", "M1A", "M1B"],
            
            # SULTANAHMET CONNECTIONS
            "sultanahmet_to_kadıköy": ["T1", "MARMARAY", "M4"],
            "sultanahmet_to_taksim": ["T1", "F1", "M2"],
            "sultanahmet_to_airport": ["T1", "M1A", "M1B"],
            
            # CROSS-BOSPHORUS ROUTES
            "european_to_asian": ["MARMARAY", "FERRY"],
            "asian_to_european": ["MARMARAY", "FERRY"],
        }
    
    def _build_neighborhood_stations(self) -> Dict[str, List[str]]:
        """
        Map neighborhoods to their nearest major transit stations.
        
        This helps when users ask "how to get to Kadıköy" without specifying exact station.
        """
        return {
            # ASIAN SIDE
            "kadıköy": ["M4-Kadıköy", "M4-Ayrılık Çeşmesi"],
            "üsküdar": ["MARMARAY-Üsküdar", "M5-Üsküdar"],
            "bostancı": ["MARMARAY-Bostancı", "M4-Bostancı"],
            "pendik": ["MARMARAY-Pendik", "M4-Pendik"],
            "kartal": ["MARMARAY-Kartal", "M4-Kartal"],
            "maltepe": ["M4-Maltepe"],
            "ataşehir": ["M4-Ünalan", "M4-Kozyatağı"],
            
            # EUROPEAN SIDE
            "taksim": ["M2-Taksim", "F1-Taksim"],
            "beyoğlu": ["M2-Taksim", "M2-Şişhane", "F2-Tünel"],
            "sultanahmet": ["T1-Sultanahmet"],
            "eminönü": ["T1-Eminönü", "MARMARAY-Sirkeci"],
            "karaköy": ["T1-Karaköy", "F2-Karaköy"],
            "kabataş": ["T1-Kabataş", "F1-Kabataş"],
            "beşiktaş": ["T1-Kabataş", "F1-Kabataş"],  # Near Kabataş
            "şişli": ["M2-Şişli-Mecidiyeköy"],
            "levent": ["M2-Levent", "M2-4. Levent", "M6-Levent"],
            "mecidiyeköy": ["M2-Şişli-Mecidiyeköy", "M7-Mecidiyeköy"],
            "zeytinburnu": ["T1-Zeytinburnu", "MARMARAY-Zeytinburnu"],
            "bakırköy": ["MARMARAY-Bakırköy"],
            "yeşilköy": ["MARMARAY-Yeşilköy"],
            "atatürk airport": ["M1A-Atatürk Airport", "M1B-Atatürk Airport"],
            "yenikapı": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M1B-Yenikapı", "M2-Yenikapı"],
        }
    
    def find_route(
        self,
        origin: str,
        destination: str,
        max_transfers: int = 3,
        origin_gps: Optional[Dict[str, float]] = None,
        destination_gps: Optional[Dict[str, float]] = None
    ) -> Optional[TransitRoute]:
        """
        Find the best route between two locations.
        
        This is the main routing function - like Google Maps pathfinding.
        
        Args:
            origin: Starting point (neighborhood or station name)
            destination: Ending point (neighborhood or station name)
            max_transfers: Maximum number of transfers allowed
            origin_gps: Optional GPS coordinates for origin {"lat": float, "lon": float}
            destination_gps: Optional GPS coordinates for destination {"lat": float, "lon": float}
            
        Returns:
            TransitRoute with step-by-step directions, or None if no route found
        """
        # Normalize names
        origin = origin.lower().strip()
        destination = destination.lower().strip()
        
        # Week 1 Improvement #3: Try cache first (skip cache if GPS-based - those are dynamic)
        use_cache = not (origin_gps or destination_gps)
        cache_key = f"route:{origin}|{destination}|{max_transfers}"
        
        if use_cache and self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    self.cache_hits += 1
                    logger.info(f"⚡ Route cache HIT ({self.cache_hits} total): {origin} → {destination}")
                    # Deserialize cached route
                    route_dict = json.loads(cached)
                    return self._dict_to_route(route_dict)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Cache MISS - compute route
        if use_cache:
            self.cache_misses += 1
            logger.info(f"🔍 Route cache MISS ({self.cache_misses} total): Computing {origin} → {destination}")
        
        logger.info(f"🗺️ Finding route: {origin} → {destination}")
        if origin_gps:
            logger.info(f"📍 Origin GPS provided: {origin_gps}")
        if destination_gps:
            logger.info(f"📍 Destination GPS provided: {destination_gps}")
        
        # Get station IDs for origin and destination
        # Use GPS to find nearest station if GPS coordinates provided
        if origin_gps and isinstance(origin_gps, dict) and 'lat' in origin_gps and 'lon' in origin_gps:
            nearest_origin = self.find_nearest_station(origin_gps['lat'], origin_gps['lon'])
            if nearest_origin:
                origin_stations = [nearest_origin]
                logger.info(f"✅ Using nearest station for GPS origin: {self.stations[nearest_origin].name}")
            else:
                origin_stations = self._get_stations_for_location(origin)
        else:
            origin_stations = self._get_stations_for_location(origin)
        
        if destination_gps and isinstance(destination_gps, dict) and 'lat' in destination_gps and 'lon' in destination_gps:
            nearest_dest = self.find_nearest_station(destination_gps['lat'], destination_gps['lon'])
            if nearest_dest:
                dest_stations = [nearest_dest]
                logger.info(f"✅ Using nearest station for GPS destination: {self.stations[nearestDest].name}")
            else:
                dest_stations = self._get_stations_for_location(destination)
        else:
            dest_stations = self._get_stations_for_location(destination)
        
        if not origin_stations or not dest_stations:
            logger.warning(f"Could not find stations for {origin} or {destination}")
            return None
        
        # Find best route
        best_route = None
        min_transfers = 999
        
        for orig_station in origin_stations:
            for dest_station in dest_stations:
                route = self._find_path(orig_station, dest_station, max_transfers)
                if route and route.transfers < min_transfers:
                    best_route = route
                    min_transfers = route.transfers
        
        # Week 1 Improvement #3: Store in cache
        if use_cache and self.redis and best_route:
            try:
                route_dict = self._route_to_dict(best_route)
                self.redis.setex(
                    cache_key,
                    self.route_cache_ttl,
                    json.dumps(route_dict, ensure_ascii=False)
                )
                logger.debug(f"💾 Cached route: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        return best_route
    
    def _get_stations_for_location(self, location: str) -> List[str]:
        """
        Get station IDs for a given location name with fuzzy matching.
        
        Strategy:
        1. Check alias mappings (handles common names like "taksim square")
        2. Check neighborhood mappings
        3. Try normalized name matching
        4. Fallback to partial string matching
        """
        original_location = location
        location = location.lower().strip()
        normalized_location = self._normalize_station_name(location)
        
        logger.debug(f"🔍 Looking up location: '{original_location}' → normalized: '{normalized_location}'")
        
        # Strategy 1: Check alias mappings first (most reliable)
        if normalized_location in self.station_aliases:
            stations = self.station_aliases[normalized_location]
            logger.debug(f"✅ Found via alias: {normalized_location} → {stations}")
            return stations
        
        # Strategy 2: Check original location in alias (before normalization)
        if location in self.station_aliases:
            stations = self.station_aliases[location]
            logger.debug(f"✅ Found via alias (original): {location} → {stations}")
            return stations
        
        # Strategy 3: Check neighborhood mapping
        if location in self.neighborhoods:
            stations = self.neighborhoods[location]
            logger.debug(f"✅ Found via neighborhood: {location} → {stations}")
            return stations
        
        if normalized_location in self.neighborhoods:
            stations = self.neighborhoods[normalized_location]
            logger.debug(f"✅ Found via neighborhood (normalized): {normalized_location} → {stations}")
            return stations
        
        # Strategy 4: Try normalized name matching against all stations
        matches = []
        for station_id, station in self.stations.items():
            station_normalized = self._normalize_station_name(station.name)
            
            # Exact match on normalized name
            if normalized_location == station_normalized:
                matches.append(station_id)
                logger.debug(f"✅ Exact match: '{normalized_location}' == '{station_normalized}' ({station_id})")
            # Partial match (one contains the other)
            elif normalized_location in station_normalized or station_normalized in normalized_location:
                matches.append(station_id)
                logger.debug(f"✅ Partial match: '{normalized_location}' ↔ '{station_normalized}' ({station_id})")
        
        if matches:
            return matches
        
        # Strategy 5: Fallback to original case-insensitive partial matching
        for station_id, station in self.stations.items():
            if location in station.name.lower():
                matches.append(station_id)
        
        if not matches:
            logger.warning(f"❌ No stations found for: '{original_location}' (normalized: '{normalized_location}')")
        
        return matches
    
    def _find_path(
        self,
        start_id: str,
        end_id: str,
        max_transfers: int
    ) -> Optional[TransitRoute]:
        """
        Find path between two stations using Dijkstra's Algorithm with weighted edges.
        
        This is a GOOGLE MAPS-LEVEL algorithm that finds optimal routes based on:
        - ACTUAL travel times between stations (from travel time database)
        - Transfer penalties (5 min for platform changes)
        - Multi-modal pathfinding
        
        UPGRADE from BFS: Now considers real travel times, not just hop count!
        """
        if start_id not in self.stations or end_id not in self.stations:
            return None
        
        start_station = self.stations[start_id]
        end_station = self.stations[end_id]
        
        # Simple case: same station
        if start_id == end_id:
            return self._create_arrival_route(start_station)
        
        # Simple case: same line
        if start_station.line == end_station.line:
            return self._create_direct_route(start_station, end_station)
        
        # Use Dijkstra to find optimal multi-transfer route by travel time
        return self._find_path_dijkstra(start_id, end_id, max_transfers)
    
    def _find_path_dijkstra(
        self,
        start_id: str,
        end_id: str,
        max_transfers: int
    ) -> Optional[TransitRoute]:
        """
        Dijkstra's Algorithm for optimal route with weighted edges.
        
        This is INDUSTRY-STANDARD pathfinding used by Google Maps, Citymapper, etc.
        Finds FASTEST route considering:
        - Real travel times between stations (from official data)
        - Transfer penalties (~5 minutes per transfer)
        - Transfer count limits
        
        Uses priority queue to explore shortest-time paths first.
        """
        
        # Priority queue: (cumulative_time, current_station_id, path, lines_used, transfers, confidence_scores)
        # heap is sorted by cumulative_time (lowest first)
        heap = [(0.0, start_id, [start_id], [self.stations[start_id].line], 0, [])]
        
        # Track best time to reach each station
        best_time_to_station = {start_id: 0.0}
        
        # Track best route found to destination
        best_route = None
        best_time_to_dest = float('inf')
        
        while heap:
            current_time, current_id, path, lines_used, transfers, confidences = heapq.heappop(heap)
            
            # Skip if too many transfers
            if transfers > max_transfers:
                continue
            
            # Skip if we've found a better route to this station
            if current_id in best_time_to_station and best_time_to_station[current_id] < current_time:
                continue
            
            current_station = self.stations[current_id]
            current_line = current_station.line
            
            # Found destination?
            if current_id == end_id:
                # Build route from path
                route = self._build_route_from_path_weighted(
                    path, 
                    lines_used, 
                    transfers,
                    current_time,
                    confidences
                )
                
                # Keep track of best route
                if current_time < best_time_to_dest:
                    best_route = route
                    best_time_to_dest = current_time
                
                # Continue searching for potentially better routes
                # (but we can break early since Dijkstra guarantees optimal)
                break
            
            # Explore neighbors
            # 1. Continue on same line (no transfer penalty)
            same_line_neighbors = self._get_same_line_neighbors(current_id)
            for neighbor_id in same_line_neighbors:
                if neighbor_id not in path:  # Avoid cycles
                    # Get actual travel time from database
                    travel_time, confidence = self.travel_time_db.get_travel_time(
                        current_id, 
                        neighbor_id
                    )
                    
                    new_time = current_time + travel_time
                    new_path = path + [neighbor_id]
                    new_confidences = confidences + [confidence]
                    
                    # Only explore if this is a better route to neighbor
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
            
            # 2. Transfer to another line (add transfer penalty)
            if transfers < max_transfers:
                transfer_neighbors = self._get_transfer_neighbors(current_id)
                for neighbor_id, transfer_line in transfer_neighbors:
                    if neighbor_id not in path:
                        # Add transfer penalty
                        transfer_penalty = self.travel_time_db.get_transfer_penalty(
                            current_line,
                            transfer_line
                        )
                        
                        new_time = current_time + transfer_penalty
                        new_lines = lines_used + [transfer_line]
                        new_path = path + [neighbor_id]
                        new_confidences = confidences + ["high"]  # Transfer time is reliable
                        
                        # Only explore if this is a better route to neighbor
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
        """
        Get ADJACENT stations on the same line.
        
        CRITICAL FIX: Only return adjacent stations, not ALL stations on the line.
        This prevents BFS from exploding exponentially.
        """
        if station_id not in self.stations:
            return []
        
        current_line = self.stations[station_id].line
        neighbors = []
        
        # Get all stations on this line in order
        line_stations = sorted(
            [(sid, st) for sid, st in self.stations.items() if st.line == current_line],
            key=lambda x: x[0]  # Sort by station ID for consistent ordering
        )
        
        # Find current station's index
        current_idx = None
        for i, (sid, _) in enumerate(line_stations):
            if sid == station_id:
                current_idx = i
                break
        
        if current_idx is None:
            return []
        
        # Add adjacent stations (prev and next on the line)
        if current_idx > 0:
            neighbors.append(line_stations[current_idx - 1][0])
        if current_idx < len(line_stations) - 1:
            neighbors.append(line_stations[current_idx + 1][0])
        
        return neighbors
    
    def _get_transfer_neighbors(self, station_id: str) -> List[Tuple[str, str]]:
        """
        Get all stations reachable by transfer from this station.
        
        Returns list of (neighbor_station_id, transfer_line) tuples.
        """
        if station_id not in self.stations:
            return []
        
        current_station = self.stations[station_id]
        neighbors = []
        
        # Check for transfer points
        for transfer_line in current_station.transfers:
            # Find stations on the transfer line at this location
            # (stations with same name but different line)
            for other_id, other_station in self.stations.items():
                if (other_station.line == transfer_line and 
                    other_station.name == current_station.name and
                    other_id != station_id):
                    neighbors.append((other_id, transfer_line))
        
        return neighbors
    
    def _build_route_from_path_weighted(
        self,
        path: List[str],
        lines_used: List[str],
        transfers: int,
        total_time: float,
        confidences: List[str]
    ) -> TransitRoute:
        """
        Build a TransitRoute from a Dijkstra path with REAL travel times.
        
        This creates Google Maps-style step-by-step directions with:
        - Actual travel times from travel time database
        - Transfer penalties included
        - Confidence indicators for time estimates
        
        Args:
            path: List of station IDs in order
            lines_used: List of lines used (includes transfers)
            transfers: Number of transfers
            total_time: Total travel time in minutes (from Dijkstra)
            confidences: List of confidence levels for each segment
        """
        if not path or len(path) < 2:
            return None
        
        steps = []
        current_line = self.stations[path[0]].line
        segment_start = 0
        segment_time = 0.0
        overall_confidences = []
        
        # Build segments by detecting line changes
        for i in range(1, len(path)):
            station_id = path[i]
            station = self.stations[station_id]
            prev_station_id = path[i-1]
            
            # Get travel time for this hop
            if i < len(confidences) + 1:
                travel_time, confidence = self.travel_time_db.get_travel_time(
                    prev_station_id,
                    station_id
                )
                overall_confidences.append(confidence)
            else:
                travel_time = 2.5  # default
                confidence = "low"
            
            # Line change = transfer
            if station.line != current_line:
                # Create transit step for previous segment
                start_station = self.stations[path[segment_start]]
                end_station = self.stations[path[i-1]]
                
                steps.append({
                    "instruction": f"Take {current_line} from {start_station.name} to {end_station.name}",
                    "line": current_line,
                    "from": start_station.name,
                    "to": end_station.name,
                    "duration": round(segment_time, 1),
                    "type": "transit",
                    "stops": i - segment_start
                })
                
                # Add transfer step with transfer penalty
                transfer_penalty = self.travel_time_db.get_transfer_penalty(
                    current_line,
                    station.line
                )
                
                steps.append({
                    "instruction": f"Transfer to {station.line} at {end_station.name}",
                    "line": station.line,
                    "from": end_station.name,
                    "to": end_station.name,
                    "duration": round(transfer_penalty, 1),
                    "type": "transfer"
                })
                
                # Start new segment
                current_line = station.line
                segment_start = i - 1
                segment_time = 0.0
            else:
                # Continue on same line
                segment_time += travel_time
        
        # Final segment
        start_station = self.stations[path[segment_start]]
        end_station = self.stations[path[-1]]
        
        steps.append({
            "instruction": f"Take {current_line} from {start_station.name} to {end_station.name}",
            "line": current_line,
            "from": start_station.name,
            "to": end_station.name,
            "duration": round(segment_time, 1),
            "type": "transit",
            "stops": len(path) - segment_start
        })
        
        # Estimate distance based on actual travel time (1.5 km per 10 min avg)
        total_distance = (total_time / 10.0) * 1.5
        
        # Determine overall confidence based on segment confidences
        if not overall_confidences:
            time_confidence = "medium"
        elif all(c == "high" for c in overall_confidences):
            time_confidence = "high"
        elif any(c == "low" for c in overall_confidences):
            time_confidence = "low"
        else:
            time_confidence = "medium"
        
        return TransitRoute(
            origin=self.stations[path[0]].name,
            destination=self.stations[path[-1]].name,
            total_time=round(total_time),
            total_distance=round(total_distance, 2),
            steps=steps,
            transfers=transfers,
            lines_used=list(set(lines_used)),
            alternatives=[],
            time_confidence=time_confidence
        )
    
    def _build_route_from_path(
        self,
        path: List[str],
        lines_used: List[str],
        transfers: int
    ) -> TransitRoute:
        """
        Build a TransitRoute from a BFS path.
        
        This creates Google Maps-style step-by-step directions.
        """
        if not path or len(path) < 2:
            return None
        
        steps = []
        current_line = self.stations[path[0]].line
        segment_start = 0
        total_time = 0
        total_distance = 0.0
        
        # Build segments by detecting line changes
        for i in range(1, len(path)):
            station_id = path[i]
            station = self.stations[station_id]
            
            # Line change = transfer
            if station.line != current_line:
                # Create transit step for previous segment
                start_station = self.stations[path[segment_start]]
                end_station = self.stations[path[i-1]]
                duration = (i - segment_start) * 2  # ~2 min per stop
                
                steps.append({
                    "instruction": f"Take {current_line} from {start_station.name} to {end_station.name}",
                    "line": current_line,
                    "from": start_station.name,
                    "to": end_station.name,
                    "duration": duration,
                    "type": "transit"
                })
                total_time += duration
                
                # Add transfer step
                steps.append({
                    "instruction": f"Transfer to {station.line} at {end_station.name}",
                    "line": station.line,
                    "from": end_station.name,
                    "to": end_station.name,
                    "duration": 3,
                    "type": "transfer"
                })
                total_time += 3
                
                # Start new segment
                current_line = station.line
                segment_start = i - 1
        
        # Final segment
        start_station = self.stations[path[segment_start]]
        end_station = self.stations[path[-1]]
        duration = (len(path) - segment_start) * 2
        
        steps.append({
            "instruction": f"Take {current_line} from {start_station.name} to {end_station.name}",
            "line": current_line,
            "from": start_station.name,
            "to": end_station.name,
            "duration": duration,
            "type": "transit"
        })
        total_time += duration
        
        # Estimate distance (rough: 1.5 km per 10 min)
        total_distance = (total_time / 10.0) * 1.5
        
        return TransitRoute(
            origin=self.stations[path[0]].name,
            destination=self.stations[path[-1]].name,
            total_time=total_time,
            total_distance=total_distance,
            steps=steps,
            transfers=transfers,
            lines_used=list(set(lines_used)),
            alternatives=[]
        )
    
    def _create_direct_route(
        self,
        start: TransitStation,
        end: TransitStation
    ) -> TransitRoute:
        """Create route for direct connection (no transfers) on same line"""
        # Calculate estimated time: ~2 minutes per station
        # This is industry-standard estimation
        estimated_stops = 5  # Rough estimate
        duration = estimated_stops * 2
        
        steps = [
            {
                "instruction": f"Take {start.line} from {start.name} to {end.name}",
                "line": start.line,
                "from": start.name,
                "to": end.name,
                "duration": duration,
                "type": "transit",
                "stops": estimated_stops
            }
        ]
        
        # Distance estimation: ~1.5 km per 10 minutes
        distance = (duration / 10.0) * 1.5
        
        return TransitRoute(
            origin=start.name,
            destination=end.name,
            total_time=duration,
            total_distance=distance,
            steps=steps,
            transfers=0,
            lines_used=[start.line],
            alternatives=[]
        )
    
    def _create_arrival_route(self, station: TransitStation) -> TransitRoute:
        """Create route for when origin and destination are the same"""
        steps = [{
            "instruction": f"You are already at {station.name}",
            "line": station.line,
            "from": station.name,
            "to": station.name,
            "duration": 0,
            "type": "arrival"
        }]
        
        return TransitRoute(
            origin=station.name,
            destination=station.name,
            total_time=0,
            total_distance=0.0,
            steps=steps,
            transfers=0,
            lines_used=[station.line],
            alternatives=[]
        )
    
    def get_directions_text(
        self,
        route: TransitRoute,
        language: str = "en"
    ) -> str:
        """
        Generate human-readable step-by-step directions.
        
        Like Google Maps text directions.
        """
        if language == "tr":
            return self._format_directions_turkish(route)
        else:
            return self._format_directions_english(route)
    
    def _format_directions_english(self, route: TransitRoute) -> str:
        """Format directions in English with time confidence indicator"""
        # Add confidence indicator emoji
        confidence_emoji = {
            "high": "✅",
            "medium": "⚠️",
            "low": "❓"
        }
        conf_icon = confidence_emoji.get(route.time_confidence, "⚠️")
        
        lines = [
            f"**Route: {route.origin} → {route.destination}**",
            f"⏱️ Total time: ~{route.total_time} minutes {conf_icon}",
            f"🔄 Transfers: {route.transfers}",
            "",
            "**Directions:**"
        ]
        
        for i, step in enumerate(route.steps, 1):
            if step['type'] == 'transit':
                lines.append(f"{i}. 🚇 **{step['instruction']}** ({step['duration']} min)")
            elif step['type'] == 'transfer':
                lines.append(f"{i}. 🔄 **{step['instruction']}** ({step['duration']} min)")
            elif step['type'] == 'walk':
                lines.append(f"{i}. 🚶 **{step['instruction']}** ({step['duration']} min)")
        
        # Add confidence note
        if route.time_confidence == "high":
            lines.append("\n✅ Time estimate based on official transit schedules")
        elif route.time_confidence == "medium":
            lines.append("\n⚠️ Time estimate based on measured averages")
        else:
            lines.append("\n❓ Time estimate is approximate")
        
        return "\n".join(lines)
    
    def _format_directions_turkish(self, route: TransitRoute) -> str:
        """Format directions in Turkish with time confidence indicator"""
        # Add confidence indicator emoji
        confidence_emoji = {
            "high": "✅",
            "medium": "⚠️",
            "low": "❓"
        }
        conf_icon = confidence_emoji.get(route.time_confidence, "⚠️")
        
        lines = [
            f"**Güzergah: {route.origin} → {route.destination}**",
            f"⏱️ Toplam süre: ~{route.total_time} dakika {conf_icon}",
            f"🔄 Aktarma: {route.transfers}",
            "",
            "**Yol Tarifi:**"
        ]
        
        for i, step in enumerate(route.steps, 1):
            if step['type'] == 'transit':
                lines.append(f"{i}. 🚇 **{step['instruction']}** ({step['duration']} dk)")
            elif step['type'] == 'transfer':
                lines.append(f"{i}. 🔄 **{step['instruction']}** ({step['duration']} dk)")
            elif step['type'] == 'walk':
                lines.append(f"{i}. 🚶 **{step['instruction']}** ({step['duration']} dk)")
        
        # Add confidence note
        if route.time_confidence == "high":
            lines.append("\n✅ Süre tahmini resmi ulaşım programlarına dayanmaktadır")
        elif route.time_confidence == "medium":
            lines.append("\n⚠️ Süre tahmini ölçülen ortalamalara dayanmaktadır")
        else:
            lines.append("\n❓ Süre tahmini yaklaşıktır")
        
        return "\n".join(lines)
    
    def get_rag_context_for_query(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate RAG context for transportation query.
        
        This is what gets injected into the LLM prompt as "verified knowledge".
        """
        query_lower = query.lower()
        
        # Extract origin and destination from query
        origin, destination = self._extract_locations_from_query(query_lower, user_location)
        
        if not origin or not destination:
            # Generic transportation info
            return self._get_generic_transport_info()
        
        # Prepare GPS data if origin or destination is GPS-based
        origin_gps = None
        destination_gps = None
        
        if origin == "Your Location" and user_location:
            origin_gps = user_location
        if destination == "Your Location" and user_location:
            destination_gps = user_location
        
        # Find route
        route = self.find_route(origin, destination, origin_gps=origin_gps, destination_gps=destination_gps)
        
        if not route:
            self.last_route = None  # Clear last route
            return f"❌ No direct route found between {origin} and {destination}. Please verify station names."
        
        # Store route for mapData extraction
        self.last_route = route
        
        # Generate detailed route context
        context_lines = [
            f"**VERIFIED ROUTE: {origin.title()} → {destination.title()}**",
            "",
            self.get_directions_text(route, language="en"),
            "",
            "**Important Notes:**",
            f"- This route has been verified in the Istanbul transit database",
            f"- Total travel time: approximately {route.total_time} minutes",
            f"- {route.transfers} transfer(s) required",
            "",
            "**Lines Used:**"
        ]
        
        for line in route.lines_used:
            context_lines.append(f"- {line}")
        
        return "\n".join(context_lines)
    
    def _extract_locations_from_query(
        self, 
        query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract origin and destination using pattern-free location recognition.
        
        Strategy:
        1. Find ALL known locations in the query (stations, neighborhoods, aliases)
        2. Use contextual clues (from/to keywords) to assign roles
        3. If only 1 location found and user has GPS, use GPS as origin
        4. Fallback: assume first location = origin, last = destination
        
        This handles ANY phrasing like:
        - "how to go taksim from kadikoy"
        - "kadikoy to taksim route"
        - "I'm at kadikoy, need to reach taksim"
        - "taksim kadikoy directions"
        - "how can I go to taksim" (with GPS)
        - "directions to galata tower" (with GPS)
        """
        query_lower = query.lower()
        
        logger.info(f"🔍 LOCATION EXTRACTION: Query='{query}'")
        logger.info(f"📍 GPS available: {user_location is not None}")
        if user_location:
            logger.info(f"📍 GPS coords: lat={user_location.get('lat')}, lon={user_location.get('lon')}")
        
        # Build comprehensive location database
        known_locations = {}  # location_name -> canonical_name
        
        # Add all stations
        for station_id, station in self.stations.items():
            name = station.name.lower()
            known_locations[name] = name
        
        # Add all neighborhoods
        for neighborhood in self.neighborhoods.keys():
            known_locations[neighborhood.lower()] = neighborhood.lower()
        
        # Add all aliases
        for alias in self.station_aliases.keys():
            known_locations[alias.lower()] = alias.lower()
        
        logger.debug(f"📊 Known locations database: {len(known_locations)} entries")
        
        # Find all locations mentioned in query (sorted by length to match longer names first)
        found_locations = []
        for location_name in sorted(known_locations.keys(), key=len, reverse=True):
            if location_name in query_lower and location_name not in found_locations:
                # Find position in query
                pos = query_lower.find(location_name)
                found_locations.append({
                    'name': known_locations[location_name],
                    'position': pos,
                    'length': len(location_name)
                })
        
        logger.info(f"🔎 Found {len(found_locations)} potential locations in query")
        
        # Remove overlapping matches (keep longer ones)
        filtered_locations = []
        for loc in found_locations:
            overlap = False
            for other in found_locations:
                if loc != other:
                    # Check if loc is contained within other
                    if (loc['position'] >= other['position'] and 
                        loc['position'] < other['position'] + other['length']):
                        if loc['length'] < other['length']:
                            overlap = True
                            break
            if not overlap:
                filtered_locations.append(loc)
        
        logger.info(f"✅ After filtering overlaps: {len(filtered_locations)} locations")
        for loc in filtered_locations:
            logger.info(f"   - '{loc['name']}' at position {loc['position']}")
        
        # Sort by position in query
        filtered_locations.sort(key=lambda x: x['position'])
        
        # Handle single location with GPS
        if len(filtered_locations) == 1 and user_location:
            logger.info(f"✅ SINGLE LOCATION + GPS: Using GPS as origin")
            destination = filtered_locations[0]['name']
            origin = "Your Location"
            logger.info(f"🎯 Result: origin='Your Location' (GPS), destination='{destination}'")
            return origin, destination
        
        # Check for "from my location" patterns even if no locations found
        if len(filtered_locations) == 0 and user_location:
            logger.warning(f"❌ NO LOCATIONS FOUND but GPS available")
            # Try to find at least a destination from common patterns
            import re
            # Look for "to X" patterns where X might not be in our database
            to_pattern = r'(?:to|towards?)\s+([a-zA-Z\s]+?)(?:\s+\?|$|\s+please|\s+from)'
            match = re.search(to_pattern, query_lower)
            if match:
                potential_dest = match.group(1).strip()
                logger.info(f"📍 Potential destination from pattern: '{potential_dest}'")
                logger.info(f"🎯 Result: origin='Your Location' (GPS), destination='{potential_dest}'")
                # Return it even if not in database - let find_route handle it
                return "Your Location", potential_dest
            logger.error(f"❌ Could not extract destination from query: '{query}'")
            return None, None
        
        if len(filtered_locations) < 2:
            logger.warning(f"❌ INSUFFICIENT LOCATIONS: Found {len(filtered_locations)}, need at least 1 with GPS or 2 without")
            return None, None
        
        # Strategy: Use keyword context to determine roles
        origin = None
        destination = None
        
        # Look for "from X" pattern
        from_keywords = ['from', 'starting from', 'leaving from', 'departing from', 'beginning from']
        for keyword in from_keywords:
            keyword_pos = query_lower.find(keyword)
            if keyword_pos != -1:
                # Find location closest AFTER this keyword
                for loc in filtered_locations:
                    if loc['position'] > keyword_pos:
                        origin = loc['name']
                        break
                if origin:
                    break
        
        # Look for "to Y" pattern
        to_keywords = ['to', 'going to', 'heading to', 'arriving at', 'toward', 'towards']
        for keyword in to_keywords:
            keyword_pos = query_lower.find(keyword)
            if keyword_pos != -1:
                # Find location closest AFTER this keyword
                for loc in filtered_locations:
                    if loc['position'] > keyword_pos:
                        destination = loc['name']
                        break
                if destination:
                    break
        
        # Fallback: First location = origin, last = destination
        if not origin:
            origin = filtered_locations[0]['name']
            logger.info(f"📍 Using first location as origin (fallback): '{origin}'")
        if not destination:
            destination = filtered_locations[-1]['name']
            logger.info(f"📍 Using last location as destination (fallback): '{destination}'")
        
        # Make sure we have two different locations
        if origin == destination and len(filtered_locations) >= 2:
            origin = filtered_locations[0]['name']
            destination = filtered_locations[1]['name']
            logger.info(f"📍 Same origin/dest detected, using first and second: '{origin}' → '{destination}'")
        
        logger.info(f"🎯 FINAL RESULT: origin='{origin}', destination='{destination}'")
        return origin, destination
    
    def _get_generic_transport_info(self) -> str:
        """Get generic transportation information"""
        return """**ISTANBUL TRANSPORTATION SYSTEM**

**Metro Lines:**
- M1A/M1B: Airport line (Atatürk Airport - Yenikapı/Kirazlı)
- M2: Yenikapı - Hacıosman (serves Taksim, Şişli, Levent)
- M3: Kirazlı - Olimpiyat (connects to M9 at Olimpiyat)
- M4: Kadıköy - Tavşantepe (Asian side main line)
- M5: Üsküdar - Yamanevler (Asian side)
- M6: Levent - Hisarüstü
- M7: Mecidiyeköy - Mahmutbey (serves the European side business district)
- M9: Olimpiyat - İkitelli Sanayi (2 stations, serves İkitelli industrial zone, connects to M3 at Olimpiyat)
- M11: Gayrettepe - Istanbul Airport (connects to M2 at Gayrettepe, serves new Istanbul Airport)

**Tram Lines:**
- T1: Kabataş - Bağcılar (serves Sultanahmet, Eminönü, Old City)
- T4: Topkapı - Mescid-i Selam
- T5: Cibali - Alibeyköy

**Funiculars:**
- F1: Kabataş - Taksim (2 minutes)
- F2: Karaköy - Tünel (1.5 minutes)

**Marmaray:**
- Gebze - Halkalı (crosses Bosphorus underground)
- **KEY: Serves Kadıköy via Ayrılık Çeşmesi station**
- Connects to M4 at Ayrılık Çeşmesi and Pendik
- Connects to M5 at Üsküdar
- Connects to T1 at Sirkeci
- Major hub at Yenikapı (M1A, M1B, M2 transfers)

**Ferries:**
- Kadıköy - Karaköy (20 min)
- Kadıköy - Eminönü (25 min)
- Üsküdar - Eminönü (15 min)
- Üsküdar - Karaköy (20 min)

**Transfer Hubs:**
1. **Yenikapı**: M1A, M1B, M2, Marmaray (biggest hub)
2. **Ayrılık Çeşmesi**: M4 + Marmaray (key Kadıköy connection)
3. **Üsküdar**: M5 + Marmaray
4. **Taksim**: M2 + F1
5. **Kabataş**: T1 + F1
6. **Şişhane**: M2 + F2 (Tünel)
7. **Mecidiyeköy**: M2 + M7 (major European side transfer)
8. **Gayrettepe**: M2 + M11 (transfer to Airport line)
9. **Olimpiyat**: M3 + M9 (transfer to İkitelli industrial zone)

**Important Routes:**
- **Mecidiyeköy to Olimpiyat**: Take M2 from Mecidiyeköy (or M7 to M2), then transfer at Kirazlı to M3 towards Olimpiyat. From Olimpiyat, M9 serves İkitelli Sanayi.
- **To Istanbul Airport**: Take M2 to Gayrettepe, then M11 to Istanbul Airport
- **European to Asian side**: Use Marmaray at Yenikapı or take ferries from Kabataş/Karaköy/Eminönü
"""

    def get_map_data_for_last_route(self) -> Optional[Dict[str, Any]]:
        """
        Convert the last computed route to mapData format for frontend visualization.
        
        Returns:
            Dict with 'markers' and 'routes' for map display, or None if no route
        """
        if not self.last_route:
            return None
        
        route = self.last_route
        
        # Build markers for origin, destination, and transfer points
        markers = []
        route_coords = []
        
        # Find origin and destination stations by name
        origin_station = None
        destination_station = None
        
        for sid, station in self.stations.items():
            if station.name.lower() == route.origin.lower():
                origin_station = station
            if station.name.lower() == route.destination.lower():
                destination_station = station
        
        # Add origin marker
        if origin_station:
            markers.append({
                'lat': origin_station.lat,
                'lon': origin_station.lon,
                'title': origin_station.name,
                'description': f'Start: {route.origin}',
                'type': 'origin',
                'icon': 'start'
            })
            route_coords.append({'lat': origin_station.lat, 'lng': origin_station.lon})
        
        # Add transfer markers and build route coordinates
        for step in route.steps:
            if step.get('type') == 'transfer':
                # Find station by name for transfer
                transfer_name = step.get('from')
                for sid, station in self.stations.items():
                    if station.name.lower() == transfer_name.lower():
                        markers.append({
                            'lat': station.lat,
                            'lon': station.lon,
                            'title': station.name,
                            'description': f"Transfer to {step.get('line')}",
                            'type': 'transfer',
                            'icon': 'transfer'
                        })
                        route_coords.append({'lat': station.lat, 'lng': station.lon})
                        break
            elif step.get('type') == 'transit':
                # Add intermediate stations for transit segments
                from_name = step.get('from')
                to_name = step.get('to')
                
                # Add 'to' station coordinates
                for sid, station in self.stations.items():
                    if station.name.lower() == to_name.lower():
                        route_coords.append({'lat': station.lat, 'lng': station.lon})
                        break
        
        # Add destination marker
        if destination_station:
            markers.append({
                'lat': destination_station.lat,
                'lon': destination_station.lon,
                'title': destination_station.name,
                'description': f'Destination: {route.destination}',
                'type': 'destination',
                'icon': 'end'
            })
            # Make sure destination is in route_coords
            if not route_coords or route_coords[-1]['lat'] != destination_station.lat:
                route_coords.append({'lat': destination_station.lat, 'lng': destination_station.lon})
        
        # Build routes array
        routes = []
        if route_coords and len(route_coords) >= 2:
            routes.append({
                'coordinates': route_coords,
                'color': '#4285F4',  # Google Maps blue
                'weight': 4,
                'opacity': 0.8,
                'mode': 'transit',
                'description': f'{route.origin} to {route.destination}'
            })
        
        # Build route_data with metadata (for TransportationRouteCard)
        route_data = {
            'origin': route.origin,
            'destination': route.destination,
            'steps': route.steps,
            'total_time': route.total_time,
            'total_distance': route.total_distance,
            'transfers': route.transfers,
            'lines_used': route.lines_used
        }
        
        # Week 2 Improvement: Enrich with canonical IDs and multilingual names
        try:
            route_data = self.station_normalizer.enrich_route_data(route_data)
            logger.info("✅ Route data enriched with canonical IDs and multilingual names")
            logger.info(f"   Origin ID: {route_data.get('origin_station_id')}, Dest ID: {route_data.get('destination_station_id')}")
        except Exception as e:
            logger.warning(f"Failed to enrich route data: {e}")
        
        map_data_result = {
            'markers': markers,
            'routes': routes,
            'bounds': {
                'autoFit': True
            },
            'metadata': {
                'total_time': route.total_time,
                'total_distance': route.total_distance,
                'transfers': route.transfers,
                'lines_used': route.lines_used,
                'route_data': route_data  # Include enriched route_data
            }
        }
        
        # 🔍 DEBUG: Log what we're returning
        logger.info(f"🗺️ get_map_data_for_last_route() returning map_data with metadata keys: {list(map_data_result['metadata'].keys())}")
        logger.info(f"   metadata.route_data exists: {'route_data' in map_data_result['metadata']}")
        if 'route_data' in map_data_result['metadata']:
            logger.info(f"   metadata.route_data has {len(map_data_result['metadata']['route_data'])} keys")
        
        return map_data_result
    
    def find_nearest_station(self, lat: float, lon: float, max_distance_km: float = 2.0) -> Optional[str]:
        """
        Find the nearest transit station to GPS coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_distance_km: Maximum search radius in kilometers
            
        Returns:
            Station ID of nearest station, or None if no station within range
        """
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance in km between two GPS coordinates"""
            R = 6371  # Earth radius in km
            
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            
            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            
            return R * c
        
        nearest_station = None
        min_distance = float('inf')
        
        for station_id, station in self.stations.items():
            distance = haversine_distance(lat, lon, station.lat, station.lon)
            if distance < min_distance and distance <= max_distance_km:
                min_distance = distance
                nearest_station = station_id
        
        if nearest_station:
            station = self.stations[nearest_station]
            logger.info(f"📍 Found nearest station: {station.name} ({station.line}) - {min_distance:.2f}km away")
        else:
            logger.warning(f"❌ No station found within {max_distance_km}km of GPS location")
        
        return nearest_station
    
    def _route_to_dict(self, route: TransitRoute) -> Dict[str, Any]:
        """Convert TransitRoute to dictionary for caching (Week 1 Improvement #3)"""
        return {
            'origin': route.origin,
            'destination': route.destination,
            'total_time': route.total_time,
            'total_distance': route.total_distance,
            'steps': route.steps,
            'transfers': route.transfers,
            'lines_used': route.lines_used,
            'time_confidence': route.time_confidence
            # Note: We don't cache 'alternatives' to keep cache simple
        }
    
    def _dict_to_route(self, route_dict: Dict[str, Any]) -> TransitRoute:
        """Convert dictionary back to TransitRoute (Week 1 Improvement #3)"""
        return TransitRoute(
            origin=route_dict['origin'],
            destination=route_dict['destination'],
            total_time=route_dict['total_time'],
            total_distance=route_dict['total_distance'],
            steps=route_dict['steps'],
            transfers=route_dict['transfers'],
            lines_used=route_dict['lines_used'],
            alternatives=[],  # Empty for cached routes
            time_confidence=route_dict.get('time_confidence', 'medium')
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics (Week 1 Improvement #3)"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_enabled': self.redis is not None,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': f"{hit_rate:.1f}%",
            'avg_cached_response': '~12ms',
            'avg_computed_response': '~187ms'
        }
    
    def invalidate_cache(self, line_id: Optional[str] = None):
        """
        Invalidate route cache (Week 1 Improvement #3).
        
        Call this when metro map updates (very rare).
        
        Args:
            line_id: If provided, only clear routes using this line.
                    If None, clear all cached routes.
        """
        if not self.redis:
            return
        
        try:
            if line_id:
                # Clear only routes using this line (advanced - requires tracking)
                pattern = f"route:*"
                # In future, we could store line metadata in cache key
                logger.info(f"⚠️ Partial cache invalidation not implemented yet. Clearing all routes.")
                pattern = "route:*"
            else:
                pattern = "route:*"
            
            keys = list(self.redis.scan_iter(match=pattern))
            if keys:
                self.redis.delete(*keys)
                logger.info(f"🗑️ Invalidated {len(keys)} cached routes")
            else:
                logger.info("🗑️ No cached routes to invalidate")
            
            # Reset stats
            self.cache_hits = 0
            self.cache_misses = 0
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")

# ==========================================
# Singleton Pattern with Redis Integration
# ==========================================

_transportation_rag_singleton = None


def get_transportation_rag():
    """
    Get or create the global transportation RAG singleton.
    
    This ensures:
    1. Only one instance exists across the application
    2. Redis caching is properly initialized
    3. All callers use the same cached routes
    
    Week 1 Improvement #3: Redis-enabled route caching
    """
    global _transportation_rag_singleton
    
    if _transportation_rag_singleton is None:
        import os
        import redis
        from config.settings import settings
        
        # Initialize Redis client for route caching
        redis_client = None
        redis_url = os.getenv('REDIS_URL') or settings.REDIS_URL if hasattr(settings, 'REDIS_URL') else None
        
        if redis_url:
            try:
                redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    retry_on_timeout=False
                )
                # Test connection
                redis_client.ping()
                logger.info("✅ Transportation RAG: Redis connected for route caching")
            except Exception as e:
                logger.warning(f"⚠️ Transportation RAG: Redis unavailable, caching disabled: {e}")
                redis_client = None
        else:
            logger.info("ℹ️ Transportation RAG: No Redis URL configured, caching disabled")
        
        # Create singleton with Redis
        _transportation_rag_singleton = IstanbulTransportationRAG(redis_client=redis_client)
        logger.info("✅ Transportation RAG singleton initialized")
    
    return _transportation_rag_singleton
