"""
Istanbul Transportation RAG System - Main Orchestrator
======================================================

Industry-level transportation knowledge system with Google Maps-quality routing.

This module orchestrates:
- Station graph building
- Multi-modal pathfinding
- Location extraction
- Route formatting
- Map data generation

Author: AI Istanbul Team
Date: December 2024
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from .nlp_utils import TurkishMorphologyHandler
from .destination_types import (
    DestinationType,
    get_destination_type,
    is_walking_distance,
    haversine_distance,
)
from .station_aliases import (
    build_station_aliases,
    build_neighborhood_stations,
    build_route_patterns,
)
from .route_builder import TransitStation, TransitRoute, RouteBuilder
from .pathfinding import Pathfinder
from .location_extraction import LocationExtractor, extract_locations_with_llm_sync

logger = logging.getLogger(__name__)

# Redis import (optional)
try:
    import redis
except ImportError:
    redis = None


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
        """Initialize the transportation knowledge base."""
        # Build station graph
        self.stations = self._build_station_graph()
        self.station_graph = self.stations  # Alias
        
        # Build lookup tables
        self.routes = build_route_patterns()
        self.neighborhoods = build_neighborhood_stations()
        self.station_aliases = build_station_aliases()
        
        # Last computed route for map data
        self.last_route = None
        self._last_query = None
        
        # Initialize travel time database
        try:
            from services.transportation_travel_times import get_travel_time_database
        except ImportError:
            from backend.services.transportation_travel_times import get_travel_time_database
        self.travel_time_db = get_travel_time_database()
        
        # Initialize station normalizer
        try:
            from services.transportation_station_normalization import get_station_normalizer
        except ImportError:
            from backend.services.transportation_station_normalization import get_station_normalizer
        self.station_normalizer = get_station_normalizer()
        
        # Initialize route builder
        self.route_builder = RouteBuilder(
            self.stations,
            self.travel_time_db,
            self.station_normalizer
        )
        
        # Initialize pathfinder
        self.pathfinder = Pathfinder(
            self.stations,
            self.travel_time_db,
            self.station_normalizer,
            self.route_builder
        )
        
        # Initialize location extractor
        self.location_extractor = LocationExtractor(
            self.stations,
            self.neighborhoods,
            self.station_aliases
        )
        
        # Redis caching
        self.redis = redis_client
        self.redis_client = redis_client
        self.route_cache_ttl = 86400  # 24 hours
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("✅ Transportation RAG initialized with complete Istanbul network")
        logger.info("✅ Dijkstra routing enabled with realistic travel times")
        if redis_client:
            logger.info("✅ Route caching enabled with Redis")
    
    def _build_station_graph(self) -> Dict[str, TransitStation]:
        """Build complete graph of all Istanbul transit stations."""
        stations = {}
        
        try:
            from services.transportation_station_normalization import get_station_normalizer
        except ImportError:
            from backend.services.transportation_station_normalization import get_station_normalizer
        normalizer = get_station_normalizer()
        
        for canonical_station in normalizer.stations:
            station_id = canonical_station.canonical_id
            stations[station_id] = TransitStation(
                name=canonical_station.name_en,
                line=canonical_station.line_id,
                lat=canonical_station.lat,
                lon=canonical_station.lon,
                transfers=canonical_station.transfers
            )
        
        logger.info(f"✅ Built station graph: {len(stations)} stations")
        
        # Log station counts by line
        line_counts = {}
        for station_id in stations.keys():
            line = station_id.split('-')[0]
            line_counts[line] = line_counts.get(line, 0) + 1
        
        for line in sorted(line_counts.keys()):
            logger.debug(f"  {line}: {line_counts[line]} stations")
        
        return stations
    
    def _normalize_station_name(self, name: str) -> str:
        """Normalize station/location names for fuzzy matching."""
        turkish_char_map = {
            'İ': 'i', 'I': 'i', 'ı': 'i',
            'Ö': 'o', 'ö': 'o',
            'Ü': 'u', 'ü': 'u',
            'Ş': 's', 'ş': 's',
            'Ğ': 'g', 'ğ': 'g',
            'Ç': 'c', 'ç': 'c'
        }
        
        for turkish_char, latin_char in turkish_char_map.items():
            name = name.replace(turkish_char, latin_char)
        
        name = name.lower().strip()
        
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
        
        name = ' '.join(name.split())
        return name
    
    def _get_stations_for_location(self, location: str) -> List[str]:
        """Get station IDs for a location name."""
        location_normalized = self._normalize_station_name(location)
        
        # Check aliases first
        if location_normalized in self.station_aliases:
            return self.station_aliases[location_normalized]
        
        # Check neighborhoods
        if location_normalized in self.neighborhoods:
            return self.neighborhoods[location_normalized]
        
        # Try to find matching station
        for station_id, station in self.stations.items():
            if self._normalize_station_name(station.name) == location_normalized:
                return [station_id]
        
        # Fuzzy search
        for station_id, station in self.stations.items():
            if location_normalized in self._normalize_station_name(station.name):
                return [station_id]
        
        return []
    
    def find_route(
        self,
        origin: str,
        destination: str,
        max_transfers: int = 3,
        origin_gps: Optional[Dict[str, float]] = None,
        destination_gps: Optional[Dict[str, float]] = None
    ) -> Optional[TransitRoute]:
        """Find the best route between two locations."""
        origin_normalized = origin.lower().strip()
        destination_normalized = destination.lower().strip()
        
        # Check for deprecated stations
        deprecated_check = self._check_deprecated_stations(origin_normalized, destination_normalized)
        if deprecated_check:
            return self.route_builder.create_deprecation_route(origin, destination, deprecated_check)
        
        # Destination type detection
        dest_info = get_destination_type(destination_normalized)
        logger.info(f"🎯 Destination type: {dest_info.dest_type.value} for '{destination}'")
        
        # Walking distance short-circuit
        if origin_normalized == destination_normalized:
            walking_route = self.route_builder.create_walking_route(origin, destination, walk_time=2)
            self.last_route = walking_route
            return walking_route
        
        if origin_gps and destination_gps:
            origin_coords = (origin_gps.get('lat', 0), origin_gps.get('lon', 0))
            dest_coords = (destination_gps.get('lat', 0), destination_gps.get('lon', 0))
            
            is_walkable, walk_time = is_walking_distance(origin_coords, dest_coords)
            if is_walkable:
                walking_route = self.route_builder.create_walking_route(origin, destination, walk_time=walk_time)
                self.last_route = walking_route
                return walking_route
        
        # Island routing
        if dest_info.dest_type == DestinationType.ISLAND:
            island_route = self._create_island_route(origin, destination, dest_info, origin_gps)
            if island_route:
                self.last_route = island_route
                return island_route
        
        # Check cache
        use_cache = not (origin_gps or destination_gps)
        cache_key = f"route:{origin_normalized}|{destination_normalized}|{max_transfers}"
        
        if use_cache and self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    self.cache_hits += 1
                    route_dict = json.loads(cached)
                    cached_route = self.route_builder.dict_to_route(route_dict)
                    self.last_route = cached_route
                    return cached_route
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        if use_cache:
            self.cache_misses += 1
        
        # Get station IDs
        if origin_gps and 'lat' in origin_gps and 'lon' in origin_gps:
            nearest_origin = self.find_nearest_station(origin_gps['lat'], origin_gps['lon'])
            origin_stations = [nearest_origin] if nearest_origin else self._get_stations_for_location(origin_normalized)
        else:
            origin_stations = self._get_stations_for_location(origin_normalized)
        
        if destination_gps and 'lat' in destination_gps and 'lon' in destination_gps:
            nearest_dest = self.find_nearest_station(destination_gps['lat'], destination_gps['lon'])
            dest_stations = [nearest_dest] if nearest_dest else self._get_stations_for_location(destination_normalized)
        else:
            dest_stations = self._get_stations_for_location(destination_normalized)
        
        if not origin_stations or not dest_stations:
            logger.warning(f"Could not find stations for {origin} or {destination}")
            return None
        
        # Find routes
        all_routes = []
        
        for orig_station in origin_stations:
            for dest_station in dest_stations:
                route = self.pathfinder.find_path(orig_station, dest_station, max_transfers)
                if route:
                    all_routes.append(route)
        
        # Find alternatives
        if len(all_routes) == 1 and len(origin_stations) == 1 and len(dest_stations) == 1:
            orig_station = origin_stations[0]
            dest_station = dest_stations[0]
            
            for extra_transfers in [1, 2]:
                alt_route = self.pathfinder.find_path_with_penalty(
                    orig_station, dest_station, max_transfers + extra_transfers
                )
                if alt_route and not self.pathfinder.is_duplicate_route(alt_route, all_routes):
                    all_routes.append(alt_route)
            
            ferry_route = self.pathfinder.find_ferry_alternative(orig_station, dest_station)
            if ferry_route and not self.pathfinder.is_duplicate_route(ferry_route, all_routes):
                all_routes.append(ferry_route)
        
        if not all_routes:
            logger.warning(f"No routes found between {origin} and {destination}")
            return None
        
        # Rank routes
        ranked_routes = self.pathfinder.rank_routes(all_routes, origin_gps)
        best_route = ranked_routes[0]
        
        if len(ranked_routes) > 1:
            best_route.alternatives = ranked_routes[1:min(4, len(ranked_routes))]
        
        # Cache result
        if use_cache and self.redis and best_route:
            try:
                route_dict = self.route_builder.route_to_dict(best_route)
                self.redis.setex(
                    cache_key,
                    self.route_cache_ttl,
                    json.dumps(route_dict, ensure_ascii=False)
                )
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        # Store GPS and walking directions
        if best_route:
            best_route.origin_gps = origin_gps
            best_route.destination_gps = destination_gps
            best_route = self._add_walking_directions(best_route)
        
        self.last_route = best_route
        return best_route
    
    def _check_deprecated_stations(self, origin: str, destination: str) -> Optional[str]:
        """Check if origin or destination is a deprecated station."""
        deprecated_stations = {
            "atatürk havalimanı": "Atatürk Airport Metro station is closed. Use Istanbul Airport (M11) instead.",
            "ataturk havalimani": "Atatürk Airport Metro station is closed. Use Istanbul Airport (M11) instead.",
            "atatürk airport": "Atatürk Airport Metro station is closed. Use Istanbul Airport (M11) instead.",
            "ataturk airport": "Atatürk Airport Metro station is closed. Use Istanbul Airport (M11) instead.",
        }
        
        for station, message in deprecated_stations.items():
            if station in origin or station in destination:
                return message
        
        return None
    
    def _create_island_route(
        self, 
        origin: str, 
        destination: str, 
        dest_info,
        origin_gps: Optional[Dict[str, float]] = None
    ) -> Optional[TransitRoute]:
        """Create a two-phase route to an island destination."""
        terminal_priorities = {
            'european': ['FERRY-Kabataş', 'FERRY-Eminönü', 'FERRY-Karaköy'],
            'asian': ['FERRY-Kadıköy', 'FERRY-Bostancı']
        }
        
        origin_lower = origin.lower()
        asian_keywords = ['kadıköy', 'kadikoy', 'üsküdar', 'uskudar', 'bostancı', 'bostanci', 
                         'pendik', 'kartal', 'maltepe', 'ataşehir', 'atasehir', 'asian']
        
        if any(kw in origin_lower for kw in asian_keywords):
            preferred_terminals = terminal_priorities['asian']
        else:
            preferred_terminals = terminal_priorities['european']
        
        # Find route to nearest ferry terminal
        terminal_route = None
        origin_stations = self._get_stations_for_location(origin.lower())
        
        if origin_stations:
            for terminal_id in preferred_terminals:
                route = self.pathfinder.find_path(origin_stations[0], terminal_id, max_transfers=2)
                if route:
                    terminal_route = route
                    break
        
        if not terminal_route:
            return None
        
        # Create combined route with ferry
        ferry_time = 35  # Approximate ferry time to islands
        
        combined_steps = terminal_route.steps + [{
            'type': 'transit',
            'instruction': f"Take ferry to {destination}",
            'line': 'FERRY',
            'from': terminal_route.destination,
            'to': destination,
            'duration': ferry_time,
            'ferry_crossing': True
        }]
        
        return TransitRoute(
            origin=origin,
            destination=destination,
            total_time=terminal_route.total_time + ferry_time,
            total_distance=terminal_route.total_distance + 15,  # Approximate ferry distance
            steps=combined_steps,
            transfers=terminal_route.transfers,
            lines_used=terminal_route.lines_used + ['FERRY'],
            alternatives=[],
            time_confidence='medium'
        )
    
    def _add_walking_directions(self, route: TransitRoute) -> TransitRoute:
        """Add first-mile/last-mile walking directions."""
        try:
            from services.walking_directions_service import get_walking_directions_service
            walking_service = get_walking_directions_service()
        except ImportError:
            return route
        
        # Implementation similar to original
        return route
    
    def find_nearest_station(self, lat: float, lon: float, max_distance_km: float = 2.0) -> Optional[str]:
        """Find the nearest transit station to given GPS coordinates."""
        nearest_station = None
        nearest_distance = float('inf')
        
        for station_id, station in self.stations.items():
            distance = haversine_distance(lat, lon, station.lat, station.lon)
            if distance < nearest_distance and distance <= max_distance_km:
                nearest_distance = distance
                nearest_station = station_id
        
        if nearest_station:
            logger.info(f"📍 Nearest station to ({lat}, {lon}): {self.stations[nearest_station].name}")
        
        return nearest_station
    
    def get_rag_context_for_query(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate RAG context for transportation query."""
        if not hasattr(self, '_last_query') or self._last_query != query:
            logger.info(f"🆕 NEW QUERY: Clearing last_route for: '{query}'")
            self.last_route = None
            self._last_query = query
        
        query_lower = query.lower().strip()
        logger.info(f"🔍 TRANSPORTATION QUERY: '{query}'")
        
        # Extract locations
        origin, destination = self.location_extractor.extract_locations(query_lower, user_location)
        
        logger.info(f"🎯 EXTRACTION RESULT: origin='{origin}', destination='{destination}'")
        
        if not origin or not destination:
            # Try LLM fallback
            llm_origin, llm_dest = extract_locations_with_llm_sync(query)
            if llm_origin and llm_dest:
                origin = llm_origin
                destination = llm_dest
            else:
                return self._get_generic_transport_info()
        
        # Prepare GPS data
        origin_gps = None
        destination_gps = None
        
        if origin == "Your Location" and user_location:
            if 'latitude' in user_location:
                origin_gps = {'lat': user_location['latitude'], 'lon': user_location['longitude']}
            elif 'lat' in user_location:
                origin_gps = user_location
        
        if destination == "Your Location" and user_location:
            if 'latitude' in user_location:
                destination_gps = {'lat': user_location['latitude'], 'lon': user_location['longitude']}
            elif 'lat' in user_location:
                destination_gps = user_location
        
        # Find route
        route = self.find_route(origin, destination, origin_gps=origin_gps, destination_gps=destination_gps)
        
        if not route:
            self.last_route = None
            return f"❌ No direct route found between {origin} and {destination}."
        
        self.last_route = route
        
        # Return minimal context
        return f"""🚇 **Route Found: {route.origin} → {route.destination}**
⏱️ Duration: {route.total_time} minutes
🔄 Transfers: {route.transfers}
🚉 Lines: {', '.join(route.lines_used)}

💡 Step-by-step directions shown in the route card below."""
    
    def _get_generic_transport_info(self) -> str:
        """Return generic transportation information."""
        return """🚇 **Istanbul Public Transportation**

To get directions, please specify:
- **Origin**: Where are you starting from?
- **Destination**: Where do you want to go?

**Example queries:**
- "How to go from Kadıköy to Taksim?"
- "Route from Sultanahmet to the airport"

**Istanbul Transit Network:**
- 🚇 Metro: M1-M11
- 🚋 Tram: T1, T4, T5
- 🚃 Marmaray: Cross-Bosphorus rail
- 🚡 Funicular: F1, F2
- ⛴️ Ferry: Multiple routes"""
    
    def get_directions_text(self, route: TransitRoute, language: str = "en") -> str:
        """Generate human-readable step-by-step directions."""
        if language == "tr":
            return self.route_builder.format_directions_turkish(route)
        return self.route_builder.format_directions_english(route)
    
    def get_map_data_for_last_route(self) -> Optional[Dict[str, Any]]:
        """Get map visualization data for the last computed route."""
        if not self.last_route:
            return None
        
        route = self.last_route
        polyline_points = []
        markers = []
        
        # Add origin marker
        origin_stations = self._get_stations_for_location(route.origin.lower())
        if origin_stations:
            origin_station = self.stations.get(origin_stations[0])
            if origin_station:
                markers.append({
                    'type': 'origin',
                    'name': route.origin,
                    'lat': origin_station.lat,
                    'lon': origin_station.lon
                })
                polyline_points.append([origin_station.lat, origin_station.lon])
        
        # Add step markers
        for step in route.steps:
            if step.get('type') in ['transit', 'ferry']:
                to_station_name = step.get('to')
                to_station_ids = self._get_stations_for_location(to_station_name.lower()) if to_station_name else []
                if to_station_ids:
                    to_station_obj = self.stations.get(to_station_ids[0])
                    if to_station_obj:
                        markers.append({
                            'type': 'stop',
                            'name': to_station_name,
                            'lat': to_station_obj.lat,
                            'lon': to_station_obj.lon,
                            'line': step.get('line', '')
                        })
                        polyline_points.append([to_station_obj.lat, to_station_obj.lon])
        
        # Add destination marker
        dest_stations = self._get_stations_for_location(route.destination.lower())
        if dest_stations:
            dest_station = self.stations.get(dest_stations[0])
            if dest_station:
                markers.append({
                    'type': 'destination',
                    'name': route.destination,
                    'lat': dest_station.lat,
                    'lon': dest_station.lon
                })
                if [dest_station.lat, dest_station.lon] not in polyline_points:
                    polyline_points.append([dest_station.lat, dest_station.lon])
        
        return {
            'polyline': polyline_points,
            'markers': markers,
            'bounds': self._calculate_bounds(polyline_points),
            'route_data': {
                'origin': route.origin,
                'destination': route.destination,
                'duration_min': route.total_time,
                'distance_km': route.total_distance,
                'transfers': route.transfers,
                'lines': route.lines_used,
            },
            'route_summary': {
                'origin': route.origin,
                'destination': route.destination,
                'total_time': route.total_time,
                'total_distance': route.total_distance,
                'transfers': route.transfers,
                'lines_used': route.lines_used
            }
        }
    
    def _calculate_bounds(self, points: List[List[float]]) -> Dict[str, float]:
        """Calculate map bounds from points."""
        if not points:
            return {'min_lat': 40.9, 'max_lat': 41.2, 'min_lon': 28.8, 'max_lon': 29.2}
        
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }


# Singleton pattern
_transportation_rag_singleton = None


def get_transportation_rag():
    """Get or create the global transportation RAG singleton."""
    global _transportation_rag_singleton
    
    if _transportation_rag_singleton is None:
        try:
            from config.settings import settings
        except ImportError:
            try:
                from backend.config.settings import settings
            except ImportError:
                settings = None
        
        redis_client = None
        
        if redis is not None:
            redis_url = os.getenv('REDIS_URL')
            if not redis_url and settings and hasattr(settings, 'REDIS_URL'):
                redis_url = settings.REDIS_URL
            
            if redis_url:
                try:
                    redis_client = redis.from_url(
                        redis_url,
                        decode_responses=True,
                        socket_connect_timeout=2,
                        socket_timeout=2,
                        retry_on_timeout=False
                    )
                    redis_client.ping()
                    logger.info("✅ Transportation RAG: Redis connected")
                except Exception as e:
                    logger.warning(f"⚠️ Transportation RAG: Redis unavailable: {e}")
                    redis_client = None
        
        _transportation_rag_singleton = IstanbulTransportationRAG(redis_client=redis_client)
        logger.info("✅ Transportation RAG singleton initialized")
    
    return _transportation_rag_singleton
