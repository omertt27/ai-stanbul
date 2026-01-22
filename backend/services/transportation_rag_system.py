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
- Week 3: Destination type system (island, ferry-only, walking distance)

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
from enum import Enum
import asyncio  # For async LLM calls
import concurrent.futures  # For sync wrapper of async LLM calls
import math  # For Haversine distance calculation
import os  # For environment variables

# Redis import (optional, for caching)
try:
    import redis
except ImportError:
    redis = None

# Station normalization is imported later in __init__ from transportation_station_normalization

logger = logging.getLogger(__name__)

# =============================================================================
# DESTINATION TYPE SYSTEM (Week 3 Fix)
# =============================================================================
class DestinationType(Enum):
    """
    Destination classification for proper routing.
    
    This prevents sending island destinations to rail-only route solvers.
    """
    STATION = "station"        # Direct transit station
    AREA = "area"              # Neighborhood/district (map to nearest station)
    ISLAND = "island"          # Ferry-only destination (Princes' Islands)
    ATTRACTION = "attraction"  # Tourist attraction (map to nearest station)
    FERRY_TERMINAL = "ferry_terminal"  # Ferry pier
    WALKING = "walking"        # Destination within walking distance


@dataclass
class DestinationInfo:
    """Information about a destination for routing."""
    name: str
    dest_type: DestinationType
    access_mode: str  # 'rail', 'ferry', 'walk', 'multi'
    terminals: List[str]  # Access points (stations/piers)
    walking_time_min: Optional[int] = None  # Minutes if walkable
    
    
# Island destinations - NEVER send to rail router
ISLAND_DESTINATIONS = {
    "buyukada": DestinationInfo(
        name="Büyükada",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
    ),
    "büyükada": DestinationInfo(
        name="Büyükada",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
    ),
    "heybeliada": DestinationInfo(
        name="Heybeliada",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
    ),
    "burgazada": DestinationInfo(
        name="Burgazada",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
    ),
    "kinaliada": DestinationInfo(
        name="Kınalıada",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
    ),
    "kınalıada": DestinationInfo(
        name="Kınalıada",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
    ),
    "princes islands": DestinationInfo(
        name="Princes' Islands",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
    ),
    "adalar": DestinationInfo(
        name="Princes' Islands",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
    ),
    "sedef adası": DestinationInfo(
        name="Sedef Adası",
        dest_type=DestinationType.ISLAND,
        access_mode="ferry",
        terminals=["FERRY-Bostancı"]
    ),
}

# Walking distance threshold in meters
WALK_THRESHOLD_METERS = 800  # ~10 minute walk


def get_destination_type(destination: str) -> DestinationInfo:
    """
    Classify a destination before routing.
    
    Args:
        destination: User-provided destination string
        
    Returns:
        DestinationInfo with routing metadata
    """
    dest_normalized = destination.lower().strip()
    
    # Check if it's an island
    if dest_normalized in ISLAND_DESTINATIONS:
        return ISLAND_DESTINATIONS[dest_normalized]
    
    # Check for island patterns
    island_patterns = ['ada', 'island', 'adası']
    for pattern in island_patterns:
        if pattern in dest_normalized:
            # Generic island - default to Büyükada ferry terminals
            return DestinationInfo(
                name=destination,
                dest_type=DestinationType.ISLAND,
                access_mode="ferry",
                terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
            )
    
    # Default: area/station (will be resolved by normal routing)
    return DestinationInfo(
        name=destination,
        dest_type=DestinationType.AREA,
        access_mode="rail",
        terminals=[]
    )


def is_walking_distance(origin_coords: Tuple[float, float], dest_coords: Tuple[float, float]) -> Tuple[bool, int]:
    """
    Check if destination is within walking distance.
    
    Args:
        origin_coords: (lat, lon) of origin
        dest_coords: (lat, lon) of destination
        
    Returns:
        Tuple of (is_walkable, estimated_walk_time_minutes)
    """
    from math import radians, cos, sin, asin, sqrt
    
    lat1, lon1 = origin_coords
    lat2, lon2 = dest_coords
    
    # Haversine formula
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    meters = 6371000 * c  # Earth radius in meters
    
    # Estimate walking time: ~80m/minute average walking speed
    walk_time_minutes = int(meters / 80)
    
    return (meters <= WALK_THRESHOLD_METERS, walk_time_minutes)


# =============================================================================
# TRANSPORT ENTITY WHITELIST (Week 3 Fix)
# =============================================================================
# These are VALID transport entities - NOT hallucinations
VALID_TRANSPORT_ENTITIES = {
    # Metro lines
    "m1", "m1a", "m1b", "m2", "m3", "m4", "m5", "m6", "m7", "m9", "m11",
    # Tram lines  
    "t1", "t4", "t5",
    # Other transit
    "marmaray", "f1", "f2", "metrobus", "ferry", "ido", "turyol", "şehir hatları",
    # Common stations mentioned
    "sirkeci", "yenikapı", "yenikapi", "ayrılık çeşmesi", "ayrilik cesmesi",
    "kadıköy", "kadikoy", "üsküdar", "uskudar", "taksim", "levent",
    "mecidiyeköy", "mecidiyekoy", "şişli", "sisli", "osmanbey",
    "kabataş", "kabatas", "karaköy", "karakoy", "eminönü", "eminonu",
    "sultanahmet", "beyazıt", "beyazit", "aksaray", "zeytinburnu",
    "bağcılar", "bagcilar", "kirazlı", "kirazli", "otogar",
    "bakırköy", "bakirkoy", "yeşilköy", "yesilkoy", "florya",
    "pendik", "kartal", "maltepe", "bostancı", "bostanci",
    "beşiktaş", "besiktas", "ortaköy", "ortakoy", "bebek",
    "hacıosman", "hacimosman", "maslak", "gayrettepe",
    "4. levent", "4 levent", "levent", "zincirlikuyu",
    # Islands
    "büyükada", "buyukada", "heybeliada", "burgazada", "kınalıada", "kinaliada",
    "adalar", "princes islands", "sedef adası",
    # Airport stations
    "istanbul havalimanı", "istanbul havalimani", "atatürk havalimanı",
    "sabiha gökçen", "sabiha gokcen",
}


def is_valid_transport_entity(entity: str) -> bool:
    """
    Check if an entity is a valid transport-related term.
    
    Use this to prevent marking transport entities as hallucinations.
    """
    entity_lower = entity.lower().strip()
    
    # Direct match
    if entity_lower in VALID_TRANSPORT_ENTITIES:
        return True
    
    # Check for partial matches (station names often appear in compound form)
    for valid_entity in VALID_TRANSPORT_ENTITIES:
        if valid_entity in entity_lower or entity_lower in valid_entity:
            return True
    
    # Check metro/tram line patterns
    if re.match(r'^m\d{1,2}$', entity_lower):
        return True
    if re.match(r'^t\d$', entity_lower):
        return True
    if re.match(r'^f\d$', entity_lower):
        return True
        
    return False


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
    ranking_scores: Dict[str, float] = None  # Ranking scores (fastest, scenic, etc.)
    
    def __post_init__(self):
        """Initialize ranking_scores if not provided"""
        if self.ranking_scores is None:
            self.ranking_scores = {}
    
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
        self.station_graph = self.stations  # Alias for backward compatibility
        self.routes = self._build_route_patterns()
        self.neighborhoods = self._build_neighborhood_stations()
        self.station_aliases = self._build_station_aliases()
        self.last_route = None  # Store last computed route for mapData extraction
        
        # Initialize travel time database for weighted routing
        try:
            from services.transportation_travel_times import get_travel_time_database
        except ImportError:
            from backend.services.transportation_travel_times import get_travel_time_database
        self.travel_time_db = get_travel_time_database()
        
        # Week 2 Improvement: Station/Line ID normalization
        try:
            from services.transportation_station_normalization import get_station_normalizer
        except ImportError:
            from backend.services.transportation_station_normalization import get_station_normalizer
        self.station_normalizer = get_station_normalizer()
        
        # Week 1 Improvement #3: Route caching
        self.redis = redis_client
        self.redis_client = redis_client  # Alias for backward compatibility
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
        - Accent removal (ı→i, ö→o, ü→u, ş→s, ğ→g, ç→c, İ→i)
        - Common suffixes (square, station, metro, tram, etc.)
        - Extra whitespace
        
        Examples:
            "Taksim Square" → "taksim"
            "Kadıköy" → "kadikoy"
            "Beşiktaş Metro" → "besiktas"
            "İkitelli" → "ikitelli"
        """
        # First, handle Turkish character mapping BEFORE lowercasing
        # This is critical because İ (capital i-dot) lowercase in Turkish is i̇ (dotted i), not i
        turkish_char_map = {
            'İ': 'i',  # Capital İ → i (MUST be done before lowercase!)
            'I': 'i',  # Capital I → i
            'ı': 'i',  # Lowercase ı → i
            'Ö': 'o',
            'ö': 'o',
            'Ü': 'u',
            'ü': 'u',
            'Ş': 's',
            'ş': 's',
            'Ğ': 'g',
            'ğ': 'g',
            'Ç': 'c',
            'ç': 'c'
        }
        
        for turkish_char, latin_char in turkish_char_map.items():
            name = name.replace(turkish_char, latin_char)
        
        # Now convert to lowercase (after Turkish char mapping)
        name = name.lower().strip()
        
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
            
            # Kadıköy area - Include BOTH M4 and FERRY stations for route alternatives
            # This allows the ranking system to compare rail vs ferry routes
            "kadikoy": ["M4-Kadıköy", "FERRY-Kadıköy"],
            "kadıköy": ["M4-Kadıköy", "FERRY-Kadıköy"],
            "kadıkoy": ["M4-Kadıköy", "FERRY-Kadıköy"],
            # Ayrılık Çeşmesi is a separate transfer station, not Kadıköy
            "ayrilik cesmesi": ["M4-Ayrılık Çeşmesi", "MARMARAY-Ayrılık Çeşmesi"],
            "ayrılık çeşmesi": ["M4-Ayrılık Çeşmesi", "MARMARAY-Ayrılık Çeşmesi"],
            
            # Taksim area - main station only (Yenikapı is for transfer, not destination)
            "taksim": ["M2-Taksim"],
            
            # Beşiktaş area
            "besiktas": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            "beşiktas": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            "beşiktaş": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            
            # Sultanahmet/Fatih area
            "sultanahmet": ["T1-Sultanahmet"],
            "sultanahmet square": ["T1-Sultanahmet"],
            "blue mosque": ["T1-Sultanahmet"],
            "hagia sophia": ["T1-Sultanahmet"],
            "ayasofya": ["T1-Sultanahmet"],
            
            # Galata/Karaköy area
            "galata": ["T1-Karaköy"],  # Galata tower is near Karaköy
            "galata tower": ["T1-Karaköy"],
            "karakoy": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
            "karaköy": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
            
            # Üsküdar area
            "uskudar": ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"],
            "üsküdar": ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"],
            "uskudar square": ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"],
            
            # Istiklal/Beyoğlu area
            "istiklal": ["M2-Taksim"],
            "istiklal street": ["M2-Taksim"],
            "istiklal caddesi": ["M2-Taksim"],
            "beyoglu": ["M2-Taksim", "T1-Karaköy"],
            "beyoğlu": ["M2-Taksim", "T1-Karaköy"],
            
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
            "eminonu": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü"],
            "eminönü": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü"],
            "spice bazaar": ["T1-Eminönü"],
            "misir carsisi": ["T1-Eminönü"],
            
            # Sirkeci
            "sirkeci": ["MARMARAY-Sirkeci", "T1-Sirkeci"],
            
            # Levent area
            "levent": ["M2-Levent", "M6-Levent"],
            "4.levent": ["M2-4. Levent"],
            "4 levent": ["M2-4. Levent"],
            
            # Şişli/Mecidiyeköy area - Updated to match canonical IDs
            "sisli": ["M2-Şişli-Mecidiyeköy"],
            "şişli": ["M2-Şişli-Mecidiyeköy"],
            "mecidiyekoy": ["M7-Mecidiyeköy", "M2-Şişli-Mecidiyeköy"],
            "mecidiyeköy": ["M7-Mecidiyeköy", "M2-Şişli-Mecidiyeköy"],
            
            # Olimpiyat/İkitelli area (M3/M9 transfer point)
            "olimpiyat": ["M3-Olimpiyat", "M9-Olimpiyat"],
            "ikitelli": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
            "İkitelli": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
            "ikitelli sanayi": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
            "İkitelli sanayi": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
            "İkitelli Sanayi": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
            
            # Bostancı area
            "bostanci": ["M4-Bostancı", "MARMARAY-Bostancı"],
            "bostancı": ["M4-Bostancı", "MARMARAY-Bostancı"],
            
            # Pendik
            "pendik": ["MARMARAY-Pendik"],
            
            # ====== LANDMARKS & TOURIST ATTRACTIONS ======
            # Palaces
            "dolmabahce": ["T1-Kabataş"],
            "dolmabahce palace": ["T1-Kabataş"],
            "dolmabahçe": ["T1-Kabataş"],
            "dolmabahçe palace": ["T1-Kabataş"],
            "dolmabahçe sarayı": ["T1-Kabataş"],
            "topkapi": ["T1-Gülhane", "T1-Sultanahmet"],
            "topkapi palace": ["T1-Gülhane", "T1-Sultanahmet"],
            "topkapı": ["T1-Gülhane", "T1-Sultanahmet"],
            "topkapı palace": ["T1-Gülhane", "T1-Sultanahmet"],
            "topkapı sarayı": ["T1-Gülhane", "T1-Sultanahmet"],
            
            # Mosques & Religious Sites
            "blue mosque": ["T1-Sultanahmet"],
            "sultan ahmed mosque": ["T1-Sultanahmet"],
            "sultanahmet camii": ["T1-Sultanahmet"],
            "suleymaniye": ["T1-Beyazıt-Kapalıçarşı"],
            "suleymaniye mosque": ["T1-Beyazıt-Kapalıçarşı"],
            "süleymaniye": ["T1-Beyazıt-Kapalıçarşı"],
            "süleymaniye camii": ["T1-Beyazıt-Kapalıçarşı"],
            
            # Museums
            "hagia sophia": ["T1-Sultanahmet"],
            "ayasofya": ["T1-Sultanahmet"],
            "aya sofya": ["T1-Sultanahmet"],
            "archaeological museum": ["T1-Gülhane"],
            "arkeoloji muzesi": ["T1-Gülhane"],
            
            # Markets & Shopping
            "grand bazaar": ["T1-Beyazıt-Kapalıçarşı"],
            "kapali carsi": ["T1-Beyazıt-Kapalıçarşı"],
            "kapalıçarşı": ["T1-Beyazıt-Kapalıçarşı"],
            "spice bazaar": ["T1-Eminönü"],
            "misir carsisi": ["T1-Eminönü"],
            "egyptian bazaar": ["T1-Eminönü"],
            
            # Galata/Beyoglu Landmarks
            "galata bridge": ["T1-Karaköy", "T1-Eminönü"],
            "galata köprüsü": ["T1-Karaköy", "T1-Eminönü"],
            
            # Parks
            "gulhane": ["T1-Gülhane"],
            "gülhane": ["T1-Gülhane"],
            "gulhane park": ["T1-Gülhane"],
            "gülhane parkı": ["T1-Gülhane"],
            
            # ====== NEIGHBORHOODS (Additional) ======
            "ortakoy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],  # Ortaköy is near Beşiktaş
            "ortaköy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            "balat": ["T4-Fener", "T4-Balat"],  # Historic neighborhood
            "fener": ["T4-Fener"],
            "fatih": ["T4-Fatih", "T1-Aksaray"],
            "aksaray": ["T1-Aksaray", "M1A-Aksaray"],
            "cihangir": ["M2-Taksim"],  # Near Taksim
            "galatasaray": ["M2-Taksim"],  # On Istiklal
            "nisantasi": ["M2-Osmanbey"],
            "nişantaşı": ["M2-Osmanbey"],
            "osmanbey": ["M2-Osmanbey"],
            "bebek": ["T4-Beşiktaş", "FERRY-Beşiktaş"],  # Near Beşiktaş
            
            # ====== GENERIC DESTINATIONS ======
            # Asian Side
            "asian side": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
            "asia": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
            "anadolu yakasi": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
            "anadolu yakası": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
            
            # European Side
            "european side": ["M2-Taksim"],
            "europe": ["M2-Taksim"],
            "avrupa yakasi": ["M2-Taksim"],
            "avrupa yakası": ["M2-Taksim"],
            
            # City Center
            "city center": ["M2-Taksim", "T1-Sultanahmet"],
            "city centre": ["M2-Taksim", "T1-Sultanahmet"],
            "center": ["M2-Taksim", "T1-Sultanahmet"],
            "centre": ["M2-Taksim", "T1-Sultanahmet"],
            "downtown": ["M2-Taksim"],
            "sehir merkezi": ["M2-Taksim", "T1-Sultanahmet"],
            "şehir merkezi": ["M2-Taksim", "T1-Sultanahmet"],
            
            # Old City / Historic Peninsula
            "old city": ["T1-Sultanahmet"],
            "historic peninsula": ["T1-Sultanahmet"],
            "tarihi yarimada": ["T1-Sultanahmet"],
            "tarihi yarımada": ["T1-Sultanahmet"],
            
            # Islands
            "princes islands": ["FERRY-Kadıköy", "FERRY-Eminönü"],  # Ferry from these
            "adalar": ["FERRY-Kadıköy", "FERRY-Eminönü"],
            "buyukada": ["FERRY-Kadıköy", "FERRY-Eminönü"],
            "büyükada": ["FERRY-Kadıköy", "FERRY-Eminönü"],
            "heybeliada": ["FERRY-Kadıköy", "FERRY-Eminönü"],
            
            # Sabiha Gökçen Airport (Asian side)
            "sabiha gokcen": ["M4-Sabiha Gökçen Havalimanı"],
            "sabiha gökçen": ["M4-Sabiha Gökçen Havalimanı"],
            "saw": ["M4-Sabiha Gökçen Havalimanı"],
            "sabiha gokcen airport": ["M4-Sabiha Gökçen Havalimanı"],
            "sabiha gökçen airport": ["M4-Sabiha Gökçen Havalimanı"],
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
        try:
            from services.transportation_station_normalization import get_station_normalizer
        except ImportError:
            from backend.services.transportation_station_normalization import get_station_normalizer
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
            "kadikoy": ["M4-Kadıköy", "M4-Ayrılık Çeşmesi"],
            "üsküdar": ["MARMARAY-Üsküdar", "M5-Üsküdar"],
            "uskudar": ["MARMARAY-Üsküdar", "M5-Üsküdar"],
            "bostancı": ["MARMARAY-Bostancı", "M4-Bostancı"],
            "bostanci": ["MARMARAY-Bostancı", "M4-Bostancı"],
            "pendik": ["MARMARAY-Pendik", "M4-Pendik"],
            "kartal": ["MARMARAY-Kartal", "M4-Kartal"],
            "maltepe": ["M4-Maltepe"],
            "ataşehir": ["M4-Ünalan", "M4-Kozyatağı"],
            "atasehir": ["M4-Ünalan", "M4-Kozyatağı"],
            
            # EUROPEAN SIDE - Historic/Tourist
            "taksim": ["M2-Taksim"],
            "beyoğlu": ["M2-Taksim", "T1-Karaköy"],
            "beyoglu": ["M2-Taksim", "T1-Karaköy"],
            "sultanahmet": ["T1-Sultanahmet"],
            "eminönü": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü", "MARMARAY-Sirkeci"],
            "eminonu": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü", "MARMARAY-Sirkeci"],
            "karaköy": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
            "karakoy": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
            "kabataş": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
            "kabatas": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
            "beşiktaş": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            "besiktas": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            "şişli": ["M2-Şişli-Mecidiyeköy"],
            "sisli": ["M2-Şişli-Mecidiyeköy"],
            "levent": ["M2-Levent", "M2-4. Levent", "M6-Levent"],
            "mecidiyeköy": ["M2-Şişli-Mecidiyeköy", "M7-Mecidiyeköy"],
            "mecidiyekoy": ["M2-Şişli-Mecidiyeköy", "M7-Mecidiyeköy"],
            "zeytinburnu": ["T1-Zeytinburnu", "MARMARAY-Zeytinburnu"],
            "bakırköy": ["MARMARAY-Bakırköy"],
            "bakirkoy": ["MARMARAY-Bakırköy"],
            "yeşilköy": ["MARMARAY-Yeşilköy"],
            "yesilkoy": ["MARMARAY-Yeşilköy"],
            
            # EUROPEAN SIDE - Additional Neighborhoods
            "fatih": ["T4-Fatih", "T1-Aksaray"],
            "aksaray": ["T1-Aksaray", "M1A-Aksaray"],
            "balat": ["T4-Fener", "T4-Balat"],
            "fener": ["T4-Fener"],
            "ortaköy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            "ortakoy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            "nişantaşı": ["M2-Osmanbey"],
            "nisantasi": ["M2-Osmanbey"],
            "osmanbey": ["M2-Osmanbey"],
            "cihangir": ["M2-Taksim"],
            "galata": ["T1-Karaköy"],
            "bebek": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
            "etiler": ["M2-4. Levent"],
            "maslak": ["M2-Hacıosman"],
            "sariyer": ["M2-Hacıosman"],
            "sarıyer": ["M2-Hacıosman"],
            
            # Airports
            "atatürk airport": ["M1A-Atatürk Havalimanı"],  # Closed airport, legacy support
            "ataturk airport": ["M1A-Atatürk Havalimanı"],
            "istanbul airport": ["M11-İstanbul Havalimanı"],
            "new airport": ["M11-İstanbul Havalimanı"],
            "sabiha gökçen": ["M4-Sabiha Gökçen Havalimanı"],
            "sabiha gokcen": ["M4-Sabiha Gökçen Havalimanı"],
            
            # Transfer Hubs
            "yenikapı": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
            "yenikapi": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
            "sirkeci": ["MARMARAY-Sirkeci", "T1-Sirkeci"],
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
        
        WEEK 3 IMPROVEMENTS:
        - Destination type detection (island, walking, etc.)
        - Walking distance short-circuit
        - Ferry routing for islands
        
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
        origin_normalized = origin.lower().strip()
        destination_normalized = destination.lower().strip()
        
        # =================================================================
        # CHECK FOR DEPRECATED STATIONS (e.g., Atatürk Airport)
        # =================================================================
        deprecated_check = self._check_deprecated_stations(origin_normalized, destination_normalized)
        if deprecated_check:
            logger.warning(f"⚠️ Deprecated station detected: {deprecated_check}")
            # Return a special route with deprecation message
            return self._create_deprecation_route(origin, destination, deprecated_check)
        
        # =================================================================
        # WEEK 3 FIX #1: Destination Type Detection
        # =================================================================
        dest_info = get_destination_type(destination_normalized)
        logger.info(f"🎯 Destination type: {dest_info.dest_type.value} for '{destination}'")
        
        # =================================================================
        # WEEK 3 FIX #2: Walking Distance Short-Circuit
        # =================================================================
        # If same origin and destination OR within walking distance, skip transit routing
        if origin_normalized == destination_normalized:
            logger.info(f"🚶 Same origin and destination - returning walking response")
            walking_route = self._create_walking_route(origin, destination, walk_time=2)
            self.last_route = walking_route
            return walking_route
        
        # Check walking distance if we have GPS for both
        if origin_gps and destination_gps:
            origin_coords = (origin_gps.get('lat', 0), origin_gps.get('lon', 0))
            dest_coords = (destination_gps.get('lat', 0), destination_gps.get('lon', 0))
            
            is_walkable, walk_time = is_walking_distance(origin_coords, dest_coords)
            if is_walkable:
                logger.info(f"🚶 Destination within walking distance ({walk_time} min) - returning walking response")
                walking_route = self._create_walking_route(origin, destination, walk_time=walk_time)
                self.last_route = walking_route
                return walking_route
        
        # =================================================================
        # WEEK 3 FIX #3: Island Routing (Ferry-Only Destinations)
        # =================================================================
        if dest_info.dest_type == DestinationType.ISLAND:
            logger.info(f"🛳️ Island destination detected - using ferry routing")
            island_route = self._create_island_route(origin, destination, dest_info, origin_gps)
            if island_route:
                self.last_route = island_route
                return island_route
            # Fall through to normal routing if island route fails
        
        # =================================================================
        # NORMAL TRANSIT ROUTING
        # =================================================================
        # Week 1 Improvement #3: Try cache first (skip cache if GPS-based - those are dynamic)
        use_cache = not (origin_gps or destination_gps)
        cache_key = f"route:{origin_normalized}|{destination_normalized}|{max_transfers}"
        
        if use_cache and self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    self.cache_hits += 1
                    logger.info(f"⚡ Route cache HIT ({self.cache_hits} total): {origin} → {destination}")
                    # Deserialize cached route
                    route_dict = json.loads(cached)
                    cached_route = self._dict_to_route(route_dict)
                    # 🔥 CRITICAL FIX: Store cached route as last_route for map visualization
                    self.last_route = cached_route
                    return cached_route
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
                origin_stations = self._get_stations_for_location(origin_normalized)
        else:
            origin_stations = self._get_stations_for_location(origin_normalized)
        
        if destination_gps and isinstance(destination_gps, dict) and 'lat' in destination_gps and 'lon' in destination_gps:
            nearest_dest = self.find_nearest_station(destination_gps['lat'], destination_gps['lon'])
            if nearest_dest:
                dest_stations = [nearest_dest]
                logger.info(f"✅ Using nearest station for GPS destination: {self.stations[nearest_dest].name}")
            else:
                dest_stations = self._get_stations_for_location(destination_normalized)
        else:
            dest_stations = self._get_stations_for_location(destination_normalized)
        
        if not origin_stations or not dest_stations:
            logger.warning(f"Could not find stations for {origin} or {destination}")
            return None
        
        # Find best route - collect alternatives for ranking
        all_routes = []
        
        for orig_station in origin_stations:
            for dest_station in dest_stations:
                route = self._find_path(orig_station, dest_station, max_transfers)
                if route:
                    all_routes.append(route)
                    logger.debug(f"Found route: {orig_station} → {dest_station} in {route.total_time:.0f} min")
        
        if not all_routes:
            logger.warning(f"No routes found between {origin} and {destination}")
            return None
        
        # Rank routes by different criteria (fastest, scenic, etc.)
        ranked_routes = self._rank_routes(all_routes, origin_gps)
        
        # Best route is the fastest (first in ranked list)
        best_route = ranked_routes[0]
        
        # Add alternatives (top 3 routes)
        if len(ranked_routes) > 1:
            best_route.alternatives = ranked_routes[1:min(4, len(ranked_routes))]
        
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
        
        # 🔥 CRITICAL FIX: Store last_route for map visualization
        # This enables get_map_data_for_last_route() to work correctly
        self.last_route = best_route
        
        return best_route
    
    def _create_walking_route(self, origin: str, destination: str, walk_time: int) -> TransitRoute:
        """Create a walking-only route for nearby destinations."""
        return TransitRoute(
            origin=origin,
            destination=destination,
            total_time=walk_time,
            total_distance=walk_time * 0.08,  # ~80m/min walking
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
    
    def _create_island_route(
        self, 
        origin: str, 
        destination: str, 
        dest_info: DestinationInfo,
        origin_gps: Optional[Dict[str, float]] = None
    ) -> Optional[TransitRoute]:
        """
        Create a two-phase route to an island destination.
        
        Phase 1: Get to ferry terminal
        Phase 2: Ferry to island
        """
        # Find the best ferry terminal based on origin
        # Prefer terminals on the same side of the city
        terminal_priorities = {
            'european': ['FERRY-Kabataş', 'FERRY-Eminönü', 'FERRY-Karaköy'],
            'asian': ['FERRY-Kadıköy', 'FERRY-Bostancı']
        }
        
        # Determine which side of Istanbul the origin is on
        origin_lower = origin.lower()
        asian_keywords = ['kadıköy', 'kadikoy', 'üsküdar', 'uskudar', 'bostancı', 'bostanci', 
                         'pendik', 'kartal', 'maltepe', 'ataşehir', 'atasehir', 'asian']
        
        if any(kw in origin_lower for kw in asian_keywords):
            preferred_terminals = terminal_priorities['asian']
        else:
            preferred_terminals = terminal_priorities['european']
        
        # Try to find route to ferry terminal
        for terminal in preferred_terminals:
            terminal_name = terminal.split('-')[1] if '-' in terminal else terminal
            
            # Find route to terminal (recursive, but without island check)
            terminal_route = self._find_route_to_terminal(origin, terminal_name, origin_gps)
            
            if terminal_route:
                # Add ferry step to island
                ferry_time = 45 if 'bostancı' in terminal.lower() else 60  # Bostancı is closer to islands
                
                # Combine routes
                combined_steps = terminal_route.steps.copy()
                combined_steps.append({
                    'type': 'ferry',
                    'line': 'İDO/Şehir Hatları',
                    'instruction': f"Take the ferry from {terminal_name} to {dest_info.name}",
                    'from_station': terminal_name,
                    'to_station': dest_info.name,
                    'duration': ferry_time,
                    'details': f"Ferry service to {dest_info.name}. Ferries run approximately every 30-60 minutes."
                })
                
                island_route = TransitRoute(
                    origin=origin,
                    destination=dest_info.name,
                    total_time=terminal_route.total_time + ferry_time,
                    total_distance=terminal_route.total_distance + 15,  # ~15km to islands
                    steps=combined_steps,
                    transfers=terminal_route.transfers + 1,
                    lines_used=terminal_route.lines_used + ['FERRY'],
                    alternatives=[],
                    time_confidence='medium'  # Ferry schedules can vary
                )
                
                logger.info(f"✅ Created island route: {origin} → {terminal_name} → {dest_info.name}")
                return island_route
        
        logger.warning(f"Could not create island route to {destination}")
        return None
    
    def _find_route_to_terminal(
        self, 
        origin: str, 
        terminal: str,
        origin_gps: Optional[Dict[str, float]] = None
    ) -> Optional[TransitRoute]:
        """Find route to a ferry terminal (internal, avoids island routing loop)."""
        origin_normalized = origin.lower().strip()
        terminal_normalized = terminal.lower().strip()
        
        # Get stations
        if origin_gps and isinstance(origin_gps, dict) and 'lat' in origin_gps and 'lon' in origin_gps:
            nearest_origin = self.find_nearest_station(origin_gps['lat'], origin_gps['lon'])
            if nearest_origin:
                origin_stations = [nearest_origin]
            else:
                origin_stations = self._get_stations_for_location(origin_normalized)
        else:
            origin_stations = self._get_stations_for_location(origin_normalized)
        
        terminal_stations = self._get_stations_for_location(terminal_normalized)
        
        if not origin_stations or not terminal_stations:
            return None
        
        # Find best route
        best_route = None
        best_time = float('inf')
        
        for orig_station in origin_stations:
            for term_station in terminal_stations:
                route = self._find_path(orig_station, term_station, max_transfers=3)
                if route and route.total_time < best_time:
                    best_route = route
                    best_time = route.total_time
        
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
        
        # Strategy 2b: Search all aliases with normalized comparison
        for alias, stations in self.station_aliases.items():
            alias_normalized = self._normalize_station_name(alias)
            if normalized_location == alias_normalized:
                logger.debug(f"✅ Found via normalized alias: {alias_normalized} → {stations}")
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
        
        # Strategy 3b: Search all neighborhoods with normalized comparison
        for neighborhood, stations in self.neighborhoods.items():
            neighborhood_normalized = self._normalize_station_name(neighborhood)
            if normalized_location == neighborhood_normalized:
                logger.debug(f"✅ Found via normalized neighborhood: {neighborhood_normalized} → {stations}")
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
        Get ONLY the physically adjacent stations on the same line.
        
        CRITICAL: Returns only the immediate neighbors (1 stop before, 1 stop after),
        NOT all stations on the line. This fixes the phantom stations issue.
        
        SPECIAL HANDLING FOR FERRIES:
        - Ferries are point-to-point connections, not linear routes
        - Returns all ferry terminals on the same ferry network (direct connections)
        - This allows ferry routes to be computed correctly
        
        The station normalizer provides stations in their correct physical sequence.
        We use that ordering to find true adjacent stations.
        """
        if station_id not in self.stations:
            return []
        
        current_line = self.stations[station_id].line
        neighbors = []
        
        # SPECIAL CASE: Ferry connections
        if current_line.upper() == "FERRY":
            # For ferries, return all other ferry terminals as direct neighbors
            # This is because ferries are point-to-point, not sequential
            for other_id, other_station in self.stations.items():
                if (other_station.line.upper() == "FERRY" and 
                    other_id != station_id):
                    neighbors.append(other_id)
            logger.debug(f"🛳️ Ferry {station_id} has {len(neighbors)} direct connections")
            return neighbors
        
        try:
            # Get stations on this line from the canonical normalizer
            # The normalizer stores stations in the correct physical sequence
            line_station_ids = self.station_normalizer.get_stations_on_line_in_order(current_line)
            
            if not line_station_ids:
                # Fallback: get all stations on this line (shouldn't happen)
                logger.warning(f"No ordered stations found for line {current_line}, using fallback")
                line_stations = [(sid, st) for sid, st in self.stations.items() if st.line == current_line]
                # Sort by longitude (west to east) - this is a reasonable approximation
                line_stations.sort(key=lambda x: x[1].gps[1] if x[1].gps else 0)
                line_station_ids = [sid for sid, _ in line_stations]
            
            # Find current station's position in the ordered list
            try:
                current_idx = line_station_ids.index(station_id)
                
                # Add previous station (if exists)
                if current_idx > 0:
                    neighbors.append(line_station_ids[current_idx - 1])
                
                # Add next station (if exists)
                if current_idx < len(line_station_ids) - 1:
                    neighbors.append(line_station_ids[current_idx + 1])
                    
            except ValueError:
                logger.warning(f"Station {station_id} not found in ordered list for line {current_line}")
                
        except Exception as e:
            logger.error(f"Error getting same-line neighbors for {station_id}: {e}")
        
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
        
        # Debug: Log the path for ferry routes
        origin_station = self.stations[path[0]]
        dest_station = self.stations[path[-1]]
        if any(self.stations[sid].line.upper() == "FERRY" for sid in path):
            logger.info(f"🛳️ Building ferry route: {origin_station.name} → {dest_station.name}")
            logger.info(f"   Path ({len(path)} stations): {' → '.join([self.stations[sid].name for sid in path])}")
        
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
                
                # Only add if not zero-length (fixes duplicate station steps)
                if start_station.name != end_station.name:
                    # Calculate stops (don't count origin, no stops for ferry)
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
                
                # Add transfer step ONLY if valid
                transfer_penalty = self.travel_time_db.get_transfer_penalty(
                    current_line,
                    station.line
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
                
                # Start new segment
                current_line = station.line
                segment_start = i  # Start from the CURRENT station (new line), not the previous one
                segment_time = 0.0
            else:
                # Continue on same line
                segment_time += travel_time
        
        # Final segment
        start_station = self.stations[path[segment_start]]
        end_station = self.stations[path[-1]]
        
        # Only add if not zero-length (fixes duplicate station steps)
        if start_station.name != end_station.name:
            # Calculate stops (don't count origin, no stops for ferry)
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
        
        # CLEANUP: Remove unnecessary same-location transfers
        filtered_steps = []
        for i, step in enumerate(steps):
            if step.get('type') == 'transfer' and step.get('from') == step.get('to'):
                if i == 0 or i == len(steps) - 1:
                    logger.debug(f"⚠️ Removing unnecessary transfer: {step.get('from')} → {step.get('line')}")
                    continue
            filtered_steps.append(step)
        
        steps = filtered_steps
        
        # IMPROVED: Calculate distance using SEGMENT-BASED logic (fixes ferry+metro bug)
        # Calculate distance for each TRANSIT step, not the full path
        total_distance = 0.0
        
        for step in steps:
            if step.get('type') != 'transit':
                continue  # Skip transfer steps
            
            # Find station IDs for this step
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
                # Fallback: estimate from time
                total_distance += (step.get('duration', 0) / 10.0) * 1.5
                continue
            
            from_st = self.stations[from_id]
            to_st = self.stations[to_id]
            
            # For FERRY steps: always use direct Haversine distance
            if step.get('ferry_crossing'):
                seg_dist = self._haversine_distance(
                    from_st.lat, from_st.lon, to_st.lat, to_st.lon
                )
                logger.debug(f"🛳️ Ferry: {from_name} → {to_name} = {seg_dist:.2f} km (direct)")
                total_distance += seg_dist
            else:
                # For rail/metro: sum distances along the line
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
                        seg_dist += self._haversine_distance(s1.lat, s1.lon, s2.lat, s2.lon)
                    
                    total_distance += seg_dist
                except (ValueError, IndexError):
                    # Fallback: direct distance
                    seg_dist = self._haversine_distance(
                        from_st.lat, from_st.lon, to_st.lat, to_st.lon
                    )
                    total_distance += seg_dist
        
        # Fallback if calculation completely fails
        if total_distance == 0:
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
        
        # CRITICAL FIX: Recalculate transfers after step cleanup
        # The original 'transfers' count is from the raw path, but we cleaned up steps
        actual_transfers = sum(1 for step in steps if step.get('type') == 'transfer')
        
        return TransitRoute(
            origin=self.stations[path[0]].name,
            destination=self.stations[path[-1]].name,
            total_time=round(total_time),
            total_distance=round(total_distance, 2),
            steps=steps,
            transfers=actual_transfers,  # Use recalculated count
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
        """
        Create route for direct connection (no transfers) on same line.
        Uses actual station ordering to calculate correct stop count.
        NOW WITH TRANSPORT-TYPE-SPECIFIC TIMING!
        """
        # Get the canonical ordering of stations on this line
        line_stations = self.station_normalizer.get_stations_on_line_in_order(start.line)
        
        # Find indices of start and end stations
        start_idx = None
        end_idx = None
        
        for idx, station_id in enumerate(line_stations):
            station = self.stations.get(station_id)
            if station:
                if station.name == start.name:
                    start_idx = idx
                if station.name == end.name:
                    end_idx = idx
        
        # Calculate stop count (don't count origin)
        if start_idx is not None and end_idx is not None:
            stops = abs(end_idx - start_idx)
        else:
            # Fallback if we can't find the stations (shouldn't happen)
            stops = 5
        
        # Get actual travel time from database
        travel_time, confidence = self.travel_time_db.get_travel_time(
            f"{start.line}-{start.name}",
            f"{end.line}-{end.name}",
            default=0.0  # Return 0 if no data, don't use fallback
        )
        
        # If no data, estimate based on stops AND transport type
        if travel_time == 0 or confidence == "low":
            # Transport-type-specific timing rules
            line = start.line.upper()
            
            if line == "FERRY":
                # Ferry: Fixed schedules, typically 15-20 min per route
                # Ferries don't have "stops" in the traditional sense
                travel_time = max(15, stops * 3.0)  # Minimum 15 min, ~3 min between piers
                confidence = "medium"
            elif line == "MARMARAY":
                # Marmaray: High-speed rail, ~2.5-3 min per stop
                travel_time = stops * 2.8
                confidence = "medium"
            elif line.startswith("M"):  # Metro lines
                # Metro: ~2-2.5 min per stop
                travel_time = stops * 2.3
                confidence = "medium"
            elif line.startswith("T"):  # Tram lines
                # Tram: Slower due to street traffic, ~2.5-3 min per stop
                travel_time = stops * 2.7
                confidence = "medium"
            elif line == "F":  # Funicular
                # Funicular: Usually short, fast trips
                travel_time = max(2, stops * 1.5)
                confidence = "medium"
            else:
                # Default fallback
                travel_time = stops * 2.5
                confidence = "low"
            
            # Add minimum time constraint (no route under 2 min unless same station)
            if stops > 0:
                travel_time = max(2, travel_time)
        
        steps = [
            {
                "instruction": f"Take {start.line} from {start.name} to {end.name}",
                "line": start.line,
                "from": start.name,
                "to": end.name,
                "duration": round(travel_time, 1),
                "type": "transit",
                "stops": stops if start.line.upper() != "FERRY" else None,  # Ferry: no stop count
                "ferry_crossing": True if start.line.upper() == "FERRY" else False
            }
        ]
        
        # IMPROVED: Calculate actual geographic distance using GPS coordinates
        # This gives much more accurate distance estimates
        distance = self._calculate_route_distance(start, end, start_idx, end_idx, line_stations)
        
        # Ferry distance validation: Flag if ferry distance > 10km (likely a bug)
        if start.line.upper() == "FERRY" and distance > 10.0:
            logger.warning(f"⚠️ FERRY DISTANCE ANOMALY: {start.name} → {end.name} = {distance:.2f}km (>10km threshold)")
            logger.warning(f"   This may indicate a polyline/Haversine calculation bug. Please review ferry route data.")
        
        # Fallback if GPS calculation fails
        if distance == 0:
            # Distance estimation based on transport type (fallback)
            line = start.line.upper()
            if line == "FERRY":
                distance = max(2.0, stops * 0.8)
            elif line == "MARMARAY":
                distance = stops * 1.5
            elif line.startswith("M"):
                distance = stops * 1.1
            elif line.startswith("T"):
                distance = stops * 0.8
            else:
                distance = (travel_time / 10.0) * 1.5
        
        return TransitRoute(
            origin=start.name,
            destination=end.name,
            total_time=round(travel_time),
            total_distance=round(distance, 2),
            steps=steps,
            transfers=0,
            lines_used=[start.line],
            alternatives=[],
            time_confidence=confidence
        )
    
    def _calculate_route_distance(
        self,
        start_station: TransitStation,
        end_station: TransitStation,
        start_idx: Optional[int],
        end_idx: Optional[int],
        line_stations: List[str]
    ) -> float:
        """
        Calculate actual route distance by summing GPS distances between consecutive stations.
        This gives accurate distance instead of time-based estimates.
        
        SPECIAL HANDLING FOR FERRIES:
        - Ferries are point-to-point, not linear routes
        - Use direct Haversine distance instead of summing through intermediate stations
        
        Returns:
            Distance in kilometers
        """
        # Special case for ferries: use direct distance
        if start_station.line.upper() == "FERRY":
            return self._haversine_distance(
                start_station.lat, start_station.lon,
                end_station.lat, end_station.lon
            )
        
        if start_idx is None or end_idx is None:
            return 0.0
        
        # Ensure start_idx < end_idx
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        total_distance = 0.0
        
        # Sum up distances between consecutive stations (for linear routes)
        for i in range(start_idx, end_idx):
            station1_id = line_stations[i]
            station2_id = line_stations[i + 1]
            
            station1 = self.stations.get(station1_id)
            station2 = self.stations.get(station2_id)
            
            if station1 and station2:
                # Use Haversine formula to calculate distance
                segment_distance = self._haversine_distance(
                    station1.lat, station1.lon,
                    station2.lat, station2.lon
                )
                total_distance += segment_distance
        
        return total_distance
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth radius in km
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = math.sin(delta_lat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * \
            math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
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
    
    def _check_deprecated_stations(self, origin: str, destination: str) -> Optional[str]:
        """
        Check if origin or destination is a deprecated station.
        
        Returns deprecation message if deprecated, None otherwise.
        """
        deprecated_stations = {
            "atatürk havalimanı": "Atatürk Airport Metro station is closed. The airport closed in 2019. Use Istanbul Airport (M11) instead.",
            "ataturk havalimani": "Atatürk Airport Metro station is closed. The airport closed in 2019. Use Istanbul Airport (M11) instead.",
            "atatürk airport": "Atatürk Airport Metro station is closed. The airport closed in 2019. Use Istanbul Airport (M11) instead.",
            "ataturk airport": "Atatürk Airport Metro station is closed. The airport closed in 2019. Use Istanbul Airport (M11) instead.",
        }
        
        origin_lower = origin.lower().strip()
        dest_lower = destination.lower().strip()
        
        for station, message in deprecated_stations.items():
            if station in origin_lower or station in dest_lower:
                return message
        
        return None
    
    def _create_deprecation_route(self, origin: str, destination: str, message: str) -> TransitRoute:
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
    
    def _rank_routes(self, routes: List[TransitRoute], origin_gps: Optional[Dict[str, float]] = None) -> List[TransitRoute]:
        """
        Rank routes by multiple criteria.
        
        Ranking factors:
        1. Total travel time (primary)
        2. Number of transfers
        3. Distance
        
        Returns routes sorted by ranking score (best first).
        """
        if not routes:
            return []
        
        # Calculate scores for each route
        for route in routes:
            # Lower is better for all factors
            time_score = route.total_time
            transfer_penalty = route.transfers * 10  # 10 min penalty per transfer
            
            # Combined score (lower is better)
            route.ranking_scores = {
                'time': time_score,
                'transfers': route.transfers,
                'combined': time_score + transfer_penalty
            }
        
        # Sort by combined score
        ranked = sorted(routes, key=lambda r: r.ranking_scores.get('combined', float('inf')))
        
        return ranked
    
    def _route_to_dict(self, route: TransitRoute) -> Dict[str, Any]:
        """Convert TransitRoute to dictionary for caching."""
        return {
            'origin': route.origin,
            'destination': route.destination,
            'total_time': route.total_time,
            'total_distance': route.total_distance,
            'steps': route.steps,
            'transfers': route.transfers,
            'lines_used': route.lines_used,
            'alternatives': [self._route_to_dict(alt) for alt in route.alternatives] if route.alternatives else [],
            'time_confidence': route.time_confidence,
            'ranking_scores': route.ranking_scores
        }
    
    def _dict_to_route(self, data: Dict[str, Any]) -> TransitRoute:
        """Convert dictionary to TransitRoute (from cache)."""
        alternatives = []
        if data.get('alternatives'):
            alternatives = [self._dict_to_route(alt) for alt in data['alternatives']]
        
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
    
    def find_nearest_station(self, lat: float, lon: float, max_distance_km: float = 2.0) -> Optional[str]:
        """
        Find the nearest transit station to given GPS coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_distance_km: Maximum distance to search (default 2km)
            
        Returns:
            Station ID of nearest station, or None if none within max_distance
        """
        nearest_station = None
        nearest_distance = float('inf')
        
        for station_id, station in self.stations.items():
            distance = self._haversine_distance(lat, lon, station.lat, station.lon)
            if distance < nearest_distance and distance <= max_distance_km:
                nearest_distance = distance
                nearest_station = station_id
        
        if nearest_station:
            logger.info(f"📍 Nearest station to ({lat}, {lon}): {self.stations[nearest_station].name} ({nearest_distance:.2f} km)")
        else:
            logger.warning(f"📍 No station found within {max_distance_km} km of ({lat}, {lon})")
        
        return nearest_station
    
    def _get_generic_transport_info(self) -> str:
        """Return generic transportation information when no route can be extracted."""
        return """🚇 **Istanbul Public Transportation**

I couldn't identify specific locations from your query. To help you with directions, please specify:
- **Origin**: Where are you starting from?
- **Destination**: Where do you want to go?

**Example queries:**
- "How to go from Kadıköy to Taksim?"
- "Route from Sultanahmet to the airport"
- "Directions from Beşiktaş to Üsküdar"

**Istanbul Transit Network:**
- 🚇 **Metro**: M1, M2, M3, M4, M5, M6, M7, M9, M11
- 🚋 **Tram**: T1, T4, T5
- 🚃 **Marmaray**: Cross-Bosphorus rail
- 🚡 **Funicular**: F1 (Kabataş-Taksim), F2 (Karaköy-Beyoğlu)
- ⛴️ **Ferry**: Kadıköy, Eminönü, Karaköy, Beşiktaş, Üsküdar

**Tips:**
- Use Marmaray or ferries to cross between European and Asian sides
- İstanbulkart works on all public transport"""
    
    def get_map_data_for_last_route(self) -> Optional[Dict[str, Any]]:
        """
        Get map visualization data for the last computed route.
        
        Returns polyline data, station markers, and route info for frontend map display.
        """
        if not self.last_route:
            logger.warning("No last_route available for map data")
            return None
        
        route = self.last_route
        
        # Build polyline from route steps
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
        
        # Add step markers and polyline
        for step in route.steps:
            if step.get('type') in ['transit', 'ferry']:
                # Get from station
                from_name = step.get('from', '')
                to_name = step.get('to', '')
                line = step.get('line', '')
                
                # Find station coordinates
                from_stations = self._get_stations_for_location(from_name.lower())
                to_stations = self._get_stations_for_location(to_name.lower())
                
                if from_stations:
                    from_station = self.stations.get(from_stations[0])
                    if from_station:
                        polyline_points.append([from_station.lat, from_station.lon])
                
                if to_stations:
                    to_station = self.stations.get(to_stations[0])
                    if to_station:
                        polyline_points.append([to_station.lat, to_station.lon])
                        markers.append({
                            'type': 'transfer' if step.get('type') == 'transfer' else 'stop',
                            'name': to_name,
                            'lat': to_station.lat,
                            'lon': to_station.lon,
                            'line': line
                        })
        
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
        """Calculate map bounds from a list of lat/lon points."""
        if not points:
            # Default to Istanbul center
            return {
                'min_lat': 40.9, 'max_lat': 41.2,
                'min_lon': 28.8, 'max_lon': 29.2
            }
        
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }
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
        """Format directions in English - simple and readable"""
        lines = [
            f"🚇 **{route.origin} → {route.destination}**",
            f"⏱️ {route.total_time} min | 🔄 {route.transfers} transfer(s)",
            ""
        ]
        
        # Simple step format: "1. Station A → Station B (Line, 10 min)"
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
    
    def _format_directions_turkish(self, route: TransitRoute) -> str:
        """Format directions in Turkish - simple and readable"""
        lines = [
            f"🚇 **{route.origin} → {route.destination}**",
            f"⏱️ {route.total_time} dk | 🔄 {route.transfers} aktarma",
            ""
        ]
        
        # Simple step format: "1. İstasyon A → İstasyon B (Hat, 10 dk)"
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
    
    def get_rag_context_for_query(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate RAG context for transportation query.
        
        This is what gets injected into the LLM prompt as "verified knowledge".
        
        NOTE: We return MINIMAL context here since:
        1. Detailed step-by-step directions are in route_data (sent to frontend)
        2. LLM should give a brief intro, not repeat all the steps
        3. Full directions appear in the RouteCard UI component
        """
        # 🔥 IMPROVED FIX: Only clear last_route if this is a NEW query
        # Check if this is the same query as last time (to prevent clearing during re-extraction)
        if not hasattr(self, '_last_query') or self._last_query != query:
            logger.info(f"🆕 NEW QUERY: Clearing last_route for: '{query}'")
            self.last_route = None
            self._last_query = query
        else:
            logger.info(f"🔁 REPEAT QUERY: Keeping last_route for: '{query}'")
        
        query_lower = query.lower().strip()
        
        logger.info(f"🔍 TRANSPORTATION QUERY: '{query}'")
        logger.info(f"📍 User location available: {user_location is not None}")
        
        # Extract origin and destination from query
        origin, destination = self._extract_locations_from_query(query_lower, user_location)
        
        logger.info(f"🎯 PATTERN EXTRACTION RESULT: origin='{origin}', destination='{destination}'")
        
        if not origin or not destination:
            # Try LLM fallback before giving up
            logger.info(f"🤖 Pattern extraction incomplete, trying LLM fallback...")
            llm_origin, llm_dest = extract_locations_with_llm_sync(query)
            if llm_origin and llm_dest:
                logger.info(f"✅ LLM FALLBACK SUCCESS: origin='{llm_origin}', destination='{llm_dest}'")
                origin = llm_origin
                destination = llm_dest
            else:
                # Generic transportation info
                logger.warning(f"⚠️ Could not extract origin/destination from query: '{query}'")
                return self._get_generic_transport_info()
        
        # Prepare GPS data if origin or destination is GPS-based
        # Normalize GPS data format ('latitude'/'longitude' → 'lat'/'lon')
        origin_gps = None
        destination_gps = None
        
        if origin == "Your Location" and user_location:
            # Normalize GPS format
            if 'latitude' in user_location and 'longitude' in user_location:
                origin_gps = {
                    'lat': user_location['latitude'],
                    'lon': user_location['longitude']
                }
            elif 'lat' in user_location and 'lon' in user_location:
                origin_gps = user_location
            logger.info(f"📍 Normalized origin GPS: {origin_gps}")
                
        if destination == "Your Location" and user_location:
            # Normalize GPS format
            if 'latitude' in user_location and 'longitude' in user_location:
                destination_gps = {
                    'lat': user_location['latitude'],
                    'lon': user_location['longitude']
                }
            elif 'lat' in user_location and 'lon' in user_location:
                destination_gps = user_location
            logger.info(f"📍 Normalized destination GPS: {destination_gps}")
        
        # Find route
        route = self.find_route(origin, destination, origin_gps=origin_gps, destination_gps=destination_gps)
        
        if not route:
            self.last_route = None  # Clear last route
            return f"❌ No direct route found between {origin} and {destination}. Please verify station names."
        
        # Store route for mapData extraction
        self.last_route = route
        
        # Return MINIMAL context - just the high-level summary
        # The detailed step-by-step is in route_data (for UI display)
        context_lines = [
            f"🚇 **Route Found: {route.origin} → {route.destination}**",
            f"⏱️ Duration: {route.total_time} minutes",
            f"🔄 Transfers: {route.transfers}",
            f"🚉 Lines: {', '.join(route.lines_used)}",
            "",
            "💡 NOTE: Step-by-step directions with map are shown in the route card below your response.",
            "Your job: Give a brief, friendly introduction (1-2 sentences). Don't repeat all the steps."
        ]
        
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
            # Try LLM fallback for location extraction
            logger.info(f"🤖 Attempting LLM fallback for location extraction...")
            llm_origin, llm_dest = extract_locations_with_llm_sync(query)
            if llm_origin and llm_dest:
                logger.info(f"✅ LLM FALLBACK SUCCESS: origin='{llm_origin}', destination='{llm_dest}'")
                return llm_origin, llm_dest
            logger.warning(f"❌ LLM fallback also failed for query: '{query}'")
            return None, None
        
        # Strategy: Use keyword context to determine roles
        origin = None
        destination = None
        
        # PRIORITY 1: Look for explicit "from X to Y" pattern first
        # This handles: "how to go from kadikoy to taksim", "from A to B", "starting from X to Y"
        from_to_pattern = r'\bfrom\s+(\w+(?:\s+\w+)?)\s+to\s+(\w+(?:\s+\w+)?)\b'
        from_to_match = re.search(from_to_pattern, query_lower, re.IGNORECASE)
        if from_to_match:
            origin_candidate = from_to_match.group(1).strip()
            dest_candidate = from_to_match.group(2).strip()
            logger.info(f"📍 PATTERN MATCH: 'from X to Y' found: '{origin_candidate}' → '{dest_candidate}'")
            
            # Verify these are valid locations
            origin_match = None
            dest_match = None
            for loc in filtered_locations:
                if origin_candidate in loc['name'] or loc['name'] in origin_candidate:
                    origin_match = loc['name']
                if dest_candidate in loc['name'] or loc['name'] in dest_candidate:
                    dest_match = loc['name']
            
            if origin_match and dest_match:
                origin = origin_match
                destination = dest_match
                logger.info(f"✅ PATTERN SUCCESS: origin='{origin}', destination='{destination}'")
        
        # If pattern didn't match, fall back to keyword analysis
        if not origin or not destination:
            # Look for "from X" pattern (but NOT when followed immediately by "to" - that's handled above)
            from_keywords = ['starting from', 'leaving from', 'departing from', 'beginning from', 'from']
            for keyword in from_keywords:
                # Use word boundary matching
                pattern = r'\b' + re.escape(keyword) + r'\s+'
                match = re.search(pattern, query_lower)
                if match:
                    keyword_end = match.end()
                    # Find location closest AFTER this keyword
                    for loc in filtered_locations:
                        if loc['position'] >= keyword_end - 1:  # Allow 1 char tolerance
                            origin = loc['name']
                            logger.info(f"📍 FROM keyword '{keyword}' → origin='{origin}'")
                            break
                    if origin:
                        break
            
            # Look for "to Y" pattern - but EXCLUDE "how to", "want to", "need to", etc.
            # These are infinitive constructs, not directional
            to_keywords = ['going to', 'heading to', 'arriving at', 'toward', 'towards', 'get to', 'travel to']
            for keyword in to_keywords:
                pattern = r'\b' + re.escape(keyword) + r'\s+'
                match = re.search(pattern, query_lower)
                if match:
                    keyword_end = match.end()
                    # Find location closest AFTER this keyword
                    for loc in filtered_locations:
                        if loc['position'] >= keyword_end - 1:
                            destination = loc['name']
                            logger.info(f"📍 TO keyword '{keyword}' → destination='{destination}'")
                            break
                    if destination:
                        break
            
            # Only use bare "to" if it's NOT part of infinitive constructs
            if not destination:
                # Find all occurrences of " to " (with spaces) that are NOT after how/want/need/etc
                infinitive_pattern = r'\b(how|want|need|try|going|have|has|had|would|will|can|could|should|must)\s+to\b'
                infinitive_matches = list(re.finditer(infinitive_pattern, query_lower))
                infinitive_positions = [m.start() for m in infinitive_matches]
                
                # Find all " to " positions
                to_pattern = r'\bto\s+'
                for match in re.finditer(to_pattern, query_lower):
                    to_pos = match.start()
                    # Skip if this "to" is part of an infinitive
                    is_infinitive = False
                    for inf_pos in infinitive_positions:
                        if inf_pos <= to_pos <= inf_pos + 15:  # Within 15 chars of infinitive start
                            is_infinitive = True
                            break
                    
                    if not is_infinitive:
                        keyword_end = match.end()
                        for loc in filtered_locations:
                            if loc['position'] >= keyword_end - 1:
                                destination = loc['name']
                                logger.info(f"📍 Bare 'to' at pos {to_pos} → destination='{destination}'")
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
        try:
            from config.settings import settings
        except ImportError:
            from backend.config.settings import settings
        
        # Initialize Redis client for route caching
        redis_client = None
        
        if redis is not None:
            redis_url = os.getenv('REDIS_URL') or (settings.REDIS_URL if hasattr(settings, 'REDIS_URL') else None)
            
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
        else:
            logger.info("ℹ️ Transportation RAG: Redis package not installed, caching disabled")
        
        # Create singleton with Redis
        _transportation_rag_singleton = IstanbulTransportationRAG(redis_client=redis_client)
        logger.info("✅ Transportation RAG singleton initialized")
    
    return _transportation_rag_singleton


# LLM client for fallback location extraction
_llm_client = None


def get_llm_client_for_extraction():
    """Get or create LLM client for location extraction fallback."""
    global _llm_client
    if _llm_client is None:
        try:
            from services.runpod_llm_client import get_llm_client
            _llm_client = get_llm_client()
            if _llm_client:
                logger.info("✅ LLM client initialized for location extraction fallback")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize LLM client for extraction: {e}")
    return _llm_client


async def extract_locations_with_llm(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Use LLM to extract origin and destination from a query when pattern matching fails.
    
    This is a fallback method that calls the RunPod LLM to intelligently
    extract location information from complex or ambiguous queries.
    
    Args:
        query: The user's transportation query
        
    Returns:
        Tuple of (origin, destination) or (None, None) if extraction fails
    """
    client = get_llm_client_for_extraction()
    if not client:
        logger.warning("⚠️ LLM client not available for fallback extraction")
        return None, None
    
    try:
        # Create a focused prompt for location extraction
        extraction_prompt = f"""Extract the origin and destination locations from this Istanbul transportation query.

Query: "{query}"

IMPORTANT INSTRUCTIONS:
1. Identify the ORIGIN (starting point) and DESTINATION (ending point)
2. Return ONLY location names, no explanations
3. If a location is unclear, use the most likely Istanbul location
4. Common Istanbul locations: Taksim, Kadıköy, Sultanahmet, Üsküdar, Beşiktaş, Eminönü, Galata, Karaköy, Şişli, Mecidiyeköy, etc.

Respond in this EXACT format:
ORIGIN: [origin location]
DESTINATION: [destination location]

If you cannot determine one, write "UNKNOWN" for that field."""

        # Call the LLM
        response = await client.generate(
            prompt=extraction_prompt,
            max_tokens=100,
            temperature=0.1  # Low temperature for deterministic extraction
        )
        
        if not response or not response.get('text'):
            logger.warning("⚠️ LLM returned empty response for location extraction")
            return None, None
        
        response_text = response['text'].strip()
        logger.info(f"🤖 LLM extraction response: {response_text}")
        
        # Parse the response
        origin = None
        destination = None
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line.upper().startswith('ORIGIN:'):
                origin = line.split(':', 1)[1].strip()
                if origin.upper() == 'UNKNOWN':
                    origin = None
            elif line.upper().startswith('DESTINATION:'):
                destination = line.split(':', 1)[1].strip()
                if destination.upper() == 'UNKNOWN':
                    destination = None
        
        logger.info(f"🤖 LLM extracted: origin='{origin}', destination='{destination}'")
        return origin, destination
        
    except Exception as e:
        logger.error(f"❌ LLM location extraction failed: {e}")
        return None, None


def extract_locations_with_llm_sync(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Synchronous wrapper for LLM location extraction.
    
    This allows calling the async LLM extraction from synchronous code.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new task in the running loop
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, extract_locations_with_llm(query))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(extract_locations_with_llm(query))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(extract_locations_with_llm(query))
    except Exception as e:
        logger.error(f"❌ Sync LLM extraction wrapper failed: {e}")
        return None, None
