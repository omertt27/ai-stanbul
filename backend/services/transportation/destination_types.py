"""
Destination Type System
=======================

Destination classification for proper routing:
- STATION: Direct transit station
- AREA: Neighborhood/district
- ISLAND: Ferry-only destination (Princes' Islands)
- ATTRACTION: Tourist attraction
- FERRY_TERMINAL: Ferry pier
- WALKING: Destination within walking distance

Author: AI Istanbul Team
Date: December 2024
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class DestinationType(Enum):
    """Destination classification for proper routing."""
    STATION = "station"
    AREA = "area"
    ISLAND = "island"
    ATTRACTION = "attraction"
    FERRY_TERMINAL = "ferry_terminal"
    WALKING = "walking"


@dataclass
class DestinationInfo:
    """Information about a destination for routing."""
    name: str
    dest_type: DestinationType
    access_mode: str  # 'rail', 'ferry', 'walk', 'multi'
    terminals: List[str]  # Access points (stations/piers)
    walking_time_min: Optional[int] = None


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
    """Classify a destination before routing."""
    dest_normalized = destination.lower().strip()
    
    if dest_normalized in ISLAND_DESTINATIONS:
        return ISLAND_DESTINATIONS[dest_normalized]
    
    island_patterns = ['ada', 'island', 'adası']
    for pattern in island_patterns:
        if pattern in dest_normalized:
            return DestinationInfo(
                name=destination,
                dest_type=DestinationType.ISLAND,
                access_mode="ferry",
                terminals=["FERRY-Kabataş", "FERRY-Eminönü", "FERRY-Kadıköy", "FERRY-Bostancı"]
            )
    
    return DestinationInfo(
        name=destination,
        dest_type=DestinationType.AREA,
        access_mode="rail",
        terminals=[]
    )


def is_walking_distance(origin_coords: Tuple[float, float], dest_coords: Tuple[float, float]) -> Tuple[bool, int]:
    """
    Check if destination is within walking distance.
    
    Returns:
        Tuple of (is_walkable, estimated_walk_time_minutes)
    """
    lat1, lon1 = origin_coords
    lat2, lon2 = dest_coords
    
    # Haversine formula
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    meters = 6371000 * c
    
    walk_time_minutes = int(meters / 80)
    
    return (meters <= WALK_THRESHOLD_METERS, walk_time_minutes)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * \
        math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


# Valid transport entities (not hallucinations)
VALID_TRANSPORT_ENTITIES = {
    # Metro lines
    "m1", "m1a", "m1b", "m2", "m3", "m4", "m5", "m6", "m7", "m9", "m11",
    # Tram lines  
    "t1", "t4", "t5",
    # Other transit
    "marmaray", "f1", "f2", "metrobus", "ferry", "ido", "turyol", "şehir hatları",
    # Common stations
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
    # Airports
    "istanbul havalimanı", "istanbul havalimani", "atatürk havalimanı",
    "sabiha gökçen", "sabiha gokcen",
}


def is_valid_transport_entity(entity: str) -> bool:
    """Check if an entity is a valid transport-related term."""
    import re
    entity_lower = entity.lower().strip()
    
    if entity_lower in VALID_TRANSPORT_ENTITIES:
        return True
    
    for valid_entity in VALID_TRANSPORT_ENTITIES:
        if valid_entity in entity_lower or entity_lower in valid_entity:
            return True
    
    if re.match(r'^m\d{1,2}$', entity_lower):
        return True
    if re.match(r'^t\d$', entity_lower):
        return True
    if re.match(r'^f\d$', entity_lower):
        return True
        
    return False
