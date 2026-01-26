#!/usr/bin/env python3
"""
Transportation RAG System - Backward Compatibility Wrapper
==========================================================

This file provides backward compatibility for code that imports from
`services.transportation_rag_system`.

The actual implementation has been refactored into the
`services.transportation` package with smaller, more maintainable modules:

- transportation/nlp_utils.py - Text processing, Turkish morphology
- transportation/destination_types.py - Island, walking detection
- transportation/station_aliases.py - Location name mappings
- transportation/route_builder.py - Route creation and formatting
- transportation/pathfinding.py - Dijkstra routing algorithm
- transportation/location_extraction.py - Query parsing
- transportation/rag_system.py - Main orchestration class

Author: AI Istanbul Team
Date: December 2024
"""

import logging

logger = logging.getLogger(__name__)

# Re-export everything from the new modular package
try:
    from services.transportation import (
        IstanbulTransportationRAG,
        get_transportation_rag,
        TransitRoute,
        TransitStation,
    )
    from services.transportation.nlp_utils import (
        TurkishMorphologyHandler,
        levenshtein_distance,
        normalized_levenshtein_similarity,
        normalize_unicode_text,
        transliterate_cyrillic_to_latin,
        remove_turkish_diacritics,
        TOURIST_DEFAULT_ORIGINS,
        AIRPORT_DESTINATIONS,
        get_time_based_suggestion,
    )
    from services.transportation.destination_types import (
        DestinationType,
        DestinationInfo,
        ISLAND_DESTINATIONS,
        WALK_THRESHOLD_METERS,
        get_destination_type,
        is_walking_distance,
        haversine_distance,
        VALID_TRANSPORT_ENTITIES,
        is_valid_transport_entity,
    )
    from services.transportation.station_aliases import (
        build_station_aliases,
        build_neighborhood_stations,
        build_route_patterns,
    )
    from services.transportation.location_extraction import (
        LocationExtractor,
        extract_locations_with_llm,
        extract_locations_with_llm_sync,
    )
    from services.transportation.route_builder import RouteBuilder
    from services.transportation.pathfinding import Pathfinder
    
    logger.info("✅ Transportation RAG loaded from modular package")
    
except ImportError as e:
    logger.warning(f"⚠️ Failed to import from modular package: {e}")
    logger.info("📦 Falling back to legacy monolithic implementation...")
    
    # Fallback: Import from the legacy file if it still exists
    # This would be the old 3455-line file renamed to _legacy.py
    try:
        from services.transportation_rag_system_legacy import *
        logger.info("✅ Transportation RAG loaded from legacy module")
    except ImportError:
        logger.error("❌ Could not load transportation RAG from any source")
        raise


# Ensure all public symbols are exported
__all__ = [
    # Main class and factory
    'IstanbulTransportationRAG',
    'get_transportation_rag',
    
    # Data classes
    'TransitRoute',
    'TransitStation',
    'DestinationType',
    'DestinationInfo',
    
    # NLP utilities
    'TurkishMorphologyHandler',
    'levenshtein_distance',
    'normalized_levenshtein_similarity',
    
    # Destination utilities
    'get_destination_type',
    'is_walking_distance',
    'is_valid_transport_entity',
    
    # Location extraction
    'extract_locations_with_llm_sync',
    
    # Constants
    'ISLAND_DESTINATIONS',
    'VALID_TRANSPORT_ENTITIES',
]
