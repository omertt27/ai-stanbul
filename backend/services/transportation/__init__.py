"""
Istanbul Transportation RAG System
==================================

Modular transportation routing system with Google Maps-quality features.

Modules:
- nlp_utils: Text processing, fuzzy matching, Turkish morphology
- destination_types: Island, walking, ferry destination handling
- station_aliases: Location name mappings (6 languages)
- route_builder: Route creation and formatting
- pathfinding: Dijkstra routing algorithm
- location_extraction: Query parsing and LLM fallback
- rag_system: Main orchestration class

Author: AI Istanbul Team
Date: December 2024
"""

from .rag_system import (
    IstanbulTransportationRAG,
    get_transportation_rag,
    TransitRoute,
    TransitStation,
)

__all__ = [
    'IstanbulTransportationRAG',
    'get_transportation_rag',
    'TransitRoute',
    'TransitStation',
]
