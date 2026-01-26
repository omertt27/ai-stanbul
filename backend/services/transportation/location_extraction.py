"""
Location Extraction from Queries
================================

Query parsing and location extraction:
- Pattern-free location recognition
- Multilingual keyword detection (6 languages)
- Fuzzy matching with Turkish morphology
- GPS fallback for "my location" queries
- LLM fallback for complex queries

Author: AI Istanbul Team
Date: December 2024
"""

import re
import logging
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any

from .nlp_utils import (
    TurkishMorphologyHandler,
    normalized_levenshtein_similarity,
    transliterate_cyrillic_to_latin,
    remove_turkish_diacritics,
)

logger = logging.getLogger(__name__)


class LocationExtractor:
    """Extract origin and destination from transportation queries."""
    
    def __init__(self, stations: Dict, neighborhoods: Dict, aliases: Dict):
        """
        Initialize with location databases.
        
        Args:
            stations: Station database {station_id: TransitStation}
            neighborhoods: Neighborhood to stations mapping
            aliases: Station aliases mapping
        """
        self.stations = stations
        self.neighborhoods = neighborhoods
        self.aliases = aliases
        self._build_known_locations()
    
    def _build_known_locations(self):
        """Build comprehensive location database for matching."""
        self.known_locations = {}
        
        # Add all stations
        for station_id, station in self.stations.items():
            name = station.name.lower()
            self.known_locations[name] = name
        
        # Add all neighborhoods
        for neighborhood in self.neighborhoods.keys():
            self.known_locations[neighborhood.lower()] = neighborhood.lower()
        
        # Add all aliases
        for alias in self.aliases.keys():
            self.known_locations[alias.lower()] = alias.lower()
        
        logger.debug(f"📊 Built location database: {len(self.known_locations)} entries")
    
    def extract_locations(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract origin and destination using pattern-free location recognition.
        
        Strategy:
        1. Find ALL known locations in the query
        2. Use contextual clues (from/to keywords) to assign roles
        3. If only 1 location found and user has GPS, use GPS as origin
        4. Fallback: assume first location = origin, last = destination
        """
        query_lower = query.lower()
        
        logger.info(f"🔍 LOCATION EXTRACTION: Query='{query}'")
        logger.info(f"📍 GPS available: {user_location is not None}")
        
        # Find all locations mentioned in query
        found_locations = self._find_locations_in_query(query_lower)
        
        # If no exact matches found, try fuzzy matching
        if len(found_locations) < 2:
            logger.info(f"🔍 Trying fuzzy matching (found {len(found_locations)} exact matches)")
            found_locations = self._fuzzy_find_locations(query_lower, found_locations)
        
        logger.info(f"🔎 Found {len(found_locations)} potential locations in query")
        
        # Remove overlapping matches
        filtered_locations = self._remove_overlapping_matches(found_locations)
        
        logger.info(f"✅ After filtering overlaps: {len(filtered_locations)} locations")
        for loc in filtered_locations:
            logger.info(f"   - '{loc['name']}' at position {loc['position']}")
        
        # Sort by position in query
        filtered_locations.sort(key=lambda x: x['position'])
        
        # Handle single location with GPS
        if len(filtered_locations) == 1 and user_location:
            logger.info(f"✅ SINGLE LOCATION + GPS: Using GPS as origin")
            destination = filtered_locations[0]['name']
            return "Your Location", destination
        
        # Handle no locations with GPS
        if len(filtered_locations) == 0 and user_location:
            logger.warning(f"❌ NO LOCATIONS FOUND but GPS available")
            to_pattern = r'(?:to|towards?)\s+([a-zA-Z\s]+?)(?:\s+\?|$|\s+please|\s+from)'
            match = re.search(to_pattern, query_lower)
            if match:
                potential_dest = match.group(1).strip()
                logger.info(f"📍 Potential destination from pattern: '{potential_dest}'")
                return "Your Location", potential_dest
            return None, None
        
        if len(filtered_locations) < 2:
            logger.warning(f"❌ INSUFFICIENT LOCATIONS: Found {len(filtered_locations)}")
            return None, None
        
        # Use keyword context to determine roles
        origin, destination = self._assign_roles_by_keywords(query_lower, filtered_locations)
        
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
    
    def _find_locations_in_query(self, query_lower: str) -> List[Dict[str, Any]]:
        """Find all exact location matches in query."""
        found_locations = []
        
        for location_name in sorted(self.known_locations.keys(), key=len, reverse=True):
            if location_name in query_lower and location_name not in [loc['name'] for loc in found_locations]:
                pos = query_lower.find(location_name)
                found_locations.append({
                    'name': self.known_locations[location_name],
                    'position': pos,
                    'length': len(location_name)
                })
        
        return found_locations
    
    def _fuzzy_find_locations(self, query_lower: str, existing: List[Dict]) -> List[Dict[str, Any]]:
        """Use fuzzy matching to find additional locations."""
        words = re.findall(r'[\w\u0400-\u04FF\u0600-\u06FF]+', query_lower)
        
        for i, word in enumerate(words):
            if len(word) >= 3:
                matched = self._fuzzy_match_location(word)
                if matched and not any(loc['name'] == matched for loc in existing):
                    pos = query_lower.find(word)
                    existing.append({
                        'name': matched,
                        'position': pos if pos >= 0 else i * 10,
                        'length': len(word),
                        'fuzzy': True
                    })
                    logger.info(f"   🔤 Fuzzy match: '{word}' → '{matched}'")
        
        return existing
    
    def _fuzzy_match_location(self, query_word: str) -> Optional[str]:
        """Fuzzy match a word against known locations."""
        if not query_word or len(query_word) < 3:
            return None
        
        query_lower = query_word.lower()
        
        # 1. Exact match
        if query_lower in self.known_locations:
            return self.known_locations[query_lower]
        
        # 2. Turkish suffix stripped
        stripped = TurkishMorphologyHandler.strip_suffixes(query_lower)
        if stripped != query_lower and stripped in self.known_locations:
            return self.known_locations[stripped]
        
        # 3. Turkish diacritics removed
        ascii_version = remove_turkish_diacritics(query_lower)
        for loc_name, canonical in self.known_locations.items():
            loc_ascii = remove_turkish_diacritics(loc_name)
            if ascii_version == loc_ascii:
                return canonical
        
        # 4. Cyrillic transliteration
        if any('\u0400' <= c <= '\u04FF' for c in query_word):
            transliterated = transliterate_cyrillic_to_latin(query_lower)
            if transliterated in self.known_locations:
                return self.known_locations[transliterated]
            for loc_name, canonical in self.known_locations.items():
                if normalized_levenshtein_similarity(transliterated, loc_name) > 0.85:
                    return canonical
        
        # 5. Partial match
        for loc_name, canonical in self.known_locations.items():
            if len(loc_name) >= 4 and query_lower.startswith(loc_name):
                return canonical
            if len(query_lower) >= 4 and loc_name.startswith(query_lower):
                return canonical
        
        # 6. Levenshtein similarity
        if len(query_lower) >= 4:
            best_match = None
            best_score = 0.0
            
            for loc_name, canonical in self.known_locations.items():
                if len(loc_name) >= 4:
                    score = normalized_levenshtein_similarity(query_lower, loc_name)
                    stripped_score = normalized_levenshtein_similarity(stripped, loc_name)
                    score = max(score, stripped_score)
                    
                    if score > 0.75 and score > best_score:
                        best_score = score
                        best_match = canonical
            
            if best_match:
                return best_match
        
        return None
    
    def _remove_overlapping_matches(self, found_locations: List[Dict]) -> List[Dict]:
        """Remove overlapping matches, keeping longer ones."""
        filtered = []
        for loc in found_locations:
            overlap = False
            for other in found_locations:
                if loc != other:
                    if (loc['position'] >= other['position'] and 
                        loc['position'] < other['position'] + other['length']):
                        if loc['length'] < other['length']:
                            overlap = True
                            break
            if not overlap:
                filtered.append(loc)
        return filtered
    
    def _assign_roles_by_keywords(
        self, 
        query_lower: str, 
        filtered_locations: List[Dict]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Use keyword context to determine origin/destination roles."""
        origin = None
        destination = None
        
        # Priority 1: Explicit "from X to Y" patterns (multilingual)
        from_to_patterns = [
            r'\bfrom\s+(\w+(?:\s+\w+)?)\s+to\s+(\w+(?:\s+\w+)?)\b',
            r"(\w+)['']?(?:den|dan)\s+(\w+)['']?(?:e|a|ye|ya)\b",
            r'(?:из|от)\s+(\w+)\s+(?:в|до|на)\s+(\w+)',
            r'(?:von)\s+(\w+)\s+(?:nach|zu|zum|zur)\s+(\w+)',
            r'(?:de|du)\s+(\w+)\s+(?:à|au|aux|vers)\s+(\w+)',
            r'من\s+(\w+)\s+(?:إلى|الى|ل)\s+(\w+)',
        ]
        
        for pattern in from_to_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE | re.UNICODE)
            if match:
                origin_candidate = match.group(1).strip()
                dest_candidate = match.group(2).strip()
                
                origin_match = None
                dest_match = None
                for loc in filtered_locations:
                    if origin_candidate in loc['name'] or loc['name'] in origin_candidate:
                        origin_match = loc['name']
                    if dest_candidate in loc['name'] or loc['name'] in dest_candidate:
                        dest_match = loc['name']
                
                if origin_match and dest_match:
                    return origin_match, dest_match
        
        # Look for "from X" keywords (multilingual)
        from_keywords = [
            'starting from', 'leaving from', 'departing from', 'from',
            'den', 'dan', 'ndan', 'nden', "'den", "'dan",
            'из', 'от', 'с ',
            'von', 'ab',
            'من', 'انطلاقاً من',
            'de', 'depuis', 'partant de',
        ]
        
        for keyword in from_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\s+'
            match = re.search(pattern, query_lower)
            if match:
                keyword_end = match.end()
                for loc in filtered_locations:
                    if loc['position'] >= keyword_end - 1:
                        origin = loc['name']
                        break
                if origin:
                    break
        
        # Look for "to Y" keywords (multilingual)
        to_keywords = [
            'going to', 'heading to', 'arriving at', 'toward', 'towards', 'get to',
            'e gitmek', 'a gitmek', 'ye gitmek', 'ya gitmek',
            'в ', 'до', 'к ', 'на ',
            'nach', 'zu', 'zum', 'zur',
            'إلى', 'الى', 'نحو',
            'à', 'vers', 'pour aller à',
        ]
        
        for keyword in to_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\s+'
            match = re.search(pattern, query_lower)
            if match:
                keyword_end = match.end()
                for loc in filtered_locations:
                    if loc['position'] >= keyword_end - 1:
                        destination = loc['name']
                        break
                if destination:
                    break
        
        return origin, destination


# LLM fallback for location extraction
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
    Use LLM to extract origin and destination when pattern matching fails.
    Supports 6 languages: English, Turkish, Russian, German, Arabic, French
    """
    client = get_llm_client_for_extraction()
    if not client:
        return None, None
    
    try:
        extraction_prompt = f"""Extract the origin and destination locations from this Istanbul transportation query.
The query may be in English, Turkish, Russian, German, Arabic, or French.

Query: "{query}"

IMPORTANT INSTRUCTIONS:
1. Identify the ORIGIN (starting point)
2. Identify the DESTINATION (ending point)
3. Return ONLY location names, no explanations
4. Common Istanbul locations: Taksim, Kadıköy, Sultanahmet, Üsküdar, Beşiktaş, Eminönü, etc.

Respond in this EXACT format:
ORIGIN: [origin location]
DESTINATION: [destination location]

If you cannot determine one, write "UNKNOWN" for that field."""

        response = await client.generate(
            prompt=extraction_prompt,
            max_tokens=100,
            temperature=0.1
        )
        
        if not response or not response.get('text'):
            return None, None
        
        response_text = response['text'].strip()
        logger.info(f"🤖 LLM extraction response: {response_text}")
        
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
        
        return origin, destination
        
    except Exception as e:
        logger.error(f"❌ LLM location extraction failed: {e}")
        return None, None


def extract_locations_with_llm_sync(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Synchronous wrapper for LLM location extraction."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, extract_locations_with_llm(query))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(extract_locations_with_llm(query))
    except RuntimeError:
        return asyncio.run(extract_locations_with_llm(query))
    except Exception as e:
        logger.error(f"❌ Sync LLM extraction wrapper failed: {e}")
        return None, None
