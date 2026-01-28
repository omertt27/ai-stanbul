"""
Location Extraction from Queries
================================

Query parsing and location extraction:
- Pattern-free location recognition
- Multilingual keyword detection (6 languages)
- Fuzzy matching with Turkish morphology
- GPS fallback for "my location" queries
- LLM fallback for complex queries
- Smart normalization to canonical station names

Author: AI Istanbul Team
Date: January 2026
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


# ============================================================
# SMART LOCATION NORMALIZER - Maps LLM output to canonical names
# ============================================================

class SmartLocationNormalizer:
    """
    Normalizes LLM-extracted location names to canonical station names.
    
    This bridges the gap between free-form LLM output and our station database.
    Uses fuzzy matching, synonym expansion, and contextual understanding.
    """
    
    # Canonical name mappings - LLM might return these variations
    CANONICAL_MAPPINGS = {
        # Taksim variations
        "taksim square": "taksim",
        "taksim meydanı": "taksim",
        "площадь таксим": "taksim",
        "place taksim": "taksim",
        "taksimplatz": "taksim",
        
        # Sultanahmet / Blue Mosque / Hagia Sophia
        "blue mosque": "sultanahmet",
        "the blue mosque": "sultanahmet",
        "hagia sophia": "sultanahmet",
        "aya sophia": "sultanahmet",
        "ayasofya": "sultanahmet",
        "aya sofya": "sultanahmet",
        "голубая мечеть": "sultanahmet",
        "святая софия": "sultanahmet",
        "mosquée bleue": "sultanahmet",
        "blaue moschee": "sultanahmet",
        "المسجد الأزرق": "sultanahmet",
        "آيا صوفيا": "sultanahmet",
        "hippodrome": "sultanahmet",
        
        # Galata Tower
        "galata tower": "galata",
        "galata kulesi": "galata",
        "галатская башня": "galata",
        "tour de galata": "galata",
        "galata turm": "galata",
        "برج غلطة": "galata",
        
        # Grand Bazaar
        "grand bazaar": "kapalıçarşı",
        "the grand bazaar": "kapalıçarşı",
        "covered bazaar": "kapalıçarşı",
        "kapali carsi": "kapalıçarşı",
        "kapalı çarşı": "kapalıçarşı",
        "гранд базар": "kapalıçarşı",
        "großer basar": "kapalıçarşı",
        "البازار الكبير": "kapalıçarşı",
        
        # Spice Bazaar / Egyptian Bazaar
        "spice bazaar": "eminönü",
        "spice market": "eminönü",
        "egyptian bazaar": "eminönü",
        "mısır çarşısı": "eminönü",
        "рынок специй": "eminönü",
        "gewürzbasar": "eminönü",
        "سوق التوابل": "eminönü",
        
        # Palaces
        "dolmabahce palace": "dolmabahçe",
        "dolmabahçe sarayı": "dolmabahçe",
        "дворец долмабахче": "dolmabahçe",
        "topkapi palace": "topkapı",
        "topkapı sarayı": "topkapı",
        "дворец топкапы": "topkapı",
        
        # Airports
        "istanbul airport": "havalimanı",
        "new airport": "havalimanı",
        "аэропорт": "havalimanı",
        "flughafen": "havalimanı",
        "aéroport": "havalimanı",
        "المطار": "havalimanı",
        "sabiha gokcen": "sabiha gökçen",
        "sabiha gokcen airport": "sabiha gökçen",
        
        # Generic areas
        "asian side": "üsküdar",
        "asia side": "üsküdar",
        "anatolian side": "üsküdar",
        "anadolu yakası": "üsküdar",
        "европейская сторона": "taksim",
        "european side": "taksim",
        "old city": "sultanahmet",
        "historic peninsula": "sultanahmet",
        "tarihi yarımada": "sultanahmet",
        "city center": "taksim",
        "downtown": "taksim",
    }
    
    @classmethod
    def normalize(cls, location: str, aliases: Dict[str, List[str]]) -> str:
        """
        Normalize a location name to a canonical form that exists in our alias database.
        
        Args:
            location: Raw location name from LLM
            aliases: Station aliases dictionary
            
        Returns:
            Normalized location name
        """
        if not location:
            return location
            
        location_lower = location.lower().strip()
        
        # Step 1: Check if it's already a canonical name in aliases
        if location_lower in aliases:
            return location_lower
        
        # Step 2: Check our predefined canonical mappings
        if location_lower in cls.CANONICAL_MAPPINGS:
            canonical = cls.CANONICAL_MAPPINGS[location_lower]
            logger.info(f"🔄 Normalized '{location}' → '{canonical}' (canonical mapping)")
            return canonical
        
        # Step 3: Remove Turkish grammatical suffixes
        stripped = cls._strip_turkish_suffixes(location_lower)
        if stripped != location_lower and stripped in aliases:
            logger.info(f"🔄 Normalized '{location}' → '{stripped}' (suffix stripping)")
            return stripped
        
        # Step 4: Try fuzzy matching against aliases
        best_match, score = cls._fuzzy_match(location_lower, aliases.keys())
        if score >= 0.85:  # High confidence threshold
            logger.info(f"🔄 Normalized '{location}' → '{best_match}' (fuzzy match, score={score:.2f})")
            return best_match
        
        # Step 5: Return as-is (the alias system will try its own matching)
        logger.info(f"ℹ️ Location '{location}' not normalized, passing through")
        return location_lower
    
    @classmethod
    def _strip_turkish_suffixes(cls, text: str) -> str:
        """Remove common Turkish grammatical suffixes."""
        suffixes = [
            "'den", "'dan", "'ten", "'tan",  # ablative with apostrophe
            "den", "dan", "ten", "tan",       # ablative
            "'e", "'a", "'ye", "'ya",         # dative with apostrophe
            "e", "a", "ye", "ya",             # dative (careful - short)
            "'de", "'da", "'te", "'ta",       # locative with apostrophe
            "de", "da", "te", "ta",           # locative
            "'nden", "nden", "'ndan", "ndan", # ablative with buffer n
            "'ne", "ne", "'na", "na",         # dative with buffer n
            "'nde", "nde", "'nda", "nda",     # locative with buffer n
        ]
        
        for suffix in sorted(suffixes, key=len, reverse=True):
            if text.endswith(suffix) and len(text) > len(suffix) + 2:
                return text[:-len(suffix)]
        
        return text
    
    @classmethod
    def _fuzzy_match(cls, text: str, candidates: List[str]) -> Tuple[str, float]:
        """Find best fuzzy match among candidates."""
        best_match = text
        best_score = 0.0
        
        # Normalize input
        text_normalized = remove_turkish_diacritics(text.lower())
        
        for candidate in candidates:
            candidate_normalized = remove_turkish_diacritics(candidate.lower())
            score = normalized_levenshtein_similarity(text_normalized, candidate_normalized)
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match, best_score


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
        self.normalizer = SmartLocationNormalizer()
        self._build_known_locations()
        
        # LLM fallback configuration
        self.use_llm_fallback = True  # Enable/disable LLM fallback
        self.llm_fallback_threshold = 1  # Use LLM if fewer than this many locations found
    
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
        user_location: Optional[Dict[str, float]] = None,
        use_llm: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract origin and destination using pattern-free location recognition.
        
        Strategy:
        1. Find ALL known locations in the query (keyword matching)
        2. Use contextual clues (from/to keywords) to assign roles
        3. If only 1 location found and user has GPS, use GPS as origin
        4. **NEW: If keyword matching fails, use LLM as intelligent fallback**
        5. Fallback: assume first location = origin, last = destination
        
        Args:
            query: User's transportation query
            user_location: Optional GPS coordinates
            use_llm: Whether to use LLM fallback (default: True)
            
        Returns:
            Tuple of (origin, destination)
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
        
        # ============================================================
        # LLM FALLBACK: Use LLM when keyword matching is insufficient
        # ============================================================
        if use_llm and self.use_llm_fallback and len(filtered_locations) < 2:
            logger.info(f"🤖 Keyword matching insufficient, trying LLM extraction...")
            llm_origin, llm_destination = self._extract_with_llm(query)
            
            if llm_origin or llm_destination:
                logger.info(f"🤖 LLM extracted: origin='{llm_origin}', destination='{llm_destination}'")
                
                # Handle special "current_location" marker
                if llm_origin == 'current_location' and user_location:
                    llm_origin = "Your Location"
                elif llm_origin == 'current_location':
                    llm_origin = None
                
                # If LLM found both, use them directly
                if llm_origin and llm_destination:
                    return llm_origin, llm_destination
                
                # If LLM found one and keyword found one, combine them
                if len(filtered_locations) == 1:
                    keyword_location = filtered_locations[0]['name']
                    if llm_destination and not llm_origin:
                        # LLM found destination, use keyword as origin
                        return keyword_location, llm_destination
                    elif llm_origin and not llm_destination:
                        # LLM found origin, use keyword as destination
                        return llm_origin, keyword_location
                
                # If only destination found by LLM and GPS available
                if llm_destination and not llm_origin and user_location:
                    return "Your Location", llm_destination
        
        # ============================================================
        # Continue with keyword-based extraction
        # ============================================================
        
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

    def _extract_with_llm(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Use LLM as intelligent fallback for location extraction.
        
        This method is called when keyword/alias matching fails to find
        sufficient locations (fewer than 2) in the user's query.
        
        The LLM handles:
        - Misspellings and typos
        - Landmark descriptions (e.g., "the big tower")
        - Multilingual queries (6 languages)
        - Turkish grammatical suffixes
        - Contextual understanding
        
        Args:
            query: User's transportation query
            
        Returns:
            Tuple of (origin, destination) - normalized to canonical names
        """
        try:
            logger.info(f"🤖 LLM fallback extraction for: '{query}'")
            origin, destination = extract_locations_with_llm_sync(query, self.aliases)
            
            if origin or destination:
                logger.info(f"✅ LLM fallback successful: origin='{origin}', destination='{destination}'")
            else:
                logger.info(f"ℹ️ LLM fallback returned no results")
            
            return origin, destination
            
        except Exception as e:
            logger.error(f"❌ LLM fallback extraction failed: {e}")
            return None, None


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


async def extract_locations_with_llm(
    query: str,
    aliases: Optional[Dict[str, List[str]]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Use LLM to extract origin and destination when pattern matching fails.
    
    Supports 6 languages: English, Turkish, Russian, German, Arabic, French
    
    Features:
    - Handles misspellings and typos
    - Understands landmark references (e.g., "the tower in Galata")
    - Handles implicit locations (e.g., "near my hotel")
    - Understands contextual clues
    - Normalizes output to canonical station names
    
    Args:
        query: User's transportation query in any supported language
        aliases: Optional alias dictionary for normalization
        
    Returns:
        Tuple of (origin, destination) - normalized location names
    """
    client = get_llm_client_for_extraction()
    if not client:
        return None, None
    
    try:
        # Enhanced multilingual extraction prompt
        extraction_prompt = f"""You are an Istanbul transportation expert. Extract the ORIGIN (starting point) and DESTINATION (ending point) from this transportation query.

QUERY: "{query}"

LANGUAGE DETECTION:
- The query may be in English, Turkish, Russian, German, Arabic, or French
- Turkish may include grammatical suffixes like -den/-dan (from), -e/-a (to), -de/-da (at)
- Russian uses Cyrillic script
- Arabic uses Arabic script

COMMON ISTANBUL LOCATIONS (use these canonical names in your response):
- Tourist areas: Taksim, Sultanahmet, Karaköy, Galata, Eminönü, Beşiktaş, Kadıköy, Üsküdar
- Landmarks: Galata Tower → "Galata", Blue Mosque → "Sultanahmet", Hagia Sophia → "Sultanahmet"
- Grand Bazaar → "Kapalıçarşı", Spice Bazaar → "Eminönü"
- Palaces: Dolmabahçe → "Dolmabahçe", Topkapi → "Topkapı"
- Airports: Istanbul Airport → "Havalimanı", Sabiha Gökçen → "Sabiha Gökçen"
- Generic: "Asian side" → "Üsküdar", "European side" → "Taksim", "Old city" → "Sultanahmet"

EXTRACTION RULES:
1. If the user says "from X to Y", X is ORIGIN and Y is DESTINATION
2. If the user says "how to get to Y" (without origin), ORIGIN is "CURRENT_LOCATION"
3. Turkish "-den/-dan" suffix indicates ORIGIN, "-e/-a" suffix indicates DESTINATION
4. Russian "из/от" indicates ORIGIN, "в/на/до" indicates DESTINATION
5. German "von/aus" indicates ORIGIN, "nach/zu" indicates DESTINATION
6. French "de/depuis" indicates ORIGIN, "à/vers" indicates DESTINATION
7. Arabic "من" indicates ORIGIN, "إلى" indicates DESTINATION

RESPOND IN THIS EXACT FORMAT (no other text):
ORIGIN: [location name or CURRENT_LOCATION or UNKNOWN]
DESTINATION: [location name or UNKNOWN]

Examples:
- "Taksimden Galata Kulesine" → ORIGIN: Taksim, DESTINATION: Galata
- "How do I get to the Blue Mosque?" → ORIGIN: CURRENT_LOCATION, DESTINATION: Sultanahmet
- "Как добраться до аэропорта?" → ORIGIN: CURRENT_LOCATION, DESTINATION: Havalimanı
- "takisme nasil giderim" (misspelled) → ORIGIN: CURRENT_LOCATION, DESTINATION: Taksim"""

        response = await client.generate(
            prompt=extraction_prompt,
            max_tokens=100,
            temperature=0.1  # Low temperature for consistent extraction
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
                if origin.upper() in ('UNKNOWN', 'N/A', 'NONE', ''):
                    origin = None
                elif origin.upper() == 'CURRENT_LOCATION':
                    origin = 'current_location'  # Special marker for GPS fallback
            elif line.upper().startswith('DESTINATION:'):
                destination = line.split(':', 1)[1].strip()
                if destination.upper() in ('UNKNOWN', 'N/A', 'NONE', ''):
                    destination = None
        
        # Normalize extracted locations to canonical names
        if aliases:
            if origin and origin != 'current_location':
                origin = SmartLocationNormalizer.normalize(origin, aliases)
            if destination:
                destination = SmartLocationNormalizer.normalize(destination, aliases)
        
        logger.info(f"✅ LLM extracted: origin='{origin}', destination='{destination}'")
        return origin, destination
        
    except Exception as e:
        logger.error(f"❌ LLM location extraction failed: {e}")
        return None, None


def extract_locations_with_llm_sync(
    query: str,
    aliases: Optional[Dict[str, List[str]]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Synchronous wrapper for LLM location extraction.
    
    Args:
        query: User's transportation query
        aliases: Optional alias dictionary for normalization
        
    Returns:
        Tuple of (origin, destination) - normalized location names
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, extract_locations_with_llm(query, aliases))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(extract_locations_with_llm(query, aliases))
    except RuntimeError:
        return asyncio.run(extract_locations_with_llm(query, aliases))
    except Exception as e:
        logger.error(f"❌ Sync LLM extraction wrapper failed: {e}")
        return None, None
