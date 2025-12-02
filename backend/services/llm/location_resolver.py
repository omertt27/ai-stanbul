"""
LLM Location Resolver
=====================

Phase 2 of LLM Enhancement: Intelligent location extraction and disambiguation.

Replaces regex-based location matching with LLM-powered understanding:
- Natural language location queries
- Ambiguity resolution
- Multi-location extraction with order preservation
- Context-aware suggestions
- Fuzzy matching fallback

Example queries handled:
- "from Sultanahmet to Galata Tower"
- "visit Blue Mosque, then Hagia Sophia, and Grand Bazaar"
- "how do I get to the tower near Taksim"
- "plan route: start at my hotel, visit museums, end at bazaar"
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from functools import lru_cache
import json

from .models import LocationResolution, LocationMatch

logger = logging.getLogger(__name__)


class LLMLocationResolver:
    """
    LLM-powered location resolver for natural language queries.
    
    Features:
    - Intelligent location extraction from conversational queries
    - Disambiguation of ambiguous locations
    - Multi-location extraction with journey order preservation
    - Confidence scoring and fallback
    - Known location database integration
    """
    
    # Known locations database (Istanbul POIs)
    KNOWN_LOCATIONS = {
        'sultanahmet': (41.0054, 28.9768),
        'blue mosque': (41.0054, 28.9768),
        'hagia sophia': (41.0086, 28.9802),
        'topkapi palace': (41.0115, 28.9833),
        'grand bazaar': (41.0108, 28.9680),
        'spice bazaar': (41.0166, 28.9703),
        'galata tower': (41.0256, 28.9742),
        'taksim square': (41.0370, 28.9857),
        'istiklal street': (41.0333, 28.9784),
        'dolmabahce palace': (41.0391, 29.0003),
        'basilica cistern': (41.0084, 28.9779),
        'maiden tower': (41.0211, 29.0043),
        'ortakoy': (41.0553, 29.0264),
        'bebek': (41.0797, 29.0434),
        'eminonu': (41.0166, 28.9703),
        'kadikoy': (40.9897, 29.0250),
        'besiktas': (41.0426, 29.0067),
        'balat': (41.0292, 28.9485),
        'fener': (41.0295, 28.9488),
        'chora church': (41.0308, 28.9387),
        'suleymaniye mosque': (41.0165, 28.9639),
        'rumeli fortress': (41.0839, 29.0566),
        'pierre loti': (41.0534, 28.9347),
        'camlica hill': (41.0217, 29.0664),
        'princes islands': (40.8562, 29.1208),
    }
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        cache_size: int = 128
    ):
        """
        Initialize LLM Location Resolver.
        
        Args:
            model: OpenAI model to use for location resolution
            temperature: Temperature for LLM (low for consistency)
            cache_size: Size of LRU cache for location queries
        """
        self.model = model
        self.temperature = temperature
        
        # Manual cache for location resolutions (avoids unhashable dict issues)
        self._resolution_cache: Dict[str, LocationResolution] = {}
        self._cache_size = cache_size
        self._cache_order: List[str] = []  # For LRU eviction
        
        logger.info(
            f"✅ LLM Location Resolver initialized "
            f"(model={model}, cache={cache_size})"
        )
    
    def resolve_locations(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> LocationResolution:
        """
        Resolve locations from natural language query.
        
        Args:
            query: Natural language location query
            user_context: Optional user context (GPS, preferences, etc.)
            
        Returns:
            LocationResolution with extracted locations, confidence, and metadata
        """
        try:
            # Create cache key
            cache_key = self._make_cache_key(query, user_context)
            
            # Check cache
            if cache_key in self._resolution_cache:
                logger.debug(f"✅ Cache hit for: '{query[:50]}'")
                # Update LRU order
                self._cache_order.remove(cache_key)
                self._cache_order.append(cache_key)
                return self._resolution_cache[cache_key]
            
            logger.debug(f"❌ Cache miss for: '{query[:50]}'")
            
            # Resolve locations (uncached)
            result = self._resolve_locations_uncached(query, user_context)
            
            # Add to cache
            self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in location resolution: {e}", exc_info=True)
            
            # Fallback to regex-based extraction
            return self._fallback_extraction(query, user_context)
    
    def _resolve_locations_uncached(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> LocationResolution:
        """
        Uncached location resolution.
        
        Args:
            query: Natural language query
            user_context: User context
            
        Returns:
            LocationResolution result
        """
        logger.info(f"🔍 Resolving locations from: '{query}'")
        
        # Build prompt for LLM
        prompt = self._build_location_prompt(query, user_context)
        
        # Call LLM
        try:
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a location extraction expert for Istanbul tourism queries. Extract locations from user queries with high accuracy."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            # Parse LLM response
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM response: {result_text}")
            
            # Parse JSON output
            resolution = self._parse_llm_response(result_text, query)
            
            # Validate and enhance with known locations
            resolution = self._enhance_with_known_locations(resolution)
            
            logger.info(
                f"✅ Resolved {len(resolution.locations)} locations "
                f"(confidence={resolution.confidence:.2f})"
            )
            
            return resolution
            
        except Exception as e:
            logger.error(f"LLM location resolution failed: {e}")
            return self._fallback_extraction(query, user_context)
    
    def _build_location_prompt(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for LLM location extraction.
        
        Args:
            query: User query
            user_context: User context
            
        Returns:
            Formatted prompt string
        """
        # Get known location names for reference
        location_names = sorted(self.KNOWN_LOCATIONS.keys())
        location_list = ", ".join(location_names[:20])  # Show subset
        
        # Build context info
        context_info = ""
        if user_context:
            if 'gps' in user_context:
                gps = user_context['gps']
                context_info += f"\n- User's current GPS: {gps.get('lat'):.4f}, {gps.get('lon'):.4f}"
            
            if 'previous_locations' in user_context:
                prev = user_context['previous_locations']
                context_info += f"\n- Recent locations: {', '.join(prev)}"
        
        prompt = f"""Extract locations from this Istanbul tourism query:

Query: "{query}"

Known Istanbul locations: {location_list}, and more...
{context_info}

Instructions:
1. Identify all location references in the query
2. Match them to known Istanbul locations (use exact names if possible)
3. Preserve the order mentioned (journey order matters!)
4. Identify the intent pattern (from-to, multi-stop, destination-only, etc.)
5. Handle ambiguity (e.g., "tower" could be Galata Tower or Maiden Tower)

Output a JSON object with this structure:
{{
  "locations": [
    {{
      "name": "Location Name",
      "matched_to": "exact known location name or null",
      "confidence": 0.0-1.0,
      "coordinates": [lat, lon] or null,
      "disambiguation_note": "optional note if ambiguous"
    }}
  ],
  "pattern": "from_to | multi_stop | destination_only | area_exploration",
  "confidence": 0.0-1.0,
  "ambiguities": ["list any ambiguous references"],
  "suggestions": ["alternative interpretations if confidence < 0.8"]
}}

Output only valid JSON, no additional text."""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response_text: str,
        original_query: str
    ) -> LocationResolution:
        """
        Parse LLM JSON response into LocationResolution.
        
        Args:
            response_text: LLM response text
            original_query: Original user query
            
        Returns:
            LocationResolution object
        """
        try:
            # Try to extract JSON from response
            # Handle cases where LLM adds extra text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)
            else:
                # Fallback: whole response is JSON
                data = json.loads(response_text)
            
            # Parse location matches
            locations = []
            for loc_data in data.get('locations', []):
                match = LocationMatch(
                    name=loc_data.get('name', ''),
                    matched_name=loc_data.get('matched_to'),
                    coordinates=tuple(loc_data['coordinates']) if loc_data.get('coordinates') else None,
                    confidence=loc_data.get('confidence', 0.5),
                    disambiguation_note=loc_data.get('disambiguation_note')
                )
                locations.append(match)
            
            # Create resolution
            resolution = LocationResolution(
                query=original_query,
                locations=locations,
                pattern=data.get('pattern', 'unknown'),
                confidence=data.get('confidence', 0.5),
                ambiguities=data.get('ambiguities', []),
                suggestions=data.get('suggestions', []),
                used_llm=True,
                fallback_used=False
            )
            
            return resolution
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            return self._fallback_extraction(original_query, None)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._fallback_extraction(original_query, None)
    
    def _enhance_with_known_locations(
        self,
        resolution: LocationResolution
    ) -> LocationResolution:
        """
        Enhance location resolution with known location database.
        
        Fills in missing coordinates and improves matching.
        
        Args:
            resolution: LocationResolution from LLM
            
        Returns:
            Enhanced LocationResolution
        """
        enhanced_locations = []
        
        for loc_match in resolution.locations:
            # If no coordinates, try to find in known locations
            if loc_match.coordinates is None:
                # Try exact match
                name_lower = loc_match.name.lower()
                if name_lower in self.KNOWN_LOCATIONS:
                    coords = self.KNOWN_LOCATIONS[name_lower]
                    loc_match = LocationMatch(
                        name=loc_match.name,
                        matched_name=name_lower,
                        coordinates=coords,
                        confidence=min(loc_match.confidence + 0.1, 1.0),
                        disambiguation_note=loc_match.disambiguation_note
                    )
                else:
                    # Try fuzzy match
                    matched_name, coords, score = self._fuzzy_match_location(loc_match.name)
                    if matched_name and score > 0.7:
                        loc_match = LocationMatch(
                            name=loc_match.name,
                            matched_name=matched_name,
                            coordinates=coords,
                            confidence=min(loc_match.confidence * score, 1.0),
                            disambiguation_note=f"Fuzzy matched to {matched_name}"
                        )
            
            enhanced_locations.append(loc_match)
        
        # Update resolution
        resolution.locations = enhanced_locations
        
        # Recalculate overall confidence
        if enhanced_locations:
            avg_confidence = sum(loc.confidence for loc in enhanced_locations) / len(enhanced_locations)
            resolution.confidence = avg_confidence
        
        return resolution
    
    def _fuzzy_match_location(
        self,
        query: str
    ) -> Tuple[Optional[str], Optional[Tuple[float, float]], float]:
        """
        Fuzzy match a location query to known locations.
        
        Args:
            query: Location query string
            
        Returns:
            Tuple of (matched_name, coordinates, confidence_score)
        """
        query_lower = query.lower().strip()
        best_match = None
        best_coords = None
        best_score = 0.0
        
        # Strategy 1: Substring match
        for loc_name, coords in self.KNOWN_LOCATIONS.items():
            if query_lower in loc_name or loc_name in query_lower:
                # Calculate score based on length ratio
                score = min(len(query_lower), len(loc_name)) / max(len(query_lower), len(loc_name))
                if score > best_score:
                    best_match = loc_name
                    best_coords = coords
                    best_score = score
        
        # Strategy 2: Word overlap
        if best_score < 0.8:
            query_words = set(query_lower.split())
            for loc_name, coords in self.KNOWN_LOCATIONS.items():
                loc_words = set(loc_name.split())
                overlap = len(query_words & loc_words)
                if overlap > 0:
                    score = overlap / max(len(query_words), len(loc_words))
                    if score > best_score:
                        best_match = loc_name
                        best_coords = coords
                        best_score = score
        
        if best_score > 0.5:
            logger.debug(f"Fuzzy matched '{query}' to '{best_match}' (score={best_score:.2f})")
            return best_match, best_coords, best_score
        
        return None, None, 0.0
    
    def _fallback_extraction(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> LocationResolution:
        """
        Fallback to regex-based location extraction.
        
        Args:
            query: User query
            user_context: User context
            
        Returns:
            LocationResolution from fallback method
        """
        logger.warning(f"⚠️ Using fallback extraction for: '{query}'")
        
        query_lower = query.lower()
        found_locations = []
        
        # Find all mentioned known locations
        for loc_name, coords in self.KNOWN_LOCATIONS.items():
            if loc_name in query_lower:
                # Find position in query for ordering
                pos = query_lower.find(loc_name)
                match = LocationMatch(
                    name=loc_name.title(),
                    matched_name=loc_name,
                    coordinates=coords,
                    confidence=0.7,  # Lower confidence for fallback
                    disambiguation_note=None
                )
                found_locations.append((pos, match))
        
        # Sort by position to preserve order
        found_locations.sort(key=lambda x: x[0])
        locations = [match for _, match in found_locations]
        
        # Determine pattern
        pattern = 'unknown'
        if len(locations) == 2:
            pattern = 'from_to'
        elif len(locations) > 2:
            pattern = 'multi_stop'
        elif len(locations) == 1:
            pattern = 'destination_only'
        
        # Calculate confidence
        confidence = 0.6 if locations else 0.0
        
        resolution = LocationResolution(
            query=query,
            locations=locations,
            pattern=pattern,
            confidence=confidence,
            ambiguities=[],
            suggestions=[],
            used_llm=False,
            fallback_used=True
        )
        
        logger.info(
            f"✅ Fallback extracted {len(locations)} locations "
            f"(confidence={confidence:.2f})"
        )
        
        return resolution
    
    def _add_to_cache(self, cache_key: str, result: LocationResolution):
        """Add result to cache with LRU eviction."""
        # Evict oldest if at capacity
        if len(self._resolution_cache) >= self._cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._resolution_cache[oldest_key]
            logger.debug(f"🗑️ Evicted cache entry: {oldest_key[:50]}")
        
        # Add new entry
        self._resolution_cache[cache_key] = result
        self._cache_order.append(cache_key)
    
    def _make_cache_key(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Create cache key for location resolution.
        
        Args:
            query: User query
            user_context: User context
            
        Returns:
            Cache key string
        """
        # Normalize query
        query_normalized = query.lower().strip()
        
        # Include relevant context in cache key
        context_key = ""
        if user_context:
            # Include GPS if present (rounded to 3 decimals)
            if 'gps' in user_context:
                gps = user_context['gps']
                lat = round(gps.get('lat', 0), 3)
                lon = round(gps.get('lon', 0), 3)
                context_key += f"|gps:{lat},{lon}"
        
        return f"{query_normalized}{context_key}"
    
    def clear_cache(self):
        """Clear the location resolution cache."""
        self._resolution_cache.clear()
        self._cache_order.clear()
        logger.info("🗑️ Location resolution cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache hit rate and size info
        """
        return {
            'size': len(self._resolution_cache),
            'max_size': self._cache_size,
            'keys': list(self._cache_order)
        }


# Global resolver instance
_location_resolver: Optional[LLMLocationResolver] = None


def get_location_resolver() -> LLMLocationResolver:
    """Get or create global location resolver instance."""
    global _location_resolver
    if _location_resolver is None:
        _location_resolver = LLMLocationResolver()
    return _location_resolver


# Convenience function for direct usage
def resolve_locations(
    query: str,
    user_context: Optional[Dict[str, Any]] = None
) -> LocationResolution:
    """
    Resolve locations from natural language query.
    
    Args:
        query: Natural language location query
        user_context: Optional user context
        
    Returns:
        LocationResolution with extracted locations
    """
    resolver = get_location_resolver()
    return resolver.resolve_locations(query, user_context)
