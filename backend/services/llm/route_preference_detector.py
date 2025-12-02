"""
LLM Route Preference Detector
==============================

Phase 4.1 of LLM Enhancement: Extract route preferences from natural language.

Gives LLM complete control over understanding HOW users want to travel:
- Optimization goals: speed, cost, scenic, accessibility, comfort
- Transport mode preferences: walk, metro, tram, bus, ferry, taxi
- Avoidance preferences: stairs, crowds, hills, transfers
- Accessibility needs: wheelchair, stroller, elderly
- Time constraints: rush, flexible, specific time
- Budget considerations: cheap, moderate, expensive

Example extractions:
- "fastest way to Taksim" → optimize_for="speed"
- "wheelchair accessible route" → accessibility="wheelchair", avoid=["stairs"]
- "scenic walk to Galata" → optimize_for="scenic", prefer_walking=True
- "I'm in a hurry" → optimize_for="speed", time_constraint="rush"
- "I have a baby stroller" → avoid=["stairs", "escalators"], accessibility="stroller"

Author: Istanbul AI Team
Date: December 2025
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from .models import RoutePreferences

logger = logging.getLogger(__name__)


class LLMRoutePreferenceDetector:
    """
    LLM-powered route preference detection.
    
    Extracts routing preferences from natural language queries, giving the LLM
    complete control over understanding user needs and constraints.
    
    This replaces hardcoded defaults with intelligent, context-aware preference detection.
    """
    
    def __init__(
        self,
        llm_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Route Preference Detector.
        
        Args:
            llm_client: LLM API client for preference extraction
            config: Configuration overrides
        """
        self.llm_client = llm_client
        
        # Configuration
        self.config = {
            'enable_llm': True,
            'fallback_to_rules': True,
            'timeout_seconds': 2,
            'cache_enabled': True,
            'min_confidence': 0.6,
            **(config or {})
        }
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'llm_detections': 0,
            'fallback_detections': 0,
            'average_latency_ms': 0.0
        }
        
        # Manual LRU cache (simple dict-based)
        self._cache = {}
        self._cache_order = []
        self._cache_max_size = 100
        
        logger.info("✅ LLM Route Preference Detector initialized")
    
    async def detect_preferences(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]] = None,
        route_context: Optional[Dict[str, Any]] = None
    ) -> RoutePreferences:
        """
        Detect route preferences from natural language query.
        
        Args:
            query: User's route request query
            user_profile: User profile with saved preferences
            route_context: Additional route context (distance, duration, etc.)
            
        Returns:
            RoutePreferences object with extracted preferences
        """
        start_time = datetime.now()
        
        try:
            self.stats['total_detections'] += 1
            
            # Check cache
            cache_key = self._make_cache_key(query, user_profile)
            if self.config['cache_enabled'] and cache_key in self._cache:
                logger.debug(f"Cache hit for preferences: '{query[:50]}...'")
                return self._cache[cache_key]
            
            # Try LLM detection
            if self.llm_client and self.config['enable_llm']:
                try:
                    preferences = await self._llm_detect_preferences(
                        query, user_profile, route_context
                    )
                    self.stats['llm_detections'] += 1
                    
                    # Cache result
                    if self.config['cache_enabled']:
                        self._add_to_cache(cache_key, preferences)
                    
                    return preferences
                    
                except Exception as e:
                    logger.warning(f"LLM preference detection failed: {e}")
                    if not self.config['fallback_to_rules']:
                        raise
            
            # Fallback to rule-based detection
            preferences = self._rule_based_detect(query, user_profile, route_context)
            self.stats['fallback_detections'] += 1
            
            # Cache result
            if self.config['cache_enabled']:
                self._add_to_cache(cache_key, preferences)
            
            return preferences
            
        finally:
            # Update stats
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.stats['average_latency_ms'] = (
                (self.stats['average_latency_ms'] * (self.stats['total_detections'] - 1) + latency_ms)
                / self.stats['total_detections']
            )
    
    async def _llm_detect_preferences(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]],
        route_context: Optional[Dict[str, Any]]
    ) -> RoutePreferences:
        """
        Use LLM to detect preferences from natural language.
        
        This is the PRIMARY method - LLM has full control.
        """
        
        # Build prompt for LLM
        prompt = self._build_detection_prompt(query, user_profile, route_context)
        
        # Call LLM
        if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
            # OpenAI-style client
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a route preference extraction expert. Extract routing preferences from user queries and return them as structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
                timeout=self.config['timeout_seconds']
            )
            
            llm_output = response.choices[0].message.content.strip()
        else:
            # Fallback: assume generate method
            llm_output = await self.llm_client.generate(prompt)
        
        # Parse LLM response
        preferences = self._parse_llm_response(llm_output, query)
        
        logger.info(
            f"✅ LLM detected preferences for '{query[:50]}...': "
            f"optimize={preferences.optimize_for}, "
            f"accessibility={preferences.accessibility}, "
            f"avoid={preferences.avoid}"
        )
        
        return preferences
    
    def _build_detection_prompt(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]],
        route_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM preference detection."""
        
        prompt = f"""Extract route preferences from this Istanbul travel query:

Query: "{query}"

User Profile: {json.dumps(user_profile or {}, indent=2)}

Route Context: {json.dumps(route_context or {}, indent=2)}

Extract the following preferences in JSON format:

1. **optimize_for**: What should the route optimize for?
   - "speed": User wants fastest route
   - "cost": User wants cheapest route
   - "scenic": User wants nice views/experience
   - "accessibility": User has accessibility needs
   - "comfort": User wants comfortable journey
   - "ease": User wants easiest/least effort
   - null: No specific optimization

2. **transport_modes**: Preferred transport modes (array)
   - Options: "walk", "metro", "tram", "bus", "ferry", "taxi", "car"
   - null or empty: No specific preference

3. **avoid**: Things to avoid (array)
   - Options: "stairs", "crowds", "hills", "transfers", "walking", "waiting", "heat", "rain"
   - null or empty: No specific avoidances

4. **accessibility**: Accessibility requirements
   - "wheelchair": Wheelchair accessible route needed
   - "stroller": Baby stroller friendly
   - "elderly": Elderly-friendly (less walking, more rests)
   - "none": No special accessibility needs
   - null: Not mentioned

5. **time_constraint**: Time constraint type
   - "rush": User is in a hurry
   - "flexible": User has flexible time
   - "specific_time": Needs to arrive by specific time
   - null: No time constraint mentioned

6. **weather_consideration**: Should weather be considered?
   - true: User mentions weather or wants weather-aware route
   - false: Weather not a concern

7. **budget**: Budget consideration
   - "cheap": User wants cheapest option
   - "moderate": Normal budget
   - "expensive": Cost not a concern
   - null: Not mentioned

8. **comfort_level**: Desired comfort level
   - "high": User wants comfortable journey
   - "medium": Normal comfort
   - "low": Doesn't care about comfort
   - null: Not mentioned

9. **max_walking_distance_km**: Maximum walking distance (number or null)

10. **max_transfers**: Maximum number of transfers (number or null)

11. **prefer_walking**: Prefer walking routes? (boolean)

12. **prefer_public_transport**: Prefer public transport? (boolean)

Examples:

Query: "fastest way to Taksim"
→ {{"optimize_for": "speed", "transport_modes": null, "avoid": null}}

Query: "wheelchair accessible route to Hagia Sophia"
→ {{"optimize_for": "accessibility", "accessibility": "wheelchair", "avoid": ["stairs"]}}

Query: "scenic walk to Galata Tower"
→ {{"optimize_for": "scenic", "prefer_walking": true, "transport_modes": ["walk"]}}

Query: "I'm in a hurry, take me to airport"
→ {{"optimize_for": "speed", "time_constraint": "rush"}}

Query: "cheapest way to Sultanahmet"
→ {{"optimize_for": "cost", "budget": "cheap"}}

Query: "I have a baby stroller"
→ {{"accessibility": "stroller", "avoid": ["stairs", "escalators"]}}

Query: "It's raining, how do I get to Blue Mosque"
→ {{"weather_consideration": true, "avoid": ["walking"]}}

Query: "I'm tired, easy route to hotel"
→ {{"optimize_for": "ease", "prefer_public_transport": true, "avoid": ["walking", "stairs"]}}

Now extract preferences from the query above. Return ONLY valid JSON, no explanation."""
        
        return prompt
    
    def _parse_llm_response(self, llm_output: str, query: str) -> RoutePreferences:
        """Parse LLM JSON response into RoutePreferences object."""
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            data = json.loads(llm_output)
            
            # Create RoutePreferences object
            preferences = RoutePreferences(
                optimize_for=data.get('optimize_for'),
                transport_modes=data.get('transport_modes'),
                avoid=data.get('avoid'),
                accessibility=data.get('accessibility'),
                time_constraint=data.get('time_constraint'),
                weather_consideration=data.get('weather_consideration', False),
                budget=data.get('budget'),
                comfort_level=data.get('comfort_level'),
                max_walking_distance_km=data.get('max_walking_distance_km'),
                max_transfers=data.get('max_transfers'),
                prefer_walking=data.get('prefer_walking', False),
                prefer_public_transport=data.get('prefer_public_transport', False),
                source='llm',
                confidence=0.9  # High confidence for LLM detection
            )
            
            return preferences
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"LLM output was: {llm_output}")
            
            # Fall back to rule-based
            return self._rule_based_detect(query, None, None)
    
    def _rule_based_detect(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]],
        route_context: Optional[Dict[str, Any]]
    ) -> RoutePreferences:
        """
        Rule-based fallback for preference detection.
        
        This is ONLY used when LLM is unavailable.
        """
        
        query_lower = query.lower()
        
        # Initialize preferences
        preferences = RoutePreferences(
            source='fallback',
            confidence=0.6
        )
        
        # Detect optimization goal
        if any(kw in query_lower for kw in ['fast', 'quick', 'hurry', 'rush', 'urgent']):
            preferences.optimize_for = 'speed'
            preferences.time_constraint = 'rush'
            # Also check for budget constraint
            if any(kw in query_lower for kw in ['cheap', 'cheapest', 'affordable']):
                preferences.budget = 'cheap'
        elif any(kw in query_lower for kw in ['cheap', 'cheapest', 'affordable', 'budget']):
            preferences.optimize_for = 'cost'
            preferences.budget = 'cheap'
        elif any(kw in query_lower for kw in ['scenic', 'beautiful', 'view', 'nice']):
            preferences.optimize_for = 'scenic'
            preferences.prefer_walking = True
        elif any(kw in query_lower for kw in ['easy', 'easiest', 'tired']):
            preferences.optimize_for = 'ease'
            preferences.avoid = ['stairs', 'walking', 'hills']
        
        # Detect accessibility
        if any(kw in query_lower for kw in ['wheelchair', 'disabled', 'accessibility']):
            preferences.accessibility = 'wheelchair'
            preferences.avoid = ['stairs']
            preferences.optimize_for = 'accessibility'
        elif any(kw in query_lower for kw in ['stroller', 'baby', 'pram']):
            preferences.accessibility = 'stroller'
            preferences.avoid = ['stairs', 'escalators']
        elif any(kw in query_lower for kw in ['elderly', 'old', 'senior']):
            preferences.accessibility = 'elderly'
            preferences.avoid = ['stairs', 'walking']
        
        # Detect transport preferences
        if any(kw in query_lower for kw in ['walk', 'walking', 'foot']):
            preferences.transport_modes = ['walk']
            preferences.prefer_walking = True
        elif any(kw in query_lower for kw in ['metro', 'subway']):
            preferences.transport_modes = ['metro']
        elif any(kw in query_lower for kw in ['tram']):
            preferences.transport_modes = ['tram']
        elif any(kw in query_lower for kw in ['bus']):
            preferences.transport_modes = ['bus']
        elif any(kw in query_lower for kw in ['ferry', 'boat']):
            preferences.transport_modes = ['ferry']
        elif any(kw in query_lower for kw in ['taxi', 'cab', 'uber']):
            preferences.transport_modes = ['taxi']
        
        # Detect avoidances
        avoid_list = []
        if 'no stairs' in query_lower or 'avoid stairs' in query_lower or 'without stairs' in query_lower:
            avoid_list.append('stairs')
        if 'no crowds' in query_lower or 'avoid crowds' in query_lower or 'crowded' in query_lower:
            avoid_list.append('crowds')
        if 'no hills' in query_lower or 'avoid hills' in query_lower:
            avoid_list.append('hills')
        if 'no transfers' in query_lower or 'avoid transfers' in query_lower or 'direct' in query_lower:
            avoid_list.append('transfers')
        if 'no walking' in query_lower or 'avoid walking' in query_lower:
            avoid_list.append('walking')
        
        if avoid_list:
            preferences.avoid = avoid_list
        
        # Detect weather consideration
        if any(kw in query_lower for kw in ['rain', 'raining', 'wet', 'weather', 'hot', 'cold']):
            preferences.weather_consideration = True
            if 'rain' in query_lower:
                if preferences.avoid:
                    preferences.avoid.append('walking')
                else:
                    preferences.avoid = ['walking']
        
        # Use user profile preferences if available
        if user_profile:
            if not preferences.accessibility and user_profile.get('accessibility'):
                preferences.accessibility = user_profile['accessibility']
            if not preferences.budget and user_profile.get('budget'):
                preferences.budget = user_profile['budget']
        
        logger.info(
            f"⚠️  Fallback preference detection for '{query[:50]}...': "
            f"optimize={preferences.optimize_for}, "
            f"accessibility={preferences.accessibility}"
        )
        
        return preferences
    
    def _make_cache_key(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for preferences."""
        profile_key = json.dumps(user_profile, sort_keys=True) if user_profile else ""
        return f"pref:{query.lower()}:{profile_key}"
    
    def _add_to_cache(self, key: str, preferences: RoutePreferences):
        """Add preferences to cache with LRU eviction."""
        # Remove oldest if cache is full
        if len(self._cache) >= self._cache_max_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
        
        # Add to cache
        self._cache[key] = preferences
        self._cache_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            **self.stats,
            'cache_size': len(self._cache),
            'llm_usage_rate': (
                self.stats['llm_detections'] / self.stats['total_detections']
                if self.stats['total_detections'] > 0 else 0
            )
        }


# Singleton instance
_preference_detector = None


def get_preference_detector(
    llm_client=None,
    config: Optional[Dict[str, Any]] = None
) -> LLMRoutePreferenceDetector:
    """
    Get or create Route Preference Detector singleton.
    
    Args:
        llm_client: LLM client (only used on first call)
        config: Configuration overrides
        
    Returns:
        LLMRoutePreferenceDetector instance
    """
    global _preference_detector
    
    if _preference_detector is None:
        _preference_detector = LLMRoutePreferenceDetector(
            llm_client=llm_client,
            config=config
        )
    
    return _preference_detector


# Convenience function
async def detect_route_preferences(
    query: str,
    user_profile: Optional[Dict[str, Any]] = None,
    route_context: Optional[Dict[str, Any]] = None,
    llm_client=None
) -> RoutePreferences:
    """
    Convenience function to detect route preferences.
    
    Args:
        query: User's route request query
        user_profile: Optional user profile
        route_context: Optional route context
        llm_client: Optional LLM client
        
    Returns:
        RoutePreferences object
    """
    detector = get_preference_detector(llm_client=llm_client)
    return await detector.detect_preferences(query, user_profile, route_context)
