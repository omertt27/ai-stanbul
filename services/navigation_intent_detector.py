"""
Navigation Intent Detector for AI Chat System
Enhanced location and navigation intent detection with OSRM integration

Detects navigation queries like:
- "How do I get from Sultanahmet to Taksim?"
- "Walking route to Blue Mosque"
- "Drive me to Galata Tower"
"""

import re
from typing import Optional, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NavigationIntent:
    """Detected navigation intent from user query"""
    intent: str = "navigation"
    origin: str = ""
    destination: str = ""
    mode: str = "walking"  # walking, driving, cycling
    include_pois: bool = False
    poi_categories: List[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'intent': self.intent,
            'origin': self.origin,
            'destination': self.destination,
            'mode': self.mode,
            'include_pois': self.include_pois,
            'poi_categories': self.poi_categories or []
        }


class NavigationIntentDetector:
    """
    Detect navigation and routing intents from natural language queries
    
    Supports multiple query patterns:
    - Point-to-point: "from X to Y"
    - Single destination: "take me to X"
    - Mode-specific: "walking route to X"
    - POI-enhanced: "route to X with museum stops"
    """
    
    # Navigation query patterns (order matters - more specific first)
    NAVIGATION_PATTERNS = [
        # Direct navigation requests with from/to
        r"(?:how (?:do i|can i|to)|show me (?:how to|the way to)) get (?:from|to) (.+?) (?:to|from) (.+?)(?:\?|$|\.)",
        r"(?:directions|route|navigate|navigation) (?:from|to) (.+?) (?:to|from) (.+?)(?:\?|$|\.)",
        r"take me (?:from|to) (.+?) (?:to|from) (.+?)(?:\?|$|\.)",
        r"show (?:me )?the way (?:from|to) (.+?) (?:to|from) (.+?)(?:\?|$|\.)",
        
        # Mode-specific with from/to
        r"(?:walk|walking) (?:from|to) (.+?) (?:to|from) (.+?)(?:\?|$|\.)",
        r"(?:drive|driving) (?:from|to) (.+?) (?:to|from) (.+?)(?:\?|$|\.)",
        r"(?:cycle|cycling|bike|biking) (?:from|to) (.+?) (?:to|from) (.+?)(?:\?|$|\.)",
        
        # Simple location-to-location
        r"(.+?) to (.+?) (?:route|directions|how)(?:\?|$|\.)",
        r"from (.+?) to (.+?)(?:\?|$|\.)",
        
        # Current location based (single destination)
        r"how (?:do i|can i|to) get to (.+?)(?: from here| from my location)?(?:\?|$|\.)",
        r"(?:directions|route|navigate|navigation) to (.+?)(?: from here| from my location)?(?:\?|$|\.)",
        r"take me to (.+?)(?:\?|$|\.)",
        r"show (?:me )?the way to (.+?)(?:\?|$|\.)",
        r"(?:walk|drive|cycle|bike) to (.+?)(?:\?|$|\.)",
        
        # Mode-specific to destination
        r"(?:walking|driving|cycling|biking) (?:directions|route) to (.+?)(?:\?|$|\.)",
    ]
    
    # POI-related keywords
    POI_KEYWORDS = [
        'stop', 'stops', 'visit', 'see', 'check out',
        'tourist', 'attraction', 'attractions', 'sights',
        'museum', 'museums', 'mosque', 'mosques',
        'palace', 'palaces', 'park', 'parks',
        'landmark', 'landmarks', 'monument', 'monuments'
    ]
    
    # Transport mode keywords
    MODE_KEYWORDS = {
        'walking': ['walk', 'walking', 'foot', 'pedestrian'],
        'driving': ['drive', 'driving', 'car', 'vehicle'],
        'cycling': ['cycle', 'cycling', 'bike', 'biking', 'bicycle']
    }
    
    def detect(self, query: str) -> Optional[NavigationIntent]:
        """
        Detect navigation intent from user query
        
        Args:
            query: User's natural language query
        
        Returns:
            NavigationIntent if navigation detected, None otherwise
        
        Example:
            >>> detector = NavigationIntentDetector()
            >>> intent = detector.detect("How do I get from Sultanahmet to Taksim?")
            >>> print(intent.origin, intent.destination, intent.mode)
            sultanahmet taksim walking
        """
        query_lower = query.lower().strip()
        
        # Detect transport mode
        mode = self._detect_mode(query_lower)
        
        # Detect POI preferences
        include_pois = self._detect_poi_preference(query_lower)
        poi_categories = self._extract_poi_categories(query_lower)
        
        # Try to extract origin and destination
        for pattern in self.NAVIGATION_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                
                if len(groups) >= 2:
                    # Two locations: origin and destination
                    origin = groups[0].strip()
                    destination = groups[1].strip()
                    
                    # Clean up location names
                    origin = self._clean_location_name(origin)
                    destination = self._clean_location_name(destination)
                    
                    return NavigationIntent(
                        intent='navigation',
                        origin=origin,
                        destination=destination,
                        mode=mode,
                        include_pois=include_pois,
                        poi_categories=poi_categories
                    )
                
                elif len(groups) == 1:
                    # Single destination, assume current location as origin
                    destination = groups[0].strip()
                    destination = self._clean_location_name(destination)
                    
                    return NavigationIntent(
                        intent='navigation',
                        origin='current_location',
                        destination=destination,
                        mode=mode,
                        include_pois=include_pois,
                        poi_categories=poi_categories
                    )
        
        return None
    
    def _detect_mode(self, query: str) -> str:
        """
        Detect transportation mode from query
        
        Args:
            query: Lowercase user query
        
        Returns:
            'walking', 'driving', or 'cycling'
        """
        for mode, keywords in self.MODE_KEYWORDS.items():
            if any(keyword in query for keyword in keywords):
                return mode
        
        return 'walking'  # Default mode
    
    def _detect_poi_preference(self, query: str) -> bool:
        """
        Detect if user wants POI recommendations along the route
        
        Args:
            query: Lowercase user query
        
        Returns:
            True if POI preference detected
        """
        return any(keyword in query for keyword in self.POI_KEYWORDS)
    
    def _extract_poi_categories(self, query: str) -> List[str]:
        """
        Extract POI categories mentioned in query
        
        Args:
            query: Lowercase user query
        
        Returns:
            List of POI categories
        """
        categories = []
        
        category_map = {
            'museum': ['museum', 'museums', 'gallery', 'galleries'],
            'mosque': ['mosque', 'mosques', 'camii'],
            'palace': ['palace', 'palaces', 'saray'],
            'park': ['park', 'parks', 'garden', 'gardens'],
            'restaurant': ['restaurant', 'restaurants', 'food', 'eat', 'dining'],
            'market': ['market', 'bazaar', 'shopping']
        }
        
        for category, keywords in category_map.items():
            if any(keyword in query for keyword in keywords):
                categories.append(category)
        
        # Default categories if POI preference detected but no specific category
        if not categories and self._detect_poi_preference(query):
            categories = ['museum', 'mosque', 'palace']
        
        return categories
    
    def _clean_location_name(self, location: str) -> str:
        """
        Clean up location name by removing common filler words
        
        Args:
            location: Raw location name from regex match
        
        Returns:
            Cleaned location name
        """
        # Remove common filler words
        filler_words = [
            'the', 'a', 'an', 'here', 'there',
            'from', 'to', 'at', 'in', 'on'
        ]
        
        words = location.split()
        cleaned_words = [
            word for word in words 
            if word not in filler_words
        ]
        
        return ' '.join(cleaned_words).strip()


# ═══════════════════════════════════════════════════════════════════
# Global Detector Instance
# ═══════════════════════════════════════════════════════════════════

_detector_instance: Optional[NavigationIntentDetector] = None


def get_navigation_detector() -> NavigationIntentDetector:
    """Get or create global navigation intent detector"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = NavigationIntentDetector()
    return _detector_instance


def detect_navigation_intent(query: str) -> Optional[Dict]:
    """
    Convenience function to detect navigation intent
    
    Args:
        query: User query string
    
    Returns:
        Dictionary with navigation intent details, or None
    
    Example:
        >>> intent = detect_navigation_intent("How do I get from Sultanahmet to Taksim?")
        >>> print(intent)
        {'intent': 'navigation', 'origin': 'sultanahmet', ...}
    """
    detector = get_navigation_detector()
    intent = detector.detect(query)
    return intent.to_dict() if intent else None


# ═══════════════════════════════════════════════════════════════════
# Testing & Examples
# ═══════════════════════════════════════════════════════════════════

def test_navigation_detection():
    """Test navigation intent detection with various queries"""
    print("\n🧠 Navigation Intent Detector - Test Suite\n")
    print("=" * 70)
    
    detector = NavigationIntentDetector()
    
    test_queries = [
        # Basic navigation
        "How do I get from Sultanahmet to Taksim?",
        "Directions from Galata Tower to Grand Bazaar",
        "Take me from Blue Mosque to Dolmabahce Palace",
        
        # Single destination
        "How do I get to Taksim?",
        "Take me to Blue Mosque",
        "Directions to Galata Tower",
        
        # Mode-specific
        "Walking route from Sultanahmet to Taksim",
        "Drive me to Dolmabahce Palace",
        "Cycling directions to Ortakoy",
        
        # With POI preferences
        "Route from Sultanahmet to Taksim with museum stops",
        "Take me to Grand Bazaar and show me mosques along the way",
        "Walking tour from Galata to Taksim visiting attractions",
        
        # Non-navigation queries (should return None)
        "What is the weather in Istanbul?",
        "Tell me about Hagia Sophia",
        "Best restaurants in Sultanahmet",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        
        intent = detector.detect(query)
        
        if intent:
            print(f"   ✅ Navigation detected!")
            print(f"      Origin: {intent.origin}")
            print(f"      Destination: {intent.destination}")
            print(f"      Mode: {intent.mode}")
            print(f"      Include POIs: {intent.include_pois}")
            if intent.poi_categories:
                print(f"      POI Categories: {', '.join(intent.poi_categories)}")
        else:
            print(f"   ❌ No navigation intent detected")
    
    print("\n" + "=" * 70)
    print("✅ Test suite completed!\n")


if __name__ == "__main__":
    test_navigation_detection()
