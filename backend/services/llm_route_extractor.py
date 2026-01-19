"""
LLM-Powered Route Data Extractor

This module uses the LLM to extract routing information from its own responses
and generates map_data + route_data dynamically, even for landmarks not in the database.

Author: AI Istanbul Team
Date: January 19, 2026
"""

import logging
import re
from typing import Dict, Optional, Any, List
import json

logger = logging.getLogger(__name__)


class LLMRouteExtractor:
    """
    Extracts routing information from LLM text responses and generates
    structured route_data and map_data for frontend visualization.
    
    This solves the problem of landmarks (like Galataport) not having
    route cards even though the LLM provides correct directions.
    """
    
    def __init__(self):
        """Initialize the route extractor."""
        # Common Istanbul landmarks and their nearest stations
        # Format: {"station": str, "line": str, "walk_min": int, "walk_m": int, "coords": [lat, lng], "station_coords": [lat, lng]}
        self.landmark_mappings = {
            # Cruise ports and waterfront
            "galataport": {
                "station": "Karaköy", "line": "T1", "walk_min": 5, "walk_m": 400,
                "coords": [41.0257, 28.9744],  # Galataport coords
                "station_coords": [41.0241, 28.9744]  # Karaköy station
            },
            "galata port": {
                "station": "Karaköy", "line": "T1", "walk_min": 5, "walk_m": 400,
                "coords": [41.0257, 28.9744],
                "station_coords": [41.0241, 28.9744]
            },
            "cruise terminal": {
                "station": "Karaköy", "line": "T1", "walk_min": 5, "walk_m": 400,
                "coords": [41.0257, 28.9744],
                "station_coords": [41.0241, 28.9744]
            },
            
            # Shopping and streets
            "istiklal": {
                "station": "Taksim", "line": "M2", "walk_min": 2, "walk_m": 150,
                "coords": [41.0369, 28.9784],
                "station_coords": [41.0370, 28.9869]
            },
            "istiklal street": {
                "station": "Taksim", "line": "M2", "walk_min": 2, "walk_m": 150,
                "coords": [41.0369, 28.9784],
                "station_coords": [41.0370, 28.9869]
            },
            "istiklal caddesi": {
                "station": "Taksim", "line": "M2", "walk_min": 2, "walk_m": 150,
                "coords": [41.0369, 28.9784],
                "station_coords": [41.0370, 28.9869]
            },
            
            # Markets
            "grand bazaar": {
                "station": "Beyazıt", "line": "T1", "walk_min": 3, "walk_m": 250,
                "coords": [41.0108, 28.9680],
                "station_coords": [41.0106, 28.9640]
            },
            "kapalı çarşı": {
                "station": "Beyazıt", "line": "T1", "walk_min": 3, "walk_m": 250,
                "coords": [41.0108, 28.9680],
                "station_coords": [41.0106, 28.9640]
            },
            "spice bazaar": {
                "station": "Eminönü", "line": "T1", "walk_min": 2, "walk_m": 150,
                "coords": [41.0166, 28.9709],
                "station_coords": [41.0170, 28.9710]
            },
            "mısır çarşısı": {
                "station": "Eminönü", "line": "T1", "walk_min": 2, "walk_m": 150,
                "coords": [41.0166, 28.9709],
                "station_coords": [41.0170, 28.9710]
            },
            
            # Attractions
            "galata tower": {
                "station": "Şişhane", "line": "M2", "walk_min": 5, "walk_m": 400,
                "coords": [41.0256, 28.9743],
                "station_coords": [41.0255, 28.9765]
            },
            "galata kulesi": {
                "station": "Şişhane", "line": "M2", "walk_min": 5, "walk_m": 400,
                "coords": [41.0256, 28.9743],
                "station_coords": [41.0255, 28.9765]
            },
            "maiden's tower": {
                "station": "Üsküdar", "line": "M5", "walk_min": 10, "walk_m": 800,
                "coords": [41.0211, 29.0040],
                "station_coords": [41.0268, 29.0155]
            },
            "kız kulesi": {
                "station": "Üsküdar", "line": "M5", "walk_min": 10, "walk_m": 800,
                "coords": [41.0211, 29.0040],
                "station_coords": [41.0268, 29.0155]
            },
            
            # Palaces
            "dolmabahçe palace": {
                "station": "Kabataş", "line": "T1", "walk_min": 10, "walk_m": 800,
                "coords": [41.0391, 29.0002],
                "station_coords": [41.0375, 28.9874]
            },
            "dolmabahçe sarayı": {
                "station": "Kabataş", "line": "T1", "walk_min": 10, "walk_m": 800,
                "coords": [41.0391, 29.0002],
                "station_coords": [41.0375, 28.9874]
            },
            "topkapı palace": {
                "station": "Sultanahmet", "line": "T1", "walk_min": 5, "walk_m": 400,
                "coords": [41.0115, 28.9833],
                "station_coords": [41.0054, 28.9768]
            },
            "topkapı sarayı": {
                "station": "Sultanahmet", "line": "T1", "walk_min": 5, "walk_m": 400,
                "coords": [41.0115, 28.9833],
                "station_coords": [41.0054, 28.9768]
            },
            
            # Mosques
            "hagia sophia": {
                "station": "Sultanahmet", "line": "T1", "walk_min": 2, "walk_m": 150,
                "coords": [41.0086, 28.9802],
                "station_coords": [41.0054, 28.9768]
            },
            "ayasofya": {
                "station": "Sultanahmet", "line": "T1", "walk_min": 2, "walk_m": 150,
                "coords": [41.0086, 28.9802],
                "station_coords": [41.0054, 28.9768]
            },
            "blue mosque": {
                "station": "Sultanahmet", "line": "T1", "walk_min": 3, "walk_m": 200,
                "coords": [41.0054, 28.9768],
                "station_coords": [41.0054, 28.9768]
            },
            "sultanahmet camii": {
                "station": "Sultanahmet", "line": "T1", "walk_min": 3, "walk_m": 200,
                "coords": [41.0054, 28.9768],
                "station_coords": [41.0054, 28.9768]
            },
            "süleymaniye mosque": {
                "station": "Beyazıt", "line": "T1", "walk_min": 8, "walk_m": 650,
                "coords": [41.0166, 28.9638],
                "station_coords": [41.0106, 28.9640]
            },
            "süleymaniye camii": {
                "station": "Beyazıt", "line": "T1", "walk_min": 8, "walk_m": 650,
                "coords": [41.0166, 28.9638],
                "station_coords": [41.0106, 28.9640]
            },
            
            # Modern areas
            "zorlu center": {
                "station": "Gayrettepe", "line": "M2", "walk_min": 15, "walk_m": 1200,
                "coords": [41.0685, 29.0079],
                "station_coords": [41.0646, 28.9962]
            },
            "cevahir": {
                "station": "Şişli", "line": "M2", "walk_min": 5, "walk_m": 400,
                "coords": [41.0583, 28.9869],
                "station_coords": [41.0603, 28.9872]
            },
            "istinye park": {
                "station": "4.Levent", "line": "M2", "walk_min": 20, "walk_m": 1600,
                "coords": [41.1099, 29.0251],
                "station_coords": [41.0816, 29.0089]
            },
        }
        
        # Patterns to extract routing information from LLM text
        self.patterns = {
            # "take X from Y to Z"
            'take_route': re.compile(
                r'take\s+(?:the\s+)?(\w+|\w+\s+\w+)\s+(?:from\s+)?(\w+(?:\s+\w+)?)\s+to\s+(\w+(?:\s+\w+)?)',
                re.IGNORECASE
            ),
            # "get off at X"
            'get_off': re.compile(r'get\s+off\s+at\s+(\w+(?:\s+\w+)?)', re.IGNORECASE),
            # "walk to X" or "walk about X minutes"
            'walk': re.compile(r'walk(?:\s+about|\s+for)?\s+(\d+)\s+(?:min|minutes?)', re.IGNORECASE),
            # "transfer at X"
            'transfer': re.compile(r'transfer\s+(?:at\s+)?(\w+(?:\s+\w+)?)', re.IGNORECASE),
            # Line mentions: "T1", "M2", "MARMARAY" etc
            'line': re.compile(r'\b([TM]\d+|F\d+|MARMARAY|METROBUS)\b', re.IGNORECASE),
        }
    
    def extract_route_from_llm_response(
        self,
        llm_response: str,
        user_query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract routing information from LLM response text.
        
        Args:
            llm_response: The LLM's text response
            user_query: Original user query
            user_location: User's GPS coordinates
            
        Returns:
            Dict with route_data and map_data, or None if extraction failed
        """
        try:
            logger.info(f"🔍 Extracting route from LLM response...")
            
            # Step 1: Check if this is a transportation response
            if not self._is_transportation_query(llm_response):
                logger.info("  → Not a transportation query")
                return None
            
            # Step 2: Extract destination from query
            destination = self._extract_destination(user_query)
            if not destination:
                logger.info("  → Could not extract destination")
                return None
            
            # Step 3: Check if destination is a known landmark
            landmark_info = self._get_landmark_info(destination)
            if not landmark_info:
                logger.info(f"  → '{destination}' is not a known landmark")
                return None
            
            logger.info(f"  ✅ Found landmark: {destination} → {landmark_info['station']}")
            
            # Step 4: Extract route steps from LLM response
            steps = self._extract_steps(llm_response)
            
            # Step 5: Extract origin from response or use user location
            origin = self._extract_origin(llm_response, user_location)
            
            # Step 6: Build route_data
            route_data = self._build_route_data(
                origin=origin,
                destination_station=landmark_info['station'],
                final_destination=destination,
                steps=steps,
                walking_info=landmark_info
            )
            
            # Step 7: Build map_data
            map_data = self._build_map_data(route_data, landmark_info)
            
            logger.info(f"  ✅ Successfully extracted route: {origin} → {landmark_info['station']} → {destination}")
            
            return {
                'route_data': route_data,
                'map_data': map_data
            }
            
        except Exception as e:
            logger.error(f"❌ Route extraction failed: {e}", exc_info=True)
            return None
    
    def _is_transportation_query(self, response: str) -> bool:
        """Check if response contains transportation-related keywords (multilingual)."""
        keywords = [
            # English
            'take', 'tram', 'metro', 'bus', 'ferry', 'walk', 'station', 'route', 'get to',
            # Turkish
            'tramvay', 'metro', 'otobüs', 'vapur', 'yürü', 'istasyon', 'rota', 'git',
            'alın', 'inin', 'aktarma', 'durak',
            # Common line names
            't1', 't4', 'm2', 'm5', 'marmaray', 'metrobüs'
        ]
        return any(kw in response.lower() for kw in keywords)
    
    def _extract_destination(self, query: str) -> Optional[str]:
        """Extract destination from user query (multilingual)."""
        # Patterns for English and Turkish
        patterns = [
            # English
            r'(?:how\s+(?:to|can\s+i)\s+)?(?:get|go)\s+to\s+(.+?)(?:\?|$)',
            r'directions?\s+to\s+(.+?)(?:\?|$)',
            r'route\s+to\s+(.+?)(?:\?|$)',
            r'way\s+to\s+(.+?)(?:\?|$)',
            # Turkish
            r'(?:nasıl\s+)?(?:giderim|gidebilirim|gidilir)\s+(.+?)(?:\?|$)',
            r'(.+?)(?:\'?(?:e|a|ye|ya))\s+nasıl\s+gid',
            r'yol\s+tarifi\s+(.+?)(?:\?|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                dest = match.group(1).strip().lower()
                # Clean up Turkish suffixes
                dest = re.sub(r'[\'\'](e|a|ye|ya|den|dan|de|da)$', '', dest)
                dest = re.sub(r'\s+', ' ', dest)
                return dest
        
        return None
    
    def _get_landmark_info(self, destination: str) -> Optional[Dict[str, Any]]:
        """Get landmark mapping info if it exists."""
        dest_lower = destination.lower().strip()
        return self.landmark_mappings.get(dest_lower)
    
    def _extract_steps(self, response: str) -> List[Dict[str, str]]:
        """Extract route steps from LLM response."""
        steps = []
        
        # Find all "take X from Y to Z" patterns
        for match in self.patterns['take_route'].finditer(response):
            line, from_station, to_station = match.groups()
            steps.append({
                'action': 'take',
                'line': line.upper(),
                'from': from_station,
                'to': to_station
            })
        
        # Find transfers
        for match in self.patterns['transfer'].finditer(response):
            station = match.group(1)
            if steps:
                steps[-1]['transfer_at'] = station
        
        # Find walking instructions
        for match in self.patterns['walk'].finditer(response):
            minutes = match.group(1)
            steps.append({
                'action': 'walk',
                'duration': int(minutes)
            })
        
        return steps
    
    def _extract_origin(self, response: str, user_location: Optional[Dict]) -> str:
        """Extract origin station from response or location."""
        # Try to find "from X" in response
        from_match = re.search(r'from\s+(\w+(?:\s+\w+)?)', response, re.IGNORECASE)
        if from_match:
            return from_match.group(1)
        
        # Default to common starting point
        return "Your Location"
    
    def _build_route_data(
        self,
        origin: str,
        destination_station: str,
        final_destination: str,
        steps: List[Dict],
        walking_info: Dict
    ) -> Dict[str, Any]:
        """Build structured route_data object."""
        
        # Calculate total time
        transit_time = len(steps) * 8  # ~8 min per step (rough estimate)
        walking_time = walking_info.get('walk_min', 5)
        total_time = transit_time + walking_time
        
        # Build step list
        route_steps = []
        for step in steps:
            if step['action'] == 'take':
                route_steps.append({
                    'type': 'transit',
                    'line': step['line'],
                    'from_station': step['from'],
                    'to_station': step['to'],
                    'duration': 8  # rough estimate
                })
        
        # Add walking step
        route_steps.append({
            'type': 'walk',
            'description': f"Walk {walking_info.get('walk_m', 400)}m to {final_destination}",
            'duration': walking_time,
            'distance': walking_info.get('walk_m', 400)
        })
        
        return {
            'origin': origin,
            'destination': destination_station,
            'final_destination': final_destination,
            'total_time': total_time,
            'total_distance': walking_info.get('walk_m', 400) / 1000,  # km
            'transfers': len(steps) - 1 if len(steps) > 1 else 0,
            'steps': route_steps,
            'lines_used': [s['line'] for s in steps if s['action'] == 'take'],
            'walking_required': True,
            'walking_distance': walking_info.get('walk_m', 400),
            'walking_time': walking_time
        }
    
    def _build_map_data(self, route_data: Dict, landmark_info: Dict) -> Dict[str, Any]:
        """Build map_data object for frontend visualization."""
        
        # Get coordinates from landmark_info
        landmark_coords = landmark_info.get('coords', [41.0257, 28.9744])  # Default to Galataport
        station_coords = landmark_info.get('station_coords', [41.0241, 28.9744])  # Default to Karaköy
        
        return {
            'type': 'route_with_landmark',
            'transit_route': {
                'origin': route_data['origin'],
                'destination': route_data['destination'],
                'steps': route_data['steps'][:-1]  # All except walking
            },
            'landmark': {
                'name': route_data['final_destination'],
                'station': route_data['destination'],
                'walking_distance': route_data['walking_distance'],
                'walking_time': route_data['walking_time'],
                'coordinates': landmark_coords  # Add landmark coordinates
            },
            'station': {
                'name': route_data['destination'],
                'coordinates': station_coords  # Add station coordinates
            },
            'metadata': {
                'total_time': route_data['total_time'],
                'total_distance': route_data['total_distance'],
                'transfers': route_data['transfers'],
                'lines_used': route_data['lines_used']
            }
        }


# Singleton instance
_llm_route_extractor = None


def get_llm_route_extractor() -> LLMRouteExtractor:
    """Get or create LLM route extractor singleton."""
    global _llm_route_extractor
    if _llm_route_extractor is None:
        _llm_route_extractor = LLMRouteExtractor()
        logger.info("✅ LLM Route Extractor initialized")
    return _llm_route_extractor


# Quick test
if __name__ == "__main__":
    extractor = LLMRouteExtractor()
    
    # Test case
    response = """Here's your route to Galataport! You can take the tram from 
    Sultanahmet and get off at the Karaköy station, then walk to Galataport, 
    which is just a short stroll away (about 5 minutes)."""
    
    query = "how can i go to galataport"
    
    result = extractor.extract_route_from_llm_response(response, query)
    
    if result:
        print("✅ Extraction successful!")
        print(json.dumps(result, indent=2))
    else:
        print("❌ Extraction failed")
