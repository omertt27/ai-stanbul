"""
Restaurant Query Handler with Map Location Support

Identifies restaurant/food queries and extracts location information
for displaying restaurants on a map.

Author: AI Istanbul Team
Date: December 26, 2025
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RestaurantQuery:
    """Represents a parsed restaurant query with map display info"""
    is_restaurant_query: bool
    cuisine_type: Optional[str]
    locations: List[str]
    map_center: Optional[Dict[str, float]]  # lat, lng for map center
    map_zoom: int
    search_radius_km: float
    original_query: str
    language: str


class RestaurantQueryHandler:
    """
    Handles restaurant/food queries and extracts location info for map display.
    
    Features:
    - Detects restaurant-related queries in 5 languages
    - Extracts cuisine type (kebab, fish, breakfast, etc.)
    - Extracts location for map centering
    - Provides coordinates for Istanbul neighborhoods
    """
    
    def __init__(self):
        self.location_coordinates = self._build_location_coordinates()
        self.cuisine_patterns = self._build_cuisine_patterns()
        self.restaurant_patterns = self._build_restaurant_patterns()
    
    def _build_location_coordinates(self) -> Dict[str, Dict[str, float]]:
        """Istanbul neighborhood coordinates for map centering"""
        return {
            # European Side - Historic
            'sultanahmet': {'lat': 41.0054, 'lng': 28.9768, 'zoom': 15},
            'السلطان أحمد': {'lat': 41.0054, 'lng': 28.9768, 'zoom': 15},  # Arabic
            'sultan ahmet': {'lat': 41.0054, 'lng': 28.9768, 'zoom': 15},  # Alternative
            'eminonu': {'lat': 41.0172, 'lng': 28.9674, 'zoom': 15},
            'eminönü': {'lat': 41.0172, 'lng': 28.9674, 'zoom': 15},
            'أمينونو': {'lat': 41.0172, 'lng': 28.9674, 'zoom': 15},  # Arabic
            'sirkeci': {'lat': 41.0144, 'lng': 28.9778, 'zoom': 15},
            'fatih': {'lat': 41.0186, 'lng': 28.9498, 'zoom': 14},
            'الفاتح': {'lat': 41.0186, 'lng': 28.9498, 'zoom': 14},  # Arabic
            'balat': {'lat': 41.0297, 'lng': 28.9483, 'zoom': 15},
            'fener': {'lat': 41.0283, 'lng': 28.9517, 'zoom': 15},
            
            # European Side - Beyoglu/Modern
            'taksim': {'lat': 41.0370, 'lng': 28.9850, 'zoom': 15},
            'تقسيم': {'lat': 41.0370, 'lng': 28.9850, 'zoom': 15},  # Arabic
            'beyoglu': {'lat': 41.0322, 'lng': 28.9770, 'zoom': 14},
            'beyoğlu': {'lat': 41.0322, 'lng': 28.9770, 'zoom': 14},
            'بيوغلو': {'lat': 41.0322, 'lng': 28.9770, 'zoom': 14},  # Arabic
            'galata': {'lat': 41.0256, 'lng': 28.9742, 'zoom': 15},
            'غلطة': {'lat': 41.0256, 'lng': 28.9742, 'zoom': 15},  # Arabic
            'karakoy': {'lat': 41.0220, 'lng': 28.9770, 'zoom': 15},
            'karaköy': {'lat': 41.0220, 'lng': 28.9770, 'zoom': 15},
            'كاراكوي': {'lat': 41.0220, 'lng': 28.9770, 'zoom': 15},  # Arabic
            'cihangir': {'lat': 41.0317, 'lng': 28.9833, 'zoom': 16},
            'istiklal': {'lat': 41.0340, 'lng': 28.9780, 'zoom': 16},
            'الاستقلال': {'lat': 41.0340, 'lng': 28.9780, 'zoom': 16},  # Arabic
            
            # European Side - Bosphorus
            'besiktas': {'lat': 41.0430, 'lng': 29.0070, 'zoom': 15},
            'beşiktaş': {'lat': 41.0430, 'lng': 29.0070, 'zoom': 15},
            'بشكتاش': {'lat': 41.0430, 'lng': 29.0070, 'zoom': 15},  # Arabic
            'ortakoy': {'lat': 41.0477, 'lng': 29.0267, 'zoom': 15},
            'ortaköy': {'lat': 41.0477, 'lng': 29.0267, 'zoom': 15},
            'اورتاكوي': {'lat': 41.0477, 'lng': 29.0267, 'zoom': 15},  # Arabic
            'bebek': {'lat': 41.0767, 'lng': 29.0433, 'zoom': 15},
            'arnavutkoy': {'lat': 41.0683, 'lng': 29.0350, 'zoom': 15},
            'arnavutköy': {'lat': 41.0683, 'lng': 29.0350, 'zoom': 15},
            'sariyer': {'lat': 41.1667, 'lng': 29.0500, 'zoom': 14},
            'sarıyer': {'lat': 41.1667, 'lng': 29.0500, 'zoom': 14},
            
            # European Side - Business
            'levent': {'lat': 41.0819, 'lng': 29.0117, 'zoom': 15},
            'sisli': {'lat': 41.0600, 'lng': 28.9870, 'zoom': 14},
            'şişli': {'lat': 41.0600, 'lng': 28.9870, 'zoom': 14},
            'شيشلي': {'lat': 41.0600, 'lng': 28.9870, 'zoom': 14},  # Arabic
            'mecidiyekoy': {'lat': 41.0678, 'lng': 28.9958, 'zoom': 15},
            'mecidiyeköy': {'lat': 41.0678, 'lng': 28.9958, 'zoom': 15},
            'maslak': {'lat': 41.1089, 'lng': 29.0200, 'zoom': 14},
            
            # Asian Side
            'kadikoy': {'lat': 40.9819, 'lng': 29.0267, 'zoom': 15},
            'kadıköy': {'lat': 40.9819, 'lng': 29.0267, 'zoom': 15},
            'كاديكوي': {'lat': 40.9819, 'lng': 29.0267, 'zoom': 15},  # Arabic
            'moda': {'lat': 40.9833, 'lng': 29.0333, 'zoom': 15},
            'uskudar': {'lat': 41.0256, 'lng': 29.0153, 'zoom': 15},
            'üsküdar': {'lat': 41.0256, 'lng': 29.0153, 'zoom': 15},
            'اسكودار': {'lat': 41.0256, 'lng': 29.0153, 'zoom': 15},  # Arabic
            'cengelkoy': {'lat': 41.0458, 'lng': 29.0517, 'zoom': 15},
            'çengelköy': {'lat': 41.0458, 'lng': 29.0517, 'zoom': 15},
            'kuzguncuk': {'lat': 41.0350, 'lng': 29.0317, 'zoom': 15},
            'bostanci': {'lat': 40.9583, 'lng': 29.0917, 'zoom': 15},
            'bostancı': {'lat': 40.9583, 'lng': 29.0917, 'zoom': 15},
            'bagdat caddesi': {'lat': 40.9667, 'lng': 29.0667, 'zoom': 14},
            
            # Landmarks
            'grand bazaar': {'lat': 41.0106, 'lng': 28.9680, 'zoom': 16},
            'kapalicarsi': {'lat': 41.0106, 'lng': 28.9680, 'zoom': 16},
            'kapalıçarşı': {'lat': 41.0106, 'lng': 28.9680, 'zoom': 16},
            'البازار الكبير': {'lat': 41.0106, 'lng': 28.9680, 'zoom': 16},  # Arabic
            'spice market': {'lat': 41.0167, 'lng': 28.9703, 'zoom': 16},
            'misir carsisi': {'lat': 41.0167, 'lng': 28.9703, 'zoom': 16},
            'سوق التوابل': {'lat': 41.0167, 'lng': 28.9703, 'zoom': 16},  # Arabic
            'galata tower': {'lat': 41.0256, 'lng': 28.9742, 'zoom': 16},
            'برج غلطة': {'lat': 41.0256, 'lng': 28.9742, 'zoom': 16},  # Arabic
            'blue mosque': {'lat': 41.0054, 'lng': 28.9768, 'zoom': 16},
            'المسجد الأزرق': {'lat': 41.0054, 'lng': 28.9768, 'zoom': 16},  # Arabic
            'hagia sophia': {'lat': 41.0086, 'lng': 28.9802, 'zoom': 16},
            'ayasofya': {'lat': 41.0086, 'lng': 28.9802, 'zoom': 16},
            'آيا صوفيا': {'lat': 41.0086, 'lng': 28.9802, 'zoom': 16},  # Arabic
            'topkapi': {'lat': 41.0115, 'lng': 28.9833, 'zoom': 15},
            'topkapı': {'lat': 41.0115, 'lng': 28.9833, 'zoom': 15},
            'توبكابي': {'lat': 41.0115, 'lng': 28.9833, 'zoom': 15},  # Arabic
            
            # Water features
            'bosphorus': {'lat': 41.0700, 'lng': 29.0500, 'zoom': 12},
            'boğaz': {'lat': 41.0700, 'lng': 29.0500, 'zoom': 12},
            'البوسفور': {'lat': 41.0700, 'lng': 29.0500, 'zoom': 12},  # Arabic
            
            # Default Istanbul center
            'istanbul': {'lat': 41.0082, 'lng': 28.9784, 'zoom': 12},
            'اسطنبول': {'lat': 41.0082, 'lng': 28.9784, 'zoom': 12},  # Arabic
        }
    
    def _build_cuisine_patterns(self) -> Dict[str, List[str]]:
        """Cuisine type detection patterns by language"""
        return {
            'kebab': [
                r'\b(kebab|kebap|döner|doner|iskender|adana|urfa)\b',
                r'\b(kebabci|kebapçı|kebapci)\b',
            ],
            'fish': [
                r'\b(fish|seafood|balik|balık|deniz\s*ürünleri|fruits?\s*de\s*mer|poisson|fisch|سمك|مأكولات\s*بحرية)\b',
            ],
            'breakfast': [
                r'\b(breakfast|kahvalti|kahvaltı|frühstück|petit[\s-]*déjeuner|فطور|إفطار)\b',
                r'\b(serpme|köy\s*kahvaltısı)\b',
            ],
            'turkish': [
                r'\b(turkish|turk|türk|turc|türkisch|تركي)\b',
                r'\b(meze|lahmacun|pide|börek|borek|mantı|manti|kofte|köfte)\b',
            ],
            'cafe': [
                r'\b(cafe|café|kafe|kahve|coffee|kaffee|قهوة|مقهى)\b',
            ],
            'rooftop': [
                r'\b(rooftop|teras|terrace|terrasse|سطح)\b',
                r'\b(view|manzara|aussicht|vue|إطلالة|منظر)\b',
            ],
            'fine_dining': [
                r'\b(fine\s*dining|upscale|luxury|michelin|lüks)\b',
            ],
        }
    
    def _build_restaurant_patterns(self) -> Dict[str, List[str]]:
        """Restaurant query detection patterns by language"""
        return {
            'english': [
                r'\b(restaurant|restaurants|cafe|cafes|bar|bars|bistro|eatery|dining)\b',
                r'\b(where\s+can\s+i\s+(eat|find|get))\b',
                r'\b(best|good|top|recommended)\s+.*(food|place\s+to\s+eat)\b',
                r'\b(looking\s+for).*(restaurant|food|eat)\b',
            ],
            'turkish': [
                r'\b(restoran|lokanta|kafe|meyhane|ocakbaşı|ocakbasi)\b',
                r'\b(yemek\s+yiyebil|nerede\s+yerim|yemek\s+için)\b',
                r'\b(en\s+iyi|güzel|iyi)\s+(restoran|lokanta|yemek)\b',
            ],
            'german': [
                r'\b(restaurant|restaurants|café|cafés|lokal|gaststätte|gasthaus)\b',
                r'\b(wo\s+kann\s+ich\s+essen)\b',
                r'\b(wo\s+finde\s+ich)\b',  # "Wo finde ich"
                r'\b(beste[sr]?|gute[sr]?)\s+.*(restaurant|essen|küche)\b',
                r'\b(fischrestaurant|türkisches\s+essen|türkisch)\b',  # Specific German food terms
            ],
            'french': [
                r'\b(restaurant|restaurants|café|cafés|bistrot|brasserie)\b',
                r'\b(où\s+(manger|trouver|puis-je))\b',
                r'\b(meilleur|bon|bonne)\s+.*(restaurant|cuisine)\b',
            ],
            'arabic': [
                r'\b(مطعم|مطاعم|مقهى|مقاهي|كافيه)\b',
                r'\b(أين\s+(أجد|يمكن|أكل))\b',
                r'\b(أفضل|أحسن)\s+.*(مطعم|طعام|أكل)\b',
            ],
        }
    
    def detect_language(self, query: str) -> str:
        """Detect query language"""
        # Arabic detection
        if re.search(r'[\u0600-\u06FF]', query):
            return 'arabic'
        
        # Turkish-specific characters
        if re.search(r'[ğĞıİöÖüÜşŞçÇ]', query):
            return 'turkish'
        
        # German-specific patterns
        if re.search(r'\b(wo|wie|welche|können|gibt)\b', query.lower()):
            return 'german'
        
        # French-specific patterns
        if re.search(r'\b(où|quel|quelle|puis|pouvez|meilleur)\b', query.lower()):
            return 'french'
        
        return 'english'
    
    def is_restaurant_query(self, query: str) -> bool:
        """Check if query is restaurant-related"""
        query_lower = query.lower()
        
        for lang_patterns in self.restaurant_patterns.values():
            for pattern in lang_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return True
        
        # Also check cuisine patterns
        for cuisine_patterns in self.cuisine_patterns.values():
            for pattern in cuisine_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return True
        
        return False
    
    def extract_cuisine_type(self, query: str) -> Optional[str]:
        """Extract cuisine type from query"""
        query_lower = query.lower()
        
        for cuisine_type, patterns in self.cuisine_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return cuisine_type
        
        return None
    
    def extract_locations(self, query: str) -> List[str]:
        """Extract location names from query"""
        query_lower = query.lower()
        found_locations = []
        
        for location in self.location_coordinates.keys():
            # Check for location in query (handle Turkish characters)
            location_pattern = re.escape(location)
            if re.search(location_pattern, query_lower, re.IGNORECASE):
                # Normalize to base name
                normalized = location.replace('ö', 'o').replace('ü', 'u').replace('ş', 's').replace('ı', 'i').replace('ğ', 'g').replace('ç', 'c')
                if normalized not in [l.replace('ö', 'o').replace('ü', 'u').replace('ş', 's').replace('ı', 'i').replace('ğ', 'g').replace('ç', 'c') for l in found_locations]:
                    found_locations.append(location.title())
        
        return found_locations
    
    def get_map_center(self, locations: List[str]) -> Tuple[Optional[Dict[str, float]], int]:
        """
        Get map center coordinates and zoom level for given locations.
        Returns (center_coords, zoom_level)
        """
        if not locations:
            # Default to Istanbul center
            return {'lat': 41.0082, 'lng': 28.9784}, 12
        
        # Use first location for centering
        location_key = locations[0].lower()
        
        # Try to find in coordinates
        for key, coords in self.location_coordinates.items():
            if key == location_key or key.replace('ö', 'o').replace('ü', 'u').replace('ş', 's').replace('ı', 'i') == location_key.replace('ö', 'o').replace('ü', 'u').replace('ş', 's').replace('ı', 'i'):
                return {'lat': coords['lat'], 'lng': coords['lng']}, coords.get('zoom', 15)
        
        # Default
        return {'lat': 41.0082, 'lng': 28.9784}, 12
    
    def parse_query(self, query: str) -> RestaurantQuery:
        """
        Parse a restaurant query and extract all relevant information.
        
        Returns RestaurantQuery with:
        - is_restaurant_query: bool
        - cuisine_type: Optional[str]
        - locations: List[str]
        - map_center: coordinates for map
        - map_zoom: zoom level
        - search_radius_km: search radius
        """
        is_restaurant = self.is_restaurant_query(query)
        cuisine = self.extract_cuisine_type(query)
        locations = self.extract_locations(query)
        language = self.detect_language(query)
        map_center, zoom = self.get_map_center(locations)
        
        # Determine search radius based on location specificity
        if locations and len(locations) == 1:
            radius = 1.0  # 1km for specific neighborhood
        elif locations:
            radius = 2.0  # 2km for multiple locations
        else:
            radius = 5.0  # 5km for general Istanbul search
        
        return RestaurantQuery(
            is_restaurant_query=is_restaurant,
            cuisine_type=cuisine,
            locations=locations,
            map_center=map_center,
            map_zoom=zoom,
            search_radius_km=radius,
            original_query=query,
            language=language
        )
    
    def get_map_display_config(self, query: str) -> Dict:
        """
        Get map display configuration for a restaurant query.
        
        Returns dict with:
        - show_map: bool
        - center: {lat, lng}
        - zoom: int
        - search_radius: float
        - markers: list of location markers
        - cuisine_filter: str or None
        """
        parsed = self.parse_query(query)
        
        if not parsed.is_restaurant_query:
            return {'show_map': False}
        
        # Build marker list for extracted locations
        markers = []
        for loc in parsed.locations:
            loc_lower = loc.lower()
            for key, coords in self.location_coordinates.items():
                if key == loc_lower or key.replace('ö', 'o').replace('ü', 'u').replace('ş', 's').replace('ı', 'i') == loc_lower.replace('ö', 'o').replace('ü', 'u').replace('ş', 's').replace('ı', 'i'):
                    markers.append({
                        'name': loc,
                        'lat': coords['lat'],
                        'lng': coords['lng'],
                        'type': 'area'
                    })
                    break
        
        return {
            'show_map': True,
            'center': parsed.map_center,
            'zoom': parsed.map_zoom,
            'search_radius_km': parsed.search_radius_km,
            'markers': markers,
            'cuisine_filter': parsed.cuisine_type,
            'locations': parsed.locations,
            'language': parsed.language,
            'query_type': 'restaurant'
        }


# Singleton instance
_restaurant_handler = None

def get_restaurant_query_handler() -> RestaurantQueryHandler:
    """Get or create restaurant query handler singleton"""
    global _restaurant_handler
    if _restaurant_handler is None:
        _restaurant_handler = RestaurantQueryHandler()
    return _restaurant_handler


# Quick test
if __name__ == "__main__":
    handler = RestaurantQueryHandler()
    
    test_queries = [
        "Where can I find the best kebab restaurant in Sultanahmet?",
        "Recommend a good seafood restaurant near Karakoy with Bosphorus view",
        "Besiktas'ta en iyi balik restorani hangisi?",
        "Wo finde ich das beste türkische Restaurant in Taksim?",
        "Où puis-je trouver un bon restaurant de kebab à Sultanahmet?",
        "أين أجد أفضل المطاعم في السلطان أحمد؟",
    ]
    
    print("=" * 70)
    print("🍽️  Restaurant Query Handler - Map Display Test")
    print("=" * 70)
    
    for query in test_queries:
        config = handler.get_map_display_config(query)
        parsed = handler.parse_query(query)
        
        print(f"\nQuery: \"{query[:50]}...\"")
        print(f"  Language: {parsed.language}")
        print(f"  Is Restaurant Query: {parsed.is_restaurant_query}")
        print(f"  Cuisine Type: {parsed.cuisine_type or 'General'}")
        print(f"  Locations: {parsed.locations or ['Istanbul (general)']}")
        print(f"  Map Center: {config.get('center', 'N/A')}")
        print(f"  Map Zoom: {config.get('zoom', 'N/A')}")
        print(f"  Search Radius: {config.get('search_radius_km', 'N/A')} km")
