"""
Advanced Entity Extractor for Istanbul AI
Extracts entities (locations, cuisines, prices, dates, etc.) from Turkish/English queries
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AdvancedEntityExtractor:
    """Extract entities from Turkish/English queries"""
    
    def __init__(self):
        self.location_patterns = self._load_location_patterns()
        self.cuisine_patterns = self._load_cuisine_patterns()
        self.price_patterns = self._compile_price_patterns()
        self.attraction_types = self._load_attraction_types()
        self.transport_modes = self._load_transport_modes()
        logger.info("✅ AdvancedEntityExtractor initialized")
    
    def extract_entities(self, query: str, intent: str) -> Dict[str, Any]:
        """Extract all relevant entities from query"""
        
        entities = {
            'locations': self._extract_locations(query),
            'cuisines': self._extract_cuisines(query) if intent == 'restaurant' else [],
            'price_range': self._extract_price_range(query),
            'dates': self._extract_dates(query),
            'time': self._extract_time(query),
            'party_size': self._extract_party_size(query),
            'preferences': self._extract_preferences(query, intent),
            'attraction_type': self._extract_attraction_type(query) if intent == 'attraction' else None,
            'transport_mode': self._extract_transport_mode(query) if intent == 'transportation' else None,
            'from_location': None,
            'to_location': None,
        }
        
        # Extract from/to locations for transportation queries
        if intent in ['transportation', 'route_planning']:
            from_loc, to_loc = self._extract_from_to_locations(query)
            entities['from_location'] = from_loc
            entities['to_location'] = to_loc
        
        # Clean up None values
        return {k: v for k, v in entities.items() if v}
    
    def _load_location_patterns(self) -> Dict[str, List[str]]:
        """Load Istanbul location patterns (districts, landmarks)"""
        
        locations = {
            # Major Districts
            'Sultanahmet': [
                r'sultanahmet', r'sultan ahmet', r'sultanahmed'
            ],
            'Taksim': [
                r'taksim', r'taqsim'
            ],
            'Kadıköy': [
                r'kadıköy', r'kadikoy', r'kadiköy'
            ],
            'Beşiktaş': [
                r'beşiktaş', r'besiktas', r'beşiktas'
            ],
            'Beyoğlu': [
                r'beyoğlu', r'beyoglu', r'pera'
            ],
            'Üsküdar': [
                r'üsküdar', r'uskudar', r'üskudar'
            ],
            'Ortaköy': [
                r'ortaköy', r'ortakoy', r'ortaköy'
            ],
            'Eminönü': [
                r'eminönü', r'eminonu', r'eminönü'
            ],
            'Galata': [
                r'galata', r'karaköy', r'karakoy'
            ],
            'Fatih': [
                r'fatih'
            ],
            'Şişli': [
                r'şişli', r'sisli'
            ],
            'Bakırköy': [
                r'bakırköy', r'bakirkoy'
            ],
            'Bebek': [
                r'bebek'
            ],
            'Arnavutköy': [
                r'arnavutköy', r'arnavutkoy'
            ],
            'Balat': [
                r'balat'
            ],
            'Fener': [
                r'fener'
            ],
            
            # Major Landmarks
            'Ayasofya': [
                r'ayasofya', r'hagia sophia', r'aya sofya', r'ayasofia'
            ],
            'Blue Mosque': [
                r'blue mosque', r'sultanahmet camii', r'sultan ahmet camii'
            ],
            'Topkapı Palace': [
                r'topkapı', r'topkapi', r'topkapı saray', r'topkapi palace'
            ],
            'Grand Bazaar': [
                r'grand bazaar', r'kapalı çarşı', r'kapali carsi', r'kapalıçarşı'
            ],
            'Spice Bazaar': [
                r'spice bazaar', r'mısır çarşısı', r'egyptian bazaar'
            ],
            'Galata Tower': [
                r'galata tower', r'galata kulesi'
            ],
            'Dolmabahçe Palace': [
                r'dolmabahçe', r'dolmabahce', r'dolmabahçe saray'
            ],
            'Bosphorus': [
                r'boğaz', r'bosphorus', r'bogaz', r'bogazici'
            ],
            'Maiden Tower': [
                r'kız kulesi', r'maiden tower', r"maiden's tower"
            ],
            'Istiklal': [
                r'istiklal', r'istiklal caddesi', r'istiklal street'
            ],
            'Çamlıca': [
                r'çamlıca', r'camlica'
            ],
        }
        
        return locations
    
    def _extract_locations(self, query: str) -> List[str]:
        """
        Extract Istanbul locations
        Examples:
        - "Sultanahmet'te" -> ["Sultanahmet"]
        - "Taksim'den Kadıköy'e" -> ["Taksim", "Kadıköy"]
        - "near Galata Tower" -> ["Galata Tower"]
        """
        locations = []
        query_lower = query.lower()
        
        # Known Istanbul districts/landmarks
        for location, patterns in self.location_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    locations.append(location)
                    break  # Found this location, move to next
        
        return list(set(locations))
    
    def _load_cuisine_patterns(self) -> Dict[str, List[str]]:
        """Load cuisine type patterns"""
        
        return {
            'turkish': [
                'türk', 'turkish', 'kebap', 'kebab', 'meze', 'köfte', 
                'pide', 'lahmacun', 'döner', 'doner'
            ],
            'seafood': [
                'balık', 'fish', 'seafood', 'deniz mahsulleri', 'sea food',
                'levrek', 'çipura', 'midye', 'karides'
            ],
            'italian': [
                'italyan', 'italian', 'pizza', 'pasta', 'risotto'
            ],
            'asian': [
                'asya', 'asian', 'sushi', 'chinese', 'japanese', 'çin',
                'japon', 'thai', 'korean', 'kore'
            ],
            'vegan': [
                'vegan', 'vejetaryen', 'vegetarian', 'vejeteryan'
            ],
            'street_food': [
                'sokak lezzet', 'street food', 'kumpir', 'simit', 
                'balık ekmek', 'fish sandwich', 'kokoreç'
            ],
            'dessert': [
                'tatlı', 'dessert', 'baklava', 'künefe', 'dondurma',
                'ice cream', 'pasta', 'cake'
            ],
            'cafe': [
                'kafe', 'cafe', 'kahve', 'coffee', 'kahvaltı', 'breakfast'
            ],
            'fast_food': [
                'fast food', 'hamburger', 'burger', 'sandwich'
            ]
        }
    
    def _extract_cuisines(self, query: str) -> List[str]:
        """
        Extract cuisine types
        Examples:
        - "Balık restoranı" -> ["seafood"]
        - "vegan friendly" -> ["vegan"]
        - "kebap" -> ["turkish"]
        """
        cuisines = []
        query_lower = query.lower()
        
        for cuisine, keywords in self.cuisine_patterns.items():
            if any(kw in query_lower for kw in keywords):
                cuisines.append(cuisine)
        
        return cuisines
    
    def _compile_price_patterns(self) -> List[tuple]:
        """Compile price range patterns"""
        
        return [
            ('budget', [
                'ucuz', 'cheap', 'budget', 'ekonomik', 'uygun', 
                '저렴', 'affordable', 'inexpensive'
            ]),
            ('mid_range', [
                'orta', 'medium', 'moderate', 'reasonable'
            ]),
            ('luxury', [
                'lüks', 'luxury', 'pahalı', 'expensive', 'premium',
                'high-end', 'upscale', 'fine dining'
            ])
        ]
    
    def _extract_price_range(self, query: str) -> Optional[str]:
        """
        Extract price range
        Examples:
        - "ucuz" -> "budget"
        - "cheap hotel" -> "budget"
        - "lüks restoran" -> "luxury"
        - "₺₺₺" -> "expensive"
        """
        query_lower = query.lower()
        
        # Check for price symbols first
        if '₺' in query or '$' in query:
            count = query.count('₺') + query.count('$')
            if count == 1:
                return 'budget'
            elif count == 2:
                return 'mid_range'
            elif count >= 3:
                return 'luxury'
        
        # Check keyword patterns
        for price_level, keywords in self.price_patterns:
            if any(word in query_lower for word in keywords):
                return price_level
        
        return None
    
    def _extract_dates(self, query: str) -> Optional[str]:
        """
        Extract dates from query
        Examples:
        - "bugün" -> today's date
        - "yarın" -> tomorrow's date
        - "15 Ocak" -> "2025-01-15"
        """
        query_lower = query.lower()
        
        # Relative dates (Turkish)
        if 'bugün' in query_lower or 'today' in query_lower:
            return datetime.now().strftime('%Y-%m-%d')
        elif 'yarın' in query_lower or 'tomorrow' in query_lower:
            return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'cumartesi' in query_lower or 'saturday' in query_lower:
            # Find next Saturday
            return self._find_next_weekday(5)  # 5 = Saturday
        elif 'pazar' in query_lower or 'sunday' in query_lower:
            return self._find_next_weekday(6)
        elif 'hafta sonu' in query_lower or 'weekend' in query_lower:
            return self._find_next_weekday(5)  # Next Saturday
        
        # TODO: Add support for specific date parsing (e.g., "15 Ocak")
        
        return None
    
    def _find_next_weekday(self, target_day: int) -> str:
        """Find the next occurrence of a weekday (0=Monday, 6=Sunday)"""
        today = datetime.now()
        days_ahead = target_day - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_date = today + timedelta(days=days_ahead)
        return next_date.strftime('%Y-%m-%d')
    
    def _extract_time(self, query: str) -> Optional[str]:
        """
        Extract time from query
        Examples:
        - "akşam" -> "evening"
        - "öğlen" -> "noon"
        - "19:00" -> "19:00"
        """
        query_lower = query.lower()
        
        # Time patterns (HH:MM format)
        time_pattern = r'\b([0-2]?[0-9]):([0-5][0-9])\b'
        match = re.search(time_pattern, query)
        if match:
            return f"{match.group(1).zfill(2)}:{match.group(2)}"
        
        # Named times (Turkish)
        time_keywords = {
            'sabah': 'morning',
            'öğlen': 'noon',
            'akşam': 'evening',
            'gece': 'night',
            'morning': 'morning',
            'noon': 'noon',
            'evening': 'evening',
            'night': 'night',
        }
        
        for keyword, time_value in time_keywords.items():
            if keyword in query_lower:
                return time_value
        
        return None
    
    def _extract_party_size(self, query: str) -> Optional[int]:
        """
        Extract party size from query
        Examples:
        - "2 kişilik" -> 2
        - "for 4 people" -> 4
        - "family" -> 4 (estimate)
        """
        # Direct numbers with people indicators
        patterns = [
            r'(\d+)\s*(?:kişi|kişilik|people|person|pax)',
            r'(?:for|için)\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        
        # Keywords implying group size
        if 'aile' in query.lower() or 'family' in query.lower():
            return 4
        elif 'çift' in query.lower() or 'couple' in query.lower():
            return 2
        elif 'grup' in query.lower() or 'group' in query.lower():
            return 6
        
        return None
    
    def _load_attraction_types(self) -> Dict[str, List[str]]:
        """Load attraction type patterns"""
        
        return {
            'museum': ['müze', 'museum', 'gallery', 'galeri'],
            'mosque': ['cami', 'camii', 'mosque'],
            'palace': ['saray', 'palace', 'kasır', 'mansion'],
            'bazaar': ['çarşı', 'carsi', 'bazaar', 'market', 'pazar'],
            'tower': ['kule', 'tower'],
            'park': ['park', 'garden', 'bahçe', 'bahce'],
            'historical': ['tarihi', 'historical', 'historic', 'antik'],
            'viewpoint': ['manzara', 'view', 'seyir', 'panorama'],
        }
    
    def _extract_attraction_type(self, query: str) -> Optional[str]:
        """Extract attraction type from query"""
        query_lower = query.lower()
        
        for attr_type, keywords in self.attraction_types.items():
            if any(kw in query_lower for kw in keywords):
                return attr_type
        
        return None
    
    def _load_transport_modes(self) -> Dict[str, List[str]]:
        """Load transportation mode patterns"""
        
        return {
            'metro': ['metro', 'metrobüs', 'metrobus', 'subway'],
            'bus': ['otobüs', 'otobus', 'bus'],
            'tram': ['tramvay', 'tram'],
            'ferry': ['vapur', 'ferry', 'feribot', 'boat'],
            'taxi': ['taksi', 'taxi', 'cab'],
            'walk': ['yürüyerek', 'yuruyerek', 'walk', 'walking', 'yaya'],
            'car': ['araba', 'car', 'drive', 'driving'],
        }
    
    def _extract_transport_mode(self, query: str) -> Optional[str]:
        """Extract transportation mode from query"""
        query_lower = query.lower()
        
        for mode, keywords in self.transport_modes.items():
            if any(kw in query_lower for kw in keywords):
                return mode
        
        return None
    
    def _extract_from_to_locations(self, query: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract from and to locations for transportation/route queries
        Examples:
        - "Taksim'den Kadıköy'e" -> ("Taksim", "Kadıköy")
        - "from Sultanahmet to Galata" -> ("Sultanahmet", "Galata")
        """
        locations = self._extract_locations(query)
        
        if len(locations) == 0:
            return None, None
        elif len(locations) == 1:
            # Only destination provided
            return None, locations[0]
        elif len(locations) >= 2:
            # Find the position of each location in the query
            query_lower = query.lower()
            location_positions = []
            
            for location in locations:
                # Find the first pattern for this location that matches
                for pattern in self.location_patterns.get(location, []):
                    match = re.search(pattern, query_lower)
                    if match:
                        location_positions.append((match.start(), location))
                        break
            
            # Sort by position in query
            location_positions.sort(key=lambda x: x[0])
            
            if len(location_positions) >= 2:
                # First location in text is "from", second is "to"
                return location_positions[0][1], location_positions[1][1]
            elif len(location_positions) == 1:
                return None, location_positions[0][1]
        
        return None, None
    
    def _extract_preferences(self, query: str, intent: str) -> List[str]:
        """
        Extract user preferences from query
        Examples:
        - "çocuk dostu" -> ["family_friendly"]
        - "romantic" -> ["romantic"]
        - "quiet" -> ["quiet"]
        """
        preferences = []
        query_lower = query.lower()
        
        preference_keywords = {
            'family_friendly': ['çocuk dostu', 'family friendly', 'family', 'kids', 'children'],
            'romantic': ['romantik', 'romantic', 'couple'],
            'quiet': ['sakin', 'quiet', 'peaceful', 'huzurlu'],
            'outdoor': ['açık hava', 'outdoor', 'dış mekan', 'terrace', 'teras'],
            'indoor': ['kapalı', 'indoor'],
            'wifi': ['wifi', 'wi-fi', 'internet'],
            'parking': ['otopark', 'parking'],
            'accessible': ['engelli', 'accessible', 'wheelchair'],
            'pet_friendly': ['evcil hayvan', 'pet friendly', 'dog friendly'],
            'halal': ['helal', 'halal'],
            'alcohol_free': ['alkolsüz', 'alcohol free', 'no alcohol'],
        }
        
        for pref, keywords in preference_keywords.items():
            if any(kw in query_lower for kw in keywords):
                preferences.append(pref)
        
        return preferences


# Singleton instance
_extractor_instance = None

def get_entity_extractor() -> AdvancedEntityExtractor:
    """Get or create entity extractor singleton"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = AdvancedEntityExtractor()
    return _extractor_instance

# Alias for backward compatibility
EntityExtractor = AdvancedEntityExtractor
