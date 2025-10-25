"""
Enhanced Entity Extractor for Istanbul AI
Advanced entity extraction with support for:
- Detailed dietary restrictions
- Nuanced price level extraction
- Temporal expression understanding
- GPS coordinate extraction
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EnhancedEntityExtractor:
    """
    Enhanced entity extractor with advanced capabilities:
    - Comprehensive dietary restriction detection
    - Multi-level price detection (symbols, ranges, keywords)
    - Complex temporal expression parsing
    - GPS coordinate extraction
    """
    
    def __init__(self):
        self.location_patterns = self._load_location_patterns()
        self.cuisine_patterns = self._load_cuisine_patterns()
        self.dietary_patterns = self._compile_dietary_patterns()
        self.price_patterns = self._compile_price_patterns()
        self.temporal_patterns = self._compile_temporal_patterns()
        self.attraction_types = self._load_attraction_types()
        self.transport_modes = self._load_transport_modes()
        logger.info("✅ EnhancedEntityExtractor initialized with advanced features")
    
    def extract_entities(self, query: str, intent: str) -> Dict[str, Any]:
        """Extract all relevant entities from query with enhanced features"""
        
        entities = {
            'locations': self._extract_locations(query),
            'gps_coordinates': self._extract_gps_coordinates(query),
            'cuisines': self._extract_cuisines(query) if intent == 'restaurant' else [],
            'dietary_restrictions': self._extract_dietary_restrictions(query),
            'price_range': self._extract_price_range(query),
            'price_level': self._extract_price_level(query),
            'dates': self._extract_dates(query),
            'time': self._extract_time(query),
            'temporal_expression': self._extract_temporal_expression(query),
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
        
        # Clean up None and empty values
        return {k: v for k, v in entities.items() if v}
    
    # ===========================================
    # DIETARY RESTRICTIONS (ENHANCED)
    # ===========================================
    
    def _compile_dietary_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive dietary restriction patterns
        Support for: vegetarian, vegan, halal, kosher, gluten-free, lactose-free, nut-free, etc.
        """
        return {
            'vegetarian': {
                'keywords': ['vegetarian', 'vejetaryen', 'vejeteryan', 'et yok', 'no meat', 'veggie'],
                'exclude_keywords': ['vegan'],  # Don't confuse vegetarian with vegan
                'confidence': 0.9
            },
            'vegan': {
                'keywords': ['vegan', 'bitkisel', 'plant based', 'plant-based', 'hayvansal ürün yok', 
                           'no animal products', 'tam vejetaryen'],
                'exclude_keywords': [],
                'confidence': 0.95
            },
            'halal': {
                'keywords': ['halal', 'helal', 'halal certified', 'helal sertifikalı', 
                           'domuz yok', 'no pork', 'islamic'],
                'exclude_keywords': [],
                'confidence': 0.95
            },
            'kosher': {
                'keywords': ['kosher', 'koşer', 'jewish', 'yahudi', 'kosher certified'],
                'exclude_keywords': [],
                'confidence': 0.95
            },
            'gluten_free': {
                'keywords': ['gluten free', 'gluten-free', 'glütensiz', 'glutensiz', 
                           'no gluten', 'celiac', 'çölyak'],
                'exclude_keywords': [],
                'confidence': 0.9
            },
            'lactose_free': {
                'keywords': ['lactose free', 'lactose-free', 'laktozsuz', 'sütsüz', 
                           'dairy free', 'dairy-free', 'no dairy', 'süt ürünü yok'],
                'exclude_keywords': [],
                'confidence': 0.9
            },
            'nut_free': {
                'keywords': ['nut free', 'nut-free', 'fındıksız', 'findik yok', 'ceviz yok',
                           'no nuts', 'nut allergy', 'fındık alerjisi'],
                'exclude_keywords': [],
                'confidence': 0.9
            },
            'egg_free': {
                'keywords': ['egg free', 'egg-free', 'yumurtasız', 'yumurta yok', 
                           'no eggs', 'egg allergy'],
                'exclude_keywords': [],
                'confidence': 0.9
            },
            'seafood_free': {
                'keywords': ['seafood free', 'no seafood', 'balık yok', 'deniz ürünü yok',
                           'shellfish free', 'kabuklu yok'],
                'exclude_keywords': [],
                'confidence': 0.85
            },
            'low_carb': {
                'keywords': ['low carb', 'low-carb', 'düşük karbonhidrat', 'keto', 'ketogenic',
                           'az karbonhidrat'],
                'exclude_keywords': [],
                'confidence': 0.85
            },
            'low_sugar': {
                'keywords': ['low sugar', 'no sugar', 'sugar free', 'şekersiz', 'az şeker',
                           'diabetic', 'diyabetik'],
                'exclude_keywords': [],
                'confidence': 0.85
            },
            'organic': {
                'keywords': ['organic', 'organik', 'bio', 'doğal', 'natural'],
                'exclude_keywords': [],
                'confidence': 0.75
            }
        }
    
    def _extract_dietary_restrictions(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract dietary restrictions with confidence scores
        
        Returns:
            List of dicts with format: {'type': 'vegan', 'confidence': 0.95}
        
        Examples:
            "vegan restaurant" -> [{'type': 'vegan', 'confidence': 0.95}]
            "gluten-free and dairy-free options" -> [
                {'type': 'gluten_free', 'confidence': 0.9},
                {'type': 'lactose_free', 'confidence': 0.9}
            ]
            "helal ve vejetaryen" -> [
                {'type': 'halal', 'confidence': 0.95},
                {'type': 'vegetarian', 'confidence': 0.9}
            ]
        """
        restrictions = []
        query_lower = query.lower()
        
        for restriction_type, pattern_info in self.dietary_patterns.items():
            # Check if any keywords match
            matched = any(keyword in query_lower for keyword in pattern_info['keywords'])
            
            # Check for exclusion keywords
            excluded = any(exc in query_lower for exc in pattern_info['exclude_keywords'])
            
            if matched and not excluded:
                restrictions.append({
                    'type': restriction_type,
                    'confidence': pattern_info['confidence']
                })
        
        return restrictions
    
    # ===========================================
    # PRICE LEVEL (ENHANCED)
    # ===========================================
    
    def _compile_price_patterns(self) -> Dict[str, Any]:
        """
        Enhanced price detection with multiple levels
        """
        return {
            'symbols': {
                '₺': 'turkish_lira',
                '$': 'dollar',
                '€': 'euro',
                '£': 'pound'
            },
            'ranges': {
                'budget': {
                    'keywords': ['ucuz', 'cheap', 'budget', 'ekonomik', 'uygun fiyat', 
                               'affordable', 'inexpensive', '저렴', 'value', 'bargain'],
                    'symbols': 1,  # ₺ or $
                    'numeric_range': (0, 100)
                },
                'mid_range': {
                    'keywords': ['orta', 'medium', 'moderate', 'reasonable', 'normal fiyat',
                               'average price', 'standard'],
                    'symbols': 2,  # ₺₺ or $$
                    'numeric_range': (100, 300)
                },
                'upscale': {
                    'keywords': ['kaliteli', 'quality', 'upscale', 'nice', 'iyi'],
                    'symbols': 3,  # ₺₺₺ or $$$
                    'numeric_range': (300, 600)
                },
                'luxury': {
                    'keywords': ['lüks', 'luxury', 'pahalı', 'expensive', 'premium', 
                               'high-end', 'fine dining', 'exclusive', 'michelin'],
                    'symbols': 4,  # ₺₺₺₺ or $$$$
                    'numeric_range': (600, float('inf'))
                }
            }
        }
    
    def _extract_price_range(self, query: str) -> Optional[str]:
        """
        Extract basic price range (backward compatible)
        Returns: "budget" | "mid_range" | "upscale" | "luxury"
        """
        query_lower = query.lower()
        
        # Check for price symbols first
        symbol_count = max(
            query.count('₺'),
            query.count('$'),
            query.count('€'),
            query.count('£')
        )
        
        if symbol_count > 0:
            if symbol_count == 1:
                return 'budget'
            elif symbol_count == 2:
                return 'mid_range'
            elif symbol_count == 3:
                return 'upscale'
            elif symbol_count >= 4:
                return 'luxury'
        
        # Check keyword patterns
        for price_level, config in self.price_patterns['ranges'].items():
            if any(word in query_lower for word in config['keywords']):
                return price_level
        
        # Check for numeric price mentions
        price_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:₺|tl|lira|$|€|£)', query_lower)
        if price_match:
            try:
                price = float(price_match.group(1).replace(',', ''))
                for level, config in self.price_patterns['ranges'].items():
                    if config['numeric_range'][0] <= price < config['numeric_range'][1]:
                        return level
            except ValueError:
                pass
        
        return None
    
    def _extract_price_level(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced price level extraction with detailed information
        
        Returns:
            {
                'level': 'mid_range',
                'symbol_count': 2,
                'currency': 'turkish_lira',
                'numeric_value': 150,
                'confidence': 0.9
            }
        """
        query_lower = query.lower()
        price_info = {
            'level': None,
            'symbol_count': 0,
            'currency': None,
            'numeric_value': None,
            'confidence': 0.0
        }
        
        # Extract currency symbols
        for symbol, currency in self.price_patterns['symbols'].items():
            count = query.count(symbol)
            if count > price_info['symbol_count']:
                price_info['symbol_count'] = count
                price_info['currency'] = currency
        
        # Extract numeric price
        price_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:₺|tl|lira|$|€|£)', query_lower)
        if price_match:
            try:
                price_info['numeric_value'] = float(price_match.group(1).replace(',', ''))
                price_info['confidence'] = 0.95
            except ValueError:
                pass
        
        # Determine price level
        if price_info['symbol_count'] > 0:
            for level, config in self.price_patterns['ranges'].items():
                if config['symbols'] == price_info['symbol_count']:
                    price_info['level'] = level
                    price_info['confidence'] = max(price_info['confidence'], 0.9)
                    break
        elif price_info['numeric_value']:
            for level, config in self.price_patterns['ranges'].items():
                if config['numeric_range'][0] <= price_info['numeric_value'] < config['numeric_range'][1]:
                    price_info['level'] = level
                    break
        else:
            # Check keywords
            for level, config in self.price_patterns['ranges'].items():
                if any(word in query_lower for word in config['keywords']):
                    price_info['level'] = level
                    price_info['confidence'] = 0.75
                    break
        
        return price_info if price_info['level'] else None
    
    # ===========================================
    # TEMPORAL EXPRESSIONS (ENHANCED)
    # ===========================================
    
    def _compile_temporal_patterns(self) -> Dict[str, Any]:
        """
        Comprehensive temporal expression patterns
        """
        return {
            'relative': {
                'now': ['şimdi', 'now', 'right now', 'an', 'moment'],
                'today': ['bugün', 'today', 'bu gün'],
                'tonight': ['bu akşam', 'tonight', 'bu gece', 'this evening'],
                'tomorrow': ['yarın', 'tomorrow', 'ertesi gün'],
                'tomorrow_morning': ['yarın sabah', 'tomorrow morning'],
                'tomorrow_evening': ['yarın akşam', 'tomorrow evening'],
                'day_after_tomorrow': ['yarından sonraki gün', 'day after tomorrow', 'öbür gün'],
                'this_week': ['bu hafta', 'this week'],
                'next_week': ['gelecek hafta', 'next week', 'önümüzdeki hafta'],
                'this_weekend': ['bu hafta sonu', 'this weekend'],
                'next_weekend': ['gelecek hafta sonu', 'next weekend'],
                'this_month': ['bu ay', 'this month'],
                'next_month': ['gelecek ay', 'next month']
            },
            'weekdays': {
                'monday': ['pazartesi', 'monday'],
                'tuesday': ['salı', 'tuesday'],
                'wednesday': ['çarşamba', 'wednesday'],
                'thursday': ['perşembe', 'thursday'],
                'friday': ['cuma', 'friday'],
                'saturday': ['cumartesi', 'saturday'],
                'sunday': ['pazar', 'sunday']
            },
            'times_of_day': {
                'early_morning': ['sabah erken', 'early morning', 'şafak'],
                'morning': ['sabah', 'morning', 'sabahleyin'],
                'late_morning': ['geç sabah', 'late morning'],
                'noon': ['öğlen', 'noon', 'öğle'],
                'afternoon': ['öğleden sonra', 'afternoon', 'ikindi'],
                'early_evening': ['akşam erken', 'early evening'],
                'evening': ['akşam', 'evening', 'akşamüstü'],
                'night': ['gece', 'night'],
                'late_night': ['gece geç', 'late night', 'gece yarısı'],
                'midnight': ['gece yarısı', 'midnight']
            },
            'durations': {
                'hour': r'(\d+)\s*(?:saat|hour|hr|h)',
                'day': r'(\d+)\s*(?:gün|day|d)',
                'week': r'(\d+)\s*(?:hafta|week|wk|w)',
                'month': r'(\d+)\s*(?:ay|month|mo|m)'
            }
        }
    
    def _extract_temporal_expression(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract complex temporal expressions
        
        Returns:
            {
                'type': 'relative' | 'absolute' | 'range',
                'expression': 'tomorrow evening',
                'parsed_date': '2025-01-16',
                'parsed_time': '19:00',
                'confidence': 0.9
            }
        
        Examples:
            "yarın akşam" -> {
                'type': 'relative',
                'expression': 'tomorrow_evening',
                'parsed_date': '2025-01-16',
                'parsed_time': 'evening',
                'confidence': 0.95
            }
            "15 Ocak saat 19:00" -> {
                'type': 'absolute',
                'expression': '15 January 19:00',
                'parsed_date': '2025-01-15',
                'parsed_time': '19:00',
                'confidence': 1.0
            }
        """
        query_lower = query.lower()
        temporal_info = {
            'type': None,
            'expression': None,
            'parsed_date': None,
            'parsed_time': None,
            'confidence': 0.0
        }
        
        # Check for relative expressions
        for rel_type, keywords in self.temporal_patterns['relative'].items():
            if any(kw in query_lower for kw in keywords):
                temporal_info['type'] = 'relative'
                temporal_info['expression'] = rel_type
                temporal_info['confidence'] = 0.9
                temporal_info['parsed_date'] = self._parse_relative_date(rel_type)
                temporal_info['parsed_time'] = self._parse_relative_time(rel_type)
                break
        
        # Check for weekdays
        if not temporal_info['type']:
            for weekday, keywords in self.temporal_patterns['weekdays'].items():
                if any(kw in query_lower for kw in keywords):
                    temporal_info['type'] = 'relative'
                    temporal_info['expression'] = weekday
                    temporal_info['confidence'] = 0.85
                    temporal_info['parsed_date'] = self._find_next_weekday_from_name(weekday)
                    break
        
        # Check for times of day
        for time_of_day, keywords in self.temporal_patterns['times_of_day'].items():
            if any(kw in query_lower for kw in keywords):
                temporal_info['parsed_time'] = time_of_day
                temporal_info['confidence'] = max(temporal_info['confidence'], 0.8)
        
        # Check for absolute time (HH:MM)
        time_match = re.search(r'\b([0-2]?[0-9]):([0-5][0-9])\b', query)
        if time_match:
            temporal_info['parsed_time'] = f"{time_match.group(1).zfill(2)}:{time_match.group(2)}"
            temporal_info['confidence'] = 1.0
            if not temporal_info['type']:
                temporal_info['type'] = 'absolute'
        
        # Check for absolute dates (DD Month or DD.MM.YYYY)
        date_match = self._parse_absolute_date(query)
        if date_match:
            temporal_info['type'] = 'absolute'
            temporal_info['parsed_date'] = date_match
            temporal_info['confidence'] = 1.0
        
        return temporal_info if temporal_info['type'] else None
    
    def _parse_relative_date(self, relative_type: str) -> Optional[str]:
        """Convert relative date expression to ISO date string"""
        today = datetime.now()
        
        date_map = {
            'now': today,
            'today': today,
            'tonight': today,
            'tomorrow': today + timedelta(days=1),
            'tomorrow_morning': today + timedelta(days=1),
            'tomorrow_evening': today + timedelta(days=1),
            'day_after_tomorrow': today + timedelta(days=2),
            'this_weekend': self._find_next_saturday(),
            'next_weekend': self._find_next_saturday() + timedelta(days=7)
        }
        
        if relative_type in date_map:
            return date_map[relative_type].strftime('%Y-%m-%d')
        
        return None
    
    def _parse_relative_time(self, relative_type: str) -> Optional[str]:
        """Extract time component from relative expression"""
        time_map = {
            'tonight': 'evening',
            'tomorrow_morning': 'morning',
            'tomorrow_evening': 'evening'
        }
        return time_map.get(relative_type)
    
    def _find_next_saturday(self) -> datetime:
        """Find the next Saturday"""
        today = datetime.now()
        days_ahead = 5 - today.weekday()  # 5 = Saturday
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def _find_next_weekday_from_name(self, weekday_name: str) -> str:
        """Find next occurrence of a weekday"""
        weekday_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        target_day = weekday_map.get(weekday_name)
        if target_day is None:
            return None
        
        today = datetime.now()
        days_ahead = target_day - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_date = today + timedelta(days=days_ahead)
        return next_date.strftime('%Y-%m-%d')
    
    def _parse_absolute_date(self, query: str) -> Optional[str]:
        """
        Parse absolute dates from query
        Supports formats: DD.MM.YYYY, DD/MM/YYYY, DD Month, Month DD
        """
        # Turkish month names
        turkish_months = {
            'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4,
            'mayıs': 5, 'haziran': 6, 'temmuz': 7, 'ağustos': 8,
            'eylül': 9, 'ekim': 10, 'kasım': 11, 'aralık': 12
        }
        
        # English month names
        english_months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
            'oct': 10, 'nov': 11, 'dec': 12
        }
        
        query_lower = query.lower()
        
        # Try DD.MM.YYYY or DD/MM/YYYY
        date_pattern = r'\b(\d{1,2})[\./](\d{1,2})[\./](\d{4})\b'
        match = re.search(date_pattern, query)
        if match:
            day, month, year = match.groups()
            try:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except ValueError:
                pass
        
        # Try DD Month (Turkish)
        for month_name, month_num in turkish_months.items():
            pattern = rf'\b(\d{{1,2}})\s+{month_name}\b'
            match = re.search(pattern, query_lower)
            if match:
                day = match.group(1)
                year = datetime.now().year
                try:
                    return f"{year}-{str(month_num).zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    pass
        
        # Try DD Month (English)
        for month_name, month_num in english_months.items():
            pattern = rf'\b(\d{{1,2}})\s+{month_name}\b'
            match = re.search(pattern, query_lower)
            if match:
                day = match.group(1)
                year = datetime.now().year
                try:
                    return f"{year}-{str(month_num).zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    pass
        
        return None
    
    # ===========================================
    # GPS COORDINATES (NEW)
    # ===========================================
    
    def _extract_gps_coordinates(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract GPS coordinates from query
        
        Supports formats:
        - Decimal: 41.0082, 28.9784
        - Degrees: 41°00'29.5"N 28°58'42.2"E
        - Google Maps links
        
        Returns:
            {
                'latitude': 41.0082,
                'longitude': 28.9784,
                'format': 'decimal',
                'confidence': 1.0
            }
        
        Examples:
            "41.0082, 28.9784" -> {'latitude': 41.0082, 'longitude': 28.9784, 'format': 'decimal'}
            "at coordinates 41.0082N 28.9784E" -> same
            "https://maps.google.com/?q=41.0082,28.9784" -> same
        """
        # Decimal coordinates (most common)
        decimal_pattern = r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)'
        match = re.search(decimal_pattern, query)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'format': 'decimal',
                    'confidence': 1.0
                }
        
        # Decimal with N/S/E/W indicators
        decimal_nsew_pattern = r'(\d+\.\d+)\s*([NS])\s*,?\s*(\d+\.\d+)\s*([EW])'
        match = re.search(decimal_nsew_pattern, query, re.IGNORECASE)
        if match:
            lat = float(match.group(1)) * (1 if match.group(2).upper() == 'N' else -1)
            lon = float(match.group(3)) * (1 if match.group(4).upper() == 'E' else -1)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'format': 'decimal_nsew',
                    'confidence': 1.0
                }
        
        # Degrees, minutes, seconds (DMS)
        dms_pattern = r'(\d+)°(\d+)\'([\d.]+)"([NS])\s*,?\s*(\d+)°(\d+)\'([\d.]+)"([EW])'
        match = re.search(dms_pattern, query, re.IGNORECASE)
        if match:
            lat_deg, lat_min, lat_sec, lat_dir = float(match.group(1)), float(match.group(2)), float(match.group(3)), match.group(4)
            lon_deg, lon_min, lon_sec, lon_dir = float(match.group(5)), float(match.group(6)), float(match.group(7)), match.group(8)
            
            # Convert to decimal
            lat = lat_deg + lat_min/60 + lat_sec/3600
            lon = lon_deg + lon_min/60 + lon_sec/3600
            
            if lat_dir.upper() == 'S':
                lat = -lat
            if lon_dir.upper() == 'W':
                lon = -lon
            
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'format': 'dms',
                    'confidence': 1.0
                }
        
        # Google Maps links
        gmaps_pattern = r'(?:maps\.google\.com|google\.com/maps).*?[?&](?:q|ll)=(-?\d+\.\d+),(-?\d+\.\d+)'
        match = re.search(gmaps_pattern, query)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'format': 'google_maps',
                    'confidence': 1.0
                }
        
        return None
    
    # ===========================================
    # EXISTING METHODS (FROM ORIGINAL)
    # ===========================================
    
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
        """Extract Istanbul locations"""
        locations = []
        query_lower = query.lower()
        
        for location, patterns in self.location_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    locations.append(location)
                    break
        
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
        """Extract cuisine types"""
        cuisines = []
        query_lower = query.lower()
        
        for cuisine, keywords in self.cuisine_patterns.items():
            if any(kw in query_lower for kw in keywords):
                cuisines.append(cuisine)
        
        return cuisines
    
    def _extract_dates(self, query: str) -> Optional[str]:
        """Extract basic date (backward compatible)"""
        query_lower = query.lower()
        
        if 'bugün' in query_lower or 'today' in query_lower:
            return datetime.now().strftime('%Y-%m-%d')
        elif 'yarın' in query_lower or 'tomorrow' in query_lower:
            return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'hafta sonu' in query_lower or 'weekend' in query_lower:
            return self._find_next_saturday().strftime('%Y-%m-%d')
        
        # Try absolute date parsing
        absolute_date = self._parse_absolute_date(query)
        if absolute_date:
            return absolute_date
        
        return None
    
    def _extract_time(self, query: str) -> Optional[str]:
        """Extract time from query"""
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
        """Extract party size from query"""
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
    
    def _extract_from_to_locations(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract from and to locations for transportation/route queries"""
        locations = self._extract_locations(query)
        
        if len(locations) == 0:
            return None, None
        elif len(locations) == 1:
            return None, locations[0]
        elif len(locations) >= 2:
            query_lower = query.lower()
            location_positions = []
            
            for location in locations:
                for pattern in self.location_patterns.get(location, []):
                    match = re.search(pattern, query_lower)
                    if match:
                        location_positions.append((match.start(), location))
                        break
            
            location_positions.sort(key=lambda x: x[0])
            
            if len(location_positions) >= 2:
                return location_positions[0][1], location_positions[1][1]
            elif len(location_positions) == 1:
                return None, location_positions[0][1]
        
        return None, None
    
    def _extract_preferences(self, query: str, intent: str) -> List[str]:
        """Extract user preferences from query"""
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
_enhanced_extractor_instance = None

def get_enhanced_entity_extractor() -> EnhancedEntityExtractor:
    """Get or create enhanced entity extractor singleton"""
    global _enhanced_extractor_instance
    if _enhanced_extractor_instance is None:
        _enhanced_extractor_instance = EnhancedEntityExtractor()
    return _enhanced_extractor_instance
