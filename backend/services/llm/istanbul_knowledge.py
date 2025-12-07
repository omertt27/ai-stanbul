"""
Istanbul-Specific Domain Knowledge Database
Phase 4 - Priority 2: Istanbul-Specific Intelligence

This module provides domain knowledge about Istanbul landmarks, neighborhoods,
transportation terms, and local context to improve signal detection accuracy.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Landmark:
    """Istanbul landmark with metadata."""
    name: str
    synonyms: List[str]
    signal: str  # Which signal to trigger
    location: Tuple[float, float]  # (lat, lon)
    nearby_transport: List[str]
    description: str
    category: str  # 'mosque', 'palace', 'tower', 'museum', etc.


@dataclass
class Neighborhood:
    """Istanbul neighborhood with characteristics."""
    name: str
    character: str
    signals: List[str]
    popular_for: List[str]
    transport_hubs: List[str]
    side: str  # 'european', 'asian', or 'both'


class IstanbulKnowledge:
    """
    Comprehensive Istanbul domain knowledge database.
    
    Provides:
    - Landmark recognition and context
    - Neighborhood information
    - Local transportation terminology
    - Cultural context
    """
    
    def __init__(self):
        """Initialize Istanbul knowledge base."""
        self.landmarks = self._init_landmarks()
        self.neighborhoods = self._init_neighborhoods()
        self.transport_terms = self._init_transport_terms()
        self.local_slang = self._init_local_slang()
        self.poi_categories = self._init_poi_categories()
        
        logger.info(
            f"✅ Istanbul Knowledge initialized: "
            f"{len(self.landmarks)} landmarks, "
            f"{len(self.neighborhoods)} neighborhoods"
        )
    
    def _init_landmarks(self) -> Dict[str, Landmark]:
        """Initialize landmark database."""
        landmarks = {
            # === Major Mosques ===
            'Blue Mosque': Landmark(
                name='Blue Mosque',
                synonyms=[
                    'Sultanahmet Mosque', 'Sultan Ahmed Mosque',
                    'Sultanahmet Camii', 'Mavi Cami'
                ],
                signal='needs_attraction',
                location=(41.0054, 28.9768),
                nearby_transport=['Sultanahmet Tram', 'T1 Tram'],
                description='Historic mosque with blue tiles',
                category='mosque'
            ),
            
            'Hagia Sophia': Landmark(
                name='Hagia Sophia',
                synonyms=[
                    'Aya Sofya', 'Ayasofya', 'Saint Sophia',
                    'Sancta Sophia'
                ],
                signal='needs_attraction',
                location=(41.0086, 28.9802),
                nearby_transport=['Sultanahmet Tram', 'T1 Tram'],
                description='Byzantine cathedral turned mosque',
                category='mosque'
            ),
            
            'Süleymaniye Mosque': Landmark(
                name='Süleymaniye Mosque',
                synonyms=[
                    'Suleymaniye Camii', 'Süleymaniye Camii'
                ],
                signal='needs_attraction',
                location=(41.0167, 28.9644),
                nearby_transport=['Eminönü Ferry', 'T1 Tram'],
                description='Ottoman imperial mosque',
                category='mosque'
            ),
            
            # === Palaces ===
            'Topkapi Palace': Landmark(
                name='Topkapi Palace',
                synonyms=[
                    'Topkapi Sarayi', 'Topkapı Sarayı',
                    'Topkapi Museum'
                ],
                signal='needs_attraction',
                location=(41.0115, 28.9833),
                nearby_transport=['Gülhane Tram', 'T1 Tram'],
                description='Ottoman palace and museum',
                category='palace'
            ),
            
            'Dolmabahçe Palace': Landmark(
                name='Dolmabahçe Palace',
                synonyms=[
                    'Dolmabahce Sarayi', 'Dolmabahçe Sarayı'
                ],
                signal='needs_attraction',
                location=(41.0391, 29.0000),
                nearby_transport=['Kabataş Tram', 'Kabataş Ferry'],
                description='Ottoman palace on Bosphorus',
                category='palace'
            ),
            
            # === Towers & Landmarks ===
            'Galata Tower': Landmark(
                name='Galata Tower',
                synonyms=[
                    'Galata Kulesi', 'Christea Turris'
                ],
                signal='needs_attraction',
                location=(41.0256, 28.9744),
                nearby_transport=['Karaköy Metro', 'Tünel Funicular'],
                description='Medieval stone tower with views',
                category='tower'
            ),
            
            'Maiden\'s Tower': Landmark(
                name='Maiden\'s Tower',
                synonyms=[
                    'Kız Kulesi', 'Kiz Kulesi', 'Leander\'s Tower'
                ],
                signal='needs_attraction',
                location=(41.0211, 29.0041),
                nearby_transport=['Üsküdar Ferry', 'Salacak Pier'],
                description='Tower on small Bosphorus island',
                category='tower'
            ),
            
            # === Markets & Bazaars ===
            'Grand Bazaar': Landmark(
                name='Grand Bazaar',
                synonyms=[
                    'Kapalı Çarşı', 'Kapali Carsi', 'Covered Market',
                    'Great Bazaar'
                ],
                signal='needs_shopping',
                location=(41.0108, 28.9680),
                nearby_transport=['Beyazıt Tram', 'T1 Tram'],
                description='Historic covered market',
                category='market'
            ),
            
            'Spice Bazaar': Landmark(
                name='Spice Bazaar',
                synonyms=[
                    'Egyptian Bazaar', 'Mısır Çarşısı', 'Misir Carsisi'
                ],
                signal='needs_shopping',
                location=(41.0166, 28.9703),
                nearby_transport=['Eminönü Ferry', 'T1 Tram'],
                description='Historic spice market',
                category='market'
            ),
            
            # === Squares ===
            'Taksim Square': Landmark(
                name='Taksim Square',
                synonyms=[
                    'Taksim Meydanı', 'Taksim Meydani'
                ],
                signal='needs_neighborhood',
                location=(41.0369, 28.9850),
                nearby_transport=['Taksim Metro', 'M2 Metro'],
                description='Central square and transport hub',
                category='square'
            ),
            
            # === Streets ===
            'Istiklal Street': Landmark(
                name='Istiklal Street',
                synonyms=[
                    'İstiklal Caddesi', 'Istiklal Caddesi',
                    'Independence Avenue'
                ],
                signal='needs_neighborhood',
                location=(41.0346, 28.9784),
                nearby_transport=['Taksim Metro', 'Nostalgic Tram'],
                description='Famous pedestrian shopping street',
                category='street'
            ),
            
            # === Waterfront ===
            'Bosphorus': Landmark(
                name='Bosphorus',
                synonyms=[
                    'Boğaziçi', 'Bogazici', 'Istanbul Strait',
                    'Bosporus'
                ],
                signal='needs_attraction',
                location=(41.1171, 29.0763),
                nearby_transport=['Various ferries', 'Bosphorus cruise'],
                description='Strait separating Europe and Asia',
                category='waterway'
            ),
            
            'Golden Horn': Landmark(
                name='Golden Horn',
                synonyms=[
                    'Haliç', 'Halic'
                ],
                signal='needs_attraction',
                location=(41.0319, 28.9504),
                nearby_transport=['Haliç Ferry', 'Golden Horn Metro'],
                description='Historic inlet of Bosphorus',
                category='waterway'
            ),
        }
        
        return landmarks
    
    def _init_neighborhoods(self) -> Dict[str, Neighborhood]:
        """Initialize neighborhood database."""
        neighborhoods = {
            'Sultanahmet': Neighborhood(
                name='Sultanahmet',
                character='Historic old city with major landmarks',
                signals=['needs_attraction', 'needs_neighborhood'],
                popular_for=['Blue Mosque', 'Hagia Sophia', 'Topkapı Palace', 'museums', 'history'],
                transport_hubs=['Sultanahmet Tram'],
                side='european'
            ),
            
            'Beyoğlu': Neighborhood(
                name='Beyoğlu',
                character='Nightlife, art galleries, cultural hub',
                signals=['needs_neighborhood', 'needs_nightlife', 'needs_restaurant'],
                popular_for=['Istiklal Street', 'bars', 'restaurants', 'nightlife', 'art'],
                transport_hubs=['Taksim Metro', 'Şişhane Metro', 'Karaköy'],
                side='european'
            ),
            
            'Kadıköy': Neighborhood(
                name='Kadıköy',
                character='Bohemian Asian-side district',
                signals=['needs_neighborhood', 'needs_restaurant', 'needs_nightlife'],
                popular_for=['cafes', 'bars', 'street markets', 'moda', 'young crowd'],
                transport_hubs=['Kadıköy Ferry', 'Kadıköy Metro'],
                side='asian'
            ),
            
            'Beşiktaş': Neighborhood(
                name='Beşiktaş',
                character='Mix of historic and modern, waterfront',
                signals=['needs_neighborhood', 'needs_restaurant'],
                popular_for=['Dolmabahçe Palace', 'waterfront', 'fish restaurants', 'shopping'],
                transport_hubs=['Beşiktaş Ferry', 'Kabataş Tram'],
                side='european'
            ),
            
            'Ortaköy': Neighborhood(
                name='Ortaköy',
                character='Bosphorus waterfront with cafes',
                signals=['needs_neighborhood', 'needs_restaurant'],
                popular_for=['Ortaköy Mosque', 'kumpir', 'waterfront cafes', 'weekend market'],
                transport_hubs=['Bus to Beşiktaş'],
                side='european'
            ),
            
            'Eminönü': Neighborhood(
                name='Eminönü',
                character='Historic commercial district',
                signals=['needs_neighborhood', 'needs_shopping', 'needs_transportation'],
                popular_for=['Spice Bazaar', 'ferries', 'Galata Bridge', 'street food'],
                transport_hubs=['Eminönü Ferry', 'Eminönü Tram'],
                side='european'
            ),
            
            'Üsküdar': Neighborhood(
                name='Üsküdar',
                character='Traditional Asian-side district',
                signals=['needs_neighborhood'],
                popular_for=['Maiden\'s Tower', 'tea gardens', 'traditional atmosphere'],
                transport_hubs=['Üsküdar Ferry', 'Marmaray'],
                side='asian'
            ),
            
            'Karaköy': Neighborhood(
                name='Karaköy',
                character='Trendy waterfront area',
                signals=['needs_neighborhood', 'needs_restaurant', 'needs_nightlife'],
                popular_for=['cafes', 'galleries', 'restaurants', 'vintage shops'],
                transport_hubs=['Karaköy Ferry', 'Karaköy Metro', 'Tünel'],
                side='european'
            ),
            
            'Balat': Neighborhood(
                name='Balat',
                character='Colorful historic neighborhood',
                signals=['needs_neighborhood', 'needs_attraction'],
                popular_for=['colorful houses', 'cafes', 'antique shops', 'churches'],
                transport_hubs=['Bus from Eminönü'],
                side='european'
            ),
            
            'Nişantaşı': Neighborhood(
                name='Nişantaşı',
                character='Upscale shopping and dining',
                signals=['needs_neighborhood', 'needs_shopping', 'needs_restaurant'],
                popular_for=['luxury shopping', 'cafes', 'fine dining', 'fashion'],
                transport_hubs=['Osmanbey Metro', 'Şişli Metro'],
                side='european'
            ),
        }
        
        return neighborhoods
    
    def _init_transport_terms(self) -> Dict[str, Dict[str, str]]:
        """Initialize local transportation terminology."""
        terms = {
            'en': {
                'dolmuş': 'shared minibus',
                'vapur': 'ferry',
                'iskele': 'ferry dock/pier',
                'durak': 'bus stop',
                'hat': 'ferry line',
                'marmaray': 'underwater train tunnel',
                'metrobüs': 'bus rapid transit',
                'tramvay': 'tram',
                'füniküler': 'funicular',
                'teleferik': 'cable car',
                'akbil': 'old transit card (deprecated)',
                'istanbulkart': 'transit card',
                'jetons': 'tokens (old system)'
            },
            'tr': {
                'dolmuş': 'dolmuş (shared minibus)',
                'vapur': 'vapur (ferry)',
                'iskele': 'iskele (pier)',
                'durak': 'durak (stop)',
                'hat': 'hat (line)',
                'marmaray': 'Marmaray',
                'metrobüs': 'Metrobüs',
                'tramvay': 'tramvay',
                'füniküler': 'füniküler',
                'teleferik': 'teleferik'
            }
        }
        
        return terms
    
    def _init_local_slang(self) -> Dict[str, str]:
        """Initialize local slang and colloquialisms."""
        slang = {
            # Location references
            'Avrupa yakası': 'European side',
            'Asya yakası': 'Asian side',
            'karşı yakaya': 'to the other side',
            'boğaza': 'to the Bosphorus',
            
            # Actions
            'vapura binmek': 'take the ferry',
            'karşıya geçmek': 'cross to the other side',
            'tramvaya binmek': 'take the tram',
            
            # Food/dining
            'midye dolma': 'stuffed mussels (street food)',
            'simit': 'sesame bread ring',
            'balık ekmek': 'fish sandwich',
            'kumpir': 'loaded baked potato',
            'çay': 'tea',
            'kahve': 'coffee',
            
            # Common phrases
            'nerede': 'where',
            'nasıl giderim': 'how do I go',
            'kaç para': 'how much money',
            'ne kadar sürer': 'how long does it take',
        }
        
        return slang
    
    def _init_poi_categories(self) -> Dict[str, List[str]]:
        """Initialize POI category keywords."""
        categories = {
            'mosque': ['cami', 'camii', 'mosque', 'masjid'],
            'palace': ['saray', 'sarayı', 'palace', 'köşk'],
            'tower': ['kule', 'kulesi', 'tower'],
            'museum': ['müze', 'müzesi', 'museum'],
            'market': ['çarşı', 'çarşısı', 'pazar', 'market', 'bazaar'],
            'square': ['meydan', 'meydanı', 'square'],
            'street': ['cadde', 'caddesi', 'sokak', 'street', 'avenue'],
            'park': ['park', 'parkı', 'bahçe', 'garden'],
            'beach': ['plaj', 'beach', 'sahil'],
        }
        
        return categories
    
    def detect_landmarks(self, query: str, language: str = 'en') -> List[Dict]:
        """
        Detect Istanbul landmarks mentioned in query.
        
        Args:
            query: User query
            language: Query language
            
        Returns:
            List of detected landmarks with metadata
        """
        query_lower = query.lower()
        detected = []
        
        for landmark_name, landmark in self.landmarks.items():
            # Check main name
            if landmark.name.lower() in query_lower:
                detected.append({
                    'name': landmark.name,
                    'signal': landmark.signal,
                    'location': landmark.location,
                    'transport': landmark.nearby_transport,
                    'category': landmark.category,
                    'match_type': 'exact'
                })
                continue
            
            # Check synonyms
            for synonym in landmark.synonyms:
                if synonym.lower() in query_lower:
                    detected.append({
                        'name': landmark.name,
                        'signal': landmark.signal,
                        'location': landmark.location,
                        'transport': landmark.nearby_transport,
                        'category': landmark.category,
                        'match_type': 'synonym'
                    })
                    break
        
        if detected:
            logger.debug(f"Detected landmarks: {[d['name'] for d in detected]}")
        
        return detected
    
    def detect_neighborhoods(self, query: str) -> List[Dict]:
        """
        Detect Istanbul neighborhoods mentioned in query.
        
        Args:
            query: User query
            
        Returns:
            List of detected neighborhoods with metadata
        """
        query_lower = query.lower()
        detected = []
        
        for neighborhood_name, neighborhood in self.neighborhoods.items():
            if neighborhood_name.lower() in query_lower:
                detected.append({
                    'name': neighborhood.name,
                    'character': neighborhood.character,
                    'signals': neighborhood.signals,
                    'popular_for': neighborhood.popular_for,
                    'transport': neighborhood.transport_hubs,
                    'side': neighborhood.side
                })
        
        if detected:
            logger.debug(f"Detected neighborhoods: {[d['name'] for d in detected]}")
        
        return detected
    
    def detect_transport_terms(self, query: str, language: str = 'en') -> List[str]:
        """
        Detect Istanbul-specific transport terms in query.
        
        Args:
            query: User query
            language: Query language
            
        Returns:
            List of detected transport terms
        """
        query_lower = query.lower()
        detected = []
        
        terms_dict = self.transport_terms.get(language, self.transport_terms['en'])
        
        for term in terms_dict.keys():
            if term.lower() in query_lower:
                detected.append(term)
        
        if detected:
            logger.debug(f"Detected transport terms: {detected}")
        
        return detected
    
    def get_suggested_signals(
        self,
        query: str,
        language: str = 'en'
    ) -> Dict[str, float]:
        """
        Get suggested signals based on Istanbul context.
        
        Args:
            query: User query
            language: Query language
            
        Returns:
            Dict of signal_name -> confidence
        """
        signals = {}
        
        # Check landmarks
        landmarks = self.detect_landmarks(query, language)
        for landmark in landmarks:
            signal = landmark['signal']
            # Landmark detection is high confidence
            signals[signal] = max(signals.get(signal, 0), 0.85)
        
        # Check neighborhoods
        neighborhoods = self.detect_neighborhoods(query)
        for neighborhood in neighborhoods:
            for signal in neighborhood['signals']:
                # Neighborhood context is medium-high confidence
                signals[signal] = max(signals.get(signal, 0), 0.75)
        
        # Check transport terms
        transport_terms = self.detect_transport_terms(query, language)
        if transport_terms:
            # Transport terminology strongly suggests transportation need
            signals['needs_transportation'] = 0.80
        
        return signals
    
    def get_context_info(
        self,
        query: str,
        language: str = 'en'
    ) -> Dict:
        """
        Get comprehensive Istanbul context for query.
        
        Args:
            query: User query
            language: Query language
            
        Returns:
            Dict with all detected Istanbul context
        """
        context = {
            'landmarks': self.detect_landmarks(query, language),
            'neighborhoods': self.detect_neighborhoods(query),
            'transport_terms': self.detect_transport_terms(query, language),
            'suggested_signals': self.get_suggested_signals(query, language)
        }
        
        # Add recommendations
        if context['landmarks']:
            context['has_landmarks'] = True
            context['landmark_names'] = [l['name'] for l in context['landmarks']]
        
        if context['neighborhoods']:
            context['has_neighborhoods'] = True
            context['neighborhood_names'] = [n['name'] for n in context['neighborhoods']]
        
        return context
    
    def get_landmark_info(self, landmark_name: str) -> Optional[Landmark]:
        """Get detailed info about a specific landmark."""
        return self.landmarks.get(landmark_name)
    
    def get_neighborhood_info(self, neighborhood_name: str) -> Optional[Neighborhood]:
        """Get detailed info about a specific neighborhood."""
        return self.neighborhoods.get(neighborhood_name)
    
    def search_landmarks_by_category(self, category: str) -> List[Landmark]:
        """Search landmarks by category."""
        return [
            landmark for landmark in self.landmarks.values()
            if landmark.category == category
        ]
    
    def get_nearby_landmarks(
        self,
        lat: float,
        lon: float,
        radius_km: float = 2.0
    ) -> List[Tuple[Landmark, float]]:
        """
        Get landmarks near a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of (Landmark, distance_km) tuples
        """
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points in km."""
            R = 6371  # Earth radius in km
            
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            
            a = (math.sin(delta_lat / 2) ** 2 +
                 math.cos(lat1_rad) * math.cos(lat2_rad) *
                 math.sin(delta_lon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            return R * c
        
        nearby = []
        for landmark in self.landmarks.values():
            distance = haversine_distance(
                lat, lon,
                landmark.location[0], landmark.location[1]
            )
            
            if distance <= radius_km:
                nearby.append((landmark, distance))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        
        return nearby


# Singleton instance
_istanbul_knowledge: Optional[IstanbulKnowledge] = None


def get_istanbul_knowledge() -> IstanbulKnowledge:
    """
    Get singleton Istanbul knowledge instance.
    
    Returns:
        IstanbulKnowledge instance
    """
    global _istanbul_knowledge
    
    if _istanbul_knowledge is None:
        _istanbul_knowledge = IstanbulKnowledge()
    
    return _istanbul_knowledge
