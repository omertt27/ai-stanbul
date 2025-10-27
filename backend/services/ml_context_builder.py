"""
ML Context Builder
Centralized context extraction for ML-enhanced handlers
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class MLContext:
    """Structured ML context for handlers"""
    user_query: str
    user_id: str
    session_id: str
    
    # Extracted features
    keywords: List[str]
    entities: Dict[str, List[str]]
    intent_hints: List[str]
    
    # Temporal context
    time_of_day: Optional[str]  # morning, afternoon, evening, night
    date_preference: Optional[str]  # today, tomorrow, this weekend, etc.
    urgency: float  # 0.0 to 1.0
    
    # User context
    user_interests: List[str]
    user_sentiment: float  # -1.0 to 1.0
    budget_level: Optional[str]  # free, budget, moderate, luxury
    
    # Location context
    current_location: Optional[str]
    location_preference: Optional[str]
    district_mentions: List[str]
    
    # Activity preferences
    indoor_outdoor_pref: Optional[str]  # indoor, outdoor, both
    crowd_preference: Optional[str]  # intimate, moderate, large
    activity_type: Optional[str]  # cultural, adventure, relaxation, etc.
    
    # Weather context
    weather_sensitive: bool
    weather_conditions: Optional[Dict[str, Any]]
    
    # Additional metadata
    conversation_history: List[Dict[str, str]]
    raw_neural_insights: Optional[Dict[str, Any]]


class MLContextBuilder:
    """
    Centralized ML Context Builder
    
    Extracts and structures context from user queries for ML-enhanced handlers.
    Provides a consistent interface for all handlers to access rich contextual information.
    """
    
    def __init__(self):
        """Initialize the ML context builder with Turkish and English support"""
        logger.info("🔧 ML Context Builder initialized (Turkish & English)")
        
        # Time-of-day patterns (Turkish & English)
        self.time_patterns = {
            'morning': [
                # English
                'morning', 'breakfast', 'early', 'dawn', 'sunrise',
                # Turkish
                'sabah', 'sabahleyin', 'kahvaltı', 'erken', 'güneş doğuşu', 'şafak'
            ],
            'afternoon': [
                # English
                'afternoon', 'lunch', 'midday', 'noon',
                # Turkish
                'öğleden sonra', 'öğle', 'öğle yemeği', 'öğlen'
            ],
            'evening': [
                # English
                'evening', 'dinner', 'sunset', 'dusk',
                # Turkish
                'akşam', 'akşamleyin', 'akşam yemeği', 'günbatımı', 'alacakaranlık'
            ],
            'night': [
                # English
                'night', 'late', 'midnight', 'after dark', 'nighttime',
                # Turkish
                'gece', 'gece vakti', 'geceleyin', 'geç', 'gece yarısı', 'karanlıkta'
            ]
        }
        
        # Date preference patterns (Turkish & English)
        self.date_patterns = {
            'today': [
                # English
                'today', 'now', 'current', 'right now',
                # Turkish
                'bugün', 'şimdi', 'şu an', 'hemen şimdi', 'güncel'
            ],
            'tomorrow': [
                # English
                'tomorrow', 'next day',
                # Turkish
                'yarın', 'ertesi gün'
            ],
            'this_weekend': [
                # English
                'this weekend', 'weekend', 'saturday', 'sunday',
                # Turkish
                'bu hafta sonu', 'hafta sonu', 'cumartesi', 'pazar'
            ],
            'this_week': [
                # English
                'this week', 'coming days',
                # Turkish
                'bu hafta', 'önümüzdeki günler'
            ],
            'next_week': [
                # English
                'next week', 'following week',
                # Turkish
                'gelecek hafta', 'önümüzdeki hafta', 'sonraki hafta'
            ],
            'this_month': [
                # English
                'this month', 'coming weeks',
                # Turkish
                'bu ay', 'önümüzdeki haftalar'
            ]
        }
        
        # Budget level patterns (Turkish & English)
        self.budget_patterns = {
            'free': [
                # English
                'free', 'no cost', 'no entrance', 'no fee', 'without paying',
                # Turkish
                'ücretsiz', 'bedava', 'parasız', 'giriş ücretsiz', 'ücret yok', 'para ödemeden'
            ],
            'budget': [
                # English
                'cheap', 'budget', 'affordable', 'inexpensive', 'economical', 'low cost',
                # Turkish
                'ucuz', 'bütçe', 'uygun fiyat', 'ekonomik', 'düşük fiyat', 'hesaplı', 'ucuza'
            ],
            'moderate': [
                # English
                'moderate', 'mid-range', 'reasonable', 'fair price',
                # Turkish
                'orta', 'orta seviye', 'makul', 'uygun', 'normal fiyat'
            ],
            'luxury': [
                # English
                'luxury', 'expensive', 'high-end', 'premium', 'upscale', 'finest',
                # Turkish
                'lüks', 'pahalı', 'yüksek kalite', 'premium', 'en iyi', 'prestijli'
            ]
        }
        
        # Indoor/outdoor patterns (Turkish & English)
        self.location_type_patterns = {
            'indoor': [
                # English
                'indoor', 'inside', 'covered', 'sheltered', 'museum', 'gallery', 'mall',
                # Turkish
                'kapalı', 'içeride', 'içerisi', 'iç mekan', 'kapalı alan', 'müze', 'galeri', 'alışveriş merkezi', 'avm'
            ],
            'outdoor': [
                # English
                'outdoor', 'outside', 'open air', 'park', 'garden', 'waterfront',
                # Turkish
                'açık hava', 'dışarıda', 'dışarısı', 'açık alan', 'park', 'bahçe', 'sahil', 'kıyı'
            ]
        }
        
        # Crowd preference patterns (Turkish & English)
        self.crowd_patterns = {
            'intimate': [
                # English
                'quiet', 'peaceful', 'less crowded', 'intimate', 'small', 'uncrowded',
                # Turkish
                'sakin', 'huzurlu', 'kalabalık olmayan', 'az kalabalık', 'küçük', 'sessiz'
            ],
            'moderate': [
                # English
                'moderate', 'normal', 'typical', 'average crowd',
                # Turkish
                'orta', 'normal', 'standart', 'tipik', 'ortalama kalabalık'
            ],
            'large': [
                # English
                'busy', 'crowded', 'popular', 'lively', 'vibrant', 'bustling',
                # Turkish
                'kalabalık', 'popüler', 'canlı', 'hareketli', 'yoğun', 'işlek'
            ]
        }
        
        # Activity type patterns (Turkish & English)
        self.activity_patterns = {
            'cultural': [
                # English
                'cultural', 'museum', 'historical', 'heritage', 'art', 'traditional',
                # Turkish
                'kültürel', 'müze', 'tarihsel', 'tarihi', 'miras', 'sanat', 'geleneksel'
            ],
            'adventure': [
                # English
                'adventure', 'exciting', 'active', 'sports', 'hiking', 'outdoor',
                # Turkish
                'macera', 'heyecan', 'heyecanlı', 'aktif', 'spor', 'doğa yürüyüşü', 'açık hava'
            ],
            'relaxation': [
                # English
                'relaxing', 'calm', 'peaceful', 'spa', 'park', 'quiet',
                # Turkish
                'rahatlatıcı', 'sakin', 'huzurlu', 'dinlendirici', 'spa', 'park', 'sessiz'
            ],
            'entertainment': [
                # English
                'entertainment', 'fun', 'show', 'concert', 'performance',
                # Turkish
                'eğlence', 'eğlenceli', 'gösteri', 'konser', 'performans', 'etkinlik'
            ],
            'culinary': [
                # English
                'food', 'dining', 'restaurant', 'cuisine', 'eating', 'tasting',
                # Turkish
                'yemek', 'lokanta', 'restoran', 'mutfak', 'yeme', 'tadım', 'lezzet'
            ],
            'shopping': [
                # English
                'shopping', 'market', 'bazaar', 'stores', 'boutiques',
                # Turkish
                'alışveriş', 'pazar', 'çarşı', 'market', 'mağaza', 'butik', 'dükkan'
            ],
            'nightlife': [
                # English
                'nightlife', 'bars', 'clubs', 'evening', 'night out',
                # Turkish
                'gece hayatı', 'bar', 'barlar', 'kulüp', 'kulüpler', 'akşam eğlencesi', 'gece çıkışı'
            ]
        }
        
        # Urgency indicators (Turkish & English)
        self.urgency_patterns = {
            'high': [
                # English
                'urgent', 'immediately', 'right now', 'asap', 'quickly', 'soon',
                # Turkish
                'acil', 'hemen', 'şimdi', 'çabuk', 'ivedi', 'derhal', 'acilen'
            ],
            'medium': [
                # English
                'today', 'this evening', 'tonight', 'soon',
                # Turkish
                'bugün', 'bu akşam', 'bu gece', 'yakında'
            ],
            'low': [
                # English
                'sometime', 'eventually', 'maybe', 'considering',
                # Turkish
                'bir ara', 'sonunda', 'belki', 'düşünüyorum'
            ]
        }
    
    def build_context(
        self,
        user_query: str,
        user_profile: Any,
        conversation_context: Any,
        entities: Optional[Dict[str, List[str]]] = None,
        neural_insights: Optional[Dict[str, Any]] = None
    ) -> MLContext:
        """
        Build comprehensive ML context from query and user information
        
        Args:
            user_query: The user's query string
            user_profile: User profile object
            conversation_context: Conversation context object
            entities: Pre-extracted entities (optional)
            neural_insights: Neural processor insights (optional)
        
        Returns:
            MLContext object with structured context information
        """
        try:
            query_lower = user_query.lower()
            
            # Extract basic identifiers
            user_id = getattr(user_profile, 'user_id', 'unknown')
            session_id = getattr(conversation_context, 'session_id', 'unknown')
            
            # Extract keywords
            keywords = self._extract_keywords(user_query)
            
            # Get entities (from parameter or extract)
            if entities is None:
                entities = {}
            
            # Extract intent hints
            intent_hints = self._extract_intent_hints(query_lower)
            
            # Extract temporal context
            time_of_day = self._extract_time_of_day(query_lower)
            date_preference = self._extract_date_preference(query_lower)
            urgency = self._calculate_urgency(query_lower)
            
            # Extract user context
            user_interests = getattr(user_profile, 'interests', [])
            user_sentiment = neural_insights.get('sentiment', 0.0) if neural_insights else 0.0
            budget_level = self._extract_budget_level(query_lower)
            
            # Extract location context
            current_location = getattr(conversation_context, 'current_location', None)
            location_preference = self._extract_location_preference(query_lower)
            district_mentions = self._extract_district_mentions(query_lower)
            
            # Extract activity preferences
            indoor_outdoor_pref = self._extract_indoor_outdoor_pref(query_lower)
            crowd_preference = self._extract_crowd_preference(query_lower)
            activity_type = self._extract_activity_type(query_lower)
            
            # Weather context
            weather_sensitive = self._is_weather_sensitive(query_lower)
            weather_conditions = None  # Could be populated from weather service
            
            # Conversation history
            conversation_history = []
            if hasattr(conversation_context, 'conversation_history'):
                history = conversation_context.conversation_history[-5:]  # Last 5 interactions
                conversation_history = [
                    {'message': h.message, 'response': h.response, 'intent': h.intent}
                    for h in history
                ]
            
            # Build context object
            context = MLContext(
                user_query=user_query,
                user_id=user_id,
                session_id=session_id,
                keywords=keywords,
                entities=entities,
                intent_hints=intent_hints,
                time_of_day=time_of_day,
                date_preference=date_preference,
                urgency=urgency,
                user_interests=user_interests,
                user_sentiment=user_sentiment,
                budget_level=budget_level,
                current_location=current_location,
                location_preference=location_preference,
                district_mentions=district_mentions,
                indoor_outdoor_pref=indoor_outdoor_pref,
                crowd_preference=crowd_preference,
                activity_type=activity_type,
                weather_sensitive=weather_sensitive,
                weather_conditions=weather_conditions,
                conversation_history=conversation_history,
                raw_neural_insights=neural_insights
            )
            
            logger.debug(f"Built ML context for query: {user_query[:50]}...")
            return context
            
        except Exception as e:
            logger.error(f"Error building ML context: {e}")
            # Return minimal context on error
            return MLContext(
                user_query=user_query,
                user_id=getattr(user_profile, 'user_id', 'unknown'),
                session_id=getattr(conversation_context, 'session_id', 'unknown'),
                keywords=[],
                entities={},
                intent_hints=[],
                time_of_day=None,
                date_preference=None,
                urgency=0.5,
                user_interests=[],
                user_sentiment=0.0,
                budget_level=None,
                current_location=None,
                location_preference=None,
                district_mentions=[],
                indoor_outdoor_pref=None,
                crowd_preference=None,
                activity_type=None,
                weather_sensitive=False,
                weather_conditions=None,
                conversation_history=[],
                raw_neural_insights=neural_insights
            )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query (Turkish & English)"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            # English stop words
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'i', 'you', 'me', 'my', 'your',
            # Turkish stop words
            'bir', 'bu', 'şu', 'o', 've', 'veya', 'ile', 'için', 'gibi', 'kadar',
            'mi', 'mı', 'mu', 'mü', 've', 'da', 'de', 'ki', 'ne', 'nasıl',
            'var', 'yok', 'ben', 'sen', 'biz', 'siz', 'onlar', 'benim', 'senin'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:15]  # Limit to top 15 keywords
    
    def _extract_intent_hints(self, query_lower: str) -> List[str]:
        """Extract intent hints from query (Turkish & English)"""
        hints = []
        
        intent_keywords = {
            'event': [
                # English
                'event', 'happening', 'concert', 'show', 'performance', 'festival',
                # Turkish
                'etkinlik', 'etkinlikler', 'oluyor', 'konser', 'gösteri', 'performans', 'festival', 'şenlik'
            ],
            'hidden_gems': [
                # English
                'hidden', 'secret', 'local', 'authentic', 'off beaten',
                # Turkish
                'gizli', 'saklı', 'yerel', 'otantik', 'az bilinen', 'keşfedilmemiş', 'turistik olmayan'
            ],
            'weather': [
                # English
                'weather', 'temperature', 'rain', 'sunny', 'forecast',
                # Turkish
                'hava', 'hava durumu', 'sıcaklık', 'derece', 'yağmur', 'güneşli', 'tahmin'
            ],
            'route': [
                # English
                'route', 'directions', 'how to get', 'plan', 'itinerary',
                # Turkish
                'rota', 'yol', 'tarif', 'nasıl gidilir', 'nasıl giderim', 'plan', 'program', 'güzergah'
            ],
            'neighborhood': [
                # English
                'neighborhood', 'district', 'area', 'which area',
                # Turkish
                'mahalle', 'semt', 'bölge', 'ilçe', 'hangi bölge', 'hangi semt'
            ],
            'restaurant': [
                # English
                'restaurant', 'eat', 'food', 'dining', 'lunch', 'dinner',
                # Turkish
                'restoran', 'lokanta', 'ye', 'yemek', 'yeme', 'öğle yemeği', 'akşam yemeği', 'nerede yenir'
            ],
            'attraction': [
                # English
                'visit', 'see', 'attraction', 'museum', 'landmark',
                # Turkish
                'ziyaret', 'gör', 'görülecek', 'gezilecek', 'yer', 'yerler', 'müze', 'anıt', 'tarihi yer'
            ]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                hints.append(intent)
        
        return hints
    
    def _extract_time_of_day(self, query_lower: str) -> Optional[str]:
        """Extract time of day preference"""
        for time_period, patterns in self.time_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return time_period
        return None
    
    def _extract_date_preference(self, query_lower: str) -> Optional[str]:
        """Extract date preference"""
        for date_type, patterns in self.date_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return date_type
        return None
    
    def _calculate_urgency(self, query_lower: str) -> float:
        """Calculate query urgency (0.0 to 1.0)"""
        if any(pattern in query_lower for pattern in self.urgency_patterns['high']):
            return 1.0
        elif any(pattern in query_lower for pattern in self.urgency_patterns['medium']):
            return 0.7
        elif any(pattern in query_lower for pattern in self.urgency_patterns['low']):
            return 0.3
        return 0.5  # Default medium urgency
    
    def _extract_budget_level(self, query_lower: str) -> Optional[str]:
        """Extract budget level preference"""
        for budget, patterns in self.budget_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return budget
        return None
    
    def _extract_location_preference(self, query_lower: str) -> Optional[str]:
        """Extract location preference from query"""
        # Look for 'in X', 'at X', 'near X' patterns
        location_patterns = [
            r'in (\w+(?:\s+\w+)?)',
            r'at (\w+(?:\s+\w+)?)',
            r'near (\w+(?:\s+\w+)?)',
            r'around (\w+(?:\s+\w+)?)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_district_mentions(self, query_lower: str) -> List[str]:
        """Extract mentioned districts/neighborhoods"""
        districts = [
            'sultanahmet', 'beyoglu', 'beyoğlu', 'galata', 'karaköy', 'karakoy',
            'taksim', 'besiktas', 'beşiktaş', 'ortakoy', 'ortaköy',
            'kadikoy', 'kadıköy', 'uskudar', 'üsküdar', 'eminonu', 'eminönü',
            'fatih', 'sisli', 'şişli', 'bakirkoy', 'bakırköy', 'bebek',
            'arnavutkoy', 'arnavutköy', 'balat', 'fener', 'cihangir',
            'nisantasi', 'nişantaşı', 'etiler', 'levent', 'maslak'
        ]
        
        mentioned = [d for d in districts if d in query_lower]
        return mentioned
    
    def _extract_indoor_outdoor_pref(self, query_lower: str) -> Optional[str]:
        """Extract indoor/outdoor preference"""
        for pref_type, patterns in self.location_type_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return pref_type
        return None
    
    def _extract_crowd_preference(self, query_lower: str) -> Optional[str]:
        """Extract crowd size preference"""
        for crowd_type, patterns in self.crowd_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return crowd_type
        return None
    
    def _extract_activity_type(self, query_lower: str) -> Optional[str]:
        """Extract activity type preference"""
        for activity, patterns in self.activity_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return activity
        return None
    
    def _is_weather_sensitive(self, query_lower: str) -> bool:
        """Determine if query is weather-sensitive (Turkish & English)"""
        weather_indicators = [
            # English
            'outdoor', 'outside', 'walk', 'park', 'garden', 'beach',
            'weather', 'rain', 'sunny', 'cold', 'hot', 'covered',
            # Turkish
            'açık hava', 'dışarıda', 'yürüyüş', 'park', 'bahçe', 'sahil', 'plaj',
            'hava', 'hava durumu', 'yağmur', 'güneşli', 'soğuk', 'sıcak', 'kapalı'
        ]
        return any(indicator in query_lower for indicator in weather_indicators)


# Singleton instance
_ml_context_builder = None


def get_ml_context_builder() -> MLContextBuilder:
    """Get or create singleton ML context builder instance"""
    global _ml_context_builder
    if _ml_context_builder is None:
        _ml_context_builder = MLContextBuilder()
    return _ml_context_builder
