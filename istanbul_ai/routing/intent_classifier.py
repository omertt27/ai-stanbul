"""
Intent Classifier - Classify user intent from messages

This module handles intent classification for user queries, determining what the user
wants to do (e.g., find restaurants, visit attractions, get transportation info).

Week 2 Refactoring: Extracted from main_system.py
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from ..core.models import ConversationContext

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Intent classification result"""
    primary_intent: str
    confidence: float = 0.0
    intents: List[str] = field(default_factory=list)
    is_multi_intent: bool = False
    multi_intent_response: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)


class IntentClassifier:
    """Classifies user intent from natural language messages"""
    
    def __init__(self):
        """Initialize the intent classifier with keyword mappings"""
        self.intent_keywords = self._initialize_intent_keywords()
        self.daily_talk_patterns = self._initialize_daily_talk_patterns()
    
    def _initialize_intent_keywords(self) -> Dict[str, List[str]]:
        """Initialize comprehensive intent keyword mappings"""
        return {
            'restaurant': [
                'eat', 'food', 'restaurant', 'lunch', 'dinner', 'breakfast', 
                'hungry', 'cuisine', 'meal', 'dining', 'cafe', 'bistro'
            ],
            'attraction': [
                # General attraction words
                'visit', 'see', 'attraction', 'attractions', 'tour', 'sightseeing', 
                'tourist', 'landmark', 'landmarks',
                # Museum keywords
                'museum', 'museums', 'gallery', 'galleries', 'exhibition', 'exhibitions', 
                'art museum', 'art museums', 'historical museum', 'history museum', 
                'archaeological', 'archaeology',
                # Specific place types
                'mosque', 'mosques', 'palace', 'palaces', 'church', 'churches', 
                'synagogue', 'monument', 'monuments', 'memorial', 'historical site', 
                'historical sites',
                # Descriptive words
                'historical', 'ancient', 'cultural site', 'heritage', 'artifact', 
                'artifacts',
                # Action words
                'explore', 'discover', 'show me', 'what to see', 'worth seeing', 
                'must see', 'should i see',
                # Specific queries
                'places to visit', 'places to see', 'what can i visit', 
                'what can i see', 'best attractions', 'top attractions', 
                'famous places', 'popular places'
            ],
            'transportation': [
                'transport', 'metro', 'bus', 'taxi', 'ferry', 'how to get', 
                'travel', 'tram', 'istanbulkart', 'public transport', 'dolmuş'
            ],
            'neighborhood': [
                'neighborhood', 'area', 'district', 'where to stay', 'which area',
                'quarter', 'region', 'location', 'suburb'
            ],
            'shopping': [
                'shop', 'shopping', 'buy', 'bazaar', 'market', 'souvenir',
                'store', 'mall', 'purchase', 'gifts'
            ],
            'events': [
                # Core event keywords
                'event', 'events', 'activity', 'activities', 'entertainment', 
                'nightlife', 'what to do', 'things to do', 'happening', 'going on',
                # Performance types
                'concert', 'concerts', 'show', 'shows', 'performance', 'performances', 
                'theater', 'theatre', 'opera', 'ballet', 'dance', 'comedy',
                'live music', 'dj', 'club', 'party', 'parties', 'celebration',
                # Event types
                'cultural', 'festival', 'festivals', 'exhibition', 'exhibitions',
                'gallery opening', 'art show', 'music event', 'art event',
                'sporting event', 'sports', 'match', 'game', 'tournament',
                # Venues
                'iksv', 'İKSV', 'salon', 'babylon', 'zorlu', 'zorlu psm',
                'cemal reşit rey', 'atatürk kültür merkezi', 'akm',
                # Temporal patterns
                'tonight', 'today', 'tomorrow', 'this weekend', 'this week', 
                'this month', 'next week', 'upcoming', 'soon', 'now', 'currently',
                'this evening', 'this afternoon', 'later today', 'upcoming events',
                # Questions
                'what\'s on', 'whats on', 'what\'s happening', 'any events',
                'any concerts', 'any shows', 'where to go', 'what\'s playing'
            ],
            'weather': [
                # Core weather keywords
                'weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy', 
                'hot', 'cold', 'warm', 'cool', 'humid', 'dry',
                # Weather conditions
                'storm', 'stormy', 'snow', 'snowy', 'drizzle', 'shower', 'showers',
                'heatwave', 'windy', 'foggy', 'misty', 'haze',
                # Measurements
                'degrees', 'celsius', 'fahrenheit', 'humidity', 'wind', 
                'precipitation', 'atmospheric', 'barometric',
                # Question patterns
                'what\'s the weather', 'how\'s the weather', 'weather like',
                'weather today', 'weather tomorrow', 'weather this week',
                'will it rain', 'is it raining', 'is it sunny', 'is it hot', 'is it cold',
                'should i bring umbrella', 'should i bring jacket', 'need umbrella',
                'what to wear', 'dress for weather', 'what should i wear',
                # Weather + activity
                'weather appropriate', 'good weather for', 'weather for walking',
                'weather for sightseeing', 'outdoor weather',
                # General
                'climate', 'meteorological', 'weather conditions', 'forecast today'
            ],
            'airport_transport': [
                'airport', 'ist', 'saw', 'atatürk', 'ataturk', 'istanbul airport', 
                'sabiha gökçen', 'sabiha gokcen', 'new airport', 'airport transfer', 
                'airport transport', 'from airport', 'to airport', 'airport shuttle', 
                'airport bus', 'airport metro', 'flight', 'departure', 'arrival', 
                'terminal', 'baggage', 'customs', 'immigration'
            ],
            'hidden_gems': [
                'hidden', 'secret', 'local', 'authentic', 'off-beaten', 
                'off the beaten path', 'unknown', 'undiscovered', 'gems', 
                'hidden gems', 'secret spots', 'local favorites', 'insider', 
                'less touristy', 'not touristy', 'avoid crowds', 'unique places', 
                'special places', 'locals know', 'local secrets', 'hidden treasures', 
                'underground', 'alternative', 'unconventional', 'non-touristy', 
                'lesser known', 'hidden places'
            ],
            'route_planning': [
                # Planning keywords
                'route', 'itinerary', 'plan', 'planning', 'schedule', 'organize',
                'day trip', 'trip planning', 'travel plan',
                # Tour types
                'tour', 'one day tour', 'two day tour', 'three day tour', 'multi-day',
                'walking tour', 'food tour', 'cultural tour', 'historical tour',
                # Duration patterns
                'one day', 'two days', '3 days', '4 days', '5 days', 'week itinerary',
                'weekend trip', 'full day', 'half day', 'morning tour', 'afternoon tour',
                # Questions
                'what should i visit', 'best route to visit', 'how to visit all',
                'efficient way to see', 'best way to see', 'plan my visit',
                'help me plan', 'organize my trip', 'create itinerary',
                # Multi-destination
                'visit all', 'see everything', 'cover all', 'comprehensive tour',
                'best order to visit', 'optimal route'
            ],
            'gps_route_planning': [
                'directions', 'navigation', 'how to get', 'from', 'to', 'nearest', 
                'distance', 'walking route', 'driving route', 'public transport route'
            ],
            'museum_route_planning': [
                'museum route', 'museum tour', 'museum plan', 'museum itinerary', 
                'museums near', 'museum walk'
            ],
            'greeting': [
                'hello', 'hi', 'merhaba', 'help', 'start', 'hey', 'good morning',
                'good afternoon', 'good evening', 'selam', 'günaydın'
            ]
        }
    
    def _initialize_daily_talk_patterns(self) -> List[str]:
        """Initialize daily talk conversation patterns"""
        greeting_patterns = [
            'hi', 'hello', 'hey', 'merhaba', 'selam', 'good morning', 
            'good afternoon', 'good evening', 'günaydın', 'iyi günler', 
            'iyi akşamlar'
        ]
        
        weather_patterns = [
            'how\'s the weather', 'what\'s the weather', 'is it raining', 
            'is it sunny', 'hava nasıl', 'yağmur yağıyor mu', 'soğuk mu', 
            'sıcak mı'
        ]
        
        casual_patterns = [
            'how are you', 'what\'s up', 'how\'s it going', 'nasılsın', 
            'ne haber', 'naber'
        ]
        
        time_patterns = [
            'what time', 'what day', 'is it open', 'saat kaç', 'açık mı'
        ]
        
        daily_life_patterns = [
            'good morning', 'good night', 'thank you', 'thanks', 'please', 
            'excuse me', 'sorry', 'günaydın', 'iyi geceler', 'teşekkürler', 
            'lütfen', 'özür dilerim'
        ]
        
        cultural_patterns = [
            'local tip', 'cultural tip', 'local advice', 'what locals do',
            'like a local', 'authentic experience', 'local culture'
        ]
        
        return (greeting_patterns + weather_patterns + casual_patterns + 
                time_patterns + daily_life_patterns + cultural_patterns)
    
    def classify_intent(
        self, 
        message: str, 
        entities: Dict, 
        context: Optional[ConversationContext] = None,
        neural_insights: Optional[Dict] = None,
        preprocessed_query: Optional[Any] = None
    ) -> IntentResult:
        """
        Classify user intent from message with contextual awareness
        
        Args:
            message: User's input message
            entities: Extracted entities from message
            context: Conversation context (optional)
            neural_insights: Neural processing insights (optional)
            preprocessed_query: Preprocessed query data (optional)
        
        Returns:
            IntentResult object with classification details
        """
        message_lower = message.lower()
        primary_intent = 'general'
        
        # PRIORITY 1: Check for greeting/general intent (most specific patterns)
        if any(keyword in message_lower for keyword in self.intent_keywords['greeting']):
            primary_intent = 'greeting'
        
        # PRIORITY 2: Check for GPS-based route planning intent (very specific)
        elif any(keyword in message_lower for keyword in self.intent_keywords['gps_route_planning']) or \
            (any(indicator in message_lower for indicator in ['from', 'to', 'near', 'closest', 'nearby', 'distance']) and 
             any(rk in message_lower for rk in ['route', 'get', 'go', 'directions'])):
            primary_intent = 'gps_route_planning'
        
        # PRIORITY 3: Check for museum route planning intent (specific)
        elif any(keyword in message_lower for keyword in self.intent_keywords['museum_route_planning']) or \
            ('museum' in message_lower and any(rk in message_lower for rk in ['route', 'plan', 'tour', 'visit'])):
            primary_intent = 'museum_route_planning'
        
        # PRIORITY 4: Check for weather intent (specific patterns before food/general)
        elif any(keyword in message_lower for keyword in self.intent_keywords['weather']):
            primary_intent = 'weather'
        
        # PRIORITY 5: Check for route planning intent (before attractions)
        elif any(keyword in message_lower for keyword in self.intent_keywords['route_planning']):
            primary_intent = 'route_planning'
        
        # PRIORITY 6: Check for airport transport intent (specific)
        elif any(keyword in message_lower for keyword in self.intent_keywords['airport_transport']):
            primary_intent = 'airport_transport'
        
        # PRIORITY 7: Check for events/activities intent (before attractions)
        elif any(keyword in message_lower for keyword in self.intent_keywords['events']):
            primary_intent = 'events'
        
        # PRIORITY 8: Check for hidden gems intent
        elif any(keyword in message_lower for keyword in self.intent_keywords['hidden_gems']):
            primary_intent = 'hidden_gems'
        
        # PRIORITY 9: Check for transportation intent
        elif (any(keyword in message_lower for keyword in self.intent_keywords['transportation']) or 
            entities.get('transportation')):
            primary_intent = 'transportation'
        
        # PRIORITY 10: Check for neighborhood/area intent
        elif (any(keyword in message_lower for keyword in self.intent_keywords['neighborhood']) or 
            entities.get('neighborhoods')):
            primary_intent = 'neighborhood'
        
        # PRIORITY 11: Check for shopping intent
        elif any(keyword in message_lower for keyword in self.intent_keywords['shopping']):
            primary_intent = 'shopping'
        
        # PRIORITY 12: Check for restaurant/food intent (broader, comes later)
        elif any(keyword in message_lower for keyword in self.intent_keywords['restaurant']) or entities.get('cuisines'):
            primary_intent = 'restaurant'
        
        # PRIORITY 13: Check for attraction/sightseeing intent (broadest, comes last)
        elif any(word in message_lower for word in ['museum', 'museums', 'gallery', 'galleries', 'exhibition', 
                                                     'attraction', 'attractions', 'landmark', 'palace', 'mosque']) or \
            any(keyword in message_lower for keyword in self.intent_keywords['attraction']) or \
            entities.get('landmarks'):
            primary_intent = 'attraction'
        
        # Calculate confidence
        confidence = self.get_intent_confidence(message, primary_intent)
        
        # Detect multiple intents
        all_intents = self.detect_multiple_intents(message, entities)
        is_multi_intent = len(all_intents) > 1
        
        return IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            intents=all_intents,
            is_multi_intent=is_multi_intent,
            multi_intent_response=None,  # Can be enhanced later
            entities=entities
        )
    
    def detect_multiple_intents(self, message: str, entities: Dict) -> List[str]:
        """
        Detect multiple intents in a single message
        
        Args:
            message: User's input message
            entities: Extracted entities from message
        
        Returns:
            List of detected intents
        """
        intents = []
        message_lower = message.lower()
        
        # Check each intent category
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                intents.append(intent)
        
        return intents if intents else ['general']
    
    def is_daily_talk_query(self, message: str) -> bool:
        """
        Check if message is a daily talk/casual conversation query
        
        Args:
            message: User's input message
        
        Returns:
            True if message is daily talk, False otherwise
        """
        message_lower = message.lower()
        
        # Check if message contains daily talk patterns
        for pattern in self.daily_talk_patterns:
            if pattern in message_lower:
                return True
        
        # Check for short casual messages (likely daily talk)
        if (len(message.split()) <= 3 and 
            any(word in message_lower for word in 
                ['hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok', 'okay'])):
            return True
        
        return False
    
    def get_intent_confidence(self, message: str, intent: str) -> float:
        """
        Calculate confidence score for a classified intent with improved algorithm
        
        Confidence factors:
        1. Keyword match count (primary)
        2. Strong indicators (keywords that clearly signal intent)
        3. Message specificity (shorter, focused messages = higher confidence)
        
        Args:
            message: User's input message
            intent: Classified intent
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if intent not in self.intent_keywords:
            return 0.0
        
        message_lower = message.lower()
        message_words = set(message_lower.split())
        keywords = self.intent_keywords[intent]
        
        # Factor 1: Count matching keywords
        matches = sum(1 for keyword in keywords if keyword in message_lower)
        
        if matches == 0:
            return 0.0
        
        # Factor 2: Strong indicators (high-value keywords that clearly signal intent)
        strong_indicators = {
            'restaurant': ['restaurant', 'restaurants', 'eat', 'dining', 'lunch', 'dinner', 'breakfast', 'cuisine'],
            'attraction': ['museum', 'museums', 'palace', 'palaces', 'mosque', 'mosques', 'attraction', 'attractions', 'landmark', 'landmarks', 'visit', 'see'],
            'transportation': ['metro', 'bus', 'ferry', 'transport', 'transportation', 'how to get', 'directions', 'tram'],
            'weather': ['weather', 'forecast', 'temperature', 'rain', 'sunny', 'cloudy', 'what\'s the weather', 'how\'s the weather', 'will it rain'],
            'events': ['event', 'events', 'concert', 'concerts', 'show', 'shows', 'festival', 'festivals', 'happening', 'tonight', 'what\'s on', 'exhibition'],
            'neighborhood': ['neighborhood', 'area', 'district', 'where to stay'],
            'shopping': ['shopping', 'bazaar', 'market', 'souvenir', 'buy'],
            'hidden_gems': ['hidden', 'secret', 'local', 'gems', 'authentic'],
            'airport_transport': ['airport', 'terminal', 'flight', 'ist', 'saw'],
            'route_planning': ['route', 'itinerary', 'plan', 'schedule', 'tour', 'day trip', 'trip planning'],
            'gps_route_planning': ['directions', 'how to get', 'from', 'to', 'navigation'],
            'greeting': ['hello', 'hi', 'merhaba', 'help']
        }
        
        # Check for strong indicators
        has_strong_indicator = False
        if intent in strong_indicators:
            has_strong_indicator = any(
                indicator in message_lower 
                for indicator in strong_indicators[intent]
            )
        
        # Factor 3: Calculate base confidence using logarithmic scale
        # This avoids penalizing intents with many keywords
        if matches >= 3:
            base_confidence = 0.95
        elif matches == 2:
            base_confidence = 0.85
        elif matches == 1:
            base_confidence = 0.75 if has_strong_indicator else 0.65
        else:
            base_confidence = 0.50
        
        # Factor 4: Boost for strong indicators
        if has_strong_indicator:
            base_confidence = min(base_confidence + 0.15, 1.0)
        
        # Factor 5: Consider message specificity
        # Shorter, focused messages should have higher confidence
        if len(message_words) <= 5 and matches >= 1:
            base_confidence = min(base_confidence + 0.10, 1.0)
        
        # Factor 6: Penalize if message is very long and vague
        if len(message_words) > 15 and matches == 1 and not has_strong_indicator:
            base_confidence = max(base_confidence - 0.15, 0.40)
        
        return round(base_confidence, 2)
