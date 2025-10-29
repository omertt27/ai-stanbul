"""
Intent Classifier - Classify user intent from messages

This module handles intent classification for user queries, determining what the user
wants to do (e.g., find restaurants, visit attractions, get transportation info).

Week 2 Refactoring: Extracted from main_system.py
"""

import logging
from typing import Dict, List, Optional
from ..core.models import ConversationContext

logger = logging.getLogger(__name__)


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
                'event', 'events', 'activity', 'activities', 'entertainment', 
                'nightlife', 'what to do', 'things to do', 'happening', 'going on', 
                'concert', 'concerts', 'show', 'shows', 'performance', 'performances', 
                'theater', 'theatre', 'cultural', 'festival', 'festivals', 
                'exhibition', 'exhibitions', 'iksv', 'İKSV', 'salon', 'babylon', 
                'music event', 'art event', 'tonight', 'this weekend', 'this week', 
                'this month'
            ],
            'weather': [
                'weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy', 
                'hot', 'cold', 'what\'s the weather', 'how\'s the weather', 
                'weather today', 'weather tomorrow', 'will it rain', 'is it sunny', 
                'degrees', 'celsius', 'fahrenheit', 'humidity', 'wind', 
                'precipitation', 'weather conditions', 'climate'
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
                'route', 'itinerary', 'plan', 'schedule', 'day trip', 
                'trip planning', 'travel plan'
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
        context: Optional[ConversationContext] = None
    ) -> str:
        """
        Classify user intent from message with contextual awareness
        
        Args:
            message: User's input message
            entities: Extracted entities from message
            context: Conversation context (optional)
        
        Returns:
            Classified intent string
        """
        message_lower = message.lower()
        
        # Check for restaurant/food intent
        if any(keyword in message_lower for keyword in self.intent_keywords['restaurant']) or entities.get('cuisines'):
            return 'restaurant'
        
        # Check for attraction/sightseeing intent (enhanced with museum keywords)
        museum_specific = any(word in message_lower for word in ['museum', 'museums', 'gallery', 'galleries', 'exhibition'])
        attraction_specific = any(word in message_lower for word in ['attraction', 'attractions', 'landmark', 'palace', 'mosque'])
        
        if (museum_specific or attraction_specific or 
            any(keyword in message_lower for keyword in self.intent_keywords['attraction']) or 
            entities.get('landmarks')):
            return 'attraction'
        
        # Check for transportation intent
        if (any(keyword in message_lower for keyword in self.intent_keywords['transportation']) or 
            entities.get('transportation')):
            return 'transportation'
        
        # Check for neighborhood/area intent
        if (any(keyword in message_lower for keyword in self.intent_keywords['neighborhood']) or 
            entities.get('neighborhoods')):
            return 'neighborhood'
        
        # Check for shopping intent
        if any(keyword in message_lower for keyword in self.intent_keywords['shopping']):
            return 'shopping'
        
        # Check for events/activities intent
        if any(keyword in message_lower for keyword in self.intent_keywords['events']):
            return 'events'
        
        # Check for weather intent
        if any(keyword in message_lower for keyword in self.intent_keywords['weather']):
            return 'weather'
        
        # Check for airport transport intent
        if any(keyword in message_lower for keyword in self.intent_keywords['airport_transport']):
            return 'airport_transport'
        
        # Check for hidden gems intent
        if any(keyword in message_lower for keyword in self.intent_keywords['hidden_gems']):
            return 'hidden_gems'
        
        # Check for route planning intent
        if any(keyword in message_lower for keyword in self.intent_keywords['route_planning']):
            return 'route_planning'
        
        # Check for GPS-based route planning intent (more specific)
        gps_route_keywords = self.intent_keywords['gps_route_planning']
        location_indicators = ['from', 'to', 'near', 'closest', 'nearby', 'distance']
        if (any(keyword in message_lower for keyword in gps_route_keywords) or 
            (any(indicator in message_lower for indicator in location_indicators) and 
             any(rk in message_lower for rk in ['route', 'get', 'go', 'directions']))):
            return 'gps_route_planning'
        
        # Check for museum route planning intent (more specific)
        museum_route_keywords = self.intent_keywords['museum_route_planning']
        if (any(keyword in message_lower for keyword in museum_route_keywords) or 
            ('museum' in message_lower and any(rk in message_lower for rk in ['route', 'plan', 'tour', 'visit']))):
            return 'museum_route_planning'
        
        # Check for greeting/general intent
        if any(keyword in message_lower for keyword in self.intent_keywords['greeting']):
            return 'greeting'
        
        return 'general'
    
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
        Calculate confidence score for a classified intent
        
        Args:
            message: User's input message
            intent: Classified intent
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if intent not in self.intent_keywords:
            return 0.0
        
        message_lower = message.lower()
        keywords = self.intent_keywords[intent]
        
        # Count matching keywords
        matches = sum(1 for keyword in keywords if keyword in message_lower)
        
        # Calculate confidence (simple approach: matches / total keywords)
        confidence = min(matches / max(len(keywords) * 0.1, 1.0), 1.0)
        
        return confidence
