# Simplified AI Intelligence Services for Enhanced Istanbul Travel Guide
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from fuzzywuzzy import fuzz, process
import re

class SimpleSessionManager:
    """Simple session manager using basic dictionary storage"""
    
    def __init__(self):
        self.sessions = {}  # In-memory storage for simplicity
        self.preferences = {}
        self.conversation_contexts = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None, user_ip: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'created_at': datetime.utcnow(),
                'last_activity': datetime.utcnow(),
                'user_ip': user_ip,
                'is_active': True
            }
            # Initialize default preferences
            self.preferences[session_id] = {
                'preferred_cuisines': [],
                'avoided_cuisines': [],
                'budget_level': 'any',
                'interests': [],
                'travel_style': 'solo',
                'preferred_districts': [],
                'confidence_score': 0.0,
                'total_interactions': 0
            }
            # Initialize conversation context
            self.conversation_contexts[session_id] = {
                'current_intent': 'initial',
                'current_location': '',
                'previous_locations': [],
                'topic_history': [],
                'conversation_stage': 'initial',
                'context_data': {}
            }
        else:
            self.sessions[session_id]['last_activity'] = datetime.utcnow()
        
        return session_id
    
    def get_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        return self.preferences.get(session_id, {})
    
    def update_preferences(self, session_id: str, updates: Dict[str, Any]):
        """Update user preferences"""
        if session_id in self.preferences:
            self.preferences[session_id].update(updates)
            self.preferences[session_id]['total_interactions'] += 1
            self.preferences[session_id]['confidence_score'] = min(1.0, 
                self.preferences[session_id].get('confidence_score', 0.0) + 0.1)
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation context"""
        return self.conversation_contexts.get(session_id, {})
    
    def update_context(self, session_id: str, updates: Dict[str, Any]):
        """Update conversation context"""
        if session_id in self.conversation_contexts:
            self.conversation_contexts[session_id].update(updates)

class EnhancedPreferenceManager:
    """Manages and learns user preferences"""
    
    def __init__(self, session_manager: SimpleSessionManager):
        self.session_manager = session_manager
    
    def learn_from_query(self, session_id: str, user_input: str, detected_intent: str):
        """Learn preferences from user queries"""
        preferences = self.session_manager.get_preferences(session_id)
        updates = {}
        user_input_lower = user_input.lower()
        
        # Learn cuisine preferences
        cuisine_keywords = {
            'turkish': ['turkish', 'ottoman', 'kebab', 'döner', 'traditional', 'lokanta'],
            'italian': ['italian', 'pizza', 'pasta', 'mediterranean'],
            'seafood': ['fish', 'seafood', 'balık', 'deniz'],
            'asian': ['asian', 'sushi', 'japanese', 'chinese'],
            'european': ['european', 'french', 'german'],
            'american': ['burger', 'american', 'fast food'],
            'vegetarian': ['vegetarian', 'vegan', 'salad'],
            'desserts': ['dessert', 'sweet', 'bakery', 'pastane', 'tatlı']
        }
        
        current_cuisines = preferences.get('preferred_cuisines', [])
        for cuisine, keywords in cuisine_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                if cuisine not in current_cuisines:
                    current_cuisines.append(cuisine)
                    updates['preferred_cuisines'] = current_cuisines
        
        # Learn budget preferences
        if any(word in user_input_lower for word in ['cheap', 'budget', 'affordable', 'ucuz']):
            updates['budget_level'] = 'budget'
        elif any(word in user_input_lower for word in ['expensive', 'luxury', 'high-end', 'premium', 'pahalı']):
            updates['budget_level'] = 'luxury'
        elif any(word in user_input_lower for word in ['moderate', 'mid-range', 'orta']):
            updates['budget_level'] = 'mid-range'
        
        # Learn interests from intent
        interest_mapping = {
            'restaurant_search': 'dining',
            'museum_query': 'museums',
            'attraction_query': 'attractions',
            'transportation_query': 'transportation',
            'nightlife_query': 'nightlife',
            'shopping_query': 'shopping',
            'culture_query': 'culture'
        }
        
        if detected_intent in interest_mapping:
            interest = interest_mapping[detected_intent]
            current_interests = preferences.get('interests', [])
            if interest not in current_interests:
                current_interests.append(interest)
                updates['interests'] = current_interests
        
        # Learn district preferences
        districts = ['sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar', 
                    'fatih', 'taksim', 'karakoy', 'ortakoy', 'bebek']
        current_districts = preferences.get('preferred_districts', [])
        for district in districts:
            if district in user_input_lower and district not in current_districts:
                current_districts.append(district)
                updates['preferred_districts'] = current_districts
        
        # Learn travel style
        if any(word in user_input_lower for word in ['family', 'kids', 'children', 'aile']):
            updates['travel_style'] = 'family'
        elif any(word in user_input_lower for word in ['couple', 'romantic', 'date', 'çift']):
            updates['travel_style'] = 'couple'
        elif any(word in user_input_lower for word in ['business', 'work', 'meeting', 'iş']):
            updates['travel_style'] = 'business'
        elif any(word in user_input_lower for word in ['group', 'friends', 'arkadaş']):
            updates['travel_style'] = 'group'
        
        if updates:
            self.session_manager.update_preferences(session_id, updates)
    
    def get_personalized_filter(self, session_id: str) -> Dict[str, Any]:
        """Get filter criteria based on user preferences"""
        return self.session_manager.get_preferences(session_id)

class IntelligentIntentRecognizer:
    """Enhanced intent recognition with fuzzy matching and context"""
    
    def __init__(self):
        self.intent_patterns = {
            'restaurant_search': {
                'keywords': ['restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal', 'lokanta', 'yemek'],
                'patterns': [
                    r'restaurants?\s+in\s+\w+', r'food\s+in\s+\w+', r'where\s+to\s+eat',
                    r'dining\s+options', r'good\s+restaurants?', r'best\s+food', r'yemek\s+nerede'
                ],
                'boost': 0.2
            },
            'transportation_query': {
                'keywords': ['transport', 'metro', 'bus', 'taxi', 'how to get', 'directions', 'ulaşım'],
                'patterns': [
                    r'how.*get.*from.*to', r'go.*from.*to', r'transport.*to',
                    r'metro.*to', r'bus.*to', r'directions', r'nasıl\s+giderim'
                ],
                'boost': 0.3
            },
            'museum_query': {
                'keywords': ['museum', 'exhibition', 'art', 'gallery', 'cultural', 'history', 'müze'],
                'patterns': [
                    r'museums?\s+in\s+\w+', r'art\s+gallery', r'exhibition',
                    r'cultural\s+sites', r'history\s+museum', r'müzeler'
                ],
                'boost': 0.2
            },
            'attraction_query': {
                'keywords': ['attraction', 'tourist', 'sightseeing', 'visit', 'places', 'landmarks', 'gezilecek'],
                'patterns': [
                    r'places\s+to\s+visit', r'tourist\s+attractions?', r'things\s+to\s+do',
                    r'sightseeing', r'landmarks?', r'what\s+to\s+see', r'gezilecek\s+yerler'
                ],
                'boost': 0.2
            },
            'shopping_query': {
                'keywords': ['shopping', 'shop', 'buy', 'market', 'bazaar', 'mall', 'alışveriş'],
                'patterns': [
                    r'shopping\s+in', r'where\s+to\s+shop', r'markets?',
                    r'bazaars?', r'buy.*souvenirs?', r'alışveriş'
                ],
                'boost': 0.2
            },
            'nightlife_query': {
                'keywords': ['nightlife', 'bars', 'clubs', 'drinks', 'party', 'night out', 'gece hayatı'],
                'patterns': [
                    r'night.*life', r'bars?\s+in', r'clubs?\s+in',
                    r'night\s+out', r'drinks?', r'party', r'gece\s+hayatı'
                ],
                'boost': 0.2
            }
        }
    
    def recognize_intent(self, user_input: str, context: Optional[Dict] = None) -> Tuple[str, float]:
        """Recognize intent with confidence scoring"""
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0.0
            
            # Keyword matching with fuzzy logic
            for keyword in config['keywords']:
                # Check exact match first
                if keyword in user_input_lower:
                    score += 0.2
                else:
                    # Use fuzzy matching for typos
                    words = user_input_lower.split()
                    best_match = process.extractOne(keyword, words)
                    if best_match and best_match[1] > 75:  # 75% similarity threshold
                        score += 0.1 * (best_match[1] / 100)
            
            # Pattern matching
            for pattern in config['patterns']:
                if re.search(pattern, user_input_lower):
                    score += config['boost']
            
            # Context boost
            if context and context.get('current_intent') == intent:
                score += 0.1
            
            intent_scores[intent] = score
        
        # Handle multi-intent queries
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_intents and sorted_intents[0][1] > 0.15:
            return sorted_intents[0][0], sorted_intents[0][1]
        else:
            return 'general_query', 0.1
    
    def extract_entities(self, user_input: str) -> Dict[str, List[str]]:
        """Extract entities from user input"""
        entities: Dict[str, List[str]] = {
            'locations': [],
            'time_references': [],
            'cuisine_types': [],
            'budget_indicators': []
        }
        
        user_input_lower = user_input.lower()
        
        # Extract locations
        locations = [
            'sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
            'fatih', 'taksim', 'karakoy', 'ortakoy', 'bebek', 'sisli'
        ]
        
        for location in locations:
            if location in user_input_lower:
                entities['locations'].append(location)
        
        # Extract time references
        time_patterns = [
            (r'morning|sabah', 'morning'),
            (r'lunch|öğle', 'lunch'),
            (r'dinner|akşam', 'dinner'),
            (r'night|gece', 'night')
        ]
        
        for pattern, time_ref in time_patterns:
            if re.search(pattern, user_input_lower):
                entities['time_references'].append(time_ref)
        
        return entities

class PersonalizedRecommendationEngine:
    """Generate personalized recommendations"""
    
    def __init__(self, session_manager: SimpleSessionManager):
        self.session_manager = session_manager
    
    def enhance_recommendations(self, session_id: str, base_results: List[Dict]) -> List[Dict]:
        """Enhance recommendations with personalization"""
        preferences = self.session_manager.get_preferences(session_id)
        
        if not preferences or not base_results:
            return base_results
        
        enhanced_results = []
        current_time = datetime.now().hour
        
        for item in base_results:
            enhanced_item = item.copy()
            score = self._calculate_personalization_score(item, preferences, current_time)
            enhanced_item['personalization_score'] = score
            enhanced_item['recommendation_reason'] = self._generate_reason(item, preferences)
            enhanced_results.append(enhanced_item)
        
        # Sort by personalization score
        return sorted(enhanced_results, key=lambda x: x.get('personalization_score', 0), reverse=True)
    
    def _calculate_personalization_score(self, item: Dict, preferences: Dict, current_time: int) -> float:
        """Calculate personalization score"""
        score = 0.5  # Base score
        
        # Cuisine matching
        item_text = f"{item.get('name', '')} {item.get('category', '')} {item.get('cuisine', '')}".lower()
        for cuisine in preferences.get('preferred_cuisines', []):
            if cuisine in item_text:
                score += 0.2
        
        # District matching
        for district in preferences.get('preferred_districts', []):
            if district in item_text:
                score += 0.15
        
        # Time-based scoring
        if current_time < 11 and any(word in item_text for word in ['cafe', 'breakfast', 'kahve']):
            score += 0.1
        elif current_time >= 18 and 'restaurant' in item_text:
            score += 0.1
        
        # Budget matching
        budget_level = preferences.get('budget_level', 'any')
        price_level = item.get('price_level', 2)
        
        if budget_level == 'budget' and price_level and price_level <= 2:
            score += 0.1
        elif budget_level == 'luxury' and price_level and price_level >= 3:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_reason(self, item: Dict, preferences: Dict) -> str:
        """Generate recommendation reason"""
        reasons = []
        
        item_text = f"{item.get('name', '')} {item.get('category', '')}".lower()
        
        # Check cuisine match
        matching_cuisines = [c for c in preferences.get('preferred_cuisines', []) if c in item_text]
        if matching_cuisines:
            reasons.append(f"matches your interest in {', '.join(matching_cuisines)} cuisine")
        
        # Check district match
        matching_districts = [d for d in preferences.get('preferred_districts', []) if d in item_text]
        if matching_districts:
            reasons.append(f"located in your preferred {', '.join(matching_districts)} area")
        
        if not reasons:
            return "popular choice among visitors"
        
        return "Recommended because it " + " and ".join(reasons)

# Global instances for the enhanced AI system
session_manager = SimpleSessionManager()
preference_manager = EnhancedPreferenceManager(session_manager)
intent_recognizer = IntelligentIntentRecognizer()
recommendation_engine = PersonalizedRecommendationEngine(session_manager)
