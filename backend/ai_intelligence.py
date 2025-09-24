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
        
        # Learn cuisine preferences - English only
        cuisine_keywords = {
            'turkish': ['turkish', 'ottoman', 'kebab', 'doner', 'traditional', 'local'],
            'italian': ['italian', 'pizza', 'pasta', 'mediterranean', 'risotto'],
            'seafood': ['fish', 'seafood', 'ocean', 'marine', 'meze'],
            'asian': ['asian', 'sushi', 'japanese', 'chinese', 'thai'],
            'european': ['european', 'french', 'german', 'continental'],
            'american': ['burger', 'american', 'fast food', 'bbq'],
            'vegetarian': ['vegetarian', 'vegan', 'salad', 'plant based'],
            'desserts': ['dessert', 'sweet', 'bakery', 'pastry', 'chocolate']
        }
        
        current_cuisines = preferences.get('preferred_cuisines', [])
        for cuisine, keywords in cuisine_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                if cuisine not in current_cuisines:
                    current_cuisines.append(cuisine)
                    updates['preferred_cuisines'] = current_cuisines
        
        # Learn budget preferences - English only
        if any(word in user_input_lower for word in ['cheap', 'budget', 'affordable', 'inexpensive']):
            updates['budget_level'] = 'budget'
        elif any(word in user_input_lower for word in ['expensive', 'luxury', 'high-end', 'premium', 'upscale']):
            updates['budget_level'] = 'luxury'
        elif any(word in user_input_lower for word in ['moderate', 'mid-range', 'reasonable']):
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
        
        # Learn travel style - English only
        if any(word in user_input_lower for word in ['family', 'kids', 'children', 'child']):
            updates['travel_style'] = 'family'
        elif any(word in user_input_lower for word in ['couple', 'romantic', 'date', 'partner']):
            updates['travel_style'] = 'couple'
        elif any(word in user_input_lower for word in ['business', 'work', 'meeting', 'corporate']):
            updates['travel_style'] = 'business'
        elif any(word in user_input_lower for word in ['group', 'friends', 'buddies']):
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
                'keywords': ['restaurant', 'restaurants', 'food', 'eat', 'eating', 'dining', 'cuisine', 'meal', 'meals',
                            'cafe', 'cafes', 'bistro', 'brasserie', 'pizza', 'burger', 'sushi', 'pasta', 
                            'turkish cuisine', 'traditional', 'local food', 'street food', 'fine dining', 
                            'casual dining', 'takeaway', 'delivery', 'brunch', 'breakfast', 'lunch', 'dinner',
                            'chef', 'kitchen', 'menu', 'taste', 'flavor', 'delicious', 'hungry', 'appetite',
                            'serve', 'serving', 'plate', 'dish', 'dishes', 'cooking', 'cook', 'cooked',
                            'eatery', 'diner', 'tavern', 'pub', 'grill', 'steakhouse', 'seafood', 'bakery',
                            'patisserie', 'dessert', 'sweet', 'ice cream', 'wine', 'bar', 'drinks'],
                'patterns': [
                    # English patterns - comprehensive
                    r'restaurants?\s+in\s+\w+', r'food\s+in\s+\w+', r'where\s+to\s+eat', r'where\s+can\s+i\s+eat',
                    r'dining\s+options', r'good\s+restaurants?', r'best\s+restaurants?', r'best\s+food', 
                    r'recommend.*restaurant', r'suggest.*restaurant', r'looking.*restaurant', r'find.*restaurant',
                    r'hungry', r'grab.*bite', r'place.*eat', r'somewhere.*eat', r'good.*food',
                    r'restaurant.*recommendations?', r'food.*recommendations?', r'places.*to.*dine',
                    r'what.*to.*eat', r'where.*should.*i.*eat', r'any.*good.*restaurants?',
                    r'restaurants?.*near', r'restaurants?.*around', r'restaurants?.*close.*to',
                    r'looking.*for.*food', r'want.*to.*eat', r'need.*food', r'craving.*food',
                    r'breakfast.*place', r'lunch.*spot', r'dinner.*restaurant', r'coffee.*shop',
                    r'i.*m.*hungry', r'we.*re.*hungry', r'getting.*hungry', r'feel.*like.*eating'
                ],
                'boost': 0.8  # High boost to strongly prioritize restaurant queries
            },
            'transportation_query': {
                'keywords': ['transport', 'transportation', 'metro', 'subway', 'bus', 'taxi', 'uber', 'lyft',
                            'how to get', 'directions', 'route', 'routes', 'travel', 'traveling', 'commute',
                            'ferry', 'boat', 'tram', 'train', 'public transport', 'getting around',
                            'navigation', 'way to', 'path to', 'journey', 'trip', 'ride', 'drive', 'walk',
                            'transport system', 'transport options', 'transport in', 'transportation in',
                            'getting around', 'how to travel', 'travel options', 'public transport',
                            'istanbulkart', 'transport card', 'transport tips', 'transport guide',
                            'airport transfer', 'airport transport', 'airport shuttle', 'airport connection',
                            'transfer options', 'transfer from', 'transfer to', 'airport to city', 'city to airport'],
                'patterns': [
                    r'how.*get.*from.*to', r'go.*from.*to', r'transport.*to', r'transportation.*to',
                    r'metro.*to', r'bus.*to', r'taxi.*to', r'directions.*to', r'route.*to',
                    r'how.*do.*i.*get.*to', r'what.*s.*the.*best.*way.*to', r'getting.*to',
                    r'travel.*from.*to', r'commute.*from.*to', r'journey.*from.*to',
                    r'transport.*in.*istanbul', r'transportation.*in.*istanbul', r'getting.*around.*istanbul',
                    r'how.*to.*get.*around', r'public.*transport.*in', r'transport.*system',
                    r'transport.*options', r'how.*to.*travel.*in', r'best.*way.*to.*travel',
                    r'airport.*transfer', r'airport.*transport', r'transfer.*from.*airport', r'transfer.*to.*airport',
                    r'airport.*to.*city', r'city.*to.*airport', r'airport.*shuttle', r'airport.*connection'
                ],
                'boost': 0.9  # High boost for transportation queries to compete with restaurants
            },
            'museum_query': {
                'keywords': ['museum', 'museums', 'exhibition', 'exhibitions', 'art', 'gallery', 'galleries',
                            'cultural', 'culture', 'history', 'historical', 'artifact', 'artifacts',
                            'collection', 'collections', 'display', 'displays', 'showcase', 'heritage',
                            'archaeological', 'archaeology', 'palace', 'palaces', 'monument', 'monuments',
                            'topkapi', 'hagia sophia', 'dolmabahce', 'istanbul modern', 'pera museum',
                            'sakip sabanci', 'chora', 'kariye', 'basilica cistern'],
                'patterns': [
                    r'museums?\s+in\s+\w+', r'art\s+gallery', r'art\s+galleries', r'exhibition',
                    r'cultural\s+sites', r'history\s+museum', r'historical\s+sites',
                    r'museums?\s+to\s+visit', r'best\s+museums?', r'museum\s+recommendations?',
                    r'art\s+museums?', r'history\s+museums?', r'archaeological\s+museums?',
                    r'which\s+museums?', r'must\s+see\s+museums?', r'famous\s+museums?',
                    r'top\s+museums?', r'museums?\s+worth\s+visiting', r'good\s+museums?',
                    r'museum\s+list', r'list.*museums?', r'show.*museums?', r'museums?\s+recommendations?',
                    r'best.*museums?.*to.*visit', r'museums?.*to.*visit.*in', r'visit.*museums?.*in'
                ],
                'boost': 0.6
            },
            'attraction_query': {
                'keywords': ['attraction', 'attractions', 'tourist', 'sightseeing', 'visit', 'visiting',
                            'places', 'landmarks', 'landmark', 'sight', 'sights', 'destinations',
                            'activities', 'activity', 'tours', 'tour', 'explore', 'exploring',
                            'must see', 'must visit', 'iconic', 'famous', 'popular', 'top rated',
                            'viewpoint', 'observation', 'scenic', 'photography', 'instagram'],
                'patterns': [
                    r'places\s+to\s+visit', r'tourist\s+attractions?', r'things\s+to\s+do',
                    r'sightseeing', r'landmarks?', r'what\s+to\s+see', r'where\s+to\s+go',
                    r'must\s+see', r'must\s+visit', r'top\s+attractions?', r'best\s+places',
                    r'famous\s+places', r'popular\s+attractions?', r'iconic\s+sites'
                ],
                'boost': 0.2
            },
            'shopping_query': {
                'keywords': ['shopping', 'shop', 'shops', 'buy', 'buying', 'purchase', 'market', 'markets',
                            'bazaar', 'bazaars', 'mall', 'malls', 'store', 'stores', 'boutique', 'boutiques',
                            'souvenirs', 'gifts', 'retail', 'commerce', 'merchandise', 'goods',
                            'shopping center', 'department store', 'outlet', 'vendors', 'stalls'],
                'patterns': [
                    r'shopping\s+in', r'where\s+to\s+shop', r'markets?', r'bazaars?', r'malls?',
                    r'buy.*souvenirs?', r'shopping.*areas?', r'best.*shops?', r'places.*to.*shop',
                    r'retail.*stores?', r'shopping.*districts?', r'commercial.*areas?'
                ],
                'boost': 0.2
            },
            'nightlife_query': {
                'keywords': ['nightlife', 'bars', 'bar', 'clubs', 'club', 'drinks', 'drinking', 'party',
                            'partying', 'night out', 'evening', 'cocktails', 'cocktail', 'lounge', 'lounges',
                            'pub', 'pubs', 'dance', 'dancing', 'music', 'live music', 'dj', 'entertainment',
                            'rooftop', 'terrace', 'wine bar', 'brewery', 'nightclub', 'discotheque'],
                'patterns': [
                    r'night.*life', r'bars?\s+in', r'clubs?\s+in', r'night\s+out', r'drinks?',
                    r'party', r'nightlife.*in', r'evening.*entertainment', r'cocktail.*bars?',
                    r'where.*to.*drink', r'best.*bars?', r'best.*clubs?', r'rooftop.*bars?'
                ],
                'boost': 0.2
            },
            'district_query': {
                'keywords': ['neighborhood', 'neighborhoods', 'district', 'districts', 'area', 'areas', 
                            'region', 'regions', 'zone', 'zones', 'quarter', 'quarters', 'sector', 'sectors',
                            'locality', 'localities', 'borough', 'boroughs', 'ward', 'wards',
                            'residential', 'commercial', 'historic', 'modern', 'trendy', 'popular'],
                'patterns': [
                    r'best\s+neighborhoods?', r'best\s+districts?', r'best\s+areas?',
                    r'neighborhoods?\s+in', r'districts?\s+in', r'areas?\s+in',
                    r'list\s+neighborhoods?', r'list\s+districts?', r'show.*neighborhoods?',
                    r'show.*districts?', r'which\s+neighborhood', r'which\s+district',
                    r'residential\s+areas?', r'popular\s+areas?', r'trendy\s+areas?',
                    r'different\s+areas?', r'various\s+districts?', r'all\s+neighborhoods?'
                ],
                'boost': 0.1  # Reduced boost to prevent overshadowing restaurant queries
            }
        }
    
    def recognize_intent(self, user_input: str, context: Optional[Dict] = None) -> Tuple[str, float]:
        """Recognize intent with confidence scoring and context awareness"""
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        # Pre-process: Check for explicit restaurant indicators
        restaurant_indicators = ['restaurant', 'restaurants', 'food', 'eat', 'eating', 'dining', 'meal', 'cafe']
        has_restaurant_indicator = any(indicator in user_input_lower for indicator in restaurant_indicators)
        
        for intent, config in self.intent_patterns.items():
            score = 0.0
            
            # Keyword matching with enhanced fuzzy logic
            keyword_matches = 0
            for keyword in config['keywords']:
                # Check exact match first (higher weight)
                if keyword in user_input_lower:
                    score += 0.25
                    keyword_matches += 1
                else:
                    # Use fuzzy matching for typos and variations
                    words = user_input_lower.split()
                    best_match = process.extractOne(keyword, words)
                    if best_match and best_match[1] > 75:  # 75% similarity threshold
                        score += 0.15 * (best_match[1] / 100)
                        keyword_matches += 1
            
            # Pattern matching with higher weight
            pattern_matches = 0
            for pattern in config['patterns']:
                if re.search(pattern, user_input_lower):
                    score += config['boost']
                    pattern_matches += 1
            
            # Special boost for restaurant queries with location context
            if intent == 'restaurant_search' and has_restaurant_indicator:
                if any(district in user_input_lower for district in ['beyoglu', 'sultanahmet', 'kadikoy', 'galata', 'besiktas', 'taksim', 'fatih']):
                    score += 0.4  # Strong boost for restaurant + location combination
            
            # Special boost for museum queries when "museum" is explicitly mentioned
            if intent == 'museum_query' and ('museum' in user_input_lower or 'museums' in user_input_lower):
                score += 0.5  # Strong boost when museum is explicitly mentioned
                # Extra boost for common museum query patterns
                if any(word in user_input_lower for word in ['best', 'top', 'visit', 'see', 'good', 'famous']):
                    score += 0.3
            
            # Context boost
            if context:
                if context.get('current_intent') == intent:
                    score += 0.15
                # If previous query was about restaurants, slightly boost restaurant intent
                if context.get('previous_intent') == 'restaurant_search' and intent == 'restaurant_search':
                    score += 0.1
            
            # Multi-keyword bonus (more keywords = higher confidence)
            if keyword_matches > 1:
                score += 0.1 * (keyword_matches - 1)
            
            # Pattern + keyword combination bonus
            if pattern_matches > 0 and keyword_matches > 0:
                score += 0.2
            
            intent_scores[intent] = score
        
        # Handle multi-intent queries and apply smart thresholds
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_intents and sorted_intents[0][1] > 0.2:  # Slightly higher threshold for better accuracy
            return sorted_intents[0][0], sorted_intents[0][1]
        else:
            # If we have restaurant indicators but low confidence, default to restaurant search
            if has_restaurant_indicator:
                return 'restaurant_search', 0.3
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
        
        # Extract time references - English only
        time_patterns = [
            (r'morning|dawn|early', 'morning'),
            (r'lunch|noon|midday', 'lunch'),
            (r'dinner|evening|supper', 'dinner'),
            (r'night|late|midnight', 'night')
        ]
        
        for pattern, time_ref in time_patterns:
            if re.search(pattern, user_input_lower):
                entities['time_references'].append(time_ref)
        
        return entities
    
    def analyze_query_context(self, user_input: str) -> Dict[str, Any]:
        """Analyze query for contextual information beyond basic intent"""
        user_input_lower = user_input.lower()
        
        analysis = {
            'locations': [],
            'cuisine_types': [],
            'price_indicators': [],
            'time_context': [],
            'group_context': None,
            'urgency_level': 'normal',
            'query_complexity': 'simple'
        }
        
        # Enhanced location detection - English only
        istanbul_locations = {
            # Main districts
            'sultanahmet': ['sultanahmet', 'sultan ahmet', 'old city', 'historic peninsula'],
            'beyoglu': ['beyoglu', 'pera', 'galatasaray', 'istiklal', 'istiklal street'],
            'galata': ['galata', 'karakoy', 'galata tower'],
            'kadikoy': ['kadikoy', 'moda', 'fenerbahce'],
            'besiktas': ['besiktas', 'ortakoy', 'dolmabahce'],
            'uskudar': ['uskudar', 'camlica', 'camlica hill'],
            'fatih': ['fatih', 'fener', 'balat', 'golden horn'],
            'sisli': ['sisli', 'nisantasi', 'osmanbey', 'mecidiyekoy'],
            'bakirkoy': ['bakirkoy', 'yesilkoy', 'ataturk airport'],
            'eminonu': ['eminonu', 'sirkeci', 'grand bazaar', 'spice bazaar'],
            
            # Neighborhoods and landmarks
            'taksim': ['taksim', 'taksim square', 'gezi park'],
            'levent': ['levent', 'etiler', 'bebek', 'maslak'],
            'bosphorus': ['bosphorus', 'bosporus', 'strait', 'waterfront'],
            'asian_side': ['asian side', 'anatolian side', 'asia'],
            'european_side': ['european side', 'europe', 'european istanbul']
        }
        
        for main_location, variants in istanbul_locations.items():
            if any(variant in user_input_lower for variant in variants):
                analysis['locations'].append(main_location)
        
        # Enhanced cuisine detection - English only
        cuisine_mapping = {
            'turkish': ['turkish', 'ottoman', 'traditional', 'local', 'authentic'],
            'kebab': ['kebab', 'doner', 'adana', 'urfa', 'grilled meat'],
            'seafood': ['seafood', 'fish', 'meze', 'ocean', 'marine'],
            'italian': ['italian', 'pizza', 'pasta', 'risotto', 'mediterranean'],
            'asian': ['asian', 'sushi', 'japanese', 'chinese', 'thai', 'korean'],
            'mediterranean': ['mediterranean', 'greek', 'middle eastern'],
            'european': ['european', 'french', 'german', 'continental'],
            'american': ['american', 'burger', 'fast food', 'bbq', 'steakhouse'],
            'vegetarian': ['vegetarian', 'vegan', 'plant based', 'salad', 'healthy'],
            'dessert': ['dessert', 'sweet', 'bakery', 'pastry', 'ice cream', 'chocolate'],
            'breakfast': ['breakfast', 'brunch', 'morning', 'coffee', 'cafe']
        }
        
        for cuisine, keywords in cuisine_mapping.items():
            if any(keyword in user_input_lower for keyword in keywords):
                analysis['cuisine_types'].append(cuisine)
        
        # Price/budget detection - English only
        budget_keywords = {
            'budget': ['cheap', 'budget', 'affordable', 'inexpensive', 'low cost', 'economical'],
            'mid_range': ['moderate', 'mid-range', 'reasonable', 'fair price', 'average price'],
            'luxury': ['expensive', 'luxury', 'high-end', 'premium', 'upscale', 'fine dining', 'elite']
        }
        
        for price_level, keywords in budget_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                analysis['price_indicators'].append(price_level)
        
        # Time context detection - English only
        time_patterns = {
            'breakfast': ['breakfast', 'morning', 'early morning', 'dawn', 'am'],
            'lunch': ['lunch', 'noon', 'midday', 'afternoon', 'lunchtime'],
            'dinner': ['dinner', 'evening', 'night', 'supper', 'pm'],
            'late_night': ['late night', 'midnight', 'after midnight', 'very late'],
            'weekend': ['weekend', 'saturday', 'sunday', 'weekends']
        }
        
        for time_type, keywords in time_patterns.items():
            if any(keyword in user_input_lower for keyword in keywords):
                analysis['time_context'].append(time_type)
        
        # Group context detection - English only
        group_indicators = {
            'family': ['family', 'kids', 'children', 'child', 'baby', 'toddler', 'parents'],
            'couple': ['couple', 'romantic', 'date', 'partner', 'spouse', 'boyfriend', 'girlfriend'],
            'business': ['business', 'work', 'meeting', 'conference', 'corporate', 'colleagues'],
            'friends': ['friends', 'group', 'buddies', 'mates', 'gang', 'crew'],
            'solo': ['alone', 'solo', 'myself', 'individual', 'single', 'by myself']
        }
        
        for group_type, keywords in group_indicators.items():
            if any(keyword in user_input_lower for keyword in keywords):
                analysis['group_context'] = group_type
                break
        
        # Urgency detection - English only
        urgency_keywords = {
            'high': ['urgent', 'now', 'immediately', 'asap', 'right now', 'quickly', 'fast'],
            'low': ['later', 'sometime', 'maybe', 'eventually', 'when possible', 'no rush']
        }
        
        for urgency, keywords in urgency_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                analysis['urgency_level'] = urgency
                break
        
        # Query complexity assessment - English only
        word_count = len(user_input.split())
        question_words = ['what', 'where', 'when', 'how', 'which', 'who', 'why', 'can', 'could', 'would', 'should']
        has_questions = any(qw in user_input_lower for qw in question_words)
        
        if word_count > 10 or len(analysis['locations']) > 1 or len(analysis['cuisine_types']) > 1:
            analysis['query_complexity'] = 'complex'
        elif has_questions or word_count > 5:
            analysis['query_complexity'] = 'moderate'
        
        return analysis

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

class SavedSessionManager:
    """Manages saved chat sessions with like/unlike functionality"""
    
    def __init__(self):
        self.saved_sessions = {}  # In-memory storage for saved sessions
    
    def save_session(self, session_id: str, messages: List[Dict], user_ip: Optional[str] = None) -> bool:
        """Save a chat session"""
        try:
            session_data = {
                'session_id': session_id,
                'messages': messages,
                'saved_at': datetime.utcnow(),
                'user_ip': user_ip,
                'title': self._generate_session_title(messages)
            }
            self.saved_sessions[session_id] = session_data
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def get_saved_sessions(self, user_ip: Optional[str] = None) -> List[Dict]:
        """Get all saved sessions, optionally filtered by user IP"""
        try:
            sessions = []
            for session_id, session_data in self.saved_sessions.items():
                if user_ip is None or session_data.get('user_ip') == user_ip:
                    sessions.append({
                        'id': session_id,
                        'title': session_data['title'],
                        'saved_at': session_data['saved_at'].isoformat(),
                        'message_count': len(session_data['messages'])
                    })
            # Sort by saved_at descending (most recent first)
            return sorted(sessions, key=lambda x: x['saved_at'], reverse=True)
        except Exception as e:
            print(f"Error getting saved sessions: {e}")
            return []
    
    def get_session_details(self, session_id: str) -> Optional[Dict]:
        """Get full details of a saved session"""
        return self.saved_sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session"""
        try:
            if session_id in self.saved_sessions:
                del self.saved_sessions[session_id]
                return True
            return False
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def _generate_session_title(self, messages: List[Dict]) -> str:
        """Generate a meaningful title from the first user message"""
        try:
            for msg in messages:
                if msg.get('role') == 'user' and msg.get('content'):
                    content = msg['content'].strip()
                    # Take first 50 characters and add ellipsis if longer
                    if len(content) > 50:
                        return content[:50].strip() + "..."
                    return content
            return "Untitled Chat"
        except Exception:
            return "Untitled Chat"

# Global instance for saved sessions
saved_session_manager = SavedSessionManager()
