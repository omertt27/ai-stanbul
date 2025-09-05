"""
Enhanced AI Istanbul Chatbot Improvements
Addresses: Context Awareness, Query Understanding, Knowledge Scope, Follow-up Questions
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ConversationContext:
    """Store conversation context for better continuity"""
    session_id: str
    previous_queries: List[str]
    previous_responses: List[str]
    mentioned_places: List[str]
    user_preferences: Dict[str, Any]
    last_recommendation_type: Optional[str]
    conversation_topics: List[str]
    user_location: Optional[str]
    timestamp: datetime

class EnhancedContextManager:
    """Manages conversation context and memory"""
    
    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}
        self.max_context_age = timedelta(hours=2)  # Context expires after 2 hours
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session"""
        if session_id in self.contexts:
            context = self.contexts[session_id]
            # Check if context is still valid
            if datetime.now() - context.timestamp < self.max_context_age:
                return context
            else:
                # Remove expired context
                del self.contexts[session_id]
        return None
    
    def update_context(self, session_id: str, query: str, response: str, 
                      places: List[str] = None, topic: str = None, 
                      location: str = None) -> None:
        """Update conversation context with new information"""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(
                session_id=session_id,
                previous_queries=[],
                previous_responses=[],
                mentioned_places=[],
                user_preferences={},
                last_recommendation_type=None,
                conversation_topics=[],
                user_location=None,
                timestamp=datetime.now()
            )
        
        context = self.contexts[session_id]
        context.previous_queries.append(query)
        context.previous_responses.append(response)
        
        # Enhanced location extraction from query
        istanbul_locations = [
            'sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
            'taksim', 'karakoy', 'ortakoy', 'bebek', 'fatih', 'sisli', 'eminonu'
        ]
        
        # Check for locations mentioned in the current query
        query_locations = []
        for loc in istanbul_locations:
            if loc in query.lower():
                query_locations.append(loc.title())
        
        # Add provided places and extracted query locations
        all_new_places = []
        if places:
            all_new_places.extend(places)
        if query_locations:
            all_new_places.extend(query_locations)
        
        if all_new_places:
            context.mentioned_places.extend(all_new_places)
            # Keep only unique places and maintain order (recent first)
            seen = set()
            unique_places = []
            for place in reversed(context.mentioned_places):
                if place.lower() not in seen:
                    seen.add(place.lower())
                    unique_places.append(place)
            context.mentioned_places = list(reversed(unique_places))
        
        if topic:
            context.conversation_topics.append(topic)
            context.last_recommendation_type = topic
        
        if location:
            context.user_location = location
        
        # Keep only last 10 exchanges to prevent memory bloat
        if len(context.previous_queries) > 10:
            context.previous_queries = context.previous_queries[-10:]
            context.previous_responses = context.previous_responses[-10:]
        
        context.timestamp = datetime.now()
    
    def extract_preferences(self, query: str, context: ConversationContext) -> None:
        """Extract user preferences from queries"""
        # Budget preferences
        if any(word in query.lower() for word in ['cheap', 'budget', 'affordable', 'inexpensive']):
            context.user_preferences['budget'] = 'low'
        elif any(word in query.lower() for word in ['expensive', 'luxury', 'high-end', 'premium']):
            context.user_preferences['budget'] = 'high'
        elif any(word in query.lower() for word in ['moderate', 'mid-range', 'reasonable']):
            context.user_preferences['budget'] = 'moderate'
        
        # Cuisine preferences
        cuisine_keywords = {
            'turkish': ['turkish', 'ottoman', 'traditional', 'local'],
            'seafood': ['seafood', 'fish', 'marine', 'ocean'],
            'vegetarian': ['vegetarian', 'vegan', 'plant-based'],
            'international': ['international', 'foreign', 'western']
        }
        
        for cuisine, keywords in cuisine_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                context.user_preferences['cuisine'] = cuisine
        
        # Dietary restrictions
        if any(word in query.lower() for word in ['allergy', 'allergic', 'gluten-free', 'dairy-free']):
            context.user_preferences['dietary_restrictions'] = True

class EnhancedQueryUnderstanding:
    """Improved query understanding with better NLP"""
    
    def __init__(self):
        # Common misspellings and corrections
        self.corrections = {
            'restaurent': 'restaurant',
            'restarunt': 'restaurant',
            'musium': 'museum',
            'musem': 'museum',
            'galery': 'gallery',
            'galeries': 'galleries',
            'allergys': 'allergies',
            'alergies': 'allergies',
            'recomend': 'recommend',
            'recomendation': 'recommendation',
            'plases': 'places',
            'atractions': 'attractions',
            'atraction': 'attraction',
            'historicall': 'historical',
            'beautifull': 'beautiful',
            'wonderfull': 'wonderful'
        }
        
        # Synonyms for better understanding
        self.synonyms = {
            'restaurant': ['eatery', 'diner', 'bistro', 'cafe', 'dining', 'food place'],
            'museum': ['gallery', 'exhibition', 'cultural center', 'art center'],
            'attraction': ['landmark', 'sight', 'place of interest', 'tourist spot'],
            'transportation': ['transport', 'travel', 'getting around', 'commute'],
            'shopping': ['buying', 'purchasing', 'markets', 'stores'],
            'nightlife': ['bars', 'clubs', 'evening', 'night out']
        }
    
    def correct_and_enhance_query(self, query: str) -> str:
        """Correct common misspellings and enhance query understanding"""
        # Fix common misspellings
        corrected = query
        for mistake, correction in self.corrections.items():
            corrected = re.sub(r'\b' + mistake + r'\b', correction, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def extract_intent_and_entities(self, query: str, context: Optional[ConversationContext] = None) -> Dict[str, Any]:
        """Enhanced intent and entity extraction with validation"""
        query_lower = query.lower()
        
        # First, validate the query for logical and geographic errors
        validation_result = self.validate_query_logic(query)
        
        # If query has serious issues, return error intent
        if not validation_result['is_valid']:
            return {
                'intent': 'validation_error',
                'confidence': 0,
                'entities': {},
                'validation_errors': validation_result,
                'all_scores': {},
                'context_location': None
            }
        
        # Check for ambiguous queries that might lead to wrong answers
        ambiguity_result = self.detect_ambiguous_queries(query)
        
        # If query is too ambiguous, return clarification intent
        if ambiguity_result['is_ambiguous']:
            return {
                'intent': 'clarification_needed',
                'confidence': 0,
                'entities': {},
                'ambiguity_info': ambiguity_result,
                'all_scores': {},
                'context_location': None
            }
        
        # Intent classification with confidence scores
        intents = {
            'restaurant_search': 0,
            'museum_inquiry': 0,
            'transportation_info': 0,
            'place_recommendation': 0,
            'shopping_info': 0,
            'nightlife_info': 0,
            'cultural_info': 0,
            'accommodation_info': 0,
            'follow_up_question': 0,
            'general_travel_info': 0
        }
        
        # Calculate intent scores with improved pattern matching
        restaurant_keywords = [r'\brestaurant\b', r'\beat\b', r'\bfood\b', r'\bdining\b', r'\bmeal\b', 
                              r'\bbreakfast\b', r'\blunch\b', r'\bdinner\b', r'\bcafe\b', r'\bbistro\b']
        museum_keywords = [r'\bmuseum\b', r'\bgallery\b', r'\bexhibition\b', r'\bart\b', r'\bhistory\b', 
                          r'\bcultural\b', r'\bartwork\b', r'\bpaintings\b']
        transport_keywords = [r'\btransport\b', r'\bmetro\b', r'\bbus\b', r'\btaxi\b', r'\bferry\b', 
                             r'\bhow to get\b', r'\btravel to\b', r'\bcommute\b', r'\bdirections\b']
        place_keywords = [r'\bplace\b', r'\battraction\b', r'\bvisit\b', r'\bsee\b', r'\blandmark\b', 
                         r'\btourist\b', r'\bsightseeing\b', r'\bexplore\b']
        
        # Use regex word boundaries to avoid false positives
        for pattern in restaurant_keywords:
            if re.search(pattern, query_lower):
                intents['restaurant_search'] += 1
        
        for pattern in museum_keywords:
            if re.search(pattern, query_lower):
                intents['museum_inquiry'] += 1
        
        for pattern in transport_keywords:
            if re.search(pattern, query_lower):
                intents['transportation_info'] += 1
        
        for pattern in place_keywords:
            if re.search(pattern, query_lower):
                intents['place_recommendation'] += 1
        
        # Check for follow-up patterns
        follow_up_patterns = [
            r'\b(what about|how about|and)\b',
            r'\b(those|these|them|that|this)\b',
            r'\b(previously|before|earlier|you mentioned|you said)\b',
            r'\b(more|additional|other|another)\b',
            r'\b(tip|tips|advice|suggest)\b'
        ]
        
        for pattern in follow_up_patterns:
            if re.search(pattern, query_lower):
                intents['follow_up_question'] += 1
        
        # Context-aware intent adjustment and location inference
        recent_locations = []
        if context:
            # Get recent locations from context
            recent_locations = context.mentioned_places[-3:] if context.mentioned_places else []
            
            # If user previously asked about restaurants and now asks follow-up
            if context.last_recommendation_type == 'restaurant' and intents['follow_up_question'] > 0:
                intents['restaurant_search'] += 2
            
            # ENHANCED: Check for context-aware location queries
            # If user mentioned a location recently and now asks generic questions
            if recent_locations and any(generic in query_lower for generic in ['places', 'restaurants', 'attractions', 'food', 'where']):
                # Boost relevant intent based on what they're asking
                if any(word in query_lower for word in ['place', 'attraction', 'visit', 'see']):
                    intents['place_recommendation'] += 3
                elif any(word in query_lower for word in ['restaurant', 'food', 'eat', 'dining']):
                    intents['restaurant_search'] += 3
        
        # Find highest scoring intent
        primary_intent = max(intents.items(), key=lambda x: x[1])
        
        # Extract entities
        entities = self.extract_entities(query, context)
        
        return {
            'intent': primary_intent[0] if primary_intent[1] > 0 else 'general_travel_info',
            'confidence': primary_intent[1],
            'entities': entities,
            'all_scores': intents,
            'context_location': recent_locations[-1] if recent_locations else None  # Most recent location
        }
    
    def extract_entities(self, query: str, context: Optional[ConversationContext] = None) -> Dict[str, List[str]]:
        """Extract entities like locations, preferences, etc."""
        entities = {
            'locations': [],
            'cuisine_types': [],
            'price_range': [],
            'dietary_restrictions': [],
            'times': [],
            'features': []
        }
        
        # Location patterns with validation
        istanbul_locations = [
            'sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
            'taksim', 'karakoy', 'ortakoy', 'bebek', 'fatih', 'sisli', 'eminonu'
        ]
        
        # Non-Istanbul locations that might cause confusion
        non_istanbul_locations = [
            'paris', 'london', 'rome', 'athens', 'madrid', 'berlin', 'moscow',
            'new york', 'manhattan', 'brooklyn', 'chicago', 'los angeles',
            'tokyo', 'bangkok', 'dubai', 'cairo', 'casablanca'
        ]
        
        # Check for non-Istanbul locations first
        for location in non_istanbul_locations:
            if location in query.lower():
                # This is a geographic error - don't add to entities
                return entities
        
        # Check explicit locations in current query
        for location in istanbul_locations:
            if re.search(r'\b' + location + r'\b', query.lower()):
                entities['locations'].append(location.title())
        
        # ENHANCED: If no explicit location in query but context has recent locations
        # and the query is generic (places, restaurants, etc.)
        if not entities['locations'] and context and context.mentioned_places:
            generic_queries = ['places', 'restaurants', 'attractions', 'museums', 'food', 'where to go', 'what to see']
            if any(generic in query.lower() for generic in generic_queries):
                # Use the most recent location from context
                recent_location = context.mentioned_places[-1]
                entities['locations'].append(recent_location)
                # Mark this as context-derived so we can handle it specially
                entities['context_derived_location'] = [recent_location]
        
        # Cuisine types
        cuisine_patterns = {
            'turkish': ['turkish', 'ottoman', 'traditional', 'local'],
            'seafood': ['seafood', 'fish', 'marine'],
            'international': ['international', 'italian', 'chinese', 'japanese'],
            'vegetarian': ['vegetarian', 'vegan']
        }
        
        for cuisine, keywords in cuisine_patterns.items():
            if any(keyword in query.lower() for keyword in keywords):
                entities['cuisine_types'].append(cuisine)
        
        # Price range
        if any(word in query.lower() for word in ['cheap', 'budget', 'affordable']):
            entities['price_range'].append('budget')
        elif any(word in query.lower() for word in ['expensive', 'luxury', 'high-end']):
            entities['price_range'].append('expensive')
        elif any(word in query.lower() for word in ['moderate', 'mid-range']):
            entities['price_range'].append('moderate')
        
        return entities
    
    def validate_query_logic(self, query: str) -> Dict[str, Any]:
        """Validate query for logical inconsistencies and geographical errors"""
        query_lower = query.lower()
        
        validation_result = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'error_type': None
        }
        
        # Geographic validation - prevent confusion with other cities
        geographic_issues = [
            {
                'pattern': r'\b(athens|greece).*istanbul|istanbul.*(athens|greece)\b',
                'issue': 'Geographic confusion: Athens is in Greece, not Istanbul',
                'suggestion': 'Did you mean Athens, Greece OR Istanbul, Turkey?'
            },
            {
                'pattern': r'\b(eiffel tower|paris).*istanbul|istanbul.*(eiffel tower|paris)\b',
                'issue': 'Geographic error: Eiffel Tower is in Paris, not Istanbul',
                'suggestion': 'Istanbul has Galata Tower and other landmarks'
            },
            {
                'pattern': r'\b(manhattan|new york|nyc)\b',
                'issue': 'Geographic error: Manhattan is in New York, not Istanbul',
                'suggestion': 'Istanbul districts include Beyoğlu, Kadıköy, Sultanahmet'
            },
            {
                'pattern': r'\b(colosseum|rome)\b',
                'issue': 'Geographic error: Colosseum is in Rome, not Istanbul',
                'suggestion': 'Istanbul has Hagia Sophia, Blue Mosque, and other historic sites'
            },
            {
                'pattern': r'\beiffel tower\b',
                'issue': 'Geographic error: Eiffel Tower is in Paris, not Istanbul',
                'suggestion': 'Istanbul has Galata Tower and other landmarks'
            }
        ]
        
        # Logical contradiction validation
        logical_issues = [
            {
                'pattern': r'\b(vegetarian|vegan).*(steakhouse|meat only)\b|\b(steakhouse|meat only).*(vegetarian|vegan)\b',
                'issue': 'Logical contradiction: Vegetarian restaurants don\'t serve meat',
                'suggestion': 'Would you like vegetarian restaurants OR steakhouses?'
            },
            {
                'pattern': r'\b(kosher|halal).*(pork|bacon|ham)\b|\b(pork|bacon|ham).*(kosher|halal)\b',
                'issue': 'Religious dietary contradiction: Kosher/Halal doesn\'t include pork',
                'suggestion': 'Would you like kosher/halal restaurants OR places serving pork?'
            },
            {
                'pattern': r'\bunderwater.*mountaintop\b|\bmountaintop.*underwater\b',
                'issue': 'Physical impossibility: Can\'t be underwater and on mountaintop',
                'suggestion': 'Would you like waterfront OR mountain view restaurants?'
            }
        ]
        
        # Temporal validation
        temporal_issues = [
            {
                'pattern': r'\byear (20[3-9]\d|2[1-9]\d\d|[3-9]\d{3})\b',
                'issue': 'Temporal error: Cannot provide information about future years',
                'suggestion': 'I can help with current or historical information'
            },
            {
                'pattern': r'\bottoman.*(195\d|19[6-9]\d|20\d\d)\b',
                'issue': 'Historical error: Ottoman Empire ended in 1922',
                'suggestion': 'Ottoman era ended in 1922, Turkey became republic in 1923'
            }
        ]
        
        # Budget reality validation
        budget_issues = [
            {
                'pattern': r'\b(free).*(luxury|expensive|high.?end)\b|\b(luxury|expensive).*(free)\b',
                'issue': 'Unrealistic budget: Luxury services aren\'t free',
                'suggestion': 'Budget restaurants start around 20-50 TRY per person'
            },
            {
                'pattern': r'\b(million|billion).*dollars?.*(meal|restaurant|food)\b',
                'issue': 'Excessive budget: Even luxury meals cost $200-500',
                'suggestion': 'Would you like luxury restaurant recommendations?'
            }
        ]
        
        # Fictional content validation
        fictional_issues = [
            {
                'pattern': r'\b(hogwarts|superman|batman|gotham|middle.?earth|wakanda)\b',
                'issue': 'Fictional content: These are not real places',
                'suggestion': 'I can help with real Istanbul locations and experiences'
            },
            {
                'pattern': r'\b(flying car|time travel|teleport|magic carpet)\b',
                'issue': 'Technology error: These technologies aren\'t available',
                'suggestion': 'I can help with current transportation options like metro, bus, taxi'
            }
        ]
        
        # Check all issue types
        all_issue_types = [
            ('geographic', geographic_issues),
            ('logical', logical_issues)
        ]
        
        for issue_type, issue_list in all_issue_types:
            for issue in issue_list:
                if re.search(issue['pattern'], query_lower):
                    validation_result['is_valid'] = False
                    validation_result['error_type'] = issue_type
                    validation_result['issues'].append(issue['issue'])
                    validation_result['suggestions'].append(issue['suggestion'])
                    break
        
        return validation_result
    
    def detect_ambiguous_queries(self, query: str) -> Dict[str, Any]:
        """Detect queries that are too ambiguous or could be misinterpreted"""
        query_lower = query.lower()
        
        ambiguity_result = {
            'is_ambiguous': False,
            'ambiguity_type': None,
            'clarification_needed': None,
            'suggested_questions': []
        }
        
        # Ultra-short queries that are too vague
        if len(query.strip()) < 3:
            ambiguity_result.update({
                'is_ambiguous': True,
                'ambiguity_type': 'too_short',
                'clarification_needed': 'Your query is too short to understand. Could you be more specific?',
                'suggested_questions': [
                    'What restaurants do you recommend in Sultanahmet?',
                    'How do I get from Taksim to Kadıköy?'
                ]
            })
            return ambiguity_result
        
        # Single word queries that could mean many things
        single_word_ambiguous = ['good', 'best', 'nice', 'great', 'amazing']
        if query.strip().lower() in single_word_ambiguous:
            ambiguity_result.update({
                'is_ambiguous': True,
                'ambiguity_type': 'single_word_vague',
                'clarification_needed': 'What specifically would you like to know about Istanbul?',
                'suggested_questions': [
                    'What are the best restaurants in Beyoğlu?',
                    'Which museums are most beautiful?'
                ]
            })
            return ambiguity_result
        
        return ambiguity_result

class EnhancedKnowledgeBase:
    """Expanded knowledge base for Istanbul"""
    
    def __init__(self):
        self.historical_info = {
            'hagia_sophia': {
                'description': 'Originally built as a Byzantine cathedral in 537 AD, later converted to a mosque, then a museum, and now a mosque again.',
                'visiting_tips': 'Best visited early morning or late afternoon to avoid crowds. Dress modestly.',
                'historical_significance': 'Symbol of both Christian and Islamic heritage in Istanbul.',
                'architecture': 'Famous for its massive dome and beautiful mosaics.'
            },
            'blue_mosque': {
                'description': 'Built between 1609-1616, famous for its six minarets and blue Iznik tiles.',
                'visiting_tips': 'Closed during prayer times. Remove shoes before entering.',
                'architectural_features': 'Six minarets, cascading domes, beautiful blue tilework.',
                'best_viewing_times': 'Early morning or sunset for photography.'
            },
            'topkapi_palace': {
                'description': 'Primary residence of Ottoman sultans for 400 years.',
                'visiting_tips': 'Allow 3-4 hours for full visit. Buy tickets online to skip lines.',
                'highlights': 'Imperial Treasury, Sacred Relics, Harem section, views of Bosphorus.',
                'historical_period': 'Ottoman Empire headquarters from 1465-1856.'
            }
        }
        
        self.cultural_etiquette = {
            'mosque_etiquette': [
                'Remove shoes before entering',
                'Dress modestly - cover shoulders and knees',
                'Women should cover their hair',
                'Be quiet and respectful',
                'Don\'t point feet toward Mecca',
                'No photography during prayer'
            ],
            'dining_etiquette': [
                'Wait for the host to begin eating',
                'Keep hands visible on the table',
                'Accept tea when offered - it\'s culturally important',
                'Tipping 10-15% is standard in restaurants',
                'Bread is sacred - don\'t waste it'
            ],
            'general_customs': [
                'Greet with handshake or slight bow',
                'Remove shoes when entering homes',
                'Dress conservatively in religious areas',
                'Learn basic Turkish phrases - locals appreciate it',
                'Bargaining is common in markets'
            ]
        }
        
        self.practical_info = {
            'currency': {
                'name': 'Turkish Lira (TRY)',
                'exchange_tips': 'Exchange at banks or official exchange offices for better rates',
                'cards_accepted': 'Credit cards widely accepted in tourist areas',
                'atms': 'ATMs available throughout the city'
            },
            'language': {
                'official': 'Turkish',
                'common_phrases': {
                    'hello': 'Merhaba',
                    'thank_you': 'Teşekkür ederim',
                    'excuse_me': 'Affedersiniz',
                    'how_much': 'Ne kadar?',
                    'where_is': 'Nerede?'
                },
                'english_level': 'Basic English in tourist areas, limited elsewhere'
            }
        }
    
    def get_historical_info(self, attraction: str) -> Optional[Dict[str, str]]:
        """Get detailed historical information about attractions"""
        return self.historical_info.get(attraction.lower().replace(' ', '_'))
    
    def get_cultural_advice(self, context: str) -> List[str]:
        """Get cultural etiquette advice for specific contexts"""
        return self.cultural_etiquette.get(context, [])
    
    def get_knowledge_response(self, query: str, intent_info: Dict[str, Any] = None) -> Optional[str]:
        """Get knowledge-based response for queries about Istanbul"""
        query_lower = query.lower()
        
        # Historical attractions
        if any(word in query_lower for word in ['hagia sophia', 'aya sofya']):
            info = self.get_historical_info('hagia_sophia')
            if info:
                return f"Hagia Sophia: {info['description']} {info['visiting_tips']}"
        
        elif any(word in query_lower for word in ['blue mosque', 'sultanahmet mosque']):
            info = self.get_historical_info('blue_mosque')
            if info:
                return f"Blue Mosque: {info['description']} {info['visiting_tips']}"
        
        elif any(word in query_lower for word in ['topkapi palace', 'topkapi']):
            info = self.get_historical_info('topkapi_palace')
            if info:
                return f"Topkapi Palace: {info['description']} {info['visiting_tips']}"
        
        # Cultural advice
        elif any(word in query_lower for word in ['mosque etiquette', 'mosque rules', 'visiting mosque']):
            advice = self.get_cultural_advice('mosque_etiquette')
            return f"Mosque etiquette tips: {', '.join(advice)}"
        
        elif any(word in query_lower for word in ['dining etiquette', 'table manners', 'eating customs']):
            advice = self.get_cultural_advice('dining_etiquette')
            return f"Turkish dining etiquette: {', '.join(advice)}"
        
        # Ottoman history
        elif any(word in query_lower for word in ['ottoman', 'ottoman empire', 'ottoman history']):
            return "The Ottoman Empire ruled from Istanbul (Constantinople) for over 600 years (1299-1922). Key landmarks include Topkapi Palace (sultan's residence), Dolmabahce Palace (later imperial palace), and many mosques like Suleymaniye and Blue Mosque. The empire's cultural legacy is visible throughout Istanbul's architecture, cuisine, and traditions."
        
        # Byzantine history  
        elif any(word in query_lower for word in ['byzantine', 'constantinople', 'byzantine empire']):
            return "Byzantine Constantinople was the Eastern Roman Empire's capital for over 1,000 years (330-1453 AD). Major Byzantine landmarks include Hagia Sophia (originally a cathedral), the Hippodrome, Basilica Cistern, and city walls. The empire's Christian heritage merged with later Islamic culture to create Istanbul's unique character."
        
        return None
        

class ContextAwareResponseGenerator:
    """Generate context-aware responses that reference previous conversations"""
    
    def __init__(self, context_manager: EnhancedContextManager, knowledge_base: EnhancedKnowledgeBase):
        self.context_manager = context_manager
        self.knowledge_base = knowledge_base
    
    def generate_follow_up_response(self, query: str, context: ConversationContext, 
                                  parsed_query: Dict[str, Any]) -> str:
        """Generate responses that reference previous conversation"""
        
        # Check if this is a follow-up to restaurant recommendations
        if context.last_recommendation_type == 'restaurant':
            if any(word in query.lower() for word in ['tip', 'tipping', 'how much']):
                return self.generate_tipping_advice(context)
            elif any(word in query.lower() for word in ['reservation', 'book', 'booking']):
                return self.generate_reservation_advice(context)
            elif any(word in query.lower() for word in ['dress code', 'what to wear']):
                return self.generate_dress_code_advice(context)
            elif any(word in query.lower() for word in ['more', 'other', 'different']):
                return f"Since you were interested in the restaurants I mentioned in {context.user_location or 'that area'}, here are some additional recommendations..."
        
        # Check if this is a follow-up to transportation info
        elif context.last_recommendation_type == 'transportation':
            if any(word in query.lower() for word in ['cost', 'price', 'how much']):
                return self.generate_transport_cost_info()
            elif any(word in query.lower() for word in ['card', 'istanbulkart', 'ticket']):
                return self.generate_istanbulkart_info()
        
        # ENHANCED: Check if this is a location-based follow-up
        # e.g., user said "beyoglu" and now asks "places"
        if context.mentioned_places and any(generic in query.lower() for generic in ['places', 'attractions', 'things to see', 'what to visit']):
            recent_location = context.mentioned_places[-1]
            return self.generate_location_specific_places_response(recent_location, query)
        
        # ENHANCED: Check if this is a restaurant follow-up with location context
        if context.mentioned_places and any(word in query.lower() for word in ['restaurants', 'food', 'where to eat', 'dining']):
            recent_location = context.mentioned_places[-1]
            return self.generate_location_specific_restaurant_response(recent_location, query)
        
        # Generic follow-up
        return f"Based on our previous conversation about {context.last_recommendation_type}, I can provide more specific information. What would you like to know?"
    
    def generate_location_specific_places_response(self, location: str, query: str) -> str:
        """Generate a location-specific response for places queries"""
        return f"Since you were asking about {location}, here are some great places to visit in that area. Let me get you specific recommendations for {location}..."
    
    def generate_location_specific_restaurant_response(self, location: str, query: str) -> str:
        """Generate a location-specific response for restaurant queries"""
        return f"Perfect! Since you mentioned {location}, I'll recommend some excellent restaurants in that neighborhood. {location} has some fantastic dining options..."
    
    def generate_response(self, query: str, ai_response: str, context: Optional[ConversationContext], 
                         intent_info: Dict[str, Any], places: List = None) -> str:
        """Generate enhanced context-aware response"""
        
        # If we have context and this is a follow-up, enhance the response
        if context and len(context.previous_queries) > 0:
            follow_up_response = self.generate_follow_up_response(query, context, intent_info)
            if follow_up_response and follow_up_response != ai_response:
                return follow_up_response
        
        # Add context references to the AI response
        enhanced_response = ai_response
        
        if context:
            # Reference previous places mentioned
            if context.mentioned_places and intent_info.get('intent') in ['restaurant', 'attraction', 'place']:
                enhanced_response += f"\n\nSince you were asking about {', '.join(context.mentioned_places[-2:])}, you might also be interested in similar areas nearby."
            
            # Add personalized recommendations based on preferences
            if context.user_preferences:
                if 'vegetarian' in context.user_preferences and intent_info.get('intent') == 'restaurant':
                    enhanced_response += "\n\n🌱 I noticed you prefer vegetarian options, so I've focused on restaurants with good vegetarian choices."
                elif 'budget' in context.user_preferences:
                    enhanced_response += "\n\n💰 I've considered your budget preferences in these recommendations."
        
        return enhanced_response
    
    def enhance_system_prompt(self, original_prompt: str, context: Optional[ConversationContext]) -> str:
        """Enhance system prompt with conversation context"""
        enhanced_prompt = original_prompt
        
        if context:
            enhanced_prompt += f"\n\nCONVERSATION CONTEXT:"
            if context.previous_queries:
                enhanced_prompt += f"\nPrevious questions: {', '.join(context.previous_queries[-3:])}"
            if context.mentioned_places:
                enhanced_prompt += f"\nPreviously mentioned places: {', '.join(context.mentioned_places)}"
            if context.user_preferences:
                enhanced_prompt += f"\nUser preferences: {context.user_preferences}"
            if context.last_recommendation_type:
                enhanced_prompt += f"\nLast recommendation type: {context.last_recommendation_type}"
        
        return enhanced_prompt
    
    def generate_tipping_advice(self, context: ConversationContext) -> str:
        """Generate tipping advice based on previous restaurant recommendations"""
        return """💡 **Tipping at Istanbul Restaurants:**

• **Standard tip**: 10-15% of the bill
• **Service charge**: Sometimes already included - check your receipt
• **Cash preferred**: While cards are accepted, cash tips are appreciated
• **Rounding up**: For casual places, rounding up the bill is common
• **Tea houses**: Small tip (1-2 TL) or rounding up is sufficient

For the restaurants I mentioned earlier, most will expect a standard 10-15% tip unless service charge is already included. Higher-end restaurants may expect closer to 15-20%."""
    
    def generate_reservation_advice(self, context: ConversationContext) -> str:
        """Generate reservation advice"""
        return """📞 **Making Reservations:**

• **Popular restaurants**: Highly recommended, especially for dinner
• **How to book**: Call directly or use apps like OpenTable, Yemeksepeti
• **Language**: Basic English is usually understood in tourist areas
• **Timing**: Turks dine late - 8-9 PM for dinner is normal
• **Group size**: Mention exact number when booking

For the restaurants I recommended, I'd suggest calling ahead, especially on weekends or if you have a larger group."""
    
    def generate_dress_code_advice(self, context: ConversationContext) -> str:
        """Generate dress code advice"""
        return """👔 **Restaurant Dress Codes in Istanbul:**

• **Casual restaurants**: Smart casual is fine
• **Upscale dining**: Business casual to formal
• **Rooftop restaurants**: Often require smart casual minimum
• **Traditional places**: Casual is acceptable
• **General rule**: Slightly more formal than you might expect

Istanbul residents generally dress well when dining out. The restaurants I mentioned earlier would be appropriate for smart casual attire."""

def enhance_main_ai_endpoint(original_function):
    """Decorator to enhance the main AI endpoint with improved capabilities"""
    
    context_manager = EnhancedContextManager()
    query_understander = EnhancedQueryUnderstanding()
    knowledge_base = EnhancedKnowledgeBase()
    response_generator = ContextAwareResponseGenerator(context_manager, knowledge_base)
    
    async def enhanced_ai_endpoint(request):
        # Get request data
        data = await request.json()
        user_input = data.get("query", data.get("user_input", ""))
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        # Get conversation context
        context = context_manager.get_context(session_id)
        
        # Correct and enhance query
        corrected_query = query_understander.correct_and_enhance_query(user_input)
        
        # Parse query with context awareness
        parsed_query = query_understander.extract_intent_and_entities(corrected_query, context)
        
        # Extract user preferences if context exists
        if context:
            context_manager.extract_preferences(corrected_query, context)
        
        # Handle follow-up questions with context
        if parsed_query['intent'] == 'follow_up_question' and context:
            response = response_generator.generate_follow_up_response(corrected_query, context, parsed_query)
            context_manager.update_context(session_id, user_input, response, topic='follow_up')
            return {"message": response, "session_id": session_id}
        
        # Call original function for main processing
        result = await original_function(request)
        
        # Update context with new interaction
        if isinstance(result, dict) and 'message' in result:
            # Extract mentioned places from response
            places = []
            # Simple extraction - in real implementation, use NLP
            istanbul_areas = ['sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas']
            for area in istanbul_areas:
                if area.lower() in result['message'].lower():
                    places.append(area.title())
            
            context_manager.update_context(
                session_id, 
                user_input, 
                result['message'],
                places=places,
                topic=parsed_query['intent'].replace('_', ' ')
            )
        
        return result
    
    return enhanced_ai_endpoint

# Additional helper functions for knowledge expansion

def get_museum_detailed_info(museum_name: str) -> Dict[str, Any]:
    """Get detailed museum information"""
    museum_db = {
        'topkapi': {
            'full_name': 'Topkapi Palace Museum',
            'opening_hours': '9:00 AM - 6:45 PM (Summer), 9:00 AM - 4:45 PM (Winter)',
            'closed': 'Tuesdays',
            'ticket_price': '100 TL (Palace), 70 TL (Harem)',
            'duration': '3-4 hours',
            'highlights': ['Imperial Treasury', 'Sacred Relics', 'Harem', 'Palace kitchens'],
            'tips': 'Buy tickets online, visit early morning, bring water'
        },
        'hagia_sophia': {
            'full_name': 'Hagia Sophia',
            'opening_hours': '24/7 (as a mosque)',
            'closed': 'During prayer times for tourists',
            'ticket_price': 'Free',
            'duration': '1-2 hours',
            'highlights': ['Byzantine mosaics', 'Islamic calligraphy', 'Massive dome'],
            'tips': 'Dress modestly, best photography in afternoon light'
        }
    }
    
    return museum_db.get(museum_name.lower(), {})

def get_neighborhood_detailed_info(neighborhood: str) -> Dict[str, Any]:
    """Get detailed neighborhood information"""
    neighborhood_db = {
        'sultanahmet': {
            'character': 'Historic old city, UNESCO World Heritage site',
            'attractions': ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace', 'Basilica Cistern'],
            'dining': ['Traditional Ottoman cuisine', 'Tourist restaurants', 'Street food'],
            'atmosphere': 'Tourist-heavy, historic, conservative',
            'best_for': 'First-time visitors, history buffs',
            'transportation': 'T1 Tram, walking distance to major sites'
        },
        'beyoglu': {
            'character': 'Vibrant cultural district, European feel',
            'attractions': ['Galata Tower', 'Istiklal Street', 'Pera Museum'],
            'dining': ['International cuisine', 'Trendy restaurants', 'Rooftop bars'],
            'atmosphere': 'Bohemian, artistic, nightlife',
            'best_for': 'Culture lovers, nightlife enthusiasts',
            'transportation': 'M2 Metro, Funicular from Karakoy'
        }
    }
    
    return neighborhood_db.get(neighborhood.lower(), {})

def generate_day_itinerary(duration: int, interests: List[str], budget: str) -> str:
    """Generate a day-by-day itinerary based on user preferences"""
    # This would be a comprehensive function to create personalized itineraries
    # For now, returning a sample structure
    
    base_itinerary = {
        1: {
            'morning': 'Sultanahmet Historic Area',
            'afternoon': 'Topkapi Palace or Hagia Sophia',
            'evening': 'Dinner in Sultanahmet'
        },
        2: {
            'morning': 'Galata Tower and Beyoglu',
            'afternoon': 'Shopping on Istiklal Street',
            'evening': 'Bosphorus cruise or ferry'
        },
        3: {
            'morning': 'Asian side - Kadikoy',
            'afternoon': 'Princes Islands or Dolmabahce Palace',
            'evening': 'Traditional Turkish bath (Hamam)'
        }
    }
    
    itinerary_text = f"🗓️ **Your {duration}-Day Istanbul Itinerary:**\n\n"
    
    for day in range(1, min(duration + 1, 4)):
        if day in base_itinerary:
            day_plan = base_itinerary[day]
            itinerary_text += f"**Day {day}:**\n"
            itinerary_text += f"• Morning: {day_plan['morning']}\n"
            itinerary_text += f"• Afternoon: {day_plan['afternoon']}\n"
            itinerary_text += f"• Evening: {day_plan['evening']}\n\n"
    
    return itinerary_text

class ValidationErrorHandler:
    """Handle various types of validation errors with helpful responses"""
    
    @staticmethod
    def generate_error_response(validation_result: Dict[str, Any]) -> str:
        """Generate helpful error responses for different validation failures"""
        error_type = validation_result.get('error_type')
        issues = validation_result.get('issues', [])
        suggestions = validation_result.get('suggestions', [])
        
        if error_type == 'geographic':
            return f"""🌍 **Geographic Clarification Needed**

I noticed you mentioned locations outside of Istanbul. {issues[0] if issues else ''}

{suggestions[0] if suggestions else ''}

I specialize in Istanbul travel information. Could you please clarify what you'd like to know about Istanbul specifically?"""
        
        elif error_type == 'logical':
            return f"""🤔 **Clarification Needed**

{issues[0] if issues else 'There seems to be a contradiction in your request.'}

{suggestions[0] if suggestions else ''}

Could you help me understand what you're looking for?"""
        
        elif error_type == 'temporal':
            return f"""⏰ **Time Period Issue**

{issues[0] if issues else 'There seems to be a time-related issue with your query.'}

{suggestions[0] if suggestions else ''}

I can help with current Istanbul information or historical facts about the city."""
        
        elif error_type == 'budget':
            return f"""💰 **Budget Clarification**

{issues[0] if issues else 'There seems to be an issue with the budget range mentioned.'}

{suggestions[0] if suggestions else ''}

Could you share a realistic budget range for your Istanbul experience?"""
        
        elif error_type == 'fictional':
            return f"""🎭 **Real World Focus**

{issues[0] if issues else 'I focus on real-world Istanbul experiences.'}

{suggestions[0] if suggestions else ''}

I'd be happy to recommend actual Istanbul attractions, restaurants, and activities!"""
        
        elif error_type == 'inappropriate':
            return f"""🛡️ **Content Guidelines**

I focus on family-friendly travel recommendations for Istanbul.

{suggestions[0] if suggestions else 'I can help with cultural attractions, restaurants, museums, and wholesome entertainment options.'}

What aspects of Istanbul would you like to explore?"""
        
        elif error_type == 'impossibility':
            return f"""🌟 **Realistic Recommendations**

{issues[0] if issues else 'I focus on experiences that are actually available in Istanbul.'}

{suggestions[0] if suggestions else ''}

Let me help you find amazing real-world experiences in Istanbul!"""
        
        else:
            return """❓ **Let me help clarify**

I want to make sure I understand your request correctly. Could you provide a bit more detail about what you're looking for in Istanbul?

I can help with:
• Restaurant recommendations
• Tourist attractions and museums  
• Transportation information
• Cultural experiences
• Neighborhood guides"""

    @staticmethod
    def generate_clarification_response(ambiguity_info: Dict[str, Any]) -> str:
        """Generate helpful clarification responses for ambiguous queries"""
        ambiguity_type = ambiguity_info.get('ambiguity_type')
        clarification = ambiguity_info.get('clarification_needed', '')
        suggestions = ambiguity_info.get('suggested_questions', [])
        
        response = f"💭 **{clarification}**\n\n"
        
        if suggestions:
            response += "Here are some example questions I can help with:\n\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"{i}. {suggestion}\n"
            response += "\n"
        
        response += "**I can help you with:**\n"
        response += "🍽️ Restaurant recommendations by area or cuisine\n"
        response += "🏛️ Museums, attractions, and historical sites\n"
        response += "🚇 Transportation between different areas\n"
        response += "🗺️ Neighborhood guides and what to see\n"
        response += "🎭 Cultural experiences and local tips\n\n"
        response += "Please let me know what specifically interests you!"
        
        return response
    
    def validate_query_logic(self, query: str) -> Dict[str, Any]:
        """Validate query for logical inconsistencies and geographical errors"""
        query_lower = query.lower()
        
        validation_result = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'error_type': None
        }
        
        # Geographic validation - prevent confusion with other cities
        geographic_issues = [
            {
                'pattern': r'\b(athens|greece).*istanbul|istanbul.*(athens|greece)\b',
                'issue': 'Geographic confusion: Athens is in Greece, not Istanbul',
                'suggestion': 'Did you mean Athens, Greece OR Istanbul, Turkey?'
            },
            {
                'pattern': r'\b(eiffel tower|paris).*istanbul|istanbul.*(eiffel tower|paris)\b',
                'issue': 'Geographic error: Eiffel Tower is in Paris, not Istanbul',
                'suggestion': 'Istanbul has Galata Tower and other landmarks'
            },
            {
                'pattern': r'\b(manhattan|new york|nyc).*istanbul|istanbul.*(manhattan|new york|nyc)\b',
                'issue': 'Geographic error: Manhattan is in New York, not Istanbul',
                'suggestion': 'Istanbul districts include Beyoğlu, Kadıköy, Sultanahmet'
            },
            {
                'pattern': r'\b(colosseum|rome).*istanbul|istanbul.*(colosseum|rome)\b',
                'issue': 'Geographic error: Colosseum is in Rome, not Istanbul',
                'suggestion': 'Istanbul has Hagia Sophia, Blue Mosque, and other historic sites'
            }
        ]
        
        # Logical contradiction validation
        logical_issues = [
            {
                'pattern': r'\b(vegetarian|vegan).*(steakhouse|meat only|beef only)\b|\b(steakhouse|meat only).*(vegetarian|vegan)\b',
                'issue': 'Logical contradiction: Vegetarian restaurants don\'t serve meat',
                'suggestion': 'Would you like vegetarian restaurants OR steakhouses?'
            },
            {
                'pattern': r'\bvegetarian.*(seafood only|fish only)\b|\b(seafood only|fish only).*vegetarian\b',
                'issue': 'Logical contradiction: Vegetarian places don\'t serve seafood',
                'suggestion': 'Would you like vegetarian restaurants OR seafood restaurants?'
            }
        ]
        
        # Check all issue types
        all_issue_types = [
            ('geographic', geographic_issues),
            ('logical', logical_issues)
        ]
        
        for issue_type, issue_list in all_issue_types:
            for issue in issue_list:
                if re.search(issue['pattern'], query_lower):
                    validation_result['is_valid'] = False
                    validation_result['error_type'] = issue_type
                    validation_result['issues'].append(issue['issue'])
                    validation_result['suggestions'].append(issue['suggestion'])
                    break  # Only report first issue of each type
        
        return validation_result
    
    def detect_ambiguous_queries(self, query: str) -> Dict[str, Any]:
        """Detect queries that are too ambiguous or could be misinterpreted"""
        query_lower = query.lower()
        
        ambiguity_result = {
            'is_ambiguous': False,
            'ambiguity_type': None,
            'clarification_needed': None,
            'suggested_questions': []
        }
        
        # Ultra-short queries that are too vague
        if len(query.strip()) < 3:
            ambiguity_result.update({
                'is_ambiguous': True,
                'ambiguity_type': 'too_short',
                'clarification_needed': 'Your query is too short to understand. Could you be more specific?',
                'suggested_questions': [
                    'What restaurants do you recommend in Sultanahmet?',
                    'How do I get from Taksim to Kadıköy?',
                    'What are the best museums in Istanbul?'
                ]
            })
            return ambiguity_result
        
        # Single word queries that could mean many things
        single_word_ambiguous = ['good', 'best', 'nice', 'great', 'amazing', 'beautiful', 'awesome', 'cool']
        if query.strip().lower() in single_word_ambiguous:
            ambiguity_result.update({
                'is_ambiguous': True,
                'ambiguity_type': 'single_word_vague',
                'clarification_needed': 'What specifically would you like to know about Istanbul?',
                'suggested_questions': [
                    'What are the best restaurants in Beyoğlu?',
                    'Which museums are most beautiful?',
                    'What are nice places to visit in Istanbul?'
                ]
            })
            return ambiguity_result
        
        return ambiguity_result
