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
        
        if places:
            context.mentioned_places.extend(places)
            # Keep only unique places
            context.mentioned_places = list(set(context.mentioned_places))
        
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
        """Enhanced intent and entity extraction"""
        query_lower = query.lower()
        
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
        
        # Calculate intent scores
        restaurant_keywords = ['restaurant', 'eat', 'food', 'dining', 'meal', 'breakfast', 'lunch', 'dinner']
        museum_keywords = ['museum', 'gallery', 'exhibition', 'art', 'history', 'cultural']
        transport_keywords = ['transport', 'metro', 'bus', 'taxi', 'ferry', 'how to get', 'travel to']
        place_keywords = ['place', 'attraction', 'visit', 'see', 'landmark', 'tourist']
        
        for keyword in restaurant_keywords:
            if keyword in query_lower:
                intents['restaurant_search'] += 1
        
        for keyword in museum_keywords:
            if keyword in query_lower:
                intents['museum_inquiry'] += 1
        
        for keyword in transport_keywords:
            if keyword in query_lower:
                intents['transportation_info'] += 1
        
        for keyword in place_keywords:
            if keyword in query_lower:
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
        
        # Context-aware intent adjustment
        if context:
            # If user previously asked about restaurants and now asks follow-up
            if context.last_recommendation_type == 'restaurant' and intents['follow_up_question'] > 0:
                intents['restaurant_search'] += 2
            # Similar logic for other types
        
        # Find highest scoring intent
        primary_intent = max(intents.items(), key=lambda x: x[1])
        
        # Extract entities
        entities = self.extract_entities(query)
        
        return {
            'intent': primary_intent[0] if primary_intent[1] > 0 else 'general_travel_info',
            'confidence': primary_intent[1],
            'entities': entities,
            'all_scores': intents
        }
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities like locations, preferences, etc."""
        entities = {
            'locations': [],
            'cuisine_types': [],
            'price_range': [],
            'dietary_restrictions': [],
            'times': [],
            'features': []
        }
        
        # Location patterns
        istanbul_locations = [
            'sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
            'taksim', 'karakoy', 'ortakoy', 'bebek', 'fatih', 'sisli', 'eminonu'
        ]
        
        for location in istanbul_locations:
            if location in query.lower():
                entities['locations'].append(location.title())
        
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
        
        # Generic follow-up
        return f"Based on our previous conversation about {context.last_recommendation_type}, I can provide more specific information. What would you like to know?"
    
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

def validate_query_logic(self, query: str) -> Dict[str, Any]:
        """Validate query for logical inconsistencies and geographical errors"""
        query_lower = query.lower()
        
        validation_result = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'error_type': None
        }
        
        # Geographical validation
        geographic_issues = [
            {
                'pattern': r'athens.*istanbul|istanbul.*athens',
                'issue': 'Geographic confusion: Athens is in Greece, not Istanbul',
                'suggestion': 'Did you mean Athens, Greece OR Istanbul, Turkey?'
            },
            {
                'pattern': r'eiffel tower.*istanbul|istanbul.*eiffel tower',
                'issue': 'Geographic error: Eiffel Tower is in Paris, not Istanbul',
                'suggestion': 'Istanbul has Galata Tower and other landmarks'
            },
            {
                'pattern': r'manhattan.*istanbul|istanbul.*manhattan',
                'issue': 'Geographic error: Manhattan is in New York, not Istanbul',
                'suggestion': 'Istanbul districts include Beyoğlu, Kadıköy, Sultanahmet'
            },
            {
                'pattern': r'colosseum.*istanbul|istanbul.*colosseum',
                'issue': 'Geographic error: Colosseum is in Rome, not Istanbul',
                'suggestion': 'Istanbul has Hagia Sophia, Blue Mosque, and other historic sites'
            }
        ]
        
        # Logical contradiction validation
        logical_issues = [
            {
                'pattern': r'vegetarian.*steakhouse|steakhouse.*vegetarian',
                'issue': 'Logical contradiction: Vegetarian restaurants don\'t serve steak',
                'suggestion': 'Would you like vegetarian restaurants OR steakhouses?'
            },
            {
                'pattern': r'vegetarian.*seafood.*only|seafood.*only.*vegetarian',
                'issue': 'Logical contradiction: Vegetarian places don\'t serve seafood',
                'suggestion': 'Would you like vegetarian restaurants OR seafood restaurants?'
            },
            {
                'pattern': r'underwater.*mountaintop|mountaintop.*underwater',
                'issue': 'Physical impossibility: Can\'t be underwater and on mountaintop',
                'suggestion': 'Would you like waterfront OR mountain view restaurants?'
            }
        ]
        
        # Temporal validation
        temporal_issues = [
            {
                'pattern': r'year 3024|3024|future.*year.*\d{4}',
                'issue': 'Temporal error: Cannot provide information about future years',
                'suggestion': 'I can help with current or historical information'
            },
            {
                'pattern': r'ottoman.*195\d|ottoman.*20\d\d',
                'issue': 'Historical error: Ottoman Empire ended in 1922',
                'suggestion': 'Ottoman era ended in 1922, Turkey became republic in 1923'
            }
        ]
        
        # Budget reality validation
        budget_issues = [
            {
                'pattern': r'1 cent|free.*luxury|luxury.*free|0 dollars?',
                'issue': 'Unrealistic budget: Luxury services aren\'t free',
                'suggestion': 'Budget restaurants start around 20-50 TRY per person'
            },
            {
                'pattern': r'million.*dollars?.*meal|meal.*million.*dollars?',
                'issue': 'Excessive budget: Even luxury meals cost $200-500',
                'suggestion': 'Would you like luxury restaurant recommendations?'
            }
        ]
        
        # Fictional content validation
        fictional_issues = [
            {
                'pattern': r'hogwarts|superman|batman|fictional.*character',
                'issue': 'Fictional content: These are not real places or people',
                'suggestion': 'I can help with real Istanbul locations and experiences'
            },
            {
                'pattern': r'flying car|time travel|teleport',
                'issue': 'Technology error: These technologies aren\'t available',
                'suggestion': 'I can help with current transportation options'
            }
        ]
        
        all_issue_types = [
            ('geographic', geographic_issues),
            ('logical', logical_issues),
            ('temporal', temporal_issues),
            ('budget', budget_issues),
            ('fictional', fictional_issues)
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
