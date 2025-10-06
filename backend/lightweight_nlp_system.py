#!/usr/bin/env python3
"""
Lightweight NLP System for Istanbul AI - No LLMs Required
========================================================

This system handles natural language queries using:
1. Intent Classification (spaCy + scikit-learn)
2. Entity Extraction (regex + keyword matching)
3. Synonym + Fuzzy Matching
4. Rule-Based Context Handling
5. Human-Like Response Templates
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
from rapidfuzz import fuzz, process
from collections import defaultdict

class QueryIntent(Enum):
    RESTAURANT_SEARCH = "restaurant_search"
    MUSEUM_INFO = "museum_info"
    CULTURAL_ETIQUETTE = "cultural_etiquette"
    DIRECTIONS = "directions"
    EVENTS = "events"
    DISTRICT_INFO = "district_info"
    TRANSPORTATION = "transportation"
    GENERAL_INFO = "general_info"
    SHOPPING = "shopping"
    NIGHTLIFE = "nightlife"

@dataclass
class ExtractedEntities:
    """Extracted entities from user query"""
    district: Optional[str] = None
    cuisine: Optional[str] = None
    budget: Optional[str] = None
    atmosphere: Optional[str] = None
    time: Optional[str] = None
    group_size: Optional[int] = None
    museum_name: Optional[str] = None
    transportation_type: Optional[str] = None
    activity_type: Optional[str] = None
    specific_location: Optional[str] = None

@dataclass
class QueryContext:
    """Context from previous queries in the session"""
    last_intent: Optional[QueryIntent] = None
    last_entities: Optional[ExtractedEntities] = None
    last_query: Optional[str] = None
    session_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.session_data is None:
            self.session_data = {}

class LightweightNLPSystem:
    """Main NLP system for processing Istanbul queries without LLMs"""
    
    def __init__(self):
        self.session_contexts = {}  # Store context per session
        self._init_keywords()
        self._init_synonyms()
        self._init_response_templates()
    
    def _init_keywords(self):
        """Initialize keyword dictionaries for entity extraction"""
        
        # Districts and neighborhoods
        self.districts = {
            'sultanahmet', 'beyoglu', 'beyoğlu', 'galata', 'karakoy', 'karaköy',
            'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'taksim', 'sisli', 'şişli',
            'fatih', 'eminonu', 'eminönü', 'ortakoy', 'ortaköy', 'bebek', 'arnavutkoy',
            'arnavutköy', 'sariyer', 'sarıyer', 'uskudar', 'üsküdar', 'moda', 'cihangir',
            'fener', 'balat', 'eyup', 'eyüp', 'nisantasi', 'nişantaşı', 'levent',
            'maslak', 'etiler', 'bagdat', 'bağdat', 'istiklal', 'galatasaray'
        }
        
        # Cuisines
        self.cuisines = {
            'turkish', 'ottoman', 'kebab', 'kebap', 'pide', 'meze', 'balik', 'balık',
            'seafood', 'lokanta', 'italian', 'french', 'chinese', 'japanese', 'sushi',
            'pizza', 'burger', 'international', 'fusion', 'vegetarian', 'vegan',
            'breakfast', 'kahvalti', 'kahvaltı', 'coffee', 'kahve', 'dessert', 'tatli',
            'tatlı', 'döner', 'doner', 'lahmacun', 'mantı', 'manti', 'kofte', 'köfte'
        }
        
        # Museums and attractions
        self.museums = {
            'hagia sophia', 'ayasofya', 'topkapi', 'topkapı', 'blue mosque', 'sultanahmet',
            'basilica cistern', 'yerebatan', 'galata tower', 'dolmabahce', 'dolmabahçe',
            'pera museum', 'istanbul modern', 'archaeology', 'arkeoloji', 'carpet museum',
            'military museum', 'askeri müze', 'naval museum', 'chora', 'kariye'
        }
        
        # Transportation
        self.transportation = {
            'metro', 'metrobus', 'metrobüs', 'bus', 'otobüs', 'tram', 'tramvay',
            'ferry', 'vapur', 'dolmuş', 'dolmus', 'taxi', 'taksi', 'uber', 'bitaksi',
            'walking', 'yürüyüş', 'bicycle', 'bisiklet', 'car', 'araba'
        }
        
        # Budget indicators
        self.budget_keywords = {
            'cheap': 'budget',
            'budget': 'budget',
            'affordable': 'budget',
            'expensive': 'upscale',
            'luxury': 'upscale',
            'mid-range': 'moderate',
            'moderate': 'moderate',
            'reasonable': 'moderate'
        }
        
        # Atmosphere keywords
        self.atmosphere_keywords = {
            'romantic', 'cozy', 'family', 'business', 'casual', 'formal', 'rooftop',
            'outdoor', 'view', 'bosphorus', 'boğaz', 'terrace', 'garden', 'quiet',
            'lively', 'traditional', 'modern', 'historic', 'trendy', 'local'
        }
        
        # Time indicators
        self.time_keywords = {
            'breakfast', 'morning', 'lunch', 'afternoon', 'dinner', 'evening',
            'night', 'late night', 'gece', 'sabah', 'öğle', 'akşam'
        }
    
    def _init_synonyms(self):
        """Initialize synonym dictionaries for fuzzy matching"""
        self.synonyms = {
            # Food synonyms
            'kebab': ['kebap', 'şiş', 'adana', 'urfa', 'döner', 'doner'],
            'fish': ['balık', 'balik', 'seafood', 'deniz ürünleri'],
            'meat': ['et', 'beef', 'lamb', 'kuzu'],
            'chicken': ['tavuk', 'piliç', 'pilić'],
            'vegetarian': ['vejetaryen', 'sebze', 'vegetable'],
            'dessert': ['tatlı', 'tatli', 'sweet', 'baklava', 'künefe'],
            'coffee': ['kahve', 'türk kahvesi', 'espresso', 'cappuccino'],
            'tea': ['çay', 'apple tea', 'elma çayı'],
            
            # Atmosphere synonyms
            'romantic': ['romantik', 'couples', 'date', 'intimate'],
            'family': ['aile', 'kids', 'children', 'çocuk'],
            'view': ['manzara', 'scenery', 'panorama', 'vista'],
            'rooftop': ['terrace', 'teras', 'çatı', 'outdoor'],
            'traditional': ['geleneksel', 'authentic', 'otantik', 'classic'],
            'modern': ['contemporary', 'trendy', 'new', 'yeni'],
            
            # District synonyms
            'old city': ['sultanahmet', 'historic peninsula', 'tarihi yarımada'],
            'new city': ['beyoğlu', 'taksim', 'modern istanbul'],
            'asian side': ['anadolu yakası', 'kadıköy', 'üsküdar'],
            'european side': ['avrupa yakası', 'beyoğlu', 'beşiktaş'],
            
            # Budget synonyms
            'cheap': ['budget', 'affordable', 'ucuz', 'ekonomik'],
            'expensive': ['luxury', 'upscale', 'pahalı', 'lüks'],
            'moderate': ['mid-range', 'reasonable', 'orta', 'makul']
        }
    
    def _init_response_templates(self):
        """Initialize human-like response templates"""
        self.templates = {
            QueryIntent.RESTAURANT_SEARCH: {
                'single': "I'd recommend **{name}**, a great {cuisine} place in {district}. {description}",
                'multiple': "Here are some excellent {cuisine} options in {district}:\n\n{restaurants}",
                'with_budget': "{name} in {district} is perfect for your {budget} budget. {description}",
                'with_atmosphere': "For a {atmosphere} experience, try {name} in {district}. {description}"
            },
            
            QueryIntent.MUSEUM_INFO: {
                'general': "**{name}** is one of Istanbul's must-visit museums. {description}",
                'practical': "**{name}** is located in {district}. {hours} {entrance_info}",
                'historical': "**{name}** dates back to {period}. {significance} {practical_info}"
            },
            
            QueryIntent.DIRECTIONS: {
                'metro': "Take the {line} metro line to {station}. {walking_directions}",
                'ferry': "You can take a ferry from {from_port} to {to_port}. {additional_info}",
                'walking': "It's about a {duration} walk from {from_location}. {route_description}"
            },
            
            QueryIntent.DISTRICT_INFO: {
                'overview': "**{district}** is known for {characteristics}. {highlights}",
                'activities': "In {district}, you can {activities}. {recommendations}"
            },
            
            QueryIntent.CULTURAL_ETIQUETTE: {
                'mosque': "When visiting mosques: {guidelines}",
                'dining': "Turkish dining etiquette: {guidelines}",
                'general': "Cultural tips for Istanbul: {guidelines}"
            }
        }
    
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the user's intent using keyword matching and patterns"""
        query_lower = query.lower()
        
        # Restaurant indicators
        restaurant_keywords = ['restaurant', 'eat', 'food', 'dining', 'meal', 'breakfast',
                             'lunch', 'dinner', 'cuisine', 'yemek', 'restoran', 'lokanta']
        food_words = any(food in query_lower for food in self.cuisines)
        restaurant_words = any(word in query_lower for word in restaurant_keywords)
        
        if restaurant_words or food_words:
            return QueryIntent.RESTAURANT_SEARCH
        
        # Museum indicators
        museum_keywords = ['museum', 'gallery', 'exhibition', 'art', 'history', 'müze', 'galeri']
        museum_names = any(museum in query_lower for museum in self.museums)
        museum_words = any(word in query_lower for word in museum_keywords)
        
        if museum_words or museum_names:
            return QueryIntent.MUSEUM_INFO
        
        # Direction indicators
        direction_keywords = ['how to get', 'directions', 'way to', 'route', 'travel to',
                            'go to', 'reach', 'nasıl gidilir', 'yol', 'ulaşım']
        if any(word in query_lower for word in direction_keywords):
            return QueryIntent.DIRECTIONS
        
        # Transportation indicators
        transport_words = any(transport in query_lower for transport in self.transportation)
        if transport_words:
            return QueryIntent.TRANSPORTATION
        
        # District info indicators
        district_words = any(district in query_lower for district in self.districts)
        info_keywords = ['about', 'what is', 'tell me', 'describe', 'hakkında', 'nedir']
        if district_words and any(word in query_lower for word in info_keywords):
            return QueryIntent.DISTRICT_INFO
        
        # Cultural etiquette indicators
        culture_keywords = ['etiquette', 'custom', 'culture', 'tradition', 'behavior',
                          'polite', 'respect', 'görgü', 'kültür', 'gelenek']
        if any(word in query_lower for word in culture_keywords):
            return QueryIntent.CULTURAL_ETIQUETTE
        
        # Events indicators
        event_keywords = ['event', 'festival', 'concert', 'show', 'exhibition',
                         'activity', 'etkinlik', 'festival', 'konser']
        if any(word in query_lower for word in event_keywords):
            return QueryIntent.EVENTS
        
        # Shopping indicators
        shopping_keywords = ['shop', 'shopping', 'buy', 'market', 'bazaar', 'mall',
                           'alışveriş', 'mağaza', 'pazar', 'çarşı']
        if any(word in query_lower for word in shopping_keywords):
            return QueryIntent.SHOPPING
        
        # Nightlife indicators
        nightlife_keywords = ['nightlife', 'bar', 'club', 'party', 'night', 'drink',
                            'gece hayatı', 'bar', 'kulüp', 'parti']
        if any(word in query_lower for word in nightlife_keywords):
            return QueryIntent.NIGHTLIFE
        
        return QueryIntent.GENERAL_INFO
    
    def extract_entities(self, query: str) -> ExtractedEntities:
        """Extract entities from the query using regex and keyword matching"""
        query_lower = query.lower()
        entities = ExtractedEntities()
        
        # Extract district
        for district in self.districts:
            if district in query_lower:
                entities.district = district.title()
                break
        
        # Extract cuisine
        for cuisine in self.cuisines:
            if cuisine in query_lower:
                entities.cuisine = cuisine
                break
        
        # Extract budget using regex patterns
        budget_patterns = [
            r'under (\d+)',
            r'less than (\d+)',
            r'below (\d+)',
            r'maximum (\d+)',
            r'budget (\d+)',
            r'(\d+) tl',
            r'(\d+) lira'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount = int(match.group(1))
                if amount < 100:
                    entities.budget = 'budget'
                elif amount < 300:
                    entities.budget = 'moderate'
                else:
                    entities.budget = 'upscale'
                break
        
        # Extract budget keywords
        for keyword, budget_type in self.budget_keywords.items():
            if keyword in query_lower:
                entities.budget = budget_type
                break
        
        # Extract atmosphere
        for atmosphere in self.atmosphere_keywords:
            if atmosphere in query_lower:
                entities.atmosphere = atmosphere
                break
        
        # Extract time
        for time_word in self.time_keywords:
            if time_word in query_lower:
                entities.time = time_word
                break
        
        # Extract group size
        group_patterns = [
            r'(\d+) people',
            r'group of (\d+)',
            r'(\d+) person',
            r'(\d+) kişi'
        ]
        
        for pattern in group_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities.group_size = int(match.group(1))
                break
        
        # Extract museum names using fuzzy matching
        for museum in self.museums:
            if museum in query_lower:
                entities.museum_name = museum
                break
        
        # Extract transportation type
        for transport in self.transportation:
            if transport in query_lower:
                entities.transportation_type = transport
                break
        
        return entities
    
    def apply_fuzzy_matching(self, query: str, entities: ExtractedEntities) -> ExtractedEntities:
        """Apply fuzzy matching for misspellings and synonyms"""
        
        # If no district found, try fuzzy matching
        if not entities.district:
            query_words = query.lower().split()
            for word in query_words:
                matches = process.extract(word, self.districts, limit=1, score_cutoff=80)
                if matches:
                    entities.district = matches[0][0].title()
                    break
        
        # If no cuisine found, try fuzzy matching
        if not entities.cuisine:
            query_words = query.lower().split()
            for word in query_words:
                matches = process.extract(word, self.cuisines, limit=1, score_cutoff=75)
                if matches:
                    entities.cuisine = matches[0][0]
                    break
        
        # Apply synonym matching
        for word in query.lower().split():
            for key, synonyms in self.synonyms.items():
                if word in synonyms and not getattr(entities, self._get_entity_field(key), None):
                    setattr(entities, self._get_entity_field(key), key)
        
        return entities
    
    def _get_entity_field(self, synonym_key: str) -> str:
        """Map synonym keys to entity fields"""
        mapping = {
            'kebab': 'cuisine',
            'fish': 'cuisine',
            'romantic': 'atmosphere',
            'family': 'atmosphere',
            'view': 'atmosphere',
            'cheap': 'budget',
            'expensive': 'budget'
        }
        return mapping.get(synonym_key, 'activity_type')
    
    def handle_context(self, query: str, session_id: str) -> QueryContext:
        """Handle context from previous queries"""
        
        # Get or create session context
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = QueryContext()
        
        context = self.session_contexts[session_id]
        
        # Check for follow-up queries
        followup_patterns = [
            r'what about',
            r'how about',
            r'and at night',
            r'for dinner',
            r'nearby',
            r'close to',
            r'around there'
        ]
        
        is_followup = any(re.search(pattern, query.lower()) for pattern in followup_patterns)
        
        if is_followup and context.last_entities:
            # This is a follow-up query, inherit context
            return context
        else:
            # Fresh query, update context
            context.last_query = query
            return context
    
    def generate_response(self, intent: QueryIntent, entities: ExtractedEntities, 
                         data: Dict[str, Any], context: QueryContext = None) -> str:
        """Generate human-like response using templates"""
        
        if intent == QueryIntent.RESTAURANT_SEARCH:
            return self._generate_restaurant_response(entities, data)
        elif intent == QueryIntent.MUSEUM_INFO:
            return self._generate_museum_response(entities, data)
        elif intent == QueryIntent.DIRECTIONS:
            return self._generate_directions_response(entities, data)
        elif intent == QueryIntent.DISTRICT_INFO:
            return self._generate_district_response(entities, data)
        elif intent == QueryIntent.CULTURAL_ETIQUETTE:
            return self._generate_culture_response(entities, data)
        else:
            return self._generate_general_response(entities, data)
    
    def _generate_restaurant_response(self, entities: ExtractedEntities, data: Dict[str, Any]) -> str:
        """Generate restaurant recommendation response"""
        if not data.get('restaurants'):
            return "I couldn't find any restaurants matching your criteria. Would you like to try a different area or cuisine?"
        
        restaurants = data['restaurants']
        
        if len(restaurants) == 1:
            restaurant = restaurants[0]
            template_key = 'single'
            
            if entities.budget:
                template_key = 'with_budget'
            elif entities.atmosphere:
                template_key = 'with_atmosphere'
            
            template = self.templates[QueryIntent.RESTAURANT_SEARCH][template_key]
            
            return template.format(
                name=restaurant.get('name', 'Unknown'),
                cuisine=entities.cuisine or 'dining',
                district=entities.district or restaurant.get('district', 'Istanbul'),
                description=restaurant.get('description', 'Highly rated local favorite.'),
                budget=entities.budget or 'any',
                atmosphere=entities.atmosphere or 'pleasant'
            )
        else:
            # Multiple restaurants
            restaurant_list = []
            for i, restaurant in enumerate(restaurants[:3], 1):
                restaurant_list.append(
                    f"{i}. **{restaurant.get('name', 'Unknown')}** - {restaurant.get('description', 'Great local spot.')}"
                )
            
            template = self.templates[QueryIntent.RESTAURANT_SEARCH]['multiple']
            return template.format(
                cuisine=entities.cuisine or 'dining',
                district=entities.district or 'Istanbul',
                restaurants='\n'.join(restaurant_list)
            )
    
    def _generate_museum_response(self, entities: ExtractedEntities, data: Dict[str, Any]) -> str:
        """Generate museum information response"""
        if not data.get('museums'):
            return "I couldn't find information about that museum. Would you like suggestions for popular museums in Istanbul?"
        
        museum = data['museums'][0]  # Take first result
        
        if entities.museum_name:
            template = self.templates[QueryIntent.MUSEUM_INFO]['historical']
            return template.format(
                name=museum.get('name', 'Museum'),
                district=museum.get('district', 'Istanbul'),
                period=museum.get('historical_period', 'ancient times'),
                significance=museum.get('significance', 'Important cultural site.'),
                practical_info=museum.get('practical_info', 'Check opening hours before visiting.')
            )
        else:
            template = self.templates[QueryIntent.MUSEUM_INFO]['general']
            return template.format(
                name=museum.get('name', 'Museum'),
                description=museum.get('description', 'A fascinating cultural experience.')
            )
    
    def _generate_directions_response(self, entities: ExtractedEntities, data: Dict[str, Any]) -> str:
        """Generate directions response"""
        if not data.get('routes'):
            return "I need more specific locations to provide directions. Could you tell me where you're starting from and where you want to go?"
        
        route = data['routes'][0]
        transport_type = entities.transportation_type or route.get('type', 'walking')
        
        if transport_type == 'metro':
            template = self.templates[QueryIntent.DIRECTIONS]['metro']
            return template.format(
                line=route.get('line', 'metro'),
                station=route.get('station', 'nearest station'),
                walking_directions=route.get('walking_info', 'Follow the signs.')
            )
        elif transport_type == 'ferry':
            template = self.templates[QueryIntent.DIRECTIONS]['ferry']
            return template.format(
                from_port=route.get('from', 'departure port'),
                to_port=route.get('to', 'destination port'),
                additional_info=route.get('info', 'Enjoy the Bosphorus views!')
            )
        else:
            template = self.templates[QueryIntent.DIRECTIONS]['walking']
            return template.format(
                duration=route.get('duration', '10-15 minutes'),
                from_location=route.get('from', 'your location'),
                route_description=route.get('description', 'Follow the main streets.')
            )
    
    def _generate_district_response(self, entities: ExtractedEntities, data: Dict[str, Any]) -> str:
        """Generate district information response"""
        district_info = data.get('district_info', {})
        
        template = self.templates[QueryIntent.DISTRICT_INFO]['overview']
        return template.format(
            district=entities.district or 'this area',
            characteristics=district_info.get('characteristics', 'its unique charm'),
            highlights=district_info.get('highlights', 'Many interesting places to explore.')
        )
    
    def _generate_culture_response(self, entities: ExtractedEntities, data: Dict[str, Any]) -> str:
        """Generate cultural etiquette response"""
        guidelines = data.get('guidelines', 'Be respectful and observe local customs.')
        
        template = self.templates[QueryIntent.CULTURAL_ETIQUETTE]['general']
        return template.format(guidelines=guidelines)
    
    def _generate_general_response(self, entities: ExtractedEntities, data: Dict[str, Any]) -> str:
        """Generate general information response"""
        return data.get('response', 'I can help you with restaurants, museums, directions, and cultural information about Istanbul. What would you like to know?')
    
    def process_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Main method to process a natural language query"""
        
        # 1. Handle context
        context = self.handle_context(query, session_id)
        
        # 2. Classify intent
        intent = self.classify_intent(query)
        
        # 3. Extract entities
        entities = self.extract_entities(query)
        
        # 4. Apply fuzzy matching
        entities = self.apply_fuzzy_matching(query, entities)
        
        # 5. Update context
        context.last_intent = intent
        context.last_entities = entities
        self.session_contexts[session_id] = context
        
        # Return structured result for the main system to handle
        return {
            'intent': intent,
            'entities': entities,
            'context': context,
            'raw_query': query,
            'session_id': session_id,
            'confidence': self._calculate_confidence(query, intent, entities)
        }
    
    def _calculate_confidence(self, query: str, intent: QueryIntent, entities: ExtractedEntities) -> float:
        """Calculate confidence score for the classification"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on clear intent indicators
        query_lower = query.lower()
        
        if intent == QueryIntent.RESTAURANT_SEARCH:
            food_words = sum(1 for cuisine in self.cuisines if cuisine in query_lower)
            confidence += min(food_words * 0.1, 0.3)
        
        elif intent == QueryIntent.MUSEUM_INFO:
            museum_words = sum(1 for museum in self.museums if museum in query_lower)
            confidence += min(museum_words * 0.2, 0.4)
        
        # Boost for extracted entities
        entity_count = sum(1 for field in ['district', 'cuisine', 'budget', 'atmosphere'] 
                          if getattr(entities, field))
        confidence += min(entity_count * 0.05, 0.2)
        
        return min(confidence, 1.0)

# Global instance
lightweight_nlp = LightweightNLPSystem()

def process_natural_query(query: str, session_id: str = "default") -> Dict[str, Any]:
    """Main function to process natural language queries"""
    return lightweight_nlp.process_query(query, session_id)
