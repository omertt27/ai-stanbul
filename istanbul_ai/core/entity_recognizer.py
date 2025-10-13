"""
Istanbul Entity Recognizer for Istanbul AI System
"""

import re
from typing import Dict, List, Set
from ..utils.constants import ISTANBUL_DISTRICTS, CUISINE_TYPES, TRANSPORT_MODES


class IstanbulEntityRecognizer:
    """
    Simplified Istanbul-specific entity recognizer
    Extracted from the monolithic system for better maintainability
    """
    
    def __init__(self):
        self.districts = set(ISTANBUL_DISTRICTS)
        self.cuisines = set(CUISINE_TYPES)
        self.transport_modes = set(TRANSPORT_MODES)
        
        # Common variations and aliases
        self.district_aliases = {
            'sultanahmet': ['sultan ahmet', 'blue mosque area', 'historic peninsula'],
            'beyoglu': ['beyoğlu', 'pera', 'galata'],
            'taksim': ['taksim square', 'istiklal'],
            'kadikoy': ['kadıköy', 'asian side'],
            'besiktas': ['beşiktaş', 'dolmabahce', 'ortakoy'],
        }
        
        # Time-related patterns
        self.time_patterns = {
            'morning': r'\b(morning|breakfast|am|early)\b',
            'afternoon': r'\b(afternoon|lunch|pm)\b',
            'evening': r'\b(evening|dinner|night)\b',
            'now': r'\b(now|current|today|right now)\b',
            'tomorrow': r'\b(tomorrow|next day)\b',
            'weekend': r'\b(weekend|saturday|sunday)\b'
        }
        
        # Budget patterns
        self.budget_patterns = {
            'budget': r'\b(cheap|budget|affordable|low cost|inexpensive)\b',
            'luxury': r'\b(luxury|expensive|high end|premium|fine dining)\b',
            'moderate': r'\b(moderate|mid range|reasonable)\b'
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        text_lower = text.lower()
        entities = {
            'districts': [],
            'cuisines': [],
            'transport': [],
            'time': [],
            'budget': [],
            'dietary': []
        }
        
        # Extract districts
        for district in self.districts:
            if district.lower() in text_lower:
                entities['districts'].append(district)
        
        # Check aliases
        for district, aliases in self.district_aliases.items():
            for alias in aliases:
                if alias in text_lower:
                    entities['districts'].append(district.title())
        
        # Extract cuisines
        for cuisine in self.cuisines:
            if cuisine.lower() in text_lower:
                entities['cuisines'].append(cuisine)
        
        # Extract transport modes
        for transport in self.transport_modes:
            if transport.lower() in text_lower:
                entities['transport'].append(transport)
        
        # Extract time references
        for time_type, pattern in self.time_patterns.items():
            if re.search(pattern, text_lower):
                entities['time'].append(time_type)
        
        # Extract budget references
        for budget_type, pattern in self.budget_patterns.items():
            if re.search(pattern, text_lower):
                entities['budget'].append(budget_type)
        
        # Extract dietary restrictions
        dietary_terms = ['vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free', 'gluten free']
        for dietary in dietary_terms:
            if dietary in text_lower:
                entities['dietary'].append(dietary)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities

    def get_location_context(self, text: str) -> Dict[str, any]:
        """Get location-specific context"""
        entities = self.extract_entities(text)
        context = {
            'has_location': len(entities['districts']) > 0,
            'primary_location': entities['districts'][0] if entities['districts'] else None,
            'location_count': len(entities['districts']),
            'is_multi_location': len(entities['districts']) > 1
        }
        return context

    def detect_intent_signals(self, text: str) -> Dict[str, float]:
        """Detect intent signals with confidence scores"""
        text_lower = text.lower()
        signals = {}
        
        # Restaurant intent signals
        restaurant_terms = ['eat', 'food', 'restaurant', 'meal', 'hungry', 'dine', 'cuisine']
        restaurant_score = sum(1 for term in restaurant_terms if term in text_lower)
        if restaurant_score > 0:
            signals['restaurant'] = min(restaurant_score / len(restaurant_terms), 1.0)
        
        # Transportation intent signals
        transport_terms = ['go', 'get to', 'transport', 'metro', 'bus', 'taxi', 'how to reach']
        transport_score = sum(1 for term in transport_terms if term in text_lower)
        if transport_score > 0:
            signals['transportation'] = min(transport_score / len(transport_terms), 1.0)
        
        # Events intent signals
        event_terms = ['event', 'happening', 'show', 'concert', 'performance', 'exhibition']
        event_score = sum(1 for term in event_terms if term in text_lower)
        if event_score > 0:
            signals['events'] = min(event_score / len(event_terms), 1.0)
        
        # Attraction intent signals
        attraction_terms = ['visit', 'see', 'attraction', 'museum', 'palace', 'tower', 'mosque']
        attraction_score = sum(1 for term in attraction_terms if term in text_lower)
        if attraction_score > 0:
            signals['attractions'] = min(attraction_score / len(attraction_terms), 1.0)
        
        return signals
