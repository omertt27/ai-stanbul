"""
Entity Extractor - Extract and enhance entities from user messages

This module handles entity extraction from user queries, wrapping the existing
IstanbulEntityRecognizer and adding contextual enhancements.

Week 2 Refactoring: Extracted from main_system.py
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from ..core.entity_recognition import IstanbulEntityRecognizer
from ..core.models import ConversationContext

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract and enhance entities from user messages"""
    
    def __init__(self, entity_recognizer: Optional[IstanbulEntityRecognizer] = None):
        """
        Initialize the entity extractor
        
        Args:
            entity_recognizer: Optional custom entity recognizer (uses default if None)
        """
        self.entity_recognizer = entity_recognizer or IstanbulEntityRecognizer()
        self.budget_keywords = self._initialize_budget_keywords()
        self.temporal_patterns = self._initialize_temporal_patterns()
    
    def _initialize_budget_keywords(self) -> Dict[str, List[str]]:
        """Initialize budget-related keywords"""
        return {
            'free': ['free', 'no cost', 'no entrance', 'no fee', 'ücretsiz', 'bedava'],
            'budget': ['cheap', 'budget', 'affordable', 'inexpensive', 'ucuz', 'ekonomik'],
            'moderate': ['moderate', 'mid-range', 'reasonable', 'orta', 'makul'],
            'expensive': ['expensive', 'luxury', 'upscale', 'fine dining', 'pahalı', 'lüks'],
            'premium': ['premium', 'high-end', 'exclusive', 'michelin', 'elit']
        }
    
    def _initialize_temporal_patterns(self) -> Dict[str, List[str]]:
        """Initialize temporal expression patterns"""
        return {
            'today': ['today', 'bugün'],
            'tonight': ['tonight', 'this evening', 'bu akşam', 'bu gece'],
            'tomorrow': ['tomorrow', 'yarın'],
            'this_weekend': ['this weekend', 'weekend', 'bu hafta sonu', 'hafta sonu'],
            'this_week': ['this week', 'bu hafta'],
            'this_month': ['this month', 'bu ay'],
            'morning': ['morning', 'breakfast', 'sabah', 'kahvaltı'],
            'afternoon': ['afternoon', 'lunch', 'öğle', 'öğlen'],
            'evening': ['evening', 'dinner', 'akşam', 'akşam yemeği'],
            'night': ['night', 'late night', 'gece', 'gece yarısı']
        }
    
    def extract_entities(
        self, 
        message: str, 
        context: Optional[ConversationContext] = None
    ) -> Dict[str, Any]:
        """
        Extract all entities from message with contextual enhancements
        
        Args:
            message: User's input message
            context: Optional conversation context
        
        Returns:
            Dictionary of extracted entities
        """
        # Use base entity recognizer
        entities = self.entity_recognizer.extract_entities(message)
        
        # Add budget information
        budget = self.extract_budget(message)
        if budget:
            entities['budget'] = budget
        
        # Add temporal information
        temporal = self.extract_temporal(message)
        if temporal:
            entities['temporal'] = temporal
        
        # Add GPS coordinates if present
        gps_coords = self.extract_gps_coordinates(message)
        if gps_coords:
            entities['gps_location'] = gps_coords
        
        # Add contextual enhancements from conversation history
        if context:
            entities = self._enhance_with_context(entities, context)
        
        return entities
    
    def extract_location(self, message: str) -> Optional[str]:
        """
        Extract location/place names from message
        
        Args:
            message: User's input message
        
        Returns:
            Location string or None
        """
        # Common Istanbul districts
        districts = [
            'sultanahmet', 'beyoglu', 'beyoğlu', 'galata', 'karaköy', 'karakoy',
            'taksim', 'besiktas', 'beşiktaş', 'ortakoy', 'ortaköy',
            'kadikoy', 'kadıköy', 'uskudar', 'üsküdar', 'eminonu', 'eminönü',
            'fatih', 'sisli', 'şişli', 'bakirkoy', 'bakırköy', 'levent',
            'maslak', 'etiler', 'nisantasi', 'nişantaşı'
        ]
        
        message_lower = message.lower()
        for district in districts:
            if district in message_lower:
                return district.title()
        
        return None
    
    def extract_budget(self, message: str) -> Optional[str]:
        """
        Extract budget/price range from message
        
        Args:
            message: User's input message
        
        Returns:
            Budget level string or None
        """
        message_lower = message.lower()
        
        # Check each budget category
        for level, keywords in self.budget_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return level
        
        return None
    
    def extract_temporal(self, message: str) -> Optional[Dict[str, str]]:
        """
        Extract temporal expressions from message
        
        Args:
            message: User's input message
        
        Returns:
            Dictionary with temporal information or None
        """
        message_lower = message.lower()
        
        for timeframe, keywords in self.temporal_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                return {
                    'timeframe': timeframe,
                    'original_text': next(
                        (kw for kw in keywords if kw in message_lower), 
                        timeframe
                    )
                }
        
        return None
    
    def extract_gps_coordinates(self, message: str) -> Optional[Tuple[float, float]]:
        """
        Extract GPS coordinates from message
        
        Args:
            message: User's input message
        
        Returns:
            Tuple of (latitude, longitude) or None
        """
        # Pattern for GPS coordinates (e.g., "41.0082, 28.9784")
        gps_pattern = r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
        match = re.search(gps_pattern, message)
        
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                
                # Validate coordinates are reasonable for Istanbul area
                if 40.5 <= lat <= 41.5 and 28.0 <= lon <= 29.5:
                    return (lat, lon)
            except ValueError:
                pass
        
        return None
    
    def extract_cuisine_type(self, message: str) -> Optional[List[str]]:
        """
        Extract cuisine types from message
        
        Args:
            message: User's input message
        
        Returns:
            List of cuisine types or None
        """
        cuisines = {
            'turkish': ['turkish', 'türk', 'ottoman', 'osmanlı'],
            'seafood': ['seafood', 'fish', 'deniz', 'balık'],
            'kebab': ['kebab', 'kebap', 'kebap'],
            'italian': ['italian', 'pizza', 'pasta', 'italyan'],
            'asian': ['asian', 'chinese', 'japanese', 'sushi', 'asya', 'çin', 'japon'],
            'mediterranean': ['mediterranean', 'akdeniz'],
            'middle_eastern': ['middle eastern', 'lebanese', 'syrian', 'lübnan', 'suriye'],
            'vegetarian': ['vegetarian', 'vegan', 'vejeteryan', 'vegan'],
            'street_food': ['street food', 'fast food', 'sokak', 'fast food']
        }
        
        message_lower = message.lower()
        found_cuisines = []
        
        for cuisine, keywords in cuisines.items():
            if any(keyword in message_lower for keyword in keywords):
                found_cuisines.append(cuisine)
        
        return found_cuisines if found_cuisines else None
    
    def extract_accessibility_needs(self, message: str) -> Optional[Dict[str, bool]]:
        """
        Extract accessibility requirements from message
        
        Args:
            message: User's input message
        
        Returns:
            Dictionary of accessibility needs or None
        """
        accessibility_keywords = {
            'wheelchair': ['wheelchair', 'accessibility', 'disabled', 'tekerlekli', 'engelli'],
            'elevator': ['elevator', 'lift', 'asansör'],
            'family_friendly': ['family', 'kids', 'children', 'stroller', 'aile', 'çocuk'],
            'parking': ['parking', 'park', 'otopark']
        }
        
        message_lower = message.lower()
        needs = {}
        
        for need, keywords in accessibility_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                needs[need] = True
        
        return needs if needs else None
    
    def extract_group_size(self, message: str) -> Optional[int]:
        """
        Extract group size from message
        
        Args:
            message: User's input message
        
        Returns:
            Number of people or None
        """
        # Patterns for group size
        patterns = [
            r'(\d+)\s*people',
            r'(\d+)\s*person',
            r'(\d+)\s*pax',
            r'(\d+)\s*kişi',
            r'group of\s*(\d+)',
            r'party of\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _enhance_with_context(
        self, 
        entities: Dict[str, Any], 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Enhance entities with conversation context
        
        Args:
            entities: Base extracted entities
            context: Conversation context
        
        Returns:
            Enhanced entities dictionary
        """
        # Add location from context if not in message
        if not entities.get('location') and hasattr(context, 'current_location'):
            entities['location'] = context.current_location
        
        # Add previous preferences from context
        if hasattr(context, 'preferences'):
            entities['context_preferences'] = context.preferences
        
        # Add conversation turn count for relevance
        if hasattr(context, 'conversation_history'):
            entities['conversation_turn'] = len(context.conversation_history)
        
        return entities
    
    def extract_all_enhanced(self, message: str) -> Dict[str, Any]:
        """
        Extract all possible entities with maximum detail
        
        Args:
            message: User's input message
        
        Returns:
            Comprehensive dictionary of all extracted entities
        """
        entities = {
            'base': self.entity_recognizer.extract_entities(message),
            'location': self.extract_location(message),
            'budget': self.extract_budget(message),
            'temporal': self.extract_temporal(message),
            'gps_coordinates': self.extract_gps_coordinates(message),
            'cuisines': self.extract_cuisine_type(message),
            'accessibility': self.extract_accessibility_needs(message),
            'group_size': self.extract_group_size(message)
        }
        
        # Remove None values
        return {k: v for k, v in entities.items() if v is not None}
