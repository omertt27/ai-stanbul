"""
Context Builder - Build enhanced context for responses

This module builds enriched context for response generation by combining:
- User profile preferences
- Conversation history
- Location information
- Temporal context (time of day, season)
- Session data

Week 7-8 Refactoring: Extracted from main_system.py
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds enriched context for response generation
    
    Combines user preferences, conversation history, location data,
    and temporal context to create comprehensive context objects.
    """
    
    def __init__(self):
        """Initialize context builder"""
        logger.info("✅ ContextBuilder initialized")
    
    def build_response_context(
        self,
        user_profile: Optional[Any] = None,
        conversation_context: Optional[Any] = None,
        message: str = "",
        entities: Optional[Dict] = None,
        neural_insights: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive response context
        
        Args:
            user_profile: User profile with preferences
            conversation_context: Conversation history and state
            message: Current message
            entities: Extracted entities
            neural_insights: ML/neural network insights
        
        Returns:
            Enriched context dictionary
        """
        context = {}
        
        # Add user preferences
        if user_profile:
            context.update(self._extract_user_preferences(user_profile))
        
        # Add conversation history
        if conversation_context:
            context.update(self._extract_conversation_history(conversation_context))
        
        # Add location context
        if entities:
            context.update(self._extract_location_context(entities))
        
        # Add temporal context
        context.update(self._extract_temporal_context())
        
        # Add neural insights if available
        if neural_insights:
            context['neural_insights'] = neural_insights
        
        # Add message metadata
        context['message'] = message
        context['message_length'] = len(message)
        context['has_entities'] = bool(entities and len(entities) > 0)
        
        return context
    
    def _extract_user_preferences(self, user_profile: Any) -> Dict[str, Any]:
        """
        Extract user preferences from profile
        
        Args:
            user_profile: User profile object
        
        Returns:
            Dictionary of user preferences
        """
        preferences = {}
        
        # Language preference
        if hasattr(user_profile, 'language_preference'):
            preferences['language'] = getattr(user_profile, 'language_preference', 'en')
        
        # Budget preference
        if hasattr(user_profile, 'budget_preference'):
            preferences['budget'] = getattr(user_profile, 'budget_preference', 'medium')
        
        # Interests
        if hasattr(user_profile, 'interests'):
            preferences['interests'] = getattr(user_profile, 'interests', [])
        
        # Dietary restrictions (for restaurant queries)
        if hasattr(user_profile, 'dietary_restrictions'):
            preferences['dietary_restrictions'] = getattr(user_profile, 'dietary_restrictions', [])
        
        # Preferred districts
        if hasattr(user_profile, 'preferred_districts'):
            preferences['preferred_districts'] = getattr(user_profile, 'preferred_districts', [])
        
        # User type (tourist, local, etc.)
        if hasattr(user_profile, 'user_type'):
            preferences['user_type'] = getattr(user_profile, 'user_type', 'tourist')
        
        # Session context from user profile
        if hasattr(user_profile, 'session_context'):
            session = getattr(user_profile, 'session_context', {})
            if isinstance(session, dict):
                preferences['session_language'] = session.get('language_preference', '')
                preferences['session_budget'] = session.get('budget_preference', '')
        
        return preferences
    
    def _extract_conversation_history(self, conversation_context: Any) -> Dict[str, Any]:
        """
        Extract conversation history and state
        
        Args:
            conversation_context: Conversation context object
        
        Returns:
            Dictionary of conversation data
        """
        history = {}
        
        # Previous queries
        if hasattr(conversation_context, 'previous_queries'):
            queries = getattr(conversation_context, 'previous_queries', [])
            history['previous_queries'] = queries
            history['query_count'] = len(queries)
        
        # Previous intents
        if hasattr(conversation_context, 'previous_intents'):
            intents = getattr(conversation_context, 'previous_intents', [])
            history['previous_intents'] = intents
        
        # Current topic/focus
        if hasattr(conversation_context, 'current_topic'):
            history['current_topic'] = getattr(conversation_context, 'current_topic', '')
        
        # Last location mentioned
        if hasattr(conversation_context, 'last_location'):
            history['last_location'] = getattr(conversation_context, 'last_location', '')
        
        # Session start time
        if hasattr(conversation_context, 'session_start'):
            history['session_start'] = getattr(conversation_context, 'session_start', None)
        
        return history
    
    def _extract_location_context(self, entities: Dict) -> Dict[str, Any]:
        """
        Extract location context from entities
        
        Args:
            entities: Extracted entities dictionary
        
        Returns:
            Dictionary of location context
        """
        location = {}
        
        # Locations
        if 'locations' in entities:
            locations = entities.get('locations', [])
            if locations:
                location['primary_location'] = locations[0] if len(locations) > 0 else None
                location['destination'] = locations[1] if len(locations) > 1 else None
                location['all_locations'] = locations
        
        # Districts
        if 'districts' in entities:
            location['districts'] = entities.get('districts', [])
        
        # Neighborhoods
        if 'neighborhoods' in entities:
            location['neighborhoods'] = entities.get('neighborhoods', [])
        
        # GPS coordinates
        if 'gps_location' in entities:
            location['gps_location'] = entities.get('gps_location')
        
        # Transport modes
        if 'transport_modes' in entities:
            location['transport_modes'] = entities.get('transport_modes', [])
        
        return location
    
    def _extract_temporal_context(self) -> Dict[str, Any]:
        """
        Extract temporal context (time of day, day of week, season)
        
        Returns:
            Dictionary of temporal context
        """
        now = datetime.now()
        
        temporal = {
            'current_time': now,
            'hour': now.hour,
            'day_of_week': now.strftime('%A'),
            'is_weekend': now.weekday() >= 5,
            'date': now.date(),
            'time_of_day': self._get_time_of_day(now.hour),
            'season': self._get_season(now.month)
        }
        
        return temporal
    
    def _get_time_of_day(self, hour: int) -> str:
        """
        Get time of day category
        
        Args:
            hour: Hour of day (0-23)
        
        Returns:
            Time of day string
        """
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _get_season(self, month: int) -> str:
        """
        Get current season (Northern Hemisphere)
        
        Args:
            month: Month number (1-12)
        
        Returns:
            Season string
        """
        if 3 <= month <= 5:
            return 'spring'
        elif 6 <= month <= 8:
            return 'summer'
        elif 9 <= month <= 11:
            return 'fall'
        else:
            return 'winter'
    
    def enhance_context_with_ml_insights(
        self,
        context: Dict[str, Any],
        ml_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance context with ML insights
        
        Args:
            context: Base context dictionary
            ml_insights: ML/neural insights
        
        Returns:
            Enhanced context
        """
        enhanced = context.copy()
        
        # Add ML predictions
        if 'intent_confidence' in ml_insights:
            enhanced['ml_confidence'] = ml_insights['intent_confidence']
        
        if 'suggested_categories' in ml_insights:
            enhanced['ml_categories'] = ml_insights['suggested_categories']
        
        if 'sentiment' in ml_insights:
            enhanced['sentiment'] = ml_insights['sentiment']
        
        if 'complexity_score' in ml_insights:
            enhanced['complexity'] = ml_insights['complexity_score']
        
        return enhanced
    
    def build_intelligent_user_context(
        self,
        message: str,
        neural_insights: Optional[Dict] = None,
        user_profile: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Build intelligent user context for ML handlers
        
        This is used by transportation and other ML-enhanced handlers.
        
        Args:
            message: User message
            neural_insights: Neural network insights
            user_profile: User profile
        
        Returns:
            Intelligent context dictionary
        """
        context = {
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add user preferences
        if user_profile:
            if hasattr(user_profile, 'language_preference'):
                context['language'] = getattr(user_profile, 'language_preference', 'en')
            if hasattr(user_profile, 'budget_preference'):
                context['budget'] = getattr(user_profile, 'budget_preference', 'medium')
        
        # Add neural insights
        if neural_insights:
            context['neural_insights'] = neural_insights
        
        # Add temporal context
        now = datetime.now()
        context['time_of_day'] = self._get_time_of_day(now.hour)
        context['is_weekend'] = now.weekday() >= 5
        
        return context
