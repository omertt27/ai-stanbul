"""
ML Context Builder for Istanbul AI Chat System

Centralizes ML-powered context extraction and enrichment across all features.
Builds comprehensive user context from queries, neural insights, and user profiles.

Features:
- Unified context extraction pipeline
- Neural insights integration
- User preference modeling
- Sentiment and intent analysis
- Multi-dimensional context building

Author: Istanbul AI Team
Date: October 27, 2025
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class MLContextBuilder:
    """
    Builds rich ML-enhanced context from user queries and neural insights.
    """
    
    def __init__(self, neural_processor, user_manager, weather_service=None):
        """
        Initialize the ML Context Builder.
        
        Args:
            neural_processor: ML model for semantic understanding
            user_manager: User profile and history manager
            weather_service: Optional weather integration
        """
        self.neural_processor = neural_processor
        self.user_manager = user_manager
        self.weather_service = weather_service
        
        logger.info("✅ MLContextBuilder initialized")
    
    # ==================== PUBLIC API ====================
    
    def build_intelligent_context(self, message: str, neural_insights: Dict[str, Any],
                                 user_profile: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """
        Build comprehensive ML-enhanced context for any query type.
        
        This is the main entry point that combines:
        - Query analysis
        - Neural insights
        - User preferences
        - Environmental context (weather, time)
        - Intent-specific context
        
        Args:
            message: User's query text
            neural_insights: ML-generated insights from neural processor
            user_profile: User's profile and preferences
            intent: Detected intent (restaurant, attraction, transport, etc.)
            
        Returns:
            Comprehensive context dictionary
        """
        try:
            logger.info(f"🧠 Building ML context for intent: {intent}")
            
            context = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_profile.get('user_id', 'unknown'),
                'intent': intent,
                'original_query': message,
                'message_lower': message.lower()
            }
            
            # 1. Core Neural Insights
            context.update(self._extract_neural_insights(neural_insights))
            
            # 2. User Profile Context
            context['user_context'] = self._build_user_context(user_profile)
            
            # 3. Environmental Context
            context['environmental'] = self._build_environmental_context()
            
            # 4. Query Analysis
            context['query_features'] = self._analyze_query(message, neural_insights)
            
            # 5. Intent-Specific Context
            context['intent_context'] = self._build_intent_specific_context(
                message, neural_insights, intent
            )
            
            # 6. Behavioral Signals
            context['behavioral'] = self._extract_behavioral_signals(
                message, neural_insights, user_profile
            )
            
            logger.debug(f"📊 ML Context built: {len(context)} top-level keys")
            return context
            
        except Exception as e:
            logger.error(f"❌ Error building ML context: {str(e)}", exc_info=True)
            return self._build_minimal_context(message, intent)
    
    # ==================== NEURAL INSIGHTS EXTRACTION ====================
    
    def _extract_neural_insights(self, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and structure key neural insights.
        
        Returns:
            Dictionary with:
            - sentiment: positive/neutral/negative
            - urgency: immediate/soon/flexible
            - confidence: float (0-1)
            - entities: List of detected entities
            - keywords: List of important keywords
            - embeddings: Query embedding vector
        """
        return {
            'sentiment': neural_insights.get('sentiment', 'neutral'),
            'urgency': neural_insights.get('urgency', 'flexible'),
            'confidence': neural_insights.get('confidence', 0.7),
            'entities': neural_insights.get('entities', []),
            'keywords': neural_insights.get('keywords', []),
            'query_embedding': neural_insights.get('query_embedding'),
            'detected_language': neural_insights.get('language', 'en'),
            'complexity': neural_insights.get('query_complexity', 'simple'),
            'special_requirements': neural_insights.get('special_requirements', [])
        }
    
    # ==================== USER CONTEXT ====================
    
    def _build_user_context(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive user context from profile and history.
        
        Returns:
            Dictionary with:
            - preferences: User preferences
            - history: Past interactions
            - profile_features: Demographic and behavioral features
            - personalization_score: How well we know the user
        """
        user_context = {
            'user_id': user_profile.get('user_id', 'unknown'),
            'is_new_user': user_profile.get('interaction_count', 0) < 3,
            'interaction_count': user_profile.get('interaction_count', 0)
        }
        
        # Preferences
        user_context['preferences'] = {
            'favorite_cuisines': user_profile.get('favorite_cuisines', []),
            'dietary_restrictions': user_profile.get('dietary_restrictions', []),
            'favorite_attraction_types': user_profile.get('favorite_attraction_types', []),
            'preferred_transport': user_profile.get('preferred_transport_mode', []),
            'budget_preference': user_profile.get('preferred_budget_level', 'moderate'),
            'language': user_profile.get('preferred_language', 'en')
        }
        
        # History
        user_context['history'] = {
            'visited_restaurants': user_profile.get('restaurant_history', []),
            'visited_attractions': user_profile.get('attraction_history', []),
            'past_queries': user_profile.get('recent_queries', []),
            'successful_recommendations': user_profile.get('successful_recs', 0)
        }
        
        # Profile features
        user_context['profile_features'] = {
            'traveler_type': user_profile.get('traveler_type', 'tourist'),  # tourist/local/business
            'accessibility_needs': user_profile.get('accessibility_needs', []),
            'travel_party': user_profile.get('typical_party_size', 2),
            'interests': user_profile.get('interests', [])
        }
        
        # Personalization score (how well we know this user)
        interaction_score = min(user_profile.get('interaction_count', 0) / 10, 1.0)
        preference_score = len(user_profile.get('favorite_cuisines', [])) / 5
        user_context['personalization_score'] = (interaction_score + preference_score) / 2
        
        return user_context
    
    # ==================== ENVIRONMENTAL CONTEXT ====================
    
    def _build_environmental_context(self) -> Dict[str, Any]:
        """
        Build environmental context (weather, time, etc.).
        
        Returns:
            Dictionary with:
            - weather: Current weather conditions
            - time: Time-based context
            - season: Current season
            - events: Major events happening
        """
        env_context = {}
        
        # Weather context
        if self.weather_service:
            try:
                weather = self.weather_service.get_current_weather()
                env_context['weather'] = {
                    'condition': weather.get('condition', 'unknown'),
                    'temperature': weather.get('temperature', 20),
                    'feels_like': weather.get('feels_like', 20),
                    'humidity': weather.get('humidity', 60),
                    'is_rainy': weather.get('condition') in ['rainy', 'stormy', 'drizzle'],
                    'is_cold': weather.get('temperature', 20) < 15,
                    'is_hot': weather.get('temperature', 20) > 30,
                    'is_good_weather': weather.get('condition') in ['sunny', 'clear', 'partly cloudy'],
                    'indoor_recommended': weather.get('condition') in ['rainy', 'stormy'] or 
                                         weather.get('temperature', 20) < 10 or 
                                         weather.get('temperature', 20) > 35
                }
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch weather: {e}")
                env_context['weather'] = self._default_weather()
        else:
            env_context['weather'] = self._default_weather()
        
        # Time context
        now = datetime.now()
        env_context['time'] = {
            'hour': now.hour,
            'day_of_week': now.strftime('%A'),
            'day_name': now.strftime('%A'),
            'month': now.month,
            'month_name': now.strftime('%B'),
            'is_weekend': now.weekday() >= 5,
            'is_morning': 6 <= now.hour < 12,
            'is_afternoon': 12 <= now.hour < 17,
            'is_evening': 17 <= now.hour < 22,
            'is_night': now.hour >= 22 or now.hour < 6,
            'is_business_hours': 9 <= now.hour < 18 and now.weekday() < 5,
            'timestamp': now.isoformat()
        }
        
        # Season context
        env_context['season'] = self._get_season(now.month)
        
        # Istanbul-specific context
        env_context['istanbul_context'] = {
            'prayer_times_relevant': True,
            'ramadan_period': self._is_ramadan_period(now),
            'tourist_season': now.month in [4, 5, 6, 7, 8, 9, 10],  # Apr-Oct
            'peak_tourist_season': now.month in [6, 7, 8]  # Jun-Aug
        }
        
        return env_context
    
    def _default_weather(self) -> Dict[str, Any]:
        """Return default weather when service unavailable."""
        return {
            'condition': 'unknown',
            'temperature': 20,
            'feels_like': 20,
            'humidity': 60,
            'is_rainy': False,
            'is_cold': False,
            'is_hot': False,
            'is_good_weather': True,
            'indoor_recommended': False
        }
    
    def _get_season(self, month: int) -> str:
        """Get current season."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def _is_ramadan_period(self, date: datetime) -> bool:
        """Check if current date is during Ramadan period (simplified)."""
        # This is a simplified check - in production, use a proper Islamic calendar library
        # Ramadan dates change each year
        return False  # Placeholder
    
    # ==================== QUERY ANALYSIS ====================
    
    def _analyze_query(self, message: str, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query structure and features.
        
        Returns:
            Dictionary with:
            - length: Query length category
            - complexity: Simple/medium/complex
            - has_constraints: Boolean
            - is_comparative: Boolean (comparing options)
            - is_exploratory: Boolean (open-ended)
            - specificity: Low/medium/high
        """
        message_lower = message.lower()
        word_count = len(message.split())
        
        analysis = {
            'word_count': word_count,
            'char_count': len(message),
            'length_category': self._categorize_length(word_count),
            'complexity': neural_insights.get('query_complexity', self._assess_complexity(message)),
            'has_question': '?' in message,
            'has_constraints': self._has_constraints(message_lower),
            'is_comparative': self._is_comparative(message_lower),
            'is_exploratory': self._is_exploratory(message_lower),
            'is_specific': self._is_specific_query(message_lower),
            'specificity_score': self._compute_specificity(message_lower, neural_insights),
            'politeness_level': self._assess_politeness(message_lower),
            'has_negations': self._has_negations(message_lower)
        }
        
        return analysis
    
    def _categorize_length(self, word_count: int) -> str:
        """Categorize query length."""
        if word_count <= 3:
            return 'very_short'
        elif word_count <= 7:
            return 'short'
        elif word_count <= 15:
            return 'medium'
        else:
            return 'long'
    
    def _assess_complexity(self, message: str) -> str:
        """Assess query complexity."""
        # Simple heuristic: count of constraints and conditions
        complexity_indicators = ['and', 'but', 'or', 'with', 'without', 'near', 'between']
        count = sum(1 for ind in complexity_indicators if ind in message.lower())
        
        if count == 0:
            return 'simple'
        elif count <= 2:
            return 'medium'
        else:
            return 'complex'
    
    def _has_constraints(self, message: str) -> bool:
        """Check if query has explicit constraints."""
        constraint_keywords = [
            'budget', 'cheap', 'expensive', 'under', 'less than', 'more than',
            'vegetarian', 'vegan', 'halal', 'kosher',
            'near', 'close to', 'walking distance',
            'open now', 'open late', 'reservation',
            'indoor', 'outdoor', 'kid-friendly', 'romantic'
        ]
        return any(kw in message for kw in constraint_keywords)
    
    def _is_comparative(self, message: str) -> bool:
        """Check if query is comparing options."""
        comparative_keywords = [
            'better', 'best', 'worse', 'worst', 'compare',
            'difference', 'versus', 'vs', 'or', 'between'
        ]
        return any(kw in message for kw in comparative_keywords)
    
    def _is_exploratory(self, message: str) -> bool:
        """Check if query is open-ended/exploratory."""
        exploratory_keywords = [
            'what', 'where', 'recommend', 'suggest', 'show me',
            'any', 'some', 'ideas', 'options', 'what about'
        ]
        return any(kw in message for kw in exploratory_keywords)
    
    def _is_specific_query(self, message: str) -> bool:
        """Check if query is specific (mentions specific places/names)."""
        # This is a simplified check - in production, use NER
        return any(word[0].isupper() for word in message.split()[1:])
    
    def _compute_specificity(self, message: str, neural_insights: Dict[str, Any]) -> float:
        """Compute query specificity score (0-1)."""
        score = 0.0
        
        # Entities increase specificity
        entities = neural_insights.get('entities', [])
        score += min(len(entities) * 0.2, 0.4)
        
        # Constraints increase specificity
        if self._has_constraints(message):
            score += 0.3
        
        # Specific names increase specificity
        if self._is_specific_query(message):
            score += 0.2
        
        # Very short queries are less specific
        if len(message.split()) <= 3:
            score *= 0.5
        
        return min(score, 1.0)
    
    def _assess_politeness(self, message: str) -> str:
        """Assess politeness level of query."""
        polite_indicators = ['please', 'could you', 'would you', 'thank you', 'thanks']
        if any(ind in message for ind in polite_indicators):
            return 'polite'
        else:
            return 'neutral'
    
    def _has_negations(self, message: str) -> bool:
        """Check if query contains negations."""
        negation_words = ['not', 'no', 'never', 'without', 'avoid', 'except', 'excluding']
        return any(neg in message for neg in negation_words)
    
    # ==================== INTENT-SPECIFIC CONTEXT ====================
    
    def _build_intent_specific_context(self, message: str, neural_insights: Dict[str, Any],
                                      intent: str) -> Dict[str, Any]:
        """
        Build context specific to the detected intent.
        
        This allows each intent handler to get pre-processed context.
        """
        message_lower = message.lower()
        
        if intent == 'restaurant':
            return self._build_restaurant_context(message_lower, neural_insights)
        elif intent == 'attraction':
            return self._build_attraction_context(message_lower, neural_insights)
        elif intent == 'transport':
            return self._build_transport_context(message_lower, neural_insights)
        elif intent == 'weather':
            return self._build_weather_context(message_lower, neural_insights)
        elif intent == 'events':
            return self._build_events_context(message_lower, neural_insights)
        elif intent == 'shopping':
            return self._build_shopping_context(message_lower, neural_insights)
        else:
            return {}
    
    def _build_restaurant_context(self, message: str, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Build restaurant-specific context."""
        return {
            'cuisine_mentioned': self._extract_cuisine(message),
            'meal_time': self._detect_meal_time(message),
            'dining_occasion': self._detect_dining_occasion(message),
            'party_size': self._extract_party_size(message),
            'budget_indicators': self._extract_budget_indicators(message)
        }
    
    def _build_attraction_context(self, message: str, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Build attraction-specific context."""
        return {
            'attraction_types': self._extract_attraction_types(message),
            'interests': self._extract_interests(message),
            'time_available': self._extract_time_available(message),
            'indoor_outdoor': self._extract_indoor_outdoor(message)
        }
    
    def _build_transport_context(self, message: str, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Build transport-specific context."""
        return {
            'origin': self._extract_location_entity(message, 'from'),
            'destination': self._extract_location_entity(message, 'to'),
            'transport_mode': self._extract_transport_mode(message),
            'departure_time': self._extract_time_reference(message)
        }
    
    def _build_weather_context(self, message: str, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Build weather-specific context."""
        return {
            'time_reference': self._extract_time_reference(message),
            'specific_metric': self._extract_weather_metric(message)
        }
    
    def _build_events_context(self, message: str, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Build events-specific context."""
        return {
            'event_type': self._extract_event_type(message),
            'date_range': self._extract_date_range(message)
        }
    
    def _build_shopping_context(self, message: str, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Build shopping-specific context."""
        return {
            'item_category': self._extract_shopping_category(message),
            'shopping_type': self._extract_shopping_type(message)
        }
    
    # ==================== BEHAVIORAL SIGNALS ====================
    
    def _extract_behavioral_signals(self, message: str, neural_insights: Dict[str, Any],
                                   user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract behavioral signals that indicate user state and needs.
        
        Returns:
            Dictionary with:
            - urgency: How urgent the request is
            - certainty: How certain the user is about what they want
            - engagement_level: How engaged the user is
            - frustration_indicators: Signs of user frustration
            - exploration_mode: User is exploring vs has specific goal
        """
        message_lower = message.lower()
        
        return {
            'urgency': self._assess_urgency(message_lower, neural_insights),
            'certainty': self._assess_certainty(message_lower),
            'engagement_level': self._assess_engagement(user_profile),
            'has_frustration': self._detect_frustration(message_lower),
            'exploration_mode': self._is_exploratory(message_lower),
            'needs_clarification': self._needs_clarification(message_lower, neural_insights),
            'is_follow_up': self._is_follow_up_query(message_lower, user_profile)
        }
    
    def _assess_urgency(self, message: str, neural_insights: Dict[str, Any]) -> str:
        """Assess urgency level."""
        # Check neural insights first
        if 'urgency' in neural_insights:
            return neural_insights['urgency']
        
        immediate_indicators = ['now', 'right now', 'immediately', 'urgent', 'asap', 'quickly']
        soon_indicators = ['soon', 'today', 'tonight', 'this evening']
        
        if any(ind in message for ind in immediate_indicators):
            return 'immediate'
        elif any(ind in message for ind in soon_indicators):
            return 'soon'
        else:
            return 'flexible'
    
    def _assess_certainty(self, message: str) -> str:
        """Assess how certain user is about what they want."""
        uncertain_indicators = ['maybe', 'perhaps', 'not sure', 'uncertain', 'any', 'some']
        certain_indicators = ['definitely', 'must', 'need', 'want', 'looking for']
        
        if any(ind in message for ind in certain_indicators):
            return 'high'
        elif any(ind in message for ind in uncertain_indicators):
            return 'low'
        else:
            return 'medium'
    
    def _assess_engagement(self, user_profile: Dict[str, Any]) -> str:
        """Assess user engagement level based on history."""
        interaction_count = user_profile.get('interaction_count', 0)
        
        if interaction_count == 0:
            return 'new'
        elif interaction_count < 3:
            return 'low'
        elif interaction_count < 10:
            return 'medium'
        else:
            return 'high'
    
    def _detect_frustration(self, message: str) -> bool:
        """Detect signs of user frustration."""
        frustration_indicators = [
            'still', 'again', 'not working', 'doesn\'t work', 'wrong',
            'no', 'nothing', 'can\'t', 'won\'t', 'doesn\'t'
        ]
        return any(ind in message for ind in frustration_indicators)
    
    def _needs_clarification(self, message: str, neural_insights: Dict[str, Any]) -> bool:
        """Check if query needs clarification."""
        # Very short or ambiguous queries
        if len(message.split()) <= 2:
            return True
        
        # Low confidence from neural processor
        if neural_insights.get('confidence', 1.0) < 0.5:
            return True
        
        return False
    
    def _is_follow_up_query(self, message: str, user_profile: Dict[str, Any]) -> bool:
        """Check if this is a follow-up to previous query."""
        follow_up_indicators = ['also', 'and', 'what about', 'how about', 'another']
        recent_queries = user_profile.get('recent_queries', [])
        
        return (any(ind in message for ind in follow_up_indicators) or 
                len(recent_queries) > 0)
    
    # ==================== HELPER EXTRACTION METHODS ====================
    
    def _extract_cuisine(self, message: str) -> Optional[str]:
        """Extract cuisine type from message."""
        cuisines = ['turkish', 'italian', 'chinese', 'japanese', 'indian', 'french', 
                   'mediterranean', 'asian', 'american', 'mexican', 'seafood']
        for cuisine in cuisines:
            if cuisine in message:
                return cuisine
        return None
    
    def _detect_meal_time(self, message: str) -> Optional[str]:
        """Detect meal time."""
        if any(kw in message for kw in ['breakfast', 'morning']):
            return 'breakfast'
        elif any(kw in message for kw in ['lunch', 'afternoon']):
            return 'lunch'
        elif any(kw in message for kw in ['dinner', 'evening']):
            return 'dinner'
        return None
    
    def _detect_dining_occasion(self, message: str) -> str:
        """Detect dining occasion."""
        if any(kw in message for kw in ['business', 'meeting', 'client']):
            return 'business'
        elif any(kw in message for kw in ['romantic', 'date', 'anniversary']):
            return 'romantic'
        elif any(kw in message for kw in ['family', 'kids', 'children']):
            return 'family'
        elif any(kw in message for kw in ['celebration', 'birthday', 'party']):
            return 'celebration'
        return 'casual'
    
    def _extract_party_size(self, message: str) -> int:
        """Extract party size."""
        numbers = re.findall(r'\b(\d+)\s*(?:people|person|pax|guest)', message)
        if numbers:
            return int(numbers[0])
        return 2  # Default
    
    def _extract_budget_indicators(self, message: str) -> List[str]:
        """Extract budget-related keywords."""
        indicators = []
        budget_keywords = {
            'cheap': 'budget',
            'affordable': 'budget',
            'expensive': 'upscale',
            'luxury': 'luxury',
            'fine dining': 'luxury',
            'budget': 'budget'
        }
        for keyword, level in budget_keywords.items():
            if keyword in message:
                indicators.append(level)
        return indicators
    
    def _extract_attraction_types(self, message: str) -> List[str]:
        """Extract attraction types."""
        types = []
        type_keywords = {
            'museum': 'museum',
            'mosque': 'religious',
            'church': 'religious',
            'palace': 'historical',
            'park': 'nature',
            'beach': 'nature',
            'market': 'shopping',
            'bazaar': 'shopping'
        }
        for keyword, type_name in type_keywords.items():
            if keyword in message:
                types.append(type_name)
        return types
    
    def _extract_interests(self, message: str) -> List[str]:
        """Extract user interests."""
        interests = []
        interest_keywords = ['history', 'art', 'architecture', 'photography', 'culture', 'food']
        for interest in interest_keywords:
            if interest in message:
                interests.append(interest)
        return interests
    
    def _extract_time_available(self, message: str) -> Optional[str]:
        """Extract time available."""
        if any(kw in message for kw in ['quick', 'short', 'hour']):
            return 'few_hours'
        elif any(kw in message for kw in ['half day', 'morning', 'afternoon']):
            return 'half_day'
        elif any(kw in message for kw in ['full day', 'whole day']):
            return 'full_day'
        return None
    
    def _extract_indoor_outdoor(self, message: str) -> Optional[str]:
        """Extract indoor/outdoor preference."""
        if 'indoor' in message or 'inside' in message:
            return 'indoor'
        elif 'outdoor' in message or 'outside' in message:
            return 'outdoor'
        return None
    
    def _extract_location_entity(self, message: str, direction: str) -> Optional[str]:
        """Extract location entity (for transport queries)."""
        # Simplified - in production, use proper NER
        if direction == 'from':
            pattern = r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        else:
            pattern = r'to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        
        match = re.search(pattern, message)
        return match.group(1) if match else None
    
    def _extract_transport_mode(self, message: str) -> Optional[str]:
        """Extract preferred transport mode."""
        modes = ['metro', 'bus', 'tram', 'ferry', 'taxi', 'walking', 'driving']
        for mode in modes:
            if mode in message:
                return mode
        return None
    
    def _extract_time_reference(self, message: str) -> Optional[str]:
        """Extract time reference."""
        if any(kw in message for kw in ['today', 'now']):
            return 'today'
        elif any(kw in message for kw in ['tomorrow']):
            return 'tomorrow'
        elif any(kw in message for kw in ['weekend']):
            return 'weekend'
        return None
    
    def _extract_weather_metric(self, message: str) -> Optional[str]:
        """Extract specific weather metric requested."""
        if 'temperature' in message or 'hot' in message or 'cold' in message:
            return 'temperature'
        elif 'rain' in message:
            return 'precipitation'
        elif 'wind' in message:
            return 'wind'
        return None
    
    def _extract_event_type(self, message: str) -> Optional[str]:
        """Extract event type."""
        event_types = ['concert', 'festival', 'exhibition', 'show', 'performance', 'sports']
        for event_type in event_types:
            if event_type in message:
                return event_type
        return None
    
    def _extract_date_range(self, message: str) -> Optional[str]:
        """Extract date range for events."""
        if any(kw in message for kw in ['tonight', 'today']):
            return 'today'
        elif any(kw in message for kw in ['this weekend', 'weekend']):
            return 'weekend'
        elif any(kw in message for kw in ['this week']):
            return 'week'
        return None
    
    def _extract_shopping_category(self, message: str) -> Optional[str]:
        """Extract shopping category."""
        categories = ['souvenir', 'clothes', 'jewelry', 'carpet', 'spices', 'antiques']
        for category in categories:
            if category in message:
                return category
        return None
    
    def _extract_shopping_type(self, message: str) -> str:
        """Extract shopping type."""
        if any(kw in message for kw in ['mall', 'shopping center']):
            return 'mall'
        elif any(kw in message for kw in ['bazaar', 'market']):
            return 'traditional'
        elif any(kw in message for kw in ['boutique', 'luxury']):
            return 'boutique'
        return 'general'
    
    # ==================== MINIMAL CONTEXT ====================
    
    def _build_minimal_context(self, message: str, intent: str) -> Dict[str, Any]:
        """Build minimal context when full context building fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'intent': intent,
            'original_query': message,
            'message_lower': message.lower(),
            'sentiment': 'neutral',
            'urgency': 'flexible',
            'minimal_context': True
        }
