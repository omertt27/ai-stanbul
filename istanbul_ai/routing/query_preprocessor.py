"""
Query Preprocessor - Preprocess and enhance queries before routing

This module handles query preprocessing, including text normalization, language detection,
and neural insights extraction.

Week 2 Refactoring: Extracted from main_system.py
"""

import logging
import re
from typing import Dict, Optional, Any
from ..core.models import UserProfile

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Preprocess and enhance user queries before routing"""
    
    def __init__(self):
        """Initialize the query preprocessor"""
        self.turkish_chars = {'ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü'}
        self.turkish_keywords = self._initialize_turkish_keywords()
        self.english_keywords = self._initialize_english_keywords()
    
    def _initialize_turkish_keywords(self) -> set:
        """Initialize Turkish language keywords"""
        return {
            'merhaba', 'selam', 'günaydın', 'iyi günler', 'iyi akşamlar',
            'teşekkür', 'teşekkürler', 'lütfen', 'özür', 'nasıl', 'nerede',
            'ne zaman', 'nedir', 'hangi', 'kaç', 'ne kadar', 'var mı',
            'yok mu', 'açık mı', 'kapalı mı', 'uzak mı', 'yakın mı',
            'restoran', 'müze', 'cami', 'saray', 'çarşı', 'pazar',
            'ulaşım', 'metro', 'otobüs', 'tramvay', 'vapur', 'taksi'
        }
    
    def _initialize_english_keywords(self) -> set:
        """Initialize English language keywords"""
        return {
            'hello', 'hi', 'good', 'morning', 'afternoon', 'evening',
            'thank', 'thanks', 'please', 'sorry', 'how', 'where',
            'when', 'what', 'which', 'many', 'much', 'there',
            'open', 'closed', 'far', 'near', 'restaurant', 'museum',
            'mosque', 'palace', 'bazaar', 'market', 'transport',
            'metro', 'bus', 'tram', 'ferry', 'taxi'
        }
    
    def preprocess_query(
        self,
        message: str,
        user_id: str,
        user_profile: UserProfile,
        neural_processor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Preprocess query with normalization, language detection, and neural insights
        
        Args:
            message: User's input message
            user_id: User identifier
            user_profile: User profile
            neural_processor: Optional neural processor for ML insights
        
        Returns:
            Dictionary with preprocessed query data
        """
        preprocessed = {
            'original_message': message,
            'normalized_message': self.normalize_text(message),
            'language': self.detect_language(message),
            'message_length': len(message),
            'word_count': len(message.split()),
            'has_question': '?' in message,
            'has_coordinates': self._has_gps_coordinates(message)
        }
        
        # Add user context
        preprocessed['user_context'] = self._build_user_context(user_profile)
        
        # Extract neural insights if processor available
        if neural_processor:
            try:
                neural_insights = self.extract_neural_insights(message, neural_processor)
                preprocessed['neural_insights'] = neural_insights
            except Exception as e:
                logger.warning(f"Failed to extract neural insights: {e}")
                preprocessed['neural_insights'] = None
        
        return preprocessed
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace and standardizing format
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Standardize punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def detect_language(self, message: str) -> str:
        """
        Detect language of message (Turkish or English)
        
        Args:
            message: User's input message
        
        Returns:
            Language code ('tr' or 'en')
        """
        message_lower = message.lower()
        
        # Check for Turkish-specific characters
        has_turkish_chars = any(char in message for char in self.turkish_chars)
        
        # Count Turkish vs English keyword matches
        turkish_matches = sum(1 for word in self.turkish_keywords 
                             if word in message_lower)
        english_matches = sum(1 for word in self.english_keywords 
                             if word in message_lower)
        
        # Decision logic
        if has_turkish_chars or turkish_matches > english_matches:
            return 'tr'
        return 'en'
    
    def extract_neural_insights(
        self, 
        message: str, 
        neural_processor: Any
    ) -> Dict[str, Any]:
        """
        Extract neural/ML insights from message
        
        Args:
            message: User's input message
            neural_processor: Neural processor instance
        
        Returns:
            Dictionary of neural insights
        """
        if not neural_processor or not hasattr(neural_processor, 'analyze_query'):
            return {}
        
        try:
            # Call neural processor
            insights = neural_processor.analyze_query(message)
            
            return {
                'sentiment': insights.get('sentiment', 'neutral'),
                'urgency': insights.get('urgency', 'normal'),
                'complexity': insights.get('complexity', 'simple'),
                'keywords': insights.get('keywords', []),
                'entities': insights.get('entities', {}),
                'temporal_context': insights.get('temporal_context'),
                'confidence': insights.get('confidence', 0.5)
            }
        except Exception as e:
            logger.error(f"Neural insights extraction failed: {e}")
            return {}
    
    def _build_user_context(self, user_profile: UserProfile) -> Dict[str, Any]:
        """
        Build user context from profile
        
        Args:
            user_profile: User profile
        
        Returns:
            User context dictionary
        """
        context = {
            'user_id': user_profile.user_id,
            'user_type': user_profile.user_type.value if hasattr(user_profile.user_type, 'value') else str(user_profile.user_type),
            'interests': user_profile.interests if hasattr(user_profile, 'interests') else [],
            'budget_range': user_profile.budget_range if hasattr(user_profile, 'budget_range') else 'moderate'
        }
        
        # Add language preference if available
        if hasattr(user_profile, 'language_preference'):
            context['language_preference'] = user_profile.language_preference
        elif hasattr(user_profile, 'session_context'):
            context['language_preference'] = user_profile.session_context.get('language_preference', 'english')
        
        # Add accessibility needs if available
        if hasattr(user_profile, 'accessibility_needs'):
            context['accessibility_needs'] = user_profile.accessibility_needs
        
        return context
    
    def _has_gps_coordinates(self, message: str) -> bool:
        """
        Check if message contains GPS coordinates
        
        Args:
            message: User's input message
        
        Returns:
            True if GPS coordinates found
        """
        gps_pattern = r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
        return bool(re.search(gps_pattern, message))
    
    def build_intelligent_user_context(
        self,
        message: str,
        neural_insights: Optional[Dict],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Build comprehensive user context for handlers
        
        Args:
            message: User's input message
            neural_insights: Neural insights if available
            user_profile: User profile
        
        Returns:
            Comprehensive user context
        """
        context = {
            'user_type': user_profile.user_type.value if hasattr(user_profile.user_type, 'value') else str(user_profile.user_type),
            'preferences': {
                'budget': user_profile.budget_range if hasattr(user_profile, 'budget_range') else 'moderate',
                'interests': user_profile.interests if hasattr(user_profile, 'interests') else []
            },
            'language': self.detect_language(message)
        }
        
        # Add neural insights if available
        if neural_insights:
            context['sentiment'] = neural_insights.get('sentiment', 'neutral')
            context['urgency'] = neural_insights.get('urgency', 'normal')
            context['temporal_context'] = neural_insights.get('temporal_context')
        
        # Add accessibility needs
        if hasattr(user_profile, 'accessibility_needs'):
            context['accessibility_needs'] = user_profile.accessibility_needs
        
        return context
    
    def is_complex_query(self, message: str, entities: Dict) -> bool:
        """
        Determine if query is complex (multi-intent, detailed requirements)
        
        Args:
            message: User's input message
            entities: Extracted entities
        
        Returns:
            True if query is complex
        """
        # Check for multiple questions
        multiple_questions = message.count('?') > 1
        
        # Check for multiple clauses
        multiple_clauses = message.count(',') >= 2 or message.count('and') >= 2
        
        # Check for many entities
        many_entities = len(entities) > 3
        
        # Check for long message
        long_message = len(message.split()) > 15
        
        # Complex if any two conditions are true
        complexity_score = sum([
            multiple_questions,
            multiple_clauses,
            many_entities,
            long_message
        ])
        
        return complexity_score >= 2
    
    def extract_query_metadata(self, message: str) -> Dict[str, Any]:
        """
        Extract metadata about the query
        
        Args:
            message: User's input message
        
        Returns:
            Query metadata
        """
        return {
            'length': len(message),
            'word_count': len(message.split()),
            'question_marks': message.count('?'),
            'exclamation_marks': message.count('!'),
            'commas': message.count(','),
            'has_numbers': bool(re.search(r'\d', message)),
            'has_urls': bool(re.search(r'http[s]?://', message)),
            'has_email': bool(re.search(r'[\w\.-]+@[\w\.-]+', message)),
            'all_caps': message.isupper(),
            'language': self.detect_language(message)
        }
