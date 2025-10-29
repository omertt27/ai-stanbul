"""
Base Handler - Abstract base class for all response handlers

This module defines the interface that all specialized handlers must implement.
It provides common functionality for language handling and response formatting.

Week 5-6 Refactoring: Extracted from main_system.py
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """
    Abstract base class for all response handlers
    
    All specialized handlers (Restaurant, Attraction, Transportation, etc.)
    must inherit from this class and implement the required methods.
    """
    
    def __init__(self):
        """Initialize the handler"""
        self.handler_name = self.__class__.__name__
        logger.info(f"✅ {self.handler_name} initialized")
    
    @abstractmethod
    def can_handle(self, intent: str, entities: Dict, context: Any) -> bool:
        """
        Determine if this handler can handle the query
        
        Args:
            intent: Classified intent (e.g., 'restaurant', 'attraction')
            entities: Extracted entities from the message
            context: Conversation context
        
        Returns:
            True if this handler can process the query, False otherwise
        """
        pass
    
    @abstractmethod
    def handle(
        self, 
        message: str, 
        intent: str, 
        entities: Dict,
        user_profile: Any,
        context: Any,
        **kwargs
    ) -> str:
        """
        Handle the query and generate a response
        
        Args:
            message: User's original message
            intent: Classified intent
            entities: Extracted entities
            user_profile: User profile with preferences
            context: Conversation context
            **kwargs: Additional handler-specific parameters
        
        Returns:
            Response string in the appropriate language
        """
        pass
    
    def _ensure_language(self, response: str, user_profile: Any, message: str = "") -> str:
        """
        Ensure response is in user's preferred language
        
        This method checks if the response matches the user's preferred language
        and provides a fallback mechanism if language detection/translation is needed.
        
        Args:
            response: Generated response
            user_profile: User profile with language preference
            message: Original user message (for language detection)
        
        Returns:
            Response in the appropriate language
        """
        try:
            # Get preferred language from profile or detect from message
            preferred_lang = getattr(user_profile, 'preferred_language', 'en')
            
            # If no preference, detect from message
            if not preferred_lang or preferred_lang == 'auto':
                preferred_lang = self._detect_language(message)
            
            # Check if response is already in correct language
            response_lang = self._detect_language(response)
            
            if response_lang == preferred_lang:
                return response
            
            # If languages don't match, return as-is
            # (Translation can be added later if needed)
            return response
            
        except Exception as e:
            logger.warning(f"Language handling error in {self.handler_name}: {e}")
            return response
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
        
        Returns:
            Language code ('en' or 'tr')
        """
        if not text:
            return 'en'
        
        # Simple Turkish character detection
        turkish_chars = {'ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü'}
        
        if any(char in text for char in turkish_chars):
            return 'tr'
        
        # Check for Turkish keywords
        turkish_keywords = {
            'merhaba', 'selam', 'günaydın', 'teşekkür', 'lütfen',
            'nerede', 'nasıl', 'ne', 'var', 'yok', 'için'
        }
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in turkish_keywords):
            return 'tr'
        
        return 'en'
    
    def _format_list_response(self, items: list, language: str = 'en') -> str:
        """
        Format a list of items into a readable response
        
        Args:
            items: List of items to format
            language: Language for formatting
        
        Returns:
            Formatted string
        """
        if not items:
            return "No items found." if language == 'en' else "Öğe bulunamadı."
        
        if len(items) == 1:
            return str(items[0])
        
        # Format as numbered list
        formatted = []
        for i, item in enumerate(items, 1):
            formatted.append(f"{i}. {item}")
        
        return "\n".join(formatted)
    
    def _truncate_response(self, response: str, max_length: int = 2000) -> str:
        """
        Truncate response if too long
        
        Args:
            response: Response text
            max_length: Maximum allowed length
        
        Returns:
            Truncated response
        """
        if len(response) <= max_length:
            return response
        
        # Truncate at last complete sentence
        truncated = response[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.8:  # If we found a sentence end near the limit
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def get_handler_info(self) -> Dict[str, Any]:
        """
        Get information about this handler
        
        Returns:
            Dictionary with handler metadata
        """
        return {
            'name': self.handler_name,
            'type': self.__class__.__name__,
            'description': self.__doc__ or 'No description available'
        }
    
    def __repr__(self) -> str:
        """String representation of handler"""
        return f"<{self.handler_name}>"
