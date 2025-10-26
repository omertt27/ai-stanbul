"""
Conversation Handler Service for Istanbul AI System
Handles greetings, thanks, planning queries, farewells, and help requests
Provides context-aware, multi-language responses
Part of Phase 2C: Conversation Enhancement
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import conversation templates
from backend.data.conversation_templates import (
    get_random_greeting,
    get_random_thanks_response,
    get_random_farewell,
    get_help_response,
    get_itinerary_by_days,
    PLANNING_HELP,
)

logger = logging.getLogger(__name__)


class ConversationHandler:
    """
    Handles conversational interactions for the Istanbul AI system.
    
    Features:
    - Multi-language greeting detection and response
    - Thank you acknowledgments
    - Trip planning assistance with itinerary generation
    - Farewell responses
    - Help and clarification requests
    - Context-aware responses
    """
    
    def __init__(self):
        """Initialize the conversation handler"""
        # Pattern definitions for intent detection
        self.greeting_patterns = [
            r'\b(hi|hello|hey|greetings?|good\s+(morning|afternoon|evening|day))\b',
            r'\b(merhaba|selam|günaydın|iyi\s+günler|hoş\s+geldiniz)\b',
            r'^(hi+|hey+|hello+)[\s!.]*$',  # Simple greetings
        ]
        
        self.thanks_patterns = [
            r'\b(thanks?|thank\s+you|thx|tysm|appreciate)\b',
            r'\b(teşekkür|teşekkürler|sağol|sağolun)\b',
            r'^(thanks?|ty|thx)[\s!.]*$',
        ]
        
        self.farewell_patterns = [
            r'\b(bye|goodbye|see\s+you|farewell|take\s+care|cya)\b',
            r'\b(güle\s+güle|hoşça\s+kal|görüşürüz|allaha\s+ısmarladık)\b',
            r'^(bye+|cya|bb)[\s!.]*$',
        ]
        
        self.planning_patterns = [
            r'\b(plan|planning|itinerary|schedule|trip)\b',
            r'\b(how\s+many\s+days?|staying\s+for|\d+\s+days?)\b',
            r'\b(visit\s+plan|travel\s+plan|help\s+me\s+plan)\b',
            r'\b(suggest.*itinerary|recommend.*itinerary|create.*plan)\b',
            r'\b(plan|planlama|program|gezi\s+planı)\b',
        ]
        
        self.help_patterns = [
            r'\b(help|assist|support|guide|confused|don\'t\s+understand)\b',
            r'\b(yardım|destek|anlamadım|anlayamadım)\b',
            r'^(help|what|huh|\?)[\s!.]*$',
            r'\b(what\s+can\s+you\s+do|how\s+do\s+you\s+work)\b',
        ]
        
        logger.info("✅ Conversation Handler initialized successfully")
    
    def detect_intent(self, query: str) -> Tuple[str, float]:
        """
        Detect the conversational intent of the query.
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (intent_type, confidence_score)
            intent_type: 'greeting', 'thanks', 'farewell', 'planning', 'help', or 'none'
            confidence_score: 0.0 to 1.0
        """
        query_lower = query.lower().strip()
        
        # Check each pattern type with priority order
        # Priority: farewell > greeting > thanks > planning > help
        
        # Check farewell (highest priority for clear exits)
        for pattern in self.farewell_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Higher confidence for exact matches
                confidence = 0.95 if len(query.split()) <= 3 else 0.85
                return ('farewell', confidence)
        
        # Check greeting
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                confidence = 0.95 if len(query.split()) <= 4 else 0.85
                return ('greeting', confidence)
        
        # Check thanks
        for pattern in self.thanks_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                confidence = 0.95 if len(query.split()) <= 5 else 0.80
                return ('thanks', confidence)
        
        # Check planning
        for pattern in self.planning_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                confidence = 0.90
                return ('planning', confidence)
        
        # Check help
        for pattern in self.help_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                confidence = 0.85
                return ('help', confidence)
        
        return ('none', 0.0)
    
    def extract_trip_duration(self, query: str) -> Optional[int]:
        """
        Extract trip duration (number of days) from query.
        
        Args:
            query: User's input query
            
        Returns:
            Number of days as integer, or None if not found
        """
        query_lower = query.lower()
        
        # Pattern 1: "X days" or "X day"
        match = re.search(r'(\d+)\s*days?', query_lower)
        if match:
            return int(match.group(1))
        
        # Pattern 2: "staying for X" or "X nights"
        match = re.search(r'(?:staying|here)\s+(?:for\s+)?(\d+)', query_lower)
        if match:
            return int(match.group(1))
        
        # Pattern 3: Number words (one, two, three, etc.)
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'bir': 1, 'iki': 2, 'üç': 3, 'dört': 4, 'beş': 5,
            'altı': 6, 'yedi': 7, 'sekiz': 8, 'dokuz': 9, 'on': 10
        }
        
        for word, number in number_words.items():
            if re.search(r'\b' + word + r'\s+days?', query_lower):
                return number
        
        return None
    
    def detect_language(self, query: str) -> str:
        """
        Detect the language of the query (Turkish or English).
        
        Args:
            query: User's input query
            
        Returns:
            'turkish' or 'english'
        """
        query_lower = query.lower()
        
        # Turkish-specific words and patterns
        turkish_indicators = [
            'merhaba', 'selam', 'teşekkür', 'sağol', 'güle', 'hoşça',
            'günaydın', 'naber', 'nasılsın', 'yardım', 'lütfen',
            'kaç', 'gün', 'gezi', 'plan', 'öner', 'göster'
        ]
        
        # Count Turkish indicators
        turkish_count = sum(1 for indicator in turkish_indicators if indicator in query_lower)
        
        # If we have Turkish indicators, it's Turkish
        if turkish_count > 0:
            return 'turkish'
        
        # Default to English
        return 'english'
    
    def handle_greeting(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Handle greeting queries.
        
        Args:
            query: User's greeting message
            context: Optional conversation context
            
        Returns:
            Response dictionary with message and metadata
        """
        language = self.detect_language(query)
        greeting = get_random_greeting(language)
        
        # Add personalization if context available
        if context and 'user_name' in context:
            if language == 'turkish':
                greeting = f"{greeting}\n\nHoş geldiniz {context['user_name']}! 🌟"
            else:
                greeting = f"{greeting}\n\nWelcome {context['user_name']}! 🌟"
        
        return {
            'message': greeting,
            'intent': 'greeting',
            'language': language,
            'confidence': 0.95
        }
    
    def handle_thanks(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Handle thank you messages.
        
        Args:
            query: User's thanks message
            context: Optional conversation context
            
        Returns:
            Response dictionary with message and metadata
        """
        language = self.detect_language(query)
        thanks_response = get_random_thanks_response(language)
        
        return {
            'message': thanks_response,
            'intent': 'thanks',
            'language': language,
            'confidence': 0.90
        }
    
    def handle_farewell(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Handle farewell messages.
        
        Args:
            query: User's farewell message
            context: Optional conversation context
            
        Returns:
            Response dictionary with message and metadata
        """
        language = self.detect_language(query)
        farewell = get_random_farewell(language)
        
        return {
            'message': farewell,
            'intent': 'farewell',
            'language': language,
            'confidence': 0.95,
            'end_conversation': True  # Signal to end conversation
        }
    
    def handle_help(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Handle help and clarification requests.
        
        Args:
            query: User's help request
            context: Optional conversation context
            
        Returns:
            Response dictionary with message and metadata
        """
        language = self.detect_language(query)
        help_response = get_help_response(language)
        
        return {
            'message': help_response,
            'intent': 'help',
            'language': language,
            'confidence': 0.85
        }
    
    def handle_planning(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Handle trip planning queries with itinerary generation.
        
        Args:
            query: User's planning query
            context: Optional conversation context with preferences
            
        Returns:
            Response dictionary with itinerary and metadata
        """
        language = self.detect_language(query)
        
        # Try to extract trip duration
        days = self.extract_trip_duration(query)
        
        if days:
            # Generate itinerary for specified duration
            itinerary = get_itinerary_by_days(days)
            response = self._format_itinerary_response(itinerary, days, language, context)
        else:
            # Ask for duration if not specified
            planning_help = PLANNING_HELP[language]
            response = f"{planning_help['intro']}\n\n{planning_help['duration_question']}"
        
        return {
            'message': response,
            'intent': 'planning',
            'language': language,
            'confidence': 0.90,
            'trip_duration': days,
            'requires_followup': days is None  # Need more info if no duration
        }
    
    def _format_itinerary_response(self, itinerary: Dict, days: int, 
                                    language: str, context: Optional[Dict] = None) -> str:
        """
        Format itinerary data into a readable response.
        
        Args:
            itinerary: Itinerary dictionary from templates
            days: Number of days
            language: Response language
            context: Optional user context (budget, interests)
            
        Returns:
            Formatted itinerary string
        """
        # Header
        if language == 'turkish':
            header = f"🗓️ **{days} Günlük İstanbul Gezisi**\n"
            header += f"_{itinerary.get('description', '')}_ \n\n"
        else:
            header = f"🗓️ **{itinerary['title']}**\n"
            header += f"_{itinerary.get('description', '')}_ \n\n"
        
        response = header
        
        # For 1-2 day itineraries, show detailed schedule
        if days <= 2 and 'schedule' in itinerary:
            response += self._format_detailed_schedule(itinerary['schedule'], language)
        elif days == 2 and 'day1' in itinerary:
            # 2-day format with separate days
            response += f"**{itinerary['day1']['title']}:**\n"
            response += self._format_detailed_schedule(itinerary['day1']['schedule'], language)
            response += f"\n**{itinerary['day2']['title']}:**\n"
            response += self._format_detailed_schedule(itinerary['day2']['schedule'], language)
        else:
            # For longer trips, show summary
            response += self._format_summary_itinerary(itinerary, language)
        
        # Budget information
        if 'total_cost' in itinerary:
            if language == 'turkish':
                response += "\n\n💰 **Tahmini Bütçe:**\n"
            else:
                response += "\n\n💰 **Estimated Budget:**\n"
            
            # Determine user's budget level from context
            budget_level = 'moderate'  # default
            if context and 'budget_level' in context:
                budget_level = context['budget_level']
            
            for level, cost in itinerary['total_cost'].items():
                marker = "➡️ " if level == budget_level else ""
                level_name = level.capitalize()
                response += f"{marker}**{level_name}:** {cost}\n"
        
        # Additional tips
        if language == 'turkish':
            response += "\n\n💡 **İpucu:** Daha detaylı bilgi için bana sorun!"
        else:
            response += "\n\n💡 **Tip:** Ask me for more details about any attraction or activity!"
        
        return response
    
    def _format_detailed_schedule(self, schedule: List[Dict], language: str) -> str:
        """Format detailed schedule for 1-2 day itineraries"""
        formatted = ""
        
        for item in schedule:
            time = item.get('time', '')
            activity = item.get('activity', '')
            details = item.get('details', '')
            cost = item.get('cost', '')
            tips = item.get('tips', '')
            
            formatted += f"\n**{time}** - {activity}\n"
            if details:
                formatted += f"_{details}_\n"
            if cost:
                formatted += f"💰 {cost}\n"
            if tips:
                formatted += f"💡 {tips}\n"
        
        return formatted
    
    def _format_summary_itinerary(self, itinerary: Dict, language: str) -> str:
        """Format summary for longer itineraries (3+ days)"""
        formatted = ""
        
        if 'summary' in itinerary:
            formatted += f"{itinerary['summary']}\n\n"
        
        if 'highlights' in itinerary:
            if language == 'turkish':
                formatted += "**Öne Çıkanlar:**\n"
            else:
                formatted += "**Highlights:**\n"
            
            for highlight in itinerary['highlights']:
                formatted += f"✨ {highlight}\n"
        
        if 'daily_highlights' in itinerary:
            formatted += "\n"
            for day_highlight in itinerary['daily_highlights']:
                formatted += f"📍 {day_highlight}\n"
        
        return formatted
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Main entry point: detect intent and route to appropriate handler.
        
        Args:
            query: User's input query
            context: Optional conversation context
            
        Returns:
            Response dictionary if conversational intent detected, None otherwise
        """
        # Detect intent
        intent, confidence = self.detect_intent(query)
        
        # If no conversational intent or low confidence, return None
        if intent == 'none' or confidence < 0.70:
            return None
        
        # Route to appropriate handler
        if intent == 'greeting':
            return self.handle_greeting(query, context)
        elif intent == 'thanks':
            return self.handle_thanks(query, context)
        elif intent == 'farewell':
            return self.handle_farewell(query, context)
        elif intent == 'planning':
            return self.handle_planning(query, context)
        elif intent == 'help':
            return self.handle_help(query, context)
        
        return None
    
    # ============ Legacy compatibility methods ============
    # These methods maintain backward compatibility with existing code
    
    def is_conversational_query(self, query: str) -> bool:
        """Check if query is conversational (legacy compatibility)"""
        intent, confidence = self.detect_intent(query)
        return intent != 'none' and confidence >= 0.70
    
    def handle_conversation(self, query: str) -> Optional[str]:
        """Main handler for conversational queries (legacy compatibility)"""
        result = self.process_query(query)
        if result:
            return result['message']
        return None
    
    def detect_greeting(self, query: str) -> Optional[str]:
        """Detect greeting type in query (legacy compatibility)"""
        intent, confidence = self.detect_intent(query)
        if intent == 'greeting' and confidence >= 0.70:
            return 'greeting'
        return None
    
    def detect_thanks(self, query: str) -> bool:
        """Detect thank you in query (legacy compatibility)"""
        intent, confidence = self.detect_intent(query)
        return intent == 'thanks' and confidence >= 0.70
    
    def detect_help_request(self, query: str) -> Optional[str]:
        """Detect help/confused queries (legacy compatibility)"""
        intent, confidence = self.detect_intent(query)
        if intent == 'help' and confidence >= 0.70:
            return 'help'
        return None
    
    def recommend_duration(self, query: str) -> Optional[str]:
        """Generate personalized itinerary based on available time (legacy compatibility)"""
        result = self.process_query(query)
        if result and result['intent'] == 'planning':
            return result['message']
        return None


# ==================== SINGLETON INSTANCE ====================

_conversation_handler_instance = None


def get_conversation_handler() -> ConversationHandler:
    """Get or create singleton conversation handler instance"""
    global _conversation_handler_instance
    
    if _conversation_handler_instance is None:
        _conversation_handler_instance = ConversationHandler()
        logger.info("💬 Conversation Handler instance created")
    
    return _conversation_handler_instance


# ==================== EXPORT ====================

__all__ = [
    'ConversationHandler',
    'get_conversation_handler',
]
