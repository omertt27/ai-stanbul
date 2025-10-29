"""
Language Handler - Manage bilingual support (English/Turkish)

This module handles language detection, switching, and bilingual response generation
for the Istanbul AI system.

Week 7-8 Refactoring: Extracted from main_system.py
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class LanguageHandler:
    """
    Handles language detection and bilingual support
    
    Supports English and Turkish with automatic detection based on:
    - User profile preferences
    - Session context
    - Message content analysis
    - Turkish character detection
    - Keyword matching
    """
    
    def __init__(self):
        """Initialize language handler with keyword dictionaries"""
        self.turkish_chars = {'ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü'}
        self.turkish_keywords = self._initialize_turkish_keywords()
        self.english_keywords = self._initialize_english_keywords()
        self.bilingual_templates = self._initialize_bilingual_templates()
        
        logger.info("✅ LanguageHandler initialized")
    
    def _initialize_turkish_keywords(self) -> Dict[str, List[str]]:
        """Initialize Turkish keyword dictionaries by category"""
        return {
            'greetings': ['merhaba', 'selam', 'günaydın', 'iyi günler', 'iyi akşamlar'],
            'thanks': ['teşekkür', 'teşekkürler', 'sağol', 'sağolun', 'minnettarım'],
            'goodbye': ['hoşça kal', 'hoşçakal', 'güle güle', 'görüşürüz', 'bay'],
            'weather': ['hava', 'hava durumu', 'sıcaklık', 'yağmur', 'güneşli', 'soğuk', 'sıcak', 'derece'],
            'how_are_you': ['nasılsın', 'nasılsınız', 'ne haber', 'naber'],
            'casual': ['güzel', 'harika', 'mükemmel', 'süper', 'tamam', 'peki'],
            'questions': ['nerede', 'nasıl', 'ne', 'hangi', 'kim', 'ne zaman', 'neden']
        }
    
    def _initialize_english_keywords(self) -> Dict[str, List[str]]:
        """Initialize English keyword dictionaries by category"""
        return {
            'greetings': ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'thanks': ['thank', 'thanks', 'appreciate'],
            'goodbye': ['bye', 'goodbye', 'see you', 'farewell'],
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'cold', 'hot', 'forecast'],
            'how_are_you': ['how are you', 'how do you do', 'how\'s it going'],
            'casual': ['nice', 'cool', 'awesome', 'great', 'perfect', 'ok', 'okay'],
            'questions': ['where', 'how', 'what', 'which', 'who', 'when', 'why']
        }
    
    def _initialize_bilingual_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize bilingual response templates"""
        return {
            'greeting_morning': {
                'tr': "🌅 Günaydın! İstanbul'u keşfetmek için harika bir gün! Bugün size nasıl yardımcı olabilirim?",
                'en': "🌅 Good morning! What a beautiful day to explore Istanbul! How can I help you discover something amazing today?"
            },
            'greeting_afternoon': {
                'tr': "☀️ İyi günler! İstanbul'u keşfetmek için mükemmel bir zaman! Bugün neyi keşfetmek istersiniz?",
                'en': "☀️ Good afternoon! Perfect time to explore Istanbul! What would you like to discover today?"
            },
            'greeting_evening': {
                'tr': "🌆 İyi akşamlar! İstanbul'un büyüleyici akşam atmosferi sizi bekliyor! Bu akşam size nasıl yardımcı olabilirim?",
                'en': "🌆 Good evening! Istanbul's evening magic awaits! How can I help you experience the city tonight?"
            },
            'thanks': {
                'tr': "🙏 Rica ederim! İstanbul'un en güzel yerlerini keşfetmenize yardımcı olmak için buradayım. Başka bilmek istediğiniz bir şey var mı?",
                'en': "🙏 You're very welcome! I'm here to help you discover the best of Istanbul. Anything else you'd like to know?"
            },
            'goodbye': {
                'tr': "👋 Güle güle! İstanbul'da harika vakit geçirin! İstediğiniz zaman soru sorabilirsiniz. İyi günler! 🌟",
                'en': "👋 Güle güle! (Goodbye in Turkish) Have a wonderful time in Istanbul! Feel free to ask me anything anytime! 🌟"
            },
            'how_are_you': {
                'tr': "😊 Ben harikayım, teşekkür ederim! İstanbul hakkında sizinle konuşmayı seviyorum. Size bugün nasıl yardımcı olabilirim?",
                'en': "😊 I'm doing great, thank you for asking! I love talking about Istanbul with visitors. How can I help you today?"
            },
            'weather': {
                'tr': "🌤️ İstanbul'da hava şu anda çok güzel! Havanın tadını çıkararak şehri keşfetmek için harika bir zaman.",
                'en': "🌤️ The weather in Istanbul is lovely right now! It's a great time to explore the city."
            },
            'casual': {
                'tr': "😊 Harika! Size daha fazla yardımcı olabilmem için ne öğrenmek istersiniz?",
                'en': "😊 Great! What would you like to learn more about so I can better assist you?"
            },
            'no_results': {
                'tr': "😔 Üzgünüm, bu kriterlere uygun sonuç bulamadım. Farklı bir şey aramayı deneyelim mi?",
                'en': "😔 I'm sorry, I couldn't find results matching those criteria. Shall we try something different?"
            },
            'error': {
                'tr': "😔 Üzgünüm, bir sorun oluştu. Lütfen tekrar deneyin veya farklı bir soru sorun.",
                'en': "😔 I'm sorry, something went wrong. Please try again or ask a different question."
            }
        }
    
    def detect_language(self, message: str, user_profile: Optional[Any] = None) -> str:
        """
        Detect language from message and user profile
        
        Priority:
        1. Session context language preference
        2. User profile language preference
        3. Message content analysis (Turkish chars)
        4. Message content analysis (keywords)
        5. Default to English
        
        Args:
            message: User's message
            user_profile: User profile with language preferences
        
        Returns:
            Language code ('en' or 'tr')
        """
        if not message:
            return 'en'
        
        # Handle Turkish character lowercasing properly
        message_lower = message.replace('İ', 'i').replace('I', 'ı').lower()
        
        # Check user profile preferences
        if user_profile:
            # Handle dict-type user profile
            if isinstance(user_profile, dict):
                # Check language key directly
                lang = user_profile.get('language', '').lower()
                if lang in ['turkish', 'tr', 'türkçe']:
                    return 'tr'
                elif lang in ['english', 'en']:
                    return 'en'
                
                # Check language_preference key
                lang = user_profile.get('language_preference', '').lower()
                if lang in ['turkish', 'tr', 'türkçe']:
                    return 'tr'
                elif lang in ['english', 'en']:
                    return 'en'
            else:
                # Handle object-type user profile
                # Check session context first
                if hasattr(user_profile, 'session_context'):
                    lang = user_profile.session_context.get('language_preference', '').lower()
                    if lang in ['turkish', 'tr', 'türkçe']:
                        return 'tr'
                    elif lang in ['english', 'en']:
                        return 'en'
                
                # Check direct attribute
                if hasattr(user_profile, 'language_preference'):
                    lang = getattr(user_profile, 'language_preference', '').lower()
                    if lang in ['turkish', 'tr', 'türkçe']:
                        return 'tr'
                    elif lang in ['english', 'en']:
                        return 'en'
        
        # Analyze message content
        # 1. Check for Turkish characters
        if any(char in message for char in self.turkish_chars):
            return 'tr'
        
        # 2. Count Turkish vs English keywords
        turkish_count = sum(
            1 for keywords in self.turkish_keywords.values()
            for keyword in keywords
            if keyword in message_lower
        )
        
        english_count = sum(
            1 for keywords in self.english_keywords.values()
            for keyword in keywords
            if keyword in message_lower
        )
        
        if turkish_count > english_count:
            return 'tr'
        elif english_count > 0:
            return 'en'
        
        # Default to English
        return 'en'
    
    def ensure_correct_language(
        self, 
        response: str, 
        user_profile: Optional[Any] = None,
        message: str = ""
    ) -> str:
        """
        Ensure response matches user's preferred language
        
        Args:
            response: Generated response
            user_profile: User profile
            message: Original message (for language detection)
        
        Returns:
            Response in correct language
        """
        # Detect user's preferred language
        preferred_lang = self.detect_language(message, user_profile)
        
        # Detect response language
        response_lang = self.detect_language(response)
        
        # If they match, return as-is
        if response_lang == preferred_lang:
            return response
        
        # If mismatch, return as-is for now
        # (Translation can be added in future if needed)
        return response
    
    def get_bilingual_response(
        self, 
        template_key: str, 
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Get a bilingual response template
        
        Args:
            template_key: Key for the template
            language: Language code ('en' or 'tr'), auto-detect if None
            **kwargs: Format arguments for template
        
        Returns:
            Formatted response string
        """
        # Default to English if language not specified
        if language is None:
            language = 'en'
        
        # Normalize language code
        if language.lower() in ['turkish', 'türkçe']:
            language = 'tr'
        elif language.lower() in ['english']:
            language = 'en'
        
        # Get template
        template = self.bilingual_templates.get(template_key, {})
        response = template.get(language, template.get('en', ''))
        
        # Format if needed
        if kwargs and response:
            try:
                response = response.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing format key in template: {e}")
        
        return response
    
    def get_greeting_response(self, user_profile: Optional[Any] = None, message: str = "") -> str:
        """
        Get appropriate greeting based on time of day and language
        
        Args:
            user_profile: User profile
            message: User's message
        
        Returns:
            Greeting response
        """
        language = self.detect_language(message, user_profile)
        current_hour = datetime.now().hour
        
        if current_hour < 12:
            return self.get_bilingual_response('greeting_morning', language)
        elif current_hour < 17:
            return self.get_bilingual_response('greeting_afternoon', language)
        else:
            return self.get_bilingual_response('greeting_evening', language)
    
    def is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        # Handle Turkish character lowercasing properly
        message_lower = message.replace('İ', 'i').replace('I', 'ı').lower()
        all_greetings = (
            self.turkish_keywords['greetings'] + 
            self.english_keywords['greetings']
        )
        return any(greeting in message_lower for greeting in all_greetings)
    
    def is_thanks(self, message: str) -> bool:
        """Check if message is a thank you"""
        # Handle Turkish character lowercasing properly
        message_lower = message.replace('İ', 'i').replace('I', 'ı').lower()
        all_thanks = (
            self.turkish_keywords['thanks'] + 
            self.english_keywords['thanks']
        )
        return any(thanks in message_lower for thanks in all_thanks)
    
    def is_goodbye(self, message: str) -> bool:
        """Check if message is a goodbye"""
        # Handle Turkish character lowercasing properly
        message_lower = message.replace('İ', 'i').replace('I', 'ı').lower()
        all_goodbye = (
            self.turkish_keywords['goodbye'] + 
            self.english_keywords['goodbye']
        )
        return any(bye in message_lower for bye in all_goodbye)
    
    def get_language_name(self, code: str) -> str:
        """
        Get full language name from code
        
        Args:
            code: Language code ('en' or 'tr')
        
        Returns:
            Full language name
        """
        return 'Turkish' if code == 'tr' else 'English'
