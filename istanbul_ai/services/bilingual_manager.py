"""
Bilingual Manager Service for Istanbul AI
Handles language detection, preference management, and bilingual content delivery

This service ensures English and Turkish language parity across the entire system.
"""

from enum import Enum
from typing import Dict, Optional, List, Any
import logging
import re

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages for the Istanbul AI system"""
    ENGLISH = "en"
    TURKISH = "tr"


class BilingualManager:
    """
    Central bilingual management service
    
    Responsibilities:
    - Language detection from user input
    - User language preference management
    - Bilingual content template management
    - Response formatting in target language
    
    Usage:
        manager = BilingualManager()
        lang = manager.detect_language("Merhaba, nasılsın?")
        response = manager.get_bilingual_response('greeting_morning', lang)
    """
    
    def __init__(self):
        """Initialize the bilingual manager with language patterns and templates"""
        
        # Language detection patterns
        self.language_patterns = {
            Language.TURKISH: {
                'greetings': [
                    'merhaba', 'selam', 'günaydın', 'iyi günler', 
                    'iyi akşamlar', 'tünaydın', 'selamlar'
                ],
                'questions': [
                    'nedir', 'nerede', 'nasıl', 'ne zaman', 'kaç', 
                    'kim', 'niçin', 'niye', 'hangi'
                ],
                'location_suffixes': [
                    'de', 'da', 'den', 'dan', 'e', 'a', 'te', 'ta',
                    'den', 'dan', 'ye', 'ya'
                ],
                'common_words': [
                    'var', 'yok', 'için', 'ile', 'gibi', 'çok', 
                    'güzel', 'iyi', 'kötü', 'ben', 'sen', 'biz'
                ],
                'verbs': [
                    'gitmek', 'giderim', 'geliyorum', 'istiyorum',
                    'önerir', 'göster', 'söyle', 'anlat'
                ]
            },
            Language.ENGLISH: {
                'greetings': [
                    'hello', 'hi', 'hey', 'good morning', 'good afternoon',
                    'good evening', 'howdy', 'greetings'
                ],
                'questions': [
                    'what', 'where', 'how', 'when', 'which', 
                    'who', 'why', 'whose'
                ],
                'articles': ['the', 'a', 'an'],
                'common_words': [
                    'is', 'are', 'was', 'were', 'have', 'has',
                    'do', 'does', 'can', 'could', 'would', 'should'
                ],
                'prepositions': [
                    'in', 'on', 'at', 'to', 'from', 'with',
                    'by', 'for', 'about', 'between'
                ]
            }
        }
        
        # Turkish-specific characters for detection
        self.turkish_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'İ', 'Ç', 'Ğ', 'Ö', 'Ş', 'Ü']
        
        # Load bilingual templates
        self.templates = self._load_templates()
        
        logger.info("✅ BilingualManager initialized with language detection and templates")
    
    def detect_language(
        self, 
        text: str, 
        user_preference: Optional[Language] = None,
        context: Optional[Dict] = None
    ) -> Language:
        """
        Detect language from text with user preference consideration
        
        Uses a scoring system based on:
        - Language-specific keywords
        - Turkish characters
        - Grammar patterns
        - User preference (as tiebreaker)
        
        Args:
            text: Input text to analyze
            user_preference: User's preferred language (optional)
            context: Additional context (optional)
            
        Returns:
            Detected Language enum
            
        Examples:
            >>> manager.detect_language("How do I get to Taksim?")
            Language.ENGLISH
            >>> manager.detect_language("Taksim'e nasıl giderim?")
            Language.TURKISH
        """
        if not text or not text.strip():
            return user_preference or Language.ENGLISH
        
        text_lower = text.lower()
        
        # Initialize scores
        turkish_score = 0
        english_score = 0
        
        # Check Turkish patterns
        for category, words in self.language_patterns[Language.TURKISH].items():
            for word in words:
                if word in text_lower:
                    turkish_score += 1
                    logger.debug(f"Turkish match: '{word}' in category '{category}'")
        
        # Check English patterns
        for category, words in self.language_patterns[Language.ENGLISH].items():
            for word in words:
                if word in text_lower:
                    english_score += 1
                    logger.debug(f"English match: '{word}' in category '{category}'")
        
        # Check for Turkish characters (strong indicator)
        turkish_char_count = sum(1 for char in text.lower() if char in self.turkish_chars)
        if turkish_char_count > 0:
            turkish_score += turkish_char_count * 2  # Weight Turkish chars heavily
            logger.debug(f"Found {turkish_char_count} Turkish characters")
        
        # Check for Turkish suffixes (grammar pattern)
        turkish_suffix_pattern = r"(de|da|den|dan|e|a|te|ta|'de|'da|'e|'a)\b"
        suffix_matches = len(re.findall(turkish_suffix_pattern, text_lower))
        if suffix_matches > 0:
            turkish_score += suffix_matches
            logger.debug(f"Found {suffix_matches} Turkish suffix patterns")
        
        # Decision logic
        if turkish_score > english_score:
            detected = Language.TURKISH
        elif english_score > turkish_score:
            detected = Language.ENGLISH
        else:
            # Tie - use user preference or default to English
            detected = user_preference or Language.ENGLISH
            logger.debug("Score tie - using preference or default")
        
        logger.info(f"🌐 Language detected: {detected.value} (TR:{turkish_score} EN:{english_score})")
        return detected
    
    def get_user_language(self, user_profile: Any) -> Language:
        """
        Get user's preferred language from their profile
        
        Args:
            user_profile: User profile object with language preference
            
        Returns:
            User's preferred Language
        """
        if not user_profile:
            return Language.ENGLISH
        
        # Check direct language preference attribute
        if hasattr(user_profile, 'language_preference'):
            lang = user_profile.language_preference
            if isinstance(lang, Language):
                return lang
            elif lang in ['turkish', 'tr', 'türkçe', 'turkce']:
                return Language.TURKISH
            elif lang in ['english', 'en', 'ingilizce']:
                return Language.ENGLISH
        
        # Check session context
        if hasattr(user_profile, 'session_context') and isinstance(user_profile.session_context, dict):
            lang = user_profile.session_context.get('language_preference')
            if lang in ['turkish', 'tr']:
                return Language.TURKISH
            elif lang in ['english', 'en']:
                return Language.ENGLISH
        
        # Check preferences dict
        if hasattr(user_profile, 'preferences') and isinstance(user_profile.preferences, dict):
            lang = user_profile.preferences.get('language')
            if lang in ['turkish', 'tr']:
                return Language.TURKISH
            elif lang in ['english', 'en']:
                return Language.ENGLISH
        
        return Language.ENGLISH
    
    def set_user_language(self, user_profile: Any, language: Language) -> None:
        """
        Set user's language preference in their profile
        
        Args:
            user_profile: User profile object
            language: Language to set
        """
        if not user_profile:
            return
        
        # Set in session context
        if hasattr(user_profile, 'session_context'):
            if not isinstance(user_profile.session_context, dict):
                user_profile.session_context = {}
            user_profile.session_context['language_preference'] = language.value
        
        # Set as direct attribute
        user_profile.language_preference = language.value
        
        logger.info(f"✅ User language preference set to: {language.value}")
    
    def get_bilingual_response(
        self, 
        key: str, 
        lang: Language, 
        **kwargs
    ) -> str:
        """
        Get response template in specified language
        
        Args:
            key: Template key (e.g., 'greeting_morning')
            lang: Target language
            **kwargs: Template variables for formatting
            
        Returns:
            Formatted response string
            
        Examples:
            >>> manager.get_bilingual_response('greeting_morning', Language.TURKISH)
            'Günaydın! ☀️ Bugün İstanbul'u keşfetmenizde size nasıl yardımcı olabilirim?'
        """
        template = self.templates.get(key, {}).get(lang)
        
        if not template:
            # Fallback to English
            template = self.templates.get(key, {}).get(Language.ENGLISH, key)
            logger.warning(f"Template '{key}' not found for {lang.value}, using English")
        
        # Format with kwargs
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable in '{key}': {e}")
            return template
        except Exception as e:
            logger.error(f"Error formatting template '{key}': {e}")
            return template
    
    def format_response(
        self, 
        response_data: Dict[str, Any], 
        lang: Language
    ) -> Dict[str, Any]:
        """
        Format a response dictionary for the target language
        
        Args:
            response_data: Response data with potentially bilingual content
            lang: Target language
            
        Returns:
            Formatted response data
        """
        formatted = response_data.copy()
        
        # If response has bilingual keys, select the right one
        if 'response' in formatted and isinstance(formatted['response'], dict):
            if lang in formatted['response']:
                formatted['response'] = formatted['response'][lang]
            elif Language.ENGLISH in formatted['response']:
                formatted['response'] = formatted['response'][Language.ENGLISH]
        
        return formatted
    
    def _load_templates(self) -> Dict[str, Dict[Language, str]]:
        """
        Load bilingual response templates
        
        Returns:
            Dictionary of template_key -> {Language -> template_string}
        """
        return {
            # Greetings
            'greeting_morning': {
                Language.ENGLISH: "Good morning! ☀️ How can I help you explore Istanbul today?",
                Language.TURKISH: "Günaydın! ☀️ Bugün İstanbul'u keşfetmenizde size nasıl yardımcı olabilirim?"
            },
            'greeting_afternoon': {
                Language.ENGLISH: "Good afternoon! 🌤️ What would you like to know about Istanbul?",
                Language.TURKISH: "İyi günler! 🌤️ İstanbul hakkında ne öğrenmek istersiniz?"
            },
            'greeting_evening': {
                Language.ENGLISH: "Good evening! 🌆 Looking for evening activities in Istanbul?",
                Language.TURKISH: "İyi akşamlar! 🌆 İstanbul'da akşam aktiviteleri mi arıyorsunuz?"
            },
            'greeting_general': {
                Language.ENGLISH: "Hello! 👋 I'm your Istanbul travel assistant. How can I help you today?",
                Language.TURKISH: "Merhaba! 👋 Ben sizin İstanbul seyahat asistanınızım. Bugün size nasıl yardımcı olabilirim?"
            },
            
            # Section Headers
            'transportation_header': {
                Language.ENGLISH: "🚇 **Istanbul Transportation Guide**",
                Language.TURKISH: "🚇 **İstanbul Ulaşım Rehberi**"
            },
            'attraction_header': {
                Language.ENGLISH: "🏛️ **Istanbul Attractions**",
                Language.TURKISH: "🏛️ **İstanbul Gezilecek Yerler**"
            },
            'restaurant_header': {
                Language.ENGLISH: "🍽️ **Restaurant Recommendations**",
                Language.TURKISH: "🍽️ **Restoran Önerileri**"
            },
            'weather_header': {
                Language.ENGLISH: "🌤️ **Istanbul Weather**",
                Language.TURKISH: "🌤️ **İstanbul Hava Durumu**"
            },
            'event_header': {
                Language.ENGLISH: "🎭 **Events in Istanbul**",
                Language.TURKISH: "🎭 **İstanbul'daki Etkinlikler**"
            },
            'neighborhood_header': {
                Language.ENGLISH: "🏘️ **Istanbul Neighborhoods**",
                Language.TURKISH: "🏘️ **İstanbul Semtleri**"
            },
            
            # Common Phrases
            'no_results': {
                Language.ENGLISH: "I couldn't find exactly what you're looking for. Could you provide more details?",
                Language.TURKISH: "Aradığınızı tam olarak bulamadım. Daha fazla detay verebilir misiniz?"
            },
            'error_message': {
                Language.ENGLISH: "Sorry, I encountered an error. Please try again.",
                Language.TURKISH: "Üzgünüm, bir hatayla karşılaştım. Lütfen tekrar deneyin."
            },
            'clarification_needed': {
                Language.ENGLISH: "Could you clarify what you're looking for? For example, a specific location or type?",
                Language.TURKISH: "Ne aradığınızı açıklayabilir misiniz? Örneğin, belirli bir yer veya tür?"
            },
            'thank_you': {
                Language.ENGLISH: "You're welcome! 😊 Let me know if you need anything else about Istanbul!",
                Language.TURKISH: "Rica ederim! 😊 İstanbul hakkında başka bir şeye ihtiyacınız olursa söyleyin!"
            },
            'goodbye': {
                Language.ENGLISH: "👋 Goodbye! Have a wonderful time in Istanbul! Feel free to ask me anything about the city anytime.",
                Language.TURKISH: "👋 Güle güle! İstanbul'da harika vakit geçirin! Şehir hakkında istediğiniz zaman bana sorabilirsiniz."
            },
            
            # Direction indicators
            'direction_from': {
                Language.ENGLISH: "from",
                Language.TURKISH: "den"
            },
            'direction_to': {
                Language.ENGLISH: "to",
                Language.TURKISH: "e"
            },
            'direction_near': {
                Language.ENGLISH: "near",
                Language.TURKISH: "yakınında"
            },
            
            # Time indicators
            'time_morning': {
                Language.ENGLISH: "morning",
                Language.TURKISH: "sabah"
            },
            'time_afternoon': {
                Language.ENGLISH: "afternoon",
                Language.TURKISH: "öğleden sonra"
            },
            'time_evening': {
                Language.ENGLISH: "evening",
                Language.TURKISH: "akşam"
            },
            'time_night': {
                Language.ENGLISH: "night",
                Language.TURKISH: "gece"
            },
            
            # Status messages
            'status_open': {
                Language.ENGLISH: "Open",
                Language.TURKISH: "Açık"
            },
            'status_closed': {
                Language.ENGLISH: "Closed",
                Language.TURKISH: "Kapalı"
            },
            'status_available': {
                Language.ENGLISH: "Available",
                Language.TURKISH: "Mevcut"
            },
            'status_unavailable': {
                Language.ENGLISH: "Unavailable",
                Language.TURKISH: "Müsait değil"
            },
            
            # Weather-specific templates
            'weather.error': {
                Language.ENGLISH: "I'm having trouble getting weather information. Let me know what you'd like to do and I can suggest activities!",
                Language.TURKISH: "Hava durumu bilgisi almakta sorun yaşıyorum. Ne yapmak istediğinizi söyleyin, size aktivite önerebilirim!"
            },
            'weather.no_activities': {
                Language.ENGLISH: "Given the current weather, I'm having trouble finding suitable activities. Would you like indoor or outdoor suggestions?",
                Language.TURKISH: "Mevcut hava durumuna göre uygun aktiviteler bulmakta zorlanıyorum. İç mekan veya dış mekan önerileri ister misiniz?"
            },
            'weather.current_header': {
                Language.ENGLISH: "🌤️ **Current Weather in Istanbul:**",
                Language.TURKISH: "🌤️ **İstanbul'da Güncel Hava Durumu:**"
            },
            'weather.temperature': {
                Language.ENGLISH: "Temperature",
                Language.TURKISH: "Sıcaklık"
            },
            'weather.more_activities': {
                Language.ENGLISH: "📋 **More weather-appropriate activities:**",
                Language.TURKISH: "📋 **Hava durumuna uygun daha fazla aktivite:**"
            },
            
            # Weather condition intros
            'weather.intro.rainy': {
                Language.ENGLISH: "☔ Perfect indoor activities for rainy weather:",
                Language.TURKISH: "☔ Yağmurlu hava için mükemmel iç mekan aktiviteleri:"
            },
            'weather.intro.hot': {
                Language.ENGLISH: "☀️ Beat the heat with these activities:",
                Language.TURKISH: "☀️ Sıcaktan kaçmak için bu aktiviteler:"
            },
            'weather.intro.clear': {
                Language.ENGLISH: "✨ Great weather! Here are the best outdoor options:",
                Language.TURKISH: "✨ Harika hava! İşte en iyi açık hava seçenekleri:"
            },
            'weather.intro.general': {
                Language.ENGLISH: "🎯 Here are the best activities for current conditions:",
                Language.TURKISH: "🎯 Mevcut koşullar için en iyi aktiviteler:"
            },
            
            # Weather comfort levels
            'weather.comfort.excellent': {
                Language.ENGLISH: "Weather comfort: Excellent",
                Language.TURKISH: "Hava konforu: Mükemmel"
            },
            'weather.comfort.good': {
                Language.ENGLISH: "Weather comfort: Good",
                Language.TURKISH: "Hava konforu: İyi"
            },
            
            # Weather forecast
            'weather.forecast.tomorrow': {
                Language.ENGLISH: "Tomorrow's forecast",
                Language.TURKISH: "Yarının tahmini"
            },
            'weather.forecast.outdoor_tip': {
                Language.ENGLISH: "Consider outdoor activities tomorrow!",
                Language.TURKISH: "Yarın açık hava aktivitelerini düşünün!"
            },
            
            # Weather tips
            'weather.tip.rainy': {
                Language.ENGLISH: "Weather tip: Bring an umbrella and wear waterproof shoes!",
                Language.TURKISH: "Hava ipucu: Şemsiye getirin ve su geçirmez ayakkabı giyin!"
            },
            'weather.tip.hot': {
                Language.ENGLISH: "Weather tip: Stay hydrated and use sunscreen!",
                Language.TURKISH: "Hava ipucu: Bol sıvı tüketin ve güneş kremi kullanın!"
            },
            'weather.tip.cold': {
                Language.ENGLISH: "Weather tip: Dress warmly in layers!",
                Language.TURKISH: "Hava ipucu: Katmanlı ve sıcak giyinin!"
            },
            
            # Common labels
            'common.match': {
                Language.ENGLISH: "Match",
                Language.TURKISH: "Eşleşme"
            },
            'common.duration': {
                Language.ENGLISH: "Duration",
                Language.TURKISH: "Süre"
            },
            'common.cost': {
                Language.ENGLISH: "Cost",
                Language.TURKISH: "Ücret"
            },
            
            # Route Planning
            'route.header': {
                Language.ENGLISH: "🗺️ **Route from {start} to {end}**",
                Language.TURKISH: "🗺️ **{start} - {end} Arası Güzergah**"
            },
            'route.recommended': {
                Language.ENGLISH: "🌟 **Recommended Route: {name}**",
                Language.TURKISH: "🌟 **Önerilen Güzergah: {name}**"
            },
            'route.match_optimized': {
                Language.ENGLISH: "(Match: {score}%, Optimized for: {goal})",
                Language.TURKISH: "(Eşleşme: %{score}, Optimize edildi: {goal})"
            },
            'route.duration': {
                Language.ENGLISH: "⏱️ Duration: {minutes} minutes",
                Language.TURKISH: "⏱️ Süre: {minutes} dakika"
            },
            'route.cost': {
                Language.ENGLISH: "💰 Cost: {cost} TL",
                Language.TURKISH: "💰 Ücret: {cost} TL"
            },
            'route.transfers': {
                Language.ENGLISH: "🔄 Transfers: {count}",
                Language.TURKISH: "🔄 Aktarma: {count}"
            },
            'route.directions': {
                Language.ENGLISH: "**Directions:**",
                Language.TURKISH: "**Yol Tarifi:**"
            },
            'route.alternatives': {
                Language.ENGLISH: "🔀 **Alternative Routes:**",
                Language.TURKISH: "🔀 **Alternatif Güzergahlar:**"
            },
            'route.alternative_item': {
                Language.ENGLISH: "{name}: {duration} min, {cost} TL, {transfers} transfer(s)",
                Language.TURKISH: "{name}: {duration} dk, {cost} TL, {transfers} aktarma"
            },
            'route.departure': {
                Language.ENGLISH: "🕐 Departure: {dep_time} | Arrival: ~{arr_time}",
                Language.TURKISH: "🕐 Kalkış: {dep_time} | Varış: ~{arr_time}"
            },
            'route.qualities': {
                Language.ENGLISH: "✨ Route qualities: {qualities}",
                Language.TURKISH: "✨ Güzergah özellikleri: {qualities}"
            },
            
            # Route optimization goals
            'route.goal.fastest': {
                Language.ENGLISH: "fastest",
                Language.TURKISH: "en hızlı"
            },
            'route.goal.cheapest': {
                Language.ENGLISH: "cheapest",
                Language.TURKISH: "en ucuz"
            },
            'route.goal.scenic': {
                Language.ENGLISH: "scenic",
                Language.TURKISH: "manzaralı"
            },
            'route.goal.comfortable': {
                Language.ENGLISH: "comfortable",
                Language.TURKISH: "konforlu"
            },
            
            # Route qualities
            'route.quality.scenic': {
                Language.ENGLISH: "Scenic views",
                Language.TURKISH: "Manzaralı"
            },
            'route.quality.comfortable': {
                Language.ENGLISH: "Comfortable",
                Language.TURKISH: "Konforlu"
            },
            'route.quality.less_crowded': {
                Language.ENGLISH: "Less crowded",
                Language.TURKISH: "Az kalabalık"
            },
            'route.quality.weather_protected': {
                Language.ENGLISH: "Weather protected",
                Language.TURKISH: "Hava korumalı"
            },
            
            # Route tips
            'route.tip.istanbul_kart': {
                Language.ENGLISH: "💡 Using Istanbul Kart saves ~30% on all public transport",
                Language.TURKISH: "💡 İstanbulKart kullanımı tüm toplu taşımada ~%30 tasarruf sağlar"
            },
            'route.tip.crowded': {
                Language.ENGLISH: "⏰ Tip: This route can be crowded during rush hours (8-9 AM, 5-7 PM)",
                Language.TURKISH: "⏰ İpucu: Bu güzergah yoğun saatlerde (08:00-09:00, 17:00-19:00) kalabalık olabilir"
            },
            'route.tip.rain_umbrella': {
                Language.ENGLISH: "☔ Weather alert: Bring an umbrella, parts of this route are outdoors",
                Language.TURKISH: "☔ Hava uyarısı: Şemsiye getirin, güzergahın bazı bölümleri açık havada"
            },
            'route.tip.ferry_views': {
                Language.ENGLISH: "⛴️ Ferry tip: Amazing Bosphorus views! Arrive 10 min early for good seats",
                Language.TURKISH: "⛴️ Vapur ipucu: Muhteşem Boğaz manzarası! İyi yer için 10 dk önce gelin"
            },
            
            # Error messages
            'route.error.no_locations': {
                Language.ENGLISH: "I need to know your start and end locations to plan a route. Where are you going?",
                Language.TURKISH: "Güzergah planlamak için başlangıç ve varış noktalarını bilmem gerekiyor. Nereye gidiyorsunuz?"
            },
            'route.error.no_suitable_route': {
                Language.ENGLISH: "I couldn't find a suitable route from {start} to {end} with your requirements. Would you like to adjust your preferences?",
                Language.TURKISH: "{start} - {end} arası isteklerinize uygun güzergah bulamadım. Tercihlerinizi değiştirmek ister misiniz?"
            },
            'route.error.planning_error': {
                Language.ENGLISH: "I'm having trouble planning that route. Could you provide more details about your start and end locations?",
                Language.TURKISH: "Bu güzergahı planlarken sorun yaşıyorum. Başlangıç ve varış noktaları hakkında daha fazla detay verebilir misiniz?"
            }
        }
    
    def get_template_keys(self) -> List[str]:
        """
        Get all available template keys
        
        Returns:
            List of template keys
        """
        return list(self.templates.keys())
    
    def add_template(self, key: str, english: str, turkish: str) -> None:
        """
        Add a new bilingual template
        
        Args:
            key: Template key
            english: English template
            turkish: Turkish template
        """
        self.templates[key] = {
            Language.ENGLISH: english,
            Language.TURKISH: turkish
        }
        logger.info(f"✅ Added new bilingual template: '{key}'")
