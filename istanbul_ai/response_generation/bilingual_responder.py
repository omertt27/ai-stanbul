"""
Bilingual Responder - Provide fallback and emergency bilingual responses

This module provides fallback responses for edge cases, errors, and situations
where the main response generation fails. All responses are bilingual (English/Turkish).

Week 7-8 Refactoring: Extracted from main_system.py
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class BilingualResponder:
    """
    Provides fallback bilingual responses for edge cases
    
    Handles:
    - Intent-based fallback responses
    - Emergency error responses
    - Clarification requests
    - No results responses
    - Generic fallback messages
    """
    
    def __init__(self):
        """Initialize bilingual responder with response templates"""
        self.fallback_templates = self._initialize_fallback_templates()
        self.emergency_templates = self._initialize_emergency_templates()
        self.clarification_templates = self._initialize_clarification_templates()
        self.no_results_templates = self._initialize_no_results_templates()
        self.special_case_templates = self._initialize_special_case_templates()
        
        logger.info("✅ BilingualResponder initialized")
    
    def _initialize_fallback_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize fallback response templates by intent"""
        return {
            'restaurant': {
                'tr': "🍽️ İstanbul'da harika restoranlar var! Size yardımcı olabilmem için lütfen daha fazla bilgi verin (bütçe, mutfak türü, konum vb.).",
                'en': "🍽️ Istanbul has amazing restaurants! Please provide more details so I can help you better (budget, cuisine type, location, etc.)."
            },
            'attraction': {
                'tr': "🏛️ İstanbul'da görülecek çok yer var! Hangi tür yerleri görmek istersiniz? (müzeler, camiler, saraylar, pazarlar vb.)",
                'en': "🏛️ There's so much to see in Istanbul! What type of attractions interest you? (museums, mosques, palaces, markets, etc.)"
            },
            'transportation': {
                'tr': "🚇 İstanbul'da ulaşım çok kolay! Nereye gitmek istiyorsunuz? Size en iyi rotayı gösterebilirim.",
                'en': "🚇 Getting around Istanbul is easy! Where would you like to go? I can show you the best route."
            },
            'hotel': {
                'tr': "🏨 İstanbul'da kalacak harika yerler var! Bütçeniz ve hangi bölgede kalmak istediğiniz hakkında bilgi verebilir misiniz?",
                'en': "🏨 Istanbul has great places to stay! Could you tell me about your budget and which area you'd prefer?"
            },
            'weather': {
                'tr': "🌤️ İstanbul'un havası mevsime göre değişir. Hangi tarihleri planlıyorsunuz?",
                'en': "🌤️ Istanbul's weather varies by season. What dates are you planning?"
            },
            'shopping': {
                'tr': "🛍️ İstanbul alışveriş cenneti! Ne tür şeyler almak istiyorsunuz? (geleneksel el sanatları, moda, elektronik vb.)",
                'en': "🛍️ Istanbul is a shopping paradise! What kind of things are you looking for? (traditional crafts, fashion, electronics, etc.)"
            },
            'general': {
                'tr': "😊 İstanbul hakkında size nasıl yardımcı olabilirim? Restoranlar, gezilecek yerler, ulaşım veya başka bir konu hakkında soru sorabilirsiniz.",
                'en': "😊 How can I help you with Istanbul? You can ask about restaurants, attractions, transportation, or anything else!"
            }
        }
    
    def _initialize_emergency_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize emergency error response templates"""
        return {
            'system_error': {
                'tr': "⚠️ Üzgünüm, teknik bir sorun yaşadım. Lütfen sorunuzu tekrar sormayı deneyin.",
                'en': "⚠️ I'm sorry, I encountered a technical issue. Please try asking your question again."
            },
            'timeout': {
                'tr': "⏱️ Üzgünüm, işlem çok uzun sürdü. Lütfen daha basit bir soru sormayı deneyin.",
                'en': "⏱️ Sorry, that took too long. Please try asking a simpler question."
            },
            'data_error': {
                'tr': "📊 Üzgünüm, verilere erişirken bir sorun oluştu. Lütfen daha sonra tekrar deneyin.",
                'en': "📊 Sorry, there was an issue accessing the data. Please try again later."
            },
            'invalid_request': {
                'tr': "❌ Sorunuzu anlayamadım. Lütfen daha açık bir şekilde sormayı deneyin.",
                'en': "❌ I couldn't understand your question. Please try asking more clearly."
            },
            'rate_limit': {
                'tr': "🚦 Çok fazla istek aldım. Lütfen birkaç saniye bekleyip tekrar deneyin.",
                'en': "🚦 Too many requests. Please wait a few seconds and try again."
            }
        }
    
    def _initialize_clarification_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize clarification request templates"""
        return {
            'location': {
                'tr': "📍 Size daha iyi yardımcı olabilmem için lütfen konum belirtin (örn: Sultanahmet, Taksim, Kadıköy).",
                'en': "📍 Please specify a location so I can help you better (e.g., Sultanahmet, Taksim, Kadıköy)."
            },
            'budget': {
                'tr': "💰 Bütçeniz hakkında bilgi verebilir misiniz? Bu bana daha uygun öneriler sunmamı sağlar.",
                'en': "💰 Could you tell me about your budget? This helps me provide more suitable recommendations."
            },
            'preferences': {
                'tr': "❤️ Tercihleriniz hakkında daha fazla bilgi verebilir misiniz? (örn: mutfak türü, atmosfer, özellikler)",
                'en': "❤️ Could you tell me more about your preferences? (e.g., cuisine type, atmosphere, features)"
            },
            'date_time': {
                'tr': "📅 Ne zaman planladığınızı belirtir misiniz? (tarih, saat veya genel zaman dilimi)",
                'en': "📅 When are you planning this? (date, time, or general time frame)"
            },
            'group_size': {
                'tr': "👥 Kaç kişi olacaksınız? Bu bana daha uygun yerler önermemi sağlar.",
                'en': "👥 How many people will there be? This helps me suggest more appropriate places."
            }
        }
    
    def _initialize_no_results_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize no results response templates"""
        return {
            'restaurant': {
                'tr': "😔 Üzgünüm, bu kriterlere uyan restoran bulamadım. Farklı bir mutfak türü veya bölge denemek ister misiniz?",
                'en': "😔 Sorry, I couldn't find restaurants matching those criteria. Would you like to try a different cuisine or area?"
            },
            'attraction': {
                'tr': "😔 Üzgünüm, bu kriterlere uyan yer bulamadım. Farklı bir kategori veya bölge denemek ister misiniz?",
                'en': "😔 Sorry, I couldn't find attractions matching those criteria. Would you like to try a different category or area?"
            },
            'transportation': {
                'tr': "😔 Üzgünüm, bu rota için bilgi bulamadım. Başlangıç ve bitiş noktalarını kontrol edebilir misiniz?",
                'en': "😔 Sorry, I couldn't find information for that route. Could you check the start and end points?"
            },
            'hotel': {
                'tr': "😔 Üzgünüm, bu kriterlere uyan konaklama bulamadım. Farklı bir bütçe veya bölge denemek ister misiniz?",
                'en': "😔 Sorry, I couldn't find accommodations matching those criteria. Would you like to try a different budget or area?"
            },
            'general': {
                'tr': "😔 Üzgünüm, bu arama için sonuç bulamadım. Sorunuzu farklı şekilde sormayı deneyin.",
                'en': "😔 Sorry, I couldn't find results for that search. Try rephrasing your question."
            }
        }
    
    def _initialize_special_case_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize special case templates (greetings, thanks, goodbye)"""
        return {
            'greeting_morning': {
                'tr': "🌅 Günaydın! İstanbul'da size nasıl yardımcı olabilirim?",
                'en': "🌅 Good morning! How can I help you with Istanbul today?"
            },
            'greeting_afternoon': {
                'tr': "☀️ İyi günler! İstanbul hakkında size nasıl yardımcı olabilirim?",
                'en': "☀️ Good afternoon! How can I help you with Istanbul?"
            },
            'greeting_evening': {
                'tr': "🌆 İyi akşamlar! İstanbul'da size nasıl yardımcı olabilirim?",
                'en': "🌆 Good evening! How can I help you with Istanbul?"
            },
            'greeting_general': {
                'tr': "👋 Merhaba! İstanbul'da size nasıl yardımcı olabilirim?",
                'en': "👋 Hello! How can I help you with Istanbul?"
            },
            'thanks': {
                'tr': "😊 Rica ederim! Başka bir konuda yardımcı olmamı ister misiniz?",
                'en': "😊 You're welcome! Is there anything else I can help you with?"
            },
            'goodbye': {
                'tr': "👋 Hoşça kalın! İstanbul'da güzel vakit geçirin! 🌟",
                'en': "👋 Goodbye! Have a wonderful time in Istanbul! 🌟"
            }
        }
    
    def get_fallback_response(
        self,
        intent: str,
        language: str = 'en'
    ) -> str:
        """
        Get fallback response for an intent
        
        Args:
            intent: Intent type (restaurant, attraction, etc.)
            language: Language code ('en' or 'tr')
        
        Returns:
            Fallback response string
        """
        lang_key = 'tr' if language == 'tr' else 'en'
        
        # Try to get specific intent fallback
        if intent in self.fallback_templates:
            return self.fallback_templates[intent][lang_key]
        
        # Return general fallback
        return self.fallback_templates['general'][lang_key]
    
    def get_emergency_response(
        self,
        error_type: str,
        language: str = 'en',
        error_details: Optional[str] = None
    ) -> str:
        """
        Get emergency response for an error
        
        Args:
            error_type: Type of error (system_error, timeout, etc.)
            language: Language code ('en' or 'tr')
            error_details: Optional error details to log
        
        Returns:
            Emergency response string
        """
        lang_key = 'tr' if language == 'tr' else 'en'
        
        if error_details:
            logger.error(f"Emergency response triggered: {error_type} - {error_details}")
        
        # Try to get specific error response
        if error_type in self.emergency_templates:
            return self.emergency_templates[error_type][lang_key]
        
        # Return generic system error
        return self.emergency_templates['system_error'][lang_key]
    
    def get_clarification_request(
        self,
        missing_info: str,
        language: str = 'en',
        context: Optional[str] = None
    ) -> str:
        """
        Get clarification request for missing information
        
        Args:
            missing_info: Type of missing info (location, budget, etc.)
            language: Language code ('en' or 'tr')
            context: Optional context to append
        
        Returns:
            Clarification request string
        """
        lang_key = 'tr' if language == 'tr' else 'en'
        
        # Try to get specific clarification
        if missing_info in self.clarification_templates:
            response = self.clarification_templates[missing_info][lang_key]
        else:
            # Generic clarification
            response = (
                "🤔 Lütfen daha fazla bilgi verebilir misiniz?" if language == 'tr'
                else "🤔 Could you provide more information?"
            )
        
        # Append context if provided
        if context:
            response += f"\n\n{context}"
        
        return response
    
    def get_no_results_response(
        self,
        query_type: str,
        language: str = 'en',
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Get response for no results found
        
        Args:
            query_type: Type of query (restaurant, attraction, etc.)
            language: Language code ('en' or 'tr')
            suggestions: Optional list of alternative suggestions
        
        Returns:
            No results response string
        """
        lang_key = 'tr' if language == 'tr' else 'en'
        
        # Try to get specific no results message
        if query_type in self.no_results_templates:
            response = self.no_results_templates[query_type][lang_key]
        else:
            response = self.no_results_templates['general'][lang_key]
        
        # Add suggestions if provided
        if suggestions:
            suggestion_header = (
                "\n\n💡 Bunları deneyebilirsiniz:" if language == 'tr'
                else "\n\n💡 You could try:"
            )
            response += suggestion_header
            for suggestion in suggestions[:3]:  # Max 3 suggestions
                response += f"\n• {suggestion}"
        
        return response
    
    def get_special_case_response(
        self,
        case_type: str,
        language: str = 'en'
    ) -> str:
        """
        Get special case response (greeting, thanks, goodbye)
        
        Args:
            case_type: Type of special case (greeting_morning, thanks, goodbye, etc.)
            language: Language code ('en' or 'tr')
        
        Returns:
            Special case response string
        """
        lang_key = 'tr' if language == 'tr' else 'en'
        
        # Try to get specific special case response
        if case_type in self.special_case_templates:
            return self.special_case_templates[case_type][lang_key]
        
        # Return general greeting as fallback
        return self.special_case_templates['greeting_general'][lang_key]
    
    def format_bilingual_list(
        self,
        items: List[str],
        header_tr: str,
        header_en: str,
        language: str = 'en'
    ) -> str:
        """
        Format a list with bilingual headers
        
        Args:
            items: List of items to format
            header_tr: Turkish header
            header_en: English header
            language: Primary language code
        
        Returns:
            Formatted bilingual list
        """
        if not items:
            return (
                "😔 Liste boş." if language == 'tr'
                else "😔 List is empty."
            )
        
        # Use primary language header
        header = header_tr if language == 'tr' else header_en
        
        response = f"{header}\n\n"
        for i, item in enumerate(items, 1):
            response += f"{i}. {item}\n"
        
        return response
    
    def get_help_message(self, language: str = 'en') -> str:
        """
        Get help message with available features
        
        Args:
            language: Language code ('en' or 'tr')
        
        Returns:
            Help message string
        """
        if language == 'tr':
            return """
🤖 **Istanbul AI - Yardım**

Size şu konularda yardımcı olabilirim:

🍽️ **Restoranlar**
- Mutfak türüne göre restoran arama
- Bütçeye uygun yerler bulma
- Popüler restoranları keşfetme

🏛️ **Gezilecek Yerler**
- Müzeler, camiler, saraylar
- Tarihi mekanlar
- Gizli cennetler

🚇 **Ulaşım**
- En hızlı rotalar
- Toplu taşıma bilgileri
- Taksi ve alternatif ulaşım

🏨 **Konaklama**
- Otel önerileri
- Bütçeye uygun seçenekler
- Bölge bazında konaklama

🌤️ **Hava Durumu**
- Güncel hava bilgileri
- Haftalık tahminler

Bir şey sormak için mesaj yazmanız yeterli! 😊
"""
        else:
            return """
🤖 **Istanbul AI - Help**

I can help you with:

🍽️ **Restaurants**
- Search by cuisine type
- Find budget-friendly options
- Discover popular restaurants

🏛️ **Attractions**
- Museums, mosques, palaces
- Historical sites
- Hidden gems

🚇 **Transportation**
- Fastest routes
- Public transport info
- Taxi and alternatives

🏨 **Accommodation**
- Hotel recommendations
- Budget-friendly options
- Area-based lodging

🌤️ **Weather**
- Current conditions
- Weekly forecasts

Just send me a message to ask anything! 😊
"""
    
    def get_welcome_message(self, language: str = 'en', user_name: Optional[str] = None) -> str:
        """
        Get welcome message for new users
        
        Args:
            language: Language code ('en' or 'tr')
            user_name: Optional user name to personalize
        
        Returns:
            Welcome message string
        """
        greeting = f"{user_name}, " if user_name else ""
        
        if language == 'tr':
            return f"""
👋 {greeting}İstanbul AI'a hoş geldiniz!

🎯 Ben İstanbul hakkında uzman yapay zeka asistanınızım. Size şunlarda yardımcı olabilirim:

✨ Harika restoranlar bul
✨ Gezilecek yerleri keşfet
✨ Ulaşım planla
✨ Otel önerileri al
✨ Hava durumunu öğren

Bana İstanbul hakkında ne sormak istersiniz? 🇹🇷
"""
        else:
            return f"""
👋 Welcome to Istanbul AI{', ' + user_name if user_name else ''}!

🎯 I'm your expert AI assistant for Istanbul. I can help you:

✨ Find amazing restaurants
✨ Discover attractions
✨ Plan transportation
✨ Get hotel recommendations
✨ Check the weather

What would you like to know about Istanbul? 🇹🇷
"""

    def get_feedback_request(self, language: str = 'en') -> str:
        """
        Get feedback request message
        
        Args:
            language: Language code ('en' or 'tr')
        
        Returns:
            Feedback request string
        """
        if language == 'tr':
            return """
💭 **Geri Bildiriminiz Önemli!**

Yardımcı olabildim mi? Deneyiminizi değerlendirmek ister misiniz?

👍 Harika yardımcı oldu
👌 Fena değildi  
👎 Geliştirilmeli

Görüşleriniz benim gelişmeme yardımcı oluyor! 🙏
"""
        else:
            return """
💭 **Your Feedback Matters!**

Was I helpful? Would you like to rate your experience?

👍 Very helpful
👌 It was okay
👎 Needs improvement

Your feedback helps me improve! 🙏
"""
