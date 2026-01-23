"""
Internationalization service for Istanbul AI chatbot backend.
Provides multi-language response translation support.
"""
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class I18nService:
    """Service for handling internationalization of chatbot responses."""
    
    def __init__(self):
        self.translations = self._load_translations()
        self.supported_languages = ["en", "tr", "de", "fr", "ar", "ru"]
        self.default_language = "en"
    
    def _load_translations(self) -> Dict[str, Dict[str, Any]]:
        """Load translation dictionaries for all supported languages."""
        translations = {
            "en": {
                "welcome": "Welcome to Istanbul AI! How can I help you explore the city?",
                "restaurant_intro": "Here are some great restaurants in {district}:",
                "museum_intro": "Discover these amazing museums:",
                "transport_intro": "Here's your transportation guide:",
                "attractions_intro": "Top attractions and places to visit:",
                "general_intro": "I'd be happy to help you explore Istanbul!",
                "error_message": "Sorry, I encountered an error. Please try rephrasing your question.",
                "no_results": "I couldn't find specific information about that. Could you try asking about restaurants, museums, or attractions?",
                "districts": {
                    "sultanahmet": "Sultanahmet",
                    "beyoglu": "Beyoğlu", 
                    "kadikoy": "Kadıköy",
                    "fatih": "Fatih",
                    "besiktas": "Beşiktaş",
                    "sisli": "Şişli",
                    "galata": "Galata",
                    "eminonu": "Eminönü"
                }
            },
            "tr": {
                "welcome": "İstanbul AI'ya hoş geldiniz! Şehri keşfetmenizde nasıl yardımcı olabilirim?",
                "restaurant_intro": "{district} bölgesindeki harika restoranlar:",
                "museum_intro": "Bu muhteşem müzeleri keşfedin:",
                "transport_intro": "İşte ulaşım rehberiniz:",
                "attractions_intro": "En popüler yerler ve görülecek yerler:",
                "general_intro": "İstanbul'u keşfetmenizde size yardımcı olmaktan mutluluk duyarım!",
                "error_message": "Üzgünüm, bir hata ile karşılaştım. Lütfen sorunuzu farklı şekilde ifade etmeyi deneyin.",
                "no_results": "Bu konu hakkında özel bilgi bulamadım. Restoranlar, müzeler veya gezilecek yerler hakkında soru sormayı deneyebilir misiniz?",
                "districts": {
                    "sultanahmet": "Sultanahmet",
                    "beyoglu": "Beyoğlu",
                    "kadikoy": "Kadıköy", 
                    "fatih": "Fatih",
                    "besiktas": "Beşiktaş",
                    "sisli": "Şişli",
                    "galata": "Galata",
                    "eminonu": "Eminönü"
                }
            },
            "de": {
                "welcome": "Willkommen bei Istanbul AI! Wie kann ich Ihnen helfen, die Stadt zu erkunden?",
                "restaurant_intro": "Hier sind einige großartige Restaurants in {district}:",
                "museum_intro": "Entdecken Sie diese erstaunlichen Museen:",
                "transport_intro": "Hier ist Ihr Transportführer:",
                "attractions_intro": "Top-Attraktionen und Sehenswürdigkeiten:",
                "general_intro": "Ich helfe Ihnen gerne dabei, Istanbul zu erkunden!",
                "error_message": "Entschuldigung, ich bin auf einen Fehler gestoßen. Bitte versuchen Sie, Ihre Frage anders zu formulieren.",
                "no_results": "Ich konnte keine spezifischen Informationen dazu finden. Könnten Sie versuchen, nach Restaurants, Museen oder Attraktionen zu fragen?",
                "districts": {
                    "sultanahmet": "Sultanahmet",
                    "beyoglu": "Beyoğlu",
                    "kadikoy": "Kadıköy",
                    "fatih": "Fatih", 
                    "besiktas": "Beşiktaş",
                    "sisli": "Şişli",
                    "galata": "Galata",
                    "eminonu": "Eminönü"
                }
            },
            "fr": {
                "welcome": "Bienvenue sur Istanbul AI ! Comment puis-je vous aider à explorer la ville ?",
                "restaurant_intro": "Voici quelques excellents restaurants dans {district} :",
                "museum_intro": "Découvrez ces musées extraordinaires :",
                "transport_intro": "Voici votre guide de transport :",
                "attractions_intro": "Principales attractions et lieux à visiter :",
                "general_intro": "Je serais ravi de vous aider à explorer Istanbul !",
                "error_message": "Désolé, j'ai rencontré une erreur. Veuillez essayer de reformuler votre question.",
                "no_results": "Je n'ai pas pu trouver d'informations spécifiques à ce sujet. Pourriez-vous essayer de demander des restaurants, des musées ou des attractions ?",
                "districts": {
                    "sultanahmet": "Sultanahmet",
                    "beyoglu": "Beyoğlu",
                    "kadikoy": "Kadıköy",
                    "fatih": "Fatih",
                    "besiktas": "Beşiktaş", 
                    "sisli": "Şişli",
                    "galata": "Galata",
                    "eminonu": "Eminönü"
                }
            },
            "ar": {
                "welcome": "مرحباً بكم في ذكاء إسطنبول! كيف يمكنني مساعدتكم في استكشاف المدينة؟",
                "restaurant_intro": "إليكم بعض المطاعم الرائعة في {district}:",
                "museum_intro": "اكتشفوا هذه المتاحف المذهلة:",
                "transport_intro": "إليكم دليل النقل الخاص بكم:",
                "attractions_intro": "أهم المعالم والأماكن التي يجب زيارتها:",
                "general_intro": "سأكون سعيداً لمساعدتكم في استكشاف إسطنبول!",
                "error_message": "آسف، واجهت خطأ. يرجى إعادة صياغة سؤالكم.",
                "no_results": "لم أستطع العثور على معلومات محددة حول ذلك. هل يمكنكم السؤال عن المطاعم أو المتاحف أو المعالم السياحية؟",
                "districts": {
                    "sultanahmet": "السلطان أحمد",
                    "beyoglu": "بيوغلو",
                    "kadikoy": "قاضي كوي",
                    "fatih": "الفاتح",
                    "besiktas": "بشيكتاش",
                    "sisli": "شيشلي",
                    "galata": "غلطة",
                    "eminonu": "أمين أونو"
                }
            },
            "ru": {
                "welcome": "Добро пожаловать в Искусственный интеллект Стамбула! Как я могу помочь вам исследовать город?",
                "restaurant_intro": "Вот несколько отличных ресторанов в районе {district}:",
                "museum_intro": "Откройте для себя эти удивительные музеи:",
                "transport_intro": "Вот ваш транспортный гид:",
                "attractions_intro": "Главные достопримечательности и места для посещения:",
                "general_intro": "Я буду рад помочь вам исследовать Стамбул!",
                "error_message": "Извините, произошла ошибка. Пожалуйста, попробуйте переформулировать ваш вопрос.",
                "no_results": "Я не смог найти конкретную информацию по этому поводу. Не могли бы вы спросить о ресторанах, музеях или достопримечательностях?",
                "districts": {
                    "sultanahmet": "Султанахмет",
                    "beyoglu": "Бейоглу",
                    "kadikoy": "Кадыкёй",
                    "fatih": "Фатих",
                    "besiktas": "Бешикташ",
                    "sisli": "Шишли",
                    "galata": "Галата",
                    "eminonu": "Эминёню"
                }
            }
        }
        
        logger.info(f"Loaded translations for languages: {list(translations.keys())}")
        return translations
    
    def get_language_from_headers(self, accept_language: Optional[str]) -> str:
        """Extract preferred language from Accept-Language header."""
        if not accept_language:
            return self.default_language
            
        # Parse Accept-Language header (simplified)
        languages = []
        for lang_part in accept_language.split(','):
            lang = lang_part.strip().split(';')[0].split('-')[0].lower()
            if lang in self.supported_languages:
                languages.append(lang)
        
        return languages[0] if languages else self.default_language
    
    def translate(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Translate a message key to the specified language.
        
        Args:
            key: Translation key (e.g., 'welcome', 'restaurant_intro')
            language: Target language code (en, tr, de, fr)
            **kwargs: Variables for string formatting
            
        Returns:
            Translated and formatted message
        """
        if language not in self.supported_languages:
            language = self.default_language
            
        try:
            # Navigate nested keys (e.g., 'districts.sultanahmet')
            translation = self.translations[language]
            for part in key.split('.'):
                translation = translation[part]
            
            # Ensure we have a string for formatting
            if isinstance(translation, str):
                # Format with provided variables
                if kwargs:
                    return translation.format(**kwargs)
                return translation
            else:
                logger.warning(f"Translation for key '{key}' is not a string: {type(translation)}")
                return str(translation)
            
        except (KeyError, TypeError) as e:
            logger.warning(f"Translation not found for key '{key}' in language '{language}': {e}")
            # Fallback to English
            try:
                translation = self.translations[self.default_language]
                for part in key.split('.'):
                    translation = translation[part]
                
                if isinstance(translation, str):
                    if kwargs:
                        return translation.format(**kwargs)
                    return translation
                else:
                    return str(translation)
            except (KeyError, TypeError):
                return f"[Translation missing: {key}]"
    
    def translate_response(self, response_data: Dict[str, Any], language: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate chatbot response data to the specified language.
        
        Args:
            response_data: Response dictionary with 'message' and optionally 'intro_message'
            language: Target language code
            
        Returns:
            Response with translated messages
        """
        if language not in self.supported_languages:
            language = self.default_language
            
        translated_response = response_data.copy()
        
        # Translate intro message if present
        if 'intro_message' in response_data:
            intro_key = response_data.get('intro_key', 'general_intro')
            intro_params = response_data.get('intro_params', {})
            translated_response['intro_message'] = self.translate(intro_key, language, **intro_params)
        
        return translated_response
    
    def get_supported_languages(self) -> list:
        """Get list of supported language codes."""
        return self.supported_languages.copy()
    
    def get_language_info(self) -> Dict[str, Dict[str, str]]:
        """Get language information with names and flags."""
        return {
            "en": {"name": "English", "flag": "🇺🇸"},
            "tr": {"name": "Türkçe", "flag": "🇹🇷"},
            "de": {"name": "Deutsch", "flag": "🇩🇪"},
            "fr": {"name": "Français", "flag": "🇫🇷"},
            "ar": {"name": "العربية", "flag": "🇸🇦"},
            "ru": {"name": "Русский", "flag": "🇷🇺"}
        }
    
    def translate_openai_response(self, english_response: str, target_language: str) -> str:
        """
        Translate an OpenAI-generated English response to the target language.
        This handles dynamic content that isn't in the pre-defined dictionary.
        """
        if target_language == "en" or target_language not in self.supported_languages:
            return english_response
        
        # For now, we'll use simple keyword-based translation patterns
        # In a production system, you'd want to use a proper translation API
        
        translation_patterns = {
            "ar": {
                # Restaurant patterns
                "Here are some great restaurants": "إليكم بعض المطاعم الرائعة",
                "in Sultanahmet": "في السلطان أحمد",
                "in Beyoğlu": "في بيوغلو", 
                "in Kadıköy": "في قاضي كوي",
                "Traditional Turkish": "تركي تقليدي",
                "Modern cuisine": "مطبخ عصري",
                "Great views": "إطلالات رائعة",
                "Historic district": "حي تاريخي",
                "Popular restaurant": "مطعم شائع",
                "Highly rated": "عالي التقييم",
                "Budget-friendly": "مناسب للميزانية",
                "Fine dining": "مطعم راقي",
                
                # Museum patterns  
                "museums to visit": "متاحف للزيارة",
                "Historical museum": "متحف تاريخي",
                "Art gallery": "معرض فني",
                "Palace": "قصر",
                
                # Transportation patterns
                "Take the metro": "خذوا المترو",
                "Bus number": "رقم الحافلة",
                "Ferry to": "عبارة إلى",
                "Taxi": "تاكسي",
                "Walking distance": "مسافة المشي",
                
                # Common phrases
                "I recommend": "أنصح بـ",
                "You can find": "يمكنكم العثور على",
                "Located in": "يقع في",
                "Open daily": "مفتوح يومياً",
                "Closed on": "مغلق يوم",
                "Price range": "نطاق الأسعار"
            },
            "tr": {
                # Add Turkish patterns if needed
            },
            "de": {
                # Add German patterns if needed  
            },
            "fr": {
                # Add French patterns if needed
            }
        }
        
        if target_language in translation_patterns:
            patterns = translation_patterns[target_language]
            translated_response = english_response
            
            # Apply translation patterns
            for english_phrase, translated_phrase in patterns.items():
                translated_response = translated_response.replace(english_phrase, translated_phrase)
            
            return translated_response
        
        return english_response
    
    def detect_query_intent_arabic(self, query: str) -> Dict[str, Any]:
        """
        Detect intent for Arabic queries using keyword matching.
        """
        query_lower = query.lower()
        
        # Restaurant keywords in Arabic
        restaurant_keywords = [
            'مطعم', 'مطاعم', 'أكل', 'طعام', 'وجبة', 'غداء', 'عشاء', 'إفطار',
            'مأكولات', 'كباب', 'دونر', 'لحم', 'سمك', 'دجاج', 'حلويات'
        ]
        
        # Museum/attraction keywords in Arabic  
        museum_keywords = [
            'متحف', 'متاحف', 'قصر', 'مسجد', 'كنيسة', 'معرض', 'فن',
            'تاريخ', 'ثقافة', 'آثار', 'تراث', 'حضارة'
        ]
        
        # Transportation keywords in Arabic
        transport_keywords = [
            'مواصلات', 'نقل', 'مترو', 'حافلة', 'تاكسي', 'عبارة', 'قطار',
            'سيارة', 'طريق', 'وصول', 'ذهاب', 'عودة'
        ]
        
        # District keywords in Arabic
        district_keywords = [
            'السلطان أحمد', 'بيوغلو', 'قاضي كوي', 'الفاتح', 'بشيكتاش',
            'شيشلي', 'غلطة', 'أمين أونو', 'حي', 'منطقة', 'مكان'
        ]
        
        # Check for restaurants
        if any(keyword in query_lower for keyword in restaurant_keywords):
            return {
                'intent': 'restaurant',
                'confidence': 0.8,
                'detected_keywords': [kw for kw in restaurant_keywords if kw in query_lower],
                'language': 'ar'
            }
        
        # Check for museums/attractions  
        if any(keyword in query_lower for keyword in museum_keywords):
            return {
                'intent': 'museum',
                'confidence': 0.8,
                'detected_keywords': [kw for kw in museum_keywords if kw in query_lower],
                'language': 'ar'
            }
            
        # Check for transportation
        if any(keyword in query_lower for keyword in transport_keywords):
            return {
                'intent': 'transportation', 
                'confidence': 0.8,
                'detected_keywords': [kw for kw in transport_keywords if kw in query_lower],
                'language': 'ar'
            }
            
        # Check for district-specific queries
        detected_districts = [kw for kw in district_keywords if kw in query_lower]
        if detected_districts:
            return {
                'intent': 'district_info',
                'confidence': 0.7,
                'detected_districts': detected_districts,
                'language': 'ar'
            }
        
        # Default to general query
        return {
            'intent': 'general',
            'confidence': 0.3,
            'language': 'ar'
        }
    
    def get_multilingual_system_prompt(self, language: str) -> str:
        """Get language-specific system prompt for OpenAI."""
        prompts = {
            "en": "You are KAM, an AI travel assistant for Istanbul. Provide helpful, detailed information about restaurants, museums, transportation, and attractions in Istanbul. Keep responses informative and friendly.",
            "tr": "Sen KAM, İstanbul için bir AI seyahat asistanısın. Restoranlar, müzeler, ulaşım ve gezilecek yerler hakkında yararlı ve detaylı bilgiler ver. Yanıtlarını bilgilendirici ve samimi tut.",
            "de": "Du bist KAM, ein AI-Reiseassistent für Istanbul. Gib hilfreiche, detaillierte Informationen über Restaurants, Museen, Transport und Sehenswürdigkeiten in Istanbul. Halte die Antworten informativ und freundlich.",
            "fr": "Tu es KAM, un assistant de voyage AI pour Istanbul. Fournis des informations utiles et détaillées sur les restaurants, musées, transports et attractions d'Istanbul. Garde les réponses informatives et amicales.",
            "ar": "أنت KAM، مساعد سفر ذكي لإسطنبول. قدم معلومات مفيدة ومفصلة حول المطاعم والمتاحف والنقل والمعالم السياحية في إسطنبول. اجعل إجاباتك غنية بالمعلومات وودودة.",
            "ru": "Ты KAM, AI-помощник по путешествиям в Стамбул. Предоставь полезную, подробную информацию о ресторанах, музеях, транспорте и достопримечательностях Стамбула. Держи ответы информативными и дружелюбными."
        }
        return prompts.get(language, prompts["en"])
    
    def should_use_ai_response(self, user_input: str, language: str) -> bool:
        """Determine if we should use AI for complex queries vs template responses."""
        # Simple greetings use templates (faster)
        simple_patterns = {
            "en": ["hello", "hi", "hey", "thanks", "thank you"],
            "tr": ["merhaba", "selam", "teşekkür", "sağol"],
            "de": ["hallo", "hi", "danke", "guten tag"],
            "fr": ["bonjour", "salut", "merci", "bonsoir"],
            "ar": ["مرحبا", "أهلا", "شكرا", "السلام"],
            "ru": ["привет", "здравствуйте", "спасибо", "добро пожаловать"]
        }
        
        user_lower = user_input.lower().strip()
        patterns = simple_patterns.get(language, [])
        
        # If it's a simple greeting, use template
        if any(pattern in user_lower for pattern in patterns) and len(user_input.strip()) < 10:
            return False
            
        # For complex questions, use AI
        return True
        

# Global instance
i18n_service = I18nService()
