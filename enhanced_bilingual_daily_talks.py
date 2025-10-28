#!/usr/bin/env python3
"""
Enhanced Bilingual Daily Talks System for AI Istanbul
======================================================

Provides natural, contextual conversations in both Turkish and English with:
- Advanced intent recognition
- Context-aware responses
- Emotional intelligence
- Cultural awareness
- Seamless language switching
- Memory of past conversations
- Personalized interactions
"""

import re
import json
import random
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Language(Enum):
    TURKISH = "tr"
    ENGLISH = "en"
    MIXED = "mixed"

class ConversationIntent(Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    SMALLTALK = "smalltalk"
    QUESTION = "question"
    HELP_REQUEST = "help_request"
    THANKS = "thanks"
    EMOTION = "emotion"
    PLANNING = "planning"

@dataclass
class UserContext:
    """User conversation context"""
    language: Language = Language.ENGLISH
    location: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    conversation_history: List[Dict] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    emotional_state: str = "neutral"
    
class EnhancedBilingualDailyTalks:
    """Enhanced bilingual conversational AI for Istanbul"""
    
    def __init__(self):
        self.load_conversation_patterns()
        self.load_cultural_knowledge()
        
    def load_conversation_patterns(self):
        """Load bilingual conversation patterns"""
        
        # Greeting patterns
        self.greetings = {
            Language.TURKISH: {
                "morning": [
                    "Günaydın! ☀️ İstanbul'da yeni bir gün başlıyor! Size nasıl yardımcı olabilirim?",
                    "Günaydın! 🌅 Bugün İstanbul'da neler keşfetmek istersiniz?",
                    "Sabah hayırlara! ☕ Kahvenizi alın, İstanbul'u keşfetmeye başlayalım!",
                    "Günaydın! 🏙️ İstanbul sabahları çok güzel, sizin için ne planlayabilirim?"
                ],
                "afternoon": [
                    "İyi günler! ☀️ Öğleden sonra İstanbul'da neler yapmak istersiniz?",
                    "Merhaba! 🌞 Gününüz nasıl geçiyor? Size yardımcı olabilir miyim?",
                    "İyi öğlenler! 🍽️ Belki güzel bir öğle yemeği önerisi?",
                    "Tünaydın! 😊 İstanbul'un tadını çıkarıyor musunuz?"
                ],
                "evening": [
                    "İyi akşamlar! 🌆 İstanbul akşamları büyülü! Ne yapmak istersiniz?",
                    "Akşamınız hayırlı olsun! ✨ Size nasıl yardımcı olabilirim?",
                    "İyi akşamlar! 🌃 Akşam için özel planlarınız var mı?",
                    "Günün sonu yaklaşıyor! 🌅 İstanbul'da güzel bir akşam geçirmeye ne dersiniz?"
                ],
                "night": [
                    "İyi geceler! 🌙 Gece İstanbul'u keşfetmek mi istiyorsunuz?",
                    "Merhaba! 🌃 Gece hayatı mı, yoksa sakin bir akşam mı?",
                    "İyi geceler! ✨ İstanbul geceleri özel, size nasıl yardımcı olabilirim?",
                    "Gecenin bu saatinde bile İstanbul canlı! 🎭 Ne arıyorsunuz?"
                ]
            },
            Language.ENGLISH: {
                "morning": [
                    "Good morning! ☀️ A new day in Istanbul begins! How can I help you today?",
                    "Good morning! 🌅 What would you like to explore in Istanbul today?",
                    "Rise and shine! ☕ Grab your coffee, let's discover Istanbul together!",
                    "Good morning! 🏙️ Istanbul mornings are beautiful, what shall we plan for you?"
                ],
                "afternoon": [
                    "Good afternoon! ☀️ What would you like to do in Istanbul this afternoon?",
                    "Hello! 🌞 How's your day going? Can I help you with anything?",
                    "Good afternoon! 🍽️ Perhaps a lovely lunch recommendation?",
                    "Hey there! 😊 Are you enjoying Istanbul so far?"
                ],
                "evening": [
                    "Good evening! 🌆 Istanbul evenings are magical! What would you like to do?",
                    "Evening greetings! ✨ How can I assist you tonight?",
                    "Good evening! 🌃 Do you have any special plans for tonight?",
                    "The day is winding down! 🌅 How about a beautiful evening in Istanbul?"
                ],
                "night": [
                    "Good night! 🌙 Want to explore Istanbul's nightlife?",
                    "Hello! 🌃 Looking for nightlife or a quiet evening?",
                    "Good evening! ✨ Istanbul nights are special, how can I help?",
                    "Even at this hour, Istanbul is alive! 🎭 What are you looking for?"
                ]
            }
        }
        
        # Casual responses
        self.casual_responses = {
            Language.TURKISH: {
                "how_are_you": [
                    "Ben harika! 😊 İstanbul'u ziyaretçilerle paylaşmayı çok seviyorum. Ya siz?",
                    "Çok iyiyim, teşekkür ederim! 🌟 İstanbul'da neler yapıyorsunuz?",
                    "Müthiş! 🎉 Size yardımcı olmak için buradayım. Neler keşfetmek istersiniz?",
                    "Harikayım! ☺️ İstanbul'un güzelliklerini paylaşmak beni mutlu ediyor."
                ],
                "good": [
                    "Ne güzel! 😊 Size nasıl yardımcı olabilirim?",
                    "Harika! 🌟 İstanbul deneyiminizi daha iyi hale getirebilir miyim?",
                    "Mükemmel! 🎯 Başka ne öğrenmek istersiniz?",
                    "Süper! 👍 Daha fazla öneri ister misiniz?"
                ],
                "thanks": [
                    "Rica ederim! 😊 Başka bir konuda yardımcı olabilir miyim?",
                    "Ne demek! 🌟 İstanbul'u keşfetmenizde yardımcı olmak benim için zevk!",
                    "Her zaman! 💙 Başka sorunuz varsa çekinmeyin!",
                    "Memnuniyetle! 😄 Size yardımcı olduğuma sevindim!"
                ],
                "confused": [
                    "Anlıyorum, açıklayayım! 🤔 Hangi konuda daha fazla bilgi istersiniz?",
                    "Üzgünüm, daha net açıklayayım. 💡 Neyi merak ediyorsunuz?",
                    "Hadi daha basit anlatayım! 😊 Size nasıl yardımcı olabilirim?",
                    "Anladım! 📝 Adım adım açıklayayım, tamam mı?"
                ]
            },
            Language.ENGLISH: {
                "how_are_you": [
                    "I'm great! 😊 I love sharing Istanbul with visitors. How about you?",
                    "I'm doing wonderfully, thank you! 🌟 What brings you to Istanbul?",
                    "Fantastic! 🎉 I'm here to help you. What would you like to explore?",
                    "I'm wonderful! ☺️ Sharing Istanbul's beauty makes me happy."
                ],
                "good": [
                    "That's great! 😊 How can I help you?",
                    "Wonderful! 🌟 Can I make your Istanbul experience even better?",
                    "Perfect! 🎯 What else would you like to know?",
                    "Super! 👍 Would you like more suggestions?"
                ],
                "thanks": [
                    "You're welcome! 😊 Can I help with anything else?",
                    "My pleasure! 🌟 I love helping people discover Istanbul!",
                    "Anytime! 💙 Don't hesitate if you have more questions!",
                    "Happy to help! 😄 Glad I could assist you!"
                ],
                "confused": [
                    "I understand, let me explain! 🤔 What would you like to know more about?",
                    "Sorry, let me clarify that. 💡 What are you curious about?",
                    "Let me break it down simply! 😊 How can I help?",
                    "Got it! 📝 Let me explain step by step, okay?"
                ]
            }
        }
        
        # Contextual responses for Istanbul topics
        self.topic_responses = {
            "weather": {
                Language.TURKISH: [
                    "İstanbul havası değişken olabilir! 🌤️ Size güncel hava durumu ve ona göre öneriler verebilirim.",
                    "Hava durumuna göre farklı önerilerim var! ☀️🌧️ Ne yapmak istersiniz?",
                    "İstanbul'da hava her şeyi değiştirir! 😊 Bugünkü planlara bakalım mı?"
                ],
                Language.ENGLISH: [
                    "Istanbul weather can be changeable! 🌤️ I can give you current weather and recommendations.",
                    "I have different suggestions based on the weather! ☀️🌧️ What would you like to do?",
                    "Weather changes everything in Istanbul! 😊 Shall we look at today's plans?"
                ]
            },
            "food": {
                Language.TURKISH: [
                    "Yemek denilince İstanbul'un binlerce seçeneği var! 🍽️ Ne tür mutfak arıyorsunuz?",
                    "İstanbul'un lezzet durağı! 😋 Geleneksel mi, modern mi tercih edersiniz?",
                    "Aç mısınız? 🤤 Size harika yerler önerebilirim! Bütçeniz nedir?"
                ],
                Language.ENGLISH: [
                    "Istanbul has thousands of dining options! 🍽️ What type of cuisine are you looking for?",
                    "Welcome to Istanbul's food paradise! 😋 Traditional or modern cuisine?",
                    "Hungry? 🤤 I can recommend amazing places! What's your budget?"
                ]
            },
            "places": {
                Language.TURKISH: [
                    "İstanbul gezilecek yerlerle dolu! 🏛️ Tarihi mi, modern mi, yoksa gizli kalmış yerler mi?",
                    "Her köşesinde tarih var bu şehirde! 🕌 Hangi semti keşfetmek istersiniz?",
                    "İstanbul'u keşfetmeye hazır mısınız? 🗺️ İlgi alanlarınız neler?"
                ],
                Language.ENGLISH: [
                    "Istanbul is full of amazing places! 🏛️ Historical, modern, or hidden gems?",
                    "Every corner has history here! 🕌 Which district would you like to explore?",
                    "Ready to discover Istanbul? 🗺️ What are your interests?"
                ]
            },
            "transport": {
                Language.TURKISH: [
                    "İstanbul'da ulaşım kolay! 🚇 Nereye gitmek istiyorsunuz?",
                    "Metro, vapur, tramvay, otobüs... 🚌 En iyi rotayı bulalım!",
                    "Ulaşım konusunda uzmanım! 🚊 Nereden nereye?"
                ],
                Language.ENGLISH: [
                    "Getting around Istanbul is easy! 🚇 Where do you want to go?",
                    "Metro, ferry, tram, bus... 🚌 Let's find the best route!",
                    "I'm an expert on transportation! 🚊 Where from and where to?"
                ]
            }
        }
        
        # Emotional responses
        self.emotional_responses = {
            "excited": {
                Language.TURKISH: [
                    "Harika bir enerji! 🎉 İstanbul size müthiş anılar yaşatacak!",
                    "Bu heyecanı sevdim! 🌟 Hemen planlamaya başlayalım!",
                    "Enerjiniz harika! 🚀 İstanbul'u birlikte keşfedelim!"
                ],
                Language.ENGLISH: [
                    "Great energy! 🎉 Istanbul will give you amazing memories!",
                    "I love this excitement! 🌟 Let's start planning right away!",
                    "Your energy is wonderful! 🚀 Let's explore Istanbul together!"
                ]
            },
            "confused": {
                Language.TURKISH: [
                    "Endişelenmeyin, her şeyi açıklayayım! 💡 Hangi konuda karar veremediz?",
                    "Çok normal, İstanbul büyük bir şehir! 🗺️ Adım adım ilerleyelim.",
                    "Yardımcı olayım! 🤝 Neyi netleştirmemi istersiniz?"
                ],
                Language.ENGLISH: [
                    "Don't worry, I'll explain everything! 💡 What are you unsure about?",
                    "That's totally normal, Istanbul is a big city! 🗺️ Let's take it step by step.",
                    "Let me help! 🤝 What would you like me to clarify?"
                ]
            },
            "tired": {
                Language.TURKISH: [
                    "Dinlenme zamanı! 😴 Size rahatlatıcı yerler önerebilirim.",
                    "Yorgunluk normal, çok geziyorsunuz! ☕ Güzel bir kafe bulalım mı?",
                    "Biraz mola verelim! 🛋️ Sakin aktiviteler ister misiniz?"
                ],
                Language.ENGLISH: [
                    "Time to rest! 😴 I can suggest relaxing spots for you.",
                    "It's normal to be tired, you've been exploring a lot! ☕ Shall we find a nice cafe?",
                    "Let's take a break! 🛋️ Would you like some calm activities?"
                ]
            }
        }
        
    def load_cultural_knowledge(self):
        """Load cultural knowledge for both languages"""
        
        self.cultural_tips = {
            Language.TURKISH: [
                "💡 **İpucu**: İstanbul'da pazarlık yapmak normaldir, özellikle çarşılarda!",
                "☕ **Kültür**: Türk kahvesi içerken acele etmeyin, sohbetin tadını çıkarın!",
                "🕌 **Saygı**: Camileri ziyaret ederken omuzları ve dizleri örtmeyi unutmayın.",
                "🍽️ **Lezzet**: Sokak yemekleri güvenlidir ve çok lezzetlidir!",
                "🚊 **Ulaşım**: İstanbulkart alın, her yerde kullanabilirsiniz!"
            ],
            Language.ENGLISH: [
                "💡 **Tip**: Bargaining is normal in Istanbul, especially in bazaars!",
                "☕ **Culture**: Don't rush your Turkish coffee, enjoy the conversation!",
                "🕌 **Respect**: Remember to cover shoulders and knees when visiting mosques.",
                "🍽️ **Taste**: Street food is safe and delicious here!",
                "🚊 **Transport**: Get an Istanbulkart, you can use it everywhere!"
            ]
        }
        
    def detect_language(self, text: str) -> Language:
        """Detect language of user input"""
        
        # Turkish indicators
        turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        turkish_words = {
            'merhaba', 'selam', 'nasılsın', 'nasılsınız', 'teşekkür', 'teşekkürler',
            'ederim', 'günaydın', 'iyi', 'akşamlar', 'geceler', 'naber', 'naber',
            'istanbul', 'nerede', 'nasıl', 'ne', 'var', 'yok', 'evet', 'hayır',
            'lütfen', 'özür', 'pardon', 'tamam', 'peki', 'güle', 'hoş', 'geldiniz'
        }
        
        # English indicators
        english_words = {
            'hello', 'hi', 'hey', 'thanks', 'thank', 'you', 'please', 'sorry',
            'good', 'morning', 'afternoon', 'evening', 'night', 'how', 'what',
            'where', 'when', 'why', 'yes', 'no', 'ok', 'okay', 'bye', 'goodbye'
        }
        
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Check for Turkish characters
        if any(char in text for char in turkish_chars):
            return Language.TURKISH
        
        # Count matches
        turkish_matches = len(words & turkish_words)
        english_matches = len(words & english_words)
        
        if turkish_matches > english_matches:
            return Language.TURKISH
        elif english_matches > turkish_matches:
            return Language.ENGLISH
        elif turkish_matches == english_matches and turkish_matches > 0:
            return Language.MIXED
        
        # Default to English
        return Language.ENGLISH
    
    def get_time_period(self) -> str:
        """Get current time period"""
        current_hour = datetime.now().hour
        
        if 6 <= current_hour < 12:
            return "morning"
        elif 12 <= current_hour < 17:
            return "afternoon"
        elif 17 <= current_hour < 21:
            return "evening"
        else:
            return "night"
    
    def detect_intent(self, text: str, language: Language) -> ConversationIntent:
        """Detect user intent from text"""
        
        text_lower = text.lower()
        
        # Greeting patterns
        greeting_patterns = {
            Language.TURKISH: ['merhaba', 'selam', 'günaydın', 'iyi günler', 'iyi akşamlar', 'hey', 'naber'],
            Language.ENGLISH: ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
        }
        
        # Farewell patterns
        farewell_patterns = {
            Language.TURKISH: ['güle güle', 'hoşça kal', 'görüşürüz', 'bay', 'bye', 'gidiyorum'],
            Language.ENGLISH: ['goodbye', 'bye', 'see you', 'farewell', 'take care', 'gotta go']
        }
        
        # Thanks patterns
        thanks_patterns = {
            Language.TURKISH: ['teşekkür', 'sağol', 'eyvallah', 'teşekkürler', 'sağolun'],
            Language.ENGLISH: ['thank', 'thanks', 'appreciate', 'grateful']
        }
        
        # Question patterns
        question_words = {
            Language.TURKISH: ['nerede', 'nasıl', 'ne zaman', 'neden', 'hangi', 'kim', 'ne', 'mi', 'mı', 'mu', 'mü'],
            Language.ENGLISH: ['where', 'how', 'when', 'why', 'which', 'who', 'what', 'can', 'could', 'would']
        }
        
        # Check patterns
        patterns = greeting_patterns.get(language, greeting_patterns[Language.ENGLISH])
        if any(pattern in text_lower for pattern in patterns):
            return ConversationIntent.GREETING
        
        patterns = farewell_patterns.get(language, farewell_patterns[Language.ENGLISH])
        if any(pattern in text_lower for pattern in patterns):
            return ConversationIntent.FAREWELL
        
        patterns = thanks_patterns.get(language, thanks_patterns[Language.ENGLISH])
        if any(pattern in text_lower for pattern in patterns):
            return ConversationIntent.THANKS
        
        patterns = question_words.get(language, question_words[Language.ENGLISH])
        if any(pattern in text_lower for pattern in patterns) or text.strip().endswith('?'):
            return ConversationIntent.QUESTION
        
        # Default to small talk
        return ConversationIntent.SMALLTALK
    
    def generate_response(
        self,
        user_input: str,
        context: Optional[UserContext] = None
    ) -> Tuple[str, UserContext]:
        """Generate contextual response"""
        
        if context is None:
            context = UserContext()
        
        # Detect language
        detected_language = self.detect_language(user_input)
        context.language = detected_language
        
        # Detect intent
        intent = self.detect_intent(user_input, detected_language)
        
        # Get time period
        time_period = self.get_time_period()
        
        # Generate response based on intent
        response = ""
        
        if intent == ConversationIntent.GREETING:
            greetings = self.greetings[detected_language][time_period]
            response = random.choice(greetings)
            
            # Add cultural tip sometimes
            if random.random() < 0.3:
                tips = self.cultural_tips[detected_language]
                response += f"\n\n{random.choice(tips)}"
                
        elif intent == ConversationIntent.THANKS:
            thanks_responses = self.casual_responses[detected_language]["thanks"]
            response = random.choice(thanks_responses)
            
        elif intent == ConversationIntent.FAREWELL:
            if detected_language == Language.TURKISH:
                response = random.choice([
                    "Güle güle! 👋 İstanbul'u keşfetmeye devam edin!",
                    "Hoşça kalın! 🌟 Tekrar görüşmek üzere!",
                    "İyi geziler! 🏙️ Size yardımcı olduğuma sevindim!",
                    "Görüşürüz! 💙 İstanbul sizi bekliyor!"
                ])
            else:
                response = random.choice([
                    "Goodbye! 👋 Continue exploring Istanbul!",
                    "Take care! 🌟 Hope to chat again soon!",
                    "Happy travels! 🏙️ Glad I could help!",
                    "See you! 💙 Istanbul awaits you!"
                ])
                
        elif intent == ConversationIntent.QUESTION:
            # For questions, provide helpful direction
            if detected_language == Language.TURKISH:
                response = "Sana yardımcı olmak isterim! 🤔 Hangi konuda bilgi istiyorsun? Gezilecek yerler, ulaşım, yemek, etkinlikler... Her konuda yardımcı olabilirim!"
            else:
                response = "I'd love to help you! 🤔 What would you like to know about? Places to visit, transportation, food, events... I can help with anything!"
                
        else:  # SMALLTALK
            # Check for specific topics
            text_lower = user_input.lower()
            topic_detected = None
            
            for topic in self.topic_responses:
                topic_keywords = {
                    "weather": ["hava", "weather", "yağmur", "rain", "güneş", "sun"],
                    "food": ["yemek", "food", "restoran", "restaurant", "aç", "hungry"],
                    "places": ["gez", "visit", "yer", "place", "müze", "museum"],
                    "transport": ["ulaşım", "transport", "metro", "bus", "otobüs"]
                }
                
                if any(keyword in text_lower for keyword in topic_keywords.get(topic, [])):
                    topic_detected = topic
                    break
            
            if topic_detected:
                responses = self.topic_responses[topic_detected][detected_language]
                response = random.choice(responses)
            else:
                # General helpful response
                helpful_responses = self.casual_responses[detected_language]["how_are_you"]
                response = random.choice(helpful_responses)
        
        # Update context
        context.conversation_history.append({
            "user": user_input,
            "ai": response,
            "intent": intent.value,
            "language": detected_language.value,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges
        if len(context.conversation_history) > 10:
            context.conversation_history = context.conversation_history[-10:]
        
        return response, context
    
    def chat(self, message: str, session_context: Optional[UserContext] = None) -> str:
        """Main chat interface"""
        response, updated_context = self.generate_response(message, session_context)
        return response


# Global instance
enhanced_daily_talks = EnhancedBilingualDailyTalks()


def process_daily_talk(message: str, context: Optional[Dict] = None) -> str:
    """Process daily talk with enhanced bilingual system"""
    try:
        # Convert dict context to UserContext if needed
        user_context = None
        if context:
            user_context = UserContext()
            if 'language' in context:
                user_context.language = Language(context['language'])
            if 'conversation_history' in context:
                user_context.conversation_history = context['conversation_history']
        
        response = enhanced_daily_talks.chat(message, user_context)
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced daily talks: {e}")
        # Fallback response
        if any(tr_char in message for tr_char in 'çğıöşüÇĞİÖŞÜ'):
            return "Merhaba! 😊 Size nasıl yardımcı olabilirim?"
        else:
            return "Hello! 😊 How can I help you today?"


if __name__ == "__main__":
    # Test the system
    print("Testing Enhanced Bilingual Daily Talks System")
    print("=" * 60)
    
    # Test Turkish
    print("\n🇹🇷 Turkish Tests:")
    print("-" * 60)
    test_messages_tr = [
        "Merhaba",
        "Nasılsın?",
        "İstanbul'da ne yapabilirim?",
        "Teşekkür ederim",
        "Güle güle"
    ]
    
    context = UserContext()
    for msg in test_messages_tr:
        response, context = enhanced_daily_talks.generate_response(msg, context)
        print(f"\nUser: {msg}")
        print(f"AI: {response}")
    
    # Test English
    print("\n\n🇬🇧 English Tests:")
    print("-" * 60)
    test_messages_en = [
        "Hello",
        "How are you?",
        "What can I do in Istanbul?",
        "Thank you",
        "Goodbye"
    ]
    
    context = UserContext()
    for msg in test_messages_en:
        response, context = enhanced_daily_talks.generate_response(msg, context)
        print(f"\nUser: {msg}")
        print(f"AI: {response}")
