"""
Istanbul Daily Talk AI - Personality Enhancement Module
======================================================

This module adds rich personality, small talk capabilities, and cultural depth
to the Istanbul Daily Talk AI system.

Features:
- Friendly, warm Istanbul local persona
- Weather and traffic small talk
- Turkish expressions and cultural phrases
- Humor and local idioms
- Sports and daily life conversations
"""

import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class PersonalityTrait(Enum):
    """Personality traits for the Istanbul AI"""
    FRIENDLY = "friendly"
    HELPFUL = "helpful"
    LOCAL_EXPERT = "local_expert"
    HUMOROUS = "humorous"
    WARM = "warm"
    ENTHUSIASTIC = "enthusiastic"


class ConversationMood(Enum):
    """Current conversation mood"""
    CASUAL = "casual"
    FRIENDLY = "friendly"
    INFORMATIVE = "informative"
    ENTHUSIASTIC = "enthusiastic"
    EMPATHETIC = "empathetic"


class IstanbulPersonality:
    """
    Istanbul AI Personality Enhancement System
    Provides warm, friendly, local expert persona with cultural depth
    """
    
    def __init__(self):
        self.logger = logger
        self.current_mood = ConversationMood.FRIENDLY
        self.traits = [
            PersonalityTrait.FRIENDLY,
            PersonalityTrait.HELPFUL,
            PersonalityTrait.LOCAL_EXPERT,
            PersonalityTrait.HUMOROUS
        ]
        
        # Initialize personality data
        self._init_greetings()
        self._init_turkish_expressions()
        self._init_small_talk_patterns()
        self._init_humor_database()
        self._init_cultural_phrases()
        
        logger.info("🎭 Istanbul Personality Enhancement Module initialized")
    
    def _init_greetings(self):
        """Initialize greeting responses with personality"""
        self.greetings = {
            'hello': [
                "Merhaba! 👋 Welcome to Istanbul! I'm your local guide ready to help you explore this amazing city!",
                "Hey there! 😊 Hoş geldiniz (welcome)! What brings you to our beautiful Istanbul today?",
                "Hello! So happy to have you here! Istanbul has so much to offer - where should we start?",
                "Merhaba friend! 🌟 Ready to discover the magic of Istanbul? I know all the best spots!",
                "Hi! Welcome to Istanbul - the city where East meets West! How can I help you today?"
            ],
            'good_morning': [
                "Günaydın! ☀️ Good morning! Istanbul mornings are magical - perfect time to explore before the crowds!",
                "Good morning! 🌅 Did you know Istanbul mornings smell like fresh simit and Turkish tea? Beautiful!",
                "Günaydın! The city is waking up beautifully today. What adventures are you planning?",
                "Morning! ☕ Time for a Turkish breakfast - çay (tea) and simit! What can I help you with today?"
            ],
            'good_evening': [
                "İyi akşamlar! 🌆 Good evening! Istanbul evenings are spectacular - the city lights up beautifully!",
                "Good evening! 🌙 The Bosphorus looks amazing at this hour. Planning some evening adventures?",
                "İyi akşamlar! Perfect time for a sunset walk or rooftop dinner. What are you thinking?",
                "Evening! 🌃 Istanbul's nightlife is just getting started. Need recommendations?"
            ],
            'how_are_you': [
                "I'm doing great, thank you! 😊 Living in Istanbul keeps me energized - there's always something exciting happening!",
                "Harika (wonderful)! Just had my morning çay and ready to help you explore Istanbul!",
                "I'm fantastic! The weather in Istanbul today is beautiful, and I'm excited to help you discover the city!",
                "Çok iyiyim (very good)! Istanbul is my home and I love sharing it with visitors like you!"
            ]
        }
    
    def _init_turkish_expressions(self):
        """Initialize Turkish expressions and phrases"""
        self.turkish_expressions = {
            'welcome': [
                "Hoş geldiniz! (Welcome!)",
                "Hoş bulduk! (We're happy you're here!)",
                "Safa geldiniz! (Welcome, honored guest!)"
            ],
            'enjoy_meal': [
                "Afiyet olsun! (Enjoy your meal!)",
                "Şerefe! (Cheers!)",
                "Ellerine sağlık! (Health to your hands - to the cook!)"
            ],
            'thank_you': [
                "Teşekkür ederim! (Thank you!)",
                "Çok teşekkürler! (Thank you very much!)",
                "Sağ ol! (Thanks - informal)"
            ],
            'goodbye': [
                "Görüşürüz! (See you!)",
                "Hoşça kal! (Goodbye!)",
                "Allah'a ısmarladık! (Goodbye - formal)",
                "İyi günler! (Have a good day!)"
            ],
            'praise': [
                "Harika! (Wonderful!)",
                "Çok güzel! (Very beautiful!)",
                "Mükemmel! (Perfect!)",
                "Bravo! (Well done!)",
                "Çok şık! (Very stylish/cool!)"
            ],
            'common_phrases': [
                "Ağzının tadını biliyorsun! (You have good taste!)",
                "İyi şanslar! (Good luck!)",
                "Kolay gelsin! (May it be easy - said to someone working)",
                "Geçmiş olsun! (Get well soon / Hope it passes)",
                "Hayırlı olsun! (May it be blessed - congratulations)"
            ]
        }
    
    def _init_small_talk_patterns(self):
        """Initialize small talk conversation patterns"""
        self.small_talk = {
            'weather': {
                'sunny': [
                    "☀️ Beautiful day in Istanbul! Perfect weather for exploring - maybe a Bosphorus cruise?",
                    "Gorgeous sunny day! Istanbul sparkles in this weather. The parks and waterfront are lovely!",
                    "What a day! ☀️ This is perfect Istanbul weather - not too hot, just right for walking around!"
                ],
                'rainy': [
                    "🌧️ A bit rainy today - typical Istanbul! Perfect time for museum visits and cozy cafes.",
                    "Rainy Istanbul has its own charm! 💧 Great day for indoor attractions and Turkish coffee.",
                    "Don't worry about the rain - Istanbul looks beautiful in any weather! Plus, fewer tourists! 😊"
                ],
                'hot': [
                    "🌞 Quite warm today! Stay hydrated and maybe head to the Bosphorus for some breeze.",
                    "Hot Istanbul summer! 🥵 Time for ice cream, shade, and waterfront spots.",
                    "Typical Istanbul summer heat! Perfect for ferry rides with that cool breeze."
                ],
                'cold': [
                    "🧣 Chilly today! Bundle up and enjoy hot salep or Turkish tea while exploring.",
                    "Cold Istanbul winter! ❄️ Perfect weather for warm köfte and cozy restaurants.",
                    "Brr, a bit cold! But Istanbul looks magical in winter - fewer crowds, beautiful atmosphere!"
                ]
            },
            'traffic': {
                'general': [
                    "🚗 Ah, Istanbul traffic - legendary! 😄 It's part of the city's character. Metro is your friend!",
                    "Traffic can be... interesting here! 🚕 That's why we have ferries - scenic AND practical!",
                    "Istanbul traffic is an adventure itself! 😅 Good thing we have great public transport!"
                ],
                'rush_hour': [
                    "⏰ Rush hour in Istanbul is no joke! Best time to grab a çay and wait it out, honestly.",
                    "Peak traffic time! 🚦 If you're traveling now, metro and tram are lifesavers.",
                    "Classic Istanbul rush hour! This is when locals become philosophers about patience. 😄"
                ],
                'weekend': [
                    "Weekend traffic to Bosphorus areas can be busy! 🚗 But worth it for the views!",
                    "Weekends, everyone heads to the coast! Ferry is the smart (and scenic) choice.",
                    "Weekend Istanbul! Bit crowded but the energy is amazing - everyone's out enjoying the city!"
                ]
            },
            'sports': {
                'football': [
                    "⚽ Football is HUGE here! Galatasaray, Fenerbahçe, Beşiktaş - the passion is real!",
                    "You like football? Istanbul lives and breathes it! Derby days are electric! ⚡",
                    "Three major teams, one city, endless passion! ⚽ Football in Istanbul is cultural!"
                ],
                'general': [
                    "Istanbul loves sports! Football mainly, but also basketball and volleyball.",
                    "Sports fan? You're in the right city! The stadium atmosphere here is legendary! 🏟️"
                ]
            },
            'daily_life': {
                'tea_culture': [
                    "☕ Çay (tea) is life in Istanbul! We drink it all day, every day. It's our social glue!",
                    "Turkish tea culture is special - it's not just a drink, it's an invitation to sit and chat! 🫖",
                    "See those tulip-shaped glasses everywhere? That's çay! Join the tradition!"
                ],
                'food_culture': [
                    "🍽️ Food is central to Istanbul life! We take our meals seriously - and joyfully!",
                    "Istanbul food culture is about sharing, talking, and taking time. Rush? What's that? 😊",
                    "Eating here is an experience! Good food + good company = Istanbul happiness formula!"
                ],
                'ferry_culture': [
                    "⛴️ Ferries aren't just transport - they're our therapy! Çay on the ferry = Istanbul zen.",
                    "Taking the ferry with a simit and çay is peak Istanbul living! Try it!",
                    "Locals use ferries as relaxation time. It's meditation with a view! 🌊"
                ]
            }
        }
    
    def _init_humor_database(self):
        """Initialize humor and funny observations"""
        self.humor = {
            'istanbul_life': [
                "Living in Istanbul: You're never lost, you're just on an adventure! 😄",
                "Istanbul rule: If GPS says 10 minutes, add 20. If it says walk, definitely walk! 🚶‍♂️",
                "Istanbul has two speeds: Crazy busy or magnificently peaceful. No in-between! 😅",
                "The city where 'just around the corner' might mean a 30-minute walk up a hill! 🏔️",
                "In Istanbul, 'I'm on my way' really means 'I just woke up' 😂 But we're friendly!"
            ],
            'traffic_jokes': [
                "Istanbul traffic: Where parallel parking becomes extreme sport! 🚗",
                "We measure distance in time here: 'How far?' 'One tea away!' ☕",
                "Istanbul traffic jam? Perfect time to make new friends in the next car! 😄",
                "Rush hour in Istanbul: When you question all your life choices but also enjoy the music! 🎵"
            ],
            'tourist_observations': [
                "First time seeing someone try to pronounce 'Üsküdar'? Priceless! 😄 (It's EWSS-koo-dar!)",
                "Tourists think 15 million people is crazy. We call it 'cozy'! 🤗",
                "The look when tourists realize Istanbul is on TWO continents! 🌍🌏 Mind. Blown.",
                "Istanbul: Where 'ancient ruins' and 'trendy clubs' are neighbors! 🏛️🎵"
            ],
            'food_humor': [
                "Turkish breakfast: When you need a table extension for all the dishes! 🍳🧀🥖",
                "Portion sizes in Istanbul: Yes.",
                "Diet in Istanbul? That's cute. Have you tried baklava? 🍯",
                "Turkish hospitality level: Force-feeding you until you promise to come back! 😊"
            ],
            'cultural_quirks': [
                "In Istanbul, everyone knows the BEST restaurant. And they're all different! 😄",
                "Turkish aunties will feed you, marry you off, and redecorate your house. All in one visit! 👵",
                "Istanbul: Where 'tea break' is a legitimate life philosophy! ☕",
                "We have more cats than people. And they run the city. Respect the cats! 🐱"
            ]
        }
    
    def _init_cultural_phrases(self):
        """Initialize cultural conversation templates"""
        self.cultural_templates = {
            'hospitality': [
                "Turkish hospitality is legendary! We say 'Misafir Allah'ın konuğudur' - guests are God's guests!",
                "In Istanbul, being a good host is an art form. You'll feel this everywhere you go! 🏠",
                "We love hosting! Don't be surprised if locals invite you for çay - that's just how we are! ☕"
            ],
            'family_values': [
                "Family is everything in Turkish culture. Sundays are for family meals that last hours! 👨‍👩‍👧‍👦",
                "You'll see multi-generational families everywhere - we like staying close! ❤️",
                "Turkish families are big, loud, loving, and involved in EVERYTHING! It's wonderful! 😊"
            ],
            'respect_culture': [
                "Respect for elders is huge here. We kiss their hands and call everyone 'Teyze' (aunt) or 'Amca' (uncle)!",
                "In Istanbul, you'll hear 'Abi' (brother) and 'Abla' (sister) a lot - it's respectful and friendly!",
                "We show respect through small gestures - offering seats, helping elders, sharing food."
            ],
            'tea_ceremony': [
                "Tea in Istanbul isn't just a drink - it's a social ritual, a pause button, a friendship maker! 🫖",
                "We offer tea to guests, customers, strangers... it's our way of saying 'you matter'! ☕",
                "Turkish tea culture: Thin-waisted glasses, two sugar cubes, endless refills, deep conversations! ❤️"
            ],
            'mosque_etiquette': [
                "Visiting mosques? Cover shoulders and knees. Ladies, grab a headscarf at the entrance (free)! 🧕",
                "Remove shoes before entering mosques - you'll find shelves or bags provided. Easy! 👟",
                "Prayer times: Be respectful and quiet. But visitors are welcome outside prayer times! 🕌"
            ],
            'bargaining': [
                "Bargaining at Grand Bazaar is expected! It's not rude - it's tradition! Start at 50-60% of asking price. 🛍️",
                "Shopping tip: Walk away if price is too high. Sellers will often call you back with better offers! 😉",
                "Bargaining is a social dance here - smile, be friendly, have fun with it! It's theater! 🎭"
            ]
        }
    
    def get_greeting(self, query: str) -> Optional[str]:
        """
        Generate appropriate greeting response with personality
        
        Args:
            query: User's greeting query
            
        Returns:
            Personalized greeting response or None
        """
        query_lower = query.lower().strip()
        
        # Morning greetings
        if any(word in query_lower for word in ['good morning', 'günaydın', 'morning']):
            return random.choice(self.greetings['good_morning'])
        
        # Evening greetings
        elif any(word in query_lower for word in ['good evening', 'iyi akşamlar', 'evening']):
            return random.choice(self.greetings['good_evening'])
        
        # How are you
        elif any(phrase in query_lower for phrase in ['how are you', 'how are u', "how're you", 'nasılsın', 'nasılsınız']):
            return random.choice(self.greetings['how_are_you'])
        
        # General hello/hi
        elif any(word in query_lower for word in ['hello', 'hi', 'hey', 'merhaba', 'selam']):
            return random.choice(self.greetings['hello'])
        
        return None
    
    def get_weather_talk(self, weather_condition: str = 'general') -> str:
        """
        Generate weather-related small talk
        
        Args:
            weather_condition: Current weather (sunny, rainy, hot, cold)
            
        Returns:
            Weather-related conversational response
        """
        condition = weather_condition.lower()
        
        if condition in self.small_talk['weather']:
            return random.choice(self.small_talk['weather'][condition])
        else:
            # Default weather talk
            return random.choice([
                "Istanbul weather can be unpredictable! That's part of its charm! 🌦️",
                "The weather here reflects the city - dynamic and ever-changing! 😊",
                "Whatever the weather, Istanbul always has something beautiful to offer!"
            ])
    
    def get_traffic_talk(self, time_context: str = 'general') -> str:
        """
        Generate traffic-related humor and advice
        
        Args:
            time_context: Time context (general, rush_hour, weekend)
            
        Returns:
            Traffic-related conversational response
        """
        context = time_context.lower()
        
        if context in self.small_talk['traffic']:
            return random.choice(self.small_talk['traffic'][context])
        else:
            return random.choice(self.small_talk['traffic']['general'])
    
    def get_daily_life_talk(self, topic: str = 'general') -> str:
        """
        Generate daily life conversation
        
        Args:
            topic: Daily life topic (tea_culture, food_culture, ferry_culture)
            
        Returns:
            Daily life conversational response
        """
        if topic in self.small_talk['daily_life']:
            return random.choice(self.small_talk['daily_life'][topic])
        
        # General daily life talk
        return random.choice([
            "Istanbul life is beautiful chaos! Fast-paced but with moments of peace. Balance! ⚖️",
            "Living here means embracing contradictions: ancient and modern, East and West! 🌍",
            "Daily life in Istanbul: Çay breaks, ferry rides, food adventures, and wonderful people! ❤️"
        ])
    
    def add_turkish_expression(self, context: str = 'general') -> str:
        """
        Add appropriate Turkish expression based on context
        
        Args:
            context: Conversation context (welcome, enjoy_meal, thank_you, etc.)
            
        Returns:
            Turkish expression with translation
        """
        if context in self.turkish_expressions:
            return random.choice(self.turkish_expressions[context])
        
        # Random praise/common phrase as default
        return random.choice(self.turkish_expressions['praise'] + self.turkish_expressions['common_phrases'])
    
    def get_humor(self, category: str = 'random') -> str:
        """
        Get humorous observation or joke
        
        Args:
            category: Humor category (istanbul_life, traffic_jokes, food_humor, etc.)
            
        Returns:
            Humorous statement
        """
        if category == 'random':
            # Pick random category
            all_humor = []
            for jokes in self.humor.values():
                all_humor.extend(jokes)
            return random.choice(all_humor)
        elif category in self.humor:
            return random.choice(self.humor[category])
        else:
            return random.choice(self.humor['istanbul_life'])
    
    def get_cultural_insight(self, topic: str = 'general') -> str:
        """
        Get cultural insight or tip
        
        Args:
            topic: Cultural topic (hospitality, family_values, etc.)
            
        Returns:
            Cultural insight
        """
        if topic in self.cultural_templates:
            return random.choice(self.cultural_templates[topic])
        
        # Random cultural insight
        all_insights = []
        for insights in self.cultural_templates.values():
            all_insights.extend(insights)
        return random.choice(all_insights)
    
    def add_personality_to_response(self, response: str, context: str = 'general') -> str:
        """
        Enhance response with personality elements
        
        Args:
            response: Base response text
            context: Response context
            
        Returns:
            Enhanced response with personality
        """
        # Add Turkish expression occasionally (20% chance)
        if random.random() < 0.2:
            expression = self.add_turkish_expression()
            response = f"{response}\n\n{expression}"
        
        # Add cultural insight occasionally (15% chance)
        if random.random() < 0.15:
            insight = self.get_cultural_insight()
            response = f"{response}\n\n💡 **Cultural Tip**: {insight}"
        
        # Add emoji personality touch
        if '!' in response and random.random() < 0.3:
            # Already enthusiastic, enhance it
            response = response.replace('!', '! 🌟', 1)
        
        return response
    
    def handle_thanks(self, query: str) -> str:
        """
        Handle thank you messages with warm personality
        
        Args:
            query: User's thank you message
            
        Returns:
            Warm, friendly response
        """
        responses = [
            "Rica ederim! (You're welcome!) 😊 So happy I could help! Enjoy Istanbul!",
            "You're very welcome! 🌟 That's what we're here for! Have an amazing time in Istanbul!",
            "Bir şey değil! (It's nothing!) ❤️ Always happy to help visitors discover Istanbul!",
            "My pleasure! 😊 Come back if you need anything else - I know this city inside out!",
            "Sağ ol! (Thanks to you!) 🌟 Enjoy exploring Istanbul - there's so much beauty here!",
            "You're welcome! 😊 Remember: In Istanbul, there are no strangers, only friends we haven't met yet!",
            "Memnun oldum! (I'm happy to help!) 🎉 Have the best time in our beautiful city!"
        ]
        return random.choice(responses)
    
    def handle_goodbye(self, query: str) -> str:
        """
        Handle goodbye messages with warm sendoff
        
        Args:
            query: User's goodbye message
            
        Returns:
            Warm farewell message
        """
        goodbyes = [
            "Görüşürüz! (See you!) 👋 Have a wonderful time in Istanbul! Come back anytime!",
            "Hoşça kal! Safe travels! 🌟 Istanbul will always be here welcoming you back!",
            "İyi günler! (Have a good day!) ☀️ Enjoy every moment in Istanbul! You'll love it!",
            "Goodbye friend! 👋 May Istanbul give you beautiful memories! Hayırlı yolculuklar! (Good travels!)",
            "See you soon! 🌟 Remember: Once you visit Istanbul, part of your heart stays here! ❤️",
            "Allah'a ısmarladık! 👋 May your Istanbul adventure be unforgettable! Tekrar görüşmek üzere! (Until we meet again!)",
            "Take care! 😊 Istanbul was lucky to have you! Come back soon - we'll be waiting! 🏙️"
        ]
        return random.choice(goodbyes)


# Create global instance
personality = IstanbulPersonality()


def enhance_response_with_personality(response: str, context: str = 'general') -> str:
    """
    Global function to enhance any response with personality
    
    Args:
        response: Original response
        context: Conversation context
        
    Returns:
        Enhanced response
    """
    return personality.add_personality_to_response(response, context)


# Convenience functions for easy access
def get_greeting(query: str) -> Optional[str]:
    """Get personalized greeting"""
    return personality.get_greeting(query)


def get_weather_talk(condition: str = 'general') -> str:
    """Get weather small talk"""
    return personality.get_weather_talk(condition)


def get_traffic_talk(time_context: str = 'general') -> str:
    """Get traffic small talk"""
    return personality.get_traffic_talk(time_context)


def handle_thanks(query: str) -> str:
    """Handle thank you messages"""
    return personality.handle_thanks(query)


def handle_goodbye(query: str) -> str:
    """Handle goodbye messages"""
    return personality.handle_goodbye(query)


def add_turkish_expression(context: str = 'general') -> str:
    """Add Turkish expression"""
    return personality.add_turkish_expression(context)


def get_humor(category: str = 'random') -> str:
    """Get humorous observation"""
    return personality.get_humor(category)


def get_cultural_insight(topic: str = 'general') -> str:
    """Get cultural insight"""
    return personality.get_cultural_insight(topic)
