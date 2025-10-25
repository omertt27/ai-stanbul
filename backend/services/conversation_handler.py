"""
Conversation Handler
Handle casual conversations, greetings, thanks, and planning queries
"""

from typing import Dict, List, Optional
import random
import re


class ConversationHandler:
    """Handle casual conversations and context-aware responses"""
    
    def __init__(self):
        self.greeting_responses = self._load_greeting_responses()
        self.thanks_responses = self._load_thanks_responses()
        self.time_recommendations = self._load_time_recommendations()
        self.help_responses = self._load_help_responses()
    
    def _load_greeting_responses(self) -> Dict[str, List[str]]:
        """Load greeting response templates"""
        return {
            'merhaba': [
                "🇹🇷 Merhaba! Welcome to Istanbul! I'm your AI guide. How can I help you explore this amazing city?",
                "👋 Merhaba! Ready to discover Istanbul? Ask me anything about restaurants, attractions, or getting around!",
                "🌟 Merhaba! I'm here to make your Istanbul experience unforgettable. What would you like to know?",
            ],
            'hello': [
                "👋 Hello! Welcome to Istanbul! I'm here to help you have an amazing experience. What would you like to explore?",
                "🌟 Hi! Great to meet you! I can help with restaurants, attractions, transportation, and local tips. What interests you?",
                "😊 Hello there! I'm your Istanbul AI guide. Ready to discover the city? Just ask me anything!",
                "🎉 Hey! Welcome! I know Istanbul inside and out. What can I help you with today?",
            ],
            'hi': [
                "👋 Hi there! Welcome to Istanbul! What brings you to this beautiful city?",
                "😊 Hi! Ready to explore Istanbul? I'm here to help with anything you need!",
                "🌟 Hey! Let's make your Istanbul visit amazing. What would you like to know?",
            ],
            'good_morning': [
                "☀️ Good morning! Perfect time to start exploring Istanbul! What are your plans for today?",
                "🌅 Good morning! Istanbul is waking up beautifully. How can I help make your day amazing?",
            ],
            'good_evening': [
                "🌙 Good evening! Istanbul comes alive at night! Looking for dinner spots or evening activities?",
                "✨ Good evening! The city is magical after dark. What would you like to explore tonight?",
            ],
            'planning': [
                "📋 Excited to help plan your Istanbul adventure! Let me know:\n\n" +
                "• How many days will you be here?\n" +
                "• What are you most interested in? (food, history, culture, nightlife)\n" +
                "• Any specific neighborhoods you want to explore?\n\n" +
                "Or just ask me specific questions - I'm here to help! 😊"
            ]
        }
    
    def _load_thanks_responses(self) -> List[str]:
        """Load thank you response templates"""
        return [
            "😊 You're welcome! Enjoy exploring Istanbul! Feel free to ask if you need anything else.",
            "🎉 My pleasure! Have a wonderful time in Istanbul! Don't hesitate to ask more questions.",
            "👍 Happy to help! Wishing you an amazing Istanbul experience! Ask anytime! 🌟",
            "💫 You're very welcome! Hope you have an incredible time in Istanbul!",
            "✨ Glad I could help! Enjoy every moment in this beautiful city!",
            "🙏 You're welcome! Have a fantastic Istanbul adventure! I'm here if you need more tips!",
            "😊 No problem at all! Make the most of your Istanbul visit! Come back with more questions anytime!",
            "🌟 Always happy to help! Enjoy Istanbul - you're going to love it!",
        ]
    
    def _load_time_recommendations(self) -> Dict[str, Dict]:
        """Load time-based itinerary recommendations"""
        return {
            '1_day': {
                'itinerary': 'Sultanahmet Focus - The Highlights',
                'realistic': "You'll see the main highlights but it will be rushed",
                'spots': [
                    'Hagia Sophia (1.5 hours)',
                    'Blue Mosque (45 min)',
                    'Grand Bazaar (1 hour)',
                    'Topkapı Palace (2 hours)',
                ],
                'tip': 'Start early (8 AM) to beat crowds. Skip the palace if tight on time.',
                'transportation': 'Walk between sites in Sultanahmet - everything is close!'
            },
            '2_days': {
                'itinerary': 'Sultanahmet + Beyoğlu - Old & New Istanbul',
                'realistic': 'Good introduction to both historical and modern Istanbul',
                'day1': {
                    'title': 'Day 1: Historical Peninsula',
                    'spots': ['Hagia Sophia', 'Blue Mosque', 'Basilica Cistern', 'Grand Bazaar'],
                    'tip': 'Lunch at local restaurant near Grand Bazaar'
                },
                'day2': {
                    'title': 'Day 2: Modern Istanbul + Bosphorus',
                    'spots': ['Galata Tower', 'Istiklal Street', 'Taksim Square', 'Bosphorus Ferry'],
                    'tip': 'Take sunset Bosphorus ferry for best views'
                },
                'transportation': 'Day 1: Walk. Day 2: Tram + Metro + Ferry'
            },
            '3_days': {
                'itinerary': 'Comprehensive Istanbul - Best Balance',
                'realistic': "Perfect! You'll experience the real Istanbul with good pacing",
                'day1': {
                    'title': 'Day 1: Sultanahmet',
                    'spots': ['Hagia Sophia', 'Blue Mosque', 'Topkapı Palace', 'Basilica Cistern'],
                    'tip': 'Book Topkapı Palace ticket online to skip lines'
                },
                'day2': {
                    'title': 'Day 2: Beyoğlu + Bosphorus',
                    'spots': ['Galata Tower', 'Istiklal Street', 'Taksim', 'Bosphorus Cruise'],
                    'tip': 'Have dinner in Karaköy - great food scene'
                },
                'day3': {
                    'title': 'Day 3: Asian Side',
                    'spots': ['Kadıköy Market', 'Moda Walk', 'Çamlıca Hill', 'Maiden\'s Tower'],
                    'tip': 'Less touristy, more authentic local experience'
                },
                'transportation': 'Use Istanbulkart for all public transport'
            },
            '4_days': {
                'itinerary': 'Extended Istanbul - Deeper Exploration',
                'realistic': 'Excellent! Time to explore neighborhoods and hidden gems',
                'highlights': [
                    'Day 1: Sultanahmet basics',
                    'Day 2: Beyoğlu + Bosphorus',
                    'Day 3: Asian side',
                    'Day 4: Balat/Fener + Ortaköy OR Princes\' Islands day trip'
                ],
                'tip': 'You have time for a Turkish bath experience and evening activities'
            },
            '5_days': {
                'itinerary': 'Deep Dive Istanbul - Like a Local',
                'realistic': 'Ideal! Time to explore like a local with no rushing',
                'highlights': [
                    'All major areas covered thoroughly',
                    'Day trips possible (Princes\' Islands, Şile, Bursa)',
                    'Time for local experiences (cooking class, hammam, etc.)',
                    'Explore hidden neighborhoods',
                    'Evening entertainment (rooftop bars, live music)'
                ],
                'tip': 'Add a day trip to Princes\' Islands or Bursa for variety'
            },
            '7_days': {
                'itinerary': 'Ultimate Istanbul Experience',
                'realistic': 'Perfect for slow travel and deep cultural immersion',
                'highlights': [
                    'All major attractions without rushing',
                    'Multiple day trips',
                    'Deep dive into specific neighborhoods',
                    'Cultural experiences (workshops, classes)',
                    'Time for spontaneous discoveries',
                    'Experience local daily life'
                ],
                'tip': 'Consider splitting time: 5 days Istanbul + 2 days Cappadocia or Bursa'
            }
        }
    
    def _load_help_responses(self) -> Dict[str, str]:
        """Load help and confused responses"""
        return {
            'help': """🤔 **Not sure what to ask? I can help with:**

**🍽️ Food & Dining:**
- Restaurant recommendations (by cuisine, area, budget)
- Traditional Turkish dishes
- Street food spots
- Budget-friendly eats

**🏛️ Places & Attractions:**
- Must-see attractions
- Hidden gems
- Museums and historical sites
- Neighborhoods to explore

**🚇 Transportation:**
- How to get around
- Metro/tram/ferry routes
- Airport transfers
- Istanbulkart information

**🗺️ Planning:**
- Itinerary suggestions
- Time-based recommendations
- Day trips
- Route planning

**💰 Budget:**
- Free things to do
- Budget-friendly options
- Cost estimates

**🌤️ Weather & Timing:**
- Weather-appropriate activities
- Best times to visit places
- Seasonal recommendations

**💡 Local Tips:**
- Insider advice
- Cultural etiquette
- Safety tips
- Turkish phrases

Just ask me anything! For example: "Best restaurants in Beyoğlu" or "Things to do on a rainy day" 😊""",
            
            'confused': """😊 **I'm here to help! Let me make it easier:**

Try asking me:
• "Best restaurants near [location]"
• "What to see in [neighborhood]"
• "How to get from [A] to [B]"
• "Free things to do in Istanbul"
• "Hidden gems in Istanbul"
• "Best places for [cuisine] food"
• "What to do with [X] days in Istanbul"

Or just tell me what you're interested in! 🌟""",
            
            'not_understood': """🤔 **I want to help, but I'm not quite sure what you're looking for.**

Could you try:
• Being more specific? (e.g., "seafood restaurants in Karaköy")
• Asking about a particular topic? (food, attractions, transportation)
• Rephrasing your question?

Type "help" to see what I can do! 😊"""
        }
    
    def detect_greeting(self, query: str) -> Optional[str]:
        """Detect greeting type in query"""
        query_lower = query.lower().strip()
        
        # Turkish greetings
        if any(word in query_lower for word in ['merhaba', 'selam', 'selamlar']):
            return 'merhaba'
        
        # Time-based greetings
        if any(phrase in query_lower for phrase in ['good morning', 'günaydın']):
            return 'good_morning'
        if any(phrase in query_lower for phrase in ['good evening', 'iyi akşamlar', 'good night']):
            return 'good_evening'
        
        # Planning queries
        if any(word in query_lower for word in ['plan', 'planning', 'itinerary', 'schedule']):
            if any(word in query_lower for word in ['help', 'need', 'want', 'how to']):
                return 'planning'
        
        # Basic greetings
        if query_lower in ['hi', 'hey', 'hello', 'yo', 'hola']:
            if len(query.split()) <= 2:  # Short greeting
                return 'hi' if query_lower in ['hi', 'hey', 'yo'] else 'hello'
        
        # Longer hello patterns
        if query_lower.startswith(('hello ', 'hi ', 'hey ')):
            return 'hello'
        
        return None
    
    def detect_thanks(self, query: str) -> bool:
        """Detect thank you in query"""
        query_lower = query.lower().strip()
        
        thanks_patterns = [
            'thank', 'thanks', 'thx', 'ty', 'thank you',
            'teşekkür', 'teşekkürler', 'sağol', 'sağolun',
            'appreciate', 'grateful', 'awesome', 'perfect',
            'great', 'excellent', 'helpful', 'nice'
        ]
        
        # Check if it's a short thanks message
        words = query_lower.split()
        if len(words) <= 4:
            return any(pattern in query_lower for pattern in thanks_patterns)
        
        # Check if thanks is prominent in longer messages
        if len(words) <= 10:
            return any(query_lower.startswith(pattern) for pattern in thanks_patterns[:8])
        
        return False
    
    def detect_help_request(self, query: str) -> Optional[str]:
        """Detect help/confused queries"""
        query_lower = query.lower().strip()
        
        # Direct help request
        if query_lower in ['help', 'help me', 'yardım', 'yardım et']:
            return 'help'
        
        # Confused/not sure
        confused_patterns = [
            "don't know", "not sure", "confused", "what can you",
            "what do you", "can you help", "i need help"
        ]
        if any(pattern in query_lower for pattern in confused_patterns):
            return 'confused'
        
        # Too vague
        if query_lower in ['?', '??', 'hm', 'hmm', 'what', 'huh']:
            return 'not_understood'
        
        return None
    
    def detect_time_duration(self, query: str) -> Optional[str]:
        """Extract time duration from query (e.g., '3 days')"""
        query_lower = query.lower()
        
        # Pattern: number + day(s)
        patterns = [
            r'(\d+)\s*day',
            r'one\s+day',
            r'two\s+days',
            r'three\s+days',
            r'a\s+day',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                if 'one' in query_lower or 'a day' in query_lower:
                    return '1_day'
                elif 'two' in query_lower:
                    return '2_days'
                elif 'three' in query_lower:
                    return '3_days'
                else:
                    try:
                        days = int(match.group(1))
                        if days == 1:
                            return '1_day'
                        elif days == 2:
                            return '2_days'
                        elif days == 3:
                            return '3_days'
                        elif days == 4:
                            return '4_days'
                        elif days in [5, 6]:
                            return '5_days'
                        elif days >= 7:
                            return '7_days'
                    except (ValueError, IndexError):
                        pass
        
        return None
    
    def handle_greeting(self, query: str) -> str:
        """Generate context-aware greeting response"""
        greeting_type = self.detect_greeting(query)
        
        if greeting_type and greeting_type in self.greeting_responses:
            responses = self.greeting_responses[greeting_type]
            return random.choice(responses)
        
        # Default greeting
        return random.choice(self.greeting_responses['hello'])
    
    def handle_thanks(self) -> str:
        """Generate friendly acknowledgment"""
        return random.choice(self.thanks_responses)
    
    def handle_help(self, help_type: str = 'help') -> str:
        """Generate help response"""
        return self.help_responses.get(help_type, self.help_responses['help'])
    
    def recommend_duration(self, query: str) -> Optional[str]:
        """Generate personalized itinerary based on available time"""
        duration = self.detect_time_duration(query)
        
        if not duration:
            return None
        
        recommendation = self.time_recommendations.get(duration)
        if not recommendation:
            return None
        
        # Format response
        response = f"**{recommendation['itinerary']}**\n\n"
        response += f"✅ **Realistic Assessment:** {recommendation['realistic']}\n\n"
        
        if duration in ['1_day', '2_days', '3_days']:
            if duration == '1_day':
                response += "**Suggested Itinerary:**\n"
                for spot in recommendation['spots']:
                    response += f"• {spot}\n"
                response += f"\n💡 **Pro Tip:** {recommendation['tip']}\n"
                response += f"🚇 **Transportation:** {recommendation['transportation']}\n"
            
            elif duration == '2_days':
                response += f"**{recommendation['day1']['title']}:**\n"
                for spot in recommendation['day1']['spots']:
                    response += f"• {spot}\n"
                response += f"💡 Tip: {recommendation['day1']['tip']}\n\n"
                
                response += f"**{recommendation['day2']['title']}:**\n"
                for spot in recommendation['day2']['spots']:
                    response += f"• {spot}\n"
                response += f"💡 Tip: {recommendation['day2']['tip']}\n\n"
                
                response += f"🚇 **Transportation:** {recommendation['transportation']}\n"
            
            elif duration == '3_days':
                for day_key in ['day1', 'day2', 'day3']:
                    day = recommendation[day_key]
                    response += f"**{day['title']}:**\n"
                    for spot in day['spots']:
                        response += f"• {spot}\n"
                    response += f"💡 Tip: {day['tip']}\n\n"
                
                response += f"🚇 **Transportation:** {recommendation['transportation']}\n"
        
        else:
            # 4+ days
            response += "**Suggested Plan:**\n"
            for highlight in recommendation['highlights']:
                response += f"• {highlight}\n"
            response += f"\n💡 **Pro Tip:** {recommendation['tip']}\n"
        
        response += "\n\nWant specific recommendations for any part of your itinerary? Just ask! 😊"
        
        return response
    
    def is_conversational_query(self, query: str) -> bool:
        """Check if query is conversational"""
        return (
            self.detect_greeting(query) is not None or
            self.detect_thanks(query) or
            self.detect_help_request(query) is not None or
            self.detect_time_duration(query) is not None
        )
    
    def handle_conversation(self, query: str) -> Optional[str]:
        """Main handler for conversational queries"""
        # Check for greeting
        if self.detect_greeting(query):
            return self.handle_greeting(query)
        
        # Check for thanks
        if self.detect_thanks(query):
            return self.handle_thanks()
        
        # Check for help
        help_type = self.detect_help_request(query)
        if help_type:
            return self.handle_help(help_type)
        
        # Check for time-based planning
        duration_response = self.recommend_duration(query)
        if duration_response:
            return duration_response
        
        return None


# Singleton instance
_conversation_handler = None

def get_conversation_handler() -> ConversationHandler:
    """Get or create conversation handler instance"""
    global _conversation_handler
    if _conversation_handler is None:
        _conversation_handler = ConversationHandler()
    return _conversation_handler
