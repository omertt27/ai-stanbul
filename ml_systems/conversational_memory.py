"""
Conversation Memory & Personality System for AI-stanbul
Adds Istanbul-style personality, slang, and conversation context memory

Features:
- Turkish-English conversational mix
- Istanbul slang ("Abi", "kanka", "yavaş yavaş")
- Conversation history tracking
- Context-aware responses
- Personality traits (friendly, helpful, local)

Priority: MEDIUM - User Experience Enhancement
Timeline: 1-2 days
Cost: $0 (local implementation)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class IstanbulPersonality:
    """
    Istanbul-style conversational personality
    
    Characteristics:
    - Friendly and professional (like a helpful local guide)
    - Uses Turkish-English mix naturally
    - Light Istanbul expressions (not too casual)
    - Patient and helpful
    - Respectful but warm
    """
    
    # Professional but warm greetings
    SLANG = {
        'greeting': [
            "Merhaba! Welcome! 👋",
            "Hoş geldiniz! How can I help? 🙌",
            "Hello! Merhaba! 😊",
            "Welcome to Istanbul! Hoş bulduk! 🌟"
        ],
        'affirmative': [
            "Tamam, anladım! ✅",
            "Exactly! Aynen! 💯",
            "Of course! Tabii! 👍",
            "Certainly! 😊"
        ],
        'thinking': [
            "Let me check that for you... 🤔",
            "Bir dakika... 💭",
            "Good question! 🧐",
            "Let me see what I can find... 📝"
        ],
        'suggestion': [
            "I recommend:",
            "Here's my suggestion:",
            "You might try:",
            "The best option would be:"
        ],
        'encouragement': [
            "Yavaş yavaş - take it step by step! 😊",
            "Kolay gelsin! I'll guide you through it. 💪",
            "No problem, I'm here to help! 🤝",
            "We'll figure this out together! 🙌"
        ],
        'farewell': [
            "Görüşürüz! See you! 👋",
            "İyi günler! Have a great day! ☀️",
            "Please visit again! 🌟",
            "Take care! 💙"
        ]
    }
    
    # Professional Istanbul expressions
    EXPRESSIONS = {
        'slow_down': "Yavaş yavaş - no need to rush! 😊",
        'no_problem': "No problem at all! 👍",
        'lets_see': "Let me check what we have... 🔍",
        'trust_me': "I know Istanbul well - trust me! 💪",
        'local_tip': "Local tip 💎:",
        'be_careful': "Please be careful! ⚠️",
        'enjoy': "İyi eğlenceler! Enjoy! 🎉",
        'have_fun': "Have a wonderful time! 😄"
    }
    
    # Contextual responses (professional tone)
    CONTEXT_RESPONSES = {
        'first_time': "First time in Istanbul? Hoş geldiniz - welcome! 🌟",
        'returning': "Welcome back! It's great to have you here again! 🤗",
        'confused': "It can be confusing at first, but I'll help you navigate! 😊",
        'lost': "Don't worry, let me help you find your way! 📍",
        'hungry': "Looking for food? Istanbul has excellent options! 🍽️",
        'tired': "You seem tired. Let me find a nice place to rest! ☕",
        'excited': "I can see you're excited! Let's make it a great experience! 🎉",
        'weather_complaint': "Yes, Istanbul weather can be unpredictable! 🌧️😄"
    }
    
    @staticmethod
    def get_greeting(time_of_day: str = None) -> str:
        """Get contextual greeting based on time (professional but warm)"""
        import random
        
        if not time_of_day:
            hour = datetime.now().hour
            if 5 <= hour < 12:
                time_of_day = 'morning'
            elif 12 <= hour < 17:
                time_of_day = 'afternoon'
            elif 17 <= hour < 21:
                time_of_day = 'evening'
            else:
                time_of_day = 'night'
        
        greetings = {
            'morning': [
                "Good morning! Günaydın! ☀️",
                "İyi sabahlar! How can I help you today? 🌅",
                "Good morning! What brings you here? ☕"
            ],
            'afternoon': [
                "Good afternoon! İyi günler! 🌤️",
                "Hello! How can I assist you? 😊",
                "İyi günler! What can I help with? 🗺️"
            ],
            'evening': [
                "Good evening! İyi akşamlar! 🌆",
                "İyi akşamlar! How can I help? 🌃",
                "Good evening! What are you looking for? 🎭"
            ],
            'night': [
                "İyi geceler! Good evening! 🌙",
                "Hello! How can I assist you tonight? �",
                "Good evening! Planning something? 🎉"
            ]
        }
        
        return random.choice(greetings.get(time_of_day, IstanbulPersonality.SLANG['greeting']))
    
    @staticmethod
    def add_personality(text: str, context: Dict[str, Any] = None) -> str:
        """Add Istanbul personality to response"""
        import random
        
        # Don't modify if already has personality markers
        if any(marker in text for marker in ['abi', 'kanka', '👋', '😊', '💪']):
            return text
        
        # Add occasional Turkish expressions
        if random.random() < 0.3:  # 30% chance
            prefix = random.choice([
                "Bak şimdi: ",
                "Şöyle söyleyeyim: ",
                "Dinle beni: ",
                "İnan bana: "
            ])
            text = prefix + text
        
        # Add friendly emojis
        if random.random() < 0.5:  # 50% chance
            emojis = ['😊', '👍', '🌟', '💙', '🙌']
            text += f" {random.choice(emojis)}"
        
        return text


class ConversationMemory:
    """
    Tracks conversation history and context
    
    Features:
    - Recent message history
    - User preferences tracking
    - Context extraction
    - Session management
    """
    
    def __init__(self, max_history: int = 10, session_timeout: int = 30):
        """
        Initialize conversation memory
        
        Args:
            max_history: Max messages to remember
            session_timeout: Minutes before session expires
        """
        self.max_history = max_history
        self.session_timeout = timedelta(minutes=session_timeout)
        
        # Memory storage: user_id -> conversation data
        self.memory: Dict[str, Dict[str, Any]] = {}
    
    def add_message(
        self,
        user_id: str,
        message: str,
        response: str,
        intent: str = None,
        entities: Dict = None
    ):
        """Add message to conversation history"""
        
        # Initialize user memory if needed
        if user_id not in self.memory:
            self.memory[user_id] = {
                'history': [],
                'preferences': {},
                'context': {},
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'message_count': 0
            }
        
        user_memory = self.memory[user_id]
        
        # Update timestamp
        user_memory['last_seen'] = datetime.now()
        user_memory['message_count'] += 1
        
        # Add to history
        user_memory['history'].append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'response': response,
            'intent': intent,
            'entities': entities or {}
        })
        
        # Keep only recent history
        if len(user_memory['history']) > self.max_history:
            user_memory['history'] = user_memory['history'][-self.max_history:]
        
        # Extract and update context
        self._update_context(user_id, message, intent, entities)
    
    def _update_context(
        self,
        user_id: str,
        message: str,
        intent: str,
        entities: Dict
    ):
        """Extract context from message"""
        
        context = self.memory[user_id]['context']
        
        # Track locations mentioned
        if entities and 'location' in entities:
            if 'mentioned_locations' not in context:
                context['mentioned_locations'] = []
            location = entities['location']
            if location not in context['mentioned_locations']:
                context['mentioned_locations'].append(location)
                context['mentioned_locations'] = context['mentioned_locations'][-5:]  # Keep last 5
        
        # Track intents
        if intent:
            if 'recent_intents' not in context:
                context['recent_intents'] = []
            context['recent_intents'].append(intent)
            context['recent_intents'] = context['recent_intents'][-5:]
        
        # Detect user state
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['lost', 'confused', 'help', 'don\'t know']):
            context['state'] = 'confused'
        elif any(word in message_lower for word in ['hungry', 'eat', 'food', 'restaurant']):
            context['state'] = 'hungry'
        elif any(word in message_lower for word in ['tired', 'rest', 'sit', 'break']):
            context['state'] = 'tired'
        elif any(word in message_lower for word in ['excited', 'amazing', 'great', 'love']):
            context['state'] = 'excited'
        
        # Track time preferences
        if any(word in message_lower for word in ['night', 'evening', 'tonight']):
            context['time_preference'] = 'night'
        elif any(word in message_lower for word in ['morning', 'breakfast']):
            context['time_preference'] = 'morning'
    
    def get_context(self, user_id: str) -> Dict[str, Any]:
        """Get user's conversation context"""
        
        if user_id not in self.memory:
            return {}
        
        user_memory = self.memory[user_id]
        
        # Check session timeout
        if datetime.now() - user_memory['last_seen'] > self.session_timeout:
            # Session expired, reset context but keep preferences
            user_memory['context'] = {}
            user_memory['history'] = []
        
        return {
            'history': user_memory['history'],
            'context': user_memory['context'],
            'preferences': user_memory['preferences'],
            'is_new_user': user_memory['message_count'] <= 2,
            'session_length': user_memory['message_count']
        }
    
    def get_personality_context(self, user_id: str) -> Dict[str, Any]:
        """Get context for personality adaptation"""
        
        context = self.get_context(user_id)
        
        return {
            'is_first_time': context.get('is_new_user', True),
            'user_state': context.get('context', {}).get('state'),
            'recent_topics': context.get('context', {}).get('recent_intents', []),
            'session_length': context.get('session_length', 0)
        }
    
    def clear_session(self, user_id: str):
        """Clear user's session"""
        if user_id in self.memory:
            self.memory[user_id]['history'] = []
            self.memory[user_id]['context'] = {}


class ConversationalEnhancer:
    """
    Main conversation enhancement system
    Combines personality + memory for natural conversations
    """
    
    def __init__(self):
        self.personality = IstanbulPersonality()
        self.memory = ConversationMemory()
    
    def enhance_response(
        self,
        user_id: str,
        message: str,
        base_response: str,
        intent: str = None,
        entities: Dict = None
    ) -> str:
        """
        Enhance response with personality and context
        
        Args:
            user_id: User identifier
            message: User's message
            base_response: System's base response
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Enhanced response with personality
        """
        
        # Get conversation context
        context = self.memory.get_personality_context(user_id)
        
        # Start with greeting for new users
        enhanced = base_response
        if context['is_first_time'] and context['session_length'] <= 1:
            greeting = self.personality.get_greeting()
            enhanced = f"{greeting}\n\n{enhanced}"
        
        # Add contextual expressions based on user state
        if context.get('user_state'):
            state = context['user_state']
            if state in self.personality.CONTEXT_RESPONSES:
                expression = self.personality.CONTEXT_RESPONSES[state]
                enhanced = f"{expression}\n\n{enhanced}"
        
        # Add personality
        enhanced = self.personality.add_personality(enhanced, context)
        
        # Store in memory
        self.memory.add_message(user_id, message, enhanced, intent, entities)
        
        return enhanced
    
    def get_conversational_prompts(self, user_id: str) -> List[str]:
        """
        Get conversation-aware prompts for LLM
        
        Returns list of context hints for better responses
        """
        
        context = self.memory.get_context(user_id)
        prompts = []
        
        # Add personality instructions
        prompts.append(
            "You are KAM, a friendly Istanbul local. Use casual Turkish-English mix. "
            "Occasional slang like 'abi' (bro), 'kanka' (buddy) is natural. "
            "Be warm, helpful, and patient."
        )
        
        # Add user context
        if context.get('is_new_user'):
            prompts.append("User is new to Istanbul - be extra welcoming and explain clearly.")
        else:
            prompts.append(f"Continuing conversation (message #{context.get('session_length', 0)})")
        
        # Add location context
        mentioned = context.get('context', {}).get('mentioned_locations', [])
        if mentioned:
            prompts.append(f"Previously discussed locations: {', '.join(mentioned[-3:])}")
        
        # Add user state
        state = context.get('context', {}).get('state')
        if state:
            prompts.append(f"User seems {state} - adjust tone accordingly.")
        
        return prompts


# Singleton instance
_conversational_enhancer = None


def get_conversational_enhancer() -> ConversationalEnhancer:
    """Get or create conversational enhancer singleton"""
    global _conversational_enhancer
    
    if _conversational_enhancer is None:
        _conversational_enhancer = ConversationalEnhancer()
    
    return _conversational_enhancer


if __name__ == "__main__":
    """Test conversational system"""
    print("🗣️ Testing Istanbul Conversational System\n")
    print("=" * 60)
    
    enhancer = ConversationalEnhancer()
    
    # Simulate conversation
    user_id = "test_user_123"
    
    conversations = [
        {
            'message': "Hi, I just arrived in Istanbul",
            'response': "Welcome! Let me help you get around the city.",
            'intent': 'greeting'
        },
        {
            'message': "I'm looking for good restaurants in Kadıköy",
            'response': "Kadıköy has amazing local spots. Try the Çarşı market area.",
            'intent': 'restaurant',
            'entities': {'location': 'Kadıköy'}
        },
        {
            'message': "I'm a bit confused about the metro system",
            'response': "No worries! The metro is easy once you know the basics.",
            'intent': 'transportation'
        },
        {
            'message': "What should I do tonight?",
            'response': "Tons of options! İstiklal Street is buzzing with nightlife.",
            'intent': 'recommendation'
        }
    ]
    
    for i, conv in enumerate(conversations, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}:")
        print(f"User: {conv['message']}")
        print(f"\nBase Response: {conv['response']}")
        
        # Enhance response
        enhanced = enhancer.enhance_response(
            user_id=user_id,
            message=conv['message'],
            base_response=conv['response'],
            intent=conv.get('intent'),
            entities=conv.get('entities')
        )
        
        print(f"\nEnhanced Response:\n{enhanced}")
    
    # Show memory
    print(f"\n{'='*60}")
    print("Conversation Memory:")
    print(f"{'='*60}")
    context = enhancer.memory.get_context(user_id)
    print(f"Session length: {context['session_length']} messages")
    print(f"Is new user: {context['is_new_user']}")
    print(f"Mentioned locations: {context.get('context', {}).get('mentioned_locations', [])}")
    print(f"Recent intents: {context.get('context', {}).get('recent_intents', [])}")
    print(f"User state: {context.get('context', {}).get('state', 'neutral')}")
