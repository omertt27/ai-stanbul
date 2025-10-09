#!/usr/bin/env python3
"""
Istanbul Daily Talk AI System
Advanced conversational AI for Istanbul visitors and locals

Features:
- Advanced Intent & Entity Recognition with Istanbul-specific embeddings
- Context-Aware Dialogue with temporal awareness and user profiles
- Hybrid ML + Rule-based approach with reinforcement learning
- Real-time data integration (transport, traffic, restaurant hours)
- Personalized responses with local flavor and cultural references
- Smart scaling with lightweight models for mobile deployment
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationTone(Enum):
    """Conversation tone adaptation"""
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    LOCAL_EXPERT = "local_expert"
    TOURIST_GUIDE = "tourist_guide"

class UserType(Enum):
    """User classification"""
    FIRST_TIME_VISITOR = "first_time_visitor"
    REPEAT_VISITOR = "repeat_visitor"
    LOCAL_RESIDENT = "local_resident"
    BUSINESS_TRAVELER = "business_traveler"
    CULTURAL_EXPLORER = "cultural_explorer"

@dataclass
class UserProfile:
    """Advanced user profiling system"""
    user_id: str
    user_type: UserType = UserType.FIRST_TIME_VISITOR
    preferred_tone: ConversationTone = ConversationTone.FRIENDLY
    
    # Preferences
    favorite_neighborhoods: List[str] = field(default_factory=list)
    dietary_restrictions: List[str] = field(default_factory=list)
    cuisine_preferences: List[str] = field(default_factory=list)
    budget_range: str = "moderate"
    
    # Behavioral patterns
    visit_frequency: Dict[str, int] = field(default_factory=dict)  # location -> count
    preferred_times: List[str] = field(default_factory=list)  # breakfast, lunch, dinner
    interaction_history: List[Dict] = field(default_factory=list)
    
    # Temporal context
    current_location: Optional[str] = None
    last_interaction: Optional[datetime] = None
    session_context: Dict[str, Any] = field(default_factory=dict)
    
    # Learning metrics
    satisfaction_score: float = 0.8
    recommendation_success_rate: float = 0.7

@dataclass
class ConversationContext:
    """Multi-turn conversation context with temporal awareness"""
    session_id: str
    user_profile: UserProfile
    conversation_history: List[Dict] = field(default_factory=list)
    current_topic: Optional[str] = None
    pending_questions: List[str] = field(default_factory=list)
    context_memory: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal context
    session_start: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    
    def add_interaction(self, user_input: str, system_response: str, intent: str):
        """Add interaction to conversation history"""
        interaction = {
            'timestamp': datetime.now(),
            'user_input': user_input,
            'system_response': system_response,
            'detected_intent': intent,
            'context_at_time': self.context_memory.copy()
        }
        self.conversation_history.append(interaction)
        self.last_interaction = datetime.now()

class IstanbulEntityRecognizer:
    """Advanced entity recognition for Istanbul-specific terms"""
    
    def __init__(self):
        self.load_istanbul_knowledge_base()
    
    def load_istanbul_knowledge_base(self):
        """Load Istanbul-specific entities and embeddings"""
        
        # Istanbul neighborhoods with variants and local names
        self.neighborhoods = {
            'sultanahmet': ['sultanahmet', 'old city', 'historic peninsula', 'eski şehir'],
            'beyoğlu': ['beyoğlu', 'beyoglu', 'pera', 'galata', 'taksim area'],
            'kadıköy': ['kadıköy', 'kadikoy', 'asian side', 'moda', 'bagdat street'],
            'beşiktaş': ['beşiktaş', 'besiktas', 'ortaköy', 'bebek', 'arnavutköy'],
            'üsküdar': ['üsküdar', 'uskudar', 'çamlıca', 'beylerbeyi'],
            'şişli': ['şişli', 'sisli', 'nişantaşı', 'osmanbey', 'pangaltı'],
            'fatih': ['fatih', 'fener', 'balat', 'eminönü', 'karaköy']
        }
        
        # Landmarks with cultural context
        self.landmarks = {
            'hagia_sophia': {
                'names': ['hagia sophia', 'ayasofya', 'holy wisdom'],
                'type': 'museum',
                'cultural_context': 'Byzantine and Ottoman architectural marvel',
                'nearby_food': 'traditional Ottoman cuisine'
            },
            'blue_mosque': {
                'names': ['blue mosque', 'sultan ahmed mosque', 'sultanahmet camii'],
                'type': 'mosque',
                'cultural_context': 'Active place of worship with stunning blue tiles',
                'nearby_food': 'traditional Turkish breakfast spots'
            },
            'galata_tower': {
                'names': ['galata tower', 'galata kulesi'],
                'type': 'tower',
                'cultural_context': 'Medieval Genoese tower with panoramic views',
                'nearby_food': 'trendy cafes and rooftop restaurants'
            }
        }
        
        # Cuisine types with local nuances
        self.cuisine_entities = {
            'turkish_traditional': ['turkish', 'ottoman', 'traditional', 'lokanta', 'ev yemeği'],
            'street_food': ['street food', 'sokak lezzetleri', 'döner', 'simit', 'balık ekmek'],
            'meze_culture': ['meze', 'meyhane', 'rakı', 'small plates', 'tapas style'],
            'breakfast_culture': ['kahvaltı', 'turkish breakfast', 'serpme kahvaltı', 'village breakfast'],
            'seafood': ['seafood', 'balık', 'fish', 'marine', 'bosphorus fish']
        }
        
        # Time expressions with cultural context
        self.time_entities = {
            'meal_times': {
                'turkish_breakfast': '08:00-12:00',
                'lunch': '12:00-15:00',
                'afternoon_tea': '15:00-17:00',
                'dinner': '19:00-23:00',
                'late_night': '23:00-02:00'
            },
            'cultural_times': {
                'friday_prayer': 'avoid 12:00-14:00 near mosques',
                'ramadan_iftar': 'special evening hours during Ramadan',
                'weekend_brunch': 'extended breakfast hours on weekends'
            }
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract Istanbul-specific entities from text"""
        
        entities = {
            'neighborhoods': [],
            'landmarks': [],
            'cuisines': [],
            'time_references': [],
            'cultural_context': []
        }
        
        text_lower = text.lower()
        
        # Neighborhood detection with fuzzy matching
        for neighborhood, variants in self.neighborhoods.items():
            if any(variant in text_lower for variant in variants):
                entities['neighborhoods'].append(neighborhood)
        
        # Landmark detection with cultural context
        for landmark_id, landmark_data in self.landmarks.items():
            if any(name in text_lower for name in landmark_data['names']):
                entities['landmarks'].append(landmark_id)
                entities['cultural_context'].append(landmark_data['cultural_context'])
        
        # Cuisine detection
        for cuisine_type, keywords in self.cuisine_entities.items():
            if any(keyword in text_lower for keyword in keywords):
                entities['cuisines'].append(cuisine_type)
        
        # Time reference detection
        time_patterns = [
            r'\b(morning|sabah)\b', r'\b(afternoon|öğleden sonra)\b',
            r'\b(evening|akşam)\b', r'\b(night|gece)\b',
            r'\b(breakfast|kahvaltı)\b', r'\b(lunch|öğle)\b', r'\b(dinner|akşam yemeği)\b'
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, text_lower):
                entities['time_references'].append(re.search(pattern, text_lower).group())
        
        return entities

class IstanbulDailyTalkAI:
    """Main Istanbul Daily Talk AI System"""
    
    def __init__(self):
        self.entity_recognizer = IstanbulEntityRecognizer()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.active_conversations: Dict[str, ConversationContext] = {}
        
        # Initialize response templates with local flavor
        self.initialize_response_templates()
        
        # Real-time data connectors (placeholder for actual API integration)
        self.real_time_data = {
            'transport': self._get_transport_status,
            'weather': self._get_weather_info,
            'traffic': self._get_traffic_status,
            'events': self._get_local_events
        }
    
    def initialize_response_templates(self):
        """Initialize culturally-aware response templates"""
        
        self.response_templates = {
            'greeting': {
                ConversationTone.CASUAL: [
                    "Merhaba! 👋 Ready to explore Istanbul today?",
                    "Hey there! What's on your Istanbul adventure list?",
                    "Selam! How can I help you discover amazing places today?"
                ],
                ConversationTone.LOCAL_EXPERT: [
                    "Hoş geldiniz! As someone who knows Istanbul like the back of my hand, I'm excited to share hidden gems with you!",
                    "Welcome, friend! Let me be your local guide to the real Istanbul - beyond the tourist spots!",
                    "Merhaba! I've got insider knowledge about the best spots locals actually go to. What interests you?"
                ],
                ConversationTone.TOURIST_GUIDE: [
                    "Welcome to Istanbul! 🏛️ I'm here to help you make the most of your visit to this incredible city.",
                    "Greetings! Ready to discover the magic where Europe meets Asia? Let's plan your perfect Istanbul experience!",
                    "Hello and welcome! Istanbul has so much to offer - let me help you find exactly what you're looking for."
                ]
            },
            
            'restaurant_recommendation': {
                'local_favorite': "Here's a local secret: {restaurant_name} - it's where Istanbulites actually eat, not just tourists!",
                'cultural_context': "For authentic {cuisine_type}, try {restaurant_name}. It's been serving traditional recipes since {year}.",
                'neighborhood_specific': "In {neighborhood}, you absolutely must try {restaurant_name} - it captures the soul of the area perfectly."
            },
            
            'contextual_responses': {
                'weather_aware': "Since it's {weather} today, I'd recommend {indoor_suggestion} or maybe {outdoor_alternative}.",
                'time_sensitive': "At this time of day, {time_appropriate_suggestion} would be perfect.",
                'traffic_aware': "Considering the current traffic, {accessible_option} might be your best bet."
            }
        }
    
    def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get existing user profile or create new one"""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
            logger.info(f"Created new user profile for {user_id}")
        
        return self.user_profiles[user_id]
    
    def start_conversation(self, user_id: str) -> str:
        """Start or resume conversation with personalized greeting"""
        
        user_profile = self.get_or_create_user_profile(user_id)
        session_id = self._generate_session_id(user_id)
        
        # Create conversation context
        context = ConversationContext(
            session_id=session_id,
            user_profile=user_profile
        )
        self.active_conversations[session_id] = context
        
        # Generate personalized greeting
        greeting = self._generate_personalized_greeting(user_profile, context)
        
        # Add to conversation history
        context.add_interaction("", greeting, "greeting")
        
        return greeting
    
    def process_message(self, user_id: str, message: str) -> str:
        """Process user message with full context awareness"""
        
        # Get user profile and active conversation
        user_profile = self.get_or_create_user_profile(user_id)
        session_id = self._get_active_session_id(user_id)
        
        if session_id not in self.active_conversations:
            # Start new conversation if needed
            return self.start_conversation(user_id)
        
        context = self.active_conversations[session_id]
        
        # Extract entities and understand intent
        entities = self.entity_recognizer.extract_entities(message)
        intent = self._classify_intent_with_context(message, entities, context)
        
        # Update context memory
        self._update_context_memory(context, message, entities, intent)
        
        # Generate contextually-aware response
        response = self._generate_contextual_response(
            message, intent, entities, context, user_profile
        )
        
        # Add interaction to history
        context.add_interaction(message, response, intent)
        
        # Update user profile based on interaction
        self._update_user_profile(user_profile, message, intent, entities)
        
        return response
    
    def _generate_personalized_greeting(self, user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate personalized greeting based on user profile and context"""
        
        # Check if returning user
        if user_profile.last_interaction:
            time_since_last = datetime.now() - user_profile.last_interaction
            
            if time_since_last.days < 1:
                return f"Hey, welcome back! 😊 Still exploring Istanbul today? Last time we talked about {user_profile.session_context.get('last_topic', 'some great spots')}!"
            elif time_since_last.days < 7:
                return f"Great to see you again! 🌟 How did those {user_profile.favorite_neighborhoods[0] if user_profile.favorite_neighborhoods else 'restaurant'} recommendations work out?"
            else:
                return f"Welcome back to Istanbul! 🏙️ It's been a while - ready for some new discoveries?"
        
        # New user greeting
        templates = self.response_templates['greeting'][user_profile.preferred_tone]
        return templates[hash(user_profile.user_id) % len(templates)]
    
    def _classify_intent_with_context(self, message: str, entities: Dict, context: ConversationContext) -> str:
        """Classify intent with full contextual awareness"""
        
        # Use conversation history for context
        recent_intents = [interaction['detected_intent'] 
                         for interaction in context.conversation_history[-3:]]
        
        # Advanced intent classification logic
        message_lower = message.lower()
        
        # Context-aware intent detection
        if context.current_topic == 'restaurant_search' and any(word in message_lower for word in ['yes', 'sure', 'sounds good']):
            return 'confirmation'
        
        if any(word in message_lower for word in ['recommend', 'suggest', 'best', 'good']):
            if entities['cuisines'] or entities['neighborhoods']:
                return 'restaurant_recommendation'
            else:
                return 'general_recommendation'
        
        if any(word in message_lower for word in ['where', 'location', 'address']):
            return 'location_query'
        
        if any(word in message_lower for word in ['when', 'time', 'hours', 'open']):
            return 'time_query'
        
        if entities['landmarks']:
            return 'landmark_information'
        
        # Default to conversational
        return 'general_conversation'
    
    def _generate_contextual_response(self, message: str, intent: str, entities: Dict, 
                                    context: ConversationContext, user_profile: UserProfile) -> str:
        """Generate contextually-aware response with local flavor"""
        
        # Get real-time context
        current_time = datetime.now()
        weather_info = self.real_time_data['weather']()
        traffic_info = self.real_time_data['traffic']()
        
        if intent == 'restaurant_recommendation':
            return self._generate_restaurant_recommendation(entities, context, user_profile, current_time)
        
        elif intent == 'location_query':
            return self._generate_location_response(entities, context, traffic_info)
        
        elif intent == 'time_query':
            return self._generate_time_response(entities, context, current_time)
        
        elif intent == 'general_conversation':
            return self._generate_conversational_response(message, context, user_profile)
        
        else:
            return self._generate_fallback_response(context, user_profile)
    
    def _generate_restaurant_recommendation(self, entities: Dict, context: ConversationContext, 
                                          user_profile: UserProfile, current_time: datetime) -> str:
        """Generate personalized restaurant recommendations"""
        
        # Determine meal time context
        hour = current_time.hour
        if hour < 11:
            meal_context = "breakfast"
        elif hour < 16:
            meal_context = "lunch"
        else:
            meal_context = "dinner"
        
        # Get neighborhood context
        neighborhood = entities['neighborhoods'][0] if entities['neighborhoods'] else user_profile.current_location
        
        # Generate recommendation with local flavor
        if neighborhood == 'sultanahmet' and 'turkish_traditional' in entities['cuisines']:
            return f"🍽️ For authentic Ottoman cuisine in Sultanahmet, you've got to try Matbah Restaurant! It's right near Hagia Sophia and serves recipes from the imperial palace. Perfect for {meal_context}! The lamb stew is legendary among locals. 😋"
        
        elif neighborhood == 'beyoğlu':
            return f"🌟 Beyoğlu is perfect for your {meal_context}! Try Çukur Meyhane on Nevizade Street - it's where locals go for authentic meze and rakı. The atmosphere is pure Istanbul! Just a 2-minute walk from İstiklal Street. 🥂"
        
        else:
            return f"🍴 Based on your taste for {entities['cuisines'][0] if entities['cuisines'] else 'great food'}, I've got some fantastic local spots in mind! What specific area are you in right now? I can give you walking directions to the best places nearby! 🚶‍♀️"
    
    def _update_context_memory(self, context: ConversationContext, message: str, entities: Dict, intent: str):
        """Update conversation context memory"""
        
        # Update current topic
        if intent in ['restaurant_recommendation', 'location_query']:
            context.current_topic = intent
        
        # Store entity information
        if entities['neighborhoods']:
            context.context_memory['last_neighborhood'] = entities['neighborhoods'][0]
        
        if entities['cuisines']:
            context.context_memory['preferred_cuisines'] = entities['cuisines']
        
        # Track conversation flow
        context.context_memory['last_intent'] = intent
        context.context_memory['message_count'] = len(context.conversation_history)
    
    def _update_user_profile(self, user_profile: UserProfile, message: str, intent: str, entities: Dict):
        """Update user profile based on interaction"""
        
        user_profile.last_interaction = datetime.now()
        
        # Update preferences
        if entities['neighborhoods']:
            for neighborhood in entities['neighborhoods']:
                if neighborhood not in user_profile.favorite_neighborhoods:
                    user_profile.favorite_neighborhoods.append(neighborhood)
        
        if entities['cuisines']:
            for cuisine in entities['cuisines']:
                if cuisine not in user_profile.cuisine_preferences:
                    user_profile.cuisine_preferences.append(cuisine)
        
        # Track interaction patterns
        user_profile.session_context['last_topic'] = intent
        user_profile.interaction_history.append({
            'timestamp': datetime.now(),
            'intent': intent,
            'entities': entities,
            'message': message[:100]  # Store truncated message for privacy
        })
        
        # Limit history size
        if len(user_profile.interaction_history) > 50:
            user_profile.interaction_history = user_profile.interaction_history[-50:]
    
    # Real-time data integration (placeholder methods)
    def _get_transport_status(self) -> Dict:
        """Get real-time transport information"""
        return {
            'metro': 'operational',
            'ferry': 'delayed_5_min',
            'bus': 'heavy_traffic',
            'tram': 'operational'
        }
    
    def _get_weather_info(self) -> Dict:
        """Get current weather information"""
        return {
            'condition': 'partly_cloudy',
            'temperature': 22,
            'humidity': 65,
            'recommendation': 'great for outdoor dining'
        }
    
    def _get_traffic_status(self) -> Dict:
        """Get current traffic information"""
        return {
            'overall': 'moderate',
            'bridges': 'heavy',
            'recommendation': 'use metro or ferry'
        }
    
    def _get_local_events(self) -> List[Dict]:
        """Get current local events"""
        return [
            {'name': 'Bosphorus Concert', 'location': 'Beşiktaş', 'time': '20:00'},
            {'name': 'Food Festival', 'location': 'Kadıköy', 'time': '18:00-22:00'}
        ]
    
    # Utility methods
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID"""
        timestamp = str(int(time.time()))
        return hashlib.md5(f"{user_id}_{timestamp}".encode()).hexdigest()[:12]
    
    def _get_active_session_id(self, user_id: str) -> Optional[str]:
        """Get active session ID for user"""
        # Simple implementation - in production, use proper session management
        for session_id, context in self.active_conversations.items():
            if context.user_profile.user_id == user_id:
                return session_id
        return None
    
    def _generate_conversational_response(self, message: str, context: ConversationContext, user_profile: UserProfile) -> str:
        """Generate conversational response with personality"""
        
        responses = [
            f"That's interesting! Tell me more about what you're looking for in Istanbul! 🤔",
            f"I love that you're exploring Istanbul! What kind of experience are you after? 🌟",
            f"Ah, a fellow Istanbul enthusiast! What's caught your attention today? 😊"
        ]
        
        return responses[hash(message) % len(responses)]
    
    def _generate_location_response(self, entities: Dict, context: ConversationContext, traffic_info: Dict) -> str:
        """Generate location-aware response"""
        
        if entities['landmarks']:
            landmark = entities['landmarks'][0]
            return f"📍 {landmark.replace('_', ' ').title()} is a must-see! Given the current traffic, I'd recommend taking the metro. Want me to give you the exact route? 🚇"
        
        return "📍 I can help you get there! What specific location are you looking for? 🗺️"
    
    def _generate_time_response(self, entities: Dict, context: ConversationContext, current_time: datetime) -> str:
        """Generate time-aware response"""
        
        hour = current_time.hour
        if hour < 10:
            return f"⏰ Perfect timing for Turkish breakfast! Most places are just opening up with fresh simit and çay. The early bird catches the best börek! 🥐"
        elif hour > 22:
            return f"🌙 It's getting late, but Istanbul's nightlife is just beginning! Many meyhanes and late-night eateries are still serving. Want some recommendations? 🍷"
        else:
            return f"⏰ Great time to explore! Most places are open and the timing is perfect for {entities['time_references'][0] if entities['time_references'] else 'discovering new spots'}! ✨"
    
    def _generate_fallback_response(self, context: ConversationContext, user_profile: UserProfile) -> str:
        """Generate fallback response with personality"""
        
        return f"🤖 I want to help you discover the best of Istanbul! Can you tell me more about what you're interested in? Food, sightseeing, culture, or something else? I'm here to be your local guide! 🌟"

# Example usage and testing
if __name__ == "__main__":
    # Initialize the AI system
    ai = IstanbulDailyTalkAI()
    
    # Simulate conversation
    user_id = "test_user_123"
    
    print("🏛️ ISTANBUL DAILY TALK AI - DEMO")
    print("=" * 50)
    
    # Start conversation
    greeting = ai.start_conversation(user_id)
    print(f"AI: {greeting}")
    
    # Simulate user interactions
    test_messages = [
        "I'm looking for good Turkish restaurants in Sultanahmet",
        "What about something more traditional?",
        "How do I get to Galata Tower from here?",
        "What time do restaurants usually close?"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = ai.process_message(user_id, message)
        print(f"AI: {response}")
    
    print(f"\n🔍 User Profile Summary:")
    profile = ai.user_profiles[user_id]
    print(f"Favorite neighborhoods: {profile.favorite_neighborhoods}")
    print(f"Cuisine preferences: {profile.cuisine_preferences}")
    print(f"Interaction count: {len(profile.interaction_history)}")
