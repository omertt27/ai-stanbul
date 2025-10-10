#!/usr/bin/env python3
"""
Istanbul Daily Talk AI System - ENHANCED WITH DEEP LEARNING
Advanced conversational AI for Istanbul visitors and locals
NOW WITH UNLIMITED DEEP LEARNING CAPABILITIES FOR 10,000+ USERS!

🚀 ENHANCED FEATURES:
- Deep Learning Enhanced Conversational AI with Advanced Neural Networks
- English-Optimized Processing for Maximum Performance  
- Advanced Intent & Entity Recognition with Istanbul-specific embeddings
- Context-Aware Dialogue with temporal awareness and user profiles
- Multimodal Support (Text, Voice, Image) - ALL FREE!
- Real-time Learning and Adaptation - UNLIMITED!
- Personalized responses with local flavor and cultural references
- Advanced Analytics and Performance Monitoring - ALWAYS ON!
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
import asyncio

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced deep learning system
try:
    from deep_learning_enhanced_ai import DeepLearningEnhancedAI, ConversationMemory, EmotionalState
    DEEP_LEARNING_AVAILABLE = True
    logger.info("🧠 Deep Learning Enhanced AI System loaded successfully!")
except ImportError as e:
    logger.warning(f"Deep Learning system not available: {e}")
    DEEP_LEARNING_AVAILABLE = False

# Import multi-intent query handler for advanced restaurant queries
try:
    from multi_intent_query_handler import MultiIntentQueryHandler, IntentType
    MULTI_INTENT_AVAILABLE = True
    logger.info("🎯 Multi-Intent Query Handler loaded successfully!")
except ImportError as e:
    logger.warning(f"Multi-Intent Query Handler not available: {e}")
    MULTI_INTENT_AVAILABLE = False

# Import priority enhancements system
try:
    from istanbul_ai_priority_enhancements import (
        IstanbulAIPriorityEnhancements, 
        EnhancedUserProfile, 
        PersonalizationLevel,
        DeepLearningFeatureType
    )
    PRIORITY_ENHANCEMENTS_AVAILABLE = True
    logger.info("🚀 Priority Enhancements System loaded successfully!")
except ImportError as e:
    logger.warning(f"Priority Enhancements not available: {e}")
    PRIORITY_ENHANCEMENTS_AVAILABLE = False

# Import neighborhood guides system
try:
    from istanbul_neighborhood_guides_system import (
        IstanbulNeighborhoodGuidesSystem, 
        VisitorType, 
        NeighborhoodCharacter,
        BestVisitingTime
    )
    NEIGHBORHOOD_GUIDES_AVAILABLE = True
    logger.info("🏘️ Neighborhood Guides System loaded successfully!")
except ImportError as e:
    NEIGHBORHOOD_GUIDES_AVAILABLE = False
    logger.warning(f"⚠️ Neighborhood Guides System not available: {e}")

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
    """🚀 ENHANCED Istanbul Daily Talk AI System with Deep Learning
    
    NOW WITH UNLIMITED DEEP LEARNING CAPABILITIES FOR 10,000+ USERS!
    ✨ ALL PREMIUM FEATURES ENABLED FOR FREE!
    🇺🇸 ENGLISH-OPTIMIZED for maximum performance!
    """
    
    def __init__(self):
        # Initialize enhanced deep learning system
        if DEEP_LEARNING_AVAILABLE:
            self.deep_learning_ai = DeepLearningEnhancedAI()
            logger.info("🧠 Deep Learning Enhanced AI integrated successfully!")
            logger.info("🚀 UNLIMITED features enabled for 10,000+ users!")
        else:
            self.deep_learning_ai = None
            logger.warning("⚠️ Running in fallback mode without deep learning")
        
        # Original components
        self.entity_recognizer = IstanbulEntityRecognizer()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.active_conversations: Dict[str, ConversationContext] = {}
        
        # Initialize multi-intent query handler for advanced restaurant queries
        if MULTI_INTENT_AVAILABLE:
            self.multi_intent_handler = MultiIntentQueryHandler()
            logger.info("🎯 Multi-Intent Query Handler integrated successfully!")
        else:
            self.multi_intent_handler = None
            logger.warning("⚠️ Multi-Intent features disabled")
        
        # Initialize priority enhancements system
        if PRIORITY_ENHANCEMENTS_AVAILABLE:
            self.priority_enhancements = IstanbulAIPriorityEnhancements()
            logger.info("🚀 Priority Enhancements System integrated successfully!")
        else:
            self.priority_enhancements = None
            logger.warning("⚠️ Priority Enhancements features disabled")
        
        # Initialize neighborhood guides system
        if NEIGHBORHOOD_GUIDES_AVAILABLE:
            self.neighborhood_guides = IstanbulNeighborhoodGuidesSystem()
            logger.info("🏘️ Neighborhood Guides System integrated successfully!")
        else:
            self.neighborhood_guides = None
            logger.warning("⚠️ Neighborhood Guides features disabled")

        # Initialize response templates with local flavor
        self.initialize_response_templates()
        
        # Real-time data connectors (placeholder for actual API integration)
        self.real_time_data = {
            'transport': self._get_transport_status,
            'weather': self._get_weather_info,
            'traffic': self._get_traffic_status,
            'events': self._get_local_events
        }
        
        # Enhanced features tracking
        self.feature_usage_stats = {
            'deep_learning_queries': 0,
            'english_optimized_responses': 0,
            'multimodal_interactions': 0,
            'voice_interactions': 0,
            'personality_adaptations': 0,
            'cultural_context_additions': 0
        }
        
        logger.info("🎉 Enhanced Istanbul Daily Talk AI System initialized with ALL features!")
        if DEEP_LEARNING_AVAILABLE:
            logger.info("🌟 Deep Learning Features: UNLIMITED & FREE for all users!")
            logger.info("🎯 English Optimization: ACTIVE for maximum performance!")
            logger.info("🤖 Advanced Analytics: ALWAYS ON!")
            logger.info("🔄 Real-time Learning: ENABLED!")
    
    def initialize_response_templates(self):
        """Initialize culturally-aware response templates"""
        
        self.response_templates = {
            'greeting': {
                'casual': [
                    "Merhaba! 👋 Ready to explore Istanbul today?",
                    "Hey there! What's on your Istanbul adventure list?",
                    "Selam! How can I help you discover amazing places today?"
                ],
                'friendly': [
                    "Welcome to Istanbul! 😊 I'm excited to help you explore this amazing city!",
                    "Hello! Ready to discover the best of Istanbul? I'm here to help!",
                    "Hi there! What would you like to know about Istanbul today?"
                ],
                'local_expert': [
                    "Hoş geldiniz! As someone who knows Istanbul like the back of my hand, I'm excited to share hidden gems with you!",
                    "Welcome, friend! Let me be your local guide to the real Istanbul - beyond the tourist spots!",
                    "Merhaba! I've got insider knowledge about the best spots locals actually go to. What interests you?"
                ],
                'tourist_guide': [
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
        """🚀 ENHANCED Process user message with Deep Learning & English Optimization
        
        NOW WITH UNLIMITED DEEP LEARNING CAPABILITIES!
        """
        
        # Get user profile and active conversation
        user_profile = self.get_or_create_user_profile(user_id)
        session_id = self._get_active_session_id(user_id)
        
        if session_id not in self.active_conversations:
            # Start new conversation if needed
            return self.start_conversation(user_id)
        
        context = self.active_conversations[session_id]
        
        # 🏘️ PRIORITY: Check for neighborhood queries before deep learning (specialized system)
        if self._is_neighborhood_query(message):
            logger.info(f"🏘️ Processing neighborhood query for {user_id}")
            current_time = datetime.now()
            response = self._process_neighborhood_query(message, user_profile, current_time)
            context.add_interaction(message, response, "neighborhood_query")
            
            # Create proper entities dict for neighborhood queries
            entities = {
                'neighborhoods': [self._extract_neighborhood_from_message(message)] if self._extract_neighborhood_from_message(message) else [],
                'cuisines': [],
                'districts': [],
                'attractions': []
            }
            self._update_user_profile(user_profile, message, "neighborhood_query", entities)
            return response
        
        # 🧠 ENHANCED: Use Deep Learning AI if available
        if self.deep_learning_ai and DEEP_LEARNING_AVAILABLE:
            try:
                # Track usage stats
                self.feature_usage_stats['deep_learning_queries'] += 1
                
                # Use advanced deep learning processing
                response = asyncio.run(
                    self.deep_learning_ai.generate_english_optimized_response(
                        message, user_id, {'context': context.__dict__}
                    )
                )
                
                self.feature_usage_stats['english_optimized_responses'] += 1
                
                # Add cultural context for Istanbul-specific queries
                if any(word in message.lower() for word in ['istanbul', 'restaurant', 'food', 'travel', 'visit']):
                    cultural_context = self.deep_learning_ai.generate_english_cultural_context("dining")
                    if cultural_context and cultural_context not in response:
                        response += f"\n\n{cultural_context}"
                        self.feature_usage_stats['cultural_context_additions'] += 1
                
                # Update conversation context
                context.add_interaction(message, response, "deep_learning_enhanced")
                
                # Update user profile with deep learning insights
                if hasattr(self.deep_learning_ai, 'user_analytics') and user_id in self.deep_learning_ai.user_analytics:
                    dl_analytics = self.deep_learning_ai.get_user_analytics(user_id)
                    self._sync_deep_learning_profile(user_profile, dl_analytics)
                
                logger.info(f"🧠 Deep Learning response generated for {user_id}")
                return response
                
            except Exception as e:
                logger.warning(f"Deep Learning processing failed, using fallback: {e}")
                # Fall through to original processing
        
        # 🔄 FALLBACK: Original processing if deep learning unavailable
        logger.info(f"📝 Using traditional processing for {user_id}")
        
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
        tone_key = user_profile.preferred_tone.value if hasattr(user_profile.preferred_tone, 'value') else str(user_profile.preferred_tone).split('.')[-1].lower()
        templates = self.response_templates['greeting'].get(tone_key, self.response_templates['greeting']['friendly'])
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
        
        # ENHANCED: Use enhanced intent classification with attractions support
        enhanced_intent = self._enhance_intent_classification(message)
        
        # If enhanced classification found attraction-related intent, use it
        if enhanced_intent in ['attraction_query', 'cultural_query', 'family_activity', 'romantic_spot', 'hidden_gem']:
            return enhanced_intent
        
        # PRIORITY: Check for restaurant queries (comprehensive handling)
        if self._is_restaurant_query(message):
            return 'restaurant_query'
        
        # Check for neighborhood queries
        if self._is_neighborhood_query(message):
            return 'neighborhood_query'
        
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
        
        # Check for neighborhood-related queries
        if self._is_neighborhood_query(message):
            return 'neighborhood_query'
        
        # Default to conversational
        return 'general_conversation'
    
    def _generate_contextual_response(self, message: str, intent: str, entities: Dict, 
                                    context: ConversationContext, user_profile: UserProfile) -> str:
        """Generate contextually-aware response with local flavor"""
        
        # Get real-time context
        current_time = datetime.now()
        weather_info = self.real_time_data['weather']()
        traffic_info = self.real_time_data['traffic']()
        
        # 🎯 ENHANCED: Use Multi-Intent Query Handler for restaurant and attraction queries
        if intent in ['restaurant_query', 'restaurant_recommendation', 'attraction_query', 'place_recommendation', 'cultural_query', 'activity_planning'] and self.multi_intent_handler:
            try:
                logger.info(f"🎯 Using Multi-Intent Handler for: {message}")
                
                # Process through multi-intent handler
                multi_intent_result = self.multi_intent_handler.analyze_query(message)
                
                # Check if this is an attraction-related query by analyzing the message content
                is_attraction_query = any(keyword in message.lower() for keyword in [
                    'attraction', 'museum', 'palace', 'mosque', 'tower', 'monument', 'historic',
                    'visit', 'see', 'explore', 'sightseeing', 'cultural', 'heritage', 'landmark',
                    'places to go', 'what to see', 'worth visiting', 'must see', 'tourist',
                    'family friendly', 'romantic', 'hidden gem'
                ])
                
                # Handle attraction-specific queries
                if is_attraction_query and self.multi_intent_handler.attractions_system:
                    try:
                        # Use the attractions system to get recommendations
                        attraction_response = self.multi_intent_handler.handle_attraction_query(
                            multi_intent_result.primary_intent, message
                        )
                        
                        if attraction_response['status'] == 'success':
                            return self._format_attraction_response_text(attraction_response, user_profile, current_time)
                        else:
                            logger.warning(f"Attraction query failed: {attraction_response.get('message', 'Unknown error')}")
                    except Exception as e:
                        logger.warning(f"Attraction query processing failed: {e}")
                
                # Handle restaurant and general queries
                multi_intent_response = multi_intent_result.response_text or "I understand your query, let me help you find the perfect place!"
                
                # Add Istanbul-specific context and local flavor
                enhanced_response = self._enhance_multi_intent_response(
                    multi_intent_response, entities, user_profile, current_time
                )
                
                return enhanced_response
                
            except Exception as e:
                logger.warning(f"Multi-intent processing failed, using fallback: {e}")
                # Fall through to traditional recommendation
        
        if intent == 'restaurant_recommendation':
            return self._generate_restaurant_recommendation(entities, context, user_profile, current_time)
        
        elif intent == 'location_query':
            return self._generate_location_response(entities, context, traffic_info)
        
        elif intent == 'time_query':
            return self._generate_time_response(entities, context, current_time)
        
        elif intent == 'general_conversation':
            return self._generate_conversational_response(message, context, user_profile)
        
        elif intent == 'neighborhood_query':
            return self._process_neighborhood_query(message, user_profile, current_time)
        
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
            return f"🌙 Late night in Istanbul! Most restaurants close around 23:00, but there are always 24/7 döner places and some areas like Nevizade stay lively. Need specific late-night recommendations? 🌃"
    
    def _generate_fallback_response(self, context: ConversationContext, user_profile: UserProfile) -> str:
        """Generate fallback response when intent classification fails"""
        
        fallback_responses = [
            "🤔 I want to make sure I give you the best advice! Could you tell me a bit more about what you're looking for in Istanbul?",
            "🌟 Istanbul has so much to offer! Are you interested in food, sightseeing, culture, or something else specific?",
            "😊 I'm here to help you discover the best of Istanbul! What would you like to know more about?"
        ]
        
        # Add context-aware fallbacks
        if len(context.conversation_history) > 0:
            last_intent = context.conversation_history[-1].get('detected_intent')
            if last_intent == 'restaurant_recommendation':
                return "🍽️ Still thinking about food? I can suggest more restaurants or help with other Istanbul questions!"
            elif last_intent == 'location_query':
                return "📍 Need more location help? I can give you directions, nearby attractions, or transport info!"
        
        return fallback_responses[hash(user_profile.user_id) % len(fallback_responses)]

    def _format_attraction_response_text(self, attraction_response: Dict[str, Any], 
                                        user_profile: UserProfile, current_time: datetime) -> str:
        """Format attraction response with Istanbul-specific context and personality"""
        
        attractions = attraction_response.get('attractions', [])
        if not attractions:
            return "🏛️ I'd love to help you discover Istanbul's amazing attractions! Could you be more specific about what you're looking for? Museums, historic sites, parks, or hidden gems? 🌟"
        
        response_parts = []
        
        # Greeting based on time and query type
        hour = current_time.hour
        if hour < 12:
            greeting = "🌅 Good morning! Perfect time to explore Istanbul's treasures!"
        elif hour < 17:
            greeting = "☀️ Great afternoon for sightseeing!"
        else:
            greeting = "🌆 Evening adventures await in Istanbul!"
        
        response_parts.append(greeting)
        
        # Add personalized intro based on attractions found
        message = attraction_response.get('message', '')
        if 'family' in message.lower():
            response_parts.append("👨‍👩‍👧‍👦 Here are some fantastic family-friendly attractions:")
        elif 'romantic' in message.lower():
            response_parts.append("💕 Here are some romantic spots perfect for couples:")
        elif 'hidden' in message.lower():
            response_parts.append("🗝️ Here are some hidden gems that locals love:")
        elif 'cultural' in message.lower():
            response_parts.append("🏛️ Here are some amazing cultural and historical sites:")
        else:
            response_parts.append("🎯 Here are some wonderful attractions I recommend:")
        
        # Format top attractions (limit to 5 for readability)
        for i, attraction in enumerate(attractions[:5]):
            attraction_text = f"\n{i+1}. **{attraction['name']}** ({attraction['district']})"
            attraction_text += f"\n   📍 {attraction['description']}"
            
            # Add practical info
            if attraction.get('entrance_fee') and attraction['entrance_fee'] != 'free':
                attraction_text += f"\n   💰 {attraction['estimated_cost']}"
            else:
                attraction_text += f"\n   🆓 Free entry!"
                
            if attraction.get('duration'):
                attraction_text += f" | ⏱️ {attraction['duration']}"
            
            # Add opening hours if available
            if attraction.get('opening_hours'):
                hours = attraction['opening_hours']
                if isinstance(hours, dict):
                    if 'daily' in hours:
                        attraction_text += f"\n   🕐 Open: {hours['daily']}"
                    elif 'monday_saturday' in hours:
                        attraction_text += f"\n   🕐 Mon-Sat: {hours['monday_saturday']}"
                else:
                    attraction_text += f"\n   🕐 {hours}"
            
            # Add special features
            features = []
            if attraction.get('is_family_friendly'):
                features.append("👨‍👩‍👧‍👦 Family-friendly")
            if attraction.get('is_romantic'):
                features.append("💕 Romantic")
            if attraction.get('is_hidden_gem'):
                features.append("🗝️ Hidden gem")
            
            if features:
                attraction_text += f"\n   ✨ {' | '.join(features)}"
            
            # Add top practical tip
            if attraction.get('practical_tips') and len(attraction['practical_tips']) > 0:
                tip = attraction['practical_tips'][0]
                attraction_text += f"\n   💡 Tip: {tip}"
            
            response_parts.append(attraction_text)
        
        # Add contextual recommendations
        if len(attractions) > 5:
            response_parts.append(f"\n✨ Plus {len(attractions) - 5} more amazing places to explore!")
        
        # Add practical advice based on time
        if hour < 10:
            response_parts.append(f"\n🌟 Pro tip: Start early to avoid crowds and enjoy the best lighting for photos!")
        elif hour > 16:
            response_parts.append(f"\n🌟 Pro tip: Many attractions have beautiful evening lighting - perfect for romantic visits!")
        
        # Add weather-appropriate suggestion if available
        weather_info = self._get_weather_info()
        if weather_info.get('condition') == 'rainy':
            indoor_count = sum(1 for attr in attractions[:5] if attr.get('weather_preference') == 'indoor')
            if indoor_count > 0:
                response_parts.append(f"\n☔ Since it's rainy today, I've included some great indoor attractions for you!")
        
        # Add follow-up engagement
        response_parts.append(f"\n❓ Would you like detailed directions to any of these places, or shall I suggest more attractions in a specific district? 🗺️")
        
        return "\n".join(response_parts)
    
    def _enhance_intent_classification(self, message: str) -> str:
        """Enhanced intent classification that includes attractions"""
        
        message_lower = message.lower()
        
        # Attraction-related keywords
        attraction_keywords = [
            'attraction', 'attractions', 'visit', 'see', 'explore', 'sightseeing',
            'museum', 'palace', 'mosque', 'tower', 'monument', 'historic', 'historical',
            'cultural', 'heritage', 'landmark', 'sight', 'sights', 'place', 'places',
            'things to do', 'what to see', 'worth visiting', 'must see', 'tourist',
            'hidden gem', 'off the beaten path', 'local favorite', 'authentic',
            'family friendly', 'romantic', 'couples', 'kids', 'children'
        ]
        
        # Cultural and historical keywords
        cultural_keywords = [
            'culture', 'cultural', 'history', 'historical', 'heritage', 'tradition',
            'art', 'architecture', 'byzantine', 'ottoman', 'ancient', 'medieval',
            'unesco', 'learn about', 'discover', 'understand'
        ]
        
        # Family activity keywords
        family_keywords = [
            'family', 'families', 'kids', 'children', 'child', 'baby', 'toddler',
            'family friendly', 'family activities', 'things to do with kids',
            'playgrounds', 'interactive', 'educational'
        ]
        
        # Romantic keywords
        romantic_keywords = [
            'romantic', 'romance', 'couples', 'date', 'anniversary', 'honeymoon',
            'valentine', 'proposal', 'intimate', 'cozy', 'sunset', 'dinner for two'
        ]
        
        # Check for attraction-related intents
        if any(keyword in message_lower for keyword in attraction_keywords):
            if any(keyword in message_lower for keyword in cultural_keywords):
                return 'cultural_query'
            elif any(keyword in message_lower for keyword in family_keywords):
                return 'family_activity'
            elif any(keyword in message_lower for keyword in romantic_keywords):
                return 'romantic_spot'
            elif 'hidden' in message_lower or 'secret' in message_lower or 'local' in message_lower:
                return 'hidden_gem'
            else:
                return 'attraction_query'
        
        # Enhanced restaurant detection
        restaurant_keywords = [
            'restaurant', 'food', 'eat', 'dining', 'meal', 'lunch', 'dinner', 'breakfast',
            'cafe', 'cuisine', 'dish', 'menu', 'chef', 'cook', 'hungry', 'delicious'
        ]
        
        if any(keyword in message_lower for keyword in restaurant_keywords):
            if any(keyword in message_lower for keyword in ['recommend', 'suggest', 'best', 'good', 'top']):
                return 'restaurant_recommendation'
            else:
                return 'restaurant_query'
        
        # Default classification logic
        if any(word in message_lower for word in ['where', 'location', 'address']):
            return 'location_query'
        elif any(word in message_lower for word in ['how', 'directions', 'route', 'way']):
            return 'route_query'  
        elif any(word in message_lower for word in ['when', 'time', 'hours', 'schedule']):
            return 'time_query'
        elif any(word in message_lower for word in ['price', 'cost', 'expensive', 'cheap']):
            return 'price_query'
        else:
            return 'general_query'

    # 🚀 ENHANCED DEEP LEARNING INTEGRATION METHODS
    
    def _sync_deep_learning_profile(self, user_profile: UserProfile, dl_analytics: Dict[str, Any]):
        """Sync traditional user profile with deep learning analytics"""
        
        try:
            # Update satisfaction score
            if 'satisfaction_score' in dl_analytics:
                user_profile.satisfaction_score = dl_analytics['satisfaction_score']
            
            # Update preferences from deep learning insights
            if 'favorite_neighborhoods' in dl_analytics:
                user_profile.favorite_neighborhoods = dl_analytics['favorite_neighborhoods']
            
            if 'favorite_cuisines' in dl_analytics:
                user_profile.cuisine_preferences = dl_analytics['favorite_cuisines']
            
            # Update success rate
            if 'average_confidence' in dl_analytics:
                confidence = dl_analytics['average_confidence']
                user_profile.recommendation_success_rate = min(1.0, confidence * 1.2)
            
            logger.info(f"✨ Synced deep learning insights for user {user_profile.user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to sync deep learning profile: {e}")

    def _enhance_multi_intent_response(self, multi_intent_response: str, entities: Dict, user_profile: UserProfile, current_time: datetime) -> str:
        """Enhance multi-intent response with Istanbul-specific context and local flavor"""
        try:
            enhanced_response = multi_intent_response
            # Add time-of-day context
            hour = current_time.hour
            if hour < 11 and 'breakfast' not in enhanced_response.lower():
                enhanced_response += "\n\n☀️ Perfect timing for Turkish breakfast! Most places serve amazing kahvaltı until 11 AM."
            elif 19 <= hour <= 21 and 'dinner' not in enhanced_response.lower():
                enhanced_response += "\n\n🌆 Great dinner timing! Istanbul restaurants really come alive in the evening."
            # Add neighborhood-specific tips
            if entities.get('neighborhoods'):
                neighborhood = entities['neighborhoods'][0]
                if neighborhood == 'sultanahmet':
                    enhanced_response += "\n\n🏛️ Pro tip: Sultanahmet can be touristy - ask locals for hidden gems nearby!"
                elif neighborhood == 'beyoğlu':
                    enhanced_response += "\n\n🎭 Beyoğlu tip: The side streets off İstiklal have amazing local spots!"
                elif neighborhood == 'kadıköy':
                    enhanced_response += "\n\n🌊 Kadıköy insider info: The Asian side has the best local food scene!"
            # Add cultural dining tips
            if 'restaurant' in enhanced_response.lower() or 'dining' in enhanced_response.lower():
                enhanced_response += "\n\n🇹🇷 Istanbul dining tip: Don't rush your meal - Turks love to linger over food and conversation!"
            # Add personalization based on user profile
            if user_profile.cuisine_preferences:
                fav_cuisine = user_profile.cuisine_preferences[0]
                if fav_cuisine not in enhanced_response.lower():
                    enhanced_response += f"\n\n😋 Since you love {fav_cuisine}, I can suggest more {fav_cuisine} spots if you'd like!"
            return enhanced_response
        except Exception as e:
            logger.warning(f"Failed to enhance multi-intent response: {e}")
            return multi_intent_response  # Return original if enhancement fails

    def _is_restaurant_query(self, message: str) -> bool:
        """Check if message is restaurant-related"""
        restaurant_keywords = [
            'restaurant', 'food', 'eat', 'dining', 'meal', 'breakfast', 'lunch', 'dinner',
            'cuisine', 'dish', 'menu', 'kitchen', 'cafe', 'bistro', 'meyhane', 'lokanta',
            'kebab', 'meze', 'baklava', 'coffee', 'tea', 'hungry', 'taste', 'flavor',
            'vegetarian', 'vegan', 'halal', 'kosher', 'gluten'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in restaurant_keywords)

    def _is_neighborhood_query(self, message: str) -> bool:
        """Check if message is neighborhood/area-related"""
        neighborhood_keywords = [
            'neighborhood', 'neighbourhood', 'area', 'district', 'quarter', 'region',
            'where to stay', 'best area', 'local area', 'explore area', 'walk around',
            'sultanahmet', 'beyoglu', 'galata', 'ortakoy', 'balat', 'kadikoy', 'besiktas',
            'taksim', 'karakoy', 'uskudar', 'fatih', 'sisli', 'bakirkoy',
            'character', 'atmosphere', 'vibe', 'feel like', 'authentic', 'local life',
            'hidden gems', 'local secrets', 'off the beaten path', 'insider tips'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in neighborhood_keywords)

    def _process_neighborhood_query(self, message: str, user_profile: UserProfile, current_time: datetime) -> str:
        """🏘️ ENHANCED: Process neighborhood-related queries with deep learning insights"""
        
        if not self.neighborhood_guides:
            return self._generate_fallback_neighborhood_response(message)
        
        try:
            logger.debug(f"🏘️ Processing neighborhood query: {message}")
            
            # Extract visitor type from user profile and message
            visitor_type = self._determine_visitor_type(user_profile, message)
            logger.debug(f"🏘️ Determined visitor type: {visitor_type}")
            
            # Extract interests from message using deep learning if available
            interests = self._extract_interests_from_message(message)
            logger.debug(f"🏘️ Extracted interests: {interests}")
            
            # Determine season
            season = self._get_current_season(current_time)
            logger.debug(f"🏘️ Current season: {season}")
            
            # Detect specific neighborhood mentions
            mentioned_neighborhood = self._extract_neighborhood_from_message(message)
            logger.debug(f"🏘️ Mentioned neighborhood: {mentioned_neighborhood}")
            
            if mentioned_neighborhood:
                # Specific neighborhood guide
                guide = self.neighborhood_guides.get_neighborhood_guide(mentioned_neighborhood)
                if guide:
                    logger.debug(f"🏘️ Found guide for {mentioned_neighborhood}")
                    return self._generate_specific_neighborhood_response(guide, visitor_type, interests, season)
                else:
                    logger.debug(f"🏘️ No guide found for {mentioned_neighborhood}")
            
            # General neighborhood recommendations
            if any(word in message.lower() for word in ['recommend', 'suggest', 'best', 'where']):
                logger.debug("🏘️ Generating personalized recommendations")
                personal_guide = self.neighborhood_guides.generate_personalized_neighborhood_guide(
                    visitor_type=visitor_type,
                    interests=interests,
                    visit_duration=self._estimate_visit_duration(message),
                    season=season
                )
                return self._generate_personalized_neighborhood_response(personal_guide, message)
            
            # Hidden gems query
            if any(word in message.lower() for word in ['hidden', 'secret', 'local', 'authentic', 'off the beaten']):
                logger.debug("🏘️ Generating hidden gems response")
                gems = self.neighborhood_guides.search_hidden_gems()
                return self._generate_hidden_gems_response(gems[:5], visitor_type)
            
            # Default neighborhood overview
            logger.debug("🏘️ Generating neighborhood overview")
            return self._generate_neighborhood_overview_response(visitor_type, interests)
            
        except Exception as e:
            logger.error(f"Error processing neighborhood query: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._generate_fallback_neighborhood_response(message)

    def _determine_visitor_type(self, user_profile: UserProfile, message: str) -> VisitorType:
        """Determine visitor type from profile and message using deep learning insights"""
        
        # Use deep learning insights if available
        if self.deep_learning_ai:
            try:
                message_analysis = self.deep_learning_ai.optimize_for_english_speakers(message)
                
                # Analyze conversation style and preferences
                if message_analysis.get("conversation_style") == "analytical":
                    return VisitorType.CULTURAL_EXPLORER
                elif message_analysis.get("conversation_style") == "creative":
                    return VisitorType.PHOTOGRAPHY_LOVER
                elif message_analysis.get("conversation_style") == "practical":
                    return VisitorType.BUDGET_TRAVELER
            except Exception as e:
                logger.debug(f"Deep learning visitor type analysis failed: {e}")
        
        # Fallback to message-based analysis
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['first time', 'never been', 'tourist']):
            return VisitorType.FIRST_TIME_TOURIST
        elif any(word in message_lower for word in ['culture', 'history', 'museum', 'traditional']):
            return VisitorType.CULTURAL_EXPLORER
        elif any(word in message_lower for word in ['food', 'eat', 'restaurant', 'cuisine']):
            return VisitorType.FOOD_ENTHUSIAST
        elif any(word in message_lower for word in ['photo', 'instagram', 'picture', 'camera']):
            return VisitorType.PHOTOGRAPHY_LOVER
        elif any(word in message_lower for word in ['local', 'authentic', 'real', 'insider']):
            return VisitorType.LOCAL_EXPERIENCE_SEEKER
        elif any(word in message_lower for word in ['budget', 'cheap', 'affordable', 'free']):
            return VisitorType.BUDGET_TRAVELER
        elif any(word in message_lower for word in ['luxury', 'expensive', 'premium', 'high-end']):
            return VisitorType.LUXURY_TRAVELER
        elif any(word in message_lower for word in ['family', 'kids', 'children', 'child']):
            return VisitorType.FAMILY_WITH_CHILDREN
        elif any(word in message_lower for word in ['young', 'backpack', 'hostel', 'party']):
            return VisitorType.YOUNG_BACKPACKER
        elif any(word in message_lower for word in ['business', 'work', 'meeting', 'conference']):
            return VisitorType.BUSINESS_TRAVELER
        else:
            return VisitorType.FIRST_TIME_TOURIST

    def _extract_interests_from_message(self, message: str) -> List[str]:
        """Extract interests from message using deep learning if available"""
        interests = []
        message_lower = message.lower()
        
        # Use deep learning analysis if available
        if self.deep_learning_ai:
            try:
                cultural_refs = self.deep_learning_ai._detect_cultural_references(message)
                if cultural_refs:
                    interests.extend(['culture', 'history'])
            except Exception as e:
                logger.debug(f"Deep learning interest extraction failed: {e}")
        
        # Traditional keyword matching
        interest_keywords = {
            'history': ['history', 'historical', 'ancient', 'old', 'past', 'heritage'],
            'culture': ['culture', 'cultural', 'traditional', 'art', 'museum', 'gallery'],
            'food': ['food', 'eat', 'restaurant', 'cuisine', 'cooking', 'taste'],
            'nightlife': ['night', 'bar', 'club', 'drink', 'party', 'entertainment'],
            'photography': ['photo', 'picture', 'camera', 'instagram', 'scenic', 'view'],
            'shopping': ['shop', 'shopping', 'buy', 'market', 'bazaar', 'store'],
            'local_life': ['local', 'authentic', 'real', 'everyday', 'neighborhood', 'community']
        }
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                interests.append(interest)
        
        return interests if interests else ['culture']  # Default interest

    def _extract_neighborhood_from_message(self, message: str) -> Optional[str]:
        """Extract specific neighborhood mentions from message"""
        neighborhood_names = {
            'sultanahmet': ['sultanahmet', 'sultan ahmet', 'blue mosque area', 'hagia sophia area'],
            'beyoglu': ['beyoglu', 'beyoğlu', 'pera', 'galata', 'istiklal', 'taksim'],
            'balat': ['balat', 'fener', 'golden horn'],
            'ortakoy': ['ortakoy', 'ortaköy', 'bosphorus bridge area'],
            'kadikoy': ['kadikoy', 'kadıköy', 'asian side', 'moda']
        }
        
        message_lower = message.lower()
        for neighborhood, variations in neighborhood_names.items():
            if any(variation in message_lower for variation in variations):
                return neighborhood
        
        return None

    def _estimate_visit_duration(self, message: str) -> str:
        """Estimate visit duration from message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['quick', 'short', 'hour', 'brief']):
            return "short"
        elif any(word in message_lower for word in ['day', 'full', 'whole', 'entire']):
            return "long"
        else:
            return "medium"

    def _get_current_season(self, current_time: datetime) -> str:
        """Get current season"""
        month = current_time.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def _generate_specific_neighborhood_response(self, guide, visitor_type: VisitorType, 
                                               interests: List[str], season: str) -> str:
        """Generate detailed response for specific neighborhood"""
        
        response = f"🏘️ **{guide.name} Neighborhood Guide**\n\n"
        
        # Character description
        response += f"**Character:** {guide.character_description}\n\n"
        
        # Atmosphere
        response += f"**Atmosphere:** {guide.atmosphere}\n\n"
        
        # Seasonal highlight
        if season in guide.seasonal_highlights:
            response += f"**{season.title()} Highlight:** {guide.seasonal_highlights[season]}\n\n"
        
        # Best visiting times
        times = [time.value.replace('_', ' ').title() for time in guide.best_visiting_times]
        response += f"**Best Times to Visit:** {', '.join(times)}\n\n"
        
        # Personalized recommendations based on interests
        if 'photography' in interests and guide.photo_opportunities:
            response += f"**📸 Photo Opportunities:** {', '.join(guide.photo_opportunities[:3])}\n\n"
        
        if 'food' in interests and guide.must_try_foods:
            response += f"**🍽️ Must-Try Foods:** {', '.join(guide.must_try_foods)}\n\n"
        
        # Hidden gems
        if guide.hidden_gems:
            response += f"**💎 Hidden Gems:**\n"
            for gem in guide.hidden_gems[:2]:
                response += f"• **{gem.name}:** {gem.description}\n"
                response += f"  *Insider tip:* {gem.insider_tip}\n"
            response += "\n"
        
        # Local insights
        if guide.local_insights:
            response += f"**🗝️ Local Insights:**\n"
            for insight in guide.local_insights[:3]:
                response += f"• {insight}\n"
            response += "\n"
        
        # Practical info
        response += f"**Practical Info:**\n"
        response += f"• Safety Rating: {guide.safety_rating}/10\n"
        response += f"• Price Level: {guide.price_level.title()}\n"
        response += f"• Walking Difficulty: {guide.walking_difficulty.title()}\n"
        response += f"• Estimated Visit: {guide.estimated_visit_duration}\n"
        
        # Cultural etiquette if relevant
        if visitor_type == VisitorType.FIRST_TIME_TOURIST and guide.local_etiquette:
            response += f"\n**🤝 Cultural Etiquette:**\n"
            for etiquette in guide.local_etiquette[:2]:
                response += f"• {etiquette}\n"
        
        return response

    def _generate_personalized_neighborhood_response(self, personal_guide: Dict[str, Any], 
                                                   original_message: str) -> str:
        """Generate response from personalized guide"""
        
        response = f"🎯 **Personalized Neighborhood Recommendations**\n\n"
        
        profile = personal_guide['visitor_profile']
        response += f"Based on your profile as a **{profile['type'].replace('_', ' ').title()}** "
        response += f"interested in **{', '.join(profile['interests'])}**, here are my top recommendations:\n\n"
        
        # Top recommendations
        for i, rec in enumerate(personal_guide['top_recommendations'], 1):
            response += f"**{i}. {rec['neighborhood']}** (Match Score: {rec['score']}/10)\n"
            response += f"*Why recommended:* {rec['why_recommended']}\n"
            response += f"*Best times:* {', '.join(rec['best_times'])}\n"
            response += f"*Seasonal highlight:* {rec['seasonal_highlight']}\n"
            response += f"*Must try:* {', '.join(rec['must_try'])}\n\n"
        
        # Hidden gems compilation
        if personal_guide['hidden_gems_compilation']:
            response += f"**💎 Curated Hidden Gems for You:**\n"
            for gem in personal_guide['hidden_gems_compilation'][:3]:
                response += f"• **{gem['name']}** ({gem['neighborhood']}): {gem['description']}\n"
            response += "\n"
        
        # Insider tips
        if personal_guide['insider_tips']:
            response += f"**🗝️ Insider Tips:**\n"
            for tip in personal_guide['insider_tips'][:3]:
                response += f"• {tip}\n"
            response += "\n"
        
        response += f"💡 *Want detailed information about any specific neighborhood? Just ask!*"
        
        return response

    def _generate_hidden_gems_response(self, gems: List[Tuple[str, Any]], 
                                     visitor_type: VisitorType) -> str:
        """Generate response focused on hidden gems"""
        
        response = f"💎 **Hidden Gems & Local Secrets**\n\n"
        response += f"As a **{visitor_type.value.replace('_', ' ').title()}**, you'll love these insider discoveries:\n\n"
        
        for neighborhood, gem in gems:
            response += f"**{gem.name}** *(in {neighborhood.title()})*\n"
            response += f"{gem.description}\n"
            response += f"*How to find:* {gem.location_hint}\n"
            response += f"*Insider tip:* {gem.insider_tip}\n"
            response += f"*Difficulty:* {gem.difficulty_to_find.title()}\n\n"
        
        response += f"🗝️ *These are the places locals don't want tourists to find - you're welcome!*"
        
        return response

    def _generate_neighborhood_overview_response(self, visitor_type: VisitorType, 
                                               interests: List[str]) -> str:
        """Generate general neighborhood overview"""
        
        response = f"🏘️ **Istanbul Neighborhoods Overview**\n\n"
        
        if not self.neighborhood_guides:
            return "I'd love to help you explore Istanbul's neighborhoods! Our detailed neighborhood guides will be available soon."
        
        summary = self.neighborhood_guides.get_all_neighborhoods_summary()
        
        response += f"I can guide you through **{summary['total_neighborhoods']} unique neighborhoods**, "
        response += f"each with its own character and **{summary['total_hidden_gems']} hidden gems** waiting to be discovered!\n\n"
        
        # Recommend neighborhoods based on visitor type
        recommendations = self.neighborhood_guides.get_recommendations_for_visitor_type(visitor_type)
        
        if recommendations:
            response += f"**Perfect for {visitor_type.value.replace('_', ' ').title()}s like you:**\n"
            for neighborhood in recommendations[:3]:
                response += f"• **{neighborhood.name}:** {neighborhood.character_description[:100]}...\n"
            response += "\n"
        
        response += f"**What would you like to explore?**\n"
        response += f"• Ask about specific neighborhoods (Sultanahmet, Beyoğlu, Balat, etc.)\n"
        response += f"• Request personalized recommendations\n"
        response += f"• Discover hidden gems and local secrets\n"
        response += f"• Get cultural insights and insider tips\n\n"
        
        response += f"💡 *Just tell me what kind of Istanbul experience you're looking for!*"
        
        return response

    def _generate_fallback_neighborhood_response(self, message: str) -> str:
        """Generate fallback response when neighborhood guides not available"""
        return (
            "🏘️ I'd love to help you explore Istanbul's amazing neighborhoods! "
            "Each area has its own unique character and hidden gems. "
            "Our comprehensive neighborhood guides with deep learning recommendations "
            "will be available soon. In the meantime, I can help you with attractions, "
            "restaurants, and general Istanbul advice!"
        )
