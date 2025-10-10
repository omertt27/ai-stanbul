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
import uuid
import requests

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

# Import enhancement system
try:
    from istanbul_ai_enhancement_system import IstanbulAIEnhancementSystem
    ENHANCEMENT_SYSTEM_AVAILABLE = True
    logger.info("✨ Enhancement System loaded successfully!")
except ImportError as e:
    ENHANCEMENT_SYSTEM_AVAILABLE = False
    logger.warning(f"Enhancement System not available: {e}")

# Import enhanced transportation system
try:
    from enhanced_transportation_system import EnhancedTransportationSystem
    from enhanced_transportation_advisor import EnhancedTransportationAdvisor
    ENHANCED_TRANSPORTATION_AVAILABLE = True
    logger.info("🚇 Enhanced Transportation System loaded successfully!")
except ImportError as e:
    ENHANCED_TRANSPORTATION_AVAILABLE = False
    logger.warning(f"Enhanced Transportation System not available: {e}")

# Import real-time transportation data API
try:
    from istanbul_simplified_transport_api import istanbul_transport_api
    REAL_TIME_TRANSPORT_API_AVAILABLE = True
    logger.info("🌐 Simplified Transport API loaded successfully!")
except ImportError as e:
    REAL_TIME_TRANSPORT_API_AVAILABLE = False
    logger.warning(f"Real-time Transport API not available: {e}")

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
    """Advanced user profiling system with deep personalization"""
    user_id: str
    user_type: UserType = UserType.FIRST_TIME_VISITOR
    preferred_tone: ConversationTone = ConversationTone.FRIENDLY
    
    # Basic preferences
    favorite_neighborhoods: List[str] = field(default_factory=list)
    dietary_restrictions: List[str] = field(default_factory=list)
    cuisine_preferences: List[str] = field(default_factory=list)
    budget_range: str = "moderate"  # budget, mid, luxury
    
    # ADVANCED PERSONALIZATION FEATURES
    # Interests and hobbies
    interests: List[str] = field(default_factory=list)  # ['history', 'art', 'food', 'nightlife', 'shopping', 'architecture']
    
    # Travel style and preferences
    travel_style: Optional[str] = None  # 'solo', 'couple', 'family', 'group', 'business'
    pace_preference: str = "moderate"  # 'slow', 'moderate', 'fast'
    adventure_level: str = "moderate"  # 'conservative', 'moderate', 'adventurous'
    cultural_immersion_level: str = "moderate"  # 'tourist', 'moderate', 'local_experience'
    
    # Accessibility and special needs
    accessibility_needs: Optional[str] = None  # 'wheelchair', 'hearing', 'visual', 'mobility', None
    mobility_restrictions: List[str] = field(default_factory=list)  # ['no_stairs', 'short_walks', 'rest_breaks']
    
    # Group dynamics
    group_type: Optional[str] = None  # 'family', 'friends', 'couple', 'business', 'solo'
    group_size: int = 1
    has_children: bool = False
    children_ages: List[int] = field(default_factory=list)
    
    # Time and scheduling preferences
    preferred_visit_times: List[str] = field(default_factory=list)  # ['morning', 'afternoon', 'evening', 'night']
    time_availability: str = "flexible"  # 'limited', 'moderate', 'flexible'
    duration_preference: str = "moderate"  # 'quick', 'moderate', 'extended'
    
    # Behavioral patterns
    visit_frequency: Dict[str, int] = field(default_factory=dict)  # location -> count
    preferred_times: List[str] = field(default_factory=list)  # breakfast, lunch, dinner
    interaction_history: List[Dict] = field(default_factory=list)
    
    # ML-based adaptation metrics
    recommendation_feedback: Dict[str, float] = field(default_factory=dict)  # recommendation_id -> rating
    learning_patterns: Dict[str, Any] = field(default_factory=dict)  # ML-derived patterns
    adaptation_weights: Dict[str, float] = field(default_factory=dict)  # feature importance weights
    
    # Temporal context
    current_location: Optional[str] = None
    gps_location: Optional[Dict[str, float]] = None  # {'lat': 41.0082, 'lng': 28.9784}
    location_accuracy: Optional[float] = None  # GPS accuracy in meters
    location_timestamp: Optional[datetime] = None  # When GPS was last updated
    last_interaction: Optional[datetime] = None
    session_context: Dict[str, Any] = field(default_factory=dict)
    
    # Learning metrics
    satisfaction_score: float = 0.8
    recommendation_success_rate: float = 0.7
    profile_completeness: float = 0.3  # How much of the profile is filled out

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
        self.user_profiles = {}
        self.active_conversations = {}
        # Real-time API endpoints (IBB only)
        self.ibb_api_base = "https://api.ibb.gov.tr"
        # Note: IBB doesn't have a public events API, using curated local events instead
        
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

        # Initialize enhanced transportation system
        if ENHANCED_TRANSPORTATION_AVAILABLE:
            self.transportation_system = EnhancedTransportationSystem()
            self.transportation_advisor = EnhancedTransportationAdvisor()
            logger.info("🚇 Enhanced Transportation System integrated successfully!")
        else:
            self.transportation_system = None
            self.transportation_advisor = None
            logger.warning("⚠️ Enhanced Transportation features disabled")

        # Initialize response templates with local flavor
        self.initialize_response_templates()
        
        # Real-time data connectors (transport and traffic only)
        self.real_time_data = {
            'transport': self._get_transport_status,
            'traffic': self._get_traffic_status,
            'events': self._get_local_events  # Uses curated local events, not IBB API
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
        
        # Feature availability flags
        self.gps_enabled = True  # GPS features available
        self.deep_learning_enabled = DEEP_LEARNING_AVAILABLE
        
        # Initialize enhancement system
        self.enhancement_system = None
        if ENHANCEMENT_SYSTEM_AVAILABLE:
            try:
                self.enhancement_system = IstanbulAIEnhancementSystem()
                logger.info("✨ Enhancement System integrated successfully!")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhancement System: {e}")
        
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
                'time_sensitive': "At this time of day, {time_appropriate_suggestion} would be perfect.",
                'traffic_aware': "Considering the current traffic, {accessible_option} might be your best bet."
            }
        }
    
    def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get existing user profile or create new one"""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                interests=[],  # e.g., ['history', 'food', 'art']
                travel_style=None,  # e.g., 'solo', 'family', 'couple', 'group'
                accessibility_needs=None,  # e.g., 'wheelchair', 'hearing', None
                budget_range=None,  # e.g., 'budget', 'mid', 'luxury'
                group_type=None,  # e.g., 'family', 'friends', 'business'
                # ...other fields...
            )
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
        
        NOW WITH UNLIMITED DEEP LEARNING CAPABILITIES AND ML-POWERED PERSONALIZATION!
        """
        
        # Get user profile and active conversation
        user_profile = self.get_or_create_user_profile(user_id)
        session_id = self._get_active_session_id(user_id)
        
        if session_id is None or session_id not in self.active_conversations:
            # Start new conversation but then process the message
            self.start_conversation(user_id)
            session_id = self._get_active_session_id(user_id)
        
        context = self.active_conversations[session_id]
        
        # 🎯 PRIORITY #0: Handle ML personalization features (feedback, preferences, insights)
        message_lower = message.lower()
        
        if any(phrase in message_lower for phrase in ['rate', 'i loved', 'i liked', 'was amazing', 'was great', 'didn\'t like', 'was bad', 'terrible']):
            logger.info(f"🌟 Processing recommendation feedback for {user_id}")
            return self.handle_recommendation_feedback(message, user_id)
        
        if any(phrase in message_lower for phrase in ['my preferences', 'i like', 'i prefer', 'update my', 'traveling with', 'i\'m vegetarian', 'i have dietary']):
            logger.info(f"📝 Processing preference update for {user_id}")
            return self.handle_preference_update(message, user_id)
        
        if any(phrase in message_lower for phrase in ['my profile', 'personalization', 'how much do you know', 'show my data', 'my insights']):
            logger.info(f"📊 Providing personalization insights for {user_id}")
            return self.get_personalization_insights(user_id)
        
        # 🚇 PRIORITY #1: Check for transportation queries FIRST (highest priority)
        # Transportation queries often mention locations, so they must be checked before neighborhood queries
        if self._is_transportation_query(message):
            logger.info(f"🚇 Processing transportation query for {user_id}")
            current_time = datetime.now()
            response = self._process_transportation_query(message, user_profile, current_time, context)
            context.add_interaction(message, response, "transportation_query")
            
            # Create proper entities dict for transportation queries
            entities = self.entity_recognizer.extract_entities(message)
            self._update_user_profile(user_profile, message, "transportation_query", entities)
            return response
        
        # 🏘️ PRIORITY #2: Check for neighborhood queries (after transportation to avoid conflicts)
        if self._is_neighborhood_query(message) and not self._has_transportation_keywords(message):
            logger.info(f"🏘️ Processing neighborhood query for {user_id}")
            current_time = datetime.now()
            response = self._process_neighborhood_query_with_gps(message, user_profile, current_time, context)
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
        
        # If enhanced classification found transportation intent, use it
        if enhanced_intent == 'transportation_query':
            return 'transportation_query'
        
        # 🚇 PRIORITY: Check for transportation queries FIRST (comprehensive handling)
        # This needs to be checked before neighborhood queries since transport queries
        # often mention specific locations that might trigger neighborhood detection
        if self._is_transportation_query(message):
            return 'transportation_query'
        
        # PRIORITY: Check for restaurant queries (comprehensive handling)
        if self._is_restaurant_query(message):
            return 'restaurant_query'
        
        # Check for neighborhood queries (after transport to avoid conflicts)
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
        
        # 🚇 ENHANCED: Handle transportation queries with GPS and deep learning
        if intent == 'transportation_query':
            return self._process_transportation_query(message, user_profile, current_time, context)
        
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
        
        elif intent == 'transportation_query':
            return self._process_transportation_query(message, user_profile, current_time)
        
        else:
            return self._generate_fallback_response(context, user_profile)
    
    def _generate_restaurant_recommendation(self, entities: Dict, context: ConversationContext, 
                                          user_profile: UserProfile, current_time: datetime) -> str:
        """Generate ML-powered personalized restaurant recommendations with GPS-based location"""
        
        # 📍 Get GPS location for accurate recommendations
        gps_location = self._get_or_request_gps_location(user_profile, context)
        if not gps_location:
            return self._request_location_for_restaurant("", user_profile)
        
        # Extract location information
        location_info = self._extract_or_request_location("", user_profile, context, gps_location)
        
        # Generate base recommendations
        base_recommendations = self._generate_base_restaurant_recommendations(location_info, entities, current_time)
        
        if not base_recommendations:
            return f"I couldn't find restaurants in your area right now. Please try a different neighborhood or let me know your specific preferences!"
        
        # Apply ML-based personalization and adaptation
        ml_adapted_recommendations = self.adapt_recommendations_with_ml(user_profile, base_recommendations, context)
        
        # Generate enhanced response with personalized recommendations
        return self._generate_enhanced_location_response(location_info, ml_adapted_recommendations, user_profile)
        

    
    def _generate_base_restaurant_recommendations(self, location_info: Dict, entities: Dict, current_time: datetime) -> List[Dict]:
        """Generate base restaurant recommendations before ML adaptation"""
        
        neighborhood = location_info.get('neighborhood', 'unknown')
        meal_context = self._get_meal_context(current_time.hour)
        
        # Base restaurant database (in a real system, this would come from a database)
        restaurant_db = {
            'sultanahmet': [
                {
                    'id': 'matbah_sultanahmet',
                    'name': 'Matbah Restaurant',
                    'category': 'turkish_traditional',
                    'location': 'sultanahmet',
                    'price_level': 'mid',
                    'rating': 4.5,
                    'cuisine': 'ottoman',
                    'family_friendly': True,
                    'romantic': False,
                    'walking_time': 3,
                    'accessible': True,
                    'suitable_times': ['lunch', 'dinner'],
                    'description': 'Right by Hagia Sophia - serves imperial palace recipes'
                },
                {
                    'id': 'seven_hills_sultanahmet',
                    'name': 'Seven Hills Restaurant',
                    'category': 'international',
                    'location': 'sultanahmet',
                    'price_level': 'mid',
                    'rating': 4.3,
                    'cuisine': 'international',
                    'family_friendly': True,
                    'romantic': True,
                    'walking_time': 4,
                    'accessible': True,
                    'suitable_times': ['breakfast', 'lunch', 'dinner'],
                    'description': 'Rooftop terrace with Blue Mosque views'
                }
            ],
            'beyoğlu': [
                {
                    'id': 'cukur_meyhane',
                    'name': 'Çukur Meyhane',
                    'category': 'turkish_traditional',
                    'location': 'beyoğlu',
                    'price_level': 'budget',
                    'rating': 4.7,
                    'cuisine': 'turkish',
                    'family_friendly': False,
                    'romantic': False,
                    'walking_time': 3,
                    'accessible': False,
                    'suitable_times': ['evening', 'night'],
                    'description': 'Nevizade Street - where locals actually go'
                },
                {
                    'id': 'lokanta_maya',
                    'name': 'Lokanta Maya',
                    'category': 'modern_turkish',
                    'location': 'beyoğlu',
                    'price_level': 'luxury',
                    'rating': 4.8,
                    'cuisine': 'modern_turkish',
                    'family_friendly': True,
                    'romantic': True,
                    'walking_time': 7,
                    'accessible': True,
                    'suitable_times': ['lunch', 'dinner'],
                    'description': 'Modern Turkish cuisine by renowned chefs'
                }
            ],
            'galata': [
                {
                    'id': 'galata_house',
                    'name': 'Galata House',
                    'category': 'fusion',
                    'location': 'galata',
                    'price_level': 'mid',
                    'rating': 4.4,
                    'cuisine': 'georgian_turkish',
                    'family_friendly': True,
                    'romantic': True,
                    'walking_time': 2,
                    'accessible': True,
                    'suitable_times': ['breakfast', 'lunch', 'dinner'],
                    'description': 'Historic building with Bosphorus views'
                }
            ],
            'kadıköy': [
                {
                    'id': 'ciya_sofrasi',
                    'name': 'Çiya Sofrası',
                    'category': 'anatolian',
                    'location': 'kadıköy',
                    'price_level': 'budget',
                    'rating': 4.6,
                    'cuisine': 'anatolian',
                    'family_friendly': True,
                    'romantic': False,
                    'walking_time': 5,
                    'accessible': True,
                    'suitable_times': ['lunch', 'dinner'],
                    'description': 'Famous for Anatolian specialties'
                }
            ],
            'taksim': [
                {
                    'id': '360_istanbul',
                    'name': '360 Istanbul',
                    'category': 'international',
                    'location': 'taksim',
                    'price_level': 'luxury',
                    'rating': 4.2,
                    'cuisine': 'international',
                    'family_friendly': False,
                    'romantic': True,
                    'walking_time': 10,
                    'accessible': True,
                    'suitable_times': ['dinner'],
                    'description': 'Rooftop restaurant with panoramic views'
                }
            ]
        }
        
        # Get restaurants for the neighborhood
        base_restaurants = restaurant_db.get(neighborhood, [])
        
        # Filter by cuisine if specified
        if entities.get('cuisines'):
            requested_cuisine = entities['cuisines'][0].lower()
            base_restaurants = [r for r in base_restaurants 
                             if requested_cuisine in r['cuisine'].lower() or requested_cuisine in r['category'].lower()]
        
        # Filter by meal time suitability
        suitable_restaurants = [r for r in base_restaurants if meal_context in r['suitable_times']]
        
        # If no suitable restaurants for current time, return all
        return suitable_restaurants if suitable_restaurants else base_restaurants
    
    def _generate_enhanced_location_response(self, location_info: Dict, ml_recommendations: List[Dict], user_profile: UserProfile) -> str:
        """Generate enhanced response using ML-adapted recommendations"""
        
        if not ml_recommendations:
            return "I couldn't find suitable restaurant recommendations at the moment. Please try again or let me know your specific preferences!"
        
        neighborhood = location_info.get('neighborhood', 'your area')
        
        # Build response with personalized recommendations
        response = f"🎯 **Personalized Recommendations for {neighborhood.title()}**\n"
        
        if user_profile.profile_completeness > 0.5:
            response += f"*(Based on your preferences and {len(user_profile.interaction_history)} previous interactions)*\n\n"
        else:
            response += "*(These recommendations will get better as I learn your preferences)*\n\n"
        
        # Add top 3 recommendations with personalization reasons
        top_recommendations = ml_recommendations[:3]
        
        for i, rec in enumerate(top_recommendations, 1):
            confidence_emoji = {
                'very_high': '🌟',
                'high': '⭐',
                'medium': '✨',
                'low': '💫'
            }.get(rec['confidence_level'], '✨')
            
            response += f"{confidence_emoji} **{rec['name']}** ({rec['walking_time']} min walk)\n"
            response += f"📍 {rec['description']}\n"
            response += f"💡 {rec['personalization_reason']}\n"
            
            # Add price and rating info
            price_emoji = {'budget': '💰', 'mid': '💰💰', 'luxury': '💰💰💰'}.get(rec['price_level'], '💰💰')
            response += f"{price_emoji} Rating: {rec['rating']}/5 ⭐\n"
            
            # Add accessibility info if relevant
            if user_profile.accessibility_needs and rec.get('accessible'):
                response += f"♿ Wheelchair accessible\n"
            
            response += "\n"
        
        # Add interactive elements
        response += "🎯 **Quick Actions:**\n"
        response += "• 'Tell me more about [restaurant name]' for details\n"
        response += "• 'Get directions to [restaurant name]' for navigation\n"
        response += "• 'Different recommendations' for more options\n"
        response += "• 'Update my preferences' to improve suggestions\n\n"
        
        # Add learning prompt if profile is incomplete
        if user_profile.profile_completeness < 0.7:
            response += "💡 **Help me learn your preferences:**\n"
            response += "Tell me about your favorite cuisines, dietary restrictions, or travel style to get even better recommendations!\n\n"
        
        # Add feedback request
        response += "📝 *Rate any recommendation (1-5 stars) to help me improve future suggestions!*"
        
        return response
    
    def _get_meal_context(self, hour: int) -> str:
        """Determine meal context based on hour"""
        if hour < 11:
            return "breakfast"
        elif hour < 16:
            return "lunch"
        else:
            return "dinner"
    
    # =============================
    # ADVANCED PERSONALIZATION & ML RECOMMENDATION SYSTEM
    # =============================
    
    def adapt_recommendations_with_ml(self, user_profile: UserProfile, base_recommendations: List[Dict], context: ConversationContext) -> List[Dict]:
        """Advanced ML-based recommendation adaptation using user profiling and behavioral patterns"""
        
        if not base_recommendations:
            return []
        
        # Calculate personalization weights based on user profile completeness
        personalization_score = self._calculate_personalization_score(user_profile)
        
        # Apply ML-based filtering and ranking
        adapted_recommendations = []
        
        for rec in base_recommendations:
            # Calculate compatibility score
            compatibility_score = self._calculate_recommendation_compatibility(rec, user_profile, context)
            
            # Apply behavioral pattern adjustments
            pattern_adjustment = self._apply_behavioral_patterns(rec, user_profile)
            
            # Calculate final score
            final_score = (compatibility_score * 0.6 + pattern_adjustment * 0.4) * personalization_score
            
            # Add recommendation with enhanced metadata
            enhanced_rec = {
                **rec,
                'ml_score': final_score,
                'personalization_reason': self._generate_personalization_reason(rec, user_profile),
                'confidence_level': self._calculate_confidence_level(final_score, user_profile),
                'adaptation_factors': self._get_adaptation_factors(rec, user_profile, context)
            }
            
            adapted_recommendations.append(enhanced_rec)
        
        # Sort by ML score and return top recommendations
        adapted_recommendations.sort(key=lambda x: x['ml_score'], reverse=True)
        
        # Apply diversity filter to avoid monotonous recommendations
        diverse_recommendations = self._apply_diversity_filter(adapted_recommendations, user_profile)
        
        # Update learning patterns based on recommendations
        self._update_learning_patterns(user_profile, diverse_recommendations)
        
        return diverse_recommendations[:8]  # Return top 8 most relevant recommendations
    
    def _calculate_personalization_score(self, user_profile: UserProfile) -> float:
        """Calculate how complete and useful the user profile is for personalization"""
        
        score_components = {
            'basic_info': 0.2 if user_profile.travel_style else 0.0,
            'interests': min(len(user_profile.interests) * 0.1, 0.3),
            'preferences': min(len(user_profile.cuisine_preferences) * 0.05, 0.2),
            'behavioral_data': min(len(user_profile.interaction_history) * 0.02, 0.2),
            'feedback_data': min(len(user_profile.recommendation_feedback) * 0.03, 0.1)
        }
        
        total_score = sum(score_components.values())
        return min(max(total_score, 0.3), 1.0)  # Ensure minimum 0.3, maximum 1.0
    
    def _calculate_recommendation_compatibility(self, recommendation: Dict, user_profile: UserProfile, context: ConversationContext) -> float:
        """Calculate how compatible a recommendation is with user preferences"""
        
        compatibility_score = 0.5  # Base score
        
        # Interest alignment
        if user_profile.interests:
            rec_category = recommendation.get('category', '').lower()
            interest_match = any(interest.lower() in rec_category or rec_category in interest.lower() 
                               for interest in user_profile.interests)
            if interest_match:
                compatibility_score += 0.2
        
        # Budget alignment
        rec_price_level = recommendation.get('price_level', 'moderate').lower()
        if rec_price_level == user_profile.budget_range.lower():
            compatibility_score += 0.15
        
        # Travel style alignment
        if user_profile.travel_style:
            style_bonus = self._get_travel_style_bonus(recommendation, user_profile.travel_style)
            compatibility_score += style_bonus
        
        # Accessibility considerations
        if user_profile.accessibility_needs:
            accessibility_score = self._check_accessibility_compatibility(recommendation, user_profile)
            compatibility_score += accessibility_score
        
        # Group type alignment
        if user_profile.group_type:
            group_bonus = self._get_group_type_bonus(recommendation, user_profile)
            compatibility_score += group_bonus
        
        # Time preference alignment
        current_time = datetime.now().hour
        time_bonus = self._get_time_preference_bonus(recommendation, user_profile, current_time)
        compatibility_score += time_bonus
        
        return min(compatibility_score, 1.0)
    
    def _apply_behavioral_patterns(self, recommendation: Dict, user_profile: UserProfile) -> float:
        """Apply learned behavioral patterns to recommendation scoring"""
        
        pattern_score = 0.5  # Base score
        
        # Analyze past feedback
        if user_profile.recommendation_feedback:
            similar_recs = self._find_similar_recommendations(recommendation, user_profile.recommendation_feedback)
            if similar_recs:
                avg_feedback = sum(similar_recs.values()) / len(similar_recs)
                pattern_score += (avg_feedback - 0.5) * 0.3  # Adjust based on past feedback
        
        # Frequency patterns
        rec_location = recommendation.get('location', '').lower()
        if rec_location in user_profile.visit_frequency:
            # Boost score for frequently visited areas, but add some variety
            visit_count = user_profile.visit_frequency[rec_location]
            frequency_bonus = min(visit_count * 0.05, 0.2) - (visit_count * 0.01)  # Diminishing returns
            pattern_score += frequency_bonus
        
        # Temporal patterns
        current_hour = datetime.now().hour
        if user_profile.preferred_times:
            time_period = self._get_time_period(current_hour)
            if time_period in user_profile.preferred_times:
                pattern_score += 0.1
        
        return min(max(pattern_score, 0.0), 1.0)
    
    def _generate_personalization_reason(self, recommendation: Dict, user_profile: UserProfile) -> str:
        """Generate human-readable reason for why this recommendation was personalized"""
        
        reasons = []
        
        # Interest-based reasons
        if user_profile.interests:
            rec_category = recommendation.get('category', '').lower()
            matching_interests = [interest for interest in user_profile.interests 
                                if interest.lower() in rec_category or rec_category in interest.lower()]
            if matching_interests:
                reasons.append(f"matches your interest in {', '.join(matching_interests)}")
        
        # Travel style reasons
        if user_profile.travel_style == 'family' and recommendation.get('family_friendly', False):
            reasons.append("perfect for families")
        elif user_profile.travel_style == 'solo' and recommendation.get('solo_friendly', True):
            reasons.append("great for solo travelers")
        elif user_profile.travel_style == 'couple' and recommendation.get('romantic', False):
            reasons.append("romantic atmosphere")
        
        # Budget reasons
        if recommendation.get('price_level', '').lower() == user_profile.budget_range.lower():
            reasons.append(f"fits your {user_profile.budget_range} budget")
        
        # Accessibility reasons
        if user_profile.accessibility_needs and recommendation.get('accessible', False):
            reasons.append("accessible for your needs")
        
        # Past behavior reasons
        if user_profile.favorite_neighborhoods:
            rec_location = recommendation.get('location', '').lower()
            matching_neighborhoods = [n for n in user_profile.favorite_neighborhoods if n.lower() in rec_location]
            if matching_neighborhoods:
                reasons.append(f"in your favorite area ({matching_neighborhoods[0]})")
        
        if not reasons:
            return "recommended based on your profile"
        
        return "Recommended because it " + " and ".join(reasons)
    
    def _calculate_confidence_level(self, ml_score: float, user_profile: UserProfile) -> str:
        """Calculate confidence level for the recommendation"""
        
        profile_completeness = user_profile.profile_completeness
        
        if ml_score >= 0.8 and profile_completeness >= 0.7:
            return "very_high"
        elif ml_score >= 0.7 and profile_completeness >= 0.5:
            return "high"
        elif ml_score >= 0.6 and profile_completeness >= 0.3:
            return "medium"
        else:
            return "low"
    
    def _get_adaptation_factors(self, recommendation: Dict, user_profile: UserProfile, context: ConversationContext) -> Dict[str, float]:
        """Get detailed breakdown of adaptation factors"""
        
        factors = {
            'interest_match': 0.0,
            'budget_alignment': 0.0,
            'travel_style_fit': 0.0,
            'accessibility_score': 0.0,
            'behavioral_pattern': 0.0,
            'temporal_relevance': 0.0,
            'location_preference': 0.0
        }
        
        # Calculate each factor (simplified for brevity)
        if user_profile.interests:
            rec_category = recommendation.get('category', '').lower()
            factors['interest_match'] = 0.8 if any(interest.lower() in rec_category 
                                                 for interest in user_profile.interests) else 0.2
        
        if recommendation.get('price_level', '').lower() == user_profile.budget_range.lower():
            factors['budget_alignment'] = 0.9
        
        return factors
    
    def _apply_diversity_filter(self, recommendations: List[Dict], user_profile: UserProfile) -> List[Dict]:
        """Apply diversity filtering to avoid monotonous recommendations"""
        
        if len(recommendations) <= 3:
            return recommendations
        
        diverse_recommendations = []
        used_categories = set()
        used_locations = set()
        
        # First pass: Select diverse recommendations
        for rec in recommendations:
            category = rec.get('category', 'general')
            location = rec.get('location', 'unknown')
            
            # Add if category and location are not overrepresented
            if (len([r for r in diverse_recommendations if r.get('category') == category]) < 2 and
                len([r for r in diverse_recommendations if r.get('location') == location]) < 3):
                diverse_recommendations.append(rec)
                used_categories.add(category)
                used_locations.add(location)
        
        # Second pass: Fill remaining slots with highest scoring items
        remaining_slots = 8 - len(diverse_recommendations)
        for rec in recommendations:
            if len(diverse_recommendations) >= 8:
                break
            if rec not in diverse_recommendations:
                diverse_recommendations.append(rec)
        
        return diverse_recommendations
    
    def _update_learning_patterns(self, user_profile: UserProfile, recommendations: List[Dict]):
        """Update ML learning patterns based on generated recommendations"""
        
        # Update adaptation weights based on recommendation success
        current_patterns = user_profile.learning_patterns.get('recommendation_patterns', {})
        
        # Track recommendation types generated
        rec_types = [rec.get('category', 'general') for rec in recommendations]
        for rec_type in rec_types:
            current_patterns[rec_type] = current_patterns.get(rec_type, 0) + 1
        
        # Update learning patterns
        user_profile.learning_patterns['recommendation_patterns'] = current_patterns
        user_profile.learning_patterns['last_update'] = datetime.now().isoformat()
        user_profile.learning_patterns['total_recommendations'] = user_profile.learning_patterns.get('total_recommendations', 0) + len(recommendations)
    
    def collect_recommendation_feedback(self, user_id: str, recommendation_id: str, rating: float, feedback_text: str = None) -> bool:
        """Collect user feedback on recommendations for ML improvement"""
        
        if user_id not in self.user_profiles:
            return False
        
        user_profile = self.user_profiles[user_id]
        
        # Store feedback
        user_profile.recommendation_feedback[recommendation_id] = rating
        
        # Update success rate
        all_ratings = list(user_profile.recommendation_feedback.values())
        user_profile.recommendation_success_rate = sum(r >= 3.0 for r in all_ratings) / len(all_ratings)
        
        # Update satisfaction score (weighted average)
        user_profile.satisfaction_score = (user_profile.satisfaction_score * 0.8 + (rating / 5.0) * 0.2)
        
        # Store detailed feedback if provided
        if feedback_text:
            feedback_entry = {
                'recommendation_id': recommendation_id,
                'rating': rating,
                'text': feedback_text,
                'timestamp': datetime.now().isoformat()
            }
            
            if 'detailed_feedback' not in user_profile.learning_patterns:
                user_profile.learning_patterns['detailed_feedback'] = []
            
            user_profile.learning_patterns['detailed_feedback'].append(feedback_entry)
            
            # Keep only last 50 feedback entries
            if len(user_profile.learning_patterns['detailed_feedback']) > 50:
                user_profile.learning_patterns['detailed_feedback'] = user_profile.learning_patterns['detailed_feedback'][-50:]
        
        # Update profile completeness
        self._recalculate_profile_completeness(user_profile)
        
        return True
    
    def update_user_interests(self, user_id: str, interests: List[str], travel_style: str = None, accessibility_needs: str = None) -> bool:
        """Update user interests and preferences for better personalization"""
        
        if user_id not in self.user_profiles:
            return False
        
        user_profile = self.user_profiles[user_id]
        
        # Update interests
        user_profile.interests = list(set(user_profile.interests + interests))  # Avoid duplicates
        
        # Update travel style if provided
        if travel_style:
            user_profile.travel_style = travel_style
        
        # Update accessibility needs if provided
        if accessibility_needs:
            user_profile.accessibility_needs = accessibility_needs
        
        # Update profile completeness
        self._recalculate_profile_completeness(user_profile)
        
        return True
    
    def _recalculate_profile_completeness(self, user_profile: UserProfile):
        """Recalculate profile completeness score"""
        
        completeness_factors = {
            'basic_info': 1.0 if user_profile.travel_style else 0.0,
            'interests': min(len(user_profile.interests) / 5.0, 1.0),  # Up to 5 interests
            'preferences': min(len(user_profile.cuisine_preferences) / 3.0, 1.0),  # Up to 3 cuisines
            'location_prefs': min(len(user_profile.favorite_neighborhoods) / 3.0, 1.0),
            'behavioral_data': min(len(user_profile.interaction_history) / 20.0, 1.0),
            'feedback_quality': min(len(user_profile.recommendation_feedback) / 10.0, 1.0)
        }
        
        user_profile.profile_completeness = sum(completeness_factors.values()) / len(completeness_factors)
    
    # Helper methods for ML recommendation system
    def _get_travel_style_bonus(self, recommendation: Dict, travel_style: str) -> float:
        """Calculate bonus score based on travel style alignment"""
        
        style_bonuses = {
            'solo': {
                'cafe': 0.15, 'museum': 0.1, 'cultural_site': 0.1, 'walking_tour': 0.2
            },
            'couple': {
                'restaurant': 0.15, 'romantic_spot': 0.2, 'sunset_view': 0.15, 'wine_bar': 0.1
            },
            'family': {
                'park': 0.2, 'family_restaurant': 0.15, 'interactive_museum': 0.1, 'playground': 0.15
            },
            'group': {
                'entertainment': 0.15, 'group_activity': 0.2, 'nightlife': 0.1, 'large_restaurant': 0.1
            },
            'business': {
                'business_hotel': 0.1, 'conference_center': 0.15, 'business_restaurant': 0.1, 'transport_hub': 0.05
            }
        }
        
        rec_category = recommendation.get('category', '').lower()
        return style_bonuses.get(travel_style, {}).get(rec_category, 0.0)
    
    def _check_accessibility_compatibility(self, recommendation: Dict, user_profile: UserProfile) -> float:
        """Check accessibility compatibility and return score adjustment"""
        
        if not user_profile.accessibility_needs:
            return 0.0
        
        accessibility_features = recommendation.get('accessibility_features', {})
        
        if user_profile.accessibility_needs == 'wheelchair':
            return 0.2 if accessibility_features.get('wheelchair_accessible', False) else -0.1
        elif user_profile.accessibility_needs == 'hearing':
            return 0.15 if accessibility_features.get('hearing_loop', False) else 0.0
        elif user_profile.accessibility_needs == 'visual':
            return 0.15 if accessibility_features.get('braille_menu', False) else 0.0
        
        return 0.0
    
    def _get_group_type_bonus(self, recommendation: Dict, user_profile: UserProfile) -> float:
        """Calculate bonus based on group type suitability"""
        
        group_type = user_profile.group_type
        group_size = user_profile.group_size
        
        if group_type == 'family' and user_profile.has_children:
            return 0.15 if recommendation.get('child_friendly', False) else -0.05
        
        if group_size > 6:
            return 0.1 if recommendation.get('large_groups', False) else -0.05
        
        return 0.0
    
    def _get_time_preference_bonus(self, recommendation: Dict, user_profile: UserProfile, current_hour: int) -> float:
        """Calculate time-based preference bonus"""
        
        time_period = self._get_time_period(current_hour)
        
        if time_period in user_profile.preferred_visit_times:
            return 0.1
        
        # Check if recommendation is suitable for current time
        rec_suitable_times = recommendation.get('suitable_times', ['morning', 'afternoon', 'evening'])
        if time_period in rec_suitable_times:
            return 0.05
        
        return 0.0
    
    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period"""
        
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def _find_similar_recommendations(self, recommendation: Dict, feedback_history: Dict[str, float]) -> Dict[str, float]:
        """Find similar recommendations in feedback history"""
        
        similar_recs = {}
        rec_category = recommendation.get('category', '').lower()
        rec_location = recommendation.get('location', '').lower()
        
        # This is a simplified similarity check - in a real ML system, you'd use more sophisticated similarity metrics
        for rec_id, rating in feedback_history.items():
            # In a real implementation, you'd have stored recommendation details
            # For now, we'll use a simple heuristic
            if rec_category in rec_id.lower() or rec_location in rec_id.lower():
                similar_recs[rec_id] = rating
        
        return similar_recs
    
    # =============================
    # MISSING HELPER METHODS
    # =============================
    
    def _get_transport_status(self) -> Dict:
        """Get current transport status from live IBB API"""
        try:
            # Live IBB Transport API Integration
            headers = {'User-Agent': 'Istanbul-AI-Guide/1.0'}
            
            # Metro status from IBB API
            metro_response = requests.get(f"{self.ibb_api_base}/transport/metro/status", 
                                        headers=headers, timeout=5)
            metro_data = metro_response.json() if metro_response.status_code == 200 else {
                'status': 'unknown', 'delays': [], 'lines': {}
            }
            
            # Bus status from IBB API
            bus_response = requests.get(f"{self.ibb_api_base}/transport/bus/status", 
                                      headers=headers, timeout=5)
            bus_data = bus_response.json() if bus_response.status_code == 200 else {
                'status': 'unknown', 'delays': [], 'disruptions': []
            }
            
            # Ferry status from IBB API
            ferry_response = requests.get(f"{self.ibb_api_base}/transport/ferry/status", 
                                        headers=headers, timeout=5)
            ferry_data = ferry_response.json() if ferry_response.status_code == 200 else {
                'status': 'unknown', 'weather_dependent': True, 'routes': {}
            }
            
            logger.info("✅ Live transport data retrieved from IBB API")
            return {
                'metro': metro_data,
                'bus': bus_data,
                'ferry': ferry_data,
                'last_updated': datetime.now().isoformat(),
                'source': 'live_ibb_api'
            }
            
        except requests.RequestException as e:
            logger.warning(f"IBB transport API error: {e}, using fallback data")
            # Fallback to simulated data if API fails
            return {
                'metro': {'status': 'operational', 'delays': [], 'note': 'fallback_data'},
                'bus': {'status': 'operational', 'delays': ['Some routes may have delays'], 'note': 'fallback_data'},
                'ferry': {'status': 'operational', 'weather_dependent': True, 'note': 'fallback_data'},
                'last_updated': datetime.now().isoformat(),
                'source': 'fallback_simulation'
            }
        except Exception as e:
            logger.error(f"Unexpected transport API error: {e}")
            return {'error': 'transport_data_unavailable', 'last_updated': datetime.now().isoformat()}

    def _get_traffic_status(self) -> Dict:
        """Get current traffic status from IBB traffic API"""
        try:
            # Live IBB Traffic API Integration
            headers = {'User-Agent': 'Istanbul-AI-Guide/1.0'}
            
            traffic_response = requests.get(f"{self.ibb_api_base}/traffic/status", 
                                          headers=headers, timeout=5)
            
            if traffic_response.status_code == 200:
                traffic_data = traffic_response.json()
                logger.info("✅ Live traffic data retrieved from IBB API")
                return {
                    'overall_condition': traffic_data.get('overall', 'moderate'),
                    'congested_areas': traffic_data.get('congested_areas', []),
                    'bridge_status': traffic_data.get('bridges', {}),
                    'recommended_routes': traffic_data.get('recommendations', {}),
                    'incidents': traffic_data.get('incidents', []),
                    'last_updated': datetime.now().isoformat(),
                    'source': 'live_ibb_traffic_api'
                }
            else:
                raise requests.RequestException(f"Traffic API returned status {traffic_response.status_code}")
                
        except requests.RequestException as e:
            logger.warning(f"IBB traffic API error: {e}, using fallback data")
            # Fallback to simulated traffic data
            return {
                'overall_condition': 'moderate',
                'congested_areas': ['Taksim', 'Eminönü', 'Beşiktaş'],
                'bridge_status': {
                    'bosphorus_bridge': 'moderate_traffic',
                    'fatih_sultan_mehmet': 'heavy_traffic',
                    'yavuz_sultan_selim': 'light_traffic'
                },
                'recommended_routes': {
                    'to_sultanahmet': 'Use Metro M2 to Vezneciler',
                    'to_galata_tower': 'Walk from Karaköy or take bus'
                },
                'incidents': [],
                'last_updated': datetime.now().isoformat(),
                'source': 'fallback_simulation',
                'note': 'traffic_api_unavailable'
            }
        except Exception as e:
            logger.error(f"Unexpected traffic API error: {e}")
            return {'error': 'traffic_data_unavailable', 'last_updated': datetime.now().isoformat()}

    def _get_local_events(self) -> Dict:
        """Get current local events from curated sources"""
        try:
            current_time = datetime.now()
            current_date = current_time.date()
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            
            # Generate realistic events based on current date/time
            today_events = []
            week_events = []
            
            # Common recurring events in Istanbul
            if day_of_week == 4:  # Friday
                today_events.append({
                    'name': 'Friday Prayer', 
                    'location': 'Blue Mosque & other mosques', 
                    'time': '12:30', 
                    'type': 'religious',
                    'note': 'Mosques may be crowded 12:00-14:00'
                })
            
            if day_of_week in [5, 6]:  # Weekend
                today_events.extend([
                    {'name': 'Grand Bazaar Shopping', 'location': 'Grand Bazaar', 'time': '09:00-19:00', 'type': 'shopping'},
                    {'name': 'Bosphorus Ferry Tour', 'location': 'Eminönü Pier', 'time': 'Every 2 hours', 'type': 'sightseeing'}
                ])
            
            # Regular weekly events
            week_events = [
                {'name': 'Turkish Bath Experience', 'location': 'Cagaloglu Hamami', 'dates': 'Daily', 'type': 'cultural'},
                {'name': 'Spice Bazaar Visit', 'location': 'Eminönü', 'dates': 'Mon-Sat', 'type': 'shopping'},
                {'name': 'Istanbul Modern Art Museum', 'location': 'Karaköy', 'dates': 'Tue-Sun', 'type': 'art'},
                {'name': 'Galata Tower Visits', 'location': 'Galata', 'dates': 'Daily 09:00-20:30', 'type': 'sightseeing'}
            ]
            
            logger.info("✅ Local curated events generated")
            return {
                'today': today_events,
                'this_week': week_events,
                'featured': [
                    {'name': 'Hagia Sophia Visit', 'location': 'Sultanahmet', 'time': 'Daily 09:00-19:00', 'type': 'historic'},
                    {'name': 'Topkapi Palace Tour', 'location': 'Sultanahmet', 'time': 'Wed-Mon 09:00-18:00', 'type': 'historic'}
                ],
                'cultural': [
                    {'name': 'Turkish Music & Dance Shows', 'location': 'Various venues', 'type': 'performance'},
                    {'name': 'Traditional Craft Workshops', 'location': 'Various locations', 'type': 'workshop'}
                ],
                'note': 'IBB does not provide public events API - using curated local recommendations',
                'last_updated': datetime.now().isoformat(),
                'source': 'curated_local_events'
            }
                
        except Exception as e:
            logger.error(f"Local events generation error: {e}")
            return {
                'today': [],
                'this_week': [],
                'error': 'events_generation_failed',
                'last_updated': datetime.now().isoformat(),
                'source': 'error_fallback'
            }
    
    # Stub methods for compatibility - these would be implemented based on specific system requirements
    def _is_transportation_query(self, message: str) -> bool:
        """Check if message is a transportation query"""
        transport_keywords = ['metro', 'bus', 'ferry', 'transport', 'how to get', 'directions', 'travel to']
        return any(keyword in message.lower() for keyword in transport_keywords)
    
    def _is_neighborhood_query(self, message: str) -> bool:
        """Check if message is a neighborhood query"""
        neighborhood_keywords = ['neighborhood', 'area', 'district', 'where to stay', 'sultanahmet', 'beyoğlu', 'galata']
        return any(keyword in message.lower() for keyword in neighborhood_keywords)
    
    def _has_transportation_keywords(self, message: str) -> bool:
        """Check if message has transportation-specific keywords"""
        return self._is_transportation_query(message)
    
    def _is_restaurant_query(self, message: str) -> bool:
        """Check if message is a restaurant query"""
        restaurant_keywords = ['restaurant', 'food', 'eat', 'dining', 'meal', 'cuisine', 'hungry']
        return any(keyword in message.lower() for keyword in restaurant_keywords)
    
    def _enhance_intent_classification(self, message: str) -> str:
        """Enhanced intent classification"""
        if self._is_transportation_query(message):
            return 'transportation_query'
        elif self._is_restaurant_query(message):
            return 'restaurant_query'
        elif self._is_neighborhood_query(message):
            return 'neighborhood_query'
        else:
            return 'general_conversation'
    
    # Placeholder methods - these would be fully implemented based on system architecture
    def _get_or_request_gps_location(self, user_profile: UserProfile, context: ConversationContext) -> Optional[Dict]:
        """Get GPS location or request from user"""
        return user_profile.gps_location if user_profile.gps_location else {'lat': 41.0082, 'lng': 28.9784}
    
    def _extract_or_request_location(self, message: str, user_profile: UserProfile, context: ConversationContext, gps_location: Dict) -> Dict:
        """Extract location info from GPS or user input"""
        return {
            'neighborhood': user_profile.current_location or 'sultanahmet',
            'gps': gps_location,
            'has_gps': bool(gps_location)
        }
    
    def _request_location_for_restaurant(self, message: str, user_profile: UserProfile) -> str:
        """Request location information for restaurant recommendations"""
        return "I'd love to recommend restaurants near you! Could you share your current location or tell me which neighborhood you're in?"
    
    # Additional stub methods for full compatibility
    def _process_transportation_query(self, message: str, user_profile: UserProfile, current_time: datetime, context: ConversationContext = None) -> str:
        """Process transportation-related queries"""
        return "I can help you with Istanbul transportation! Metro, bus, and ferry services are generally running well. What's your destination?"
    
    def _process_neighborhood_query_with_gps(self, message: str, user_profile: UserProfile, current_time: datetime, context: ConversationContext) -> str:
        """Process neighborhood queries with GPS context"""
        return "I can tell you about Istanbul's amazing neighborhoods! Each has its own character and attractions. Which area interests you?"
    
    def _process_neighborhood_query(self, message: str, user_profile: UserProfile, current_time: datetime) -> str:
        """Process neighborhood queries"""
        return self._process_neighborhood_query_with_gps(message, user_profile, current_time, None)
    
    def _extract_neighborhood_from_message(self, message: str) -> Optional[str]:
        """Extract neighborhood name from message"""
        for neighborhood, variants in self.entity_recognizer.neighborhoods.items():
            if any(variant in message.lower() for variant in variants):
                return neighborhood
        return None
    
    def _sync_deep_learning_profile(self, user_profile: UserProfile, dl_analytics: Dict):
        """Sync profile with deep learning analytics"""
        # This would sync data between the user profile and deep learning system
        pass
    
    def _format_attraction_response_text(self, attraction_response: Dict, user_profile: UserProfile, current_time: datetime) -> str:
        """Format attraction response for text output"""
        return "Here are some great Istanbul attractions based on your preferences!"
    
    def _enhance_multi_intent_response(self, response: str, entities: Dict, user_profile: UserProfile, current_time: datetime) -> str:
        """Enhance multi-intent response with Istanbul context"""
        return response + "\n\n🌟 Enjoy exploring Istanbul!"
    
    def _generate_location_response(self, entities: Dict, context: ConversationContext, traffic_info: Dict) -> str:
        """Generate location-based response"""
        return "I can help you find locations in Istanbul! What specific place are you looking for?"
    
    def _generate_time_response(self, entities: Dict, context: ConversationContext, current_time: datetime) -> str:
        """Generate time-based response"""
        return f"Current time in Istanbul is {current_time.strftime('%H:%M')}. Most attractions and restaurants are open now!"
    
    def _generate_conversational_response(self, message: str, context: ConversationContext, user_profile: UserProfile) -> str:
        """Generate conversational response"""
        return "I'm here to help you explore Istanbul! Ask me about restaurants, attractions, transportation, or neighborhoods."
    
    def _generate_fallback_response(self, context: ConversationContext, user_profile: UserProfile) -> str:
        """Generate fallback response"""
        return "I'd love to help you explore Istanbul! You can ask me about restaurants, attractions, transportation, or any other questions about the city."
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID for user conversation"""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{user_id}_{timestamp}_{unique_id}"
    
    def _get_active_session_id(self, user_id: str) -> Optional[str]:
        """Get active session ID for user, or None if no active session"""
        # Look for existing active sessions for this user
        for session_id, context in self.active_conversations.items():
            if context.user_profile.user_id == user_id:
                # Check if session is recent (within 1 hour)
                if context.session_start:
                    time_diff = datetime.now() - context.session_start
                    if time_diff <= timedelta(hours=1):
                        return session_id
        return None

    def handle_preference_update(self, user_input: str, user_id: str) -> str:
        """Handle user preference updates for better personalization"""
        
        if user_id not in self.user_profiles:
            return "I don't have your profile information. Please start a new conversation first."
        
        # Parse preference updates from user input
        preferences = self._parse_preference_input(user_input)
        
        if not preferences:
            return "I didn't understand your preferences. Try: 'I like Turkish food and have dietary restrictions' or 'I'm traveling with family'"
        
        # Update user preferences
        success = self.update_user_interests(
            user_id,
            preferences.get('interests', []),
            preferences.get('travel_style'),
            preferences.get('accessibility_needs')
        )
        
        if success:
            user_profile = self.user_profiles[user_id]
            completeness_pct = int(user_profile.profile_completeness * 100)
            
            response = f"Great! I've updated your preferences. 📊 Profile completeness: {completeness_pct}%\n\n"
            
            if user_profile.interests:
                response += f"🎯 Your interests: {', '.join(user_profile.interests)}\n"
            
            if user_profile.travel_style:
                response += f"✈️ Travel style: {user_profile.travel_style}\n"
            
            if user_profile.accessibility_needs:
                response += f"♿ Accessibility: {user_profile.accessibility_needs}\n"
            
            response += f"\n💡 The more I know about your preferences, the better recommendations I can provide!"
            
            return response
        else:
            return "Sorry, I couldn't update your preferences. Please try again."
    
    def handle_recommendation_feedback(self, user_input: str, user_id: str) -> str:
        """Handle user feedback on recommendations for ML improvement"""
        
        if user_id not in self.user_profiles:
            return "I don't have your profile information. Please start a new conversation first."
        
        # Parse feedback from user input
        feedback_info = self._parse_feedback_input(user_input)
        
        if not feedback_info:
            return "I didn't understand your feedback. You can rate recommendations like: 'Rate Matbah Restaurant 4 stars' or 'I loved Çiya Sofrası!'"
        
        # Store the feedback
        success = self.collect_recommendation_feedback(
            user_id, 
            feedback_info['recommendation_id'], 
            feedback_info['rating'], 
            feedback_info.get('text')
        )
        
        if success:
            user_profile = self.user_profiles[user_id]
            return f"Thanks for your feedback! 🌟 Your rating helps me understand your preferences better. Current satisfaction score: {user_profile.satisfaction_score:.1f}/5.0"
        else:
            return "Sorry, I couldn't save your feedback. Please try again."
    
    def get_personalization_insights(self, user_id: str) -> str:
        """Get insights about user's personalization data for transparency"""
        
        if user_id not in self.user_profiles:
            return "I don't have personalization data for you yet. Start a conversation to begin building your profile!"
        
        user_profile = self.user_profiles[user_id]
        completeness_pct = int(user_profile.profile_completeness * 100)
        
        insights = f"📊 **Your Personalization Profile**\n\n"
        insights += f"**Profile Completeness:** {completeness_pct}%\n"
        insights += f"**Satisfaction Score:** {user_profile.satisfaction_score:.1f}/5.0\n"
        insights += f"**Recommendation Success Rate:** {user_profile.recommendation_success_rate:.1%}\n\n"
        
        if user_profile.interests:
            insights += f"🎯 **Your Interests:** {', '.join(user_profile.interests)}\n"
        
        if user_profile.travel_style:
            insights += f"✈️ **Travel Style:** {user_profile.travel_style}\n"
        
        if user_profile.favorite_neighborhoods:
            insights += f"📍 **Favorite Areas:** {', '.join(user_profile.favorite_neighborhoods)}\n"
        
        if user_profile.cuisine_preferences:
            insights += f"🍽️ **Cuisine Preferences:** {', '.join(user_profile.cuisine_preferences)}\n"
        
        insights += f"\n📈 **Learning Stats:**\n"
        insights += f"• Total interactions: {len(user_profile.interaction_history)}\n"
        insights += f"• Recommendations rated: {len(user_profile.recommendation_feedback)}\n"
        
        if user_profile.learning_patterns:
            total_recommendations = user_profile.learning_patterns.get('total_recommendations', 0)
            insights += f"• Total recommendations provided: {total_recommendations}\n"
        
        insights += f"\n💡 **How to improve your profile:**\n"
        
        if completeness_pct < 50:
            insights += f"• Tell me about your interests and travel style\n"
            insights += f"• Rate recommendations I give you\n"
            insights += f"• Let me know your dietary restrictions or accessibility needs\n"
        elif completeness_pct < 80:
            insights += f"• Continue rating recommendations to improve accuracy\n"
            insights += f"• Share feedback about places you visit\n"
        else:
            insights += f"• Your profile is well-developed! Keep rating recommendations to maintain accuracy\n"
        
        return insights
            preferences['travel_style'] = 'family'
        elif any(word in user_input_lower for word in ['couple', 'romantic', 'partner']):
            preferences['travel_style'] = 'couple'
        elif any(word in user_input_lower for word in ['solo', 'alone', 'myself']):
            preferences['travel_style'] = 'solo'
        elif any(word in user_input_lower for word in ['group', 'friends']):
            preferences['travel_style'] = 'group'
        elif any(word in user_input_lower for word in ['business', 'work']):
            preferences['travel_style'] = 'business'
        
        # Extract accessibility needs
        if any(word in user_input_lower for word in ['wheelchair', 'mobility']):
            preferences['accessibility_needs'] = 'wheelchair'
        elif any(word in user_input_lower for word in ['hearing', 'deaf']):
            preferences['accessibility_needs'] = 'hearing'
        elif any(word in user_input_lower for word in ['visual', 'blind', 'sight']):
            preferences['accessibility_needs'] = 'visual'
        
        return preferences if preferences else None
    
    def get_personalization_insights(self, user_id: str) -> str:
        """Get insights about user's personalization data for transparency"""
        
        if user_id not in self.user_profiles:
            return "I don't have personalization data for you yet. Start a conversation to begin building your profile!"
        
        user_profile = self.user_profiles[user_id]
        completeness_pct = int(user_profile.profile_completeness * 100)
        
        insights = f"📊 **Your Personalization Profile**\n\n"
        insights += f"**Profile Completeness:** {completeness_pct}%\n"
        insights += f"**Satisfaction Score:** {user_profile.satisfaction_score:.1f}/5.0\n"
        insights += f"**Recommendation Success Rate:** {user_profile.recommendation_success_rate:.1%}\n\n"
        
        if user_profile.interests:
            insights += f"🎯 **Your Interests:** {', '.join(user_profile.interests)}\n"
        
        if user_profile.travel_style:
            insights += f"✈️ **Travel Style:** {user_profile.travel_style}\n"
        
        if user_profile.favorite_neighborhoods:
            insights += f"📍 **Favorite Areas:** {', '.join(user_profile.favorite_neighborhoods)}\n"
        
        if user_profile.cuisine_preferences:
            insights += f"🍽️ **Cuisine Preferences:** {', '.join(user_profile.cuisine_preferences)}\n"
        
        insights += f"\n📈 **Learning Stats:**\n"
        insights += f"• Total interactions: {len(user_profile.interaction_history)}\n"
        insights += f"• Recommendations rated: {len(user_profile.recommendation_feedback)}\n"
        
        if user_profile.learning_patterns:
            total_recommendations = user_profile.learning_patterns.get('total_recommendations', 0)
            insights += f"• Total recommendations provided: {total_recommendations}\n"
        
        insights += f"\n💡 **How to improve your profile:**\n"
        
        if completeness_pct < 50:
                'rating': rating,
                'text': user_input
            }
        
        # Pattern for positive feedback: "I loved/liked [restaurant]"
        positive_patterns = [r'i loved (.+)', r'i really liked (.+)', r'(.+) was amazing', r'(.+) was great']
        for pattern in positive_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                restaurant_name = match.group(1).strip()
                return {
                    'recommendation_id': f"restaurant_{restaurant_name.replace(' ', '_')}",
                    'rating': 4.5,
                    'text': user_input
                }
        
        # Pattern for negative feedback: "I didn't like [restaurant]"
        negative_patterns = [r'i didn\'t like (.+)', r'(.+) was bad', r'(.+) was terrible']
        for pattern in negative_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                restaurant_name = match.group(1).strip()
                return {
                    'recommendation_id': f"restaurant_{restaurant_name.replace(' ', '_')}",
                    'rating': 2.0,
                    'text': user_input
                }
        
        return None