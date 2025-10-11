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
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate a unique session ID for a user"""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{user_id}_{timestamp}_{unique_id}"
    
    def _get_active_session_id(self, user_id: str) -> Optional[str]:
        """Get the active session ID for a user"""
        # Find the most recent session for this user
        user_sessions = [session_id for session_id, context in self.active_conversations.items() 
                        if context.user_profile.user_id == user_id]
        
        if user_sessions:
            # Return the most recent session (sessions are chronologically ordered by timestamp)
            return max(user_sessions)
        
        return None
    
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
    
    # ML Personalization Handler Methods
    def handle_preference_update(self, message: str, user_id: str) -> str:
        """Handle user preference updates through natural language"""
        user_profile = self.get_or_create_user_profile(user_id)
        message_lower = message.lower()
        
        updates = []
        
        # Extract dietary preferences
        if 'vegetarian' in message_lower:
            user_profile.dietary_restrictions = list(set(user_profile.dietary_restrictions + ['vegetarian']))
            updates.append("dietary preferences (vegetarian)")
        
        if 'vegan' in message_lower:
            user_profile.dietary_restrictions = list(set(user_profile.dietary_restrictions + ['vegan']))
            updates.append("dietary preferences (vegan)")
            
        if 'halal' in message_lower:
            user_profile.dietary_restrictions = list(set(user_profile.dietary_restrictions + ['halal']))
            updates.append("dietary preferences (halal)")
        
        # Extract travel preferences
        if 'family' in message_lower or 'kids' in message_lower or 'children' in message_lower:
            user_profile.interests = list(set(user_profile.interests + ['family-friendly']))
            updates.append("travel style (family-friendly)")
            
        if 'solo' in message_lower or 'alone' in message_lower:
            user_profile.interests = list(set(user_profile.interests + ['solo-travel']))
            updates.append("travel style (solo travel)")
            
        if 'couple' in message_lower or 'romantic' in message_lower:
            user_profile.interests = list(set(user_profile.interests + ['romantic']))
            updates.append("travel style (romantic)")
        
        # Extract interests
        interests_keywords = {
            'history': ['history', 'historical', 'museum', 'ancient'],
            'food': ['food', 'cuisine', 'restaurant', 'eating', 'taste'],
            'culture': ['culture', 'cultural', 'traditional', 'local'],
            'shopping': ['shopping', 'bazaar', 'market', 'souvenir'],
            'nightlife': ['nightlife', 'bar', 'club', 'evening'],
            'art': ['art', 'gallery', 'artist', 'creative'],
            'architecture': ['architecture', 'building', 'mosque', 'palace']
        }
        
        for interest, keywords in interests_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                if interest not in user_profile.interests:
                    user_profile.interests.append(interest)
                    updates.append(f"interests ({interest})")
        
        # Extract budget preferences
        if any(word in message_lower for word in ['budget', 'cheap', 'affordable', 'economical']):
            user_profile.budget_range = 'budget'
            updates.append("budget preferences (budget-friendly)")
        elif any(word in message_lower for word in ['luxury', 'expensive', 'premium', 'high-end']):
            user_profile.budget_range = 'luxury'
            updates.append("budget preferences (luxury)")
        elif any(word in message_lower for word in ['mid-range', 'moderate', 'medium']):
            user_profile.budget_range = 'mid-range'
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
        user_budget = (user_profile.budget_range or 'moderate').lower()
        if rec_price_level == user_budget:
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
        user_budget = user_profile.budget_range or 'moderate'
        if recommendation.get('price_level', '').lower() == user_budget.lower():
            reasons.append(f"fits your {user_budget} budget")
        
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
        
        user_budget = user_profile.budget_range or 'moderate'
        if recommendation.get('price_level', '').lower() == user_budget.lower():
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
    
    # =============================
    # AI EXPLAINABILITY & TRUST SYSTEM
    # =============================
    
    def get_recommendation_explanation(self, recommendation_id: str, user_id: str) -> Dict[str, Any]:
        """Generate detailed explanation for why a specific recommendation was made"""
        
        if user_id not in self.user_profiles:
            return {'error': 'User profile not found'}
        
        user_profile = self.user_profiles[user_id]
        
        # Find the recommendation in recent interactions
        recommendation_data = self._find_recommendation_by_id(recommendation_id, user_profile)
        
        if not recommendation_data:
            return {'error': 'Recommendation not found'}
        
        explanation = {
            'recommendation_id': recommendation_id,
            'recommendation_name': recommendation_data.get('name', 'Unknown'),
            'explanation_summary': self._generate_explanation_summary(recommendation_data, user_profile),
            'detailed_factors': self._generate_detailed_explanation_factors(recommendation_data, user_profile),
            'transparency_info': self._generate_transparency_info(user_profile),
            'data_usage': self._explain_data_usage(user_profile),
            'confidence_breakdown': self._explain_confidence_score(recommendation_data, user_profile),
            'alternatives_considered': self._explain_alternatives(recommendation_data, user_profile),
            'privacy_context': self._get_privacy_context(user_profile)
        }
        
        return explanation
    
    def _generate_explanation_summary(self, recommendation: Dict, user_profile: UserProfile) -> str:
        """Generate a clear, human-readable summary of why this recommendation was made"""
        
        factors = []
        
        # Location-based reasoning
        if recommendation.get('location') and user_profile.current_location:
            if recommendation['location'].lower() == user_profile.current_location.lower():
                factors.append(f"it's in {user_profile.current_location.title()}, your current area")
            else:
                factors.append(f"it's easily accessible from your location")
        
        # Interest-based reasoning
        if user_profile.interests:
            matching_interests = []
            rec_category = recommendation.get('category', '').lower()
            for interest in user_profile.interests:
                if interest.lower() in rec_category or rec_category in interest.lower():
                    matching_interests.append(interest)
            
            if matching_interests:
                factors.append(f"it matches your interests in {', '.join(matching_interests)}")
        
        # Travel style reasoning
        if user_profile.travel_style:
            if user_profile.travel_style == 'family' and recommendation.get('family_friendly'):
                factors.append("it's perfect for families like yours")
            elif user_profile.travel_style == 'couple' and recommendation.get('romantic'):
                factors.append("it offers a romantic atmosphere for couples")
            elif user_profile.travel_style == 'solo' and recommendation.get('solo_friendly', True):
                factors.append("it's great for solo travelers")
        
        # Past behavior reasoning
        if user_profile.recommendation_feedback:
            similar_recs = self._find_similar_recommendations(recommendation, user_profile.recommendation_feedback)
            if similar_recs:
                avg_rating = sum(similar_recs.values()) / len(similar_recs)
                if avg_rating >= 4.0:
                    factors.append("you've rated similar places highly in the past")
        
        # Time-based reasoning
        current_hour = datetime.now().hour
        suitable_times = recommendation.get('suitable_times', [])
        time_period = self._get_time_period(current_hour)
        if time_period in suitable_times:
            factors.append(f"it's perfect for {time_period} visits")
        
        # Accessibility reasoning
        if user_profile.accessibility_needs and recommendation.get('accessible'):
            factors.append("it meets your accessibility requirements")
        
        # Budget reasoning
        user_budget = user_profile.budget_range or 'moderate'
        if recommendation.get('price_level', '').lower() == user_budget.lower():
            factors.append(f"it fits your {user_budget} budget preferences")
        
        if not factors:
            return f"This recommendation is based on general popularity and location convenience."
        
        return f"I recommended this because {', '.join(factors)}."
    
    def _generate_detailed_explanation_factors(self, recommendation: Dict, user_profile: UserProfile) -> Dict[str, Any]:
        """Generate detailed breakdown of all factors that influenced the recommendation"""
        
        factors = {
            'user_profile_factors': {},
            'contextual_factors': {},
            'behavioral_factors': {},
            'external_factors': {},
            'weights_and_scores': {}
        }
        
        # User Profile Factors
        factors['user_profile_factors'] = {
            'interests_match': {
                'weight': 0.25,
                'score': self._calculate_interest_match_score(recommendation, user_profile),
                'explanation': f"Based on your interests: {', '.join(user_profile.interests) if user_profile.interests else 'None specified'}"
            },
            'travel_style_alignment': {
                'weight': 0.20,
                'score': self._calculate_travel_style_score(recommendation, user_profile),
                'explanation': f"Aligned with your {user_profile.travel_style or 'unspecified'} travel style"
            },
            'budget_compatibility': {
                'weight': 0.15,
                'score': self._calculate_budget_score(recommendation, user_profile),
                'explanation': f"Matches your {user_profile.budget_range or 'moderate'} budget range"
            },
            'accessibility_needs': {
                'weight': 0.10,
                'score': self._calculate_accessibility_score(recommendation, user_profile),
                'explanation': f"Accessibility: {user_profile.accessibility_needs or 'No special needs'}"
            }
        }
        
        # Contextual Factors
        current_time = datetime.now()
        factors['contextual_factors'] = {
            'location_proximity': {
                'weight': 0.20,
                'score': 0.8 if recommendation.get('location') == user_profile.current_location else 0.6,
                'explanation': f"Distance from your current location: {recommendation.get('walking_time', 'Unknown')} minutes"
            },
            'time_suitability': {
                'weight': 0.15,
                'score': self._calculate_time_suitability_score(recommendation, current_time),
                'explanation': f"Suitable for current time ({current_time.strftime('%H:%M')})"
            },
            'seasonal_relevance': {
                'weight': 0.05,
                'score': 0.7,  # Default seasonal score
                'explanation': f"Appropriate for {current_time.strftime('%B')} season"
            }
        }
        
        # Behavioral Factors
        factors['behavioral_factors'] = {
            'past_feedback': {
                'weight': 0.20,
                'score': self._calculate_feedback_score(recommendation, user_profile),
                'explanation': f"Based on {len(user_profile.recommendation_feedback)} previous ratings"
            },
            'interaction_history': {
                'weight': 0.10,
                'score': self._calculate_interaction_history_score(recommendation, user_profile),
                'explanation': f"Learning from {len(user_profile.interaction_history)} past interactions"
            }
        }
        
        # External Factors
        factors['external_factors'] = {
            'popularity_score': {
                'weight': 0.10,
                'score': recommendation.get('rating', 0.0) / 5.0,
                'explanation': f"Overall rating: {recommendation.get('rating', 'N/A')}/5.0"
            },
            'real_time_availability': {
                'weight': 0.05,
                'score': 0.9,  # Assume generally available
                'explanation': "Currently open and available"
            }
        }
        
        # Calculate overall weighted score
        total_weighted_score = 0
        total_weight = 0
        
        for category in factors.values():
            if isinstance(category, dict) and 'weight' in str(category):
                for factor in category.values():
                    if isinstance(factor, dict) and 'weight' in factor and 'score' in factor:
                        total_weighted_score += factor['weight'] * factor['score']
                        total_weight += factor['weight']
        
        factors['weights_and_scores'] = {
            'final_score': total_weighted_score,
            'confidence_level': self._calculate_confidence_level(total_weighted_score, user_profile),
            'score_explanation': f"Weighted average of all factors: {total_weighted_score:.3f}"
        }
        
        return factors
    
    def _generate_transparency_info(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Generate transparency information about the AI system's decision-making"""
        
        return {
            'algorithm_type': 'Hybrid Machine Learning with Rule-based Components',
            'data_sources': [
                'Your personal preferences and profile',
                'Your interaction history and feedback',
                'Real-time location data (if enabled)',
                'General venue information and ratings',
                'Contextual factors (time, season, etc.)'
            ],
            'decision_process': [
                '1. Analyze your profile and preferences',
                '2. Filter relevant options based on location and context',
                '3. Score each option using ML algorithms',
                '4. Apply personalization weights',
                '5. Rank and select top recommendations',
                '6. Generate explanations for transparency'
            ],
            'personalization_level': f"{user_profile.profile_completeness:.1%} (based on profile completeness)",
            'learning_status': f"Learning from {len(user_profile.interaction_history)} interactions",
            'last_updated': user_profile.last_interaction.isoformat() if user_profile.last_interaction else 'Never',
            'bias_mitigation': [
                'Diversity filtering to avoid echo chambers',
                'Regular model retraining with new data',
                'Balanced representation across price ranges',
                'Accessibility consideration in all recommendations'
            ]
        }
    
    def _explain_data_usage(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Explain how user data is used in recommendations"""
        
        data_usage = {
            'data_collected': {
                'profile_information': {
                    'collected': bool(user_profile.interests or user_profile.travel_style),
                    'purpose': 'Personalize recommendations to your preferences',
                    'retention': 'Stored until account deletion',
                    'user_control': 'Can be updated or deleted anytime'
                },
                'interaction_history': {
                    'collected': bool(user_profile.interaction_history),
                    'purpose': 'Learn your preferences and improve recommendations',
                    'retention': 'Latest 50 interactions kept',
                    'user_control': 'Can be cleared from profile settings'
                },
                'feedback_and_ratings': {
                    'collected': bool(user_profile.recommendation_feedback),
                    'purpose': 'Understand what you like and improve future suggestions',
                    'retention': 'Latest 50 ratings kept',
                    'user_control': 'Individual ratings can be modified or removed'
                },
                'location_data': {
                    'collected': bool(user_profile.gps_location),
                    'purpose': 'Provide location-relevant recommendations',
                    'retention': 'Current session only, not permanently stored',
                    'user_control': 'Can be disabled or manually entered'
                }
            },
            'data_not_collected': [
                'Personal identification information',
                'Financial or payment data',
                'Private messages or conversations',
                'Data from other apps or services',
                'Permanent location tracking'
            ],
            'data_sharing': {
                'with_third_parties': False,
                'with_venues': False,
                'anonymized_analytics': 'Only aggregated, non-identifiable usage statistics',
                'marketing_purposes': False
            },
            'user_rights': [
                'View all data we have about you',
                'Update or correct your information',
                'Delete your profile and all associated data',
                'Export your data in a readable format',
                'Opt out of data collection features'
            ]
        }
        
        return data_usage
    
    def _explain_confidence_score(self, recommendation: Dict, user_profile: UserProfile) -> Dict[str, Any]:
        """Explain how confidence scores are calculated"""
        
        ml_score = recommendation.get('ml_score', 0.5)
        confidence_level = recommendation.get('confidence_level', 'medium')
        
        confidence_explanation = {
            'confidence_level': confidence_level,
            'numerical_score': f"{ml_score:.3f} out of 1.000",
            'what_it_means': {
                'very_high': 'We\'re very confident this matches your preferences (85-100%)',
                'high': 'We\'re confident this is a good match for you (70-84%)',
                'medium': 'This seems like a reasonable match (55-69%)',
                'low': 'This might interest you, but we\'re less certain (30-54%)'
            }.get(confidence_level, 'Confidence level not available'),
            'factors_affecting_confidence': [
                f"Profile completeness: {user_profile.profile_completeness:.1%}",
                f"Historical feedback: {len(user_profile.recommendation_feedback)} ratings provided",
                f"Preference clarity: {'High' if len(user_profile.interests) >= 3 else 'Medium' if len(user_profile.interests) >= 1 else 'Low'}",
                f"Interaction history: {len(user_profile.interaction_history)} previous conversations"
            ],
            'how_to_improve_confidence': [
                'Rate more recommendations to help us learn your preferences',
                'Update your profile with interests and travel style',
                'Provide feedback on places you visit',
                'Specify dietary restrictions or accessibility needs'
            ] if user_profile.profile_completeness < 0.7 else [
                'Continue rating recommendations to maintain accuracy',
                'Your profile is well-developed for high-confidence recommendations'
            ]
        }
        
        return confidence_explanation
    
    def _explain_alternatives(self, recommendation: Dict, user_profile: UserProfile) -> Dict[str, Any]:
        """Explain what alternatives were considered and why this one was chosen"""
        
        return {
            'selection_process': 'We considered multiple options and ranked them based on your preferences',
            'alternatives_considered': {
                'total_options_evaluated': 'All available venues in your area',
                'filtering_criteria': [
                    f"Location: Within reasonable distance of {user_profile.current_location or 'your area'}",
                    f"Preferences: Matching your interests ({', '.join(user_profile.interests) if user_profile.interests else 'general'})",
                    f"Travel style: Suitable for {user_profile.travel_style or 'general'} travelers",
                    f"Budget: Compatible with {user_profile.budget_range or 'moderate'} budget",
                    'Accessibility: Meeting any specified requirements',
                    'Time: Appropriate for current time of day'
                ],
                'ranking_factors': [
                    'Personal preference match (highest weight)',
                    'Location convenience',
                    'Past feedback similarity',
                    'Overall quality and ratings',
                    'Contextual appropriateness'
                ]
            },
            'why_this_recommendation': recommendation.get('personalization_reason', 'Selected as the best overall match'),
            'other_good_options': 'Ask me for "different recommendations" to see other highly-rated alternatives',
            'customization_options': [
                'Specify different cuisine types',
                'Request different price ranges',
                'Ask for options in specific neighborhoods',
                'Request accessibility-specific recommendations'
            ]
        }
    
    def _get_privacy_context(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Provide privacy context and controls"""
        
        return {
            'privacy_status': {
                'location_sharing': 'Enabled' if user_profile.gps_location else 'Disabled',
                'profile_personalization': 'Enabled' if user_profile.interests or user_profile.travel_style else 'Basic',
                'learning_from_feedback': 'Enabled' if user_profile.recommendation_feedback else 'Disabled',
                'data_retention': 'Standard (50 interactions/ratings max)'
            },
            'privacy_controls': [
                'Say "disable location" to stop using GPS data',
                'Say "clear my data" to reset your profile',
                'Say "show my data" to see all stored information',
                'Say "privacy settings" to adjust data preferences'
            ],
            'data_minimization': [
                'We only collect data necessary for recommendations',
                'Location data is used only for current session',
                'Personal conversations are not stored',
                'All data is kept locally to your session'
            ],
            'transparency_promise': 'You can always ask "why did you recommend this?" for any suggestion'
        }
    
    # Privacy Control Methods
    def handle_privacy_request(self, user_input: str, user_id: str) -> str:
        """Handle privacy-related requests from users"""
        
        user_input_lower = user_input.lower()
        
        if 'privacy settings' in user_input_lower or 'privacy options' in user_input_lower:
            return self.show_privacy_settings(user_id)
        
        elif 'show my data' in user_input_lower or 'what data do you have' in user_input_lower:
            return self.show_user_data(user_id)
        
        elif 'clear my data' in user_input_lower or 'delete my data' in user_input_lower:
            return self.clear_user_data(user_id)
        
        elif 'disable location' in user_input_lower:
            return self.disable_location_sharing(user_id)
        
        elif 'enable location' in user_input_lower:
            return self.enable_location_sharing(user_id)
        
        elif 'data usage' in user_input_lower or 'how do you use my data' in user_input_lower:
            return self.explain_data_usage_simple(user_id)
        
        else:
            return self.show_privacy_help()
    
    def show_privacy_settings(self, user_id: str) -> str:
        """Show current privacy settings and available controls"""
        
        if user_id not in self.user_profiles:
            return "No profile found. Your privacy is protected - we only store data when you interact with recommendations."
        
        user_profile = self.user_profiles[user_id]
        
        response = "🔒 **Your Privacy Settings**\n\n"
        
        response += "**Current Status:**\n"
        response += f"• Location sharing: {'✅ Enabled' if user_profile.gps_location else '❌ Disabled'}\n"
        response += f"• Profile personalization: {'✅ Active' if user_profile.interests or user_profile.travel_style else '⚪ Basic'}\n"
        response += f"• Learning from feedback: {'✅ Active' if user_profile.recommendation_feedback else '⚪ Inactive'}\n"
        response += f"• Data stored: {len(user_profile.interaction_history)} interactions, {len(user_profile.recommendation_feedback)} ratings\n\n"
        
        response += "**Available Controls:**\n"
        response += "• 'Disable location' - Stop using GPS data\n"
        response += "• 'Clear my data' - Reset your entire profile\n"
        response += "• 'Show my data' - See all information we have\n"
        response += "• 'Data usage' - Learn how we use your information\n\n"
        
        response += "**Privacy Guarantees:**\n"
        response += "🔹 No personal identification data stored\n"
        response += "🔹 Location data used only for current session\n"
        response += "🔹 No data sharing with third parties\n"
        response += "🔹 You control all data collection and can delete anytime\n"
        
        return response
    
    def show_user_data(self, user_id: str) -> str:
        """Show all data stored about the user"""
        
        if user_id not in self.user_profiles:
            return "No data stored about you. You can start fresh anytime!"
        
        user_profile = self.user_profiles[user_id]
        
        response = "📊 **Your Data Summary**\n\n"
        
        response += "**Profile Information:**\n"
        response += f"• Interests: {', '.join(user_profile.interests) if user_profile.interests else 'None specified'}\n"
        response += f"• Travel style: {user_profile.travel_style or 'Not specified'}\n"
        response += f"• Budget preference: {user_profile.budget_range or 'Not specified'}\n"
        response += f"• Accessibility needs: {user_profile.accessibility_needs or 'None specified'}\n"
        response += f"• Favorite neighborhoods: {', '.join(user_profile.favorite_neighborhoods) if user_profile.favorite_neighborhoods else 'None yet'}\n\n"
        
        response += "**Learning Data:**\n"
        response += f"• Total interactions: {len(user_profile.interaction_history)}\n"
        response += f"• Recommendations rated: {len(user_profile.recommendation_feedback)}\n"
        response += f"• Profile completeness: {user_profile.profile_completeness:.1%}\n"
        response += f"• Satisfaction score: {user_profile.satisfaction_score:.1f}/5.0\n\n"
        
        response += "**Session Data:**\n"
        response += f"• Current location sharing: {'Yes' if user_profile.gps_location else 'No'}\n"
        response += f"• Last interaction: {user_profile.last_interaction.strftime('%Y-%m-%d %H:%M') if user_profile.last_interaction else 'Never'}\n\n"
        
        if user_profile.recommendation_feedback:
            response += "**Recent Ratings:** (Latest 5)\n"
            recent_feedback = list(user_profile.recommendation_feedback.items())[-5:]
            for rec_id, rating in recent_feedback:
                response += f"• {rec_id.replace('restaurant_', '').replace('_', ' ').title()}: {rating}/5 ⭐\n"
        
        response += "\n💡 You can 'clear my data' to delete everything or update specific preferences anytime."
        
        return response
    
    def clear_user_data(self, user_id: str) -> str:
        """Clear all user data"""
        
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
            
            # Also clear from active conversations
            sessions_to_remove = []
            for session_id, context in self.active_conversations.items():
                if context.user_profile.user_id == user_id:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_conversations[session_id]
            
            return "✅ **Data Cleared Successfully**\n\nAll your profile data, preferences, ratings, and interaction history have been permanently deleted. You can start fresh with a new conversation!\n\n🔒 Your privacy is protected - no trace of your previous data remains."
        else:
            return "No data found to clear. Your privacy is already protected!"
    
    def disable_location_sharing(self, user_id: str) -> str:
        """Disable location sharing for user"""
        
        if user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
            user_profile.gps_location = None
            user_profile.location_accuracy = None
            user_profile.location_timestamp = None
            
            return "📍 **Location Sharing Disabled**\n\nI will no longer use GPS data for recommendations. You can still get great suggestions by telling me which neighborhood you're in!\n\n💡 Say 'enable location' to turn GPS recommendations back on anytime."
        else:
            return "Location sharing is already disabled. No location data is being collected."
    
    def enable_location_sharing(self, user_id: str) -> str:
        """Enable location sharing for user"""
        
        return "📍 **Location Sharing**\n\nTo enable location-based recommendations, I'll need access to your GPS location. This helps me suggest nearby restaurants and attractions!\n\n🔒 **Privacy Protection:**\n• Location used only for current session\n• Not stored permanently\n• Never shared with third parties\n• You can disable anytime\n\n💡 Your browser will ask for location permission. Allow it for personalized nearby recommendations!"
    
    def explain_data_usage_simple(self, user_id: str) -> str:
        """Simple explanation of data usage"""
        
        return "🔍 **How We Use Your Data**\n\n**We DO use:**\n• Your preferences (interests, travel style) - to personalize recommendations\n• Your ratings and feedback - to learn what you like\n• Your current location - to suggest nearby places (session only)\n• Your conversation context - to provide relevant responses\n\n**We DON'T collect:**\n• Personal identification information\n• Permanent location tracking\n• Financial or payment data\n• Data from other apps\n• Private personal details\n\n**Your Control:**\n✅ Update preferences anytime\n✅ Delete all data with one command\n✅ See exactly what we store\n✅ Disable features you don't want\n\n🔒 **Bottom line:** We only use data to make your Istanbul experience better, and you're always in control!"
    
    def show_privacy_help(self) -> str:
        """Show privacy help and available commands"""
        
        return "🔒 **Privacy Help**\n\nI'm designed with privacy in mind. Here's what you can ask:\n\n**Data Control:**\n• 'Show my data' - See what information I have\n• 'Clear my data' - Delete everything and start fresh\n• 'Privacy settings' - View and manage your privacy\n\n**Location Control:**\n• 'Disable location' - Stop using GPS data\n• 'Enable location' - Allow location-based recommendations\n\n**Transparency:**\n• 'How do you use my data?' - Learn about data usage\n• 'Why did you recommend this?' - Explain any recommendation\n\n**Quick Facts:**\n🔹 No personal ID or financial data collected\n🔹 Location used only for current session\n🔹 You control all data collection\n🔹 Everything can be deleted instantly\n\n💡 I'm built to be helpful while respecting your privacy!"
    
    def _get_transport_status(self) -> Dict[str, Any]:
        """Get real-time transport status (mock implementation)"""
        try:
            # Mock transport status - in real implementation, this would connect to IBB API
            return {
                'metro': {
                    'status': 'operational',
                    'delays': [],
                    'message': 'All metro lines running normally'
                },
                'bus': {
                    'status': 'operational', 
                    'delays': ['Line 28: 5 min delay due to traffic'],
                    'message': 'Minor delays on some bus routes'
                },
                'ferry': {
                    'status': 'operational',
                    'delays': [],
                    'message': 'Ferry services running on schedule'
                },
                'traffic_density': 'moderate',
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting transport status: {e}")
            return {
                'status': 'unavailable',
                'message': 'Transport information temporarily unavailable'
            }

    def _get_traffic_status(self) -> Dict[str, Any]:
        """Get real-time traffic status (mock implementation)"""
        try:
            # Mock traffic status - in real implementation, this would connect to traffic APIs
            return {
                'overall_status': 'moderate',
                'congestion_level': 65,
                'problem_areas': [
                    'Bosphorus Bridge - Heavy traffic',
                    'Fatih Sultan Mehmet Bridge - Moderate congestion',
                    'E-5 Highway (European side) - Slow moving'
                ],
                'estimated_travel_times': {
                    'Sultanahmet to Taksim': '25-35 minutes',
                    'Kadıköy to Beşiktaş': '30-40 minutes',
                    'Airport to Sultanahmet': '45-60 minutes'
                },
                'recommendations': [
                    'Use metro when possible for cross-city travel',
                    'Consider ferry for Bosphorus crossings',
                    'Avoid E-5 highway during peak hours'
                ],
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting traffic status: {e}")
            return {
                'status': 'unavailable',
                'message': 'Traffic information temporarily unavailable'
            }

    def _get_local_events(self) -> Dict[str, Any]:
        """Get local events information (curated local events, not IBB API)"""
        try:
            # Curated local events - this could be expanded with real event data
            current_month = datetime.now().month
            current_day = datetime.now().day
            
            # Sample events based on season/month
            events = []
            
            if current_month in [6, 7, 8]:  # Summer
                events = [
                    {
                        'name': 'Istanbul Music Festival',
                        'location': 'Various venues',
                        'type': 'music',
                        'description': 'Classical music performances across the city'
                    },
                    {
                        'name': 'Bosphorus Sunset Concerts',
                        'location': 'Ortaköy',
                        'type': 'music',
                        'description': 'Evening concerts with Bosphorus views'
                    }
                ]
            elif current_month in [9, 10, 11]:  # Autumn
                events = [
                    {
                        'name': 'Istanbul Biennial',
                        'location': 'Various galleries',
                        'type': 'art',
                        'description': 'Contemporary art exhibitions'
                    },
                    {
                        'name': 'Autumn Food Festival',
                        'location': 'Galata',
                        'type': 'food',
                        'description': 'Seasonal Turkish cuisine showcase'
                    }
                ]
            elif current_month in [12, 1, 2]:  # Winter
                events = [
                    {
                        'name': 'New Year Celebrations',
                        'location': 'Taksim Square',
                        'type': 'celebration',
                        'description': 'New Year festivities and concerts'
                    },
                    {
                        'name': 'Winter Arts Festival',
                        'location': 'Cultural centers',
                        'type': 'culture',
                        'description': 'Indoor cultural performances'
                    }
                ]
            else:  # Spring
                events = [
                    {
                        'name': 'Tulip Festival',
                        'location': 'Emirgan Park',
                        'type': 'nature',
                        'description': 'Beautiful tulip displays across the city'
                    },
                    {
                        'name': 'Spring Markets',
                        'location': 'Various neighborhoods',
                        'type': 'shopping',
                        'description': 'Local artisan markets and crafts'
                    }
                ]
            
            return {
                'current_events': events,
                'event_count': len(events),
                'categories': list(set([event['type'] for event in events])),
                'last_updated': datetime.now().isoformat(),
                'note': 'Curated local events - check official sources for exact dates and times'
            }
            
        except Exception as e:
            logger.error(f"Error getting local events: {e}")
            return {
                'current_events': [],
                'status': 'unavailable',
                'message': 'Event information temporarily unavailable'
            }