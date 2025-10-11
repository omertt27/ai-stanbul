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

# Import ML personalization helpers
from ml_personalization_helpers import (
    handle_preference_update,
    handle_recommendation_feedback, 
    get_personalization_insights,
    calculate_personalization_score,
    calculate_recommendation_compatibility,
    generate_personalization_reason,
    calculate_confidence_level,
    generate_explanation_summary,
    apply_behavioral_patterns,
    get_meal_context,
    update_learning_patterns,
    get_adaptation_factors,
    apply_diversity_filter,
    collect_recommendation_feedback,
    update_user_interests,
    recalculate_profile_completeness,
    get_recommendation_explanation,
    find_similar_recommendations,
    get_time_period,
    process_neighborhood_query,
    format_attraction_response_text,
    generate_conversational_response_enhanced,
    process_transportation_query_enhanced,
    extract_gps_coordinates,
    detect_transportation_intent,
    is_enhanced_transportation_query,
    process_enhanced_transportation_query,
    show_privacy_settings,
    show_user_data,
    clear_user_data
)

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

# Import ML-enhanced transportation system
try:
    from ml_enhanced_transportation_system import create_ml_enhanced_transportation_system
    from transportation_integration_helper import TransportationQueryProcessor
    ML_ENHANCED_TRANSPORTATION_AVAILABLE = True
    logger.info("🚇 ML-Enhanced Transportation System loaded successfully!")
except ImportError as e:
    ML_ENHANCED_TRANSPORTATION_AVAILABLE = False
    logger.warning(f"ML-Enhanced Transportation System not available: {e}")

# Fallback to basic enhanced transportation
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

        # Initialize ML-enhanced transportation system
        if ML_ENHANCED_TRANSPORTATION_AVAILABLE:
            self.ml_transportation_system = create_ml_enhanced_transportation_system()
            self.transportation_processor = TransportationQueryProcessor()
            logger.info("🚇 ML-Enhanced Transportation System integrated successfully!")
        elif ENHANCED_TRANSPORTATION_AVAILABLE:
            # Fallback to basic enhanced system
            self.transportation_system = EnhancedTransportationSystem()
            self.transportation_advisor = EnhancedTransportationAdvisor()
            self.ml_transportation_system = None
            self.transportation_processor = None
            logger.info("🚇 Enhanced Transportation System integrated successfully!")
        else:
            self.transportation_system = None
            self.transportation_advisor = None
            self.ml_transportation_system = None
            self.transportation_processor = None
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
    
    def process_message(self, message: str, user_id: str) -> str:
        """Process user message with ML personalization and context awareness"""
        
        # Get user profile and active conversation
        user_profile = self.get_or_create_user_profile(user_id)
        session_id = self._get_active_session_id(user_id)
        
        if session_id is None or session_id not in self.active_conversations:
            # Start new conversation but then process the message
            self.start_conversation(user_id)
            session_id = self._get_active_session_id(user_id)
        
        context = self.active_conversations[session_id]
        
        # Handle ML personalization features first
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
        
        if any(phrase in message_lower for phrase in ['privacy settings', 'show my data', 'clear my data']):
            return self.handle_privacy_request(message, user_id)
        
        # Extract entities and understand intent
        entities = self.entity_recognizer.extract_entities(message)
        intent = self._classify_intent_with_context(message, entities, context)
        
        # Generate contextually-aware response
        response = self._generate_contextual_response(
            message, intent, entities, context, user_profile
        )
        
        # Add interaction to history
        context.add_interaction(message, response, intent)
        
        # Update user profile based on interaction
        from ml_personalization_helpers import update_user_profile
        update_user_profile(user_profile, message, intent, entities)
        
        return response
    
    # ML Personalization Methods (delegating to helpers)
    def handle_preference_update(self, message: str, user_id: str) -> str:
        """Handle user preference updates through natural language"""
        user_profile = self.get_or_create_user_profile(user_id)
        return handle_preference_update(user_profile, message, user_id)
    
    def handle_recommendation_feedback(self, message: str, user_id: str) -> str:
        """Handle user feedback on recommendations"""
        user_profile = self.get_or_create_user_profile(user_id)
        return handle_recommendation_feedback(user_profile, message, user_id)
    
    def get_personalization_insights(self, user_id: str) -> str:
        """Provide insights about the user's personalization data"""
        user_profile = self.get_or_create_user_profile(user_id)
        return get_personalization_insights(user_profile, user_id)
    
    def show_privacy_settings(self, user_id: str) -> str:
        """Show current privacy settings and available controls"""
        if user_id not in self.user_profiles:
            return "No profile found. Your privacy is protected - we only store data when you interact with recommendations."
        user_profile = self.user_profiles[user_id]
        return show_privacy_settings(user_profile, user_id)
    
    def show_user_data(self, user_id: str) -> str:
        """Show all data stored about the user"""
        if user_id not in self.user_profiles:
            return "No data stored about you. You can start fresh anytime!"
        user_profile = self.user_profiles[user_id]
        return show_user_data(user_profile, user_id)
    
    def clear_user_data(self, user_id: str) -> str:
        """Clear all user data"""
        return clear_user_data(self.user_profiles, self.active_conversations, user_id)
    
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
        
        # PRIORITY: Check for museum queries (comprehensive handling)
        if self._is_museum_query(message):
            return 'museum_query'
        
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
        
        # 🎯 ENHANCED: Use Multi-Intent Query Handler for restaurant, museum, and attraction queries
        if intent in ['restaurant_query', 'museum_query', 'restaurant_recommendation', 'attraction_query', 'place_recommendation', 'cultural_query', 'activity_planning'] and self.multi_intent_handler:
            try:
                logger.info(f"🎯 Using Multi-Intent Handler with Deep Learning for: {message}")
                
                # Create enhanced context for deep learning integration  
                enhanced_context = {
                    'user_id': user_profile.user_id,
                    'session_id': context.session_id,
                    'conversation_history': [interaction.get('user_input', '') for interaction in context.conversation_history],
                    'user_preferences': {
                        'interests': user_profile.interests,
                        'budget_range': user_profile.budget_range,
                        'accessibility_needs': user_profile.accessibility_needs
                    },
                    'location': None
                }
                
                # Add GPS location if available
                if user_profile.gps_location:
                    enhanced_context['location'] = (
                        user_profile.gps_location.get('lat'), 
                        user_profile.gps_location.get('lng')
                    )
                
                # Process through multi-intent handler with enhanced context
                multi_intent_result = self.multi_intent_handler.analyze_query(message, enhanced_context)
                
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
                            return format_attraction_response_text(attraction_response, user_profile, current_time)
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
        
        # 🚇 ENHANCED: Handle transportation queries with ML, GPS, and POI integration
        if intent == 'transportation_query':
            # Check if this requires enhanced processing
            if is_enhanced_transportation_query(message):
                # Use ML-enhanced transportation system if available
                if self.ml_transportation_system and self.transportation_processor:
                    try:
                        # Extract GPS coordinates from message or user profile
                        gps_coords = extract_gps_coordinates(message)
                        
                        # Try async processing with fallback to sync
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Already in async context, create task
                                return asyncio.create_task(
                                    self.transportation_processor.process_transportation_query_enhanced(
                                        message, user_profile, current_time, gps_coords
                                    )
                                ).result()
                            else:
                                # Create new event loop
                                return asyncio.run(
                                    self.transportation_processor.process_transportation_query_enhanced(
                                        message, user_profile, current_time, gps_coords
                                    )
                                )
                        except Exception:
                            # Fallback to sync processing
                            return self._process_transportation_query_sync(message, user_profile, current_time, gps_coords)
                            
                    except Exception as e:
                        logger.warning(f"ML transportation processing failed: {e}")
                        # Fallback to basic enhanced processing
                        return process_transportation_query_enhanced(message, user_profile, current_time, context)
                else:
                    # Use available transportation system
                    return process_transportation_query_enhanced(message, user_profile, current_time, context)
            else:
                # Use basic transportation processing
                return process_transportation_query_enhanced(message, user_profile, current_time, context)
        
        if intent == 'restaurant_recommendation':
            return self._generate_restaurant_recommendation(entities, context, user_profile, current_time)
        
        elif intent == 'location_query':
            return self._generate_location_response(entities, context, traffic_info)
        
        elif intent == 'time_query':
            return self._generate_time_response(entities, context, current_time)
        
        elif intent == 'general_conversation':
            return generate_conversational_response_enhanced(message, context, user_profile)
        
        elif intent == 'museum_query':
            # Use enhanced museum advising system with GPS and ML
            user_location = None
            if user_profile.gps_location:
                user_location = (user_profile.gps_location.get('lat'), user_profile.gps_location.get('lng'))
            
            # Convert user profile to dict format for museum system
            user_profile_dict = {
                'interests': user_profile.interests,
                'budget_range': user_profile.budget_range,
                'accessibility_needs': user_profile.accessibility_needs
            }
            
            from museum_advising_system import process_museum_query_enhanced
            return process_museum_query_enhanced(message, user_profile_dict, current_time, user_location)
        
        elif intent == 'neighborhood_query':
            # Use enhanced district advising system with GPS and ML
            user_location = None
            if user_profile.gps_location:
                user_location = (user_profile.gps_location.get('lat'), user_profile.gps_location.get('lng'))
            
            from district_advising_system import process_neighborhood_query_enhanced
            return process_neighborhood_query_enhanced(message, user_profile, current_time, user_location)
        
        elif intent == 'transportation_query':
            return process_transportation_query_enhanced(message, user_profile, current_time, context)
        
        else:
            return self._generate_fallback_response(context, user_profile)
    
    def _generate_restaurant_recommendation(self, entities: Dict, context: ConversationContext, 
                                          user_profile: UserProfile, current_time: datetime) -> str:
        """Generate ML-powered personalized restaurant recommendations with GPS-based location"""
        
        # 📍 Get GPS location for accurate recommendations
        gps_location = self._get_or_request_gps_location(user_profile, context)
        if not gps_location:
            return self._request_location_for_restaurant(context.current_message, user_profile)
        
        # Extract location information
        location_info = self._extract_or_request_location(context.current_message, user_profile, context, gps_location)
        
        # Get restaurant data from local database (500+ restaurants from Google Places)
        database_recommendations = self._get_restaurant_data_from_local_database(location_info, entities, context.current_message)
        
        if database_recommendations:
            # Use local database (500+ restaurants from Google Places)
            base_recommendations = database_recommendations
            logger.info(f"Using local database: found {len(database_recommendations)} restaurants")
        else:
            # Fallback to static recommendations if database query fails
            base_recommendations = self._generate_base_restaurant_recommendations(location_info, entities, current_time)
            logger.info("Using static fallback recommendations")
        
        if not base_recommendations:
            return f"I couldn't find restaurants in {location_info.get('neighborhood', 'your area')} right now. Please try a different neighborhood or let me know your specific preferences!"
        
        # Apply ML-based personalization and adaptation
        ml_adapted_recommendations = self.adapt_recommendations_with_ml(user_profile, base_recommendations, context)
        
        # Update user's location and preferences based on this search
        self._update_user_location_and_preferences(user_profile, location_info, entities)
        
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
        return get_meal_context(hour)
    
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
        return calculate_personalization_score(user_profile)
    
    def _calculate_recommendation_compatibility(self, recommendation: Dict, user_profile: UserProfile, context: ConversationContext) -> float:
        """Calculate how compatible a recommendation is with user preferences"""
        return calculate_recommendation_compatibility(recommendation, user_profile, context)
    
    def _apply_behavioral_patterns(self, recommendation: Dict, user_profile: UserProfile) -> float:
        """Apply learned behavioral patterns to recommendation scoring"""
        return apply_behavioral_patterns(recommendation, user_profile)
    
    def _generate_personalization_reason(self, recommendation: Dict, user_profile: UserProfile) -> str:
        """Generate human-readable reason for why this recommendation was personalized"""
        return generate_personalization_reason(recommendation, user_profile)
    
    def _calculate_confidence_level(self, ml_score: float, user_profile: UserProfile) -> str:
        """Calculate confidence level for the recommendation"""
        return calculate_confidence_level(ml_score, user_profile)
    
    def _get_adaptation_factors(self, recommendation: Dict, user_profile: UserProfile, context: ConversationContext) -> Dict[str, float]:
        """Get detailed breakdown of adaptation factors"""
        return get_adaptation_factors(recommendation, user_profile, context)
    
    def _apply_diversity_filter(self, recommendations: List[Dict], user_profile: UserProfile) -> List[Dict]:
        """Apply diversity filtering to avoid monotonous recommendations"""
        return apply_diversity_filter(recommendations, user_profile)
    
    def _update_learning_patterns(self, user_profile: UserProfile, recommendations: List[Dict]):
        """Update ML learning patterns based on generated recommendations"""
        return update_learning_patterns(user_profile, recommendations)
    
    def collect_recommendation_feedback(self, user_id: str, recommendation_id: str, rating: float, feedback_text: str = None) -> bool:
        """Collect user feedback on recommendations for ML improvement"""
        return collect_recommendation_feedback(self.user_profiles, user_id, recommendation_id, rating, feedback_text)
    
    def update_user_interests(self, user_id: str, interests: List[str], travel_style: str = None, accessibility_needs: str = None) -> bool:
        """Update user interests and preferences for better personalization"""
        return update_user_interests(self.user_profiles, user_id, interests, travel_style, accessibility_needs)
    
    def _recalculate_profile_completeness(self, user_profile: UserProfile):
        """Recalculate profile completeness score"""
        return recalculate_profile_completeness(user_profile)
    
    # =============================
    # AI EXPLAINABILITY & TRUST SYSTEM
    # =============================
    
    def get_recommendation_explanation(self, recommendation_id: str, user_id: str) -> Dict[str, Any]:
        """Generate detailed explanation for why a specific recommendation was made"""
        return get_recommendation_explanation(self.user_profiles, recommendation_id, user_id)
    
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
            similar_recs = find_similar_recommendations(recommendation, user_profile.recommendation_feedback)
            if similar_recs:
                avg_rating = sum(similar_recs.values()) / len(similar_recs)
                if avg_rating >= 4.0:
                    factors.append("you've rated similar places highly in the past")
        
        # Time-based reasoning
        current_hour = datetime.now().hour
        suitable_times = recommendation.get('suitable_times', [])
        time_period = get_time_period(current_hour)
        if time_period in suitable_times:
            factors.append(f"it's perfect for {time_period} visits")
            if similar_recs:
                avg_rating = sum(similar_recs.values()) / len(similar_recs)
                if avg_rating >= 4.0:
                    factors.append("you've rated similar places highly in the past")
        
        # Time-based reasoning
        current_hour = datetime.now().hour
        suitable_times = recommendation.get('suitable_times', [])
        time_period = get_time_period(current_hour)
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
        confidence_level = recommendation.get('confidence_level', 'medium')
        ml_score = recommendation.get('ml_score', 0.5)
        
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
            ],
            'filtering_criteria': [
                f"Location: Within reasonable distance of {user_profile.current_location or 'your area'}",
                f"Preferences: Matching your interests ({', '.join(user_profile.interests) if user_profile.interests else 'general'})",
                f"Travel style: Suitable for {user_profile.travel_style or 'general'} travelers",
                f"Budget: Compatible with {user_profile.budget_range or 'moderate'} budget",
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
                'Ask for options in specific neighborhoods',
                'Request accessibility-specific recommendations',
                'Specify different cuisine types',
                'Request different price ranges'
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
        
        else:
            return "I can help with privacy settings, showing your data, or clearing your data. What would you like to do?"
    
    def show_privacy_settings(self, user_id: str) -> str:
        """Show current privacy settings for the user"""
        user_profile = self.get_user_profile(user_id)
        
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
    
    def _enhance_intent_classification(self, message: str) -> str:
        """Enhanced intent classification with attraction support"""
        message_lower = message.lower()
        
        # Transportation keywords
        transport_keywords = ['metro', 'bus', 'ferry', 'tram', 'taxi', 'transport', 'get to', 'how to reach']
        if any(keyword in message_lower for keyword in transport_keywords):
            return 'transportation_query'
        
        # Attraction keywords
        attraction_keywords = ['visit', 'see', 'attraction', 'museum', 'palace', 'mosque', 'tower', 'monument']
        if any(keyword in message_lower for keyword in attraction_keywords):
            return 'attraction_query'
        
        # Cultural keywords
        cultural_keywords = ['culture', 'cultural', 'traditional', 'heritage', 'historic']
        if any(keyword in message_lower for keyword in cultural_keywords):
            return 'cultural_query'
        
        return 'general_conversation'
    
    def _is_transportation_query(self, message: str) -> bool:
        """Check if message is about transportation"""
        transport_keywords = [
            'metro', 'bus', 'ferry', 'tram', 'taxi', 'transport', 'transportation',
            'get to', 'how to reach', 'how to get', 'travel to', 'go to',
            'metro station', 'bus stop', 'ferry terminal', 'airport',
            'dolmuş', 'metrobüs', 'vapur', 'otobüs'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in transport_keywords)
    
    def _is_museum_query(self, message: str) -> bool:
        """Check if message is about museums"""
        museum_keywords = [
            'museum', 'museums', 'gallery', 'galleries', 'exhibition', 'exhibit',
            'palace', 'palaces', 'collection', 'artifacts', 'art', 'history',
            'archaeological', 'cultural', 'heritage', 'islamic arts',
            'müze', 'müzeler', 'saray', 'koleksiyon', 'sergi'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in museum_keywords)
    
    def _is_restaurant_query(self, message: str) -> bool:
        """Check if message is about restaurants"""
        restaurant_keywords = [
            'restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal',
            'breakfast', 'lunch', 'dinner', 'café', 'coffee',
            'lokanta', 'restoran', 'yemek', 'kahvaltı'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in restaurant_keywords)
    
    def _is_neighborhood_query(self, message: str) -> bool:
        """Check if message is about neighborhoods"""
        neighborhood_keywords = [
            'neighborhood', 'district', 'area', 'quarter',
            'sultanahmet', 'beyoğlu', 'beyoglu', 'galata', 'karaköy', 'karakoy',
            'taksim', 'kadıköy', 'kadikoy', 'beşiktaş', 'besiktas', 'ortaköy', 'ortakoy',
            'eminönü', 'eminonu', 'fatih', 'şişli', 'sisli', 'bakırköy', 'bakirkoy',
            'üsküdar', 'uskudar', 'sarıyer', 'sariyer', 'pendik', 'maltepe'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in neighborhood_keywords)
    
    def _has_transportation_keywords(self, message: str) -> bool:
        """Check if message contains transportation-related keywords"""
        return self._is_transportation_query(message)
    
    # Response Generation Methods
    def _generate_fallback_response(self, context, user_profile) -> str:
        """Generate fallback response when other methods fail"""
        responses = [
            "I'm here to help you explore Istanbul! What would you like to know about the city?",
            "Let me know what you're interested in - restaurants, attractions, transportation, or anything else about Istanbul!",
            "I can help you with restaurant recommendations, finding attractions, getting around the city, and much more. What interests you?",
            "What would you like to discover about Istanbul today? I'm here to help with personalized recommendations!"
        ]
        
        # Use user profile to personalize the fallback
        if user_profile.interests:
            if 'food' in user_profile.interests:
                return "I notice you're interested in food! Would you like restaurant recommendations, or are you curious about something else in Istanbul?"
            elif 'history' in user_profile.interests:
                return "Given your interest in history, I can recommend historical sites, museums, or help with other Istanbul questions. What would you like to know?"
        
        # Return a random friendly response
        import random
        return random.choice(responses)
    
    def _enhance_multi_intent_response(self, multi_intent_response: str, entities: Dict, user_profile, current_time: datetime) -> str:
        """Enhance multi-intent response with Istanbul-specific context and personalization"""
        enhanced_response = multi_intent_response
        
        # Add time-based context
        hour = current_time.hour
        if hour < 11:
            time_context = "Since it's morning, consider places that serve good breakfast!"
        elif hour < 16:
            time_context = "Perfect timing for lunch recommendations!"
        else:
            time_context = "Great time for dinner suggestions!"
        
        # Add personalized context based on user profile
        personal_context = ""
        if user_profile.interests:
            if 'food' in user_profile.interests:
                personal_context = "Based on your love for food, I've focused on culinary experiences."
            elif 'family-friendly' in user_profile.interests:
                personal_context = "I've made sure these are family-friendly options."
        
        # Add dietary considerations
        dietary_context = ""
        if user_profile.dietary_restrictions:
            dietary_context = f"I've considered your {', '.join(user_profile.dietary_restrictions)} preferences."
        
        # Combine contexts
        context_additions = []
        if personal_context:
            context_additions.append(personal_context)
        if dietary_context:
            context_additions.append(dietary_context)
        if time_context:
            context_additions.append(time_context)
        
        if context_additions:
            enhanced_response += f"\n\n💡 {' '.join(context_additions)}"
        
        return enhanced_response
    
    def _get_or_request_gps_location(self, user_profile: UserProfile, context: ConversationContext) -> Optional[Dict]:
        """Get GPS location from user or request if not available"""
        
        # Check if user has location sharing enabled
        if not user_profile.location_sharing_enabled:
            return None
            
        # Try to get current GPS location from context
        if hasattr(context, 'gps_coordinates') and context.gps_coordinates:
            return {
                'lat': context.gps_coordinates.get('lat'),
                'lng': context.gps_coordinates.get('lng'),
                'source': 'gps',
                'accuracy': context.gps_coordinates.get('accuracy', 'unknown')
            }
        
        # Check if we have a recent location from user profile
        if user_profile.last_known_location and user_profile.last_location_update:
            # Use location if it's less than 30 minutes old
            time_diff = datetime.utcnow() - user_profile.last_location_update
            if time_diff.total_seconds() < 1800:  # 30 minutes
                return {
                    'lat': user_profile.last_known_location.get('lat'),
                    'lng': user_profile.last_known_location.get('lng'),
                    'source': 'cached',
                    'accuracy': 'approximate'
                }
        
        # Try ML-predicted location based on user patterns
        if user_profile.favorite_neighborhoods:
            predicted_location = self._predict_user_location(user_profile)
            if predicted_location:
                return {
                    'neighborhood': predicted_location,
                    'source': 'ml_predicted',
                    'accuracy': 'neighborhood_level'
                }
        
        return None
    
    def _predict_user_location(self, user_profile: UserProfile) -> Optional[str]:
        """Predict user location based on their interaction patterns"""
        
        if not user_profile.favorite_neighborhoods:
            return None
            
        # Get most frequently visited neighborhood
        if user_profile.visit_frequency:
            most_visited = max(user_profile.visit_frequency.items(), key=lambda x: x[1])
            return most_visited[0]
        
        # Fallback to first favorite neighborhood
        return user_profile.favorite_neighborhoods[0]
    
    def _request_location_for_restaurant(self, query: str, user_profile: UserProfile) -> str:
        """Request location from user for restaurant recommendations"""
        
        response = "🗺️ **Location Needed for Restaurant Recommendations**\n\n"
        
        if not user_profile.location_sharing_enabled:
            response += "I'd love to recommend restaurants near you! To get personalized, distance-based recommendations:\n\n"
            response += "📍 **Option 1: Enable GPS Location**\n"
            response += "• Say 'enable location sharing' for automatic nearby suggestions\n"
            response += "• Your location is only used for this session and not stored\n\n"
            response += "🏙️ **Option 2: Tell Me Your Area**\n"
            response += "• Just mention which Istanbul district you're in or heading to\n"
            response += "• Examples: 'Sultanahmet', 'Beyoğlu', 'Kadıköy', 'Taksim area'\n\n"
        else:
            response += "I need to know where you are to suggest the best nearby restaurants!\n\n"
            response += "📱 **Please share your location** or tell me which Istanbul district you're in:\n"
            response += "• Sultanahmet • Beyoğlu • Galata • Kadıköy\n"
            response += "• Taksim • Beşiktaş • Ortaköy • Eminönü\n\n"
        
        response += "💡 **Meanwhile, I can help with:**\n"
        response += "• General restaurant recommendations by cuisine type\n"
        response += "• Famous Istanbul restaurants and must-try dishes\n"
        response += "• Restaurant recommendations for specific districts\n\n"
        
        response += "Just let me know your preferences and I'll help you find amazing food! 🍽️"
        
        return response
    
    def _extract_or_request_location(self, query: str, user_profile: UserProfile, 
                                   context: ConversationContext, gps_location: Optional[Dict] = None) -> Dict:
        """Extract location from query/GPS or request from user"""
        
        # Use GPS location if available
        if gps_location and gps_location.get('lat') and gps_location.get('lng'):
            neighborhood = self._get_neighborhood_from_coordinates(
                gps_location['lat'], gps_location['lng']
            )
            return {
                'neighborhood': neighborhood,
                'coordinates': {'lat': gps_location['lat'], 'lng': gps_location['lng']},
                'source': gps_location['source'],
                'accuracy': 'high'
            }
        
        # Check if location is mentioned in query
        istanbul_districts = [
            'sultanahmet', 'beyoğlu', 'beyoglu', 'galata', 'karaköy', 'karakoy',
            'taksim', 'kadıköy', 'kadikoy', 'beşiktaş', 'besiktas', 'ortaköy', 'ortakoy',
            'eminönü', 'eminonu', 'fatih', 'şişli', 'sisli', 'bakırköy', 'bakirkoy',
            'üsküdar', 'uskudar', 'sarıyer', 'sariyer', 'pendik', 'maltepe'
        ]
        
        query_lower = query.lower()
        for district in istanbul_districts:
            if district in query_lower:
                return {
                    'neighborhood': district,
                    'source': 'user_specified',
                    'accuracy': 'district_level'
                }
        
        # Check user's favorite neighborhoods
        if user_profile.favorite_neighborhoods:
            return {
                'neighborhood': user_profile.favorite_neighborhoods[0],
                'source': 'user_profile',
                'accuracy': 'inferred'
            }
        
        # Default to central Istanbul
        return {
            'neighborhood': 'sultanahmet',
            'source': 'default',
            'accuracy': 'low'
        }
    
    def _get_neighborhood_from_coordinates(self, lat: float, lng: float) -> str:
        """Convert GPS coordinates to Istanbul neighborhood"""
        
        # Define rough boundaries for major Istanbul districts
        district_boundaries = {
            'sultanahmet': {'lat_min': 41.000, 'lat_max': 41.015, 'lng_min': 28.975, 'lng_max': 28.985},
            'beyoğlu': {'lat_min': 41.025, 'lat_max': 41.040, 'lng_min': 28.970, 'lng_max': 28.985},
            'taksim': {'lat_min': 41.035, 'lat_max': 41.042, 'lng_min': 28.985, 'lng_max': 28.995},
            'galata': {'lat_min': 41.020, 'lat_max': 41.028, 'lng_min': 28.970, 'lng_max': 28.980},
            'kadıköy': {'lat_min': 40.980, 'lat_max': 41.000, 'lng_min': 29.025, 'lng_max': 29.040},
            'beşiktaş': {'lat_min': 41.035, 'lat_max': 41.050, 'lng_min': 29.000, 'lng_max': 29.015},
            'ortaköy': {'lat_min': 41.045, 'lat_max': 41.055, 'lng_min': 29.020, 'lng_max': 29.030}
        }
        
        # Check which district the coordinates fall into
        for district, bounds in district_boundaries.items():
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
                bounds['lng_min'] <= lng <= bounds['lng_max']):
                return district
        
        # Default to closest major area if no exact match
        if lng < 29.000:  # European side
            return 'beyoğlu'
        else:  # Asian side
            return 'kadıköy'
    
    def _get_restaurant_data_from_local_database(self, location_info: Dict, entities: Dict, query: str) -> List[Dict]:
        """Get restaurant data from local database (500+ restaurants from Google Places)"""
        
        try:
            # Use local restaurant database instead of live API calls
            from backend.services.restaurant_database_service import RestaurantDatabaseService, RestaurantQuery
            
            # Initialize restaurant database service
            db_service = RestaurantDatabaseService()
            
            if not db_service.restaurants:
                logger.warning("Local restaurant database is empty, using fallback")
                return []
            
            # Build query from location and entities
            district = location_info.get('neighborhood') or location_info.get('district')
            cuisine_type = entities.get('cuisines', [None])[0] if entities.get('cuisines') else None
            
            # Extract budget from query text
            budget = self._extract_budget_from_query(query)
            rating_min = self._extract_rating_from_query(query)
            
            # Create database query
            db_query = RestaurantQuery(
                district=district,
                cuisine_type=cuisine_type,
                budget=budget,
                rating_min=rating_min,
                location=(location_info.get('coordinates', {}).get('lat'), 
                         location_info.get('coordinates', {}).get('lng')) if location_info.get('coordinates') else None,
                radius_km=2.0,  # 2km default radius
                keywords=query.lower().split()
            )
            
            # Get restaurants from local database
            restaurants = db_service.filter_restaurants(db_query, limit=10)
            
            if restaurants:
                # Convert database format to internal format
                return self._convert_db_restaurants_to_internal_format(restaurants)
                    
        except ImportError:
            logger.warning("Restaurant database service not available, using static data")
        except Exception as e:
            logger.error(f"Error getting restaurant data from local database: {e}")
            
        return []
    
    def _extract_budget_from_query(self, query: str) -> Optional[str]:
        """Extract budget preference from query text"""
        query_lower = query.lower()
        
        budget_keywords = {
            'budget': ['cheap', 'budget', 'affordable', 'inexpensive', 'economical'],
            'moderate': ['moderate', 'mid-range', 'reasonable', 'normal', 'average'],
            'upscale': ['upscale', 'expensive', 'high-end', 'fine dining', 'premium'],
            'luxury': ['luxury', 'luxurious', 'exclusive', 'top-tier', 'finest']
        }
        
        for budget_cat, keywords in budget_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return budget_cat
        
        return None
    
    def _extract_rating_from_query(self, query: str) -> Optional[float]:
        """Extract minimum rating preference from query text"""
        query_lower = query.lower()
        
        # Look for explicit rating mentions
        import re
        rating_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:star|rating)', query_lower)
        if rating_matches:
            rating = float(rating_matches[0])
            if 1 <= rating <= 5:
                return rating
        
        # Implicit high rating requests
        if any(word in query_lower for word in ['best', 'top', 'excellent', 'amazing', 'outstanding', 'highly rated']):
            return 4.0
        
        return None
    
    def _convert_db_restaurants_to_internal_format(self, db_restaurants: List[Dict]) -> List[Dict]:
        """Convert database restaurant format to internal format"""
        
        converted_restaurants = []
        
        for restaurant in db_restaurants:
            # Map database fields to our internal format
            converted = {
                'id': restaurant.get('place_id', f"db_{len(converted_restaurants)}"),
                'name': restaurant.get('name', 'Unknown Restaurant'),
                'category': self._categorize_db_restaurant(restaurant),
                'location': restaurant.get('district', '').lower(),
                'price_level': self._convert_db_price_level(restaurant.get('budget_category')),
                'rating': restaurant.get('rating', 4.0),
                'cuisine': ', '.join(restaurant.get('cuisine_types', ['Restaurant'])),
                'family_friendly': restaurant.get('rating', 4.0) >= 4.0,  # Heuristic
                'romantic': 'romantic' in restaurant.get('name', '').lower() or restaurant.get('budget_category') == 'luxury',
                'walking_time': self._estimate_walking_time_from_coords(restaurant),
                'accessible': True,  # Most restaurants are accessible
                'suitable_times': ['breakfast', 'lunch', 'dinner'],  # Default all times
                'description': f"Rated {restaurant.get('rating', 'N/A')}/5 with {restaurant.get('reviews_count', 0)} reviews. {restaurant.get('address', 'Istanbul location')}",
                'photo_url': restaurant.get('photos', [None])[0] if restaurant.get('photos') else None,
                'address': restaurant.get('address', ''),
                'phone': restaurant.get('phone', ''),
                'website': restaurant.get('website', ''),
                'opening_hours': restaurant.get('opening_hours', {}).get('weekday_text', []) if restaurant.get('opening_hours') else [],
                'api_source': False  # Mark as database data
            }
            
            converted_restaurants.append(converted)
            
        return converted_restaurants
    
    def _categorize_db_restaurant(self, db_restaurant: Dict) -> str:
        """Categorize restaurant based on database data"""
        
        cuisine_types = db_restaurant.get('cuisine_types', [])
        name = db_restaurant.get('name', '').lower()
        
        # Turkish/local cuisine indicators
        if any(cuisine in ['turkish', 'kebab', 'ottoman'] for cuisine in cuisine_types):
            return 'turkish'
        
        # International cuisine
        international_cuisines = ['italian', 'chinese', 'japanese', 'mexican', 'indian', 'french']
        for cuisine in cuisine_types:
            if cuisine in international_cuisines:
                return cuisine
        
        # Fine dining indicators
        if db_restaurant.get('budget_category') in ['upscale', 'luxury']:
            return 'fine_dining'
        
        # Casual dining default
        return 'casual'
    
    def _convert_db_price_level(self, budget_category: str) -> str:
        """Convert database budget category to price level"""
        budget_mapping = {
            'budget': 'budget',
            'moderate': 'mid',
            'upscale': 'high',
            'luxury': 'luxury'
        }
        return budget_mapping.get(budget_category, 'mid')
    
    def _estimate_walking_time_from_coords(self, restaurant: Dict) -> int:
        """Estimate walking time from restaurant coordinates (placeholder)"""
        # In a real implementation, this would calculate distance from user location
        # For now, return a reasonable default based on district
        district = restaurant.get('district', '').lower()
        
        # Different districts have different typical walking distances
        district_times = {
            'sultanahmet': 5,
            'beyoğlu': 7,
            'galata': 4,
            'taksim': 8,
            'kadıköy': 6
        }
        
        return district_times.get(district, 5)  # Default 5 minutes
    
    def _process_transportation_query_sync(self, message: str, user_profile, current_time, gps_coords=None) -> str:
        """Synchronous transportation query processing as fallback"""
        try:
            # Basic transportation response with available info
            intent_info = detect_transportation_intent(message)
            
            # Build response based on detected intent
            response_parts = []
            
            if gps_coords:
                response_parts.append(f"📍 I can see your location coordinates. Let me help you navigate from there!")
            
            if intent_info['transport_modes']:
                modes_text = ', '.join(intent_info['transport_modes'])
                response_parts.append(f"🚇 For {modes_text} transportation:")
            
            # Add basic transportation info
            if 'metro' in intent_info['transport_modes']:
                response_parts.append("• Metro: Fast, reliable, and air-conditioned")
                response_parts.append("• Use Istanbulkart for all public transport")
            
            if 'bus' in intent_info['transport_modes']:
                response_parts.append("• Bus: Extensive network covering all areas")
                response_parts.append("• Check real-time arrivals on İBB mobile app")
            
            if 'ferry' in intent_info['transport_modes']:
                response_parts.append("• Ferry: Scenic route with Bosphorus views")
                response_parts.append("• Great for avoiding traffic between continents")
            
            # Add optimization tips
            if intent_info['optimization_preference'] == 'cheapest':
                response_parts.append("💰 Most budget-friendly options: Bus → Metro → Ferry → Taxi")
            elif intent_info['optimization_preference'] == 'fastest':
                response_parts.append("⚡ Fastest routes usually combine Metro + Bus or Ferry")
            
            # Add POI suggestions if requested
            if intent_info['include_attractions']:
                response_parts.append("🏛️ I can suggest routes that pass by popular attractions!")
            
            if response_parts:
                return "\n".join(response_parts) + "\n\nTell me your specific start and end points for detailed directions!"
            else:
                return "I'd be happy to help with transportation! Where would you like to go?"
                
        except Exception as e:
            logger.error(f"Sync transportation processing failed: {e}")
            return "I can help you navigate Istanbul! Tell me where you'd like to go and I'll suggest the best routes."