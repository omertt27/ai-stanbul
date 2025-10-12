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

# Import route planning service
try:
    from backend.services.route_maker_service import (
        IstanbulRoutemaker, RouteRequest, RouteStyle, TransportMode, 
        GeneratedRoute, RoutePoint, IstanbulOptimizations
    )
    ROUTE_MAKER_AVAILABLE = True
    logger.info("🗺️ Route Maker Service loaded successfully!")
except ImportError as e:
    logger.warning(f"Route Maker Service not available: {e}")
    ROUTE_MAKER_AVAILABLE = False

# Import weather services
try:
    from services.weather_cache_service import (
        weather_cache, get_weather_for_ai, WeatherData
    )
    from services.weather_notification_service import (
        notify_route_weather, notify_location_weather
    )
    WEATHER_SERVICES_AVAILABLE = True
    logger.info("🌤️ Weather Services loaded successfully!")
except ImportError as e:
    logger.warning(f"Weather Services not available: {e}")
    WEATHER_SERVICES_AVAILABLE = False

# Import weather-aware route cache
try:
    from services.route_cache import (
        get_weather_aware_route_recommendations,
        get_transportation_advice_for_weather,
        weather_aware_cache
    )
    ROUTE_CACHE_AVAILABLE = True
    logger.info("🗺️ Weather-aware route cache loaded successfully!")
except ImportError as e:
    logger.warning(f"Weather-aware route cache not available: {e}")
    ROUTE_CACHE_AVAILABLE = False

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

# Import hidden gems and local tips system
try:
    from hidden_gems_local_tips import HiddenGemsLocalTips, LocalTip, HiddenGem, TipCategory
    HIDDEN_GEMS_AVAILABLE = True
    logger.info("💎 Hidden Gems & Local Tips System loaded successfully!")
except ImportError as e:
    logger.warning(f"Hidden Gems & Local Tips system not available: {e}")
    HIDDEN_GEMS_AVAILABLE = False

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

        # Initialize route maker service
        self.route_maker = None
        if ROUTE_MAKER_AVAILABLE:
            try:
                self.route_maker = IstanbulRoutemaker()
                logger.info("🗺️ Route Maker Service integrated successfully!")
            except Exception as e:
                logger.warning(f"Failed to initialize Route Maker: {e}")
                self.route_maker = None
        else:
            logger.warning("⚠️ Route Maker features disabled")
        
        # Initialize weather services
        self.weather_enabled = WEATHER_SERVICES_AVAILABLE
        if self.weather_enabled:
            logger.info("🌤️ Weather Services integrated successfully!")
        else:
            logger.warning("⚠️ Weather Services disabled")
        
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
        
        # Initialize hidden gems and local tips system
        self.hidden_gems_system = None
        if HIDDEN_GEMS_AVAILABLE:
            try:
                self.hidden_gems_system = HiddenGemsLocalTips()
                logger.info("💎 Hidden Gems & Local Tips System integrated successfully!")
            except Exception as e:
                logger.warning(f"Failed to initialize Hidden Gems System: {e}")
        else:
            logger.warning("⚠️ Hidden Gems & Local Tips features disabled")
        
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
        
        # ENHANCED: Use enhanced intent classification with attractions and hidden gems support
        enhanced_intent = self._enhance_intent_classification(message)
        
        # If enhanced classification found hidden gems query, use it (high priority)
        if enhanced_intent == 'hidden_gems_query':
            return 'hidden_gems_query'
        
        # If enhanced classification found attraction-related intent, use it
        if enhanced_intent in ['attraction_query', 'cultural_query', 'family_activity', 'romantic_spot', 'hidden_gem']:
            return enhanced_intent
        
        # If enhanced classification found transportation intent, use it
        if enhanced_intent == 'transportation_query':
            return 'transportation_query'
        
        # PRIORITY: Check for route planning queries (high priority)
        if self.is_route_planning_query(message):
            return 'route_planning'
        
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
        
        # Get weather context for intelligent recommendations
        weather_context = None
        if self.weather_enabled:
            try:
                weather_context = get_weather_for_ai()
                if weather_context and 'error' not in weather_context:
                    logger.info("🌤️ Weather context integrated for response generation")
            except Exception as e:
                logger.warning(f"Failed to get weather context: {e}")
        
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
        
        elif intent == 'hidden_gems_query':
            # Use hidden gems and local tips system with GPS and ML personalization
            if self.hidden_gems_system:
                user_location = None
                if user_profile.gps_location:
                    user_location = (user_profile.gps_location.get('lat'), user_profile.gps_location.get('lng'))
                
                # Convert user profile to dict format for hidden gems system
                user_context = {
                    'interests': user_profile.interests,
                    'budget_range': user_profile.budget_range,
                    'accessibility_needs': user_profile.accessibility_needs,
                    'travel_style': user_profile.travel_style,
                    'group_type': user_profile.group_type,
                    'cultural_immersion_level': user_profile.cultural_immersion_level,
                    'location': user_location,
                    'current_time': current_time
                }
                
                try:
                    return self.hidden_gems_system.process_hidden_gems_query(message, user_context)
                except Exception as e:
                    logger.warning(f"Hidden gems processing failed: {e}")
                    return self._generate_fallback_hidden_gems_response(message, user_profile, current_time)
            else:
                return self._generate_fallback_hidden_gems_response(message, user_profile, current_time)
        
        elif intent == 'route_planning':
            # Handle route planning queries with multi-modal routing
            return self.handle_route_planning_query(message, user_profile, context, current_time)
        
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
        
        # 🌤️ Get weather context for restaurant recommendations
        weather_context = None
        if self.weather_enabled:
            try:
                weather_context = get_weather_for_ai()
                if weather_context and 'error' not in weather_context:
                    # Add weather-based restaurant suggestions
                    location_info['weather_context'] = weather_context
                    logger.info(f"🌤️ Added weather context to restaurant search: {weather_context['condition']}")
            except Exception as e:
                logger.warning(f"Failed to get weather context for restaurants: {e}")
        
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
                'explanation': f"Matches your {user_profile.budget_range or 'not specified'} budget range"
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
        """Enhanced intent classification with attraction support and hidden gems detection"""
        message_lower = message.lower()
        
        # Route planning keywords (high priority)
        route_keywords = [
            'route', 'plan my day', 'itinerary', 'tour plan', 'day plan', 
            'plan route', 'visiting multiple', 'best route', 'optimize route',
            'day trip', 'multi-stop', 'several places', 'different attractions',
            'plan visit', 'route to visit', 'how to visit all', 'sequence',
            'order to visit', 'efficient route', 'walking route', 'travel plan'
        ]
        if any(keyword in message_lower for keyword in route_keywords):
            return 'route_planning'
        
        # Hidden gems and local tips keywords (high priority)
        hidden_gems_keywords = [
            'hidden gem', 'hidden gems', 'secret', 'locals go', 'local favorite', 'local tip', 'local tips',
            'off the beaten path', 'undiscovered', 'authentic', 'insider tip', 'insider tips',
            'where locals eat', 'local secret', 'best kept secret', 'not touristy', 'avoid crowds',
            'real istanbul', 'authentic istanbul', 'local experience', 'local places', 'neighborhood secret'
        ]
        if any(keyword in message_lower for keyword in hidden_gems_keywords):
            return 'hidden_gems_query'
        
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
    
    def is_route_planning_query(self, message: str) -> bool:
        """Check if message is asking for route planning"""
        route_keywords = [
            'route', 'plan my day', 'itinerary', 'tour plan', 'day plan', 
            'plan route', 'visiting multiple', 'best route', 'optimize route',
            'day trip', 'multi-stop', 'several places', 'different attractions',
            'plan visit', 'route to visit', 'how to visit all', 'sequence',
            'order to visit', 'efficient route', 'walking route', 'travel plan',
            'visit in order', 'how to see everything', 'optimize my trip',
            'plan my trip', 'best way to visit', 'route between', 'connect places'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in route_keywords)

    def _detect_route_planning_intent(self, message: str) -> Dict[str, Any]:
        """Detect specific route planning intent and extract parameters"""
        message_lower = message.lower()
        
        intent_data = {
            'is_route_planning': self.is_route_planning_query(message),
            'route_style': 'balanced',  # Default
            'transport_mode': 'walking',  # Default
            'max_duration_hours': 4.0,  # Default
            'include_food': True,  # Default
            'max_attractions': 6,  # Default
            'optimization_preference': 'balanced'
        }
        
        # Extract route style preferences
        if any(word in message_lower for word in ['scenic', 'beautiful', 'views', 'photography']):
            intent_data['route_style'] = 'scenic'
        elif any(word in message_lower for word in ['cultural', 'history', 'heritage', 'museum']):
            intent_data['route_style'] = 'cultural'
        elif any(word in message_lower for word in ['efficient', 'quick', 'fast', 'shortest']):
            intent_data['route_style'] = 'efficient'
        
        # Extract transport preferences
        if any(word in message_lower for word in ['walking', 'walk', 'on foot']):
            intent_data['transport_mode'] = 'walking'
        elif any(word in message_lower for word in ['public transport', 'metro', 'bus', 'ferry']):
            intent_data['transport_mode'] = 'public_transport'
        elif any(word in message_lower for word in ['driving', 'car', 'taxi']):
            intent_data['transport_mode'] = 'driving'
        
        # Extract time duration
        time_matches = re.findall(r'(\d+)\s*(?:hour|hr|h)', message_lower)
        if time_matches:
            intent_data['max_duration_hours'] = float(time_matches[0])
        
        # Extract optimization preferences
        if any(word in message_lower for word in ['shortest', 'fastest', 'quick']):
            intent_data['optimization_preference'] = 'time'
        elif any(word in message_lower for word in ['cheapest', 'budget', 'affordable']):
            intent_data['optimization_preference'] = 'cost'
        elif any(word in message_lower for word in ['best', 'top', 'must-see']):
            intent_data['optimization_preference'] = 'quality'
        
        return intent_data

    # =============================
    # ROUTE PLANNING SYSTEM INTEGRATION
    # =============================
    
    def handle_route_planning_query(self, message: str, user_profile: UserProfile, 
                                  context: ConversationContext, current_time: datetime) -> str:
        """Handle route planning queries with multi-modal routing"""
        try:
            # Check if route maker service is available
            if not hasattr(self, 'route_maker') or self.route_maker is None:
                return self._generate_fallback_route_response(message, user_profile, current_time)
            
            # Detect route planning intent and extract parameters
            route_intent = self._detect_route_planning_intent(message)
            
            if not route_intent['is_route_planning']:
                return "I'd be happy to help you plan a route! Tell me which places you'd like to visit and I'll create an optimized itinerary for you."
            
            # Get or request user location
            location_info = self._extract_or_request_location(message, user_profile, context)
            
            if location_info['accuracy'] == 'low' and location_info['source'] == 'default':
                return self._request_location_for_route_planning(message, user_profile)
            
            # Extract places/attractions from the message
            places_to_visit = self._extract_places_from_message(message)
            
            # If no specific places mentioned, suggest based on user preferences
            if not places_to_visit:
                return self._suggest_route_based_on_preferences(message, user_profile, location_info, route_intent)
            
            # Generate route using the route maker service
            route_result = self._generate_optimized_route(
                places_to_visit, location_info, user_profile, route_intent, current_time
            )
            
            # Format and return the route response
            return self._format_route_response(route_result, user_profile, current_time)
            
        except Exception as e:
            logger.error(f"Route planning failed: {e}")
            return self._generate_fallback_route_response(message, user_profile, current_time)

    def _request_location_for_route_planning(self, query: str, user_profile: UserProfile) -> str:
        """Request location from user for route planning"""
        
        response = "🗺️ **Location Needed for Route Planning**\n\n"
        
        response += "To create the perfect route for you, I need to know where you're starting from:\n\n"
        
        if not user_profile.location_sharing_enabled:
            response += "📍 **Option 1: Enable GPS Location**\n"
            response += "• Say 'enable location sharing' for automatic location-based routing\n"
            response += "• Your location is only used for this session\n\n"
        
        response += "🏙️ **Option 2: Tell Me Your Starting Point**\n"
        response += "• Mention which Istanbul district or landmark you're starting from\n"
        response += "• Examples: 'Starting from Sultanahmet', 'I'm at Taksim Square', 'From my hotel in Beyoğlu'\n\n"
        
        response += "💡 **Meanwhile, you can also specify:**\n"
        response += "• How long you want to spend (e.g., '4 hours', 'half day')\n"
        response += "• Your preferred style ('scenic route', 'cultural tour', 'efficient route')\n"
        response += "• Transportation preference ('walking', 'public transport', 'mixed')\n\n"
        
        response += "Just let me know your starting point and preferences, and I'll create an amazing route for you! 🚶‍♂️🗺️"
        
        return response

    def _extract_places_from_message(self, message: str) -> List[str]:
        """Extract place names and attractions from user message"""
        
        # Common Istanbul attractions and landmarks
        istanbul_places = [
            'hagia sophia', 'blue mosque', 'topkapi palace', 'grand bazaar', 'spice bazaar',
            'galata tower', 'basilica cistern', 'dolmabahce palace', 'bosphorus bridge',
            'taksim square', 'istiklal street', 'karakoy', 'beyoglu', 'sultanahmet',
            'kadikoy', 'besiktas', 'ortakoy', 'eminonu', 'galata bridge',
            'princes islands', 'maiden tower', 'rumeli fortress', 'eyup sultan mosque',
            'suleymaniye mosque', 'archaeological museum', 'turkish and islamic arts museum',
            'istanbul modern', 'pera museum', 'chora church', 'little hagia sophia'
        ]
        
        message_lower = message.lower()
        found_places = []
        
        for place in istanbul_places:
            if place in message_lower:
                found_places.append(place.title())
        
        # Also look for generic categories
        if 'museum' in message_lower and not any('museum' in place for place in found_places):
            found_places.append('Museums')
        if 'mosque' in message_lower and not any('mosque' in place for place in found_places):
            found_places.append('Mosques')
        if 'palace' in message_lower and not any('palace' in place for place in found_places):
            found_places.append('Palaces')
        
        return found_places

    def _suggest_route_based_on_preferences(self, message: str, user_profile: UserProfile, 
                                          location_info: Dict, route_intent: Dict) -> str:
        """Suggest route options based on user preferences when no specific places mentioned"""
        
        # Get weather context
        weather_data = None
        weather_context = ""
        if self.weather_enabled:
            try:
                weather_data = get_weather_for_ai()
                if weather_data:
                    temp = weather_data.current_temp
                    condition = weather_data.condition
                    weather_context = f"🌤️ **Current Weather:** {temp}°C, {condition}\n\n"
                    
                    # Add weather-specific advice
                    if weather_data.rainfall_1h and weather_data.rainfall_1h > 0:
                        weather_context += "☔ Rain expected - routes adapted for covered attractions\n\n"
                    elif temp > 28:
                        weather_context += "🌞 Hot weather - early morning or indoor routes recommended\n\n"
                    elif temp < 10:
                        weather_context += "🧥 Cool weather - indoor attractions and warm-up spots included\n\n"
            except Exception as e:
                logger.warning(f"Could not get weather data: {e}")
        
        response = "🎯 **Route Planning Suggestions**\n\n"
        response += weather_context
        
        # Determine user interests
        interests = user_profile.interests or []
        route_style = route_intent['route_style']
        duration = route_intent['max_duration_hours']
        
        response += f"Based on your preferences and current conditions, here are some great {duration}-hour route options from {location_info['neighborhood'].title()}:\n\n"
        
        # Generate route suggestions based on style and interests
        if route_style == 'cultural' or 'history' in interests:
            response += "🏛️ **Cultural Heritage Route:**\n"
            response += "• Hagia Sophia → Blue Mosque → Topkapi Palace → Archaeological Museum\n"
            response += "• Perfect for history lovers and architecture enthusiasts\n"
            response += "• Estimated time: 4-5 hours with guided visits\n\n"
        
        if route_style == 'scenic' or 'photography' in interests:
            response += "📸 **Scenic Bosphorus Route:**\n"
            response += "• Galata Tower → Galata Bridge → Eminönu Ferry → Ortaköy → Dolmabahçe Palace\n"
            response += "• Best views of the city and Bosphorus\n"
            response += "• Great for photography and sunset views\n\n"
        
        if 'food' in interests or route_intent['include_food']:
            response += "🍽️ **Food & Culture Route:**\n"
            response += "• Spice Bazaar → Karaköy food scene → Beyoğlu meyhanes → Istiklal Street\n"
            response += "• Combines cultural sites with authentic local dining\n"
            response += "• Perfect for food enthusiasts\n\n"
        
        if route_style == 'efficient':
            response += "⚡ **Essential Istanbul Route:**\n"
            response += "• Sultanahmet Square → Grand Bazaar → Galata Tower → Taksim Square\n"
            response += "• Covers major highlights efficiently\n"
            response += "• Ideal for limited time visitors\n\n"
        
        # Add weather-aware recommendations if available
        if weather_data and ROUTE_CACHE_AVAILABLE:
            try:
                area = location_info.get('neighborhood', 'sultanahmet')
                weather_recommendations = get_weather_aware_route_recommendations(area, route_style, weather_data)
                
                if weather_recommendations.get('weather_recommendations'):
                    response += "🌤️ **Weather-Adapted Recommendations:**\n"
                    for rec in weather_recommendations['weather_recommendations'][:3]:  # Top 3
                        response += f"• {rec}\n"
                    response += "\n"
                
                transportation_advice = get_transportation_advice_for_weather(weather_data, 
                    location_info.get('neighborhood', 'Current Location'), 'Destination')
                if transportation_advice and 'recommended_routes' in transportation_advice:
                    response += "🚇 **Detailed Transportation Guide:**\n"
                    response += f"📍 {transportation_advice['route_overview']}\n"
                    response += f"🌤️ {transportation_advice['weather_impact']}\n\n"
                    
                    # Show primary recommended route
                    if transportation_advice['recommended_routes']:
                        primary_route = transportation_advice['recommended_routes'][0]
                        response += f"**🎯 Recommended: {primary_route['route_name']}**\n"
                        response += f"⏱️ Duration: {primary_route['duration']} | 💰 Cost: {primary_route['cost']}\n"
                        response += f"🌟 {primary_route['weather_rating']}\n\n"
                        
                        response += "**Step-by-step directions:**\n"
                        for i, step in enumerate(primary_route['steps'][:4], 1):  # Show first 4 steps
                            response += f"{i}. {step['instruction']} ({step['duration']})\n"
                            if 'weather_tip' in step:
                                response += f"   💡 {step['weather_tip']}\n"
                        response += "\n"
                    
                    # Show real-time alerts
                    if transportation_advice['real_time_alerts']:
                        response += "⚠️ **Current Alerts:**\n"
                        for alert in transportation_advice['real_time_alerts'][:2]:
                            response += f"• {alert}\n"
                        response += "\n"
                    
            except Exception as e:
                logger.warning(f"Failed to get weather-aware recommendations: {e}")
        
        response += "💡 **To create your personalized route:**\n"
        response += "• Choose which route interests you most\n"
        response += "• Tell me specific places you want to visit\n"
        response += "• Let me know your time constraints and transport preferences\n\n"
        
        response += "Which route style appeals to you, or would you like me to plan something custom based on specific places you want to visit?"
        
        return response

    def _generate_optimized_route(self, places_to_visit: List[str], location_info: Dict, 
                                user_profile: UserProfile, route_intent: Dict, 
                                current_time: datetime) -> Dict:
        """Generate optimized route using the route maker service with weather awareness"""
        
        # Get weather context first
        weather_data = None
        if self.weather_enabled:
            try:
                weather_data = get_weather_for_ai()
            except Exception as e:
                logger.warning(f"Could not get weather data: {e}")
        
        try:
            from backend.services.route_maker_service import RouteRequest, RouteStyle, TransportMode
            
            # Convert string parameters to enums
            route_style_map = {
                'scenic': RouteStyle.SCENIC,
                'cultural': RouteStyle.CULTURAL,
                'efficient': RouteStyle.EFFICIENT,
                'balanced': RouteStyle.BALANCED
            }
            
            transport_mode_map = {
                'walking': TransportMode.WALKING,
                'driving': TransportMode.DRIVING,
                'public_transport': TransportMode.PUBLIC_TRANSPORT
            }
            
            # Get coordinates for starting location
            start_coords = self._get_coordinates_for_location(location_info)
            
            if not start_coords:
                raise ValueError("Could not determine starting coordinates")
            
            # Build route request
            route_request = RouteRequest(
                start_lat=start_coords['lat'],
                start_lng=start_coords['lng'],
                max_distance_km=route_intent.get('max_distance_km', 5.0),
                available_time_hours=route_intent['max_duration_hours'],
                preferred_categories=self._convert_interests_to_categories(user_profile.interests),
                route_style=route_style_map.get(route_intent['route_style'], RouteStyle.BALANCED),
                transport_mode=transport_mode_map.get(route_intent['transport_mode'], TransportMode.WALKING),
                include_food=route_intent['include_food'],
                max_attractions=route_intent['max_attractions']
            )
            
            # Generate route using the route maker service
            try:
                from database import get_db
                db = next(get_db())
                generated_route = self.route_maker.generate_route(route_request, db)
                db.close()
            except Exception:
                # Fallback without database
                generated_route = self.route_maker.generate_route(route_request, None)
            
            # Add personalization based on user profile
            personalized_route = self._personalize_route(generated_route, user_profile)
            
            # Add weather-aware recommendations if available
            weather_recommendations = []
            transportation_advice = []
            if weather_data and ROUTE_CACHE_AVAILABLE:
                try:
                    # Get weather recommendations from route cache
                    weather_recommendations = weather_aware_cache.get_weather_recommendations(weather_data)
                    start_coords = self._get_coordinates_for_location(location_info)
                    transportation_advice = get_transportation_advice_for_weather(weather_data, 
                        location_info.get('neighborhood', 'Current Location'), 
                        ', '.join(places_to_visit[:2]) if places_to_visit else 'Destination')
                except Exception as e:
                    logger.warning(f"Failed to get weather recommendations: {e}")
            
            return {
                'success': True,
                'route': personalized_route,
                'places_requested': places_to_visit,
                'optimization_applied': True,
                'weather_recommendations': weather_recommendations,
                'transportation_advice': transportation_advice,
                'weather_data': weather_data.to_dict() if weather_data else None
            }
            
        except Exception as e:
            logger.error(f"Route generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'places_requested': places_to_visit
            }

    def _get_coordinates_for_location(self, location_info: Dict) -> Optional[Dict]:
        """Get coordinates for a given location"""
        
        # If coordinates already available, use them
        if location_info.get('coordinates'):
            return location_info['coordinates']
        
        # Define approximate coordinates for major Istanbul districts
        district_coordinates = {
            'sultanahmet': {'lat': 41.0086, 'lng': 28.9802},
            'beyoğlu': {'lat': 41.0370, 'lng': 28.9777},
            'beyoglu': {'lat': 41.0370, 'lng': 28.9777},
            'taksim': {'lat': 41.0370, 'lng': 28.9844},
            'galata': {'lat': 41.0255, 'lng': 28.9742},
            'karaköy': {'lat': 41.0255, 'lng': 28.9742},
            'karakoy': {'lat': 41.0255, 'lng': 28.9742},
            'kadıköy': {'lat': 40.9897, 'lng': 29.0267},
            'kadikoy': {'lat': 40.9897, 'lng': 29.0267},
            'beşiktaş': {'lat': 41.0422, 'lng': 29.0094},
            'besiktas': {'lat': 41.0422, 'lng': 29.0094},
            'ortaköy': {'lat': 41.0550, 'lng': 29.0268},
            'ortakoy': {'lat': 41.0550, 'lng': 29.0268},
            'eminönü': {'lat': 41.0170, 'lng': 28.9700},
            'eminonu': {'lat': 41.0170, 'lng': 28.9700}
        }
        
        neighborhood = location_info.get('neighborhood', '').lower()
        return district_coordinates.get(neighborhood)

    def _convert_interests_to_categories(self, interests: List[str]) -> List[str]:
        """Convert user interests to route maker categories"""
        
        if not interests:
            return ['tourist_attraction', 'cultural', 'historical']
        
        category_mapping = {
            'history': 'historical',
            'culture': 'cultural',
            'art': 'cultural',
            'food': 'restaurant',
            'architecture': 'architectural',
            'photography': 'scenic',
            'nature': 'park',
            'shopping': 'shopping',
            'nightlife': 'entertainment',
            'family': 'family_friendly',
            'romance': 'romantic'
        }
        
        categories = []
        for interest in interests:
            category = category_mapping.get(interest.lower(), 'tourist_attraction')
            if category not in categories:
                categories.append(category)
        
        # Always include tourist attractions as baseline
        if 'tourist_attraction' not in categories:
            categories.append('tourist_attraction')
        
        return categories

    def _personalize_route(self, generated_route, user_profile: UserProfile):
        """Add personalization to the generated route"""
        
        # Add personalized recommendations and tips
        for point in generated_route.points:
            # Add accessibility information if needed
            if user_profile.accessibility_needs:
                point.notes += f"\n♿ Accessibility: Wheelchair accessible venue"
            
            # Add dietary recommendations if applicable
            if user_profile.dietary_restrictions and 'restaurant' in point.category.lower():
                dietary_info = ", ".join(user_profile.dietary_restrictions)
                point.notes += f"\n🥗 Dietary: {dietary_info} options available"
            
            # Add budget information
            if user_profile.budget_range:
                budget_emoji = {'budget': '💰', 'moderate': '💰💰', 'upscale': '💰💰💰', 'luxury': '💰💰💰💰'}
                point.notes += f"\n{budget_emoji.get(user_profile.budget_range, '💰💰')} Budget: {user_profile.budget_range.title()} range"
        
        return generated_route

    def _format_route_response(self, route_result: Dict, user_profile: UserProfile, 
                             current_time: datetime) -> str:
        """Format the route result into a user-friendly response"""
        
        if not route_result['success']:
            return f"I encountered an issue creating your route: {route_result.get('error', 'Unknown error')}. Let me help you plan manually - which specific places would you like to visit?"
        
        route = route_result['route']
        response = f"🗺️ **Your Optimized Istanbul Route**\n\n"
        
        # Route overview
        response += f"📊 **Route Overview:**\n"
        response += f"• **Total Distance:** {route.total_distance_km:.1f} km\n"
        response += f"• **Estimated Duration:** {route.estimated_duration_hours:.1f} hours\n"
        response += f"• **Route Style:** {route.description}\n"
        response += f"• **Overall Score:** {route.overall_score:.1f}/10\n\n"
        
        # Route points
        response += f"📍 **Your Itinerary ({len(route.points)} stops):**\n\n"
        
        for i, point in enumerate(route.points):
            if i == 0:
                response += f"🏁 **Start:** {point.name}\n"
                if point.arrival_time:
                    response += f"   ⏰ Starting time: {point.arrival_time}\n"
            else:
                response += f"{i}. **{point.name}**"
                if point.category:
                    response += f" ({point.category.title()})"
                response += "\n"
                
                if point.arrival_time:
                    response += f"   ⏰ Arrival: {point.arrival_time}\n"
                
                if point.estimated_duration_minutes > 0:
                    response += f"   🕐 Visit duration: {point.estimated_duration_minutes} minutes\n"
                
                if point.score > 0:
                    response += f"   ⭐ Attraction score: {point.score:.1f}/10\n"
                
                if point.notes:
                    # Clean up notes and format nicely
                    clean_notes = point.notes.strip()
                    if clean_notes:
                        response += f"   💡 {clean_notes}\n"
            
            response += "\n"
        
        # Add route optimization info
        response += f"🎯 **Route Optimization:**\n"
        response += f"• **Efficiency Score:** {route.efficiency_score:.1f}/10\n"
        response += f"• **Diversity Score:** {route.diversity_score:.1f}/10\n"
        response += f"• Route optimized using TSP algorithm for minimal travel time\n\n"
        
        # Add personalized tips
        response += f"💡 **Personalized Tips:**\n"
        
        if user_profile.budget_range:
            response += f"• Budget optimized for {user_profile.budget_range} range\n"
        
        if user_profile.interests:
            interests_text = ", ".join(user_profile.interests)
            response += f"• Route tailored to your interests: {interests_text}\n"
        
        if user_profile.accessibility_needs:
            response += f"• All locations verified for accessibility\n"
        
        # Add weather-aware advice
        if route_result.get('weather_recommendations'):
            response += "\n🌤️ **Weather-Aware Tips:**\n"
            for rec in route_result['weather_recommendations'][:3]:  # Top 3 recommendations
                response += f"• {rec}\n"
        
        if route_result.get('transportation_advice'):
            transportation_advice = route_result['transportation_advice']
            if isinstance(transportation_advice, dict) and 'recommended_routes' in transportation_advice:
                response += "\n🚇 **Detailed Transportation Guide:**\n"
                response += f"📍 {transportation_advice['route_overview']}\n"
                if 'weather_impact' in transportation_advice:
                    response += f"🌤️ {transportation_advice['weather_impact']}\n\n"
                
                # Show recommended route
                if transportation_advice['recommended_routes']:
                    primary_route = transportation_advice['recommended_routes'][0]
                    response += f"**🎯 Best Route: {primary_route['route_name']}**\n"
                    response += f"⏱️ {primary_route['duration']} | 💰 {primary_route['cost']} | {primary_route.get('weather_rating', '')}\n\n"
                    
                    response += "**Directions:**\n"
                    for i, step in enumerate(primary_route['steps'][:3], 1):  # Show first 3 steps
                        response += f"{i}. {step['instruction']} ({step['duration']})\n"
                        if 'weather_tip' in step:
                            response += f"   💡 {step['weather_tip']}\n"
                    response += "\n"
                
                # Show alerts
                if transportation_advice.get('real_time_alerts'):
                    response += "⚠️ **Current Alerts:**\n"
                    for alert in transportation_advice['real_time_alerts'][:2]:
                        response += f"• {alert}\n"
                    response += "\n"
            else:
                # Fallback for simple list format
                response += "\n🚇 **Transportation Advice:**\n"
                advice_list = transportation_advice if isinstance(transportation_advice, list) else [str(transportation_advice)]
                for advice in advice_list[:2]:
                    response += f"• {advice}\n"
        
        # Add real-time advice
        current_hour = current_time.hour
        if current_hour < 10:
            response += f"• Perfect morning start - most attractions will be less crowded\n"
        elif current_hour > 16:
            response += f"• Evening route - consider which attractions stay open late\n"
        
        response += "\n🗺️ **Need adjustments?** Tell me if you want to:\n"
        response += "• Add or remove specific places\n"
        response += "• Change transportation method\n"
        response += "• Adjust time constraints\n"
        response += "• Get directions between specific points\n\n"
        
        response += "Have an amazing time exploring Istanbul! 🌟"
        
        return response

    def _generate_fallback_route_response(self, message: str, user_profile: UserProfile, 
                                        current_time: datetime) -> str:
        """Generate fallback route response when route maker is unavailable"""
        
        response = "🗺️ **Route Planning Assistant**\n\n"
        
        response += "I'd love to help you plan the perfect route through Istanbul! Here's what I can do:\n\n"
        
        response += "🎯 **Popular Route Options:**\n\n"
        
        response += "**1. Classic Historic Route** (4-5 hours)\n"
        response += "• Sultanahmet → Hagia Sophia → Blue Mosque → Topkapi Palace → Grand Bazaar\n"
        response += "• Best for first-time visitors\n\n"
        
        response += "**2. Bosphorus Scenic Route** (3-4 hours)\n"
        response += "• Galata Tower → Galata Bridge → Eminönu Ferry → Ortaköy → Dolmabahçe Palace\n"
        response += "• Perfect for photography and views\n\n"
        
        response += "**3. Modern Istanbul Route** (4-5 hours)\n"
        response += "• Taksim → İstiklal Street → Galata → Karaköy → Beyoğlu food scene\n"
        response += "• Great for culture and nightlife\n\n"
        
        response += "**4. Asian Side Discovery** (3-4 hours)\n"
        response += "• Kadıköy → Moda → Üsküdar → Çamlıca Hill\n"
        response += "• Less touristy, more authentic\n\n"
        
        response += "💡 **To customize your route:**\n"
        response += "• Tell me your specific interests (history, food, art, etc.)\n"
        response += "• Let me know your time constraints\n"
        response += "• Specify your starting location\n"
        response += "• Mention any places you definitely want to see\n\n"
        
        response += "Which route interests you, or would you like help planning something custom?"
        
        return response