# -*- coding: utf-8 -*-
"""
Istanbul Daily Talk AI - Main System
The main orchestration class for the Istanbul AI system.
"""

import json
import logging
import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from .core.models import UserProfile, ConversationContext
from .core.entity_recognition import IstanbulEntityRecognizer
from .core.response_generator import ResponseGenerator
from .core.user_management import UserManager

# Week 1 Modularization: Import initialization modules
from .initialization import ServiceInitializer, HandlerInitializer, SystemConfig

# Week 2 Modularization: Import routing layer modules
from .routing import (
    IntentClassifier,
    EntityExtractor,
    QueryPreprocessor,
    ResponseRouter
)

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import hidden gems and price filtering services (after logging is configured)
try:
    from backend.services.hidden_gems_handler import HiddenGemsHandler
    HIDDEN_GEMS_HANDLER_AVAILABLE = True
    logger.info("✅ Hidden Gems Handler loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Hidden Gems Handler not available: {e}")
    HIDDEN_GEMS_HANDLER_AVAILABLE = False

try:
    from backend.services.price_filter_service import PriceFilterService
    PRICE_FILTER_AVAILABLE = True
    logger.info("✅ Price Filter Service loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Price Filter Service not available: {e}")
    PRICE_FILTER_AVAILABLE = False

# Import Events Service
try:
    from backend.services.events_service import get_events_service
    EVENTS_SERVICE_AVAILABLE = True
    logger.info("✅ Events Service loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Events Service not available: {e}")
    EVENTS_SERVICE_AVAILABLE = False

# Import Weather Recommendations Service
try:
    from backend.services.weather_recommendations import get_weather_recommendations_service
    WEATHER_RECOMMENDATIONS_AVAILABLE = True
    logger.info("✅ Weather Recommendations Service loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Weather Recommendations Service not available: {e}")
    WEATHER_RECOMMENDATIONS_AVAILABLE = False

# Import advanced transportation system
try:
    import sys
    import os
    # Add parent directory to path to access transportation modules
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from enhanced_transportation_integration import TransportationQueryProcessor, create_ml_enhanced_transportation_system, GPSLocation
    ADVANCED_TRANSPORT_AVAILABLE = True
    logger.info("✅ Advanced transportation system loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Advanced transportation system not available: {e}")
    ADVANCED_TRANSPORT_AVAILABLE = False

# Import Transfer Instructions & Map Visualization Integration
try:
    from transportation_chat_integration import TransportationChatIntegration
    TRANSFER_MAP_INTEGRATION_AVAILABLE = True
    logger.info("✅ Transfer Instructions & Map Visualization integration loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Transfer Instructions & Map Visualization not available: {e}")
    TRANSFER_MAP_INTEGRATION_AVAILABLE = False

# Import ML-Enhanced Daily Talks Bridge
try:
    from ml_enhanced_daily_talks_bridge import MLEnhancedDailyTalksBridge, process_enhanced_daily_talk
    ML_DAILY_TALKS_AVAILABLE = True
    logger.info("✅ ML-Enhanced Daily Talks Bridge loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ ML-Enhanced Daily Talks Bridge not available: {e}")
    ML_DAILY_TALKS_AVAILABLE = False

# Import Enhanced Bilingual Daily Talks System (PRIMARY)
try:
    import sys
    import os
    # Add parent directory to path to import enhanced_bilingual_daily_talks
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from enhanced_bilingual_daily_talks import EnhancedBilingualDailyTalks, UserContext as DailyTalkContext, Language
    ENHANCED_DAILY_TALKS_AVAILABLE = True
    logger.info("✅ Enhanced Bilingual Daily Talks System loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Bilingual Daily Talks System not available: {e}")
    ENHANCED_DAILY_TALKS_AVAILABLE = False

# Import Lightweight Neural Query Enhancement System (Budget-Friendly!)
try:
    from backend.services.lightweight_neural_query_enhancement import (
        get_lightweight_neural_processor,
        LightweightNeuralInsights
    )
    NEURAL_QUERY_ENHANCEMENT_AVAILABLE = True
    logger.info("✅ Neural Query Enhancement System loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Neural Query Enhancement System not available: {e}")
    NEURAL_QUERY_ENHANCEMENT_AVAILABLE = False

# Import Multi-Intent Query Handler for complex multi-part queries
try:
    from multi_intent_query_handler import MultiIntentQueryHandler, MultiIntentResult
    MULTI_INTENT_HANDLER_AVAILABLE = True
    logger.info("✅ Multi-Intent Query Handler loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Multi-Intent Query Handler not available: {e}")
    MULTI_INTENT_HANDLER_AVAILABLE = False

# Import Advanced Personalization System
try:
    from backend.services.advanced_personalization import AdvancedPersonalizationSystem
    ADVANCED_PERSONALIZATION_AVAILABLE = True
    logger.info("✅ Advanced Personalization System loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Advanced Personalization System not available: {e}")
    ADVANCED_PERSONALIZATION_AVAILABLE = False

# Import Real-time Feedback Loop System
try:
    from backend.services.feedback_loop import FeedbackLoopSystem
    FEEDBACK_LOOP_AVAILABLE = True
    logger.info("✅ Real-time Feedback Loop System loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Real-time Feedback Loop System not available: {e}")
    FEEDBACK_LOOP_AVAILABLE = False

# Import ML-Enhanced Handlers
try:
    from istanbul_ai.handlers.event_handler import create_ml_enhanced_event_handler
    from istanbul_ai.handlers.hidden_gems_handler import create_ml_enhanced_hidden_gems_handler
    from istanbul_ai.handlers.weather_handler import create_ml_enhanced_weather_handler
    from istanbul_ai.handlers.route_planning_handler import create_ml_enhanced_route_planning_handler
    from istanbul_ai.handlers.neighborhood_handler import create_ml_enhanced_neighborhood_handler
    ML_ENHANCED_HANDLERS_AVAILABLE = True
    logger.info("✅ ML-Enhanced Handlers loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ ML-Enhanced Handlers not available: {e}")
    ML_ENHANCED_HANDLERS_AVAILABLE = False


class IstanbulDailyTalkAI:
    """🚀 ENHANCED Istanbul Daily Talk AI System with Deep Learning
    
    Advanced conversational AI for Istanbul visitors and locals with comprehensive
    recommendations, cultural context, and personalized experiences.
    """
    
    def __init__(self):
        """Initialize the Istanbul AI system with modular architecture (Week 1)"""
        logger.info("🚀 Initializing Istanbul Daily Talk AI System (Modular Architecture)...")
        
        # Week 1 Modularization: Get system configuration
        self.config = SystemConfig()
        
        # Week 1 Modularization: Initialize core components
        self.entity_recognizer = IstanbulEntityRecognizer()
        self.response_generator = ResponseGenerator()
        self.user_manager = UserManager()
        
        # Week 1 Modularization: Initialize all services using ServiceInitializer
        service_initializer = ServiceInitializer()
        services = service_initializer.initialize_all_services()
        
        # Assign initialized services to instance variables
        for service_name, service_instance in services.items():
            setattr(self, service_name, service_instance)
        
        logger.info(f"✅ Initialized {len(services)} services via ServiceInitializer")
        
        # Week 1 Modularization: Initialize ML handlers using HandlerInitializer
        handler_initializer = HandlerInitializer(
            events_service=self.events_service,
            hidden_gems_handler=self.hidden_gems_handler,
            weather_client=self.weather_client,
            weather_recommendations=self.weather_recommendations,
            gps_route_planner=self.gps_route_planner,
            advanced_route_planner=self.advanced_route_planner,
            transport_processor=self.transport_processor,
            response_generator=self.response_generator,
            neural_processor=self.neural_processor,
            ml_context_builder=self.ml_context_builder
        )
        
        handlers = handler_initializer.initialize_all_handlers()
        
        # Assign initialized handlers to instance variables
        for handler_name, handler_instance in handlers.items():
            setattr(self, handler_name, handler_instance)
        
        logger.info(f"✅ Initialized {len(handlers)} ML handlers via HandlerInitializer")
        
        # Week 2 Modularization: Initialize routing layer components
        logger.info("🎯 Initializing routing layer components...")
        
        # Initialize Intent Classifier
        self.intent_classifier = IntentClassifier(
            multi_intent_handler=self.multi_intent_handler,
            neural_processor=self.neural_processor
        )
        
        # Initialize Entity Extractor (wraps and enhances entity_recognizer)
        self.entity_extractor = EntityExtractor(
            entity_recognizer=self.entity_recognizer
        )
        
        # Initialize Query Preprocessor
        self.query_preprocessor = QueryPreprocessor(
            neural_processor=self.neural_processor
        )
        
        # Initialize Response Router
        self.response_router = ResponseRouter(
            ml_restaurant_handler=handlers.get('ml_restaurant_handler'),
            ml_attraction_handler=handlers.get('ml_attraction_handler'),
            ml_event_handler=handlers.get('ml_event_handler'),
            ml_hidden_gems_handler=handlers.get('ml_hidden_gems_handler'),
            ml_weather_handler=handlers.get('ml_weather_handler'),
            ml_route_planning_handler=handlers.get('ml_route_planning_handler'),
            ml_neighborhood_handler=handlers.get('ml_neighborhood_handler'),
            response_generator=self.response_generator,
            conversation_handler=self.conversation_handler,
            price_filter_service=self.price_filter_service,
            hidden_gems_handler=self.hidden_gems_handler
        )
        
        logger.info("✅ Routing layer components initialized successfully!")
        
        # System status
        self.system_ready = True
        logger.info("✅ Istanbul Daily Talk AI System initialized successfully!")
        
        # Log cache integration status
        self._log_cache_status()
    
    def _log_cache_status(self):
        """Log ML cache integration status"""
        if hasattr(self, 'gps_route_planner') and self.gps_route_planner:
            if hasattr(self.gps_route_planner, 'ml_cache') and self.gps_route_planner.ml_cache:
                logger.info("✅ ML Prediction Cache integrated into GPS Route Planner")
            else:
                logger.warning("⚠️ ML Prediction Cache not available in GPS Route Planner")
        
        if hasattr(self, 'ml_transport_system') and self.ml_transport_system:
            if hasattr(self.ml_transport_system, 'ibb_client'):
                ibb_client = self.ml_transport_system.ibb_client
                if hasattr(ibb_client, 'ml_cache') and ibb_client.ml_cache:
                    logger.info("✅ ML Prediction Cache integrated into Transportation System")
                else:
                    logger.warning("⚠️ ML Prediction Cache not available in Transportation System")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get ML cache statistics from all integrated systems
        
        Returns:
            Dictionary with cache stats from route planner and transportation system
        """
        stats = {
            'route_planner_cache': None,
            'transportation_cache': None,
            'overall_status': 'unavailable'
        }
        
        try:
            # Get route planner cache stats
            if hasattr(self, 'gps_route_planner') and self.gps_route_planner:
                if hasattr(self.gps_route_planner, 'ml_cache') and self.gps_route_planner.ml_cache:
                    stats['route_planner_cache'] = self.gps_route_planner.ml_cache.get_stats()
                    logger.info("📊 Route Planner Cache Stats retrieved")
            
            # Get transportation system cache stats
            if hasattr(self, 'ml_transport_system') and self.ml_transport_system:
                if hasattr(self.ml_transport_system, 'ibb_client'):
                    ibb_client = self.ml_transport_system.ibb_client
                    if hasattr(ibb_client, 'get_cache_stats'):
                        stats['transportation_cache'] = ibb_client.get_cache_stats()
                        logger.info("📊 Transportation Cache Stats retrieved")
            
            # Determine overall status
            if stats['route_planner_cache'] or stats['transportation_cache']:
                stats['overall_status'] = 'active'
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            stats['error'] = str(e)
            return stats
    
    def invalidate_user_cache(self, user_id: str) -> Dict[str, bool]:
        """
        Invalidate all cached data for a specific user
        
        Use when user preferences change or profile updates
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary showing which caches were invalidated
        """
        result = {
            'route_planner': False,
            'transportation': False
        }
        
        try:
            # Invalidate route planner cache
            if hasattr(self, 'gps_route_planner') and self.gps_route_planner:
                if hasattr(self.gps_route_planner, 'ml_cache') and self.gps_route_planner.ml_cache:
                    self.gps_route_planner.ml_cache.invalidate_user(user_id)
                    result['route_planner'] = True
                    logger.info(f"🗑️ Invalidated route planner cache for user {user_id}")
            
            # Note: Transportation cache is typically not user-specific
            # but we could invalidate patterns if needed
            
            return result
            
        except Exception as e:
            logger.error(f"Error invalidating user cache: {e}")
            result['error'] = str(e)
            return result
    
    def start_conversation(self, user_id: str) -> str:
        """Start a new conversation with personalized greeting"""
        try:
            # Start user session
            session_id = self.user_manager.start_conversation(user_id)
            user_profile = self.user_manager.get_or_create_user_profile(user_id)
            context = self.user_manager.get_conversation_context(session_id)
            
            # Generate personalized greeting
            greeting = self._generate_personalized_greeting(user_profile, context)
            
            # Record interaction
            context.add_interaction("", greeting, "greeting")
            
            return greeting
            
        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            return "🌟 Welcome to Istanbul! I'm your AI guide ready to help you discover this amazing city. How can I assist you today?"
    
    def process_message(self, message: str, user_id: str, return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Process user message and generate response (Week 2: Using modular routing layer)
        
        Args:
            message: User's input message
            user_id: User identifier
            return_structured: If True, return dict with response and map_data; if False, return string (default)
            
        Returns:
            If return_structured=False: String response (backward compatible)
            If return_structured=True: Dict with 'response' (str) and 'map_data' (dict) keys
        """
        try:
            # Validate inputs
            if not isinstance(user_id, str):
                logger.warning(f"Invalid user_id type: {type(user_id)}. Converting to string.")
                user_id = str(user_id)
            
            if not isinstance(message, str):
                logger.warning(f"Invalid message type: {type(message)}. Converting to string.")
                message = str(message)
            
            # Get or create user context
            user_profile = self.user_manager.get_or_create_user_profile(user_id)
            session_id = self.user_manager._get_active_session_id(user_id)
            
            if not session_id:
                session_id = self.user_manager.start_conversation(user_id)
            
            context = self.user_manager.get_conversation_context(session_id)
            
            # Week 2: Use QueryPreprocessor for comprehensive query analysis
            preprocessed_query = self.query_preprocessor.preprocess_query(
                message=message,
                user_profile=user_profile,
                context=context,
                session_id=session_id
            )
            
            # Extract neural insights from preprocessed query
            neural_insights = preprocessed_query.neural_insights
            
            # Check if this is a daily talk query (casual conversation, greetings, weather, etc.)
            if self._is_daily_talk_query(message):
                return self._handle_daily_talk_query(message, user_id, session_id, user_profile, context, neural_insights)
            
            # Week 2: Use EntityExtractor for comprehensive entity extraction
            entities = self.entity_extractor.extract_entities(
                message=message,
                neural_insights=neural_insights,
                user_context=preprocessed_query.user_context
            )
            
            # Week 2: Use IntentClassifier for intelligent intent classification
            intent_result = self.intent_classifier.classify_intent(
                message=message,
                entities=entities,
                context=context,
                neural_insights=neural_insights,
                preprocessed_query=preprocessed_query
            )
            
            # Handle multi-intent queries if detected
            if intent_result.is_multi_intent and intent_result.multi_intent_response:
                context.add_interaction(message, intent_result.multi_intent_response, 'multi_intent')
                if return_structured:
                    return {
                        'response': intent_result.multi_intent_response,
                        'map_data': {},
                        'intent': 'multi_intent',
                        'intents': intent_result.intents
                    }
                return intent_result.multi_intent_response
            
            # Week 2: Use ResponseRouter for intelligent response generation
            response_result = self.response_router.route_query(
                message=message,
                intent=intent_result.primary_intent,
                entities=entities,
                user_profile=user_profile,
                context=context,
                neural_insights=neural_insights,
                return_structured=return_structured
            )
            
            # Extract response text and map_data
            if return_structured and isinstance(response_result, dict):
                response_text = response_result.get('response', '')
                map_data = response_result.get('map_data', {})
            else:
                # Backward compatible - response_result is a string
                response_text = response_result if isinstance(response_result, str) else str(response_result)
                map_data = {}
            
            # Record interaction (use text only)
            context.add_interaction(message, response_text, intent_result.primary_intent)
            
            # Record interaction for personalization system
            if self.personalization_system:
                try:
                    interaction_data = {
                        'type': intent_result.primary_intent,
                        'item_id': f"{intent_result.primary_intent}_{datetime.now().timestamp()}",
                        'item_data': self._extract_personalization_data(entities, intent_result.primary_intent),
                        'rating': 0.7,  # Default positive rating (can be updated with feedback)
                        'timestamp': datetime.now().isoformat()
                    }
                    self.personalization_system.record_interaction(user_id, interaction_data)
                    logger.debug(f"Recorded interaction for personalization: {intent_result.primary_intent}")
                except Exception as e:
                    logger.warning(f"Failed to record personalization interaction: {e}")
            
            # Generate unique interaction ID for feedback
            interaction_id = f"{user_id}_{datetime.now().timestamp()}"
            
            # Return structured or string response based on parameter
            if return_structured:
                return {
                    'response': response_text,
                    'map_data': map_data,
                    'intent': intent_result.primary_intent,
                    'entities': entities,
                    'interaction_id': interaction_id,  # For feedback submission
                    'confidence': intent_result.confidence
                }
            else:
                return response_text
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Provide safe defaults for fallback response
            try:
                if 'context' not in locals():
                    # Create a minimal context for fallback
                    context = ConversationContext()
                if 'user_profile' not in locals():
                    # Create a minimal user profile for fallback
                    user_profile = UserProfile()
                    user_profile.user_id = user_id if isinstance(user_id, str) else "unknown_user"
                return self.response_generator._generate_fallback_response(context, user_profile)
            except Exception as fallback_error:
                logger.error(f"Error in fallback response: {fallback_error}")
                return "🌟 Welcome to Istanbul! I'm your AI guide ready to help you discover this amazing city. How can I assist you today?"

    def _is_daily_talk_query(self, message: str) -> bool:
        """Detect if the message is a daily talk query (casual conversation, greetings, weather, etc.)"""
        
        message_lower = message.lower().strip()
        
        # CRITICAL: Exclude museum/attraction/event queries FIRST (EXPANDED LIST)
        # These should NEVER be classified as daily talk
        exclude_patterns = [
            # Museum and gallery keywords
            'museum', 'museums', 'gallery', 'galleries', 'exhibition', 'exhibitions',
            'art museum', 'history museum', 'archaeological',
            
            # Attraction keywords
            'attraction', 'attractions', 'place to visit', 'places to visit', 'what to see', 'what to visit',
            'palace', 'palaces', 'mosque', 'mosques', 'church', 'churches',
            'landmark', 'landmarks', 'monument', 'monuments', 'memorial',
            'historical site', 'historical sites', 'historic',
            
            # Park and outdoor keywords
            'park', 'parks', 'garden', 'gardens', 'playground', 'green space',
            
            # Sightseeing action words
            'visit', 'see', 'show me', 'recommend', 'suggest', 'tell me about',
            'tour', 'sightseeing', 'explore', 'discover',
            
            # Events and entertainment keywords (IMPORTANT: These are NOT daily talk!)
            'event', 'events', 'concert', 'concerts', 'show', 'shows', 'performance',
            'theater', 'theatre', 'festival', 'festivals', 'cultural events',
            'happening', 'going on', 'activities', 'entertainment', 'nightlife',
            'iksv', 'İKSV', 'salon', 'babylon', 'music event', 'art event',
            
            # Specific services
            'restaurant', 'restaurants', 'eat', 'food', 'dining', 'lunch', 'dinner',
            'transport', 'metro', 'bus', 'ferry', 'how to get', 'directions',
            'shopping', 'bazaar', 'market', 'buy', 'souvenir',
            'hotel', 'stay', 'accommodation', 'neighborhood', 'district', 'area'
        ]
        
        # If message contains any specific travel/tourism intent, it's NOT daily talk
        if any(pattern in message_lower for pattern in exclude_patterns):
            return False
        
        # Greeting patterns
        greeting_patterns = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how r u', 'whats up', "what's up", 'merhaba', 'selam'
        ]
        
        # Weather patterns (only pure weather talk, not "weather-appropriate activities")
        weather_patterns = [
            'what is the weather', 'how is the weather', "what's the weather",
            'weather forecast', 'weather today', 'temperature today', 'will it rain',
            'is it raining', 'is it sunny'
        ]
        
        # Casual conversation patterns
        casual_patterns = [
            'how is your day', 'nice day', 'beautiful day', 'lovely weather',
            'what a day', 'nice weather', 'good day', 'having a good time',
            'nice to meet', 'pleasure to meet', 'thanks', 'thank you', 'bye', 'goodbye'
        ]
        
        # Time-based patterns
        time_patterns = [
            'good morning', 'good afternoon', 'good evening', 'good night'
        ]
        
        # Simple questions about the day
        daily_life_patterns = [
            'how is the day', 'what should i do today', 'any suggestions for today',
            'whats happening today', "what's happening today", 'today', 'this morning',
            'this afternoon', 'this evening', 'tonight'
        ]
        
        # Cultural and local tips
        cultural_patterns = [
            'local tip', 'cultural tip', 'local advice', 'what locals do',
            'like a local', 'authentic experience', 'local culture'
        ]
        
        all_patterns = (greeting_patterns + weather_patterns + casual_patterns + 
                       time_patterns + daily_life_patterns + cultural_patterns)
        
        # Check if message contains daily talk patterns
        for pattern in all_patterns:
            if pattern in message_lower:
                return True
        
        # Check for short casual messages (likely daily talk)
        if len(message.split()) <= 3 and any(word in message_lower for word in 
                                            ['hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok', 'okay']):
            return True
        
        return False

    def _handle_daily_talk_query(self, message: str, user_id: str, session_id: str, 
                                user_profile: UserProfile, context: ConversationContext,
                                neural_insights: Optional[Dict] = None) -> str:
        """Handle daily talk queries through enhanced bilingual system with fallback"""
        
        # PRIMARY: Try Enhanced Bilingual Daily Talks System
        if ENHANCED_DAILY_TALKS_AVAILABLE and self.enhanced_daily_talks:
            try:
                # Prepare context for enhanced daily talks
                daily_talk_context = DailyTalkContext()
                
                # Map language preference to Language enum
                lang_pref = getattr(user_profile, 'language_preference', 'english').lower()
                if lang_pref == 'turkish' or lang_pref == 'tr':
                    daily_talk_context.language = Language.TURKISH
                elif lang_pref == 'english' or lang_pref == 'en':
                    daily_talk_context.language = Language.ENGLISH
                else:
                    daily_talk_context.language = Language.MIXED
                
                # Set user location if available
                if hasattr(user_profile, 'current_location') and user_profile.current_location:
                    daily_talk_context.location = user_profile.current_location
                
                # Set interests
                if hasattr(user_profile, 'interests') and user_profile.interests:
                    daily_talk_context.interests = user_profile.interests
                
                # Transfer conversation history
                if hasattr(context, 'conversation_history') and context.conversation_history:
                    # Convert conversation history to daily talk format
                    for hist in context.conversation_history[-5:]:  # Last 5 exchanges
                        if isinstance(hist, dict):
                            daily_talk_context.conversation_history.append({
                                'user': hist.get('user_message', ''),
                                'ai': hist.get('response', ''),
                                'intent': hist.get('intent', 'unknown'),
                                'timestamp': hist.get('timestamp', datetime.now().isoformat())
                            })
                
                # Generate response using enhanced bilingual system
                response, updated_context = self.enhanced_daily_talks.generate_response(
                    message, daily_talk_context
                )
                
                # Record interaction in main context
                context.add_interaction(message, response, 'daily_talk')
                
                logger.info(f"✅ Enhanced bilingual daily talk response generated (lang: {updated_context.language.value})")
                return response
                
            except Exception as e:
                logger.error(f"Error in enhanced daily talks: {e}")
                # Fall through to legacy ML bridge
        
        # FALLBACK 1: Try ML-Enhanced Daily Talks Bridge (legacy)
        if ML_DAILY_TALKS_AVAILABLE and self.daily_talks_bridge:
            try:
                # Prepare context data for the ML bridge
                user_type_value = 'first_time_visitor'  # default
                if hasattr(user_profile, 'user_type'):
                    if hasattr(user_profile.user_type, 'value'):
                        user_type_value = user_profile.user_type.value
                    elif isinstance(user_profile.user_type, dict):
                        user_type_value = user_profile.user_type.get('value', 'first_time_visitor')
                    else:
                        user_type_value = str(user_profile.user_type)
                
                context_data = {
                    'location': getattr(user_profile, 'current_location', None),
                    'preferences': {
                        'interests': getattr(user_profile, 'interests', []),
                        'user_type': user_type_value,
                        'language_preference': getattr(user_profile, 'language_preference', 'english')
                    },
                    'mood': getattr(context, 'current_mood', None),
                    'weather': None,  # Could be enhanced with real weather data
                    'time_of_day': datetime.now().strftime('%H:%M')
                }
                
                # Process through ML-enhanced daily talks bridge using asyncio
                import asyncio
                try:
                    # Try to get existing event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, fall back to basic response
                        return self._generate_basic_daily_talk_response(message, user_profile, context)
                    else:
                        # No loop running, we can use run_until_complete
                        bridge_result = loop.run_until_complete(
                            self.daily_talks_bridge.process_daily_talk_request(
                                message, user_id, session_id, context_data
                            )
                        )
                except RuntimeError:
                    # No event loop, create one
                    bridge_result = asyncio.run(
                        self.daily_talks_bridge.process_daily_talk_request(
                            message, user_id, session_id, context_data
                        )
                    )
                
                # Extract response from bridge result
                if isinstance(bridge_result, dict) and 'response' in bridge_result:
                    response_data = bridge_result['response']
                    if isinstance(response_data, dict) and 'message' in response_data:
                        response = response_data['message']
                    elif isinstance(response_data, str):
                        response = response_data
                    else:
                        response = str(response_data)
                else:
                    response = str(bridge_result)
                
                # Record interaction
                context.add_interaction(message, response, 'daily_talk')
                
                logger.info("✅ ML bridge daily talk response generated (legacy fallback)")
                return response
                
            except Exception as e:
                logger.error(f"Error in ML daily talks bridge: {e}")
                # Fall through to basic response
        
        # FALLBACK 2: Basic daily talk response
        return self._generate_basic_daily_talk_response(message, user_profile, context)
    
    def _generate_basic_daily_talk_response(self, message: str, user_profile: UserProfile, 
                                          context: ConversationContext) -> str:
        """Generate basic bilingual daily talk response as final fallback"""
        
        message_lower = message.lower()
        current_hour = datetime.now().hour
        
        # Detect language preference from multiple sources
        # 1. Check session_context first
        lang = user_profile.session_context.get('language_preference', 'english').lower()
        # 2. Fallback to direct attribute if available
        if hasattr(user_profile, 'language_preference'):
            lang = getattr(user_profile, 'language_preference', lang).lower()
        
        is_turkish = lang in ['turkish', 'tr', 'türkçe']
        
        # Turkish greeting keywords
        turkish_greetings = ['merhaba', 'selam', 'günaydın', 'iyi günler', 'iyi akşamlar']
        # English greeting keywords
        english_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        
        # Detect language from message if not set in profile
        has_turkish_greeting = any(greet in message_lower for greet in turkish_greetings)
        has_english_greeting = any(greet in message_lower for greet in english_greetings)
        
        # Override language detection based on actual message
        if has_turkish_greeting and not has_english_greeting:
            is_turkish = True
        elif has_english_greeting and not has_turkish_greeting:
            is_turkish = False
        
        # === GREETING RESPONSES ===
        if any(greeting in message_lower for greeting in turkish_greetings + english_greetings):
            if is_turkish:
                if current_hour < 12:
                    return "🌅 Günaydın! İstanbul'u keşfetmek için harika bir gün! Bugün size nasıl yardımcı olabilirim?"
                elif current_hour < 17:
                    return "☀️ İyi günler! İstanbul'u keşfetmek için mükemmel bir zaman! Bugün neyi keşfetmek istersiniz?"
                else:
                    return "🌆 İyi akşamlar! İstanbul'un büyüleyici akşam atmosferi sizi bekliyor! Bu akşam size nasıl yardımcı olabilirim?"
            else:
                if current_hour < 12:
                    return "🌅 Good morning! What a beautiful day to explore Istanbul! How can I help you discover something amazing today?"
                elif current_hour < 17:
                    return "☀️ Good afternoon! Perfect time to explore Istanbul! What would you like to discover today?"
                else:
                    return "🌆 Good evening! Istanbul's evening magic awaits! How can I help you experience the city tonight?"
        
        # === WEATHER RESPONSES ===
        weather_keywords_en = ['weather', 'temperature', 'rain', 'sunny', 'cold', 'hot', 'forecast']
        weather_keywords_tr = ['hava', 'hava durumu', 'sıcaklık', 'yağmur', 'güneşli', 'soğuk', 'sıcak', 'derece']
        
        if any(w in message_lower for w in weather_keywords_en + weather_keywords_tr):
            if is_turkish or any(w in message_lower for w in weather_keywords_tr):
                return ("🌤️ İstanbul'da hava şu anda çok güzel! Havanın tadını çıkararak şehri keşfetmek için "
                        "harika bir zaman. Size şehirde gezebileceğiniz yerler veya yapabileceğiniz aktiviteler "
                        "önerebilirim. Ne yapmak istersiniz?")
            else:
                return ("🌤️ The weather in Istanbul is lovely right now! It's a great time to explore the city. "
                        "I can suggest places to visit or activities that would be perfect for today's conditions. "
                        "What interests you?")
        
        # === THANK YOU RESPONSES ===
        thanks_keywords_en = ['thank', 'thanks', 'appreciate']
        thanks_keywords_tr = ['teşekkür', 'teşekkürler', 'sağol', 'sağolun', 'minnettarım']
        
        if any(t in message_lower for t in thanks_keywords_en + thanks_keywords_tr):
            if is_turkish or any(t in message_lower for t in thanks_keywords_tr):
                return ("🙏 Rica ederim! İstanbul'un en güzel yerlerini keşfetmenize yardımcı olmak için buradayım. "
                        "Başka bilmek istediğiniz bir şey var mı?")
            else:
                return ("🙏 You're very welcome! I'm here to help you discover the best of Istanbul. "
                        "Anything else you'd like to know?")
        
        # === GOODBYE RESPONSES ===
        bye_keywords_en = ['bye', 'goodbye', 'see you', 'farewell']
        bye_keywords_tr = ['hoşça kal', 'hoşçakal', 'güle güle', 'görüşürüz', 'bay']
        
        if any(b in message_lower for b in bye_keywords_en + bye_keywords_tr):
            if is_turkish or any(b in message_lower for b in bye_keywords_tr):
                return ("👋 Güle güle! İstanbul'da harika vakit geçirin! İstediğiniz zaman soru sorabilirsiniz. "
                        "İyi günler! 🌟")
            else:
                return ("👋 Güle güle! (Goodbye in Turkish) Have a wonderful time in Istanbul! "
                        "Feel free to ask me anything anytime! 🌟")
        
        # === HOW ARE YOU RESPONSES ===
        how_are_you_en = ['how are you', 'how do you do', 'how\'s it going']
        how_are_you_tr = ['nasılsın', 'nasılsınız', 'ne haber', 'naber']
        
        if any(h in message_lower for h in how_are_you_en + how_are_you_tr):
            if is_turkish or any(h in message_lower for h in how_are_you_tr):
                return ("😊 Ben harikayım, teşekkür ederim! İstanbul hakkında sizinle konuşmayı seviyorum. "
                        "Size bugün nasıl yardımcı olabilirim? Restoran önerileri, kültürel mekanlar, "
                        "ya da şehirde dolaşma konusunda yardım edebilirim.")
            else:
                return ("😊 I'm doing great, thank you for asking! I love talking about Istanbul with visitors. "
                        "How can I help you today? I can provide restaurant recommendations, cultural insights, "
                        "or help you navigate the city.")
        
        # === CASUAL/SMALL TALK RESPONSES ===
        casual_en = ['nice', 'cool', 'awesome', 'great', 'perfect', 'ok', 'okay']
        casual_tr = ['güzel', 'harika', 'mükemmel', 'süper', 'tamam', 'peki']
        
        if any(c in message_lower for c in casual_en + casual_tr):
            if is_turkish or any(c in message_lower for c in casual_tr):
                return ("😊 Harika! Size daha fazla yardımcı olabilmem için ne öğrenmek istersiniz? "
                        "Restoran tavsiyeleri, müze ve turistik yerler, ulaşım bilgileri, veya İstanbul hakkında "
                        "başka konularda yardımcı olabilirim. Neyle ilgileniyorsunuz?")
            else:
                return ("😊 Great! What would you like to learn more about? I can help with restaurant recommendations, "
                        "museums and attractions, transportation info, or any other questions about Istanbul. "
                        "What interests you most?")
        
        # === DEFAULT FALLBACK RESPONSE ===
        if is_turkish:
            return ("😊 Ben sizin İstanbul yapay zeka rehberinizim, her zaman yardıma hazırım! "
                    "Restoran önerileri, kültürel içgörüler, şehirde gezinme yardımı veya başka herhangi "
                    "bir konuda size yardımcı olabilirim. İstanbul hakkında en çok neyi merak ediyorsunuz?")
        else:
            return ("😊 I'm your Istanbul AI guide, always ready to help! Whether you want restaurant recommendations, "
                    "cultural insights, or help getting around the city, just let me know. "
                    "What interests you most about Istanbul?")
    
    def _generate_personalized_greeting(self, user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate personalized greeting based on user profile"""
        
        # Check if returning user
        if len(context.conversation_history) > 0:
            return f"🎯 Welcome back! Ready to discover more of Istanbul's magic? What adventure are we planning today?"
        
        # Personalize based on user type and preferences
        greetings = {
            'first_time_visitor': "🌟 Merhaba! Welcome to Istanbul for the first time! I'm your personal AI guide, excited to help you discover this incredible city where two continents meet. What brings you to Istanbul?",
            'repeat_visitor': "🎭 Welcome back to Istanbul! As a returning visitor, I can help you discover hidden gems and new experiences beyond the typical tourist spots. What would you like to explore this time?",
            'local_resident': "🏠 Merhaba! As a local, you know Istanbul well, but I can still help you discover new neighborhoods, restaurants, or experiences you might have missed. What's on your mind today?",
            'business_traveler': "💼 Welcome to Istanbul! I know your time is precious during business travel. I can help you find efficient ways to experience the city's highlights or great places for business meals. How can I assist?",
            'cultural_explorer': "🎨 Hoş geldiniz! Perfect timing for a cultural explorer - Istanbul offers layers of Byzantine, Ottoman, and modern Turkish culture. I'm excited to guide you through authentic experiences. Where shall we start?"
        }
        
        base_greeting = greetings.get(user_profile.user_type.value, greetings['first_time_visitor'])
        
        # Add personalization if we have interests
        if user_profile.interests:
            interests_text = ', '.join(user_profile.interests[:2])
            base_greeting += f" I see you're interested in {interests_text} - I have some amazing recommendations for you!"
        
        return base_greeting
    
    def _classify_intent_with_context(self, message: str, entities: Dict, context: ConversationContext) -> str:
        """DEPRECATED: Legacy intent classification method (Week 2 refactoring)
        
        This method has been replaced by IntentClassifier in the routing layer.
        Kept for backward compatibility only.
        
        Use: self.intent_classifier.classify_intent() instead
        """
        logger.warning("⚠️ Using deprecated _classify_intent_with_context method. Use IntentClassifier instead.")
        
        # Delegate to the new routing layer for backward compatibility
        intent_result = self.intent_classifier.classify_intent(
            message=message,
            entities=entities,
            context=context
        )
        return intent_result.primary_intent
        
        message_lower = message.lower()
        
        # Delegate to IntentClassifier for backward compatibility
        logger.warning("⚠️ Using deprecated _classify_intent_with_context logic. This method will be removed in future versions.")
        
        # Use the new IntentClassifier
        intent_result = self.intent_classifier.classify_intent(
            message=message,
            entities=entities,
            context=context
        )
        
        return intent_result.primary_intent
    
    def _generate_contextual_response(self, message: str, intent: str, entities: Dict,
                                    user_profile: UserProfile, context: ConversationContext, 
                                    neural_insights: Optional[Dict] = None, return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """DEPRECATED: Legacy response generation method (Week 2 refactoring)
        
        This method has been replaced by ResponseRouter in the routing layer.
        Kept for backward compatibility only.
        
        Use: self.response_router.route_query() instead
        """
        logger.warning("⚠️ Using deprecated _generate_contextual_response method. Use ResponseRouter instead.")
        
        # Delegate to the new routing layer for backward compatibility
        return self.response_router.route_query(
            message=message,
            intent=intent,
            entities=entities,
            user_profile=user_profile,
            context=context,
            neural_insights=neural_insights,
            return_structured=return_structured
        )
    
    def _generate_transportation_response(self, message: str, entities: Dict, user_profile: UserProfile, 
                                        context: ConversationContext, neural_insights: Optional[Dict] = None,
                                        return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate transportation response with neural insights
        
        Args:
            message: User's query
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
            neural_insights: ML-powered insights (sentiment, temporal context, keywords, etc.)
            return_structured: Whether to return structured response
            
        Returns:
            Transportation response (string or dict based on return_structured)
        """
        try:
            # Extract temporal and sentiment context from neural insights
            temporal_context = neural_insights.get('temporal_context') if neural_insights else None
            sentiment = neural_insights.get('sentiment') if neural_insights else None
            
            logger.info(f"🧠 Transportation query with ML insights: temporal={temporal_context}, sentiment={sentiment}")
            
            # Check if this is a specific route request
            route_indicators = [
                'from', 'to', 'how to get', 'how do i get', 'how can i get', 
                'how to go', 'how do i go', 'how can i go',
                'directions', 'route from', 'route to', 'way to get', 'way to go',
                'get to', 'go to', 'travel to', 'reach'
            ]
            is_route_query = any(indicator in message.lower() for indicator in route_indicators)
            
            # Use new transfer instructions & map visualization integration if available
            if is_route_query and TRANSFER_MAP_INTEGRATION_AVAILABLE and self.transportation_chat:
                logger.info("🗺️ Using Transfer Instructions & Map Visualization system")
                
                # Extract locations from entities if available
                user_location = None
                destination = None
                
                if 'location' in entities and entities['location']:
                    locations = entities['location']
                    if len(locations) >= 2:
                        user_location = locations[0]
                        destination = locations[1]
                    elif len(locations) == 1:
                        destination = locations[0]
                
                # Process the transportation query (async)
                import asyncio
                loop = asyncio.get_event_loop()
                
                # Build intelligent user context using ML insights
                user_context = self._build_intelligent_user_context(message, neural_insights, user_profile)
                
                result = loop.run_until_complete(
                    self.transportation_chat.handle_transportation_query(
                        query=message,
                        user_location=user_location,
                        destination=destination,
                        user_context=user_context
                    )
                )
                
                if result.get('success'):
                    response_text = result.get('response_text', '')
                    map_data = result.get('map_data', {})
                    
                    if return_structured:
                        return {
                            'response': response_text,
                            'map_data': map_data,
                            'detailed_route': result.get('detailed_route'),
                            'alternatives': result.get('alternatives', []),
                            'fare_info': result.get('fare_info'),
                            'transfer_count': result.get('transfer_count', 0),
                            'total_time': result.get('total_time', 0)
                        }
                    else:
                        return response_text
                else:
                    # If clarification needed or error, fall through to other systems
                    if result.get('needs_clarification'):
                        return result.get('response_text', '')
            
            # Check if this is a GPS route request (use GPS route planner)
            if is_route_query:
                logger.info("🗺️ Using GPS route planner for route-specific query")
                return self._generate_gps_route_response(message, entities, user_profile, context)
            
            # Use advanced transportation system if available for general transport info
            if ADVANCED_TRANSPORT_AVAILABLE and self.transport_processor:
                logger.info("🚇 Using advanced transportation system with IBB API")
                
                # Process query through advanced system using the actual message
                enhanced_response = self.transport_processor.process_transportation_query_sync(
                    message, entities, user_profile
                )
                
                if enhanced_response and enhanced_response.strip():
                    if return_structured:
                        return {
                            'response': enhanced_response,
                            'map_data': {}
                        }
                    else:
                        return enhanced_response
                    
            # Fallback to improved static response
            logger.info("🚇 Using fallback transportation system")
            response = self._get_fallback_transportation_response(entities, user_profile, context)
            
            if return_structured:
                return {
                    'response': response,
                    'map_data': {}
                }
            else:
                return response
            
        except Exception as e:
            logger.error(f"Transportation query error: {e}")
            response = self._get_fallback_transportation_response(entities, user_profile, context)
            if return_structured:
                return {
                    'response': response,
                    'map_data': {}
                }
            else:
                return response

    def _get_fallback_transportation_response(self, entities: Dict, user_profile: UserProfile, 
                                            context: ConversationContext) -> str:
        """Fallback transportation response with correct information"""
        current_time = datetime.now().strftime("%H:%M")
        
        return f"""🚇 **Istanbul Transportation Guide**
📍 **Live Status** (Updated: {current_time})

**🎫 Essential Transport Card:**
• **Istanbulkart**: Must-have for official public transport (13 TL + credit)
• Available at metro stations, kiosks, and ferry terminals
• Works on metro, tram, bus, ferry, and metrobüs (NOT on dolmuş - cash only)

**🚇 Metro Lines:**
• **M1A**: Yenikapı ↔ Atatürk Airport (closed) - serves Grand Bazaar area
• **M2**: Vezneciler ↔ Hacıosman (serves Taksim, Şişli, Levent)
• **M4**: Kadıköy ↔ Sabiha Gökçen Airport (Asian side)
• **M11**: IST Airport ↔ Gayrettepe (new airport connection)
• **M6**: Levent ↔ Boğaziçi Üniversitesi

**🚋 Historic Trams:**
• **T1**: Kabataş ↔ Bağcılar (connects Sultanahmet, Eminönü, Karaköy)
• **Nostalgic Tram**: Taksim ↔ Tünel (historic Istiklal Street)

**⛴️ Ferries (Most Scenic!):**
• **Eminönü ↔ Kadıköy**: 20 minutes, beautiful city views
• **Karaköy ↔ Üsküdar**: Quick cross-Bosphorus connection
• **Bosphorus Tours**: 1.5-hour scenic cruises (90-150 TL)

**🚌 Buses & Dolmuş:**
• Extensive network but can be crowded
• Dolmuş (shared taxis) follow set routes - cash payment only, no Istanbulkart
• Look for destination signs in Turkish and English

**💡 Pro Tips:**
• Download Citymapper or Moovit apps for real-time directions
• Rush hours: 7:30-9:30 AM, 5:30-7:30 PM
• Ferries often faster than road transport across Bosphorus
• Keep Istanbulkart handy - inspectors check frequently
• Metro runs until midnight, limited night bus service

**🎯 Popular Routes:**
• **IST Airport → Sultanahmet**: M11 + M2 + T1 (60 min, ~20 TL)
• **Taksim → Sultanahmet**: M2 + T1 (25 min, ~7 TL)  
• **Sultanahmet → Galata Tower**: T1 + M2 (25 min)
• **European → Asian side**: Ferry from Eminönü/Karaköy

Need specific route directions? Tell me your starting point and destination!"""
    
    def _generate_shopping_response(self, entities: Dict, user_profile: UserProfile, 
                                   context: ConversationContext, neural_insights: Dict = None) -> str:
        """Generate comprehensive shopping response (ML-enhanced)"""
        
        return """🛍️ **Istanbul Shopping Paradise**

**🏛️ Historic Markets:**
• **Grand Bazaar (Kapalıçarşı)**: 4,000 shops, carpets, jewelry, ceramics
  - Hours: 9:00-19:00 (closed Sundays)
  - Haggling expected, start at 30-40% of asking price
• **Spice Bazaar (Mısır Çarşısı)**: Turkish delight, spices, tea
  - Perfect for authentic food souvenirs
• **Arasta Bazaar**: Smaller, less crowded alternative near Blue Mosque

**🛒 Modern Shopping:**
• **Istinye Park**: Luxury brands, beautiful architecture
• **Cevahir**: Largest mall in Europe, all price ranges
• **Kanyon**: Upscale shopping in trendy Levent

**🎨 Unique Districts:**
• **Nişantaşı**: Turkish designers, high-end fashion
• **Galata/Beyoğlu**: Vintage shops, antiques, indie boutiques
• **Çukurcuma**: Antique furniture, vintage items

**🎁 Best Souvenirs:**
• **Turkish Carpets**: Hand-woven, get certificates of authenticity
• **Ceramics**: Traditional Kütahya and İznik designs
• **Turkish Delight (Lokum)**: Hacı Bekir (since 1777) is the original
• **Evil Eye (Nazar Boncuğu)**: Protection charm in all sizes
• **Turkish Tea & Coffee**: Freshly ground, try Selamlique or Kurukahveci Mehmet

**💰 Budget Tips:**
• **High-end**: Nişantaşı, Istinye Park (100-1000+ TL)
• **Mid-range**: Grand Bazaar after haggling (50-300 TL)  
• **Budget**: Mahmutpaşa district, local markets (10-100 TL)

**🎯 Haggling Guide:**
• Expected in bazaars, not in modern stores
• Be respectful and smile
• Start at 40% of asking price
• Walk away if not satisfied - often they'll call you back
• Cash payments often get better prices

**📍 Shopping Routes:**
• **Historic**: Sultanahmet → Grand Bazaar → Spice Bazaar
• **Modern**: Taksim → Nişantaşı → Istinye Park  
• **Alternative**: Galata → Karaköy → Çukurcuma

What type of shopping interests you most? I can provide specific store recommendations!"""
    
    def _generate_events_response(self, entities: Dict, user_profile: UserProfile, 
                                 context: ConversationContext, current_time: datetime,
                                 neural_insights: Dict = None) -> str:
        """Generate events and activities response with ML-enhanced temporal parsing"""
        
        # Get the original query message
        query_message = context.conversation_history[-1].message if context.conversation_history else ""
        
        # Try to use events service for temporal parsing
        events_content = ""
        if self.events_service:
            try:
                # Parse temporal query (today, tonight, this weekend, etc.)
                timeframe = self.events_service.parse_temporal_query(query_message)
                
                if timeframe:
                    # Get events for the specified timeframe
                    events = self.events_service.get_events_by_timeframe(timeframe)
                    
                    if events:
                        # Format events using the service's formatter
                        events_content = self.events_service.format_events_response(
                            events, 
                            timeframe.get('label'),
                            include_iksv=True
                        )
                        # Return the formatted events directly
                        return events_content
                
            except Exception as e:
                logger.error(f"Error using events service: {e}")
        
        # Try to get live IKSV events through ML Daily Talks Bridge
        live_events_section = ""
        if self.daily_talks_bridge:
            try:
                import asyncio
                # Create context for IKSV events fetching
                from ml_enhanced_daily_talks_bridge import ConversationContext as MLContext, UserProfile as MLProfile
                
                ml_user_profile = MLProfile(
                    user_id=user_profile.user_id,
                    preferences={"interests": ["culture", "events", "arts"]},
                    interaction_history=[],
                    personality_traits={},
                    location_preferences={},
                    activity_patterns={}
                )
                ml_context = MLContext(context.session_id, ml_user_profile, [])
                ml_context.current_location = context.current_location or "Istanbul"
                
                # Fetch live IKSV events
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't await in sync function, skip live events for now
                        pass
                    else:
                        iksv_events = loop.run_until_complete(
                            self.daily_talks_bridge._fetch_relevant_iksv_events(ml_context)
                        )
                        
                        if iksv_events and len(iksv_events) > 0:
                            live_events_section = "\n**🎭 Live İKSV Events:**\n"
                            for i, event in enumerate(iksv_events[:3]):  # Show top 3
                                title = event.get('title', 'Event')
                                venue = event.get('venue', 'Istanbul')
                                date = event.get('date', 'Check schedule')
                                live_events_section += f"• **{title}** - {venue}\n"
                                if date != 'Check schedule':
                                    live_events_section += f"  📅 {date}\n"
                            live_events_section += "  💡 Visit iksv.org for tickets and details\n"
                except Exception:
                    # Fallback to static content if async fails
                    pass
                    
            except Exception as e:
                logger.debug(f"Could not fetch live IKSV events: {e}")
        
        # Add seasonal events if events service is available
        seasonal_section = ""
        if self.events_service:
            try:
                seasonal_events = self.events_service.get_seasonal_events()
                if seasonal_events:
                    seasonal_section = "\n**🎪 Current Season Highlights:**\n"
                    for event in seasonal_events[:3]:
                        seasonal_section += f"• **{event['name']}** ({event.get('month', 'TBA')})\n"
                        if event.get('description'):
                            seasonal_section += f"  {event['description']}\n"
            except Exception as e:
                logger.debug(f"Could not get seasonal events: {e}")
        
        return f"""🎭 **Istanbul Events & Activities**
{live_events_section}
{seasonal_section}

**🎨 Cultural Events:**
• **Istanbul Modern**: Contemporary art exhibitions, Bosphorus views
• **Pera Museum**: Rotating exhibitions, Orientalist paintings
• **Turkish baths (Hamam)**: Cagaloglu Hamami (historic), Kilic Ali Pasha
• **Traditional Shows**: Whirling Dervishes at various venues

**🌙 Evening Entertainment:**
• **Bosphorus Dinner Cruise**: Dinner with city lights (150-300 TL)
• **Rooftop Bars**: 360 Istanbul, Mikla, Leb-i Derya
• **Live Music**: Babylon, Salon IKSV, Nardis Jazz Club
• **Traditional Music**: Turkish folk at cultural centers

**🌊 Bosphorus Activities:**
• **Ferry Tours**: Public ferries (15 TL) vs private tours (100+ TL)
• **Sunset Cruises**: Most romantic time, book in advance
• **Fishing Tours**: Traditional experience with local fishermen
• **Water Sports**: Kayaking, boat rentals in calmer areas

**🎯 Time-Based Recommendations:**

**Morning (9-12):**
• Museum visits before crowds
• Traditional Turkish breakfast experiences
• Bosphorus morning ferry rides

**Afternoon (12-17):**
• Shopping in covered markets
• Turkish bath experiences  
• Neighborhood walking tours

**Evening (17-22):**
• Sunset from Galata Tower
• Traditional dinner with entertainment
• Istiklal Street evening stroll (1 hour)

**Night (22+):**
• Rooftop bar hopping in Beyoğlu
• Traditional meyhane (tavern) experiences
• Late-night Bosphorus illumination tours

**💡 Booking Tips:**
• Many cultural venues offer online booking
• Friday evenings are busiest for entertainment
• Traditional shows often include dinner packages
• Check weather for outdoor activities

**🎫 Useful Apps:**
• Biletix: Major event ticketing
• Istanbul Municipality: Free cultural events
• Time Out Istanbul: Current happenings

💡 **Tip:** Ask me about events "today", "tonight", "this weekend" or specific days for detailed schedules!"""
    
    def _generate_route_planning_response(self, message: str, user_profile: UserProfile, 
                                        context: ConversationContext, neural_insights: Dict = None) -> str:
        """Generate ML-enhanced route planning response"""
        
        return """🗺️ **Istanbul Itinerary Planning**

**🌅 Classic One-Day Route:**
**Morning (9-12):**
• Start at **Hagia Sophia** (1 hour)
• Walk to **Blue Mosque** (30 min)
• **Traditional breakfast** nearby (45 min)

**Afternoon (12-17):**
• **Grand Bazaar** shopping (1-2 hours) 
• Walk to **Spice Bazaar** (30 min)
• **Ferry to Asian side** for views (30 min)
• Return and explore **Galata area** (1 hour)

**Evening (17-21):**
• **Galata Tower** for sunset (45 min)
• **Dinner in Beyoğlu** (1.5 hours)
• **Istiklal Street** evening stroll (1 hour)

**🏛️ History-Focused Route:**
• **Topkapi Palace** (2-3 hours) → **Hagia Sophia** → **Basilica Cistern**
• **Blue Mosque** → **Hippodrome** → **Turkish & Islamic Arts Museum**
• Transport: All walkable in Sultanahmet area

**🍽️ Food Tour Route:**
• **Traditional breakfast** in Sultanahmet
• **Street food** at Eminönü (balık ekmek, simit)
• **Lunch** at historic restaurant (Pandeli)
• **Turkish delight tasting** at Spice Bazaar
• **Dinner** with Bosphorus view in Beyoğlu

**🌉 Cross-Continental Route:**
• Morning: **European side** historic sites
• Midday: **Ferry across Bosphorus** (scenic!)
• Afternoon: **Asian side** (Kadıköy market, Moda walk)
• Evening: **Return via ferry** for sunset views

**⏰ Time Optimization:**
• **Half day (4 hours)**: Sultanahmet core sites
• **Full day (8 hours)**: Add shopping + one neighborhood
• **Two days**: Split historic/modern, include Asian side
• **Three days**: Add Bosphorus cruise, second neighborhoods

**🎯 Personalized Suggestions:**

**For Art Lovers:**
Istanbul Modern → Pera Museum → Galata Tower → Street art in Karaköy

**For Food Enthusiasts:**  
Market tours → Cooking class → Traditional restaurants → Street food crawl

**For History Buffs:**
Archaeological Museum → Topkapi → Hagia Sophia → Byzantine sites

**💡 Practical Tips:**
• Buy **Museum Pass** (₺850) for 12+ museums
• Start early (9 AM) to avoid crowds
• Wear comfortable shoes
• Keep Istanbulkart handy
• Plan indoor backup for weather
• Book dinner reservations in advance

**🚇 Transport Integration:**
• **Sultanahmet Tram** connects all historic sites
• **Ferry rides** double as sightseeing
• **Metro + tram combos** for cross-district travel
• **Walking** often faster than transport in old city

How many days do you have? What are your main interests? I can create a detailed personalized itinerary!"""
    
    def _generate_greeting_response(self, user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate friendly greeting response"""
        
        return """🌟 **Merhaba! Welcome to your Istanbul adventure!**

I'm your personal AI guide for this incredible city where Europe meets Asia! Whether you're here for the first time or returning to discover new experiences, I'm excited to help you explore Istanbul's magic.

**🎯 I can help you with:**
• **🍽️ Restaurant recommendations** - From street food to Ottoman cuisine
• **🏛️ Historic attractions** - Byzantine, Ottoman, and modern sites  
• **🏘️ Neighborhood guides** - Each district has unique character
• **🚇 Transportation** - Navigate the city like a local
• **🛍️ Shopping** - From Grand Bazaar to modern districts
• **🎭 Events & activities** - Cultural experiences and entertainment
• **🗺️ Route planning** - Personalized itineraries for your time

**💡 Just tell me:**
• What interests you most?
• How much time do you have?
• Any dietary restrictions or accessibility needs?
• Traveling solo, couple, family, or group?
• Budget preferences?

**🌅 Quick suggestions to get started:**
• "Show me the best Turkish breakfast spots"
• "Plan a one-day historic tour"  
• "Where should I stay in Istanbul?"
• "How do I get from airport to city center?"
• "What's the best way to see the Bosphorus?"

What would you like to explore first? I'm here to make your Istanbul experience unforgettable! ✨"""
    
    def _detect_multiple_intents(self, message: str, entities: Dict) -> List[str]:
        """Detect multiple intents in a single message"""
        
        intents = []
        message_lower = message.lower()
        
        intent_keywords = {
            'restaurant': ['eat', 'food', 'restaurant', 'lunch', 'dinner'],
            'attraction': ['visit', 'see', 'attraction', 'museum'],
            'transportation': ['transport', 'metro', 'bus', 'how to get'],
            'neighborhood': ['neighborhood', 'area', 'district'],
            'shopping': ['shop', 'shopping', 'buy', 'bazaar', 'market', 'souvenir']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                intents.append(intent)
        
        return intents if intents else ['general']
    
    def _generate_location_aware_museum_response(self, message: str, entities: Dict, 
                                               user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate location-aware museum recommendations using simplified detection"""
        
        intent_keywords = {
            'restaurant': ['eat', 'food', 'restaurant', 'lunch', 'dinner'],
            'attraction': ['visit', 'see', 'attraction', 'museum'],
            'transportation': ['transport', 'metro', 'bus', 'how to get'],
            'neighborhood': ['neighborhood', 'area', 'district'],
            'shopping': ['shop', 'shopping', 'buy', 'bazaar', 'market', 'souvenir']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                intents.append(intent)
        
        return intents if intents else ['general']
    
    def _generate_location_aware_museum_response(self, message: str, entities: Dict, 
                                               user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate location-aware museum recommendations using simplified detection"""
        
        if not self.museum_generator:
            # Fallback to basic response if museum system not available
            return self.response_generator.generate_comprehensive_recommendation(
                'attraction', entities, user_profile, context
            )
        
        # Step 1: Simple location detection from message text
        detected_location = self._detect_location_from_message(message)
        
        if detected_location:
            logger.info(f"🌍 Location detected: {detected_location}")
        
        # Generate base museum response
        museum_response = self.museum_generator.generate_museum_recommendation(message)
        
        # Enhance with location-specific information
        if detected_location:
            location_enhancement = self._generate_location_specific_museum_info(detected_location, message)
            if location_enhancement:
                museum_response = f"{museum_response}\n\n{location_enhancement}"
        
        # Add current hours information using Google Maps checker
        hours_info = self._add_current_museum_hours(message)
        if hours_info:
            museum_response = f"{museum_response}\n\n{hours_info}"
        
        return museum_response

    def _generate_location_specific_museum_info(self, location: str, message: str) -> Optional[str]:
        """Generate location-specific museum information"""
        location_museums = {
            'sultanahmet': {
                'museums': ['Hagia Sophia', 'Topkapi Palace', 'Istanbul Archaeology Museums', 'Turkish and Islamic Arts Museum'],
                'info': '📍 Sultanahmet is the heart of historic Istanbul with world-class museums within walking distance.'
            },
            'beyoglu': {
                'museums': ['Istanbul Modern', 'Pera Museum', 'SALT Beyoğlu'],
                'info': '🎨 Beyoğlu features contemporary art museums and cultural centers near Istiklal Street.'
            },
            'beşiktaş': {
                'museums': ['Dolmabahçe Palace', 'Naval Museum', 'Painting and Sculpture Museum'],
                'info': '🏰 Beşiktaş offers Ottoman palaces and specialized museums along the Bosphorus.'
            }
        }
        
        if location in location_museums:
            data = location_museums[location]
            info = f"\n{data['info']}\n"
            info += f"📍 **Museums in {location.title()}:**\n"
        
        return None

    def _generate_advanced_museum_response(self, message: str, entities: Dict, 
                                         user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate advanced ML-powered museum recommendations with detailed information"""
        
        if not self.advanced_museum_system:
            logger.warning("Advanced museum system not available, falling back")
            return self._generate_location_aware_museum_response(message, entities, user_profile, context)
        
        try:
            logger.info(f"🎨 Processing museum query with Advanced Museum System: {message}")
            
            # Extract GPS location if available from entities or user profile
            gps_location = None
            if 'location' in entities and entities['location']:
                loc = entities['location']
                if isinstance(loc, dict) and 'lat' in loc and 'lon' in loc:
                    gps_location = (loc['lat'], loc['lon'])
            
            # Convert user profile to dict for the museum system
            user_profile_dict = {
                'interests': user_profile.interests if hasattr(user_profile, 'interests') else [],
                'budget_range': user_profile.budget_range if hasattr(user_profile, 'budget_range') else 'moderate',
                'accessibility_needs': user_profile.accessibility_needs if hasattr(user_profile, 'accessibility_needs') else False
            }
            
            # Get museum recommendations using advanced system methods
            from datetime import datetime
            message_lower = message.lower()
            
            # Determine what type of query this is
            museums = []
            
            # Check for specific museum name
            for museum_id, museum in self.advanced_museum_system.museums.items():
                if museum.name.lower() in message_lower:
                    museums = [museum]
                    break
            
            # If no specific museum, get recommendations based on filters
            if not museums:
                # Check for category filters
                if any(word in message_lower for word in ['art', 'modern art', 'contemporary']):
                    from museum_advising_system import MuseumCategory
                    museums = self.advanced_museum_system.get_museums_by_category(MuseumCategory.ART_MODERN)
                elif any(word in message_lower for word in ['historical', 'history', 'archaeological']):
                    from museum_advising_system import MuseumCategory
                    museums = self.advanced_museum_system.get_museums_by_category(MuseumCategory.HISTORICAL)
                elif any(word in message_lower for word in ['archaeological', 'archaeology']):
                    from museum_advising_system import MuseumCategory
                    museums = self.advanced_museum_system.get_museums_by_category(MuseumCategory.ARCHAEOLOGICAL)
                
                # Check for district filters
                elif 'sultanahmet' in message_lower or 'fatih' in message_lower:
                    museums = self.advanced_museum_system.get_museums_by_district('Sultanahmet')
                elif 'beyoglu' in message_lower or 'beyoğlu' in message_lower:
                    museums = self.advanced_museum_system.get_museums_by_district('Beyoğlu')
                elif 'besiktas' in message_lower or 'beşiktaş' in message_lower:
                    museums = self.advanced_museum_system.get_museums_by_district('Beşiktaş')
                
                # Check for budget filters
                elif any(word in message_lower for word in ['free', 'no entrance', 'no fee']):
                    museums = self.advanced_museum_system.get_museums_by_price_range(0)
                
                # GPS-based search if location provided
                elif gps_location:
                    museums = self.advanced_museum_system.get_museums_by_location(gps_location, radius_km=5.0)
                
                # Default: Get general recommendations
                else:
                    museums = self.advanced_museum_system.get_museum_recommendations(
                        user_profile_dict, 
                        user_location=gps_location
                    )
            
            # Format detailed response
            if museums:
                response = self._format_detailed_museums_response(museums, gps_location, message)
                logger.info(f"✅ Generated detailed museum response with {len(museums)} museums")
                return response
            else:
                logger.warning("No museums found, falling back")
                return self._generate_location_aware_museum_response(message, entities, user_profile, context)
                
        except Exception as e:
            logger.error(f"Error in advanced museum response: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_location_aware_museum_response(message, entities, user_profile, context)

    def _generate_advanced_attractions_response(self, message: str, entities: Dict,
                                               user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate advanced attractions recommendations with category/district filtering"""
        
        if not self.advanced_attractions_system:
            logger.warning("Advanced attractions system not available, falling back")
            return self.response_generator.generate_comprehensive_recommendation(
                'attraction', entities, user_profile, context
            )
        
        try:
            logger.info(f"🌟 Processing attractions query with Advanced Attractions System: {message}")
            
            message_lower = message.lower()
            
            # ENHANCED: Detect family-friendly queries
            family_keywords = [
                'family', 'families', 'kids', 'children', 'child', 'kid',
                'toddler', 'toddlers', 'baby', 'babies', 'infant',
                'age', 'aged', 'year old', 'years old',
                'son', 'daughter', 'boy', 'girl', 'boys', 'girls',
                'playground', 'play area', 'family-friendly',
                'suitable for kids', 'suitable for children',
                'safe for kids', 'safe for children'
            ]
            is_family_query = any(keyword in message_lower for keyword in family_keywords)
            
            # ENHANCED: Detect weather-specific queries
            weather_keywords = {
                'indoor': 'indoor',
                'outdoor': 'outdoor',
                'rainy': 'indoor',
                'rain': 'indoor',
                'sunny': 'outdoor',
                'hot': 'outdoor',
                'cold': 'indoor',
                'covered': 'covered'
            }
            weather_preference = None
            for keyword, pref in weather_keywords.items():
                if keyword in message_lower:
                    weather_preference = pref
                    break
            
            # ENHANCED: Detect GPS coordinates more reliably
            gps_pattern = r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
            gps_match = re.search(gps_pattern, message)
            user_location = None
            if gps_match:
                try:
                    lat, lon = float(gps_match.group(1)), float(gps_match.group(2))
                    user_location = (lat, lon)
                    logger.info(f"📍 GPS coordinates detected: {user_location}")
                except ValueError:
                    pass
            
            # Try to find specific attraction using search
            search_results = self.advanced_attractions_system.search_attractions(message)
            
            # If we have a strong match (score > 0.7), show that attraction
            if search_results and search_results[0][1] > 0.7:
                top_attraction = search_results[0][0]
                return self._format_single_attraction(top_attraction)
            
            # Otherwise, get recommendations based on filters
            category = self._extract_attraction_category(message)
            district = self._extract_district(message)
            
            # Build recommendations based on filters
            attractions = []
            
            if category:
                # ENHANCED: More complete category mapping with museum, park, monument
                category_map = {
                    'museum': 'MUSEUM',
                    'park': 'PARK_GARDEN',
                    'monument': 'HISTORICAL_MONUMENT',
                    'historic': 'HISTORICAL_MONUMENT',
                    'religious': 'RELIGIOUS_SITE',
                    'palace': 'PALACE_MANSION',
                    'market': 'MARKET_BAZAAR',
                    'waterfront': 'WATERFRONT',
                    'tower': 'VIEWPOINT',
                    'modern': 'CULTURAL_CENTER',
                    'family': 'FAMILY_ATTRACTION',
                    'romantic': 'ROMANTIC_SPOT'
                }
                category_enum_name = category_map.get(category, 'HISTORICAL_MONUMENT')
                logger.info(f"🎯 Category detected: {category} → {category_enum_name}")
                
                # Find matching enum value
                from istanbul_attractions_system import AttractionCategory
                for cat_enum in AttractionCategory:
                    if cat_enum.name == category_enum_name:
                        attractions = self.advanced_attractions_system.get_attractions_by_category(cat_enum)
                        logger.info(f"📊 Found {len(attractions)} attractions in category {category_enum_name}")
                        break
            
            elif district:
                attractions = self.advanced_attractions_system.get_attractions_by_district(district)
            
            # If no filters matched or no results, use general recommendations
            if not attractions:
                # Build preference dict from user profile
                preferences = {
                    'interests': user_profile.interests if hasattr(user_profile, 'interests') else [],
                    'budget': user_profile.budget_range if hasattr(user_profile, 'budget_range') else 'moderate',
                    'accessibility': user_profile.accessibility_needs if hasattr(user_profile, 'accessibility_needs') else False
                }
                attractions = self.advanced_attractions_system.get_attraction_recommendations(preferences)
            
            # ENHANCED: Apply family-friendly filter if detected
            if is_family_query and attractions:
                family_friendly_attractions = [
                    a for a in attractions 
                    if hasattr(a, 'is_family_friendly') and a.is_family_friendly
                ]
                if family_friendly_attractions:
                    attractions = family_friendly_attractions
                    logger.info(f"👨‍👩‍👧 Applied family-friendly filter: {len(attractions)} attractions")
            
            # ENHANCED: Apply weather filter if detected
            if weather_preference and attractions:
                from istanbul_attractions_system import WeatherPreference
                weather_filtered = []
                for a in attractions:
                    if hasattr(a, 'weather_preference'):
                        if weather_preference == 'indoor' and a.weather_preference in [WeatherPreference.INDOOR, WeatherPreference.ALL_WEATHER]:
                            weather_filtered.append(a)
                        elif weather_preference == 'outdoor' and a.weather_preference in [WeatherPreference.OUTDOOR, WeatherPreference.ALL_WEATHER]:
                            weather_filtered.append(a)
                        elif weather_preference == 'covered' and a.weather_preference in [WeatherPreference.COVERED, WeatherPreference.ALL_WEATHER]:
                            weather_filtered.append(a)
                if weather_filtered:
                    attractions = weather_filtered
                    logger.info(f"🌤️ Applied weather filter ({weather_preference}): {len(attractions)} attractions")
            
            # ENHANCED: Sort by GPS distance if location provided
            if user_location and attractions:
                def calculate_distance(attraction):
                    if hasattr(attraction, 'coordinates') and attraction.coordinates:
                        from math import radians, sin, cos, sqrt, atan2
                        lat1, lon1 = user_location
                        lat2, lon2 = attraction.coordinates
                        # Convert to radians
                        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                        # Haversine formula
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * atan2(sqrt(a), sqrt(1-a))
                        r = 6371  # Earth's radius in kilometers
                        return c * r
                    return float('inf')  # Default to infinite distance if no coordinates
                
                attractions.sort(key=calculate_distance)
                logger.info(f"📍 Sorted {len(attractions)} attractions by GPS distance")
            
            # Format response
            if attractions:
                return self._format_attractions_list(attractions, category=category, district=district)
            else:
                logger.warning("Advanced Attractions System returned no results, falling back")
                return self.response_generator.generate_comprehensive_recommendation(
                    'attraction', entities, user_profile, context
                )

        except Exception as e:
            logger.error(f"Error in advanced attractions response: {e}")
            import traceback
            traceback.print_exc()
            return self.response_generator.generate_comprehensive_recommendation(
                'attraction', entities, user_profile, context
            )

    def _format_single_attraction(self, attraction) -> str:
        """Format a single attraction detail"""
        response = f"🌟 **{attraction.name}**\n\n"
        response += f"📍 **Location:** {attraction.district} • {attraction.address}\n"
        response += f"🎯 **Category:** {attraction.category.value.replace('_', ' ').title()}\n"
        
        # Use 'duration' (correct attribute name)
        if hasattr(attraction, 'duration') and attraction.duration:
            response += f"⏰ **Visit Duration:** {attraction.duration}\n"
        
        response += f"🎫 **Entrance Fee:** {attraction.entrance_fee.value.replace('_', ' ').title()}\n"
        
        # Only show rating if available
        if hasattr(attraction, 'rating') and hasattr(attraction, 'reviews_count'):
            response += f"⭐ **Rating:** {attraction.rating}/5 ({attraction.reviews_count} reviews)\n"
        
        response += f"\n📖 **About:**\n{attraction.description}\n\n"
        
        if hasattr(attraction, 'highlights') and attraction.highlights:
            response += f"✨ **Highlights:**\n"
            for highlight in attraction.highlights[:5]:
                response += f"• {highlight}\n"
            response += "\n"
        
        if hasattr(attraction, 'best_time') and attraction.best_time:
            response += f"🕐 **Best Time:** {attraction.best_time}\n"
        
        if hasattr(attraction, 'accessibility_features') and attraction.accessibility_features:
            response += f"♿ **Accessibility:** {', '.join(attraction.accessibility_features)}\n"
        
        if hasattr(attraction, 'practical_tips') and attraction.practical_tips:
            response += f"\n💡 **Pro Tips:**\n"
            for tip in attraction.practical_tips[:3]:
                response += f"• {tip}\n"
        
        return response

    def _format_attractions_list(self, attractions: List, category: Optional[str] = None, district: Optional[str] = None) -> str:
        """Format a list of attractions with category-specific headers"""
        # ENHANCED: Category-specific headers
        category_headers = {
            'museum': ('🏛️', 'Museums in Istanbul', 'museum'),
            'park': ('🌳', 'Parks & Gardens in Istanbul', 'park'),
            'monument': ('🗿', 'Historical Monuments in Istanbul', 'monument'),
            'historic': ('🏛️', 'Historical Sites in Istanbul', 'historical site'),
            'religious': ('🕌', 'Religious Sites in Istanbul', 'religious site'),
            'palace': ('👑', 'Palaces in Istanbul', 'palace'),
            'market': ('🛍️', 'Markets & Bazaars in Istanbul', 'market'),
            'waterfront': ('⛵', 'Waterfront Attractions in Istanbul', 'waterfront'),
            'tower': ('🗼', 'Viewpoints & Towers in Istanbul', 'viewpoint'),
            'modern': ('🎨', 'Modern Attractions in Istanbul', 'modern attraction'),
            'family': ('👨‍👩‍👧', 'Family-Friendly Attractions in Istanbul', 'family attraction'),
            'romantic': ('💑', 'Romantic Spots in Istanbul', 'romantic spot')
        }
        
        # Build header
        if category and category in category_headers:
            emoji, title, singular = category_headers[category]
            if district:
                header = f"{emoji} **{title.replace('in Istanbul', f'in {district.title()}')}**\n\n"
            else:
                header = f"{emoji} **{title}**\n\n"
        elif district:
            header = f"🌟 **Top Attractions in {district.title()}**\n\n"
        else:
            header = "🌟 **Recommended Attractions in Istanbul**\n\n"
        
        response = header
        response += f"I found **{len(attractions)}** amazing {category_headers.get(category, ('', '', 'attraction'))[2]}{'s' if len(attractions) > 1 else ''} for you:\n\n"
        
        # Format each attraction
        for i, attraction in enumerate(attractions[:8], 1):  # Limit to 8 for readability
            response += f"{i}. **{attraction.name}**\n"
            response += f"   📍 {attraction.district}"
            
            # Show category if not filtering by category
            if not category or category != attraction.category.value.replace('_', ' ').lower():
                response += f" | {attraction.category.value.replace('_', ' ').title()}"
            
            response += "\n"
            
            # Build details line with available attributes
            details = []
            if hasattr(attraction, 'duration') and attraction.duration:
                details.append(f"⏰ {attraction.duration}")
            details.append(f"🎫 {attraction.entrance_fee.value.replace('_', ' ').title()}")
            if hasattr(attraction, 'rating'):
                details.append(f"⭐ {attraction.rating}/5")

            
            response += f"   {' | '.join(details)}\n"
            
            # Show description (truncated)
            if len(attraction.description) > 150:
                response += f"   {attraction.description[:147]}...\n\n"
            else:
                response += f"   {attraction.description}\n\n"
        
        # Add helpful footer
        if len(attractions) > 8:
            response += f"\n📋 *Showing top 8 of {len(attractions)} results. Ask for specific details or use filters to narrow down!*\n"
        
        response += "\n💡 **Tip:** Ask me about any specific attraction for detailed information!"
        
        return response

    def _extract_attraction_category(self, message: str) -> Optional[str]:
        """Extract attraction category from user message"""
        message_lower = message.lower()
        
        # ENHANCED: More comprehensive category mapping with museum, park, monument
        category_keywords = {
            'museum': ['museum', 'museums', 'gallery', 'galleries', 'exhibition', 'exhibitions'],
            'park': ['park', 'parks', 'garden', 'gardens', 'green space', 'playground', 'outdoor space'],
            'monument': ['monument', 'monuments', 'landmark', 'landmarks', 'memorial'],
            'historic': ['historic', 'historical', 'ancient', 'old', 'heritage', 'byzantine', 'ottoman empire'],
            'religious': ['mosque', 'mosques', 'church', 'churches', 'synagogue', 'religious', 'spiritual', 'temple'],
            'palace': ['palace', 'palaces', 'sultan', 'imperial'],
            'market': ['market', 'bazaar', 'shopping', 'shop', 'souvenir'],
            'waterfront': ['waterfront', 'bosphorus', 'sea', 'coast', 'pier', 'ferry', 'seaside'],
            'tower': ['tower', 'towers', 'view', 'panorama', 'viewpoint', 'observation'],
            'modern': ['modern', 'contemporary', 'new', 'current'],
            'family': ['family', 'kids', 'children', 'family-friendly'],
            'romantic': ['romantic', 'couple', 'couples', 'date', 'sunset']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        
        return None

    def _extract_district(self, message: str) -> Optional[str]:
        """Extract district/neighborhood from user message"""
        message_lower = message.lower()
        
        districts = [
            'sultanahmet', 'beyoglu', 'beyoğlu', 'galata', 'karaköy', 'karakoy',
            'taksim', 'besiktas', 'beşiktaş', 'ortakoy', 'ortaköy',
            'kadikoy', 'kadıköy', 'uskudar', 'üsküdar', 'eminonu', 'eminönü',
            'fatih', 'sisli', 'şişli', 'bakirkoy', 'bakırköy'
        ]
        
        for district in districts:
            if district in message_lower:
                return district
        
        return None

    def _format_detailed_museums_response(self, museums: List, gps_location: Optional[Tuple[float, float]], query: str) -> str:
        """Format detailed museum information with rich context"""
        from datetime import datetime
        
        # Build header based on query context
        query_lower = query.lower()
        if any(word in query_lower for word in ['art', 'modern']):
            header = "🎨 **Art Museums in Istanbul**\n\n"
        elif any(word in query_lower for word in ['historical', 'history']):
            header = "🏛️ **Historical Museums in Istanbul**\n\n"
        elif any(word in query_lower for word in ['free', 'no entrance']):
            header = "🎫 **Free Museums in Istanbul**\n\n"
        elif gps_location:
            header = "📍 **Museums Near You**\n\n"
        else:
            header = "🏛️ **Recommended Museums in Istanbul**\n\n"
        
        response = header
        current_time = datetime.now()
        current_day = current_time.weekday()
        
        # Limit to top 5 museums for readability
        museums_to_show = museums[:5]
        
        response += f"I found {len(museums_to_show)} excellent museum{'s' if len(museums_to_show) > 1 else ''} for you:\n\n"
        
        for i, museum in enumerate(museums_to_show, 1):
            response += "=" * 60 + "\n\n"
            response += f"**{i}. {museum.name}**\n"
            response += f"📍 **Location:** {museum.district} • {museum.address}\n"
            response += f"🎯 **Category:** {museum.category.value.replace('_', ' ').title()}\n"
            
            # Price and opening hours
            if museum.price_tl == 0:
                response += f"🎫 **Entry:** FREE ✨\n"
            else:
                response += f"🎫 **Entry:** {museum.price_tl} TL ({museum.price_category.value.title()})\n"
            
            # Current opening status
            hours_today = museum.opening_hours.get_today_hours(current_day)
            is_open = museum.opening_hours.is_open(current_day, current_time.time())
            status_emoji = "🟢" if is_open else "🔴"
            response += f"⏰ **Today's Hours:** {hours_today} {status_emoji}\n"
            
            # Distance if GPS available
            if gps_location and museum.coordinates:
                distance = self._calculate_haversine_distance(gps_location, museum.coordinates)
                response += f"🚶 **Distance:** {distance:.1f} km from you\n"
            
            # Highlights
            if museum.highlights:
                response += f"✨ **Must-See Highlights:**\n"
                for highlight in museum.highlights[:4]:
                    response += f"  • {highlight}\n"
                response += "\n"
            
            # Facilities and features
            features = []
            if museum.accessibility:
                features.append("♿ Wheelchair accessible")
            if museum.family_friendly:
                features.append("👨‍👩‍👧 Family-friendly")
            if museum.photography_allowed:
                features.append("📸 Photography allowed")
            if museum.guided_tours:
                features.append("🎧 Guided tours")
            if museum.audio_guide:
                features.append("🔊 Audio guide")
            if museum.cafe:
                features.append("☕ Café")
            if museum.gift_shop:
                features.append("🎁 Gift shop")
            if features:
                response += f"🏢 **Facilities:** {' | '.join(features)}\n\n"
            
            # Nearby attractions
            if museum.nearby_attractions:
                response += f"🗺️ **Nearby:** {', '.join(museum.nearby_attractions[:3])}\n"
            
            # Nearby restaurants
            if museum.nearby_restaurants:
                response += f"🍽️ **Dining:** {', '.join(museum.nearby_restaurants[:2])}\n"
            
            response += "\n"
            
        # Add practical tips at the end
        response += "=" * 60 + "\n\n"
        response += "💡 **Planning Your Visit:**\n"
        response += "• Museum Pass Istanbul (₺850) covers 12+ museums for 5 days\n"
        response += "• Start early (9 AM) to avoid crowds\n"
        response += "• Wear comfortable walking shoes\n"
        response += "• Keep Istanbulkart handy for transport\n"
        response += "• Plan indoor backup for weather\n"
        response += "• Book dinner reservations in advance\n"
        
        # Also check for multiple question marks or commas (indicating multiple sub-queries)
        multiple_questions = message.count('?') > 1
        multiple_clauses = message.count(',') >= 2
        
        # Query is complex if it has:
        # - 2+ complexity indicators, OR
        # - Multiple questions, OR  
        # - 3+ filter words AND 2+ clauses
        complexity_indicators = ['museum', 'exhibition', 'art', 'history', 'cultural', 'istanbul', 'visit', 'see', 'recommend']
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in message_lower)
        
        is_complex = (complexity_count >= 2 or
                      multiple_questions or
                      (category and district and len(attractions) > 2) or  # Category + district filter with multiple results
                      (complexity_count >= 1 and multiple_clauses)
                     )
        if is_complex:
            logger.info(f"🔍 Complex query detected (indicators: {complexity_count}, questions: {multiple_questions}, clauses: {multiple_clauses})")
        
        return response
    
    def _calculate_haversine_distance(self, coords_1: Tuple[float, float], coords_2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two sets of coordinates"""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1 = coords_1
        lat2, lon2 = coords_2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Earth's radius in kilometers
        return c * r

    def _extract_personalization_data(self, entities: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """
        Extract data from entities for personalization learning
        
        Args:
            entities: Extracted entities from user query
            intent: Detected intent
        
        Returns:
            Dict with personalization-relevant data
        """
        personalization_data = {}
        
        # Extract cuisine preferences
        if 'cuisine' in entities:
            personalization_data['cuisine'] = entities['cuisine'][0] if entities['cuisine'] else None
        
        # Extract price range preferences
        if 'price_range' in entities:
            personalization_data['price_range'] = entities['price_range'][0] if entities['price_range'] else None
        
        # Extract district preferences
        if 'district' in entities or 'neighborhood' in entities:
            district = (entities.get('district', []) + entities.get('neighborhood', []))
            personalization_data['district'] = district[0] if district else None
        
        # Map intent to activity/attraction types
        intent_to_activity = {
            'restaurant': {'activity_type': 'dining'},
            'attraction': {'activity_type': 'cultural', 'attraction_type': 'historical'},
            'museum': {'activity_type': 'cultural', 'attraction_type': 'museum'},
            'event': {'activity_type': 'entertainment'},
            'nightlife': {'activity_type': 'nightlife'},
            'shopping': {'activity_type': 'shopping'},
            'hidden_gem': {'activity_type': 'adventure'},
            'park': {'activity_type': 'relaxation', 'attraction_type': 'nature'}
        }
        
        if intent in intent_to_activity:
            personalization_data.update(intent_to_activity[intent])
        
        # Extract time of day context
        time_entities = entities.get('time', [])
        if time_entities:
            time_str = str(time_entities[0]).lower()
            if any(t in time_str for t in ['morning', 'breakfast', 'sabah']):
                personalization_data['time_of_day'] = 'morning'
            elif any(t in time_str for t in ['afternoon', 'lunch', 'öğle']):
                personalization_data['time_of_day'] = 'afternoon'
            elif any(t in time_str for t in ['evening', 'dinner', 'akşam']):
                personalization_data['time_of_day'] = 'evening'
            elif any(t in time_str for t in ['night', 'gece']):
                personalization_data['time_of_day'] = 'night'
        
        # Extract transportation preferences if relevant
        if intent == 'transportation':
            transport_keywords = {
                'metro': ['metro', 'subway'],
                'bus': ['bus', 'otobüs'],
                'tram': ['tram', 'tramvay'],
                'ferry': ['ferry', 'vapur', 'boat'],
                'taxi': ['taxi', 'taksi', 'uber']
            }
            for mode, keywords in transport_keywords.items():
                if any(kw in str(entities).lower() for kw in keywords):
                    personalization_data['transportation_mode'] = mode
                    break
        
        return personalization_data
    
    def submit_feedback(self, user_id: str, interaction_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit user feedback for an interaction
        
        Args:
            user_id: User identifier
            interaction_id: Interaction identifier from response
            feedback_data: Feedback data including satisfaction_score, was_helpful, etc.
        
        Returns:
            Dict with submission status and feedback summary
        """
        if not self.feedback_loop_system:
            return {
                'status': 'error',
                'message': 'Feedback system not available'
            }
        
        try:
            # Submit to feedback loop system
            result = self.feedback_loop_system.submit_feedback(
                user_id=user_id,
                interaction_id=interaction_id,
                feedback_data=feedback_data
            )
            
            # Update personalization if rating is provided
            if self.personalization_system and 'satisfaction_score' in feedback_data:
                try:
                    # Normalize satisfaction score to 0-1 range
                    normalized_rating = feedback_data['satisfaction_score'] / 5.0
                    
                    interaction_data = {
                        'type': feedback_data.get('intent', 'unknown'),
                        'item_id': interaction_id,
                        'item_data': feedback_data.get('item_data', {}),
                        'rating': normalized_rating,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.personalization_system.record_interaction(user_id, interaction_data)
                except Exception as e:
                    logger.warning(f"Failed to update personalization with feedback: {e}")
            
            return {
                'status': 'success',
                'message': 'Feedback submitted successfully',
                'feedback_id': result.get('feedback_id'),
                'aggregate_metrics': self.feedback_loop_system.get_aggregate_metrics()
            }
        
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return {
                'status': 'error',
                'message': f'Failed to submit feedback: {str(e)}'
            }
    
    def get_feedback_metrics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get feedback metrics for dashboard/monitoring
        
        Args:
            filters: Optional filters (intent, feature, time_range, etc.)
        
        Returns:
            Dict with feedback metrics and insights
        """
        if not self.feedback_loop_system:
            return {
                'status': 'error',
                'message': 'Feedback system not available'
            }
        
        try:
            metrics = {
                'aggregate': self.feedback_loop_system.get_aggregate_metrics(),
                'by_intent': self.feedback_loop_system.get_feedback_by_intent(),
                'by_feature': self.feedback_loop_system.get_feedback_by_feature(),
                'improvement_status': self.feedback_loop_system.get_improvement_status()
            }
            
            # Apply filters if provided
            if filters:
                if 'intent' in filters:
                    metrics['filtered_intent'] = metrics['by_intent'].get(filters['intent'], {})
                if 'feature' in filters:
                    metrics['filtered_feature'] = metrics['by_feature'].get(filters['feature'], {})
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting feedback metrics: {e}")
            return {
                'status': 'error',
                'message': f'Failed to get metrics: {str(e)}'
            }
    
    def get_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get personalization insights for a user
        
        Args:
            user_id: User identifier
        
        Returns:
            Dict with user preferences, similar users, and personalization status
        """
        if not self.personalization_system:
            return {
                'status': 'error',
                'message': 'Personalization system not available'
            }
        
        try:
            insights = self.personalization_system.get_user_insights(user_id)
            
            # Add A/B test assignments
            insights['ab_tests'] = {
                'response_format': self.personalization_system.get_response_format(user_id),
                'recommendation_algo': self.personalization_system.ab_testing.user_assignments.get(
                    user_id, {}
                ).get('recommendation_algo', 'hybrid')
            }
            
            return {
                'status': 'success',
                'insights': insights
            }
        
        except Exception as e:
            logger.error(f"Error getting personalization insights: {e}")
            return {
                'status': 'error',
                'message': f'Failed to get insights: {str(e)}'
            }
    
    def apply_personalization(self, user_id: str, candidates: List[Dict[str, Any]], 
                            top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Apply personalization to recommendation candidates
        
        Args:
            user_id: User identifier
            candidates: List of candidate recommendations
            top_n: Number of recommendations to return
        
        Returns:
            Personalized list of recommendations
        """
        if not self.personalization_system or not candidates:
            return candidates[:top_n]
        
        try:
            personalized = self.personalization_system.personalize_recommendations(
                user_id=user_id,
                candidates=candidates,
                top_n=top_n
            )
            
            logger.info(f"Applied personalization for user {user_id}: {len(candidates)} -> {len(personalized)} items")
            return personalized
        
        except Exception as e:
            logger.warning(f"Failed to apply personalization: {e}")
            return candidates[:top_n]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for monitoring
        
        Returns:
            Dict with system metrics, feedback data, and improvement status
        """
        dashboard = {
            'system_status': {
                'ready': self.system_ready,
                'components': {
                    'personalization': self.personalization_system is not None,
                    'feedback_loop': self.feedback_loop_system is not None,
                    'neural_processor': self.neural_processor is not None,
                    'ml_handlers': ML_ENHANCED_HANDLERS_AVAILABLE
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add feedback metrics
        if self.feedback_loop_system:
            try:
                dashboard['feedback'] = {
                    'aggregate': self.feedback_loop_system.get_aggregate_metrics(),
                    'by_intent': self.feedback_loop_system.get_feedback_by_intent(),
                    'by_feature': self.feedback_loop_system.get_feedback_by_feature(),
                    'improvement_status': self.feedback_loop_system.get_improvement_status()
                }
            except Exception as e:
                logger.error(f"Error getting feedback data for dashboard: {e}")
                dashboard['feedback'] = {'error': str(e)}
        
        # Add A/B test results
        if self.personalization_system:
            try:
                dashboard['ab_tests'] = {
                    'response_format': self.personalization_system.ab_testing.get_experiment_results('response_format'),
                    'recommendation_algo': self.personalization_system.ab_testing.get_experiment_results('recommendation_algo')
                }
            except Exception as e:
                logger.error(f"Error getting A/B test data for dashboard: {e}")
                dashboard['ab_tests'] = {'error': str(e)}
        
        return dashboard
    
    def health_check(self) -> Dict[str, Any]:
        """
        System health check endpoint for monitoring and uptime tracking
        
        Returns:
            Dict with health status, component availability, and system metrics
            
        Example:
            >>> ai = IstanbulDailyTalkAI()
            >>> health = ai.health_check()
            >>> print(f"Status: {health['status']}, Score: {health['health_score']}")
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_check': 'passed',
            'system_ready': self.system_ready
        }
        
        # Core Components Check
        core_components = {
            'entity_recognizer': self.entity_recognizer is not None,
            'response_generator': self.response_generator is not None,
            'user_manager': self.user_manager is not None,
        }
        
        # ML/AI Components Check
        ml_components = {
            'neural_processor': self.neural_processor is not None,
            'ml_context_builder': self.ml_context_builder is not None,
            'personalization_system': self.personalization_system is not None,
            'feedback_loop_system': self.feedback_loop_system is not None,
        }
        
        # Service Components Check
        service_components = {
            'hidden_gems_handler': self.hidden_gems_handler is not None,
            'price_filter_service': self.price_filter_service is not None,
            'conversation_handler': self.conversation_handler is not None,
            'events_service': self.events_service is not None,
            'weather_recommendations': self.weather_recommendations is not None,
            'location_detector': self.location_detector is not None,
            'transport_processor': self.transport_processor is not None,
            'weather_client': self.weather_client is not None,
        }
        
        # ML Handlers Check
        ml_handlers = {
            'ml_event_handler': hasattr(self, 'ml_event_handler') and self.ml_event_handler is not None,
            'ml_hidden_gems_handler': hasattr(self, 'ml_hidden_gems_handler') and self.ml_hidden_gems_handler is not None,
            'ml_weather_handler': hasattr(self, 'ml_weather_handler') and self.ml_weather_handler is not None,
            'ml_route_planning_handler': hasattr(self, 'ml_route_planning_handler') and self.ml_route_planning_handler is not None,
            'ml_neighborhood_handler': hasattr(self, 'ml_neighborhood_handler') and self.ml_neighborhood_handler is not None,
        }
        
        # Calculate component health scores
        core_health = sum(core_components.values()) / len(core_components) * 100 if core_components else 0
        ml_health = sum(ml_components.values()) / len(ml_components) * 100 if ml_components else 0
        service_health = sum(service_components.values()) / len(service_components) * 100 if service_components else 0
        ml_handlers_health = sum(ml_handlers.values()) / len(ml_handlers) * 100 if ml_handlers else 0
        
        # Overall health score (weighted average)
        overall_health = (
            core_health * 0.40 +      # Core components are critical (40%)
            ml_health * 0.25 +         # ML components are important (25%)
            service_health * 0.25 +    # Services are important (25%)
            ml_handlers_health * 0.10  # ML handlers are nice-to-have (10%)
        )
        
        # Determine overall status
        if overall_health >= 90:
            health_status['status'] = 'healthy'
        elif overall_health >= 70:
            health_status['status'] = 'degraded'
        else:
            health_status['status'] = 'unhealthy'
        
        health_status['health_score'] = round(overall_health, 2)
        health_status['components'] = {
            'core': {
                'health': round(core_health, 2),
                'details': core_components
            },
            'ml': {
                'health': round(ml_health, 2),
                'details': ml_components
            },
            'services': {
                'health': round(service_health, 2),
                'details': service_components
            },
            'ml_handlers': {
                'health': round(ml_handlers_health, 2),
                'details': ml_handlers
            }
        }
        
        # Add feedback system health if available
        if self.feedback_loop_system:
            try:
                feedback_health = self.feedback_loop_system.get_improvement_status()
                health_status['feedback_system'] = {
                    'health_score': feedback_health.get('health_score', 0),
                    'status': feedback_health.get('status', 'unknown'),
                    'avg_satisfaction': feedback_health.get('avg_satisfaction', 0),
                    'total_ratings': feedback_health.get('total_ratings', 0)
                }
            except Exception as e:
                health_status['feedback_system'] = {'error': str(e)}
        
        # Add system metrics
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            health_status['system_metrics'] = {
                'memory_usage_mb': round(process.memory_info().rss / 1024 / 1024, 2),
                'cpu_percent': process.cpu_percent(interval=0.1),
            }
        except Exception as e:
            logger.debug(f"Could not get system metrics: {e}")
        
        # Add recommendations if system is degraded
        if health_status['status'] in ['degraded', 'unhealthy']:
            recommendations = []
            if core_health < 100:
                recommendations.append("Core components missing - check initialization errors")
            if ml_health < 50:
                recommendations.append("ML components unavailable - advanced features may not work")
            if service_health < 70:
                recommendations.append("Some services unavailable - functionality may be limited")
            health_status['recommendations'] = recommendations
        
        return health_status
