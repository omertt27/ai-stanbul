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


class IstanbulDailyTalkAI:
    """🚀 ENHANCED Istanbul Daily Talk AI System with Deep Learning
    
    Advanced conversational AI for Istanbul visitors and locals with comprehensive
    recommendations, cultural context, and personalized experiences.
    """
    
    def __init__(self):
        """Initialize the Istanbul AI system"""
        logger.info("🚀 Initializing Istanbul Daily Talk AI System...")
        
        # Initialize core components
        self.entity_recognizer = IstanbulEntityRecognizer()
        self.response_generator = ResponseGenerator()
        self.user_manager = UserManager()
        
        # Initialize hidden gems handler
        if HIDDEN_GEMS_HANDLER_AVAILABLE:
            try:
                self.hidden_gems_handler = HiddenGemsHandler()
                logger.info("💎 Hidden Gems Handler initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize hidden gems handler: {e}")
                self.hidden_gems_handler = None
        else:
            self.hidden_gems_handler = None
        
        # Initialize price filter service
        if PRICE_FILTER_AVAILABLE:
            try:
                self.price_filter_service = PriceFilterService()
                logger.info("💰 Price Filter Service initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize price filter: {e}")
                self.price_filter_service = None
        else:
            self.price_filter_service = None
        
        # Initialize conversation handler
        try:
            from backend.services.conversation_handler import get_conversation_handler
            self.conversation_handler = get_conversation_handler()
            logger.info("💬 Conversation Handler initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize conversation handler: {e}")
            self.conversation_handler = None
        
        # Initialize events service
        if EVENTS_SERVICE_AVAILABLE:
            try:
                self.events_service = get_events_service()
                logger.info("🎭 Events Service initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize events service: {e}")
                self.events_service = None
        else:
            self.events_service = None
        
        # Initialize weather recommendations service
        if WEATHER_RECOMMENDATIONS_AVAILABLE:
            try:
                self.weather_recommendations = get_weather_recommendations_service()
                logger.info("🌤️ Weather Recommendations Service initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize weather recommendations: {e}")
                self.weather_recommendations = None
        else:
            self.weather_recommendations = None
        
        # Initialize location detector if available
        try:
            from .services.intelligent_location_detector import IntelligentLocationDetector
            self.location_detector = IntelligentLocationDetector()
            logger.info("📍 Intelligent Location Detector loaded successfully!")
        except ImportError as e:
            logger.warning(f"Location detection not available: {e}")
            self.location_detector = None
        
        # Initialize advanced transportation system
        if ADVANCED_TRANSPORT_AVAILABLE:
            try:
                self.transport_processor = TransportationQueryProcessor()
                self.ml_transport_system = create_ml_enhanced_transportation_system()
                logger.info("🚇 Advanced transportation system with IBB API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize advanced transportation: {e}")
                self.transport_processor = None
                self.ml_transport_system = None
        else:
            self.transport_processor = None
            self.ml_transport_system = None
        
        # Initialize Transfer Instructions & Map Visualization Integration
        if TRANSFER_MAP_INTEGRATION_AVAILABLE:
            try:
                self.transportation_chat = TransportationChatIntegration()
                logger.info("🗺️ Transfer Instructions & Map Visualization integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize transfer instructions integration: {e}")
                self.transportation_chat = None
        else:
            self.transportation_chat = None

        # Initialize ML-Enhanced Daily Talks Bridge
        if ML_DAILY_TALKS_AVAILABLE:
            try:
                self.daily_talks_bridge = MLEnhancedDailyTalksBridge()
                logger.info("🤖 ML-Enhanced Daily Talks Bridge initialized")
            except Exception as e:
                logger.error(f"Failed to initialize daily talks bridge: {e}")
                self.daily_talks_bridge = None
        else:
            self.daily_talks_bridge = None

        # Initialize Lightweight Neural Query Enhancement System (Budget-Friendly!)
        if NEURAL_QUERY_ENHANCEMENT_AVAILABLE:
            try:
                self.neural_processor = get_lightweight_neural_processor()
                logger.info("🧠 Lightweight Neural Query Enhancement System initialized (CPU-optimized, <100ms latency)")
            except Exception as e:
                logger.error(f"Failed to initialize neural query processor: {e}")
                self.neural_processor = None
        else:
            self.neural_processor = None

        # Initialize museum system with location integration
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from museum_response_generator import MuseumResponseGenerator
            from google_maps_hours_checker import GoogleMapsHoursChecker
            from updated_museum_database import UpdatedIstanbulMuseumDatabase
            
            self.museum_generator = MuseumResponseGenerator()
            self.hours_checker = GoogleMapsHoursChecker()
            self.museum_db = UpdatedIstanbulMuseumDatabase()
            logger.info("🏛️ Museum system with location integration loaded successfully!")
        except ImportError as e:
            logger.warning(f"Museum system not available: {e}")
            self.museum_generator = None
            self.hours_checker = None
            self.museum_db = None

        # Initialize advanced museum system (ML-powered with GPS, filtering, typo correction)
        try:
            from museum_advising_system import IstanbulMuseumSystem
            self.advanced_museum_system = IstanbulMuseumSystem()
            logger.info("🎨 Advanced Museum System (ML-powered) loaded successfully!")
        except ImportError as e:
            logger.warning(f"Advanced Museum System not available: {e}")
            self.advanced_museum_system = None

        # Initialize advanced attractions system (78+ curated attractions with category/district filtering)
        try:
            from istanbul_attractions_system import IstanbulAttractionsSystem
            self.advanced_attractions_system = IstanbulAttractionsSystem()
            logger.info("🌟 Advanced Attractions System loaded successfully!")
        except ImportError as e:
            logger.warning(f"Advanced Attractions System not available: {e}")
            self.advanced_attractions_system = None

        # Initialize Multi-Intent Query Handler for complex multi-part queries
        if MULTI_INTENT_HANDLER_AVAILABLE:
            try:
                self.multi_intent_handler = MultiIntentQueryHandler()
                logger.info("🎯 Multi-Intent Query Handler initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize Multi-Intent Query Handler: {e}")
                self.multi_intent_handler = None
        else:
            self.multi_intent_handler = None

        # Initialize enhanced museum route planner
        try:
            from enhanced_museum_route_planner import EnhancedMuseumRoutePlanner
            self.museum_route_planner = EnhancedMuseumRoutePlanner()
            logger.info("🗺️ Enhanced Museum Route Planner loaded successfully!")
        except ImportError as e:
            logger.warning(f"Enhanced Museum Route Planner not available: {e}")
            self.museum_route_planner = None

        # Initialize enhanced GPS route planner with fallback location detection
        try:
            from enhanced_gps_route_planner import EnhancedGPSRoutePlanner
            self.gps_route_planner = EnhancedGPSRoutePlanner()
            logger.info("🗺️ Enhanced GPS Route Planner with fallback location detection loaded successfully!")
        except ImportError as e:
            logger.warning(f"Enhanced GPS Route Planner not available: {e}")
            self.gps_route_planner = None
        
        # Initialize Enhanced Route Planner V2 (advanced multi-feature planner)
        try:
            from enhanced_route_planner_v2 import EnhancedRoutePlannerV2
            self.advanced_route_planner = EnhancedRoutePlannerV2()
            logger.info("🧭 Enhanced Route Planner V2 loaded successfully!")
        except ImportError as e:
            logger.warning(f"Enhanced Route Planner V2 not available: {e}")
            self.advanced_route_planner = None

        # Initialize Weather System
        try:
            from backend.api_clients.enhanced_weather import EnhancedWeatherClient
            self.weather_client = EnhancedWeatherClient()
            logger.info("🌤️ Enhanced Weather System loaded successfully!")
        except ImportError as e:
            logger.warning(f"Weather System not available: {e}")
            self.weather_client = None

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
        """Process user message and generate response
        
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
            
            # Step 3: Lightweight Neural Query Enhancement - Fast & budget-friendly!
            neural_insights = None
            if NEURAL_QUERY_ENHANCEMENT_AVAILABLE and self.neural_processor:
                try:
                    # Skip neural processing if already in an async context (avoid deadlock)
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, skip neural processing to avoid deadlock
                        logger.debug("Skipping neural processing (already in async context)")
                        neural_insights = None
                    except RuntimeError:
                        # No event loop running, safe to create one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            neural_result = loop.run_until_complete(
                                self.neural_processor.process_query(
                                    query=message,
                                    context={
                                        'session_id': session_id,
                                        'user_profile': {
                                            'interests': getattr(user_profile, 'interests', []),
                                            'user_type': getattr(user_profile, 'user_type', 'first_time_visitor'),
                                            'language_preference': getattr(user_profile, 'language_preference', 'english')
                                        }
                                    }
                                )
                                    )
                            
                            # Convert LightweightNeuralInsights to dict format
                            neural_insights = {
                                'entities': {entity['type']: [entity['text']] for entity in neural_result.entities},
                                'intent': {
                                    'primary': neural_result.intent,
                                    'confidence': neural_result.intent_confidence
                                },
                                'sentiment': neural_result.sentiment,
                                'keywords': neural_result.keywords,
                                'complexity': neural_result.query_complexity,
                                'location_context': neural_result.location_context,
                                'temporal_context': neural_result.temporal_context
                            }
                            
                            logger.info(f"✨ Lightweight neural processing complete ({neural_result.processing_time_ms:.1f}ms) - "
                                       f"Intent: {neural_result.intent}, Confidence: {neural_result.intent_confidence:.2f}")
                        finally:
                            loop.close()
                except Exception as e:
                    logger.warning(f"Neural processing failed, continuing with standard processing: {e}")
                    neural_insights = None
            
            # Check if this is a daily talk query (casual conversation, greetings, weather, etc.)
            if self._is_daily_talk_query(message):
                return self._handle_daily_talk_query(message, user_id, session_id, user_profile, context, neural_insights)
            
            # Extract entities from message (enhanced with neural insights if available)
            entities = self.entity_recognizer.extract_entities(message)
            
            # Merge neural entities with traditional entities
            if neural_insights and 'entities' in neural_insights:
                neural_entities = neural_insights['entities']
                for entity_type, entity_values in neural_entities.items():
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].extend(entity_values)
            
            # Check for multi-intent queries using MultiIntentQueryHandler
            # OPTIMIZED: Only invoke for truly complex queries with multiple components
            multi_intent_result = None
            if self.multi_intent_handler and self._is_complex_multi_intent_query(message):
                try:
                    multi_intent_context = {
                        'user_profile': {
                            'interests': getattr(user_profile, 'interests', []),
                            'user_type': getattr(user_profile, 'user_type', 'first_time_visitor'),
                            'language_preference': getattr(user_profile, 'language_preference', 'english')
                        },
                        'conversation_history': context.interactions[-3:] if hasattr(context, 'interactions') else [],
                        'entities': entities
                    }
                    multi_intent_result = self.multi_intent_handler.analyze_query(message, multi_intent_context)
                    
                    # If complex multi-intent query detected, handle it specially
                    if multi_intent_result and multi_intent_result.query_complexity > 0.7:  # Raised threshold from 0.6 to 0.7
                        logger.info(f"🎯 Multi-intent query detected (complexity: {multi_intent_result.query_complexity:.2f})")
                        if hasattr(multi_intent_result, 'response_text') and multi_intent_result.response_text:
                            context.add_interaction(message, multi_intent_result.response_text, 'multi_intent')
                            return multi_intent_result.response_text
                except Exception as e:
                    logger.warning(f"Multi-intent processing failed, continuing with standard processing: {e}")
                    multi_intent_result = None
            
            # Classify intent with context (use neural intent if available and confident)
            # Lower confidence threshold for museum/attraction queries since ML is better at detecting them
            neural_confidence_threshold = 0.7
            if neural_insights and neural_insights.get('intent'):
                neural_intent = neural_insights['intent']['primary']
                neural_confidence = neural_insights.get('intent', {}).get('confidence', 0)
                
                # Use lower threshold for attraction-related intents (ML is good at these)
                if neural_intent in ['attraction', 'sightseeing', 'museum', 'landmark']:
                    neural_confidence_threshold = 0.5  # More lenient for places/attractions
                
                if neural_confidence > neural_confidence_threshold:
                    intent = neural_intent
                    logger.info(f"Using neural intent: {intent} (confidence: {neural_confidence:.2f})")
                else:
                    intent = self._classify_intent_with_context(message, entities, context)
                    logger.info(f"Using traditional intent: {intent} (neural confidence too low: {neural_confidence:.2f})")
            else:
                intent = self._classify_intent_with_context(message, entities, context)
                logger.info(f"Using traditional intent: {intent}")
            
            # Generate contextual response (enhanced with neural insights)
            response_result = self._generate_contextual_response(
                message, intent, entities, user_profile, context, neural_insights, return_structured=return_structured
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
            context.add_interaction(message, response_text, intent)
            
            # Return structured or string response based on parameter
            if return_structured:
                return {
                    'response': response_text,
                    'map_data': map_data,
                    'intent': intent,
                    'entities': entities
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
        
        # CRITICAL: Exclude museum/attraction queries FIRST (EXPANDED LIST)
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
        """Handle daily talk queries through ML-enhanced bridge, enhanced with neural insights"""
        
        if not ML_DAILY_TALKS_AVAILABLE or not self.daily_talks_bridge:
            # Fallback to basic daily talk response
            return self._generate_basic_daily_talk_response(message, user_profile, context)
        
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
                    # If loop is already running, create a task but don't await it directly
                    # Instead, use the comprehensive daily talks system synchronously
                    bridge_result = asyncio.create_task(
                        self.daily_talks_bridge.process_daily_talk_request(
                            message, user_id, session_id, context_data
                        )
                    )
                    # For now, fall back to basic response to avoid blocking
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
            
            return response
            
        except Exception as e:
            logger.error(f"Error in ML daily talks bridge: {e}")
            # Fallback to basic response
            return self._generate_basic_daily_talk_response(message, user_profile, context)
    
    def _generate_basic_daily_talk_response(self, message: str, user_profile: UserProfile, 
                                          context: ConversationContext) -> str:
        """Generate basic daily talk response as fallback"""
        
        message_lower = message.lower()
        current_hour = datetime.now().hour
        
        # Greeting responses
        if any(greeting in message_lower for greeting in ['hi', 'hello', 'hey', 'merhaba']):
            if current_hour < 12:
                return "🌅 Good morning! What a beautiful day to explore Istanbul! How can I help you discover something amazing today?"
            elif current_hour < 17:
                return "☀️ Good afternoon! Perfect time to explore Istanbul! What would you like to discover today?"
            else:
                return "🌆 Good evening! Istanbul's evening magic awaits! How can I help you experience the city tonight?"
        
        # Enhanced weather responses
        if any(weather in message_lower for weather in ['weather', 'temperature', 'rain', 'sunny', 'cold', 'hot']):
            return self._generate_weather_aware_response(message, user_profile, context)
        
        # Thank you responses
        if any(thanks in message_lower for thanks in ['thank', 'thanks']):
            return "🙏 You're very welcome! I'm here to help you discover the best of Istanbul. Anything else you'd like to know?"
        
        # Goodbye responses
        if any(bye in message_lower for bye in ['bye', 'goodbye', 'see you']):
            return "👋 Güle güle! (Goodbye in Turkish) Have a wonderful time in Istanbul! Feel free to ask me anything anytime!"
        
        # Default casual response
        return "😊 I'm your Istanbul AI guide, always ready to help! Whether you want restaurant recommendations, cultural insights, or help getting around the city, just let me know. What interests you most about Istanbul?"
    
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
        """Classify user intent with contextual awareness"""
        
        message_lower = message.lower()
        
        # Restaurant/food intent
        food_keywords = ['eat', 'food', 'restaurant', 'lunch', 'dinner', 'breakfast', 'hungry', 'cuisine']
        if any(keyword in message_lower for keyword in food_keywords) or entities.get('cuisines'):
            return 'restaurant'
        
        # Attraction/sightseeing intent (ENHANCED with more museum/attraction keywords)
        attraction_keywords = [
            # General attraction words
            'visit', 'see', 'attraction', 'attractions', 'tour', 'sightseeing', 'tourist', 'landmark', 'landmarks',
            # Museum keywords
            'museum', 'museums', 'gallery', 'galleries', 'exhibition', 'exhibitions', 'art museum', 'art museums',
            'historical museum', 'history museum', 'archaeological', 'archaeology',
            # Specific place types
            'mosque', 'mosques', 'palace', 'palaces', 'church', 'churches', 'synagogue',
            'monument', 'monuments', 'memorial', 'historical site', 'historical sites',
            # Descriptive words for museums
            'historical', 'ancient', 'cultural site', 'heritage', 'artifact', 'artifacts',
            # Action words for attractions
            'explore', 'discover', 'show me', 'what to see', 'worth seeing', 'must see', 'should i see',
            # Specific queries
            'places to visit', 'places to see', 'what can i visit', 'what can i see',
            'best attractions', 'top attractions', 'famous places', 'popular places'
        ]
        
        # Also check if message is asking about specific types
        museum_specific = any(word in message_lower for word in ['museum', 'museums', 'gallery', 'galleries', 'exhibition'])
        attraction_specific = any(word in message_lower for word in ['attraction', 'attractions', 'landmark', 'palace', 'mosque'])
        
        if museum_specific or attraction_specific or any(keyword in message_lower for keyword in attraction_keywords) or entities.get('landmarks'):
            return 'attraction'
        
        # Transportation intent
        transport_keywords = ['transport', 'metro', 'bus', 'taxi', 'ferry', 'how to get', 'travel']
        if any(keyword in message_lower for keyword in transport_keywords) or entities.get('transportation'):
            return 'transportation'
        
        # Neighborhood/area intent
        area_keywords = ['neighborhood', 'area', 'district', 'where to stay', 'which area']
        if any(keyword in message_lower for keyword in area_keywords) or entities.get('neighborhoods'):
            return 'neighborhood'
        
        # Shopping intent
        shopping_keywords = ['shop', 'shopping', 'buy', 'bazaar', 'market', 'souvenir']
        if any(keyword in message_lower for keyword in shopping_keywords):
            return 'shopping'
        
        # Events/activities intent
        event_keywords = ['event', 'activity', 'entertainment', 'nightlife', 'what to do']
        if any(keyword in message_lower for keyword in event_keywords):
            return 'events'
        
        # Weather intent
        weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy', 'hot', 'cold', 
                           'what\'s the weather', 'how\'s the weather', 'weather today', 'weather tomorrow',
                           'will it rain', 'is it sunny', 'degrees', 'celsius', 'fahrenheit', 'humidity',
                           'wind', 'precipitation', 'weather conditions', 'climate']
        if any(keyword in message_lower for keyword in weather_keywords):
            return 'weather'
        
        # Airport transport intent
        airport_keywords = [
            'airport', 'ist', 'saw', 'atatürk', 'ataturk', 'istanbul airport', 'sabiha gökçen', 
            'sabiha gokcen', 'new airport', 'airport transfer', 'airport transport', 'from airport',
            'to airport', 'airport shuttle', 'airport bus', 'airport metro', 'flight', 'departure',
            'arrival', 'terminal', 'baggage', 'customs', 'immigration'
        ]
        if any(keyword in message_lower for keyword in airport_keywords):
            return 'airport_transport'
        
        # Hidden gems intent
        hidden_gems_keywords = [
            'hidden', 'secret', 'local', 'authentic', 'off-beaten', 'off the beaten path', 'unknown', 
            'undiscovered', 'gems', 'hidden gems', 'secret spots', 'local favorites', 'insider',
            'less touristy', 'not touristy', 'avoid crowds', 'unique places', 'special places',
            'locals know', 'local secrets', 'hidden treasures', 'underground', 'alternative',
            'unconventional', 'non-touristy', 'lesser known', 'hidden places'
        ]
        if any(keyword in message_lower for keyword in hidden_gems_keywords):
            return 'hidden_gems'
        
        # Route planning intent
        route_keywords = ['route', 'itinerary', 'plan', 'schedule', 'day trip']
        if any(keyword in message_lower for keyword in route_keywords):
            return 'route_planning'
        
        # GPS-based route planning intent (more specific)
        gps_route_keywords = ['directions', 'navigation', 'how to get', 'from', 'to', 'nearest', 'distance', 'walking route', 'driving route', 'public transport route']
        location_indicators = ['from', 'to', 'near', 'closest', 'nearby', 'distance']
        if (any(keyword in message_lower for keyword in gps_route_keywords) or 
            any(indicator in message_lower for indicator in location_indicators) and any(rk in message_lower for rk in ['route', 'get', 'go', 'directions'])):
            return 'gps_route_planning'
        
        # Museum route planning intent (more specific)
        museum_route_keywords = ['museum route', 'museum tour', 'museum plan', 'museum itinerary', 'museums near', 'museum walk']
        if any(keyword in message_lower for keyword in museum_route_keywords) or \
           ('museum' in message_lower and any(rk in message_lower for rk in ['route', 'plan', 'tour', 'visit'])):
            return 'museum_route_planning'
        
        # Greeting/general intent
        greeting_keywords = ['hello', 'hi', 'merhaba', 'help', 'start']
        if any(keyword in message_lower for keyword in greeting_keywords):
            return 'greeting'
        
        return 'general'
    
    def _generate_contextual_response(self, message: str, intent: str, entities: Dict,
                                    user_profile: UserProfile, context: ConversationContext, 
                                    neural_insights: Optional[Dict] = None, return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate contextual response based on intent and entities, enhanced with neural insights
        
        Args:
            return_structured: If True, return dict with response and map_data; if False, return string
        """
        
        current_time = datetime.now()
        
        # Check for conversational queries first (thanks, help, planning)
        if self.conversation_handler:
            try:
                if self.conversation_handler.is_conversational_query(message):
                    conv_response = self.conversation_handler.handle_conversation(message)
                    if conv_response:
                        if return_structured:
                            return {
                                'response': conv_response,
                                'map_data': {},
                                'conversation_type': 'conversational'
                            }
                        return conv_response
            except Exception as e:
                logger.error(f"Conversation handler error: {e}")
        
        # Check for budget-related queries first using price filter service
        if self.price_filter_service:
            budget_level = self.price_filter_service.detect_budget_query(message)
            
            # Handle explicit free/budget attraction queries
            if budget_level == 'free' and intent in ['attraction', 'activity', 'things_to_do']:
                response = self.price_filter_service.format_free_attractions_response()
                if return_structured:
                    return {
                        'response': response,
                        'map_data': {},
                        'budget_level': 'free'
                    }
                return response
            
            # Handle budget restaurant queries
            if budget_level in ['free', 'budget'] and intent == 'restaurant':
                response = self.price_filter_service.format_budget_eats_response()
                if return_structured:
                    return {
                        'response': response,
                        'map_data': {},
                        'budget_level': budget_level
                    }
                return response
        
        # Check for hidden gems query using hidden gems handler
        if self.hidden_gems_handler and self.hidden_gems_handler.detect_hidden_gems_query(message):
            # Override intent to hidden_gems if detected
            intent = 'hidden_gems'
        
        # Use response generator for comprehensive responses
        if intent == 'attraction':
            # Check if this is a museum query - use advanced museum system if available
            message_lower = message.lower()
            museum_keywords = ['museum', 'museums', 'gallery', 'exhibition', 'art', 'historical sites', 'cultural sites']
            attractions_keywords = ['attraction', 'attractions', 'place', 'places', 'landmark', 'landmarks', 'sight', 'sights', 'visit', 'see', 'tower', 'palace', 'mosque', 'bazaar']
            
            # Route to advanced museum system if museum query and advanced system available
            if any(keyword in message_lower for keyword in museum_keywords):
                if self.advanced_museum_system:
                    response = self._generate_advanced_museum_response(message, entities, user_profile, context)
                    # Add weather context to response
                    if isinstance(response, str):
                        response = self._add_weather_context_to_attractions(response)
                    return response
                elif self.museum_generator:
                    response = self._generate_location_aware_museum_response(message, entities, user_profile, context)
                    if isinstance(response, str):
                        response = self._add_weather_context_to_attractions(response)
                    return response
            
            # Route to advanced attractions system if attraction query and advanced system available
            elif any(keyword in message_lower for keyword in attractions_keywords):
                if self.advanced_attractions_system:
                    response = self._generate_advanced_attractions_response(message, entities, user_profile, context)
                    if isinstance(response, str):
                        response = self._add_weather_context_to_attractions(response)
                    return response
            
            # Fallback to basic response generator with weather context
            response = self.response_generator.generate_comprehensive_recommendation(
                intent, entities, user_profile, context, return_structured=return_structured
            )
            if isinstance(response, str):
                response = self._add_weather_context_to_attractions(response)
            return response
        elif intent in ['restaurant', 'neighborhood']:
            return self.response_generator.generate_comprehensive_recommendation(
                intent, entities, user_profile, context, return_structured=return_structured
            )
        
        # Handle specific intents
        elif intent == 'transportation':
            return self._generate_transportation_response(message, entities, user_profile, context, return_structured=return_structured)
        
        elif intent == 'shopping':
            return self._generate_shopping_response(entities, user_profile, context)
        
        elif intent == 'events':
            return self._generate_events_response(entities, user_profile, context, current_time)
        
        elif intent == 'weather':
            return self._generate_weather_response(message, entities, user_profile, context)
        
        elif intent == 'airport_transport':
            return self.response_generator.generate_comprehensive_recommendation(
                intent, entities, user_profile, context, return_structured=return_structured
            )
        
        elif intent == 'hidden_gems':
            # Use specialized hidden gems handler
            if self.hidden_gems_handler:
                try:
                    # Extract location from entities if available
                    location = None
                    if 'location' in entities and entities['location']:
                        location = entities['location'][0] if isinstance(entities['location'], list) else entities['location']
                    
                    # Get hidden gems
                    gems = self.hidden_gems_handler.get_hidden_gems(
                        location=location,
                        limit=5
                    )
                    
                    # Format response
                    response = self.hidden_gems_handler.format_hidden_gem_response(gems, location)
                    
                    if return_structured:
                        return {
                            'response': response,
                            'map_data': {},
                            'gems': gems
                        }
                    return response
                    
                except Exception as e:
                    logger.error(f"Hidden gems handler error: {e}")
            
            # Fallback to response generator
            return self.response_generator.generate_comprehensive_recommendation(
                intent, entities, user_profile, context, return_structured=return_structured
            )
        
        elif intent == 'route_planning':
            return self._generate_route_planning_response(message, user_profile, context)
        
        elif intent == 'gps_route_planning':
            return self._generate_gps_route_response(message, entities, user_profile, context)
        
        elif intent == 'museum_route_planning':
            return self._generate_museum_route_response(message, entities, user_profile, context)
        
        elif intent == 'greeting':
            # Use conversation handler if available for better greetings
            if self.conversation_handler:
                try:
                    response = self.conversation_handler.handle_conversation(message)
                    if response:
                        return response
                except Exception as e:
                    logger.error(f"Conversation handler error: {e}")
            
            # Fallback to original greeting
            return self._generate_greeting_response(user_profile, context)
        
        else:
            # Multiple intents detected, enhance response
            detected_intents = self._detect_multiple_intents(message, entities)
            if len(detected_intents) > 1:
                base_response = self.response_generator.generate_comprehensive_recommendation(
                    detected_intents[0], entities, user_profile, context
                )
                return self.response_generator._enhance_multi_intent_response(
                    base_response, detected_intents, entities, user_profile
                )
            else:
                return self.response_generator._generate_fallback_response(context, user_profile)
    
    def _generate_transportation_response(self, message: str, entities: Dict, user_profile: UserProfile, 
                                        context: ConversationContext, return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate comprehensive transportation response with advanced AI and real-time data"""
        try:
            # Check if this is a specific route request
            route_indicators = ['from', 'to', 'how to get', 'directions', 'route from', 'route to']
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
                result = loop.run_until_complete(
                    self.transportation_chat.handle_transportation_query(
                        query=message,
                        user_location=user_location,
                        destination=destination,
                        user_context={
                            'has_luggage': False,
                            'time_sensitive': False,
                            'accessibility_needs': []
                        }
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
                                   context: ConversationContext) -> str:
        """Generate comprehensive shopping response"""
        
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
                                 context: ConversationContext, current_time: datetime) -> str:
        """Generate events and activities response with temporal parsing and live IKSV data"""
        
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
                                        context: ConversationContext) -> str:
        """Generate route planning response"""
        
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
• Buy **Museum Pass** (325 TL) for multiple sites
• Start early (9 AM) to avoid crowds
• Wear comfortable walking shoes
• Keep **Istanbulkart** handy for transport
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
            'shopping': ['shop', 'shopping', 'buy', 'bazaar']
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
            for museum in data['museums']:
                info += f"• {museum}\n"
            return info
        
        return None

    def _add_current_museum_hours(self, message: str) -> Optional[str]:
        """Add current museum hours information"""
        # This is a simplified version. The full implementation would use the hours_checker
        message_lower = message.lower()
        
        # Check if asking about specific museums
        if 'hagia sophia' in message_lower or 'ayasofya' in message_lower:
            return "⏰ **Hagia Sophia:** Open 09:00-19:00 (closed during prayer times)"
        elif 'topkapi' in message_lower:
            return "⏰ **Topkapi Palace:** Open 09:00-18:00 (closed Tuesdays)"
        elif 'istanbul modern' in message_lower:
            return "⏰ **Istanbul Modern:** Open 10:00-18:00 (Thursdays until 20:00, closed Mondays)"
        
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
        response += "• Many museums are closed Mondays - plan accordingly\n"
        response += "• Morning visits (9-11am) are less crowded\n"
        response += "• Guided tours highly recommended for historical museums\n\n"
        response += "🎯 **Need help planning a museum route?** Just ask!"
        
        return response
    def _calculate_haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two GPS coordinates using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        lat1, lon1 = point1
        lat2, lon2 = point2
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Earth's radius in kilometers
        return c * r

    def _is_complex_multi_intent_query(self, message: str) -> bool:
        """
        Detect if query is truly complex enough to warrant multi-intent processing.
        
        Simple queries like "Show me museums" should NOT trigger multi-intent handler.
        """
        message_lower = message.lower()
        
        # Indicators of complex multi-intent queries
        complexity_indicators = [
            # Multiple filter combinations
            ' and ', ' with ', ' plus ', ' also ',
            ' near ', ' nearby ', ' close to ',
            ' then ', ' after ', ' before ',
            # Multiple entities
            'museum and restaurant', 'attraction and food', 'places and restaurants',
            'monuments and eat', 'visit and dine', 'see and eat',
            # Route planning with multiple stops
            'from', 'to', 'via', 'through',
            # Multiple time/condition clauses
            'if ', ' when ', ' during ',
            # Comparison queries
            'versus', 'vs', 'compare', 'between',
            'or', 'either'
        ]
        
        # Count complexity indicators detected
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in message_lower)
        
        # Also check for multiple question marks or commas (indicating multiple sub-queries)
        multiple_questions = message.count('?') > 1
        multiple_clauses = message.count(',') >= 2
        
        # Query is complex if it has:
        # - 2+ complexity indicators, OR
        # - Multiple questions, OR  
        # - 3+ filter words AND 2+ clauses
        is_complex = (
            complexity_count >= 2 or
            multiple_questions or
            (complexity_count >= 1 and multiple_clauses)
        )
        
        if is_complex:
            logger.info(f"🔍 Complex query detected (indicators: {complexity_count}, questions: {multiple_questions}, clauses: {multiple_clauses})")
        
        return is_complex

    def _generate_weather_response(self, message: str, entities: Dict, user_profile: UserProfile, 
                                 context: ConversationContext) -> str:
        """Generate weather response using the enhanced weather client and recommendations service"""
        try:
            if not self.weather_client:
                return "🌤️ Weather service is temporarily unavailable. Please try again later."
            
            # Get current weather data for Istanbul
            weather_data = self.weather_client.get_current_weather()
            
            if not weather_data:
                return "🌤️ Unable to fetch current weather data. Please try again later."
            
            # Extract weather data
            temp = weather_data.get('temperature', 'N/A')
            feels_like = weather_data.get('feels_like', temp)
            description = weather_data.get('description', 'N/A')
            humidity = weather_data.get('humidity', 'N/A')
            wind_speed = weather_data.get('wind_speed', 'N/A')
            pressure = weather_data.get('pressure', 'N/A')
            visibility = weather_data.get('visibility', 'N/A')
            current_time = datetime.now().strftime("%H:%M")
            
            # Get weather-based activity recommendations
            recommendations = self._get_weather_based_recommendations(weather_data)
            
            # Build response
            response = f"""🌤️ **Istanbul Weather** (Updated: {current_time})

**📊 Current Conditions:**
• **Temperature**: {temp}°C (feels like {feels_like}°C)
• **Conditions**: {description.title()}
• **Humidity**: {humidity}%
• **Wind**: {wind_speed} km/h
• **Pressure**: {pressure} hPa
• **Visibility**: {visibility} km

{recommendations}

**💡 Travel Tips:**
• Check weather before outdoor activities
• Istanbul weather can change quickly
• Consider indoor alternatives during rain
• Dress in layers for temperature changes
• Bosphorus can be windier than city center

Would you like specific recommendations based on today's weather conditions?"""
            
            return response
            
        except Exception as e:
            logger.error(f"Weather response error: {e}")
            return """🌤️ **Weather Service Temporarily Unavailable**
            
I'm currently unable to fetch real-time weather data for Istanbul. 

**🌍 General Istanbul Weather Info:**
• **Spring (Mar-May)**: Mild, 15-25°C, occasional rain
• **Summer (Jun-Aug)**: Hot, 25-35°C, mostly sunny
• **Fall (Sep-Nov)**: Pleasant, 15-25°C, some rain
• **Winter (Dec-Feb)**: Cool, 5-15°C, frequent rain

**☔ Weather Preparation:**
• Umbrella recommended year-round
• Layers useful for temperature changes
• Waterproof jacket for winter visits
• Sunscreen for summer sightseeing

Please try asking about the weather again in a few moments!"""
    
    def _get_weather_based_recommendations(self, weather_data: Dict) -> str:
        """Generate activity recommendations based on current weather"""
        try:
            temp = weather_data.get('temperature', 20)
            description = weather_data.get('description', '').lower()
            
            # Determine weather condition
            condition = 'clear'
            if any(word in description for word in ['rain', 'drizzle', 'shower']):
                condition = 'rain'
            elif 'snow' in description:
                condition = 'snow'
            elif 'cloud' in description:
                condition = 'cloudy'
            
            # Use weather recommendations service if available
            if self.weather_recommendations:
                try:
                    weather_response = self.weather_recommendations.format_weather_activities_response(
                        temperature=temp,
                        weather_condition=condition,
                        limit=5
                    )
                    return "\n**🎯 Weather-Based Recommendations:**\n" + weather_response
                except Exception as e:
                    logger.error(f"Weather recommendations service error: {e}")
                    # Fall through to fallback recommendations
            
            # Fallback recommendations (original logic)
            recommendations = "\n**🎯 Weather-Based Recommendations:**\n"
            wind_speed = weather_data.get('wind_speed', 0)
            
            # Temperature-based recommendations
            if temp >= 25:
                recommendations += "• **Hot Weather** 🌡️: Visit air-conditioned museums, take Bosphorus ferry for breeze\n"
                recommendations += "  **Top Picks**: \n"
                recommendations += "  - Basilica Cistern (cool underground atmosphere)\n"
                recommendations += "  - Grand Bazaar (covered market, naturally cool)\n"
                recommendations += "  - Bosphorus Ferry (refreshing sea breeze)\n"
                recommendations += "  - Turkish Bath Experience (Çemberlitaş Hamamı)\n"
                recommendations += "• **Stay Cool**: Drink plenty of water, seek shade between 12-4 PM\n"
            elif temp >= 15:
                recommendations += "• **Pleasant Weather** ☀️: Perfect for walking tours and outdoor sightseeing\n"
                recommendations += "  **Top Picks**:\n"
                recommendations += "  - Sultanahmet Square & Blue Mosque\n"
                recommendations += "  - Galata Bridge walk to Galata Tower\n"
                recommendations += "  - Princes' Islands day trip\n"
                recommendations += "  - Gülhane Park or Emirgan Park\n"
                recommendations += "• **Perfect Time**: Ideal for most outdoor activities!\n"
            else:
                recommendations += "• **Cool Weather** 🧥: Focus on indoor attractions and cozy experiences\n"
                recommendations += "  **Top Picks**:\n"
                recommendations += "  - Topkapı Palace (indoor sections)\n"
                recommendations += "  - Turkish Bath Experience (warm & relaxing)\n"
                recommendations += "  - Grand Bazaar & Spice Bazaar (covered)\n"
                recommendations += "  - Istanbul Modern Art Museum\n"
                recommendations += "• **Warm Up**: Try traditional Turkish tea at Pierre Loti Café\n"
            
            # Weather condition recommendations
            if condition == 'rain':
                recommendations += "\n• **Rainy Day** ☔: Perfect for covered markets & underground attractions\n"
                recommendations += "  **Stay Dry Options**:\n"
                recommendations += "  - Grand Bazaar (4,000 covered shops)\n"
                recommendations += "  - Basilica Cistern (underground wonder)\n"
                recommendations += "  - Spice Bazaar (aromatic covered market)\n"
                recommendations += "  - Hagia Sophia & Blue Mosque (indoor sections)\n"
                recommendations += "  - Museum Route: Archaeological, Mosaic, or Istanbul Modern\n"
            elif 'clear' in description or 'sunny' in description:
                recommendations += "\n• **Sunny Day** ☀️: Excellent for Bosphorus tours and panoramic views\n"
                recommendations += "  **Don't Miss**:\n"
                recommendations += "  - Galata Tower (360° city views)\n"
                recommendations += "  - Bosphorus Sunset Cruise\n"
                recommendations += "  - Çamlıca Hill (highest viewpoint)\n"
                recommendations += "  - Ortaköy waterfront\n"
                recommendations += "  - Outdoor markets in Kadıköy\n"
            elif condition == 'cloudy':
                recommendations += "\n• **Cloudy Day** ⛅: Great for photography and comfortable walking\n"
                recommendations += "  **Perfect For**:\n"
                recommendations += "  - Photography tours (soft lighting)\n"
                recommendations += "  - Balat & Fener colorful streets\n"
                recommendations += "  - City exploration without harsh sun\n"
                recommendations += "  - Longer walking tours\n"
            
            # Wind recommendations
            if wind_speed > 20:
                recommendations += "\n• **Windy Conditions** 💨: Be cautious on bridges & high viewpoints\n"
                recommendations += "  **Better Choices**: Sheltered streets of Sultanahmet, indoor museums\n"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Weather recommendations error: {e}")
            return "\n**🎯 General Recommendations:**\n• Check current conditions before heading out\n• Have indoor backup plans ready\n"
    
    def _add_weather_context_to_attractions(self, response: str, weather_data: Optional[Dict] = None) -> str:
        """Add weather context and recommendations to attraction responses"""
        if not weather_data and self.weather_client:
            try:
                weather_data = self.weather_client.get_current_weather()
            except Exception as e:
                logger.debug(f"Could not fetch weather for context: {e}")
                return response
        
        if not weather_data:
            return response
        
        # Extract weather conditions
        temp = weather_data.get('temperature', 20)
        description = weather_data.get('description', '').lower()
        condition = 'clear'
        
        # Determine weather condition
        if any(word in description for word in ['rain', 'drizzle', 'shower']):
            condition = 'rain'
        elif 'snow' in description:
            condition = 'snow'
        elif 'cloud' in description:
            condition = 'cloudy'
        
        # Add weather context header
        weather_note = "\n\n**🌤️ Weather Tip for Today:**\n"
        
        # Use weather recommendations service if available
        if self.weather_recommendations:
            try:
                # Get 3 quick weather-appropriate suggestions
                weather_response = self.weather_recommendations.format_weather_activities_response(
                    temperature=temp,
                    weather_condition=condition,
                    limit=3
                )
                weather_note += weather_response
                return response + "\n" + weather_note
            except Exception as e:
                logger.error(f"Weather recommendations error: {e}")
                # Fall through to basic weather note
        
        # Fallback to basic weather note
        if condition == 'rain':
            weather_note += "It's rainy today - prioritize indoor attractions or covered areas!\n"
            weather_note += "💡 The suggestions above include indoor/covered options perfect for this weather.\n"
        elif temp > 28:
            weather_note += f"It's quite hot today ({temp}°C) - stay hydrated and take breaks in air-conditioned spaces!\n"
            weather_note += "💡 Consider visiting during early morning or late afternoon.\n"
        elif temp < 10:
            weather_note += f"It's chilly today ({temp}°C) - dress warmly and enjoy cozy indoor attractions!\n"
            weather_note += "💡 Perfect weather for museums and Turkish baths.\n"
        else:
            weather_note += f"Pleasant weather today ({temp}°C, {description}) - great for exploring!\n"
            weather_note += "💡 Perfect conditions for outdoor sightseeing.\n"
        
        return response + weather_note
