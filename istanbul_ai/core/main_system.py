"""
Main Istanbul Daily Talk AI System - Simplified and Modular
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..core.user_profile import UserProfile, UserType
from ..core.conversation_context import ConversationContext
from ..core.entity_recognizer import IstanbulEntityRecognizer
from ..utils.constants import ConversationTone, DEFAULT_RESPONSES

# Configure logger first
logger = logging.getLogger(__name__)

# Import advanced transportation system
try:
    import sys
    import os
    # Add parent directory to path to access transportation modules
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from enhanced_transportation_integration import TransportationQueryProcessor, create_ml_enhanced_transportation_system, GPSLocation
    ADVANCED_TRANSPORT_AVAILABLE = True
    logger.info("✅ Advanced transportation system loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Advanced transportation system not available: {e}")
    ADVANCED_TRANSPORT_AVAILABLE = False

# Import industry-level routing system
try:
    from services.routing_service_adapter import get_routing_service, RoutingServiceAdapter
    ROUTING_SERVICE_AVAILABLE = True
    logger.info("✅ Industry-level routing service loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Industry-level routing service not available: {e}")
    ROUTING_SERVICE_AVAILABLE = False


class IstanbulDailyTalkAI:
    """
    Simplified Istanbul Daily Talk AI System
    Refactored from monolithic 2915-line file for better maintainability
    """
    
    def __init__(self):
        self.logger = logger
        self.entity_recognizer = IstanbulEntityRecognizer()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Initialize integrations (simplified)
        self._init_integrations()
        
        # Initialize services
        self._init_services()
        
        logger.info("🎉 Enhanced Istanbul Daily Talk AI System initialized (Modular Architecture)")

    def _init_integrations(self):
        """Initialize external integrations"""
        try:
            # Initialize advanced transportation system
            if ADVANCED_TRANSPORT_AVAILABLE:
                self.transport_processor = TransportationQueryProcessor()
                self.ml_transport_system = create_ml_enhanced_transportation_system()
                logger.info("🚇 Advanced transportation system with IBB API initialized")
            else:
                self.transport_processor = None
                self.ml_transport_system = None
            
            # Initialize industry-level routing service
            if ROUTING_SERVICE_AVAILABLE:
                self.routing_service = get_routing_service()
                if self.routing_service and self.routing_service.is_initialized:
                    logger.info("🗺️ Industry-level routing service initialized successfully")
                else:
                    self.routing_service = None
                    logger.warning("⚠️ Routing service could not be initialized")
            else:
                self.routing_service = None
                
            # Try to load external integrations
            self._load_events_integration()
            self._load_route_integration()
            self._load_deep_learning()
            logger.info("✅ External integrations loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Some integrations not available: {e}")
            self._init_fallback_systems()

    def _load_events_integration(self):
        """Load events integration"""
        try:
            from monthly_events_scheduler import MonthlyEventsScheduler
            self.events_scheduler = MonthlyEventsScheduler()
            self.events_available = True
            logger.info("🎭 Events integration loaded")
        except ImportError:
            self.events_available = False
            logger.warning("⚠️ Events integration not available")

    def _load_route_integration(self):
        """Load route integration"""
        try:
            from services.route_maker import IstanbulRoutemaker
            self.route_maker = IstanbulRoutemaker()
            self.routing_available = True
            logger.info("🗺️ Route integration loaded")
        except ImportError:
            self.routing_available = False
            logger.warning("⚠️ Route integration not available")

    def _load_deep_learning(self):
        """Load deep learning components"""
        try:
            from deep_learning_enhanced_ai import DeepLearningEnhancedAI
            self.deep_learning_ai = DeepLearningEnhancedAI()
            self.deep_learning_available = True
            logger.info("🧠 Deep learning integration loaded")
        except ImportError:
            self.deep_learning_available = False
            logger.warning("⚠️ Deep learning not available")

    def _init_fallback_systems(self):
        """Initialize fallback systems when integrations are not available"""
        self.events_available = False
        self.routing_available = False
        self.deep_learning_available = False

    def _init_services(self):
        """Initialize advanced services"""
        try:
            from ..services.intelligent_location_detector import IntelligentLocationDetector
            from ..services.gps_location_service import GPSLocationService
            from ..services.neighborhood_guide_service import NeighborhoodGuideService
            
            self.location_detector = IntelligentLocationDetector()
            self.gps_service = GPSLocationService()
            self.neighborhood_guide = NeighborhoodGuideService()
            self.advanced_services_available = True
            logger.info("✅ Advanced location and neighborhood guide services loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Advanced services not available: {e}")
            self.location_detector = None
            self.neighborhood_guide = None
            self.gps_service = None
            self.advanced_services_available = False
        
        # Initialize restaurant handler
        try:
            from ..handlers.enhanced_restaurant_handler import EnhancedRestaurantHandler
            self.restaurant_handler = EnhancedRestaurantHandler()
            self.restaurant_available = True
            logger.info("🍽️ Restaurant Recommendation System loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Restaurant Handler not available: {e}")
            self.restaurant_handler = None
            self.restaurant_available = False
        
        # Initialize multi-intent query handler
        try:
            from multi_intent_query_handler import MultiIntentQueryHandler
            self.multi_intent_handler = MultiIntentQueryHandler()
            self.multi_intent_available = True
            logger.info("✅ Multi-Intent Query Handler loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Multi-Intent Query Handler not available: {e}")
            self.multi_intent_handler = None
            self.multi_intent_available = False
        
        # Initialize personality enhancement module
        try:
            from ..services.personality_enhancement import IstanbulPersonality
            self.personality = IstanbulPersonality()
            self.personality_available = True
            logger.info("🎭 Personality Enhancement Module loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Personality Enhancement not available: {e}")
            self.personality = None
            self.personality_available = False
        
        # Initialize museum advising system with comprehensive database
        try:
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            backend_dir = os.path.join(parent_dir, 'backend')
            if backend_dir not in sys.path:
                sys.path.append(backend_dir)
            
            from accurate_museum_database import IstanbulMuseumDatabase
            self.museum_database = IstanbulMuseumDatabase()
            self.museum_available = True
            logger.info("🏛️ Museum Advising System loaded successfully (40 museums)")
        except ImportError as e:
            logger.warning(f"⚠️ Museum Database not available: {e}")
            self.museum_database = None
            self.museum_available = False

    def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
            logger.info(f"Created new user profile for {user_id}")
        return self.user_profiles[user_id]

    def get_or_create_conversation_context(self, session_id: str, user_profile: UserProfile) -> ConversationContext:
        """Get or create conversation context"""
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = ConversationContext(
                session_id=session_id,
                user_profile=user_profile
            )
        return self.conversation_contexts[session_id]

    def process_message(self, user_input: str, user_id: str, gps_location: Optional[Dict] = None) -> str:
        """
        Main message processing method - enhanced with personality and multi-intent support
        
        Args:
            user_input: User's query text
            user_id: Unique user identifier
            gps_location: Optional GPS location dict with 'latitude', 'longitude', 'name'
        """
        try:
            # Get user profile and context
            user_profile = self.get_or_create_user_profile(user_id)
            session_id = f"session_{user_id}"
            context = self.get_or_create_conversation_context(session_id, user_profile)
            
            # 🗺️ STEP 1: INTELLIGENT LOCATION DETECTION (Use for all queries)
            detected_location = None
            location_context = {}
            
            if self.advanced_services_available and self.location_detector:
                try:
                    # First, check if GPS location provided
                    if gps_location:
                        location_context['gps_location'] = gps_location
                        detected_location = gps_location.get('name', 'Current Location')
                        logger.info(f"📍 GPS Location provided: {detected_location} ({gps_location.get('latitude')}, {gps_location.get('longitude')})")
                    
                    # Then, detect location from query text (works for "restaurants in Beyoğlu" etc.)
                    location_result = self.location_detector.detect_location(user_input)
                    if location_result and location_result.get('location'):
                        text_location = location_result.get('location')
                        location_context['text_location'] = text_location
                        if not detected_location:  # Use text location if no GPS
                            detected_location = text_location
                        logger.info(f"📍 Text location detected: {text_location}")
                        
                        # Get neighborhood info if available
                        if location_result.get('coordinates'):
                            location_context['coordinates'] = location_result['coordinates']
                        if location_result.get('district'):
                            location_context['district'] = location_result['district']
                except Exception as e:
                    logger.warning(f"⚠️ Location detection error: {e}")
            
            # Store location in context for follow-up queries
            if detected_location:
                context.last_location = detected_location
                context.location_context = location_context
            
            # 🧠 STEP 2: MULTI-INTENT ANALYSIS (Analyze query complexity and intents)
            intent_analysis = None
            is_complex_query = False
            
            if self.multi_intent_available and self.multi_intent_handler:
                try:
                    intent_analysis = self.multi_intent_handler.analyze_query(user_input)
                    if intent_analysis:
                        # Check if this is a multi-intent query (e.g., "museums and restaurants near Taksim")
                        intents_str = str(intent_analysis).lower()
                        intent_count = sum([
                            'museum' in intents_str,
                            'restaurant' in intents_str or 'food' in intents_str,
                            'transport' in intents_str,
                            'event' in intents_str,
                            'attraction' in intents_str
                        ])
                        is_complex_query = intent_count > 1
                        logger.info(f"🧠 Intent Analysis: {intent_count} intents detected, Complex: {is_complex_query}")
                except Exception as e:
                    logger.warning(f"⚠️ Intent analysis error: {e}")
            
            # 🎭 STEP 3: PERSONALITY ENHANCEMENT: Check for greetings, goodbyes, thanks with warm responses
            if self.personality_available and self.personality:
                # Handle greetings
                greeting_response = self.personality.get_greeting(user_input)
                if greeting_response:
                    logger.info("🎭 Personality: Warm greeting response")
                    return greeting_response
                
                # Handle thank you messages
                if any(word in user_input.lower() for word in ['thank', 'thanks', 'teşekkür', 'sağol', 'appreciate']):
                    logger.info("🎭 Personality: Grateful response")
                    return self.personality.handle_thanks(user_input)
                
                # Handle goodbyes
                if any(word in user_input.lower() for word in ['goodbye', 'bye', 'see you', 'görüşürüz', 'hoşçakal', 'cya']):
                    logger.info("🎭 Personality: Warm goodbye")
                    return self.personality.handle_goodbye(user_input)
                
                # Handle small talk - Weather
                if any(word in user_input.lower() for word in ['weather', 'hava', 'rain', 'sunny', 'hot', 'cold']):
                    logger.info("🎭 Personality: Weather small talk")
                    # Determine weather condition from query
                    weather_condition = 'general'
                    if 'rain' in user_input.lower() or 'rainy' in user_input.lower():
                        weather_condition = 'rainy'
                    elif 'sunny' in user_input.lower() or 'sun' in user_input.lower():
                        weather_condition = 'sunny'
                    elif 'hot' in user_input.lower():
                        weather_condition = 'hot'
                    elif 'cold' in user_input.lower():
                        weather_condition = 'cold'
                    return self.personality.get_weather_talk(weather_condition)
                
                # Handle small talk - Traffic
                if any(word in user_input.lower() for word in ['traffic', 'trafik', 'congestion', 'jam']):
                    logger.info("🎭 Personality: Traffic small talk")
                    # Determine time context
                    import datetime
                    hour = datetime.datetime.now().hour
                    if 7 <= hour <= 9 or 17 <= hour <= 19:
                        time_context = 'rush_hour'
                    elif hour >= 20 or hour <= 6:
                        time_context = 'evening'
                    else:
                        time_context = 'general'
                    return self.personality.get_traffic_talk(time_context)
                
                # Handle cultural questions
                if any(phrase in user_input.lower() for phrase in ['culture', 'custom', 'tradition', 'etiquette', 'turkish people']):
                    logger.info("🎭 Personality: Cultural insight")
                    cultural_response = self._handle_cultural_query(user_input)
                    return self.personality.add_personality_to_response(cultural_response, 'cultural')
            
            # 🏛️ STEP 4: MUSEUM SYSTEM - Handle museum queries with comprehensive database
            if self.museum_available and self.museum_database:
                museum_keywords = [
                    'museum', 'müze', 'palace', 'saray', 'mosque', 'cami',
                    'topkapi', 'hagia sophia', 'ayasofya', 'blue mosque', 'archaeology',
                    # Contemporary art spaces
                    'contemporary art', 'modern art', 'art gallery', 'exhibition',
                    'arter', 'salt', 'galata', 'dirimart', 'pi artworks', 'mixer',
                    'elgiz', 'akbank sanat', 'borusan', 'art museum', 'contemporary',
                    'gallery', 'sanat', 'galeri', 'sergi'
                ]
                if any(word in user_input.lower() for word in museum_keywords):
                    logger.info(f"🏛️ Museum System: Processing museum query (Location: {detected_location or 'Not specified'})")
                    museum_response = self._handle_museum_query(user_input, detected_location, location_context)
                    if museum_response:
                        # Add personality touch to museum responses
                        if self.personality_available and self.personality:
                            return self.personality.add_personality_to_response(museum_response, 'informative')
                        return museum_response
            
            # 🍽️ RESTAURANT SYSTEM: Handle restaurant/food queries with advanced matching
            restaurant_keywords = [
                'restaurant', 'food', 'eat', 'dining', 'restoran', 'yemek',
                'cuisine', 'dish', 'meal', 'breakfast', 'lunch', 'dinner', 
                'brunch', 'cafe', 'cafeteria', 'bistro',
                # Cuisine types
                'turkish', 'ottoman', 'seafood', 'kebab', 'meze', 'baklava',
                'pide', 'lahmacun', 'doner', 'meyhane', 'street food',
                'italian', 'french', 'asian', 'mediterranean', 'international',
                # Dietary
                'vegetarian', 'vegan', 'halal', 'kosher', 'gluten', 'gluten-free',
                'lactose', 'dietary', 'allergy',
                # Price and quality
                'cheap', 'expensive', 'budget', 'luxury', 'affordable',
                # Specific foods
                'balik ekmek', 'simit', 'borek', 'manti', 'kofte', 'dolma'
            ]
            
            if any(word in user_input.lower() for word in restaurant_keywords):
                logger.info(f"🍽️ Restaurant System: Processing query (Location: {detected_location or 'Not specified'})")
                
                # ✅ FIX 3.1: TYPO CORRECTION - Correct common typos before processing
                corrected_query, was_corrected = self._correct_restaurant_typos(user_input)
                if was_corrected:
                    logger.info(f"✏️ Typo correction: '{user_input}' → '{corrected_query}'")
                    user_input = corrected_query  # Use corrected version
                
                # ✅ FIX 3.2: CONFLICT DETECTION - Check for contradictory requirements
                conflicts = self._detect_query_conflicts(user_input)
                if conflicts:
                    logger.info(f"⚔️ Detected {len(conflicts)} query conflict(s)")
                    # Return clarification message about conflicts
                    conflict_response = "🍽️ **I want to help you find the perfect restaurant!**\n\n"
                    conflict_response += "I noticed some conflicting preferences in your request:\n\n"
                    for i, conflict in enumerate(conflicts, 1):
                        conflict_response += f"{i}. {conflict['message']}\n\n"
                    conflict_response += "💡 **Let me know your preference, and I'll give you the best recommendations!**"
                    
                    if self.personality_available and self.personality:
                        return self.personality.add_personality_to_response(conflict_response, 'helpful')
                    return conflict_response
                
                # ✅ FIX 3.3: AMBIGUITY DETECTION - Check if query is too vague
                if self._is_ambiguous_restaurant_query(user_input, detected_location):
                    logger.info("❓ Ambiguous restaurant query detected - asking for clarification")
                    ambiguous_response = self._handle_ambiguous_restaurant_query(user_input, detected_location)
                    
                    if self.personality_available and self.personality:
                        return self.personality.add_personality_to_response(ambiguous_response, 'helpful')
                    return ambiguous_response
                
                # 🧠 If multi-intent detected restaurant intent, use it for complex queries
                if is_complex_query and intent_analysis and 'restaurant' in str(intent_analysis).lower():
                    try:
                        logger.info("🧠 Multi-Intent Handler: Processing complex restaurant query with location context")
                        multi_response = self.multi_intent_handler.generate_response(intent_analysis, user_input)
                        if multi_response and len(multi_response) > 50:  # Valid response
                            if self.personality_available and self.personality:
                                return self.personality.add_personality_to_response(multi_response, 'helpful')
                            return multi_response
                    except Exception as e:
                        logger.warning(f"⚠️ Multi-intent processing error: {e}")
                
                # 🍽️ Use dedicated restaurant handler with detected location
                if self.restaurant_available and self.restaurant_handler:
                    try:
                        restaurant_response = self.restaurant_handler.handle_restaurant_query(
                            user_input=user_input,
                            user_profile=user_profile,
                            detected_location=detected_location,
                            location_context=location_context,  # ✅ FIXED: Pass full location context
                            context=context
                        )
                        if restaurant_response:
                            # Add personality touch to restaurant responses
                            if self.personality_available and self.personality:
                                return self.personality.add_personality_to_response(restaurant_response, 'helpful')
                            return restaurant_response
                    except Exception as e:
                        logger.warning(f"⚠️ Restaurant handler error: {e}")
                
                # Final fallback: Generate helpful restaurant response
                return self._generate_restaurant_fallback(user_input, detected_location)
            
            # 🚇 STEP 6: TRANSPORTATION QUERIES - Use industry-level routing system
            transport_keywords = ['metro', 'bus', 'tram', 'ferry', 'transport', 'how to get', 'direction', 'route',
                                'marmaray', 'metrobüs', 'nostalgic tram', 'from', 'to', 'go to', 'travel',
                                'nasıl gidebilirim', 'nasıl giderim', 'ulaşım', 'yol', 'aktarma']
            
            if any(word in user_input.lower() for word in transport_keywords):
                logger.info(f"🚇 Transportation query detected (Location: {detected_location or 'Not specified'})")
                
                # Priority 1: Try industry-level routing service for route planning
                if self.routing_service and self.routing_service.is_initialized:
                    try:
                        routing_response = self.routing_service.process_routing_query(user_input)
                        if routing_response:
                            logger.info("🗺️ Route planned successfully using industry-level routing system")
                            # Add personality touch to routing responses
                            if self.personality_available and self.personality:
                                return self.personality.add_personality_to_response(routing_response, 'helpful')
                            return routing_response
                    except Exception as e:
                        logger.warning(f"⚠️ Routing service error: {e}")
                
                # Priority 2: Use advanced transportation system for general transport info
                if self.transport_processor:
                    try:
                        transport_response = self._handle_transportation_query(user_input, detected_location, location_context)
                        if transport_response:
                            if self.personality_available and self.personality:
                                return self.personality.add_personality_to_response(transport_response, 'helpful')
                            return transport_response
                    except Exception as e:
                        logger.warning(f"⚠️ Advanced transport system error: {e}")
                
                # Fallback: Basic transportation guidance
                return self._generate_transportation_fallback(user_input, detected_location)
            
            # 🧠 STEP 7: MULTI-INTENT HANDLER - Use for complex multi-category queries
            # Already analyzed at top, now generate response if detected
            if is_complex_query and intent_analysis:
                logger.info("🧠 Multi-Intent: Processing complex multi-category query")
                try:
                    response = self.multi_intent_handler.generate_response(intent_analysis, user_input)
                    if response and len(response) > 50:
                        if self.personality_available and self.personality:
                            return self.personality.add_personality_to_response(response, 'helpful')
                        return response
                except Exception as e:
                    logger.warning(f"⚠️ Multi-intent response generation error: {e}")
            
            # 🧠 STEP 8: SINGLE-INTENT QUERIES - Use intent analysis for routing
            if intent_analysis and not is_complex_query:
                response = self._process_with_intent_analysis(intent_analysis, user_input, user_profile, context, detected_location, location_context)
            else:
                # Fallback to traditional processing
                response = self._process_traditional(user_input, user_profile, context, detected_location)
            
            # 🎭 PERSONALITY ENHANCEMENT: Add personality touch to all responses
            if self.personality_available and self.personality and response:
                response = self.personality.add_personality_to_response(response, 'general')
            
            # Add to conversation history
            context.add_interaction(user_input, response, "processed")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._generate_fallback_response(user_input)
    
    def _handle_cultural_query(self, user_input: str) -> str:
        """Handle cultural and etiquette questions with personality"""
        query_lower = user_input.lower()
        
        # Specific cultural topics
        if 'tea' in query_lower or 'çay' in query_lower:
            return self.personality.get_daily_life_talk('tea_culture')
        elif 'food' in query_lower and 'culture' in query_lower:
            return self.personality.get_daily_life_talk('food_culture')
        elif 'mosque' in query_lower or 'prayer' in query_lower:
            return self.personality.get_cultural_insight('mosque_etiquette')
        elif 'bargain' in query_lower or 'haggle' in query_lower or 'bazaar' in query_lower:
            return self.personality.get_cultural_insight('bargaining')
        elif 'hospitality' in query_lower or 'host' in query_lower:
            return self.personality.get_cultural_insight('hospitality')
        elif 'family' in query_lower:
            return self.personality.get_cultural_insight('family_values')
        elif 'respect' in query_lower or 'polite' in query_lower:
            return self.personality.get_cultural_insight('respect_culture')
        else:
            # General cultural response
            return """🏛️ **Turkish Culture & Etiquette in Istanbul**

Istanbul is a beautiful blend of Eastern and Western cultures! Here are some key cultural insights:

**Social Customs**:
• Turkish people are incredibly hospitable - "Misafir Allah'ın konuğudur" (Guests are God's guests!)
• It's common to be offered çay (tea) - accepting is polite and shows respect
• Greeting with a handshake is common; close friends kiss on both cheeks

**Mosque Etiquette**:
• Remove shoes before entering (shelves provided)
• Dress modestly: cover shoulders and knees
• Ladies should cover their hair (free headscarves at entrance)
• Be quiet and respectful during prayer times

**Dining Etiquette**:
• Say "Afiyet olsun!" (Enjoy your meal!) before eating
• It's polite to try everything offered
• Sharing food is a sign of friendship

**Shopping Culture**:
• Bargaining is expected at bazaars (not rude, it's tradition!)
• Start at 50-60% of asking price
• It's a social interaction, smile and have fun!

**Daily Life**:
• Family is very important in Turkish culture
• Respect for elders is deeply ingrained
• Public displays of affection are generally modest

Enjoy experiencing Turkish hospitality and culture! 🇹🇷"""
    
    def _handle_museum_query(self, user_input: str, detected_location: Optional[str] = None, location_context: Optional[Dict] = None) -> Optional[str]:
        """
        Handle museum-related queries using the comprehensive museum database
        
        Args:
            user_input: User's query text
            detected_location: Location detected from query or GPS
            location_context: Additional location context (GPS coords, district, etc.)
        """
        query_lower = user_input.lower()
        
        # Get all museums from database
        all_museums = self.museum_database.get_all_museums()
        
        # Specific museum lookup (40+ museums including contemporary art spaces)
        museum_keywords = {
            'hagia sophia': 'hagia_sophia',
            'ayasofya': 'hagia_sophia',
            'topkapi': 'topkapi_palace',
            'topkapı': 'topkapi_palace',
            'blue mosque': 'blue_mosque',
            'sultan ahmed': 'blue_mosque',
            'dolmabahce': 'dolmabahce_palace',
            'dolmabahçe': 'dolmabahce_palace',
            'archaeology': 'archaeology_museum',
            'archaeological': 'archaeology_museum',
            'galata tower': 'galata_tower',
            'pera': 'pera_museum',
            'istanbul modern': 'istanbul_modern',
            'chora': 'chora_church',
            'kariye': 'chora_church',
            'basilica cistern': 'basilica_cistern',
            'cistern': 'basilica_cistern',
            'beylerbeyi': 'beylerbeyi_palace',
            'rumeli': 'rumeli_fortress',
            'suleymaniye': 'suleymaniye_mosque',
            'süleymaniye': 'suleymaniye_mosque',
            # Contemporary Art Museums
            'arter': 'arter',
            'salt galata': 'salt_galata',
            'salt beyoglu': 'salt_beyoglu',
            'salt beyoğlu': 'salt_beyoglu',
            'dirimart': 'dirimart_dolapdere',
            'pi artworks': 'pi_artworks',
            'mixer': 'mixer',
            'elgiz': 'elgiz_museum',
            'akbank sanat': 'akbank_sanat',
            'borusan': 'borusan_contemporary',
            'borusan contemporary': 'borusan_contemporary'
        }
        
        # Check for specific museum
        for keyword, museum_id in museum_keywords.items():
            if keyword in query_lower:
                museum = all_museums.get(museum_id)
                if museum:
                    return self._format_museum_info(museum)
        
        # Handle general museum queries
        if any(word in query_lower for word in ['list', 'show', 'all', 'museums', 'recommend']):
            # Filter by district if specified
            district_keywords = {
                'sultanahmet': 'Sultanahmet',
                'beyoğlu': 'Beyoğlu',
                'beyoglu': 'Beyoğlu',
                'kadıköy': 'Kadıköy',
                'kadikoy': 'Kadıköy',
                'beşiktaş': 'Beşiktaş',
                'besiktas': 'Beşiktaş',
                'fatih': 'Fatih'
            }
            
            filtered_museums = []
            target_district = None
            
            for keyword, district in district_keywords.items():
                if keyword in query_lower:
                    target_district = district
                    break
            
            if target_district:
                # Filter by district
                for museum_id, museum in all_museums.items():
                    if target_district.lower() in museum.location.lower():
                        filtered_museums.append(museum)
                
                if filtered_museums:
                    return self._format_museum_list(filtered_museums, f"Museums in {target_district}")
            
            # If no district specified, show top recommendations
            top_museums = ['hagia_sophia', 'topkapi_palace', 'blue_mosque', 'dolmabahce_palace', 
                          'archaeology_museum', 'basilica_cistern', 'galata_tower', 'pera_museum']
            recommended = [all_museums[mid] for mid in top_museums if mid in all_museums]
            return self._format_museum_list(recommended, "Top Museum Recommendations")
        
        # Handle opening hours queries
        if any(word in query_lower for word in ['open', 'hours', 'time', 'when']):
            # Try to find which museum they're asking about
            for keyword, museum_id in museum_keywords.items():
                if keyword in query_lower:
                    museum = all_museums.get(museum_id)
                    if museum:
                        hours_info = f"🕐 **{museum.name} - Opening Hours**\n\n"
                        for day, hours in museum.opening_hours.items():
                            hours_info += f"**{day.capitalize()}**: {hours}\n"
                        hours_info += f"\n💰 **Entrance Fee**: {museum.entrance_fee}\n"
                        hours_info += f"🚪 **Closing Days**: {', '.join(museum.closing_days) if museum.closing_days else 'Open daily'}"
                        return hours_info
        
        # Handle price/fee queries
        if any(word in query_lower for word in ['price', 'fee', 'cost', 'ticket', 'how much']):
            response = "💰 **Museum Entrance Fees in Istanbul**\n\n"
            for museum_id in ['hagia_sophia', 'topkapi_palace', 'blue_mosque', 'dolmabahce_palace', 
                             'archaeology_museum', 'basilica_cistern']:
                museum = all_museums.get(museum_id)
                if museum:
                    response += f"• **{museum.name}**: {museum.entrance_fee}\n"
            response += "\n💡 **Tip**: Many museums offer discounted Museum Pass Istanbul cards!"
            return response
        
        return None
    
    def _format_museum_info(self, museum) -> str:
        """Format detailed museum information"""
        response = f"🏛️ **{museum.name}**\n\n"
        response += f"📍 **Location**: {museum.location}\n"
        response += f"📅 **Period**: {museum.historical_period}\n"
        response += f"🏗️ **Built**: {museum.construction_date}\n"
        
        if museum.architect:
            response += f"👷 **Architect**: {museum.architect}\n"
        
        response += f"\n💰 **Entrance Fee**: {museum.entrance_fee}\n"
        response += f"⏰ **Best Time to Visit**: {museum.best_time_to_visit}\n"
        response += f"⏱️ **Recommended Duration**: {museum.visiting_duration}\n"
        
        response += f"\n📸 **Photography**: {'✅ Allowed' if museum.photography_allowed else '❌ Not allowed'}\n"
        response += f"♿ **Accessibility**: {museum.accessibility}\n"
        
        response += f"\n✨ **Historical Significance**:\n{museum.historical_significance}\n"
        
        if museum.must_see_highlights:
            response += f"\n🎯 **Must-See Highlights**:\n"
            for highlight in museum.must_see_highlights[:5]:
                response += f"• {highlight}\n"
        
        if museum.nearby_attractions:
            response += f"\n🗺️ **Nearby Attractions**:\n"
            for attraction in museum.nearby_attractions[:3]:
                response += f"• {attraction}\n"
        
        return response
    
    def _format_museum_list(self, museums: List, title: str) -> str:
        """Format a list of museums"""
        response = f"🏛️ **{title}**\n\n"
        response += f"I found {len(museums)} amazing museums for you!\n\n"
        
        for i, museum in enumerate(museums[:8], 1):
            response += f"{i}. **{museum.name}**\n"
            response += f"   📍 {museum.location}\n"
            response += f"   💰 {museum.entrance_fee}\n"
            response += f"   ⏰ {museum.best_time_to_visit}\n\n"
        
        if len(museums) > 8:
            response += f"\n...and {len(museums) - 8} more museums! Ask about a specific museum for details.\n"
        
        response += "\n💡 Ask me about any specific museum for detailed information!"
        return response
    
    def search_museums(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for museums based on query text.
        Returns list of museum dictionaries with rich metadata for API responses.
        """
        if not self.museum_available or not self.museum_database:
            return []
        
        query_lower = query.lower()
        all_museums = self.museum_database.get_all_museums()
        matching_museums = []
        
        # Search by name, category, or description
        for museum_id, museum in all_museums.items():
            if (query_lower in museum.name.lower() or 
                query_lower in museum.category.lower() or
                query_lower in museum.description.lower() or
                any(query_lower in h.lower() for h in museum.must_see_highlights)):
                
                # Convert museum object to rich dictionary
                museum_dict = {
                    'id': museum_id,
                    'name': museum.name,
                    'type': 'museum',
                    'category': museum.category,
                    'coordinates': [museum.coordinates['lat'], museum.coordinates['lng']],
                    'location': museum.location,
                    'description': museum.description,
                    'highlights': museum.must_see_highlights[:5],
                    'local_tips': [
                        f"Best time: {museum.best_time_to_visit}",
                        f"Visit duration: {museum.visit_duration}",
                        f"Accessibility: {museum.accessibility_info}"
                    ],
                    'opening_hours': self._format_opening_hours(museum.opening_hours),
                    'entrance_fee': museum.entrance_fee,
                    'best_time_to_visit': museum.best_time_to_visit,
                    'visit_duration': museum.visit_duration,
                    'accessibility': museum.accessibility_info,
                    'nearby_transport': ', '.join(museum.how_to_get_there[:2]) if museum.how_to_get_there else 'Public transport available',
                    'nearby_attractions': museum.nearby_attractions[:3],
                    'website': museum.official_website,
                    'phone': museum.phone,
                    'insider_tips': museum.insider_tips[:3]
                }
                matching_museums.append(museum_dict)
        
        # If no matches by content, try category-based search
        if not matching_museums:
            if any(word in query_lower for word in ['art', 'modern', 'contemporary', 'painting']):
                category_filter = 'Art Museum'
            elif any(word in query_lower for word in ['history', 'archaeological', 'ancient']):
                category_filter = 'Historical Museum'
            elif any(word in query_lower for word in ['palace', 'ottoman', 'sultan']):
                category_filter = 'Palace'
            elif any(word in query_lower for word in ['mosque', 'religious', 'islamic']):
                category_filter = 'Mosque'
            else:
                # Return top museums by default
                top_ids = ['hagia_sophia', 'topkapi_palace', 'istanbul_modern', 'pera_museum', 'archaeology_museum']
                for museum_id in top_ids:
                    if museum_id in all_museums:
                        museum = all_museums[museum_id]
                        matching_museums.append(self._museum_to_dict(museum, museum_id))
                return matching_museums[:5]
            
            for museum_id, museum in all_museums.items():
                if category_filter.lower() in museum.category.lower():
                    matching_museums.append(self._museum_to_dict(museum, museum_id))
        
        return matching_museums[:10]  # Return top 10 matches
    
    def _museum_to_dict(self, museum, museum_id: str) -> Dict[str, Any]:
        """Convert museum object to dictionary"""
        return {
            'id': museum_id,
            'name': museum.name,
            'type': 'museum',
            'category': museum.category,
            'coordinates': [museum.coordinates['lat'], museum.coordinates['lng']],
            'location': museum.location,
            'description': museum.description,
            'highlights': museum.must_see_highlights[:5],
            'local_tips': [
                f"Best time: {museum.best_time_to_visit}",
                f"Visit duration: {museum.visit_duration}",
                f"Accessibility: {museum.accessibility_info}"
            ],
            'opening_hours': self._format_opening_hours(museum.opening_hours),
            'entrance_fee': museum.entrance_fee,
            'best_time_to_visit': museum.best_time_to_visit,
            'visit_duration': museum.visit_duration,
            'accessibility': museum.accessibility_info,
            'nearby_transport': ', '.join(museum.how_to_get_there[:2]) if museum.how_to_get_there else 'Public transport available',
            'nearby_attractions': museum.nearby_attractions[:3],
            'website': museum.official_website,
            'phone': museum.phone,
            'insider_tips': museum.insider_tips[:3]
        }
    
    def _format_opening_hours(self, hours_dict: Dict[str, str]) -> str:
        """Format opening hours dictionary to string"""
        if not hours_dict:
            return "Hours vary, please check website"
        
        # Try to find a typical day pattern
        weekday_hours = hours_dict.get('Monday') or hours_dict.get('Tuesday') or hours_dict.get('Wednesday')
        weekend_hours = hours_dict.get('Saturday') or hours_dict.get('Sunday')
        
        if weekday_hours:
            return f"Weekdays: {weekday_hours}" + (f", Weekends: {weekend_hours}" if weekend_hours and weekend_hours != weekday_hours else "")
        
        return "Check website for hours"
    
    @property
    def museum_system(self):
        """Alias for museum_database for backward compatibility with backend"""
        return self.museum_database
    
    def _process_with_intent_analysis(self, intent_analysis: Any, user_input: str, user_profile: UserProfile, 
                                      context: ConversationContext, detected_location: Optional[str] = None,
                                      location_context: Optional[Dict] = None) -> str:
        """
        Process single-intent queries using intent analysis
        Routes to appropriate handler based on detected intent
        """
        try:
            # Use multi-intent handler's response generation
            response = self.multi_intent_handler.generate_response(intent_analysis, user_input)
            if response and len(response) > 50:
                return response
        except Exception as e:
            logger.warning(f"⚠️ Intent-based processing error: {e}")
        
        # Fallback if intent processing fails
        return self._process_traditional(user_input, user_profile, context, detected_location)
    
    def _process_traditional(self, user_input: str, user_profile: UserProfile, context: ConversationContext, 
                           detected_location: Optional[str] = None) -> str:
        """Traditional fallback processing with location awareness"""
        location_info = f" in {detected_location}" if detected_location else ""
        return f"I'm here to help you discover Istanbul{location_info}! You can ask me about museums, attractions, restaurants, transportation, events, and more. What would you like to know?"
    
    def _generate_fallback_response(self, user_input: str) -> str:
        """Generate fallback response"""
        return f"I'm your Istanbul AI assistant! I can help you with museums, attractions, restaurants, transportation, and more. What would you like to know about Istanbul?"
    
    def _generate_restaurant_fallback(self, user_input: str, detected_location: Optional[str] = None) -> str:
        """Generate fallback response for restaurant queries when handler is not available"""
        query_lower = user_input.lower()
        
        # Build a helpful response based on query content
        response = "🍽️ **Istanbul Dining Recommendations**\n\n"
        
        # Location-specific recommendations
        if detected_location:
            location_lower = detected_location.lower()
            if 'beyoğlu' in location_lower or 'beyoglu' in location_lower or 'taksim' in location_lower:
                response += f"**Great dining options in {detected_location}**:\n"
                response += "• **Istiklal Street**: Hundreds of restaurants, cafes, and eateries\n"
                response += "• **Asmalımescit**: Famous for meyhanes (Turkish taverns)\n"
                response += "• **Nevizade Street**: Traditional Turkish cuisine and seafood\n"
            elif 'sultanahmet' in location_lower:
                response += f"**Dining near {detected_location}**:\n"
                response += "• **Balıkçı Sabahattin**: Historic seafood restaurant\n"
                response += "• **Köfteci Selim**: Famous for Turkish meatballs\n"
                response += "• **Serbethane**: Ottoman cuisine in historic setting\n"
            elif 'kadıköy' in location_lower or 'kadikoy' in location_lower:
                response += f"**Kadıköy food scene** (Asian side):\n"
                response += "• **Çiya Sofrası**: Authentic Anatolian cuisine\n"
                response += "• **Kadıköy Market**: Street food and fresh produce\n"
                response += "• **Moda cafes**: Trendy brunch spots with sea views\n"
            else:
                response += f"Looking for restaurants in **{detected_location}**?\n"
        
        # Cuisine-specific recommendations
        if any(word in query_lower for word in ['seafood', 'fish', 'balik']):
            response += "\n🐟 **Best Seafood Areas**:\n"
            response += "• **Kumkapı**: Traditional fish restaurants\n"
            response += "• **Ortaköy**: Bosphorus-side dining\n"
            response += "• **Balıkçı Sabahattin**: Historic Sultanahmet restaurant\n"
        
        if any(word in query_lower for word in ['vegetarian', 'vegan']):
            response += "\n🥗 **Vegetarian/Vegan Options**:\n"
            response += "• **Zencefil**: Popular vegetarian restaurant in Beyoğlu\n"
            response += "• **Bi Nevi Deli**: Vegan-friendly organic cafe\n"
            response += "• Most Turkish restaurants offer excellent meze (vegetarian appetizers)\n"
        
        if any(word in query_lower for word in ['cheap', 'budget', 'affordable']):
            response += "\n💰 **Budget-Friendly Options**:\n"
            response += "• **Simit + Çay**: Traditional breakfast (10-20 TL)\n"
            response += "• **Lokantas**: Local cafeterias with daily specials (50-100 TL)\n"
            response += "• **Street food**: Döner, balık ekmek, midye (20-50 TL)\n"
        
        if any(word in query_lower for word in ['kebab', 'kebap', 'kofte']):
            response += "\n🍖 **Best Kebab Places**:\n"
            response += "• **Hamdi Restaurant**: Famous for kebabs with Golden Horn views\n"
            response += "• **Zübeyir Ocakbaşı**: Traditional grill house in Beyoğlu\n"
            response += "• **Sultanahmet Köftecisi**: Historic meatball restaurant\n"
        
        # Add general tips
        response += "\n\n💡 **Dining Tips**:\n"
        response += "• Try a **Turkish breakfast** (kahvaltı) - it's legendary!\n"
        response += "• **Meyhanes** are Turkish taverns perfect for evening dining\n"
        response += "• **Lokantas** offer affordable, home-style Turkish food\n"
        response += "• Say '**Afiyet olsun!**' (Bon appétit!) before eating\n"
        
        return response
    
    def _detect_location_from_query(self, user_input: str) -> Optional[str]:
        """Detect location mentioned in query (fallback method)"""
        query_lower = user_input.lower()
        
        # Common Istanbul districts and neighborhoods
        locations = {
            'beyoğlu': 'Beyoğlu',
            'beyoglu': 'Beyoğlu',
            'taksim': 'Taksim',
            'sultanahmet': 'Sultanahmet',
            'kadıköy': 'Kadıköy',
            'kadikoy': 'Kadıköy',
            'beşiktaş': 'Beşiktaş',
            'besiktas': 'Beşiktaş',
            'ortaköy': 'Ortaköy',
            'ortakoy': 'Ortaköy',
            'eminönü': 'Eminönü',
            'eminonu': 'Eminönü',
            'galata': 'Galata',
            'karaköy': 'Karaköy',
            'karakoy': 'Karaköy',
            'fatih': 'Fatih',
            'üsküdar': 'Üsküdar',
            'uskudar': 'Üsküdar',
            'şişli': 'Şişli',
            'sisli': 'Şişli',
            'nişantaşı': 'Nişantaşı',
            'nisantasi': 'Nişantaşı'
        }
        
        for key, value in locations.items():
            if key in query_lower:
                return value
        
        return None
    
    def _detect_query_conflicts(self, user_input: str) -> List[Dict[str, str]]:
        """
        Detect conflicting requirements in restaurant queries
        
        Returns:
            List of conflict dictionaries with type and message
        """
        conflicts = []
        query_lower = user_input.lower()
        
        # Price conflicts
        if ('cheap' in query_lower or 'budget' in query_lower or 'affordable' in query_lower) and \
           ('luxury' in query_lower or 'expensive' in query_lower or 'fine dining' in query_lower or 'michelin' in query_lower):
            conflicts.append({
                'type': 'price_conflict',
                'message': "I notice you're looking for both budget-friendly and luxury options. Would you like affordable restaurants with a premium feel, or should I focus on one or the other?"
            })
        
        # Dietary conflicts
        vegan_vegetarian = any(word in query_lower for word in ['vegan', 'vegetarian', 'plant-based', 'veggie'])
        meat_focused = any(word in query_lower for word in ['steakhouse', 'bbq', 'kebab', 'meat', 'steak'])
        
        if vegan_vegetarian and meat_focused:
            conflicts.append({
                'type': 'dietary_conflict',
                'message': "I see you're interested in both vegan/vegetarian options and meat-focused restaurants. Would you like restaurants that offer both options, or are you looking for separate recommendations?"
            })
        
        # Cuisine conflicts
        italian = 'italian' in query_lower
        turkish = any(word in query_lower for word in ['turkish', 'ottoman', 'traditional'])
        authentic = 'authentic' in query_lower
        
        if italian and turkish and authentic:
            conflicts.append({
                'type': 'cuisine_conflict',
                'message': "You mentioned both Italian and Turkish cuisine. Would you like fusion restaurants, or are you looking for two separate recommendations?"
            })
        
        # Ambiance conflicts
        quiet = any(word in query_lower for word in ['quiet', 'peaceful', 'calm', 'silent'])
        live_music = any(word in query_lower for word in ['live music', 'entertainment', 'band', 'performance'])
        
        if quiet and live_music:
            conflicts.append({
                'type': 'ambiance_conflict',
                'message': "You mentioned both quiet atmosphere and live music. Would you like a restaurant with outdoor quiet seating and indoor music, or should I prioritize one?"
            })
        
        # Time conflicts
        breakfast = 'breakfast' in query_lower
        late_night = any(word in query_lower for word in ['late night', 'after midnight', 'late'])
        
        if breakfast and late_night:
            conflicts.append({
                'type': 'time_conflict',
                'message': "Are you looking for a 24/7 restaurant that serves breakfast, or separate breakfast and late-night options?"
            })
        
        return conflicts
    
    def _is_ambiguous_restaurant_query(self, user_input: str, detected_location: Optional[str]) -> bool:
        """
        Check if restaurant query is too ambiguous to provide good recommendations
        
        Returns:
            True if query needs clarification
        """
        query_lower = user_input.lower()
        
        # Very short queries
        word_count = len(query_lower.split())
        if word_count <= 2 and any(word in query_lower for word in ['food', 'restaurant', 'eat', 'dining']):
            return True
        
        # Generic queries without specifics
        generic_patterns = [
            'best food',
            'good place',
            'where to eat',
            'find food',
            'somewhere to eat'
        ]
        
        if any(pattern in query_lower for pattern in generic_patterns):
            # Check if we have any specifics
            has_location = detected_location is not None or any(
                loc in query_lower for loc in ['beyoğlu', 'beyoglu', 'sultanahmet', 'taksim', 'kadıköy', 'kadikoy']
            )
            has_cuisine = any(
                cuisine in query_lower for cuisine in ['turkish', 'italian', 'seafood', 'kebab', 'pizza', 'sushi']
            )
            has_price = any(
                price in query_lower for price in ['cheap', 'expensive', 'budget', 'luxury', 'affordable']
            )
            has_dietary = any(
                diet in query_lower for diet in ['vegetarian', 'vegan', 'halal', 'kosher', 'gluten']
            )
            
            # If missing most context, it's ambiguous
            specifics_count = sum([has_location, has_cuisine, has_price, has_dietary])
            if specifics_count <= 1:
                return True
        
        return False
    
    def _handle_ambiguous_restaurant_query(self, query: str, detected_location: Optional[str]) -> str:
        """
        Handle ambiguous restaurant queries with helpful clarification prompts
        
        Args:
            query: User's query
            detected_location: Any detected location
            
        Returns:
            Clarification message with helpful prompts
        """
        response = "🍽️ **I'd love to help you find the perfect restaurant!**\n\n"
        response += "To give you the best recommendations, could you tell me a bit more?\n\n"
        
        # Ask for missing information
        missing = []
        
        if not detected_location:
            missing.append("📍 **Location**: Which area? (Beyoğlu, Sultanahmet, Kadıköy, etc.)")
        
        query_lower = query.lower()
        
        # Check for cuisine
        cuisines = ['turkish', 'italian', 'seafood', 'kebab', 'asian', 'french', 'mediterranean']
        if not any(c in query_lower for c in cuisines):
            missing.append("🍲 **Cuisine**: Turkish, seafood, Italian, vegetarian, etc.?")
        
        # Check for price
        if not any(p in query_lower for p in ['cheap', 'expensive', 'budget', 'luxury', 'affordable']):
            missing.append("💰 **Budget**: Budget-friendly, mid-range, or luxury?")
        
        # Check for occasion/mood
        if not any(o in query_lower for o in ['romantic', 'family', 'business', 'casual', 'formal']):
            missing.append("✨ **Occasion**: Casual dining, romantic, family-friendly, business?")
        
        # Show what we're missing
        for item in missing[:3]:  # Show top 3 missing items
            response += f"• {item}\n"
        
        response += "\n**Example queries**:\n"
        response += "• 'Seafood restaurants in Beyoğlu'\n"
        response += "• 'Budget-friendly Turkish food near Taksim'\n"
        response += "• 'Vegetarian restaurants with Bosphorus view'\n"
        response += "• 'Romantic Italian restaurant for anniversary'\n"
        
        response += "\n💡 **Or just tell me more about what you're in the mood for!**"
        
        return response
    
    def _correct_restaurant_typos(self, query: str) -> tuple[str, bool]:
        """
        Correct common typos in restaurant queries using fuzzy matching
        
        Returns:
            Tuple of (corrected_query, was_corrected)
        """
        original = query
        corrected = query
        
        # Try importing thefuzz for advanced fuzzy matching
        try:
            from thefuzz import fuzz, process
            FUZZY_AVAILABLE = True
        except ImportError:
            FUZZY_AVAILABLE = False
        
        # Common restaurant terms for fuzzy matching
        restaurant_terms = [
            'restaurant', 'vegetarian', 'seafood', 'halal', 'expensive', 'cheap',
            'budget', 'luxury', 'cuisine', 'traditional', 'modern', 'rooftop',
            'breakfast', 'lunch', 'dinner', 'brunch', 'dessert', 'cafe'
        ]
        
        # Location terms for fuzzy matching
        location_terms = [
            'Beyoğlu', 'Kadıköy', 'Istanbul', 'Taksim', 'Sultanahmet', 
            'Beşiktaş', 'Ortaköy', 'Eminönü', 'Galata', 'Uskudar',
            'Sisli', 'Fatih', 'Karakoy', 'Besiktas', 'Bakirkoy'
        ]
        
        # Exact typo corrections (fast path)
        typo_corrections = {
            'restourant': 'restaurant',
            'resturant': 'restaurant',
            'restraunt': 'restaurant',
            'restarant': 'restaurant',
            'vegiterian': 'vegetarian',
            'vegeterian': 'vegetarian',
            'seafod': 'seafood',
            'halaal': 'halal',
            'were to': 'where to',
            'were can': 'where can',
            'expensiv': 'expensive',
            'beyoglu': 'Beyoğlu',
            'kadikoy': 'Kadıköy',
            'istambul': 'Istanbul',
            'constantinople': 'Istanbul',
            'taxim': 'Taksim',
            'taksim square': 'Taksim',
            'sultanahmed': 'Sultanahmet',
            'besiktas': 'Beşiktaş',
            'ortakoy': 'Ortaköy',
            'eminonu': 'Eminönü'
        }
        
        # Fast exact matching first
        import re
        query_lower = corrected.lower()
        for typo, correct in typo_corrections.items():
            if typo in query_lower:
                pattern = re.compile(re.escape(typo), re.IGNORECASE)
                corrected = pattern.sub(correct, corrected)
        
        # Fuzzy matching for remaining terms (if available)
        if FUZZY_AVAILABLE:
            words = corrected.split()
            corrected_words = []
            
            for word in words:
                word_lower = word.lower().strip('.,!?')
                corrected_word = word
                
                # Skip short words and already corrected words
                if len(word_lower) < 4:
                    corrected_words.append(word)
                    continue
                
                # Check restaurant terms
                best_match = process.extractOne(word_lower, restaurant_terms, scorer=fuzz.ratio)
                if best_match and best_match[1] >= 80:  # 80% similarity threshold
                    if best_match[1] < 100:  # Only if not exact match
                        corrected_word = best_match[0]
                        self.logger.debug(f"Fuzzy corrected '{word}' to '{corrected_word}' (score: {best_match[1]})")
                
                # Check location terms if no restaurant match
                if corrected_word == word:
                    best_match = process.extractOne(word_lower, location_terms, scorer=fuzz.ratio)
                    if best_match and best_match[1] >= 80:
                        if best_match[1] < 100:
                            corrected_word = best_match[0]
                            self.logger.debug(f"Fuzzy corrected location '{word}' to '{corrected_word}' (score: {best_match[1]})")
                
                corrected_words.append(corrected_word)
            
            corrected = ' '.join(corrected_words)
        
        was_corrected = (corrected != original)
        if was_corrected:
            self.logger.info(f"🔧 Typo correction: '{original}' -> '{corrected}'")
        
        return corrected, was_corrected
    
    def _handle_transportation_query(self, user_input: str, detected_location: Optional[str] = None, 
                                     location_context: Optional[Dict] = None) -> Optional[str]:
        """
        Handle transportation queries using advanced transport system
        
        Args:
            user_input: User's query text
            detected_location: Location detected from query or GPS
            location_context: Additional location context (GPS coords, district, etc.)
        
        Returns:
            Transportation response or None if not available
        """
        if not self.transport_processor:
            return None
        
        try:
            # Use the advanced transportation system for general info
            query_lower = user_input.lower()
            
            # Build response based on query type
            response_parts = []
            
            # General transport info queries
            if any(word in query_lower for word in ['how does', 'how do i use', 'explain', 'what is', 'tell me about']):
                if 'metro' in query_lower:
                    response_parts.append("🚇 **Istanbul Metro System**\n")
                    response_parts.append("Istanbul has an extensive metro network covering both European and Asian sides:\n\n")
                    response_parts.append("**Major Lines:**")
                    response_parts.append("• **M1** (Red): Airport - Yenikapı - Kirazlı")
                    response_parts.append("• **M2** (Green): Yenikapı - Hacıosman (connects to Marmaray)")
                    response_parts.append("• **M3** (Blue): Kirazlı - Olimpiyat - Başakşehir")
                    response_parts.append("• **M4** (Pink): Kadıköy - Tavşantepe (Asian side)")
                    response_parts.append("• **M5** (Purple): Üsküdar - Çekmeköy (Asian side)")
                    response_parts.append("• **M6** (Brown): Levent - Hisarüstü")
                    response_parts.append("• **M7** (Light Blue): Mecidiyeköy - Mahmutbey")
                    response_parts.append("\n💳 **Payment:** Use Istanbul Card (Istanbulkart)")
                    response_parts.append("💰 **Cost:** Single ride ~₺15-20 (with discount card)")
                    
                elif 'marmaray' in query_lower:
                    response_parts.append("🚄 **Marmaray Rail System**\n")
                    response_parts.append("The first rail tunnel connecting Europe and Asia!\n\n")
                    response_parts.append("**Route:** Halkalı (Europe) ↔ Gebze (Asia)")
                    response_parts.append("**Key Stops:** Halkalı, Bakırköy, Yenikapı, Sirkeci, Ayrılıkçeşmesi, Üsküdar, Kadıköy, Gebze")
                    response_parts.append("\n**Connections:**")
                    response_parts.append("• Metro M2 at Yenikapı")
                    response_parts.append("• Metro M4 at Ayrılıkçeşmesi")
                    response_parts.append("• Metrobüs at multiple stations")
                    response_parts.append("\n⏱️ **Frequency:** Every 3-15 minutes depending on time")
                    response_parts.append("💰 **Cost:** Same as metro (~₺15-20)")
                
                elif 'bus' in query_lower or 'otobüs' in query_lower:
                    response_parts.append("🚌 **Istanbul Bus System**\n")
                    response_parts.append("Extensive bus network covering all neighborhoods:\n\n")
                    response_parts.append("**Types:**")
                    response_parts.append("• **IETT Buses:** Main public buses (red)")
                    response_parts.append("• **Metrobüs:** Rapid transit buses on dedicated lanes")
                    response_parts.append("• **Private Buses:** Yellow and white buses")
                    response_parts.append("\n💳 **Payment:** Istanbul Card required")
                    response_parts.append("💰 **Cost:** ~₺15 per ride")
                    response_parts.append("\n📱 **Tip:** Use Istanbul public transport apps for real-time info!")
                
                elif 'ferry' in query_lower or 'vapur' in query_lower:
                    response_parts.append("⛴️ **Istanbul Ferry System**\n")
                    response_parts.append("Experience Istanbul by sea!\n\n")
                    response_parts.append("**Main Routes:**")
                    response_parts.append("• Eminönü ↔ Kadıköy (Most popular)")
                    response_parts.append("• Karaköy ↔ Kadıköy")
                    response_parts.append("• Beşiktaş ↔ Üsküdar")
                    response_parts.append("• Bosphorus tours (Eminönü - Princes' Islands)")
                    response_parts.append("\n💳 **Payment:** Istanbul Card or token")
                    response_parts.append("💰 **Cost:** ~₺15-25 depending on route")
                    response_parts.append("⏱️ **Duration:** 15-25 minutes for short crossings")
                    response_parts.append("\n🌅 **Tip:** Best views during sunset!")
            
            if response_parts:
                return "\n".join(response_parts)
            
            # If no specific info query, return None to try routing
            return None
            
        except Exception as e:
            logger.error(f"Error in transportation query handler: {e}")
            return None
    
    def _generate_transportation_fallback(self, user_input: str, detected_location: Optional[str] = None) -> str:
        """
        Generate fallback response for transportation queries
        
        Args:
            user_input: User's query text
            detected_location: Location detected from query or GPS
        
        Returns:
            Helpful transportation fallback response
        """
        response_parts = []
        response_parts.append("🚇 **Istanbul Public Transportation**\n")
        
        if detected_location:
            response_parts.append(f"Getting around {detected_location}:\n")
        
        response_parts.append("\n**Transportation Options:**")
        response_parts.append("• 🚇 **Metro:** Fast, modern, covers major districts")
        response_parts.append("• 🚄 **Marmaray:** Connects Europe and Asia under the Bosphorus")
        response_parts.append("• 🚌 **Buses:** Extensive network, IETT and Metrobüs")
        response_parts.append("• 🚊 **Trams:** Historic and modern lines (Istiklal, Sultanahmet)")
        response_parts.append("• ⛴️ **Ferries:** Scenic Bosphorus crossings")
        response_parts.append("• 🚕 **Taxis:** Yellow cabs and ride-sharing apps")
        
        response_parts.append("\n💳 **Istanbul Card (Istanbulkart):**")
        response_parts.append("Essential for public transport! Available at kiosks and machines.")
        response_parts.append("Offers discounted fares and easy transfers.")
        
        response_parts.append("\n💡 **Tips:**")
        response_parts.append("• Download İBB mobile app for real-time schedules")
        response_parts.append("• Peak hours: 7-9 AM, 5-7 PM (can be crowded)")
        response_parts.append("• Most lines run 6 AM - 12 AM")
        
        response_parts.append("\n🗺️ **For specific route planning:**")
        response_parts.append("Ask me: 'How do I get from [origin] to [destination]?'")
        response_parts.append("Example: 'How do I get from Taksim to Kadıköy?'")
        
        return "\n".join(response_parts)
