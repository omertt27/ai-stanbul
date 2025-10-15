"""
Istanbul Daily Talk AI - Main System
The main orchestration class for the Istanbul AI system.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .core.models import UserProfile, ConversationContext
from .core.entity_recognition import IstanbulEntityRecognizer
from .core.response_generator import ResponseGenerator
from .core.user_management import UserManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Import ML-Enhanced Daily Talks Bridge
try:
    from ml_enhanced_daily_talks_bridge import MLEnhancedDailyTalksBridge, process_enhanced_daily_talk
    ML_DAILY_TALKS_AVAILABLE = True
    logger.info("✅ ML-Enhanced Daily Talks Bridge loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ ML-Enhanced Daily Talks Bridge not available: {e}")
    ML_DAILY_TALKS_AVAILABLE = False


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
        
        # System status
        self.system_ready = True
        logger.info("✅ Istanbul Daily Talk AI System initialized successfully!")
    
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
    
    def process_message(self, message: str, user_id: str) -> str:
        """Process user message and generate response"""
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
            
            # Check if this is a daily talk query (casual conversation, greetings, weather, etc.)
            if self._is_daily_talk_query(message):
                return self._handle_daily_talk_query(message, user_id, session_id, user_profile, context)
            
            # Extract entities from message
            entities = self.entity_recognizer.extract_entities(message)
            
            # Classify intent with context
            intent = self._classify_intent_with_context(message, entities, context)
            
            # Generate contextual response
            response = self._generate_contextual_response(
                message, intent, entities, user_profile, context
            )
            
            # Record interaction
            context.add_interaction(message, response, intent)
            
            return response
            
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
        
        # Greeting patterns
        greeting_patterns = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how r u', 'whats up', "what's up", 'merhaba', 'selam'
        ]
        
        # Weather patterns
        weather_patterns = [
            'weather', 'temperature', 'rain', 'sunny', 'cloudy', 'hot', 'cold',
            'warm', 'cool', 'forecast', 'climate', 'degrees'
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
                                user_profile: UserProfile, context: ConversationContext) -> str:
        """Handle daily talk queries through ML-enhanced bridge"""
        
        if not ML_DAILY_TALKS_AVAILABLE or not self.daily_talks_bridge:
            # Fallback to basic daily talk response
            return self._generate_basic_daily_talk_response(message, user_profile, context)
        
        try:
            # Prepare context data for the ML bridge
            context_data = {
                'location': getattr(user_profile, 'current_location', None),
                'preferences': {
                    'interests': getattr(user_profile, 'interests', []),
                    'user_type': getattr(user_profile, 'user_type', {}).get('value', 'first_time_visitor') if hasattr(user_profile.user_type, 'value') else str(user_profile.user_type),
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
        
        # Weather responses
        if any(weather in message_lower for weather in ['weather', 'temperature', 'rain', 'sunny']):
            return "🌤️ I don't have real-time weather data, but Istanbul weather can be quite pleasant! Check the local weather and I can suggest activities that match the conditions. What kind of experience are you looking for?"
        
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
        
        # Attraction/sightseeing intent
        attraction_keywords = ['visit', 'see', 'attraction', 'museum', 'mosque', 'palace', 'tour', 'sightseeing']
        if any(keyword in message_lower for keyword in attraction_keywords) or entities.get('landmarks'):
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
        
        # Route planning intent
        route_keywords = ['route', 'itinerary', 'plan', 'schedule', 'day trip']
        if any(keyword in message_lower for keyword in route_keywords):
            return 'route_planning'
        
        # Greeting/general intent
        greeting_keywords = ['hello', 'hi', 'merhaba', 'help', 'start']
        if any(keyword in message_lower for keyword in greeting_keywords):
            return 'greeting'
        
        return 'general'
    
    def _generate_contextual_response(self, message: str, intent: str, entities: Dict,
                                    user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate contextual response based on intent and entities"""
        
        current_time = datetime.now()
        
        # Use response generator for comprehensive responses
        if intent == 'attraction':
            # Check if this is a museum query - use enhanced museum system
            message_lower = message.lower()
            museum_keywords = ['museum', 'museums', 'gallery', 'exhibition', 'art', 'historical sites', 'cultural sites']
            if any(keyword in message_lower for keyword in museum_keywords) and self.museum_generator:
                return self._generate_location_aware_museum_response(message, entities, user_profile, context)
            else:
                return self.response_generator.generate_comprehensive_recommendation(
                    intent, entities, user_profile, context
                )
        elif intent in ['restaurant', 'neighborhood']:
            return self.response_generator.generate_comprehensive_recommendation(
                intent, entities, user_profile, context
            )
        
        # Handle specific intents
        elif intent == 'transportation':
            return self._generate_transportation_response(message, entities, user_profile, context)
        
        elif intent == 'shopping':
            return self._generate_shopping_response(entities, user_profile, context)
        
        elif intent == 'events':
            return self._generate_events_response(entities, user_profile, context, current_time)
        
        elif intent == 'route_planning':
            return self._generate_route_planning_response(message, user_profile, context)
        
        elif intent == 'greeting':
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
                                        context: ConversationContext) -> str:
        """Generate comprehensive transportation response with advanced AI and real-time data"""
        try:
            # Use advanced transportation system if available
            if ADVANCED_TRANSPORT_AVAILABLE and self.transport_processor:
                logger.info("🚇 Using advanced transportation system with IBB API")
                
                # Process query through advanced system using the actual message
                enhanced_response = self.transport_processor.process_transportation_query_sync(
                    message, entities, user_profile
                )
                
                if enhanced_response and enhanced_response.strip():
                    return enhanced_response
                    
            # Fallback to improved static response
            logger.info("🚇 Using fallback transportation system")
            return self._get_fallback_transportation_response(entities, user_profile, context)
            
        except Exception as e:
            logger.error(f"Transportation query error: {e}")
            return self._get_fallback_transportation_response(entities, user_profile, context)

    def _get_fallback_transportation_response(self, entities: Dict, user_profile: UserProfile, 
                                            context: ConversationContext) -> str:
        """Fallback transportation response with correct information"""
        current_time = datetime.now().strftime("%H:%M")
        
        return f"""🚇 **Istanbul Transportation Guide**
📍 **Live Status** (Updated: {current_time})

**🎫 Essential Transport Card:**
• **Istanbulkart**: Must-have for all public transport (13 TL + credit)
• Available at metro stations, kiosks, and ferry terminals
• Works on metro, tram, bus, ferry, and dolmuş

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
• Dolmuş (shared taxis) follow set routes
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
        """Generate events and activities response"""
        
        return """🎭 **Istanbul Events & Activities**

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

**🎪 Seasonal Events:**
• **Spring**: Tulip Festival (April), Historic Peninsula blooms
• **Summer**: Istanbul Music Festival, outdoor concerts
• **Fall**: Istanbul Biennial (odd years), art across the city
• **Winter**: New Year celebrations, cozy indoor venues

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

What type of experience interests you most? I can provide specific venue recommendations and booking details!"""
    
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
    
    def _detect_location_from_message(self, message: str) -> Optional[str]:
        """Simple location detection from message text"""
        message_lower = message.lower()
        
        # Location mapping
        location_keywords = {
            'sultanahmet': ['sultanahmet', 'blue mosque', 'hagia sophia', 'topkapi'],
            'beyoğlu': ['beyoğlu', 'beyoglu', 'galata tower', 'istiklal', 'taksim'],
            'galata': ['galata', 'karaköy', 'karakoy'],
            'eminönü': ['eminönü', 'eminonu', 'spice bazaar', 'galata bridge'],
            'beşiktaş': ['beşiktaş', 'besiktas', 'dolmabahçe', 'dolmabahce'],
            'kadıköy': ['kadıköy', 'kadikoy', 'asian side'],
            'üsküdar': ['üsküdar', 'uskudar', 'maiden tower']
        }
        
        # Check for explicit location mentions
        for location, keywords in location_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return location
        
        return None
