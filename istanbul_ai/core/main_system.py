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

    def process_message(self, user_input: str, user_id: str) -> str:
        """
        Main message processing method - enhanced with personality and multi-intent support
        """
        try:
            # Get user profile and context
            user_profile = self.get_or_create_user_profile(user_id)
            session_id = f"session_{user_id}"
            context = self.get_or_create_conversation_context(session_id, user_profile)
            
            # 🎭 PERSONALITY ENHANCEMENT: Check for greetings, goodbyes, thanks with warm responses
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
            
            # 🏛️ MUSEUM SYSTEM: Handle museum queries with comprehensive database
            if self.museum_available and self.museum_database:
                if any(word in user_input.lower() for word in ['museum', 'müze', 'palace', 'saray', 'mosque', 'cami', 'topkapi', 'hagia sophia', 'ayasofya', 'blue mosque', 'archaeology']):
                    logger.info("🏛️ Museum System: Processing museum query")
                    museum_response = self._handle_museum_query(user_input)
                    if museum_response:
                        # Add personality touch to museum responses
                        if self.personality_available and self.personality:
                            return self.personality.add_personality_to_response(museum_response, 'informative')
                        return museum_response
            
            # Use multi-intent handler if available for enhanced processing
            if self.multi_intent_available and self.multi_intent_handler:
                response = self._process_with_multi_intent(user_input, user_profile, context)
            else:
                # Fallback to traditional processing
                response = self._process_traditional(user_input, user_profile, context)
            
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
    
    def _handle_museum_query(self, user_input: str) -> Optional[str]:
        """Handle museum-related queries using the comprehensive museum database"""
        query_lower = user_input.lower()
        
        # Get all museums from database
        all_museums = self.museum_database.get_all_museums()
        
        # Specific museum lookup
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
            'süleymaniye': 'suleymaniye_mosque'
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
    
    def _process_with_multi_intent(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> str:
