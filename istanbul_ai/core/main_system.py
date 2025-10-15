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
    
    from enhanced_transportation_system import TransportationQueryProcessor
    from ml_enhanced_transportation_system import create_ml_enhanced_transportation_system, GPSLocation
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
        Main message processing method - enhanced with multi-intent support
        """
        try:
            # Get user profile and context
            user_profile = self.get_or_create_user_profile(user_id)
            session_id = f"session_{user_id}"
            context = self.get_or_create_conversation_context(session_id, user_profile)
            
            # Use multi-intent handler if available for enhanced processing
            if self.multi_intent_available and self.multi_intent_handler:
                response = self._process_with_multi_intent(user_input, user_profile, context)
            else:
                # Fallback to traditional processing
                response = self._process_traditional(user_input, user_profile, context)
            
            # Add to conversation history
            context.add_interaction(user_input, response, "processed")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._generate_fallback_response(user_input)

    def _determine_primary_intent(self, intent_signals: Dict[str, float]) -> str:
        """Determine primary intent from signals with improved logic"""
        if not intent_signals:
            return "general_info"
        
        # Return intent with highest confidence, but with minimum threshold
        max_intent, max_confidence = max(intent_signals.items(), key=lambda x: x[1])
        
        # Only return intent if confidence is above threshold
        if max_confidence >= 0.3:
            return max_intent
        else:
            return "general_info"

    def _process_by_intent(self, user_input: str, intent: str, entities: Dict, user_profile: UserProfile) -> str:
        """Process message by detected intent - legacy method"""
        return self._process_by_intent_enhanced(user_input, intent, entities, user_profile)

    def _process_by_intent_enhanced(self, user_input: str, intent: str, entities: Dict, user_profile: UserProfile) -> str:
        """Process message by detected intent with enhanced routing"""
        
        # Get context for enhanced processing
        session_id = f"session_{user_profile.user_id}"
        context = self.get_or_create_conversation_context(session_id, user_profile)
        
        # Check for neighborhood/district-specific queries first (Priority #1)
        user_input_lower = user_input.lower()
        matched_districts = []
        
        if self.neighborhood_guide:
            matched_districts = self.neighborhood_guide.find_districts_by_keywords(user_input)
            
            # Enhanced neighborhood detection keywords
            neighborhood_keywords = [
                'neighborhood', 'district', 'area', 'guide', 'vibe', 'atmosphere', 'character',
                'compare', 'comparison', 'vs', 'versus', 'which is better',
                'complete guide', 'comprehensive guide', 'everything about',
                'besiktas', 'beşiktaş', 'kadikoy', 'kadıköy', 'sultanahmet', 
                'sisli', 'şişli', 'uskudar', 'üsküdar', 'fatih', 'sariyer', 'sarıyer'
            ]
            
            # Multi-intent neighborhood queries (e.g., "restaurants in Kadıköy", "nightlife in Beşiktaş")
            district_context_keywords = [
                f"{query_type} in" for query_type in ['restaurants', 'nightlife', 'attractions', 'shopping', 'food', 'bars', 'clubs', 'sights']
            ]
            
            has_neighborhood_context = (
                any(word in user_input_lower for word in neighborhood_keywords) or
                any(phrase in user_input_lower for phrase in district_context_keywords) or
                matched_districts or
                # Check for specific district mentions with context
                any(f"in {district}" in user_input_lower for district in ['kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'sultanahmet', 'sisli', 'şişli', 'uskudar', 'üsküdar', 'fatih', 'sariyer', 'sarıyer'])
            )
            
            if has_neighborhood_context:
                return self._handle_neighborhood_guide_query(user_input, entities, user_profile, context)
        
        # Map different intent formats to consistent handlers
        elif intent in ["restaurant", "restaurant_query"] or self._is_restaurant_query(user_input):
            return self._handle_restaurant_query(user_input, entities, user_profile, context)
        elif intent in ["events", "events_query"] or any(word in user_input.lower() for word in ['event', 'happening', 'show', 'performance', 'exhibition']):
            return self._handle_events_query(user_input, entities, user_profile)
        elif intent in ["transportation", "transportation_query"] or any(word in user_input.lower() for word in ['transport', 'metro', 'bus', 'taxi', 'how to get']):
            return self._handle_transportation_query(user_input, entities, user_profile)
        elif intent in ["attractions", "attraction_query"] or any(word in user_input.lower() for word in ['visit', 'see', 'attraction', 'museum', 'palace']):
            return self._handle_attractions_query(user_input, entities, user_profile)
        else:
            return self._handle_general_query(user_input, entities, user_profile)

    def _handle_restaurant_query(self, user_input: str, entities: Dict, user_profile: UserProfile, context: Optional['ConversationContext'] = None) -> str:
        """Handle restaurant-related queries with enhanced recommendations and location detection"""
        
        # Get location information - either from query or intelligent detection
        districts = entities.get('districts', [])
        detected_location = None
        
        if districts:
            detected_location = districts[0]
        elif context:
            # Use intelligent location detection if no explicit location in query
            detected_location = self._detect_user_location(user_input, user_profile, context)
        
        cuisines = entities.get('cuisines', [])
        dietary = entities.get('dietary', [])
        budget = entities.get('budget', [])
        time_refs = entities.get('time', [])
        
        # Check for specific food items and complex requirements
        query_lower = user_input.lower()
        
        # Handle specific food/drink queries
        if 'turkish coffee' in query_lower or ('coffee' in query_lower and 'dessert' in query_lower):
            return self._handle_coffee_dessert_query(user_input, entities)
        elif 'pide' in query_lower:
            return self._handle_pide_query(user_input, entities)
        elif 'street food' in query_lower:
            return self._handle_street_food_query(user_input, entities)
        elif 'gluten-free' in query_lower or 'gluten free' in query_lower:
            return self._handle_gluten_free_query(user_input, entities)
        elif 'lactose-free' in query_lower or 'lactose free' in query_lower or 'dairy intolerance' in query_lower:
            return self._handle_lactose_free_query(user_input, entities)
        elif 'meyhane' in query_lower or ('traditional' in query_lower and 'music' in query_lower):
            return self._handle_meyhane_query(user_input, entities)
        elif 'waterfront' in query_lower and 'fish' in query_lower:
            return self._handle_waterfront_fish_query(user_input, entities)
        elif any(word in query_lower for word in ['diabetic', 'sugar-free', 'pregnant', 'pregnancy', 'wheelchair', 'accessible']):
            return self._handle_complex_dietary_query(user_input, entities)
        elif 'lactose-free' in query_lower or 'lactose free' in query_lower:
            return self._handle_lactose_free_query(user_input, entities)
        
        response = "🍽️ **Restaurant Recommendations for Istanbul**\n\n"
        
        # Enhanced location-specific recommendations with context awareness
        if detected_location:
            # Get detection method for more personalized messaging
            detection_method = context.get_context('location_detection_method') if context else 'unknown'
            
            if detection_method == 'explicit_query':
                response += f"🗺️ **Best restaurants in {detected_location} (as requested):**\n\n"
            elif detection_method == 'proximity_inference':
                response += f"🗺️ **Nearby restaurants in {detected_location} area:**\n\n"
            elif detection_method == 'user_profile':
                response += f"🗺️ **Restaurants near your location in {detected_location}:**\n\n"
            elif detection_method == 'context_memory':
                response += f"🗺️ **Restaurants in {detected_location} (from our conversation):**\n\n"
            elif detection_method == 'conversation_history':
                response += f"🗺️ **Restaurants in {detected_location} (based on your recent interests):**\n\n"
            elif detection_method == 'favorite_neighborhood':
                response += f"🗺️ **Restaurants in your favorite area, {detected_location}:**\n\n"
            else:
                response += f"🗺️ **Best restaurants in {detected_location}:**\n\n"
                
            district_name = detected_location
            
            if detected_location.lower() == 'sultanahmet':
                restaurants = [
                    ("Pandeli", "Historic Ottoman restaurant above Spice Bazaar", "Traditional Turkish", "$$"),
                    ("Hamdi Restaurant", "Famous for İskender kebab with Bosphorus view", "Turkish Grill", "$$"),
                    ("Seven Hills Restaurant", "Rooftop dining with Blue Mosque view", "Turkish/International", "$$$")
                ]
            elif district_name.lower() in ['beyoglu', 'galata']:
                restaurants = [
                    ("Mikla", "Award-winning modern Turkish cuisine", "Fine Dining", "$$$$"),
                    ("Karakoy Lokantasi", "Ottoman-inspired elegant dining", "Turkish", "$$$"),
                    ("Zubeyir Ocakbasi", "Authentic grill house loved by locals", "Turkish Grill", "$$")
                ]
            elif district_name.lower() == 'kadikoy':
                restaurants = [
                    ("Ciya Sofrasi", "Traditional Anatolian cuisine", "Turkish Regional", "$$"),
                    ("Kanaat Lokantasi", "Historic neighborhood restaurant", "Turkish Home Cooking", "$"),
                    ("Sur Balik", "Fresh seafood on Asian side", "Seafood", "$$$")
                ]
            elif district_name.lower() == 'taksim':
                restaurants = [
                    ("360 Istanbul", "Panoramic city views with international menu", "International", "$$$"),
                    ("Masa Restaurant", "Contemporary Turkish in stylish setting", "Modern Turkish", "$$$"),
                    ("Leb-i Derya", "Trendy rooftop with Bosphorus views", "Mediterranean", "$$$")
                ]
            elif district_name.lower() == 'besiktas':
                restaurants = [
                    ("Tugra Restaurant", "Ottoman palace cuisine", "Ottoman", "$$$$"),
                    ("Feriye Palace", "Historic waterfront dining", "Turkish", "$$$"),
                    ("Poseidon", "Fresh seafood by the Bosphorus", "Seafood", "$$$")
                ]
            elif district_name.lower() == 'sisli':
                restaurants = [
                    ("Spago", "International fine dining", "International", "$$$$"),
                    ("Vogue Restaurant", "Stylish dining with city views", "International", "$$$"),
                    ("Local Brasserie", "French-Turkish fusion", "Fusion", "$$$")
                ]
            elif district_name.lower() == 'nisantasi':
                restaurants = [
                    ("Nicole Restaurant", "Michelin-starred fine dining", "Fine Dining", "$$$$$"),
                    ("Seasons Restaurant", "Elegant Four Seasons dining", "International", "$$$$"),
                    ("House Cafe", "Trendy cafe with good food", "Cafe/International", "$$$")
                ]
            elif district_name.lower() == 'eminonu':
                restaurants = [
                    ("Pandeli", "Historic restaurant in Spice Bazaar", "Ottoman", "$$"),
                    ("Hamdi Restaurant", "Famous for Adana kebab", "Turkish Grill", "$$"),
                    ("Balık Pazarı", "Fresh fish market restaurants", "Seafood", "$$")
                ]
            elif district_name.lower() == 'levent':
                restaurants = [
                    ("Sunset Grill & Bar", "International with Bosphorus view", "International", "$$$"),
                    ("Ulus 29", "Upscale dining with panoramic views", "International", "$$$$"),
                    ("Park Fora", "Business district favorite", "Turkish/International", "$$$")
                ]
            else:
                # Provide realistic restaurant names for other districts
                restaurants = [
                    ("Lokanta Maya", "Contemporary Turkish cuisine", "Modern Turkish", "$$$"),
                    ("Çiya Sofrası", "Regional Anatolian specialties", "Turkish Regional", "$$"),
                    ("Pandeli", "Historic Ottoman dining", "Traditional Turkish", "$$")
                ]
        else:
            response += "🌟 **Top Istanbul Restaurant Picks:**\n\n"
            restaurants = [
                ("Nusr-Et Steakhouse", "Famous Turkish steakhouse", "Steakhouse", "$$$$"),
                ("Balikci Sabahattin", "Historic seafood institution", "Seafood", "$$$"),
                ("Develi", "Legendary kebab restaurant", "Turkish Grill", "$$"),
                ("Sunset Grill & Bar", "International cuisine with Bosphorus view", "International", "$$$")
            ]
        
        # Display restaurants
        for i, (name, desc, cuisine, price) in enumerate(restaurants, 1):
            response += f"**{i}. {name}** {price}\n"
            response += f"   {desc}\n"
            response += f"   🍽️ Cuisine: {cuisine}\n\n"
        
        # Add specific filters and handle special requirements
        filter_notes = []
        special_requirements = []
        
        # Check for specific timing requirements
        if '7 am' in query_lower or 'early breakfast' in query_lower:
            special_requirements.append("⏰ **Early Opening (7 AM):** Hotel restaurants, airport cafes, 24-hour diners")
            filter_notes.append("🌅 Early breakfast (7 AM opening)")
        
        # Check for family requirements
        if 'family' in query_lower or 'kids' in query_lower or 'children' in query_lower:
            special_requirements.append("👨‍👩‍👧‍👦 **Family Features:** High chairs, kids menu, spacious seating")
            filter_notes.append("👨‍👩‍👧‍👦 Family-friendly with kids amenities")
        
        # Check for outdoor seating
        if 'outdoor' in query_lower or 'terrace' in query_lower:
            special_requirements.append("🌳 **Outdoor Options:** Terrace seating, garden areas, street-side tables")
            filter_notes.append("🌳 Outdoor seating available")
        
        # Check for large groups
        if 'large group' in query_lower or 'big group' in query_lower:
            special_requirements.append("👥 **Group Dining:** Reservations recommended, private dining rooms available")
            filter_notes.append("👥 Suitable for large groups")
        
        # Check for delivery
        if 'deliver' in query_lower or 'takeout' in query_lower:
            special_requirements.append("🚚 **Delivery Options:** Yemeksepeti, Getir, Trendyol Yemek apps")
            filter_notes.append("🚚 Delivery service available")
        
        # Check for romantic settings
        if 'romantic' in query_lower:
            special_requirements.append("💕 **Romantic Features:** Intimate lighting, Bosphorus views, quiet atmosphere")
            filter_notes.append("💕 Romantic atmosphere")
        
        # Check for live music
        if 'live music' in query_lower or 'entertainment' in query_lower:
            special_requirements.append("🎵 **Entertainment:** Live Turkish music, traditional performances")
            filter_notes.append("🎵 Live music and entertainment")
        
        if cuisines:
            filter_notes.append(f"🥘 Specializing in {', '.join(cuisines)} cuisine")
        if dietary:
            filter_notes.append(f"🌱 With {', '.join(dietary)} options")
        if budget:
            budget_level = budget[0]
            if budget_level == 'budget':
                filter_notes.append("💰 Budget-friendly options ($ - $$)")
            elif budget_level == 'luxury':
                filter_notes.append("✨ Premium dining experiences ($$$$ - $$$$$)")
        if time_refs:
            if 'morning' in time_refs and '7 am' not in query_lower:
                filter_notes.append("🌅 Great for breakfast")
            elif 'evening' in time_refs:
                filter_notes.append("🌙 Perfect for dinner")
        
        if filter_notes:
            response += "**Your preferences:**\n" + "\n".join(f"• {note}" for note in filter_notes) + "\n\n"
        
        if special_requirements:
            response += "**Special Requirements:**\n" + "\n".join(f"• {req}" for req in special_requirements) + "\n\n"
        
        # Add helpful tips
        response += "💡 **Helpful Tips:**\n"
        response += "• Make reservations for upscale restaurants\n"
        response += "• Try 'meze' (appetizers) for authentic experience\n"
        response += "• Most restaurants open around 12:00 for lunch\n"
        response += "• Dinner typically starts after 19:00\n\n"
        
        response += "🗺️ Would you like directions to any of these restaurants or more specific recommendations?"
        
        return response

    def _handle_events_query(self, user_input: str, entities: Dict, user_profile: UserProfile) -> str:
        """Handle events-related queries with comprehensive fallback"""
        districts = entities.get('districts', [])
        time_refs = entities.get('time', [])
        
        # Try to use real events scheduler if available
        if self.events_available:
            try:
                events = self.events_scheduler.get_current_events()
                if events:
                    response = "🎭 **Current Events in Istanbul**\n\n"
                    for i, event in enumerate(events[:6], 1):
                        response += f"**{i}. {event.get('title', 'Event')}**\n"
                        if event.get('date'):
                            response += f"📅 {event['date']}\n"
                        if event.get('venue'):
                            response += f"📍 {event['venue']}\n"
                        if event.get('category'):
                            response += f"🎨 Category: {event['category']}\n"
                        response += "\n"
                    
                    response += "💡 **Tips:**\n"
                    response += "• Check venue websites for ticket availability\n"
                    response += "• Many venues are accessible by metro and bus\n"
                    response += "• Consider booking in advance for popular events\n\n"
                    response += "🗺️ Would you like directions to any venue or more event details?"
                    return response
            except Exception as e:
                logger.warning(f"Events integration error: {e}")
        
        # Enhanced fallback events response
        response = "🎭 **Cultural Events & Activities in Istanbul**\n\n"
        
        # Time-specific recommendations
        if 'now' in time_refs or 'today' in time_refs:
            response += "**Today's Highlights:**\n\n"
        elif 'weekend' in time_refs:
            response += "**Weekend Events:**\n\n"
        elif 'evening' in time_refs:
            response += "**Evening Events:**\n\n"
        else:
            response += "**Current & Upcoming Events:**\n\n"
        
        # Location-specific events
        if districts:
            district = districts[0].lower()
            if district == 'sultanahmet':
                events_list = [
                    ("Hagia Sophia Evening Tours", "Guided tours with special lighting", "Hagia Sophia Museum", "Historical"),
                    ("Ottoman Palace Concerts", "Classical music in historic setting", "Topkapi Palace", "Music"),
                    ("Blue Mosque Cultural Talks", "Islamic art and architecture", "Blue Mosque Complex", "Cultural")
                ]
            elif district == 'beyoglu':
                events_list = [
                    ("Contemporary Art Exhibitions", "Modern Turkish and international art", "Istanbul Modern", "Art"),
                    ("Galata Tower Night Shows", "Panoramic city views with entertainment", "Galata Tower", "Entertainment"),
                    ("İstiklal Street Performances", "Street artists and musicians", "İstiklal Avenue", "Street Art")
                ]
            else:
                events_list = [
                    ("Neighborhood Cultural Center", "Local performances and exhibitions", f"{district.title()} Cultural Center", "Mixed"),
                    ("Community Arts Festival", "Local artists and craftspeople", f"{district.title()} Square", "Festival"),
                    ("Traditional Music Evening", "Folk and classical Turkish music", f"{district.title()} Concert Hall", "Music")
                ]
        else:
            events_list = [
                ("İKSV Cultural Events", "International Istanbul festivals", "Various venues citywide", "International"),
                ("Istanbul Biennial", "Contemporary art from around the world", "Multiple locations", "Art"),
                ("Turkish State Opera", "Classical and modern opera performances", "Atatürk Cultural Center", "Opera"),
                ("Bosphorus Concert Series", "Outdoor concerts with water views", "Bosphorus venues", "Music"),
                ("Historic Peninsula Tours", "Archaeological and cultural walks", "Old City", "Cultural"),
                ("Whirling Dervish Ceremonies", "Traditional Sufi spiritual performances", "Cultural centers", "Spiritual")
            ]
        
        # Display events
        for i, (title, desc, venue, category) in enumerate(events_list, 1):
            response += f"**{i}. {title}**\n"
            response += f"   {desc}\n"
            response += f"   📍 Venue: {venue}\n"
            response += f"   🎨 Category: {category}\n\n"
        
        # Add helpful information
        response += "🎫 **Event Information:**\n"
        response += "• Most cultural events: 19:00-21:00\n"
        response += "• Museum exhibitions: 09:00-17:00 (closed Mondays)\n"
        response += "• Concert venues: Check specific showtimes\n"
        response += "• Many events offer English translations\n\n"
        
        response += "💡 **Getting There:**\n"
        response += "• Metro: Most venues accessible via M2 line\n"
        response += "• Tram: Convenient for Old City events\n"
        response += "• Ferry: Scenic route to Bosphorus venues\n\n"
        
        response += "🎟️ Would you like ticket information or directions to any venue?"
        
        return response

    def _handle_transportation_query(self, user_input: str, entities: Dict, user_profile: UserProfile) -> str:
        """Handle transportation-related queries with advanced AI and real-time data"""
        try:
            # Use advanced transportation system if available
            if ADVANCED_TRANSPORT_AVAILABLE and self.transport_processor:
                logger.info("🚇 Using advanced transportation system with IBB API")
                
                # Process query through enhanced system
                enhanced_response = self.transport_processor.process_transportation_query_sync(
                    user_input, entities, user_profile
                )
                
                if enhanced_response and enhanced_response.strip():
                    return enhanced_response
                    
            # Fallback to improved static response
            logger.info("🚇 Using fallback transportation system")
            return self._get_fallback_transportation_response(user_input, entities, user_profile)
            
        except Exception as e:
            logger.error(f"Transportation query error: {e}")
            return self._get_fallback_transportation_response(user_input, entities, user_profile)
    
    def _get_fallback_transportation_response(self, user_input: str, entities: Dict, user_profile: UserProfile) -> str:
        """Fallback transportation response with correct information"""
        districts = entities.get('districts', [])
        transport_modes = entities.get('transport', [])
        
        response = "🚇 **Transportation in Istanbul**\n\n"
        
        if districts:
            response += f"Getting to/from {', '.join(districts)}:\n\n"
        
        # Real-time status indicator
        current_time = datetime.now().strftime("%H:%M")
        response += f"📍 **Live Status** (Updated: {current_time})\n\n"
        
        response += "**Metro Lines:**\n"
        response += "• M1A: Yenikapı ↔ Halkalı (serves Aksaray, Grand Bazaar)\n"
        response += "• M2: Vezneciler ↔ Hacıosman (serves Taksim, Şişli, Maslak)\n"
        response += "• M3: Kirazlı ↔ Başakşehir (business districts)\n"
        response += "• M4: Kadıköy ↔ Tavşantepe (Asian side main line)\n"
        response += "• M7: Kabataş ↔ Mahmutbey (Golden Horn bridge line)\n"  
        response += "• M11: Gayrettepe ↔ IST Airport (new airport express)\n\n"
        
        response += "**Popular Routes:**\n"
        response += "• Airport to Taksim: M11 → M2 (45 min)\n"
        response += "• Sultanahmet to Asian side: T1 → M2 → M4 (30 min)\n"
        response += "• Grand Bazaar to Galata: M1A → M2 (20 min)\n\n"
        
        response += "**Other Transport:**\n"
        response += "• 🚌 İETT Bus: City-wide network\n"
        response += "• 🚋 Tram: T1 (Historic Peninsula), T4 (Topkapı)\n" 
        response += "• ⛴️ Ferry: Bosphorus & Golden Horn routes\n"
        response += "• 🚕 Taxi: BiTaksi, Uber available\n\n"
        
        response += "💳 **Payment:** İstanbulkart for all public transport\n"
        response += "💡 Need specific directions? Ask: 'How to get from X to Y?'"
        
        return response

    def _handle_attractions_query(self, user_input: str, entities: Dict, user_profile: UserProfile) -> str:
        """Handle attractions-related queries"""
        districts = entities.get('districts', [])
        
        response = "🏛️ **Istanbul Attractions**\n\n"
        
        if districts:
            response += f"Top attractions in {', '.join(districts)}:\n\n"
        else:
            response += "Must-visit attractions:\n\n"
        
        attractions = [
            ("Hagia Sophia", "Sultanahmet", "Historic Byzantine cathedral and Ottoman mosque"),
            ("Blue Mosque", "Sultanahmet", "Stunning Ottoman architecture with six minarets"),
            ("Topkapi Palace", "Sultanahmet", "Former Ottoman palace with treasury and views"),
            ("Galata Tower", "Galata", "Medieval tower with panoramic city views"),
            ("Grand Bazaar", "Beyazıt", "Historic covered market with 4,000 shops")
        ]
        
        for i, (name, district, desc) in enumerate(attractions, 1):
            response += f"**{i}. {name}** ({district})\n   {desc}\n\n"
        
        response += "📍 Would you like directions or opening hours for any of these?"
        
        return response

    def _handle_general_query(self, user_input: str, entities: Dict, user_profile: UserProfile) -> str:
        """Handle general queries"""
        return f"""Hello! 🌟 Welcome to Istanbul!

I can help you with:
• 🍽️ Restaurant recommendations
• 🎭 Current events and cultural activities  
• 🚇 Transportation and directions
• 🏛️ Attractions and sightseeing
• 🏘️ Neighborhood guides and district information
• 💡 Local tips and hidden gems

**Examples**:
• "Restaurants in Kadıköy"
• "What's the vibe in Beşiktaş?"
• "Complete guide to Sultanahmet"
• "Compare Şişli and Üsküdar"

What would you like to explore in Istanbul today?"""

    def _handle_neighborhood_guide_query(self, user_input: str, entities: Dict, user_profile: UserProfile, context: Optional['ConversationContext'] = None) -> str:
        """Handle neighborhood/district-specific queries with comprehensive guidance"""
        
        if not self.neighborhood_guide:
            return f"""🏘️ **Neighborhood Guide Service**

I'd love to help with neighborhood information, but the neighborhood guide service is currently not available.

**I can still help you with:**
• 🍽️ Restaurant recommendations by district
• 🎭 Events and cultural activities
• 🚇 Transportation information
• 🏛️ Attractions and sightseeing

Try asking: "Restaurants in [district name]" or "How to get to [location]" """
        
        user_input_lower = user_input.lower()
        
        # Find districts mentioned in the query
        matched_districts = self.neighborhood_guide.find_districts_by_keywords(user_input)
        
        # Enhanced district detection - try alternative spellings and patterns
        if not matched_districts:
            # Manual district detection for common variations
            district_patterns = {
                'kadıköy': ['kadikoy', 'kadıköy', 'kadiköy'],
                'beşiktaş': ['besiktas', 'beşiktaş', 'besiktaş'],
                'sultanahmet': ['sultanahmet'],
                'şişli': ['sisli', 'şişli', 'sisli'],
                'üsküdar': ['uskudar', 'üsküdar', 'uskudar'],
                'fatih': ['fatih'],
                'sarıyer': ['sariyer', 'sarıyer', 'sariyer']
            }
            
            for standard_name, variations in district_patterns.items():
                if any(variation in user_input_lower for variation in variations):
                    matched_districts = [standard_name]
                    break
        
        # Check for specific query types
        if any(word in user_input_lower for word in ['compare', 'difference', 'versus', 'vs', 'which is better']):
            return self._handle_district_comparison_query(user_input, matched_districts, user_profile)
        
        elif any(word in user_input_lower for word in ['character', 'vibe', 'atmosphere', 'feel', 'what is like']):
            return self._handle_district_character_query(user_input, matched_districts, user_profile)
        
        elif any(word in user_input_lower for word in ['how to get', 'transport', 'travel to', 'go to']):
            return self._handle_district_transport_query(user_input, matched_districts, user_profile)
        
        elif any(word in user_input_lower for word in ['restaurants', 'eat', 'food', 'dining']):
            return self._handle_district_dining_query(user_input, matched_districts, user_profile)
        
        elif any(word in user_input_lower for word in ['nightlife', 'bars', 'clubs', 'night']):
            return self._handle_district_nightlife_query(user_input, matched_districts, user_profile)
        
        elif any(word in user_input_lower for word in ['attractions', 'sights', 'visit', 'see']):
            return self._handle_district_attractions_query(user_input, matched_districts, user_profile)
        
        elif any(word in user_input_lower for word in ['shopping', 'shop', 'buy', 'mall']):
            return self._handle_district_shopping_query(user_input, matched_districts, user_profile)
        
        elif any(word in user_input_lower for word in ['guide', 'comprehensive', 'everything', 'complete']):
            return self._handle_comprehensive_district_guide(user_input, matched_districts, user_profile, context)
        
        else:
            # General district information
            return self._handle_general_district_info(user_input, matched_districts, user_profile)
    
    def _handle_district_comparison_query(self, user_input: str, districts: List[str], user_profile: UserProfile) -> str:
        """Handle district comparison queries"""
        if len(districts) < 2:
            return """🏘️ **District Comparison**

To compare neighborhoods, please mention at least two districts. For example:
• "Compare Beşiktaş and Kadıköy"
• "What's the difference between Sultanahmet and Fatih?"
• "Which is better for nightlife: Beşiktaş vs Şişli?"

Available districts: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer"""
        
        comparison = self.neighborhood_guide.get_district_comparison(districts)
        
        response = "🏘️ **District Comparison**\n\n"
        
        for district_name, info in comparison["districts"].items():
            response += f"**{district_name}**\n"
            response += f"• **Type**: {info['type'].replace('_', ' ').title()}\n"
            response += f"• **Vibe**: {info['vibe']}\n"
            response += f"• **Best Time**: {', '.join(info['best_time'])}\n"
            response += f"• **Budget**: {info['budget_estimate']}\n"
            response += f"• **Categories**: {', '.join(info['categories'])}\n\n"
        
        if comparison["summary"]:
            response += "**Quick Summary:**\n"
            for category, district_list in comparison["summary"].items():
                if district_list:
                    response += f"• **{category.replace('_', ' ').title()}**: {', '.join(district_list)}\n"
        
        return response
    
    def _handle_district_character_query(self, user_input: str, districts: List[str], user_profile: UserProfile) -> str:
        """Handle queries about district character and atmosphere"""
        if not districts:
            return """🏘️ **Neighborhood Character Guide**

Please specify a district to learn about its character. For example:
• "What's the vibe in Kadıköy?"
• "Tell me about Beşiktaş atmosphere"
• "What is Sultanahmet like?"

Available districts: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer"""
        
        response = "🏘️ **Neighborhood Character**\n\n"
        
        for district_key in districts:
            character = self.neighborhood_guide.get_neighborhood_character(district_key)
            if character:
                district = self.neighborhood_guide.get_district_guide(district_key)
                response += f"**{district.name}**\n"
                response += f"🎭 **Vibe**: {character.vibe}\n"
                response += f"👥 **Crowd**: {character.crowd}\n"
                response += f"🌟 **Atmosphere**: {character.atmosphere}\n"
                response += f"⏰ **Best Time**: {', '.join([t.value for t in character.best_time])}\n"
                
                if character.local_saying:
                    response += f"💬 **Local Saying**: {character.local_saying}\n"
                
                if character.insider_tip:
                    response += f"💡 **Insider Tip**: {character.insider_tip}\n"
                
                response += "\n"
        
        return response
    
    def _handle_district_transport_query(self, user_input: str, districts: List[str], user_profile: UserProfile) -> str:
        """Handle transportation queries for districts"""
        if not districts:
            return """🚇 **District Transportation Guide**

Please specify a district for transportation information. For example:
• "How to get to Beşiktaş?"
• "Transportation to Kadıköy"
• "Best way to reach Sultanahmet"

Available districts: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer"""
        
        response = "🚇 **Transportation Guide**\n\n"
        
        for district_key in districts:
            district = self.neighborhood_guide.get_district_guide(district_key)
            if district:
                response += f"**Getting to {district.name}:**\n"
                for transport_type, details in district.getting_there.items():
                    response += f"• **{transport_type.title()}**: {details}\n"
                response += "\n"
        
        return response
    
    def _handle_district_dining_query(self, user_input: str, districts: List[str], user_profile: UserProfile) -> str:
        """Handle dining queries for specific districts"""
        if not districts:
            return """🍽️ **District Dining Guide**

Please specify a district for restaurant recommendations. For example:
• "Restaurants in Kadıköy"
• "Where to eat in Beşiktaş?"
• "Best food in Sultanahmet"

Available districts: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer"""
        
        response = "🍽️ **District Dining Guide**\n\n"
        
        for district_key in districts:
            restaurants = self.neighborhood_guide.get_recommendations_by_category(district_key, "restaurants")
            if restaurants:
                district = self.neighborhood_guide.get_district_guide(district_key)
                response += f"**{district.name} Restaurants:**\n\n"
                
                for rest in restaurants:
                    response += f"**{rest.name}** ({rest.type})\n"
                    response += f"📍 {rest.address}\n"
                    response += f"📝 {rest.description}\n"
                    
                    if rest.opening_hours:
                        response += f"🕐 {rest.opening_hours}\n"
                    
                    if rest.insider_tip:
                        response += f"💡 **Tip**: {rest.insider_tip}\n"
                    
                    response += f"🏆 Local Favorite: {'Yes' if rest.local_favorite else 'No'}\n"
                    response += f"🌍 Tourist Friendly: {'Yes' if rest.tourist_friendly else 'No'}\n\n"
        
        return response
    
    def _handle_district_nightlife_query(self, user_input: str, districts: List[str], user_profile: UserProfile) -> str:
        """Handle nightlife queries for specific districts"""
        if not districts:
            return """🌙 **District Nightlife Guide**

Please specify a district for nightlife recommendations. For example:
• "Nightlife in Beşiktaş"
• "Bars in Kadıköy"
• "Clubs in Şişli"

Available districts: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer"""
        
        response = "🌙 **District Nightlife Guide**\n\n"
        
        for district_key in districts:
            nightlife = self.neighborhood_guide.get_recommendations_by_category(district_key, "nightlife")
            if nightlife:
                district = self.neighborhood_guide.get_district_guide(district_key)
                response += f"**{district.name} Nightlife:**\n\n"
                
                for venue in nightlife:
                    response += f"**{venue.name}** ({venue.type})\n"
                    response += f"📍 {venue.address}\n"
                    response += f"📝 {venue.description}\n"
                    
                    if venue.opening_hours:
                        response += f"🕐 {venue.opening_hours}\n"
                    
                    if venue.insider_tip:
                        response += f"💡 **Tip**: {venue.insider_tip}\n"
                    
                    response += f"🏆 Local Favorite: {'Yes' if venue.local_favorite else 'No'}\n\n"
            else:
                district = self.neighborhood_guide.get_district_guide(district_key)
                if district:
                    response += f"**{district.name}**: Limited nightlife options, more family-oriented district.\n\n"
        
        return response
    
    def _handle_district_attractions_query(self, user_input: str, districts: List[str], user_profile: UserProfile) -> str:
        """Handle attraction queries for specific districts"""
        if not districts:
            return """🏛️ **District Attractions Guide**

Please specify a district for attraction recommendations. For example:
• "Attractions in Sultanahmet"
• "What to see in Fatih?"
• "Sights in Üsküdar"

Available districts: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer"""
        
        response = "🏛️ **District Attractions Guide**\n\n"
        
        for district_key in districts:
            attractions = self.neighborhood_guide.get_recommendations_by_category(district_key, "attractions")
            if attractions:
                district = self.neighborhood_guide.get_district_guide(district_key)
                response += f"**{district.name} Attractions:**\n\n"
                
                for attr in attractions:
                    response += f"**{attr.name}** ({attr.type})\n"
                    response += f"📍 {attr.address}\n"
                    response += f"📝 {attr.description}\n"
                    
                    if attr.opening_hours:
                        response += f"🕐 {attr.opening_hours}\n"
                    
                    if attr.insider_tip:
                        response += f"💡 **Tip**: {attr.insider_tip}\n"
                    
                    response += f"🏆 Local Favorite: {'Yes' if attr.local_favorite else 'No'}\n\n"
        
        return response
    
    def _handle_district_shopping_query(self, user_input: str, districts: List[str], user_profile: UserProfile) -> str:
        """Handle shopping queries for specific districts"""
        if not districts:
            return """🛍️ **District Shopping Guide**

Please specify a district for shopping recommendations. For example:
• "Shopping in Şişli"
• "Where to shop in Fatih?"
• "Malls in Beşiktaş"

Available districts: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer"""
        
        response = "🛍️ **District Shopping Guide**\n\n"
        
        for district_key in districts:
            shopping = self.neighborhood_guide.get_recommendations_by_category(district_key, "shopping")
            if shopping:
                district = self.neighborhood_guide.get_district_guide(district_key)
                response += f"**{district.name} Shopping:**\n\n"
                
                for shop in shopping:
                    response += f"**{shop.name}** ({shop.type})\n"
                    response += f"📍 {shop.address}\n"
                    response += f"📝 {shop.description}\n"
                    
                    if shop.opening_hours:
                        response += f"🕐 {shop.opening_hours}\n"
                    
                    if shop.insider_tip:
                        response += f"💡 **Tip**: {shop.insider_tip}\n"
                    
                    response += f"🏆 Local Favorite: {'Yes' if shop.local_favorite else 'No'}\n\n"
            else:
                district = self.neighborhood_guide.get_district_guide(district_key)
                if district:
                    response += f"**{district.name}**: Limited shopping options, more focused on other activities.\n\n"
        
        return response
    
    def _handle_comprehensive_district_guide(self, user_input: str, districts: List[str], user_profile: UserProfile, context: Optional['ConversationContext'] = None) -> str:
        """Handle comprehensive district guide requests"""
        if not districts:
            return """📖 **Comprehensive District Guide**

Please specify a district for a complete guide. For example:
• "Complete guide to Kadıköy"
• "Everything about Beşiktaş"
• "Comprehensive Sultanahmet guide"

Available districts: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer"""
        
        # Use first district for comprehensive guide
        district_key = districts[0]
        
        # Get user interests from profile if available
        user_interests = None
        if hasattr(user_profile, 'preferences') and user_profile.preferences:
            user_interests = user_profile.preferences.get('interests', [])
        
        guide = self.neighborhood_guide.generate_comprehensive_guide(district_key, user_interests)
        
        if "error" in guide:
            return f"❌ {guide['error']}"
        
        response = f"📖 **Comprehensive Guide to {guide['district_name']}**\n\n"
        
        # Character
        character = guide['character']
        response += "🎭 **Character & Atmosphere**\n"
        response += f"• **Vibe**: {character['vibe']}\n"
        response += f"• **Crowd**: {character['crowd']}\n"
        response += f"• **Atmosphere**: {character['atmosphere']}\n"
        response += f"• **Best Time**: {', '.join(character['best_time'])}\n"
        
        if character['local_saying']:
            response += f"• **Local Saying**: {character['local_saying']}\n"
        
        if character['insider_tip']:
            response += f"• **Insider Tip**: {character['insider_tip']}\n"
        
        response += "\n"
        
        # Transportation
        response += "🚇 **Getting There**\n"
        for transport_type, details in guide['getting_there'].items():
            response += f"• **{transport_type.title()}**: {details}\n"
        response += "\n"
        
        # Recommendations (top picks)
        if guide['recommendations']:
            response += "🌟 **Top Recommendations**\n"
            for category, recs in guide['recommendations'].items():
                if recs:  # Only show categories with recommendations
                    response += f"\n**{category.title()}:**\n"
                    for rec in recs[:2]:  # Show top 2 per category
                        response += f"• **{rec['name']}**: {rec['description']}\n"
        
        # Practical information
        response += f"\n💰 **Budget Estimate**: {guide['budget_estimate']}\n"
        response += f"🛡️ **Safety**: {guide['safety_notes']}\n\n"
        
        # Quick tips
        if guide['practical_tips']:
            response += "💡 **Quick Tips**:\n"
            for tip in guide['practical_tips'][:3]:
                response += f"• {tip}\n"
        
        response += "\n🗺️ *For walking routes, local customs, and detailed recommendations, ask for specific categories!*"
        
        return response
    
    def _handle_general_district_info(self, user_input: str, districts: List[str], user_profile: UserProfile) -> str:
        """Handle general district information requests"""
        if not districts:
            available_districts = ", ".join(self.neighborhood_guide.get_all_districts())
            return f"""🏘️ **Istanbul Neighborhood Guide**

I can provide detailed information about Istanbul's neighborhoods!

**Available Districts**: {available_districts}

**Ask me about**:
• Character and atmosphere
• Transportation options  
• Restaurants and dining
• Nightlife and bars
• Attractions and sights
• Shopping areas
• Comprehensive guides
• District comparisons

**Examples**:
• "What's the vibe in Kadıköy?"
• "Restaurants in Beşiktaş"
• "Complete guide to Sultanahmet"
• "Compare Şişli and Üsküdar"

What would you like to know about Istanbul's neighborhoods?"""
        
        response = "🏘️ **District Information**\n\n"
        
        for district_key in districts:
            district = self.neighborhood_guide.get_district_guide(district_key)
            if district:
                response += f"**{district.name}** ({district.district_type.value.replace('_', ' ').title()})\n"
                response += f"🎭 {district.character.vibe}\n"
                response += f"💰 Budget: {district.budget_estimate}\n"
                response += f"📍 Categories: {', '.join(district.recommendations.keys())}\n\n"
        
        response += "💡 *Ask for specific information like 'restaurants in [district]' or 'nightlife in [district]' for detailed recommendations!*"
        
        return response

    def _enhance_multi_intent_response(self, user_input: str, intents: List[str], entities: Dict, user_profile: UserProfile) -> str:
        """Handle multi-intent queries with enhanced responses"""
        if not intents:
            return self._generate_fallback_response(user_input)
        
        primary_intent = intents[0]
        response = self._process_by_intent(user_input, primary_intent, entities, user_profile)
        
        # Add suggestions for other detected intents
        if len(intents) > 1:
            suggestions = []
            for intent in intents[1:]:
                if intent == "restaurant":
                    suggestions.append("🍽️ Would you also like restaurant recommendations?")
                elif intent == "events":
                    suggestions.append("🎭 Interested in current events and shows?")
                elif intent == "transportation":
                    suggestions.append("🚇 Need help with transportation?")
                elif intent == "attractions":
                    suggestions.append("🏛️ Want to explore more attractions?")
            
            if suggestions:
                response += "\n\n**You might also be interested in:**\n" + "\n".join(suggestions)
        
        return response

    def _get_or_request_gps_location(self, user_profile: UserProfile) -> Optional[Dict[str, float]]:
        """Get user's GPS location or request it"""
        if user_profile.gps_location:
            return user_profile.gps_location
        
        # For now, return None - in a real app, this would trigger location permission request
        return None

    def classify_intent(self, user_input: str) -> str:
        """Public interface for intent classification - used by external systems"""
        return self._enhance_intent_classification(user_input)
    
    def _enhance_intent_classification(self, user_input: str) -> str:
        """Enhanced intent classification with better complex query handling"""
        intent_signals = self.entity_recognizer.detect_intent_signals(user_input)
        primary_intent = self._determine_primary_intent(intent_signals)
        
        user_lower = user_input.lower()
        
        # Priority 1: Enhanced restaurant detection - include Turkish dining culture terms
        restaurant_keywords = [
            'restaurant', 'eat', 'food', 'meal', 'hungry', 'dine', 'dining', 'cuisine', 'breakfast', 'lunch', 'dinner',
            'pide', 'kebab', 'meze', 'coffee', 'dessert', 'baklava', 'turkish coffee', 'street food',
            'seafood', 'vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free', 'lactose-free', 'diabetic',
            'ottoman cuisine', 'anatolian', 'meyhane', 'döner', 'köfte', 'fish', 'bosphorus',
            'rooftop', 'romantic dinner', 'family-friendly', 'kids menu', 'delivery', 'takeout',
            'outdoor seating', 'live music', 'fine dining', 'budget', 'cheap', 'expensive', 'luxury',
            'business lunch', 'celebration', 'anniversary', 'waterfront', 'wi-fi', 'digital nomad',
            'parking', 'pet-friendly', 'wheelchair accessible', 'private dining', 'corporate',
            'unlimited çay', 'iftar', 'ramadan', 'çay', 'tea', 'wine pairing', 'alcohol',
            'cooking class', 'chef table', 'celebrity', 'palace', '24-hour', 'late night',
            'golden horn', 'fusion', 'organic', 'healthy', 'pregnant', 'sugar-free'
        ]
        
        # Check for restaurant keywords
        if any(word in user_lower for word in restaurant_keywords):
            return 'restaurant_query'
        
        # Priority 2: Events detection - but exclude dining-related cultural activities
        events_keywords = ['event', 'show', 'performance', 'exhibition', 'concert', 'happening', 'cultural']
        if any(word in user_lower for word in events_keywords):
            # Check if it's actually about dining culture (e.g., "meyhane with traditional music")
            dining_context = any(word in user_lower for word in ['restaurant', 'eat', 'food', 'dining', 'meyhane'])
            if dining_context:
                return 'restaurant_query'
            else:
                return 'events_query'
        
        # Priority 3: Attraction detection
        elif any(word in user_lower for word in ['visit', 'see', 'attraction', 'museum', 'palace', 'mosque', 'tower']):
            return 'attraction_query'
        
        # Priority 4: Transportation detection  
        elif any(word in user_lower for word in ['transport', 'metro', 'bus', 'taxi', 'how to get', 'directions']):
            return 'transportation_query'
        
        return primary_intent

    def _is_restaurant_query(self, user_input: str) -> bool:
        """Check if query is restaurant-related for backward compatibility"""
        restaurant_keywords = ['restaurant', 'eat', 'food', 'meal', 'hungry', 'dine', 'cuisine', 'breakfast', 'lunch', 'dinner']
        return any(keyword in user_input.lower() for keyword in restaurant_keywords)

    def _handle_coffee_dessert_query(self, user_input: str, entities: Dict) -> str:
        """Handle Turkish coffee and dessert queries"""
        return """🍽️ **Best Turkish Coffee & Desserts in Istanbul**

☕ **Top Turkish Coffee Spots:**

**1. Kurukahveci Mehmet Efendi** $
   Historic coffee roastery since 1871
   📍 Location: Eminönü, near Spice Bazaar
   ☕ Specialty: Traditional Turkish coffee

**2. Fazıl Bey'in Türk Kahvesi** $
   Authentic neighborhood coffee house
   📍 Location: Kadıköy
   ☕ Specialty: Hand-ground Turkish coffee

**3. Mandabatmaz** $
   Famous tiny coffee shop in Beyoğlu
   📍 Location: Olivia Passage, Beyoğlu
   ☕ Specialty: Strong Turkish coffee

🧁 **Best Dessert Places:**

**4. Koçak Baklava** $$
   Premium baklava since 1956
   🍯 Specialty: Pistachio baklava, künefe

**5. Güllüoğlu Baklava** $$
   Famous Gaziantep baklava house
   🍯 Specialty: Traditional baklava varieties

**6. Café Privato** $$
   Modern dessert cafe with Ottoman sweets
   🍮 Specialty: Muhallebi, rice pudding

💡 **Tips:**
• Turkish coffee is served with Turkish delight
• Best enjoyed slowly with good company
• Try baklava with pistachio from Gaziantep
• Many places open early for breakfast coffee

☕ Perfect spots for authentic Turkish coffee culture!"""

    def _handle_pide_query(self, user_input: str, entities: Dict) -> str:
        """Handle Turkish pide queries"""
        return """🍽️ **Best Turkish Pide Places in Istanbul**

🥖 **Authentic Pide Restaurants:**

**1. Develi Pide** $$
   Famous for Gaziantep-style pide
   📍 Location: Samatya & multiple locations
   🥩 Specialty: Lamb pide, cheese pide

**2. Borsam Taşfırın** $
   Traditional stone oven pide house
   📍 Location: Karaköy
   🔥 Specialty: Stone-baked pide varieties

**3. Hamdi Pide** $$
   Legendary pide restaurant
   📍 Location: Eminönü, near Golden Horn
   🧀 Specialty: Cheese pide, mixed pide

**4. Cennet Pide** $
   Local favorite for cheap good pide
   📍 Location: Various neighborhoods
   💰 Specialty: Budget-friendly pide options

**5. Çamlıca Pide** $$
   Family-run pide house
   📍 Location: Asian side locations
   🥩 Specialty: Meat pide, Turkish sausage pide

💡 **Pide Tips:**
• Best eaten fresh from stone ovens
• Try 'karışık' (mixed) pide for variety
• Pide shops usually open from lunch till late
• Served with fresh salad and ayran (yogurt drink)

🥖 Enjoy authentic Turkish pide experience!"""

    def _handle_street_food_query(self, user_input: str, entities: Dict) -> str:
        """Handle street food queries"""
        return """🍽️ **Best Street Food Spots in Istanbul**

🌯 **Top Street Food Areas:**

**1. Eminönü - Fish Sandwich Boats**
   Famous floating fish sandwich vendors
   📍 Location: Galata Bridge area
   🐟 Specialty: Fresh grilled fish sandwiches

**2. Taksim Çiçek Pasajı**
   Historic flower passage with meyhanes
   📍 Location: İstiklal Street
   🍺 Specialty: Meze, rakı, street musicians

**3. Kadıköy Market Area**
   Local street food paradise
   📍 Location: Kadıköy ferry area
   🥙 Specialty: Döner, kokoreç, midye dolma

**4. Galata Bridge Lower Level**
   Restaurant strip under the bridge
   📍 Location: Golden Horn
   🦐 Specialty: Fresh seafood, balık ekmek

**5. Ortaköy Square**
   Weekend street food hub
   📍 Location: Bosphorus waterfront
   🥔 Specialty: Kumpir (stuffed baked potato)

**Popular Street Foods:**
• 🌯 Döner kebab - Rotating meat wraps
• 🥪 Balık ekmek - Fish sandwiches
• 🦪 Midye dolma - Stuffed mussels
• 🥔 Kumpir - Loaded baked potatoes
• 🌭 Kokoreç - Grilled lamb intestines
• 🧀 Tost - Turkish grilled cheese

💡 **Street Food Tips:**
• Look for busy stalls with high turnover
• Peak times: lunch and evening
• Most vendors speak basic English
• Cash only at most street stalls

🌯 Experience authentic Istanbul street culture!"""

    def _handle_gluten_free_query(self, user_input: str, entities: Dict) -> str:
        """Handle gluten-free restaurant queries"""
        districts = entities.get('districts', [])
        location_text = f" in {districts[0]}" if districts else ""
        
        return f"""🍽️ **Gluten-Free Restaurants{location_text}**

🌾 **Gluten-Free Friendly Restaurants:**

**1. Cooklife**
   Dedicated gluten-free bakery & cafe
   📍 Location: Bebek, Bomonti
   ✅ Specialty: GF bread, pastries, meals

**2. Zencefil Restaurant**
   Vegetarian with many GF options
   📍 Location: Galata
   🥗 Specialty: Organic, gluten-free dishes

**3. Seasons Restaurant (Four Seasons)**
   Upscale dining with GF menu
   📍 Location: Sultanahmet
   ⭐ Specialty: International GF fine dining

**4. House Cafe**
   Popular chain with GF options
   📍 Location: Multiple locations
   🍰 Specialty: GF desserts, salads

**5. Mikla Restaurant**
   Award-winning with GF accommodations
   📍 Location: Beyoğlu
   🏆 Specialty: Modern Turkish GF dishes

**Turkish GF Options:**
• 🍖 Grilled meats (without marinade)
• 🥗 Fresh salads and vegetables
• 🧀 Turkish cheese varieties
• 🫒 Olive oil dishes
• 🍯 Rice-based desserts (avoid wheat-based)

💡 **GF Tips in Istanbul:**
• Learn key phrases: "Gluten yok" (no gluten)
• Many traditional dishes naturally GF
• Check rice vs bulgur in dishes
• Ask about cross-contamination
• Hotel concierges can help translate dietary needs

🌾 Safe and delicious gluten-free dining in Istanbul!"""

    def _handle_lactose_free_query(self, user_input: str, entities: Dict) -> str:
        """Handle lactose-free and dairy intolerance queries"""
        districts = entities.get('districts', [])
        location_text = f" in {districts[0]}" if districts else ""
        
        return f"""🍽️ **Lactose-Free Restaurants{location_text}**

🥛 **Dairy-Free Friendly Options:**

**1. Zencefil Restaurant**
   Vegetarian with extensive dairy-free menu
   📍 Location: Galata
   🌱 Specialty: Organic, plant-based dishes

**2. Cooklife**
   Health-focused with fresh preparations
   📍 Location: Bebek, Bomonti
   ✅ Specialty: Fresh juices, wholesome meals

**3. Pandora Bookstore Cafe**
   Cultural cafe with dairy alternatives
   📍 Location: Beyoğlu
   📚 Specialty: Oat milk coffee, vegan pastries

**4. Neolokal**
   Modern Turkish with health-conscious preparations
   📍 Location: Galata
   🍽️ Specialty: Lactose-free Turkish cuisine adaptations

**5. Seasons Restaurant (Four Seasons)**
   Upscale dining with special dietary menus
   📍 Location: Sultanahmet
   ⭐ Specialty: Custom lactose-free fine dining

**Turkish Lactose-Free Options:**
• 🥩 Grilled meats without dairy marinades
• 🥗 Fresh vegetable dishes with olive oil
• 🍚 Rice-based dishes (pilav varieties)
• 🫒 Traditional olive oil mezze
• 🥜 Nuts and dried fruits

💡 **Lactose-Free Tips:**
• Learn to say: "Süt ürünü yok" (no dairy products)
• Turkish cuisine has many naturally dairy-free dishes
• Ask about cooking oils (some use butter)
• Traditional Turkish coffee is naturally dairy-free
• Many restaurants can accommodate with advance notice

🥛 Safe and delicious dairy-free dining in Istanbul!"""

    def _handle_meyhane_query(self, user_input: str, entities: Dict) -> str:
        """Handle Turkish meyhane (tavern) queries with traditional music"""
        return """🍽️ **Authentic Turkish Meyhanes with Traditional Music**

🎵 **Traditional Meyhanes:**

**1. Nevizade Sokak Meyhanes**
   Historic meyhane street in Beyoğlu
   📍 Location: Nevizade Street, Beyoğlu
   🎼 Specialty: Live fasıl music, classic meze

**2. Çiçek Pasajı (Flower Passage)**
   Historic covered passage with multiple meyhanes
   📍 Location: İstiklal Street, Beyoğlu
   🌸 Specialty: Traditional atmosphere, street musicians

**3. Refik Restaurant**
   Century-old family meyhane
   📍 Location: Tünel, Beyoğlu
   🏛️ Specialty: Historic setting, live Turkish music

**4. Sofyalı 9**
   Modern take on traditional meyhane
   📍 Location: Asmalımescit, Beyoğlu
   🎭 Specialty: Contemporary atmosphere, quality meze

**5. Yakup 2**
   Authentic neighborhood meyhane
   📍 Location: Asmalımescit, Beyoğlu
   🍷 Specialty: Traditional rakı culture, live music

**6. Pandeli**
   Historic Ottoman meyhane atmosphere
   📍 Location: Eminönü, Spice Bazaar
   🏺 Specialty: Ottoman-era ambiance, traditional service

**Meyhane Experience:**
• 🥃 **Rakı** - Traditional anise-flavored spirit
• 🧄 **Meze** - Small appetizer dishes (20+ varieties)
• 🎵 **Fasıl** - Traditional Turkish folk music
• 🕘 **Late dining** - Usually starts after 8 PM
• 👥 **Social dining** - Best experienced with friends

💡 **Meyhane Tips:**
• Order multiple meze to share
• Rakı is traditionally mixed with water
• Music gets livelier as the night progresses
• Reservations recommended for weekends
• Learn some Turkish songs for sing-alongs!

🎵 Experience authentic Turkish tavern culture!"""

    def _handle_waterfront_fish_query(self, user_input: str, entities: Dict) -> str:
        """Handle waterfront dining with fresh fish queries"""
        return """🍽️ **Waterfront Dining with Fresh Bosphorus Fish**

🐟 **Best Waterfront Fish Restaurants:**

**1. Lacivert Restaurant**
   Upscale waterfront dining with Bosphorus views
   📍 Location: Anadolu Hisarı, Asian side
   🌊 Specialty: Fresh daily catch, panoramic water views

**2. Poseidon Restaurant**
   Historic fish restaurant by the water
   📍 Location: Bebek, Bosphorus shore
   🦐 Specialty: Grilled fish, waterfront terrace

**3. Balıkçı Sabahattin**
   Traditional fish house since 1927
   📍 Location: Sultanahmet (historic fish specialist)
   🐟 Specialty: Classic Turkish fish preparations

**4. Sur Balık**
   Fresh seafood on the Asian side
   📍 Location: Kadıköy waterfront
   🦑 Specialty: Daily catch from local fishermen

**5. Galata Bridge Fish Restaurants**
   Multiple restaurants under the bridge
   📍 Location: Golden Horn waterfront
   🌉 Specialty: Casual dining with water views

**6. Feriye Palace Restaurant**
   Ottoman palace turned restaurant
   📍 Location: Ortaköy, Bosphorus shore
   🏰 Specialty: Elegant waterfront dining, fresh fish

**Fresh Fish Experience:**
• 🐟 **Daily catch** displayed on ice
• 🔥 **Grilled whole fish** - Turkish specialty
• 🥗 **Seasonal salads** and mezze starters
• 🍷 **Turkish white wines** pair perfectly
• 🌅 **Sunset dining** on terraces

💡 **Waterfront Fish Tips:**
• Ask to see the daily catch selection
• Grilled levrek (sea bass) and çipura (sea bream) are excellent
• Order rakı or white wine with fish
• Best times: lunch or early dinner for views
• Reservations essential for waterfront tables

🌊 Fresh from the Bosphorus to your plate!"""

    def _handle_complex_dietary_query(self, user_input: str, entities: Dict) -> str:
        """Handle complex dietary restrictions and medical needs"""
        query_lower = user_input.lower()
        
        if 'diabetic' in query_lower or 'sugar-free' in query_lower:
            return self._handle_diabetic_query(user_input, entities)
        elif 'pregnant' in query_lower or 'pregnancy' in query_lower:
            return self._handle_pregnancy_query(user_input, entities)
        elif 'wheelchair' in query_lower or 'accessible' in query_lower:
            return self._handle_accessibility_query(user_input, entities)
        else:
            return self._handle_restaurant_query(user_input, entities, None)

    def _handle_diabetic_query(self, user_input: str, entities: Dict) -> str:
        """Handle diabetic and sugar-free dietary needs"""
        return """🍽️ **Diabetic-Friendly & Sugar-Free Restaurants**

🩺 **Diabetic-Safe Dining Options:**

**1. Zencefil Restaurant**
   Organic vegetarian with sugar-free options
   📍 Location: Galata
   🥗 Specialty: Fresh salads, sugar-free desserts

**2. Cooklife**
   Health-focused with diabetic-friendly menu
   📍 Location: Bebek, Bomonti
   ✅ Specialty: Sugar-free baked goods, low-carb options

**3. Seasons Restaurant (Four Seasons)**
   Fine dining with medical dietary accommodation
   📍 Location: Sultanahmet
   ⭐ Specialty: Custom diabetic-friendly tasting menus

**4. Neolokal**
   Modern Turkish with health-conscious preparations
   📍 Location: Galata
   🍽️ Specialty: Fresh, unprocessed Turkish ingredients

**Diabetic-Safe Turkish Foods:**
• 🥩 **Grilled meats** - No sugar marinades
• 🥗 **Fresh vegetables** - Olive oil preparations
• 🧀 **Turkish cheese** - Natural, unprocessed
• 🫒 **Olives and nuts** - Healthy snacks
• 🥬 **Salads** - Ask for dressing on the side

⚠️ **Foods to Avoid:**
• Turkish desserts (baklava, künefe)
• Sweetened Turkish tea/coffee
• Fruit juices and sodas
• Honey-glazed dishes

💡 **Diabetic Dining Tips:**
• Learn: "Şekersiz" (sugar-free) and "Az şekerli" (low sugar)
• Always ask about hidden sugars in sauces
• Stick to grilled/steamed preparations
• Carry glucose meter for monitoring
• Hotel concierge can help explain dietary needs

🩺 Safe and delicious dining for diabetic guests!"""

    def _handle_pregnancy_query(self, user_input: str, entities: Dict) -> str:
        """Handle pregnancy-safe dining options"""
        return """🍽️ **Pregnancy-Safe Healthy Restaurants**

🤱 **Pregnancy-Friendly Dining:**

**1. Zencefil Restaurant**
   Organic vegetarian with fresh ingredients
   📍 Location: Galata
   🥗 Specialty: Organic vegetables, no processed foods

**2. Cooklife**
   Health-focused with fresh preparations
   📍 Location: Bebek, Bomonti
   ✅ Specialty: Fresh juices, wholesome meals

**3. Seasons Restaurant (Four Seasons)**
   High-end dining with quality control
   📍 Location: Sultanahmet
   ⭐ Specialty: Premium ingredients, safe preparations

**4. House Cafe**
   Modern cafe with fresh options
   📍 Location: Multiple locations
   ☕ Specialty: Fresh salads, reliable food safety

**Pregnancy-Safe Turkish Foods:**
• 🥩 **Well-cooked meats** - Avoid rare preparations
• 🥗 **Fresh vegetables** - Well-washed salads
• 🍞 **Fresh bread** - Traditional Turkish bread
• 🧀 **Pasteurized cheese** - Turkish white cheese
• 🍵 **Herbal teas** - Avoid excessive caffeine

⚠️ **Foods to Avoid:**
• Raw or undercooked fish
• Unpasteurized dairy products
• Raw eggs in preparations
• Excessive caffeine (Turkish coffee)
• Unwashed fruits/vegetables

💡 **Pregnancy Dining Tips:**
• Choose restaurants with good hygiene standards
• Ask for meat to be well-cooked
• Avoid street food during pregnancy
• Drink bottled water
• Fresh fruit juices are great vitamin sources

🤱 Nutritious and safe dining for expecting mothers!"""

    def _handle_accessibility_query(self, user_input: str, entities: Dict) -> str:
        """Handle wheelchair accessibility and mobility needs"""
        return """🍽️ **Wheelchair Accessible Restaurants**

♿ **Fully Accessible Dining Options:**

**1. Seasons Restaurant (Four Seasons)**
   5-star hotel with full accessibility
   📍 Location: Sultanahmet
   ♿ Features: Wheelchair ramps, accessible restrooms

**2. Mikla Restaurant**
   Modern restaurant with accessibility features
   📍 Location: Beyoğlu
   🏢 Features: Elevator access, wide aisles

**3. Sunset Grill & Bar**
   Upscale dining with accessibility
   📍 Location: Ulus
   🌅 Features: Ground level access, accessible parking

**4. Mall Restaurants (Kanyon, Zorlu Center)**
   Shopping mall dining with full accessibility
   📍 Location: Levent, Beşiktaş
   🛒 Features: Wheelchair access, accessible facilities

**5. Hotel Restaurants**
   Most 4-5 star hotel restaurants are accessible
   📍 Location: Throughout Istanbul
   🏨 Features: Ramps, wide doorways, accessible restrooms

**Accessibility Features to Look For:**
• ♿ **Wheelchair ramps** - Entrance access
• 🚪 **Wide doorways** - Easy navigation
• 🚻 **Accessible restrooms** - Proper facilities
• 🅿️ **Accessible parking** - Close to entrance
• 🛗 **Elevator access** - For upper floors

💡 **Accessibility Tips:**
• Call ahead to confirm accessibility features
• Hotel restaurants usually have best access
• Newer restaurants generally more accessible
• Shopping mall restaurants are reliably accessible
• Tourist areas have better accessibility compliance

♿ Comfortable and accessible dining experiences!"""

    def _detect_user_location(self, user_input: str, user_profile: UserProfile, context: 'ConversationContext') -> Optional[str]:
        """Intelligently detect user's current location using advanced location detection service"""
        
        # Use advanced location detector if available
        if self.advanced_services_available and self.location_detector:
            try:
                result = self.location_detector.detect_location(user_input, user_profile, context)
                
                if result.location:
                    # Update context with detection method and metadata
                    context.set_context('current_detected_location', result.location)
                    context.set_context('location_detection_method', result.detection_method)
                    context.set_context('location_confidence', result.confidence)
                    context.set_context('location_fallbacks', result.fallback_locations)
                    
                    # Update user profile if it's an explicit mention
                    if result.detection_method == 'explicit_query':
                        user_profile.update_location(result.location)
                    
                    logger.info(f"Advanced location detection: {result.location} (method: {result.detection_method}, confidence: {result.confidence})")
                    return result.location
                else:
                    logger.info("Advanced location detector found no definitive location")
                    return None
                    
            except Exception as e:
                logger.warning(f"Advanced location detection failed, falling back to basic detection: {e}")
        
        # Fallback to basic location detection
        return self._detect_user_location_basic(user_input, user_profile, context)

    def _detect_user_location_basic(self, user_input: str, user_profile: UserProfile, context: 'ConversationContext') -> Optional[str]:
        """Basic location detection fallback method"""
        
        # Priority 1: Explicit location mentioned in current query
        entities = self.entity_recognizer.extract_entities(user_input)
        districts = entities.get('districts', [])
        if districts:
            location = districts[0]
            user_profile.update_location(location)  # Update user profile
            context.set_context('current_detected_location', location)
            context.set_context('location_detection_method', 'explicit_query')
            logger.info(f"Location detected from query: {location}")
            return location
        
        # Enhanced contextual location detection for proximity indicators
        proximity_indicators = ['nearby', 'around here', 'close by', 'walking distance', 'in the area']
        if any(indicator in user_input.lower() for indicator in proximity_indicators):
            # Look for most recent location mention in context
            recent_location = self._get_most_recent_location_from_context(context)
            if recent_location:
                context.set_context('current_detected_location', recent_location)
                context.set_context('location_detection_method', 'proximity_inference')
                logger.info(f"Location inferred from proximity indicator: {recent_location}")
                return recent_location
        
        # Priority 2: Check user profile current location
        if user_profile.current_location:
            context.set_context('location_detection_method', 'user_profile')
            logger.info(f"Using user profile location: {user_profile.current_location}")
            return user_profile.current_location
        
        # Priority 3: Check context memory for previous location mentions
        if context.get_context('current_detected_location'):
            location = context.get_context('current_detected_location')
            context.set_context('location_detection_method', 'context_memory')
            logger.info(f"Using context location: {location}")
            return location
        
        # Priority 4: Analyze recent conversation history with weighted recency
        recent_locations = self._extract_locations_from_history_weighted(context.conversation_history)
        if recent_locations:
            # Use most recent or frequently mentioned location with recency bias
            best_location = self._select_best_location_from_history(recent_locations)
            if best_location:
                context.set_context('current_detected_location', best_location)
                context.set_context('location_detection_method', 'conversation_history')
                logger.info(f"Using location from weighted history analysis: {best_location}")
                return best_location
        
        # Priority 5: Check user's favorite neighborhoods with preference weighting
        if user_profile.favorite_neighborhoods:
            # Consider user type and preferences to select best favorite neighborhood
            location = self._select_best_favorite_neighborhood(user_profile)
            context.set_context('location_detection_method', 'favorite_neighborhood')
            logger.info(f"Using selected favorite neighborhood: {location}")
            return location
        
        # Priority 6: Use GPS coordinates if available to determine nearest district
        if user_profile.gps_location:
            nearest_district = self._get_nearest_district_from_gps(user_profile.gps_location)
            if nearest_district:
                context.set_context('location_detection_method', 'gps_coordinates')
                logger.info(f"Using GPS-derived location: {nearest_district}")
                return nearest_district
        
        logger.info("No location detected, will provide general recommendations")
        return None

    def _get_most_recent_location_from_context(self, context: 'ConversationContext') -> Optional[str]:
        """Get the most recently mentioned location from conversation context"""
        # Check last 5 interactions for location mentions
        recent_history = context.conversation_history[-5:] if len(context.conversation_history) > 5 else context.conversation_history
        
        for interaction in reversed(recent_history):  # Start from most recent
            user_input = interaction.get('user_input', '')
            system_response = interaction.get('system_response', '')
            
            # Check user input first
            entities = self.entity_recognizer.extract_entities(user_input)
            districts = entities.get('districts', [])
            if districts:
                return districts[0]
            
            # Check system response for location patterns
            for district in self._get_known_districts():
                if district.lower() in system_response.lower():
                    return district
        
        return None

    def _extract_locations_from_history_weighted(self, conversation_history: List[Dict]) -> List[Dict]:
        """Extract locations from history with recency weighting"""
        weighted_locations = []
        # Look at last 10 interactions with decreasing weight for older interactions
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        
        for i, interaction in enumerate(reversed(recent_history)):
            weight = 1.0 - (i * 0.1)  # Recent interactions have higher weight
            user_input = interaction.get('user_input', '')
            system_response = interaction.get('system_response', '')
            
            # Extract from user input
            entities = self.entity_recognizer.extract_entities(user_input)
            districts = entities.get('districts', [])
            for district in districts:
                weighted_locations.append({
                    'location': district,
                    'weight': weight,
                    'source': 'user_input',
                    'interaction_index': len(recent_history) - i - 1
                })
            
            # Extract from system response
            for district in self._get_known_districts():
                if district.lower() in system_response.lower():
                    weighted_locations.append({
                        'location': district,
                        'weight': weight * 0.8,  # System mentions have slightly lower weight
                        'source': 'system_response',
                        'interaction_index': len(recent_history) - i - 1
                    })
        
        return weighted_locations

    def _select_best_location_from_history(self, weighted_locations: List[Dict]) -> Optional[str]:
        """Select the best location from weighted history analysis"""
        if not weighted_locations:
            return None
        
        # Calculate total weight for each location
        location_scores = {}
        for loc_data in weighted_locations:
            location = loc_data['location']
            weight = loc_data['weight']
            
            if location not in location_scores:
                location_scores[location] = {
                    'total_weight': 0,
                    'mention_count': 0,
                    'most_recent_index': -1
                }
            
            location_scores[location]['total_weight'] += weight
            location_scores[location]['mention_count'] += 1
            location_scores[location]['most_recent_index'] = max(
                location_scores[location]['most_recent_index'],
                loc_data['interaction_index']
            )
        
        # Select location with highest combined score (weight + recency + frequency)
        best_location = None
        best_score = 0
        
        for location, score_data in location_scores.items():
            # Combined score: total weight + recency bonus + frequency bonus
            combined_score = (
                score_data['total_weight'] +
                (score_data['most_recent_index'] * 0.1) +  # Recency bonus
                (score_data['mention_count'] * 0.2)  # Frequency bonus
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_location = location
        
        return best_location

    def _select_best_favorite_neighborhood(self, user_profile: UserProfile) -> str:
        """Select the best favorite neighborhood based on user preferences and context"""
        if not user_profile.favorite_neighborhoods:
            return None
        
        # For now, use first favorite, but could be enhanced with:
        # - User type preferences (tourists prefer tourist areas, locals prefer authentic areas)
        # - Time of day (nightlife areas for evening, business areas for lunch)
        # - Query type (cultural areas for culture queries, food areas for restaurant queries)
        
        primary_favorite = user_profile.favorite_neighborhoods[0]
        
        # Add some intelligence based on user type
        if hasattr(user_profile, 'user_type'):
            if user_profile.user_type == UserType.TOURIST:
                # Tourists might prefer more accessible/famous areas
                tourist_friendly = ['Sultanahmet', 'Taksim', 'Beyoğlu', 'Galata']
                for neighborhood in user_profile.favorite_neighborhoods:
                    if neighborhood in tourist_friendly:
                        return neighborhood
            elif user_profile.user_type == UserType.LOCAL:
                # Locals might prefer authentic neighborhood experiences
                authentic_areas = ['Kadıköy', 'Beşiktaş', 'Balat', 'Fener', 'Cihangir']
                for neighborhood in user_profile.favorite_neighborhoods:
                    if neighborhood in authentic_areas:
                        return neighborhood
        
        return primary_favorite

    def _get_known_districts(self) -> List[str]:
        """Get list of known Istanbul districts"""
        return [
            'Sultanahmet', 'Beyoğlu', 'Galata', 'Taksim', 'Kadıköy', 'Beşiktaş', 
            'Şişli', 'Nişantaşı', 'Levent', 'Etiler', 'Ortaköy', 'Üsküdar',
            'Eminönü', 'Karaköy', 'Cihangir', 'Asmalımescit', 'Arnavutköy',
            'Bebek', 'Bostancı', 'Fenerbahçe', 'Moda', 'Balat', 'Fener'
        ]

    def _get_nearest_district_from_gps(self, gps_coords: Dict[str, float]) -> Optional[str]:
        """Determine nearest district from GPS coordinates"""
        lat, lng = gps_coords.get('lat'), gps_coords.get('lng')
        if not lat or not lng:
            return None
        
        # Istanbul district center coordinates (approximate)
        district_coords = {
            'Sultanahmet': {'lat': 41.0086, 'lng': 28.9802},
            'Beyoğlu': {'lat': 41.0362, 'lng': 28.9773},
            'Taksim': {'lat': 41.0370, 'lng': 28.9850},
            'Kadıköy': {'lat': 40.9833, 'lng': 29.0333},
            'Beşiktaş': {'lat': 41.0422, 'lng': 29.0097},
            'Galata': {'lat': 41.0256, 'lng': 28.9744},
            'Levent': {'lat': 41.0766, 'lng': 29.0092},
            'Şişli': {'lat': 41.0608, 'lng': 28.9866}
        }
        
        min_distance = float('inf')
        nearest_district = None
        
        for district, coords in district_coords.items():
            # Simple distance calculation (Euclidean)
            distance = ((lat - coords['lat'])**2 + (lng - coords['lng'])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                nearest_district = district
        
        return nearest_district

    def _process_with_multi_intent(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> str:
        """Process message using multi-intent handler with location detector integration"""
        try:
            # PRIORITY CHECK: Neighborhood guide queries should be handled first
            user_input_lower = user_input.lower()
            entities = self.entity_recognizer.extract_entities(user_input)
            matched_districts = []
            
            if self.neighborhood_guide:
                matched_districts = self.neighborhood_guide.find_districts_by_keywords(user_input)
                
                # Enhanced neighborhood detection
                neighborhood_keywords = [
                    'neighborhood', 'district', 'area', 'guide', 'vibe', 'atmosphere', 'character',
                    'compare', 'comparison', 'vs', 'versus', 'which is better',
                    'complete guide', 'comprehensive guide', 'everything about',
                    'besiktas', 'beşiktaş', 'kadikoy', 'kadıköy', 'sultanahmet', 
                    'sisli', 'şişli', 'uskudar', 'üsküdar', 'fatih', 'sariyer', 'sarıyer'
                ]
                
                district_context_keywords = [
                    f"{query_type} in" for query_type in ['restaurants', 'nightlife', 'attractions', 'shopping', 'food', 'bars', 'clubs', 'sights']
                ]
                
                has_neighborhood_context = (
                    any(word in user_input_lower for word in neighborhood_keywords) or
                    any(phrase in user_input_lower for phrase in district_context_keywords) or
                    matched_districts or
                    any(f"in {district}" in user_input_lower for district in ['kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'sultanahmet', 'sisli', 'şişli', 'uskudar', 'üsküdar', 'fatih', 'sariyer', 'sarıyer'])
                )
                
                if has_neighborhood_context:
                    logger.info(f"🏘️ Priority routing to neighborhood guide for query with districts: {matched_districts}")
                    return self._handle_neighborhood_guide_query(user_input, entities, user_profile, context)
            
            # Create context for multi-intent handler
            multi_intent_context = {
                'user_id': user_profile.user_id,
                'session_id': context.session_id,
                'conversation_history': context.get_recent_interactions(5),
                'user_preferences': user_profile.preferences,
                'location': getattr(user_profile, 'current_location', None)
            }
            
            # Analyze query with multi-intent handler
            multi_intent_result = self.multi_intent_handler.analyze_query(user_input, multi_intent_context)
            
            logger.info(f"🎯 Multi-intent analysis: Primary={multi_intent_result.primary_intent.type.value}, "
                       f"Secondary={[i.type.value for i in multi_intent_result.secondary_intents]}")
            
            # Process based on primary intent with location integration
            if multi_intent_result.primary_intent.type.value in ['location_search', 'route_planning']:
                return self._handle_location_intent(user_input, multi_intent_result, user_profile, context)
            elif multi_intent_result.primary_intent.type.value == 'recommendation':
                return self._handle_recommendation_intent(user_input, multi_intent_result, user_profile, context)
            elif multi_intent_result.primary_intent.type.value == 'information_request':
                return self._handle_information_intent(user_input, multi_intent_result, user_profile, context)
            else:
                # Handle other intents or use execution plan
                return self._execute_multi_intent_plan(multi_intent_result, user_profile, context)
            
        except Exception as e:
            logger.error(f"Multi-intent processing failed: {e}")
            return self._process_traditional(user_input, user_profile, context)

    def _handle_location_intent(self, user_input: str, multi_intent_result, user_profile: UserProfile, context: ConversationContext) -> str:
        """Handle location-related intents with intelligent location detection"""
        
        # Use intelligent location detector for enhanced location detection
        if self.advanced_services_available and self.location_detector:
            location_result = self.location_detector.detect_location(user_input, user_profile, context)
            
            if location_result:
                # Update context with detected location
                context.set_context('current_detected_location', location_result.location)
                context.set_context('location_confidence', location_result.confidence)
                context.set_context('detection_method', location_result.detection_method)
                
                # Check for transportation context
                if hasattr(location_result, 'context_match') and location_result.context_match:
                    if 'transportation' in location_result.context_match:
                        return self._handle_transportation_query_enhanced(user_input, location_result, user_profile, context)
                
                # Handle regular location search
                return self._handle_location_search_enhanced(user_input, location_result, user_profile, context)
        
        # Fallback to traditional location handling
        entities = self.entity_recognizer.extract_entities(user_input)
        return self._handle_restaurant_query(user_input, entities, user_profile, context)

    def _handle_recommendation_intent(self, user_input: str, multi_intent_result, user_profile: UserProfile, context: ConversationContext) -> str:
        """Handle recommendation intents with location awareness"""
        
        # Detect location for recommendation context
        detected_location = None
        if self.advanced_services_available and self.location_detector:
            location_result = self.location_detector.detect_location(user_input, user_profile, context)
            if location_result:
                detected_location = location_result.location
        
        # Extract entities for recommendation processing
        entities = self.entity_recognizer.extract_entities(user_input)
        
        # Update entities with detected location if available
        if detected_location and 'districts' not in entities:
            entities['districts'] = [detected_location]
        
        # Process as enhanced restaurant query
        return self._handle_restaurant_query(user_input, entities, user_profile, context)

    def _handle_information_intent(self, user_input: str, multi_intent_result, user_profile: UserProfile, context: ConversationContext) -> str:
        """Handle information requests"""
        
        # Check if it's location-related information
        if any(intent.type.value in ['location_search', 'route_planning'] for intent in multi_intent_result.secondary_intents):
            return self._handle_location_intent(user_input, multi_intent_result, user_profile, context)
        
        # Handle general information requests
        entities = self.entity_recognizer.extract_entities(user_input)
        return self._handle_general_query(user_input, entities, user_profile)

    def _execute_multi_intent_plan(self, multi_intent_result, user_profile: UserProfile, context: ConversationContext) -> str:
        """Execute the multi-intent execution plan"""
        
        # For now, return the generated response from multi-intent handler
        if multi_intent_result.response_text:
            return multi_intent_result.response_text
        
        # Fallback to processing primary intent
        entities = self.entity_recognizer.extract_entities("")
        return self._handle_general_query("", entities, user_profile)

    def _handle_transportation_query_enhanced(self, user_input: str, location_result, user_profile: UserProfile, context: ConversationContext) -> str:
        """Enhanced transportation query handling with location context"""
        
        transportation_info = f"""🚇 **Transportation Information for {location_result.location}**

📍 **Location**: {location_result.location}
🎯 **Detection Method**: {location_result.detection_method}
✨ **Confidence**: {location_result.confidence:.1%}

🚊 **Transportation Options:**

"""
        
        # Add transportation modes based on location
        if location_result.location in ['Taksim', 'Şişli', 'Levent']:
            transportation_info += """**Metro**: M2 Metro Line available
**Bus**: Multiple bus routes
**Taxi**: Readily available
**Walking**: Central location, walkable to many areas

"""
        elif location_result.location in ['Sultanahmet', 'Eminönü']:
            transportation_info += """**Tram**: T1 Tram Line
**Ferry**: Ferry connections from Eminönü
**Bus**: City bus connections
**Walking**: Historic peninsula, many attractions walkable

"""
        elif location_result.location in ['Kadıköy', 'Üsküdar']:
            transportation_info += """**Ferry**: Ferry connections to European side
**Bus**: Extensive bus network
**Metro**: Metro connections available
**Taxi**: Available for cross-city travel

"""
        else:
            transportation_info += """**Mixed Transportation**: Various options available
**Bus**: City bus connections
**Taxi**: Available throughout the city
**Metro/Tram**: Check nearest stations

"""
        
        transportation_info += """💡 **Tips:**
• Use Istanbul transportation app for real-time schedules
• Get an Istanbulkart for easy payment
• Consider traffic when planning your route
• Ferry connections offer scenic routes"""
        
        return transportation_info

    def _handle_location_search_enhanced(self, user_input: str, location_result, user_profile: UserProfile, context: ConversationContext) -> str:
        """Enhanced location search with intelligent detection results"""
        
        # Build enhanced response with location detection metadata
        entities = self.entity_recognizer.extract_entities(user_input)
        entities['districts'] = [location_result.location]
        
        # Add location confidence to response
        base_response = self._handle_restaurant_query(user_input, entities, user_profile, context)
        
        # Enhance with detection metadata
        location_info = f"\n\n📍 **Location Detection**: {location_result.location} (confidence: {location_result.confidence:.1%})"
        if location_result.fallback_locations:
            location_info += f"\n🔄 **Alternative areas**: {', '.join(location_result.fallback_locations[:3])}"
        
        return base_response + location_info

    def _process_traditional(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> str:
        """Traditional processing method as fallback"""
        
        # Extract entities first
        entities = self.entity_recognizer.extract_entities(user_input)
        
        # Use enhanced intent classification for better routing
        primary_intent = self._enhance_intent_classification(user_input)
        
        # Process based on intent with improved routing
        return self._process_by_intent_enhanced(user_input, primary_intent, entities, user_profile)

    def _generate_fallback_response(self, user_input: str) -> str:
        """Generate fallback response when processing fails"""
        return """🤔 I'm having trouble understanding your request right now.

🌟 **I can help you with:**
• 🍽️ Restaurant recommendations ("restaurants in Kadıköy")
• 🏘️ Neighborhood guides ("what's Beşiktaş like?")
• 🎭 Events and cultural activities
• 🚇 Transportation and directions
• 🏛️ Attractions and sightseeing
• 💡 Local tips and hidden gems

**Try asking:**
• "Best restaurants in [neighborhood]"
• "What's the vibe in [district]?"
• "How to get to [location]?"
• "What to see in Istanbul?"

Please rephrase your question, and I'll do my best to help! 😊"""
