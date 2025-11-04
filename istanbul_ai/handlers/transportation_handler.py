"""
Transportation Handler for Istanbul AI
Handles: Public transport, metro, bus, ferry, route planning, GPS navigation

This handler consolidates all transportation-related functionality that was
previously scattered in main_system.py into a dedicated, ML-enhanced handler.

🌐 BILINGUAL SUPPORT: Full English/Turkish parity for all transportation responses
"""

from typing import Dict, Optional, List, Any, Union
import logging
from datetime import datetime

# Import bilingual support
try:
    from ..services.bilingual_manager import BilingualManager, Language
    BILINGUAL_AVAILABLE = True
except ImportError:
    BILINGUAL_AVAILABLE = False
    Language = None

logger = logging.getLogger(__name__)


class TransportationHandler:
    """
    ML-Enhanced Transportation Response Handler
    
    Capabilities:
    - Public transport information (metro, bus, tram, ferry)
    - Route planning with real-time data
    - GPS-based navigation
    - Transfer instructions with map visualization
    - Station/stop information
    - Fare information
    - IBB API integration for live data
    """
    
    def __init__(
        self,
        transportation_chat=None,
        transport_processor=None,
        gps_route_service=None,
        bilingual_manager=None,
        map_integration_service=None,
        transfer_map_integration_available: bool = False,
        advanced_transport_available: bool = False,
        llm_service=None,
        gps_location_service=None
    ):
        """
        Initialize transportation handler with required services.
        
        Args:
            transportation_chat: TransportationMapChat service for map visualization
            transport_processor: AdvancedTransportationProcessor for IBB API integration
            gps_route_service: GPSRouteService for GPS-based navigation
            bilingual_manager: BilingualManager for language support
            map_integration_service: MapIntegrationService for map visualization
            transfer_map_integration_available: Flag for transfer map feature
            advanced_transport_available: Flag for advanced transport feature
            llm_service: Optional LLM service for GPS-aware transportation advice
            gps_location_service: GPS location service for district detection
        """
        self.transportation_chat = transportation_chat
        self.transport_processor = transport_processor
        self.gps_route_service = gps_route_service
        self.bilingual_manager = bilingual_manager
        self.map_integration_service = map_integration_service
        
        # LLM + GPS integration
        self.llm_service = llm_service
        self.gps_location_service = gps_location_service
        
        # Feature availability flags
        self.transfer_map_integration_available = transfer_map_integration_available
        self.advanced_transport_available = advanced_transport_available
        
        # Initialize service availability flags
        self.has_transportation_chat = transportation_chat is not None
        self.has_transport_processor = transport_processor is not None
        self.has_gps_service = gps_route_service is not None
        self.has_bilingual = bilingual_manager is not None and BILINGUAL_AVAILABLE
        self.has_maps = map_integration_service is not None and map_integration_service.is_enabled()
        self.has_llm = llm_service is not None
        self.has_gps_location = gps_location_service is not None
        
        logger.info(
            f"Transportation Handler initialized - "
            f"TransportChat: {self.has_transportation_chat}, "
            f"AdvancedTransport: {self.has_transport_processor}, "
            f"GPS: {self.has_gps_service}, "
            f"Bilingual: {self.has_bilingual}, "
            f"Maps: {self.has_maps}, "
            f"TransferMap: {transfer_map_integration_available}, "
            f"LLM: {self.has_llm}, "
            f"GPSLocation: {self.has_gps_location}"
        )
    
    def _get_language(self, context) -> str:
        """
        Extract language from context.
        
        Args:
            context: Conversation context
            
        Returns:
            Language code ('en' or 'tr')
        """
        if not context:
            return 'en'
        
        # Check if language is in context
        if hasattr(context, 'language'):
            lang = context.language
            if hasattr(lang, 'value'):
                return lang.value  # Language enum
            return lang if lang in ['en', 'tr'] else 'en'
        
        # Default to English
        return 'en'
    
    def can_handle(self, message: str, entities: Dict[str, Any]) -> bool:
        """
        Determine if this handler can process the given query.
        
        Args:
            message: User's query
            entities: Extracted entities
            
        Returns:
            True if this is a transportation query
        """
        message_lower = message.lower()
        
        # Transportation keywords
        transport_keywords = [
            'metro', 'bus', 'tram', 'ferry', 'transport', 'transportation',
            'train', 'subway', 'istanbulkart', 'public transport',
            'funicular', 'metrobus', 'metrobüs', 'dolmuş', 'dolmus',
            'how to get', 'how do i get', 'how can i get',
            'how to go', 'directions', 'route', 'travel to', 'reach',
            'from', 'way to', 'navigate', 'navigation'
        ]
        
        return any(keyword in message_lower for keyword in transport_keywords)
    
    def handle(
        self,
        message: str,
        entities: Dict[str, Any],
        user_profile=None,
        context=None,
        neural_insights: Optional[Dict] = None,
        return_structured: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Main entry point for transportation queries.
        
        Args:
            message: User's query
            entities: Extracted entities (locations, transport types, etc.)
            user_profile: User preferences and history
            context: Conversation context (includes language)
            neural_insights: ML-powered insights (sentiment, temporal context, keywords)
            return_structured: Whether to return structured data
            
        Returns:
            Response string or dict based on return_structured
        """
        try:
            # 🌐 BILINGUAL: Extract language from context
            language = self._get_language(context)
            logger.info(f"🚇 Transportation query (lang: {language})")
            
            # Extract ML insights
            temporal_context = neural_insights.get('temporal_context') if neural_insights else None
            sentiment = neural_insights.get('sentiment') if neural_insights else None
            
            logger.info(
                f"🧠 Transportation query with ML insights: "
                f"temporal={temporal_context}, sentiment={sentiment}"
            )
            
            # Classify query type
            query_type = self._classify_transport_query(message, entities)
            
            # Route to appropriate handler (all handlers now receive language)
            if query_type == 'route_planning':
                return self._handle_route_planning(
                    message, entities, user_profile, context, 
                    neural_insights, return_structured, language
                )
            elif query_type == 'gps_navigation':
                return self._handle_gps_navigation(
                    message, entities, user_profile, context, 
                    return_structured, language
                )
            elif query_type == 'station_info':
                return self._handle_station_info(
                    message, entities, user_profile, context, 
                    return_structured, language
                )
            else:
                return self._handle_general_transport(
                    message, entities, user_profile, context, 
                    return_structured, language
                )
                
        except Exception as e:
            logger.error(f"Transportation handler error: {e}", exc_info=True)
            language = self._get_language(context)
            return self._get_fallback_response(entities, user_profile, context, return_structured, language)
    
    def _classify_transport_query(self, message: str, entities: Dict) -> str:
        """
        Classify the type of transportation query.
        
        Args:
            message: User's query
            entities: Extracted entities
            
        Returns:
            Query type: 'route_planning', 'gps_navigation', 'station_info', or 'general'
        """
        message_lower = message.lower()
        
        # Route planning indicators
        route_indicators = [
            'from', 'to', 'how to get', 'how do i get', 'how can i get',
            'how to go', 'how do i go', 'how can i go',
            'directions', 'route from', 'route to', 'way to get', 'way to go',
            'get to', 'go to', 'travel to', 'reach'
        ]
        
        # GPS navigation indicators
        gps_indicators = ['navigate', 'navigation', 'gps', 'turn by turn']
        
        # Station info indicators
        station_indicators = [
            'station', 'stop', 'terminal', 'pier', 'iskele',
            'which line', 'what line', 'metro line', 'tram line'
        ]
        
        if any(indicator in message_lower for indicator in route_indicators):
            if any(indicator in message_lower for indicator in gps_indicators):
                return 'gps_navigation'
            return 'route_planning'
        elif any(indicator in message_lower for indicator in station_indicators):
            return 'station_info'
        else:
            return 'general'
    
    def _handle_route_planning(
        self,
        message: str,
        entities: Dict,
        user_profile,
        context,
        neural_insights: Optional[Dict],
        return_structured: bool,
        language: str = 'en'
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle route planning queries with transfer instructions and map visualization.
        
        Args:
            message: User's query
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
            neural_insights: ML insights
            return_structured: Whether to return structured response
            language: Target language ('en' or 'tr')
            
        Returns:
            Route planning response
        """
        # Use Transfer Instructions & Map Visualization system if available
        if self.transfer_map_integration_available and self.has_transportation_chat:
            logger.info("🗺️ Using Transfer Instructions & Map Visualization system")
            
            # Extract locations from entities
            user_location = None
            destination = None
            
            if 'location' in entities and entities['location']:
                locations = entities['location']
                if len(locations) >= 2:
                    user_location = locations[0]
                    destination = locations[1]
                elif len(locations) == 1:
                    destination = locations[0]
            
            # 📍 CRITICAL: Check for GPS location from user_profile if not in entities
            # This handles queries like "How can I go to Taksim from my location?"
            if not user_location and user_profile and hasattr(user_profile, 'current_location'):
                gps_location = user_profile.current_location
                if gps_location and isinstance(gps_location, tuple) and len(gps_location) == 2:
                    user_location = f"{gps_location[0]:.6f},{gps_location[1]:.6f}"
                    logger.info(f"📍 Using GPS location from user profile: {user_location}")
            
            # Also check if query mentions "my location", "from here", "current position"
            message_lower = message.lower()
            if any(phrase in message_lower for phrase in [
                'my location', 'my position', 'from here', 'where i am', 'current location',
                'current position', 'from my location', 'from my position'
            ]):
                if not user_location and user_profile and hasattr(user_profile, 'current_location'):
                    gps_location = user_profile.current_location
                    if gps_location and isinstance(gps_location, tuple) and len(gps_location) == 2:
                        user_location = f"{gps_location[0]:.6f},{gps_location[1]:.6f}"
                        logger.info(f"📍 Detected 'my location' phrase, using GPS: {user_location}")
            
            # Build intelligent user context using ML insights
            user_context = self._build_intelligent_user_context(
                message, neural_insights, user_profile
            )
            
            # Process the transportation query (async)
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                
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
                    
                    # ==================== LLM ENHANCEMENT ====================
                    # Build GPS context and enhance with LLM if available
                    gps_context = self._build_gps_context(user_profile)
                    
                    if self.has_llm and gps_context.get('has_gps'):
                        try:
                            # Prepare route data for LLM
                            route_data = {
                                'duration': result.get('total_time', 0),
                                'distance': result.get('distance', 0),
                                'transfer_count': result.get('transfer_count', 0),
                                'steps': result.get('detailed_route', []),
                                'alternatives': result.get('alternatives', [])
                            }
                            
                            # Get LLM-enhanced advice
                            llm_advice = self._enhance_with_llm(
                                route_data=route_data,
                                gps_context=gps_context,
                                destination=destination or "your destination",
                                user_preferences={}
                            )
                            
                            if llm_advice:
                                # Prepend LLM advice to existing response
                                response_text = f"{llm_advice}\n\n{response_text}"
                                logger.info("✨ Response enhanced with LLM advice")
                        except Exception as e:
                            logger.warning(f"LLM enhancement failed, using original response: {e}")
                    # ==================== END LLM ENHANCEMENT ====================
                    
                    if return_structured:
                        return {
                            'response': response_text,
                            'map_data': map_data,
                            'detailed_route': result.get('detailed_route'),
                            'alternatives': result.get('alternatives', []),
                            'fare_info': result.get('fare_info'),
                            'transfer_count': result.get('transfer_count', 0),
                            'total_time': result.get('total_time', 0),
                            'gps_context': gps_context,  # Include GPS context
                            'handler': 'transportation_handler',
                            'success': True
                        }
                    else:
                        return response_text
                else:
                    # If clarification needed or error, fall through to other systems
                    if result.get('needs_clarification'):
                        if return_structured:
                            return {
                                'response': result.get('response_text', ''),
                                'needs_clarification': True,
                                'handler': 'transportation_handler',
                                'success': False
                            }
                        else:
                            return result.get('response_text', '')
            except Exception as e:
                logger.error(f"Transfer map system error: {e}", exc_info=True)
                # Fall through to next system
        
        # Use GPS route planner for route-specific queries
        logger.info("🗺️ Using GPS route planner for route-specific query")
        return self._handle_gps_navigation(
            message, entities, user_profile, context, return_structured
        )
    
    def _handle_gps_navigation(
        self,
        message: str,
        entities: Dict,
        user_profile,
        context,
        return_structured: bool,
        language: str = 'en'
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle GPS-based navigation queries.
        
        Args:
            message: User's query
            entities: Extracted entities
            user_profile: User profile (contains current_location if GPS enabled)
            context: Conversation context
            return_structured: Whether to return structured response
            language: Target language
            
        Returns:
            GPS navigation response
        """
        if not self.has_gps_service:
            logger.warning("⚠️ GPS Route Service not available, using fallback")
            return self._get_fallback_response(
                entities, user_profile, context, return_structured
            )
        
        # 📍 CRITICAL: Extract GPS location from user_profile
        user_gps_location = None
        if user_profile and hasattr(user_profile, 'current_location'):
            gps_location = user_profile.current_location
            if gps_location and isinstance(gps_location, tuple) and len(gps_location) == 2:
                user_gps_location = gps_location
                logger.info(f"📍 GPS Navigation using user location: {user_gps_location[0]:.6f}, {user_gps_location[1]:.6f}")
        
        # Check if query explicitly mentions "my location"
        message_lower = message.lower()
        if any(phrase in message_lower for phrase in [
            'my location', 'from here', 'where i am', 'current location', 'my position'
        ]):
            if user_gps_location:
                logger.info("� User explicitly requested navigation from their location")
            else:
                # GPS not enabled - provide helpful message
                gps_prompt = (
                    "To navigate from your current location, please enable GPS. "
                    "Click the 📍 GPS button to allow location access. "
                    "Alternatively, tell me your starting location (e.g., 'from Taksim')."
                ) if language == 'en' else (
                    "Mevcut konumunuzdan yol tarifi almak için lütfen GPS'i etkinleştirin. "
                    "Konum erişimine izin vermek için 📍 GPS düğmesine tıklayın. "
                    "Alternatif olarak başlangıç konumunuzu söyleyin (örn. 'Taksim'den')."
                )
                
                if return_structured:
                    return {
                        'response': gps_prompt,
                        'handler': 'transportation_handler',
                        'method': 'gps_navigation',
                        'success': False,
                        'needs_gps': True
                    }
                else:
                    return gps_prompt
        
        try:
            logger.info("🗺️ Delegating to GPS Route Service")
            response = self.gps_route_service.generate_route_response(
                message, entities, user_profile, context
            )
            
            # ==================== LLM ENHANCEMENT ====================
            # Build GPS context and enhance response with LLM if available
            gps_context = self._build_gps_context(user_profile)
            
            if self.has_llm and gps_context.get('has_gps'):
                try:
                    # Extract destination from entities or message
                    destination = None
                    if 'location' in entities and entities['location']:
                        locations = entities['location']
                        destination = locations[-1] if locations else None
                    
                    if destination:
                        # Create simplified route data (GPS service response is text)
                        route_data = {
                            'text_response': response,
                            'has_route': True
                        }
                        
                        # Get LLM-enhanced advice
                        llm_advice = self._enhance_with_llm(
                            route_data=route_data,
                            gps_context=gps_context,
                            destination=destination,
                            user_preferences={}
                        )
                        
                        if llm_advice:
                            # Prepend LLM advice to GPS route response
                            response = f"{llm_advice}\n\n{response}"
                            logger.info("✨ GPS navigation enhanced with LLM advice")
                except Exception as e:
                    logger.warning(f"LLM enhancement failed, using original response: {e}")
            # ==================== END LLM ENHANCEMENT ====================
            
            if return_structured:
                return {
                    'response': response,
                    'handler': 'transportation_handler',
                    'method': 'gps_navigation',
                    'success': True,
                    'gps_location': user_gps_location,
                    'gps_context': gps_context  # Include GPS context
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error in GPS route response: {e}", exc_info=True)
            return self._get_fallback_response(
                entities, user_profile, context, return_structured
            )
    
    def _handle_station_info(
        self,
        message: str,
        entities: Dict,
        user_profile,
        context,
        return_structured: bool,
        language: str = 'en'
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle station/stop information queries.
        
        Args:
            message: User's query
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
            return_structured: Whether to return structured response
            
        Returns:
            Station information response
        """
        # Use advanced transportation system if available
        if self.advanced_transport_available and self.has_transport_processor:
            logger.info("🚇 Using advanced transportation system for station info")
            
            try:
                enhanced_response = self.transport_processor.process_transportation_query_sync(
                    message, entities, user_profile
                )
                
                if enhanced_response and enhanced_response.strip():
                    if return_structured:
                        return {
                            'response': enhanced_response,
                            'handler': 'transportation_handler',
                            'method': 'advanced_system',
                            'success': True
                        }
                    else:
                        return enhanced_response
            except Exception as e:
                logger.error(f"Advanced transport system error: {e}", exc_info=True)
        
        # Fallback to general transport info
        return self._handle_general_transport(
            message, entities, user_profile, context, return_structured
        )
    
    def _handle_general_transport(
        self,
        message: str,
        entities: Dict,
        user_profile,
        context,
        return_structured: bool,
        language: str = 'en'
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle general transportation queries.
        
        Args:
            message: User's query
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
            return_structured: Whether to return structured response
            
        Returns:
            General transportation information
        """
        # Use advanced transportation system if available
        if self.advanced_transport_available and self.has_transport_processor:
            logger.info("🚇 Using advanced transportation system with IBB API")
            
            try:
                enhanced_response = self.transport_processor.process_transportation_query_sync(
                    message, entities, user_profile
                )
                
                if enhanced_response and enhanced_response.strip():
                    if return_structured:
                        return {
                            'response': enhanced_response,
                            'handler': 'transportation_handler',
                            'method': 'advanced_system',
                            'success': True
                        }
                    else:
                        return enhanced_response
            except Exception as e:
                logger.error(f"Advanced transport system error: {e}", exc_info=True)
        
        # Fallback to static response
        logger.info("🚇 Using fallback transportation system")
        return self._get_fallback_response(
            entities, user_profile, context, return_structured
        )
    
    def _build_intelligent_user_context(
        self,
        message: str,
        neural_insights: Optional[Dict],
        user_profile
    ) -> Dict[str, Any]:
        """
        Build intelligent user context from ML insights and user profile.
        
        Args:
            message: User's query
            neural_insights: ML-powered insights
            user_profile: User profile
            
        Returns:
            User context dictionary
        """
        context = {}
        
        if neural_insights:
            context['temporal_context'] = neural_insights.get('temporal_context')
            context['sentiment'] = neural_insights.get('sentiment')
            context['urgency'] = neural_insights.get('urgency')
            context['keywords'] = neural_insights.get('keywords', [])
        
        if user_profile:
            if hasattr(user_profile, 'preferences'):
                context['preferences'] = user_profile.preferences
            if hasattr(user_profile, 'accessibility_needs'):
                context['accessibility_needs'] = user_profile.accessibility_needs
            if hasattr(user_profile, 'budget_range'):
                context['budget_range'] = user_profile.budget_range
        
        return context
    
    def _build_gps_context(self, user_profile) -> Dict[str, Any]:
        """
        Build GPS context from user profile for LLM enhancement.
        
        Args:
            user_profile: User profile with optional GPS location
            
        Returns:
            Dictionary with GPS context:
            {
                'gps_location': (lat, lon) or None,
                'district': str or None,
                'confidence': float,
                'has_gps': bool
            }
        """
        gps_context = {
            'gps_location': None,
            'district': None,
            'confidence': 0.0,
            'has_gps': False
        }
        
        # Extract GPS from user profile
        if user_profile and hasattr(user_profile, 'current_location'):
            gps_location = user_profile.current_location
            if gps_location and isinstance(gps_location, tuple) and len(gps_location) == 2:
                gps_context['gps_location'] = gps_location
                gps_context['has_gps'] = True
                
                # Detect district using GPS location service
                if self.has_gps_location:
                    try:
                        district_info = self.gps_location_service.get_district_from_coordinates(
                            gps_location[0], gps_location[1]
                        )
                        if district_info:
                            gps_context['district'] = district_info.get('district')
                            gps_context['confidence'] = district_info.get('confidence', 0.0)
                            logger.info(
                                f"📍 Detected district: {gps_context['district']} "
                                f"(confidence: {gps_context['confidence']:.2f})"
                            )
                    except Exception as e:
                        logger.warning(f"District detection failed: {e}")
        
        return gps_context
    
    def _enhance_with_llm(
        self,
        route_data: Dict[str, Any],
        gps_context: Dict[str, Any],
        destination: str,
        user_preferences: Optional[Dict] = None
    ) -> str:
        """
        Enhance transportation response with LLM-generated advice.
        
        Args:
            route_data: Route information (duration, distance, steps)
            gps_context: GPS context from _build_gps_context
            destination: Destination name
            user_preferences: Optional user preferences
            
        Returns:
            LLM-generated transportation advice (concise, Google Maps-style)
        """
        if not self.has_llm:
            return ""
        
        try:
            # Prepare context for LLM
            origin_district = gps_context.get('district', 'your location')
            origin_coords = gps_context.get('gps_location')
            
            # Build user preferences (time of day, budget, accessibility, etc.)
            prefs = user_preferences or {}
            
            # Get LLM advice
            llm_advice = self.llm_service.get_transportation_advice(
                origin=origin_coords or origin_district,
                destination=destination,
                route_data=route_data,
                user_preferences=prefs,
                gps_context=gps_context
            )
            
            logger.info(f"✨ LLM transportation advice generated ({len(llm_advice)} chars)")
            return llm_advice
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return ""
    
    def _get_fallback_response(
        self,
        entities: Dict,
        user_profile,
        context,
        return_structured: bool,
        language: str = 'en'
    ) -> Union[str, Dict[str, Any]]:
        """
        Provide fallback response when advanced features fail.
        
        Args:
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
            return_structured: Whether to return structured response
            language: Target language ('en' or 'tr')
            
        Returns:
            Fallback transportation information
        """
        current_time = datetime.now().strftime("%H:%M")
        
        # 🌐 BILINGUAL: Select response based on language
        if language == 'tr':
            response = self._get_turkish_fallback_response(current_time)
        else:
            response = self._get_english_fallback_response(current_time)
        
        if return_structured:
            return {
                'response': response,
                'map_data': {},
                'handler': 'transportation_handler',
                'method': 'fallback',
                'language': language,
                'success': True,
                'is_fallback': True
            }
        else:
            return response
    
    def _get_english_fallback_response(self, current_time: str) -> str:
        """Generate English fallback response."""
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
    
    def _get_turkish_fallback_response(self, current_time: str) -> str:
        """Generate Turkish fallback response."""
        return f"""🚇 **İstanbul Ulaşım Rehberi**
📍 **Canlı Durum** (Güncellenme: {current_time})

**🎫 Temel Ulaşım Kartı:**
• **İstanbulkart**: Resmi toplu taşıma için olmazsa olmaz (13 TL + bakiye)
• Metro istasyonları, büfeler ve vapur iskelelerinde satılır
• Metro, tramvay, otobüs, vapur ve metrobüste geçerli (Dolmuşta geçersiz - nakit)

**🚇 Metro Hatları:**
• **M1A**: Yenikapı ↔ Atatürk Havalimanı (kapalı) - Kapalıçarşı bölgesine hizmet
• **M2**: Vezneciler ↔ Hacıosman (Taksim, Şişli, Levent'e hizmet)
• **M4**: Kadıköy ↔ Sabiha Gökçen Havalimanı (Anadolu yakası)
• **M11**: İST Havalimanı ↔ Gayrettepe (yeni havalimanı bağlantısı)
• **M6**: Levent ↔ Boğaziçi Üniversitesi

**🚋 Tarihi Tramvaylar:**
• **T1**: Kabataş ↔ Bağcılar (Sultanahmet, Eminönü, Karaköy'ü bağlar)
• **Nostaljik Tramvay**: Taksim ↔ Tünel (tarihi İstiklal Caddesi)

**⛴️ Vapurlar (En Manzaralı!):**
• **Eminönü ↔ Kadıköy**: 20 dakika, muhteşem şehir manzarası
• **Karaköy ↔ Üsküdar**: Hızlı Boğaz geçişi
• **Boğaz Turları**: 1.5 saatlik manzaralı geziler (90-150 TL)

**🚌 Otobüsler ve Dolmuşlar:**
• Geniş ağ ama kalabalık olabilir
• Dolmuşlar belirli güzergahlarda çalışır - sadece nakit, İstanbulkart yok
• İşaretler Türkçe ve İngilizce

**💡 Pratik İpuçları:**
• Citymapper veya Moovit uygulamalarını indirin
• Yoğun saatler: 07:30-09:30, 17:30-19:30
• Vapurlar Boğaz geçişinde genellikle daha hızlı
• İstanbulkartı hazır tutun - denetim sık yapılır
• Metro gece yarısına kadar çalışır, sınırlı gece otobüsü hizmeti

**🎯 Popüler Rotalar:**
• **İST Havalimanı → Sultanahmet**: M11 + M2 + T1 (60 dk, ~20 TL)
• **Taksim → Sultanahmet**: M2 + T1 (25 dk, ~7 TL)  
• **Sultanahmet → Galata Kulesi**: T1 + M2 (25 dk)
• **Avrupa → Asya yakası**: Eminönü/Karaköy'den vapur

Belirli bir rota için yardım mı gerekiyor? Başlangıç ve varış noktanızı söyleyin!"""
