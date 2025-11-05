"""
Transportation Handler for Istanbul AI
Handles: Public transport, metro, bus, ferry, route planning, GPS navigation

This handler consolidates all transportation-related functionality that was
previously scattered in main_system.py into a dedicated, ML-enhanced handler.

🌐 BILINGUAL SUPPORT: Full English/Turkish parity for all transportation responses
🌤️ WEATHER-AWARE: Integrates weather data for route recommendations (Step 3.2)
💎 HIDDEN GEMS: Contextual local recommendations in route responses (Step 2)

Updated: November 5, 2025 - Added hidden gems integration to LLM responses
Previous: November 5, 2025 - Added weather-aware transportation advice
Previous: November 4, 2025 - Added LLM and GPS integration
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

# Import Marmaray data (Step 4.2)
try:
    from backend.data.marmaray_stations import (
        get_marmaray_recommendation,
        find_nearest_marmaray_station,
        crosses_bosphorus,
        MARMARAY_INFO
    )
    MARMARAY_AVAILABLE = True
except ImportError:
    MARMARAY_AVAILABLE = False
    logger.warning("Marmaray station data not available")

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
    - Weather-aware route recommendations (Step 3.2)
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
        gps_location_service=None,
        weather_service=None,
        hidden_gems_context_service=None
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
            weather_service: Optional weather service for weather-aware route advice (Step 3.2)
            hidden_gems_context_service: Optional hidden gems service for contextual recommendations
        """
        self.transportation_chat = transportation_chat
        self.transport_processor = transport_processor
        self.gps_route_service = gps_route_service
        self.bilingual_manager = bilingual_manager
        self.map_integration_service = map_integration_service
        
        # LLM + GPS + Weather + Hidden Gems integration
        self.llm_service = llm_service
        self.gps_location_service = gps_location_service
        self.weather_service = weather_service
        self.hidden_gems_context_service = hidden_gems_context_service
        
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
        self.has_weather = weather_service is not None
        self.has_hidden_gems = hidden_gems_context_service is not None
        
        logger.info(
            f"Transportation Handler initialized - "
            f"TransportChat: {self.has_transportation_chat}, "
            f"AdvancedTransport: {self.has_transport_processor}, "
            f"GPS: {self.has_gps_service}, "
            f"Bilingual: {self.has_bilingual}, "
            f"Maps: {self.has_maps}, "
            f"TransferMap: {transfer_map_integration_available}, "
            f"LLM: {self.has_llm}, "
            f"GPSLocation: {self.has_gps_location}, "
            f"Weather: {self.has_weather}, "
            f"HiddenGems: {self.has_hidden_gems}"
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
    
    # ===== WEATHER-AWARE TRANSPORTATION (Step 3.2) =====
    
    async def _get_weather_data_for_route(self) -> Optional[Dict[str, Any]]:
        """
        Get current weather data for route planning
        
        Returns:
            Weather data dict or None if service unavailable
        """
        if not self.has_weather:
            return None
        
        try:
            weather = await self.weather_service.get_current_weather("Istanbul")
            return {
                'temperature': weather.get('temperature'),
                'conditions': weather.get('condition'),
                'description': weather.get('description'),
                'humidity': weather.get('humidity'),
                'wind_speed': weather.get('wind_speed')
            }
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch weather data: {e}")
            return None
    
    def _enhance_route_with_weather_advice(
        self,
        route_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]],
        gps_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance route response with weather-aware LLM advice
        
        This implements Step 3.2: Weather-aware transportation recommendations
        
        Args:
            route_data: Route information (duration, distance, transit types)
            weather_data: Current weather data
            gps_context: Optional GPS location context
            
        Returns:
            Enhanced route data with weather-aware advice
        """
        # Return original route if weather or LLM unavailable
        if not weather_data or not self.has_llm:
            return route_data
        
        try:
            from ml_systems.context_aware_prompts import ContextAwarePromptEngine
            
            engine = ContextAwarePromptEngine()
            
            # Extract route details
            origin = route_data.get('origin', 'your location')
            destination = route_data.get('destination', 'destination')
            duration = route_data.get('duration', 'unknown')
            distance = route_data.get('distance', 'unknown')
            transit_types = route_data.get('transit_types', [])
            
            # Create weather-aware transportation prompt
            prompt = engine.create_transportation_advice_prompt(
                origin=origin,
                destination=destination,
                route_data={
                    'duration': duration,
                    'distance': distance,
                    'transit_types': transit_types
                },
                weather_data=weather_data,
                gps_context=gps_context
            )
            
            # Generate concise LLM advice (max 100 tokens for brevity)
            logger.info("🤖 Generating weather-aware transportation advice...")
            llm_advice = self.llm_service.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            
            # Truncate to 3 sentences if needed
            if llm_advice:
                llm_advice = engine.truncate_to_sentences(llm_advice, max_sentences=3)
                logger.info("✅ Weather-aware advice generated successfully")
            
            # Add weather advice to route data
            route_data['weather_advice'] = llm_advice
            route_data['weather_context'] = {
                'temperature': weather_data.get('temperature'),
                'conditions': weather_data.get('conditions'),
                'description': weather_data.get('description')
            }
            
            return route_data
            
        except Exception as e:
            logger.error(f"❌ Error generating weather-aware advice: {e}")
            # Return original route without weather advice
            return route_data
    
    def _check_weather_impact_on_transit(
        self,
        transit_types: List[str],
        weather_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Check if weather significantly impacts the transit modes
        
        Args:
            transit_types: List of transit modes (ferry, bus, metro, etc.)
            weather_data: Current weather data
            
        Returns:
            Warning message if weather impacts transit, None otherwise
        """
        conditions = weather_data.get('conditions', '').lower()
        temp = weather_data.get('temperature', 20)
        wind_speed = weather_data.get('wind_speed', 0)
        
        warnings = []
        
        # Check ferry in bad weather
        if any('ferry' in t.lower() or 'vapur' in t.lower() for t in transit_types):
            if 'rain' in conditions or 'storm' in conditions:
                warnings.append("⚠️ Ferry service may be delayed or uncomfortable in rain")
            elif wind_speed and wind_speed > 40:
                warnings.append("⚠️ Ferry service may be affected by high winds")
        
        # Check outdoor waiting in extreme weather
        if temp < 0:
            warnings.append("🧥 Very cold weather - consider metro/underground options")
        elif temp > 35:
            warnings.append("☀️ Very hot weather - seek air-conditioned options")
        
        # Check rain impact on bus stops
        if 'rain' in conditions or 'shower' in conditions:
            if any('bus' in t.lower() or 'otobüs' in t.lower() for t in transit_types):
                warnings.append("☔ Bus stops may be wet - bring umbrella")
        
        return " ".join(warnings) if warnings else None
    
    # ===== MARMARAY INTEGRATION (Step 4.2) =====
    
    def _check_marmaray_option(
        self,
        origin_coords: Dict[str, float],
        dest_coords: Dict[str, float],
        weather_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if Marmaray is a good option for this route (Step 4.2)
        
        Args:
            origin_coords: {'lat': float, 'lng': float}
            dest_coords: {'lat': float, 'lng': float}
            weather_data: Current weather information
            
        Returns:
            Marmaray recommendation dict or None
        """
        if not MARMARAY_AVAILABLE:
            return None
        
        try:
            # Get weather conditions string
            weather_conditions = None
            if weather_data:
                weather_conditions = weather_data.get('conditions', weather_data.get('description', ''))
            
            # Get Marmaray recommendation
            recommendation = get_marmaray_recommendation(
                origin_lat=origin_coords['lat'],
                origin_lon=origin_coords['lng'],
                dest_lat=dest_coords['lat'],
                dest_lon=dest_coords['lng'],
                weather_conditions=weather_conditions
            )
            
            if recommendation['use_marmaray']:
                logger.info(f"✅ Marmaray recommended: {recommendation['recommendation_strength']}")
                return recommendation
            else:
                logger.info(f"ℹ️  Marmaray not applicable: {recommendation.get('reason', 'Unknown')}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking Marmaray option: {e}")
            return None
    
    def _create_marmaray_llm_advice(
        self,
        query: str,
        regular_route: Dict[str, Any],
        marmaray_option: Dict[str, Any],
        weather_data: Dict[str, Any]
    ) -> str:
        """
        Generate LLM advice comparing regular route with Marmaray option (Step 4.2)
        
        Args:
            query: User's original query
            regular_route: Regular route data from OSRM
            marmaray_option: Marmaray recommendation data
            weather_data: Current weather
            
        Returns:
            LLM-generated advice string
        """
        if not self.has_llm:
            # Fallback to template-based advice
            return self._create_marmaray_template_advice(marmaray_option, weather_data)
        
        try:
            from ml_systems.context_aware_prompts import ContextAwarePromptEngine
            
            engine = ContextAwarePromptEngine()
            
            # Extract origin/destination from query
            origin = query.split(' from ')[-1].split(' to ')[0] if ' from ' in query else 'origin'
            destination = query.split(' to ')[-1] if ' to ' in query else 'destination'
            
            # Create prompt
            prompt = engine.create_marmaray_comparison_prompt(
                origin=origin,
                destination=destination,
                regular_route={
                    'duration': regular_route.get('duration', 'unknown'),
                    'distance': regular_route.get('distance', 'unknown')
                },
                marmaray_route={
                    'duration': marmaray_option.get('travel_time_minutes', 'unknown'),
                    'undersea_time': marmaray_option.get('undersea_crossing_time', 4)
                },
                weather_data=weather_data
            )
            
            # Generate LLM response
            logger.info("🤖 Generating Marmaray comparison advice with LLM...")
            llm_advice = self.llm_service.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            
            # Validate and truncate if needed
            if engine.validate_response_length(llm_advice, max_words=80):
                return llm_advice
            else:
                truncated = engine.truncate_to_sentences(llm_advice, max_sentences=3)
                return truncated
                
        except Exception as e:
            logger.error(f"Error generating Marmaray LLM advice: {e}")
            return self._create_marmaray_template_advice(marmaray_option, weather_data)
    
    def _create_marmaray_template_advice(
        self,
        marmaray_option: Dict[str, Any],
        weather_data: Dict[str, Any]
    ) -> str:
        """
        Create template-based Marmaray advice (fallback) (Step 4.2)
        
        Args:
            marmaray_option: Marmaray recommendation data
            weather_data: Current weather
            
        Returns:
            Template-based advice string
        """
        strength = marmaray_option.get('recommendation_strength', 'recommended')
        travel_time = marmaray_option.get('travel_time_minutes', 'unknown')
        origin_station = marmaray_option.get('origin_station', {}).get('name', 'nearest station')
        dest_station = marmaray_option.get('dest_station', {}).get('name', 'destination station')
        
        # Weather-based recommendations
        conditions = weather_data.get('conditions', '').lower() if weather_data else ''
        
        if 'rain' in conditions or 'snow' in conditions:
            weather_tip = "Perfect for rainy weather - completely underground and weather-independent!"
        elif 'wind' in conditions:
            weather_tip = "Better than ferry in windy conditions - smooth underground crossing."
        else:
            weather_tip = "Fast and reliable underground crossing."
        
        if strength == 'highly_recommended':
            return f"🚇 Marmaray is highly recommended! {origin_station} → {dest_station} takes ~{travel_time} minutes. {weather_tip}"
        elif strength == 'alternative':
            return f"🚇 Marmaray is a good alternative: {origin_station} → {dest_station} (~{travel_time} min). {weather_tip}"
        else:
            return f"🚇 Consider Marmaray: {origin_station} → {dest_station} takes ~{travel_time} minutes. {weather_tip}"
    
    def _enhance_route_with_marmaray(
        self,
        route_response: Dict[str, Any],
        origin_coords: Dict[str, float],
        dest_coords: Dict[str, float],
        weather_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance route response with Marmaray option if applicable (Step 4.2)
        
        Args:
            route_response: Original route response
            origin_coords: Origin coordinates
            dest_coords: Destination coordinates
            weather_data: Current weather data
            
        Returns:
            Enhanced route response with Marmaray option
        """
        if not MARMARAY_AVAILABLE:
            return route_response
        
        # Check if Marmaray is applicable
        marmaray_option = self._check_marmaray_option(
            origin_coords=origin_coords,
            dest_coords=dest_coords,
            weather_data=weather_data
        )
        
        if not marmaray_option:
            # No Marmaray option, return original response
            return route_response
        
        # Get regular route data
        regular_route = route_response.get('route', {})
        
        # Generate LLM advice comparing options
        if weather_data:
            marmaray_advice = self._create_marmaray_llm_advice(
                query=route_response.get('query', ''),
                regular_route=regular_route,
                marmaray_option=marmaray_option,
                weather_data=weather_data
            )
        else:
            marmaray_advice = self._create_marmaray_template_advice(
                marmaray_option=marmaray_option,
                weather_data=weather_data or {}
            )
        
        # Add Marmaray option to response
        route_response['marmaray_option'] = {
            'available': True,
            'recommended': marmaray_option.get('use_marmaray', False),
            'strength': marmaray_option.get('recommendation_strength', 'recommended'),
            'travel_time_minutes': marmaray_option.get('travel_time_minutes'),
            'origin_station': marmaray_option.get('origin_station'),
            'dest_station': marmaray_option.get('dest_station'),
            'undersea_crossing_time': marmaray_option.get('undersea_crossing_time', 4),
            'advantages': marmaray_option.get('advantages', []),
            'llm_advice': marmaray_advice,
            'transfer_info': marmaray_option.get('transfer_info'),
            'weather_independent': True
        }
        
        logger.info(f"✅ Marmaray option added to route response ({marmaray_option.get('recommendation_strength')})")
        
        return route_response
    
    # ===== GPS AND LLM HELPER METHODS =====
    
    def _build_gps_context(self, user_profile) -> Dict[str, Any]:
        """
        Build GPS location context from user profile
        
        Args:
            user_profile: User profile with GPS location data
            
        Returns:
            GPS context dict with location info
        """
        gps_context = {
            'has_gps': False,
            'location': None,
            'district': None,
            'coordinates': None
        }
        
        if not user_profile or not hasattr(user_profile, 'current_location'):
            return gps_context
        
        gps_location = user_profile.current_location
        if not gps_location or not isinstance(gps_location, tuple) or len(gps_location) != 2:
            return gps_context
        
        gps_context['has_gps'] = True
        gps_context['coordinates'] = {
            'lat': gps_location[0],
            'lng': gps_location[1]
        }
        gps_context['location'] = f"{gps_location[0]:.6f},{gps_location[1]:.6f}"
        
        # Try to get district name from GPS service
        if self.gps_location_service:
            try:
                district = self.gps_location_service.get_district_from_coords(
                    gps_location[0],
                    gps_location[1]
                )
                gps_context['district'] = district
            except Exception as e:
                logger.warning(f"Could not get district from GPS: {e}")
        
        return gps_context
    
    def _build_intelligent_user_context(
        self,
        message: str,
        neural_insights: Optional[Dict],
        user_profile
    ) -> Dict[str, Any]:
        """
        Build intelligent user context using ML insights
        
        Args:
            message: User's query
            neural_insights: ML-generated insights
            user_profile: User profile
            
        Returns:
            User context dict for enhanced responses
        """
        context = {
            'preferences': {},
            'travel_style': 'balanced',
            'accessibility_needs': False,
            'language': 'en',
            'has_istanbulkart': True
        }
        
        # Extract preferences from neural insights
        if neural_insights:
            context['preferences'] = neural_insights.get('preferences', {})
            context['travel_style'] = neural_insights.get('travel_style', 'balanced')
        
        # Get language preference
        if user_profile and hasattr(user_profile, 'language'):
            context['language'] = user_profile.language
        
        # Detect accessibility needs from message
        accessibility_keywords = [
            'wheelchair', 'accessible', 'disabled', 'mobility',
            'elevator', 'ramp', 'stroller', 'pram'
        ]
        if any(keyword in message.lower() for keyword in accessibility_keywords):
            context['accessibility_needs'] = True
        
        return context
    
    def _enhance_with_llm(
        self,
        route_data: Dict[str, Any],
        gps_context: Dict[str, Any],
        destination: str,
        user_preferences: Dict[str, Any]
    ) -> Optional[str]:
        """
        Enhance route response with LLM-generated natural language advice
        
        This implements the LLM integration from the Transportation System Analysis
        & Enhancement Plan, generating natural, conversational responses instead of
        structured JSON.
        
        🆕 NOW INCLUDES HIDDEN GEMS CONTEXT - Step 2 Integration Complete
        
        Args:
            route_data: Route information (duration, distance, steps, alternatives)
            gps_context: GPS location context
            destination: Destination name
            user_preferences: User preferences (travel_style, accessibility, etc.)
            
        Returns:
            LLM-generated advice string or None if generation fails
        """
        if not self.has_llm:
            logger.info("LLM service not available for route enhancement")
            return None
        
        try:
            from ml_systems.context_aware_prompts import ContextAwarePromptEngine
            
            engine = ContextAwarePromptEngine()
            
            # Extract route details
            duration = route_data.get('duration', 'unknown')
            distance = route_data.get('distance', 'unknown')
            transfer_count = route_data.get('transfer_count', 0)
            steps = route_data.get('steps', [])
            alternatives = route_data.get('alternatives', [])
            
            # Determine origin from GPS context
            origin = gps_context.get('district', 'your location')
            if not origin or origin == 'your location':
                if gps_context.get('has_gps'):
                    origin = 'your current location'
            
            # 💎 GET HIDDEN GEMS CONTEXT
            hidden_gems_text = ""
            if self.has_hidden_gems:
                try:
                    origin_district = gps_context.get('district')
                    destination_district = self._extract_district_from_destination(destination)
                    
                    logger.info(f"💎 Checking hidden gems: origin={origin_district}, dest={destination_district}")
                    
                    gems_data = self.hidden_gems_context_service.get_gems_for_route(
                        origin_district=origin_district,
                        destination_district=destination_district,
                        max_gems_per_district=2
                    )
                    
                    # Add to prompt if gems found
                    if gems_data.get('destination_text'):
                        hidden_gems_text = gems_data['destination_text']
                        logger.info(f"💎 Found hidden gems for destination: {destination_district}")
                    elif gems_data.get('origin_text'):
                        hidden_gems_text = gems_data['origin_text']
                        logger.info(f"💎 Found hidden gems for origin: {origin_district}")
                    
                    if hidden_gems_text:
                        logger.info("✨ Hidden gems context added to LLM prompt")
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch hidden gems: {e}")
                    hidden_gems_text = ""
            
            # Create transportation advice prompt with hidden gems
            prompt = f"""You are KAM, a friendly Istanbul tour guide. Generate a natural, helpful response about this transportation route.

Route Information:
- From: {origin}
- To: {destination}
- Duration: {duration} minutes
- Distance: {distance} meters
- Transfers: {transfer_count}

Travel Style: {user_preferences.get('travel_style', 'balanced')}

{hidden_gems_text}

Respond with:
1. A friendly greeting acknowledging the route
2. Brief summary of the journey (1-2 sentences)
3. ONE hidden gem recommendation (if available from the list above) - be specific with the name

Keep it conversational, concise (max 4 sentences), and include relevant emojis (🚇🚋🚶‍♂️⛴️💎).

Response:"""
            
            # Generate LLM response
            logger.info("🤖 Generating LLM-enhanced route advice with hidden gems...")
            llm_advice = self.llm_service.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            
            if llm_advice:
                # Truncate to ensure conciseness (4 sentences max with gems)
                llm_advice = engine.truncate_to_sentences(llm_advice, max_sentences=4)
                logger.info("✅ LLM route advice generated successfully with hidden gems")
                return llm_advice
            else:
                logger.warning("LLM generated empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating LLM route advice: {e}")
            return None

    # ===== END GPS AND LLM HELPER METHODS =====

    # ===== FALLBACK RESPONSES =====
    
    def _get_fallback_response(
        self,
        entities: Dict,
        user_profile,
        context,
        return_structured: bool,
        language: str = 'en'
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate fallback response for general transportation queries
        
        Args:
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
            return_structured: Whether to return structured response
            language: Target language
            
        Returns:
            Fallback transportation information
        """
        current_time = datetime.now().strftime("%H:%M")
        
        # Get language-specific response
        if language == 'tr':
            response_text = self._get_turkish_fallback_response(current_time)
        else:
            response_text = self._get_english_fallback_response(current_time)
        
        if return_structured:
            return {
                'response': response_text,
                'handler': 'transportation_handler',
                'method': 'fallback',
                'success': True
            }
        else:
            return response_text
    
    def _get_english_fallback_response(self, current_time: str) -> str:
        """
        Generate English fallback response based on current time.
        
        Args:
            current_time: Current time string (HH:MM)
            
        Returns:
            Fallback response text
        """
        # Parse current time
        try:
            hour, minute = map(int, current_time.split(':'))
        except ValueError:
            hour, minute = 12, 0  # Default to noon on error
        
        # Determine time-based response
        if hour < 6:
            return "🌙 It's quite early, public transport is limited. Consider taking a taxi or waiting until morning."
        elif hour < 9:
            return "🚌 Morning! Buses and metros are starting to run. Check the latest schedules for updates."
        elif hour < 17:
            return "🚆 Daytime! Trains and buses are frequent. Safe travels!"
        elif hour < 20:
            return "🌆 Evening! Public transport is still active, but check the last departures for your route."
        else:
            return "🌙 It's late, some transport options may be limited. Plan your return trip accordingly."
    
    def _get_turkish_fallback_response(self, current_time: str) -> str:
        """
        Generate Turkish fallback response based on current time.
        
        Args:
            current_time: Current time string (HH:MM)
            
        Returns:
            Fallback response text
        """
        # Parse current time
        try:
            hour, minute = map(int, current_time.split(':'))
        except ValueError:
            hour, minute = 12, 0  # Default to noon on error
        
        # Determine time-based response
        if hour < 6:
            return "🌙 Çok erken, toplu taşıma sınırlı. Taksiyle gitmeyi veya sabahı beklemeyi düşünün."
        elif hour < 9:
            return "🚌 Sabah! Otobüsler ve metrolar çalışmaya başlıyor. Güncel tarifeleri kontrol edin."
        elif hour < 17:
            return "🚆 Gün içi! Trenler ve otobüsler sık sık çalışıyor. İyi yolculuklar!"
        elif hour < 20:
            return "🌆 Akşam! Toplu taşıma hala aktif, ancak seferleri kontrol edin."
        else:
            return "🌙 Geç oldu, bazı ulaşım seçenekleri sınırlı olabilir. Dönüşünüzü buna göre planlayın."
    
    def _extract_district_from_destination(self, destination: str) -> Optional[str]:
        """
        Extract district name from destination string.
        
        This method attempts to identify district names in destination strings
        to provide relevant hidden gems recommendations.
        
        Args:
            destination: Destination string (e.g., "Taksim Square", "Karaköy", "Sultanahmet")
            
        Returns:
            District name if found, None otherwise
        """
        if not destination:
            return None
        
        destination_lower = destination.lower()
        
        # Known districts and their keywords (based on hidden gems database)
        # Keys are lowercase to match hidden_gems_database.py district keys
        district_keywords = {
            'beyoğlu': ['beyoglu', 'beyoğlu', 'taksim', 'galata', 'istiklal', 'karakoy', 'karaköy', 'cihangir'],
            'beşiktaş': ['besiktas', 'beşiktaş', 'ortakoy', 'ortaköy'],
            'kadıköy': ['kadikoy', 'kadıköy', 'moda'],
            'üsküdar': ['uskudar', 'üsküdar'],
            'fatih': ['sultanahmet', 'fatih', 'eminonu', 'eminönü'],
            'sarıyer': ['sariyer', 'sarıyer', 'emirgan', 'istinye']
        }
        
        # Check for matches
        for district, keywords in district_keywords.items():
            if any(keyword in destination_lower for keyword in keywords):
                logger.debug(f"💎 Matched destination '{destination}' to district '{district}'")
                return district
        
        # Return None if no district identified
        logger.debug(f"💎 No district match found for destination: {destination}")
        return None
