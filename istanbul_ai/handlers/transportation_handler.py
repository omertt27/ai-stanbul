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
        transfer_map_integration_available: bool = False,
        advanced_transport_available: bool = False
    ):
        """
        Initialize transportation handler with required services.
        
        Args:
            transportation_chat: TransportationMapChat service for map visualization
            transport_processor: AdvancedTransportationProcessor for IBB API integration
            gps_route_service: GPSRouteService for GPS-based navigation
            bilingual_manager: BilingualManager for language support
            transfer_map_integration_available: Flag for transfer map feature
            advanced_transport_available: Flag for advanced transport feature
        """
        self.transportation_chat = transportation_chat
        self.transport_processor = transport_processor
        self.gps_route_service = gps_route_service
        self.bilingual_manager = bilingual_manager
        
        # Feature availability flags
        self.transfer_map_integration_available = transfer_map_integration_available
        self.advanced_transport_available = advanced_transport_available
        
        # Initialize service availability flags
        self.has_transportation_chat = transportation_chat is not None
        self.has_transport_processor = transport_processor is not None
        self.has_gps_service = gps_route_service is not None
        self.has_bilingual = bilingual_manager is not None and BILINGUAL_AVAILABLE
        
        logger.info(
            f"Transportation Handler initialized - "
            f"TransportChat: {self.has_transportation_chat}, "
            f"AdvancedTransport: {self.has_transport_processor}, "
            f"GPS: {self.has_gps_service}, "
            f"Bilingual: {self.has_bilingual}, "
            f"TransferMap: {transfer_map_integration_available}"
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
                    
                    if return_structured:
                        return {
                            'response': response_text,
                            'map_data': map_data,
                            'detailed_route': result.get('detailed_route'),
                            'alternatives': result.get('alternatives', []),
                            'fare_info': result.get('fare_info'),
                            'transfer_count': result.get('transfer_count', 0),
                            'total_time': result.get('total_time', 0),
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
            user_profile: User profile
            context: Conversation context
            return_structured: Whether to return structured response
            
        Returns:
            GPS navigation response
        """
        if not self.has_gps_service:
            logger.warning("⚠️ GPS Route Service not available, using fallback")
            return self._get_fallback_response(
                entities, user_profile, context, return_structured
            )
        
        try:
            logger.info("🗺️ Delegating to GPS Route Service")
            response = self.gps_route_service.generate_route_response(
                message, entities, user_profile, context
            )
            
            if return_structured:
                return {
                    'response': response,
                    'handler': 'transportation_handler',
                    'method': 'gps_navigation',
                    'success': True
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
