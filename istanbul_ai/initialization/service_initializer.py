"""
Service Initialization Module
Handles initialization of all external services and dependencies for Istanbul AI system
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class ServiceInitializer:
    """
    Initialize and manage external services
    
    Responsibilities:
    - Initialize all external services with proper error handling
    - Track initialization status and errors
    - Provide initialization reports for debugging
    - Manage service dependencies
    """
    
    def __init__(self):
        """Initialize the service initializer"""
        self.services: Dict[str, Any] = {}
        self.initialization_errors: List[Tuple[str, Exception]] = []
        self.initialization_order: List[str] = []
    
    def initialize_all_services(self) -> Dict[str, Any]:
        """
        Initialize all services with proper error handling
        
        Returns:
            Dictionary of initialized services
        """
        logger.info("🔧 Starting service initialization...")
        
        # Core services (no dependencies)
        self.services['hidden_gems'] = self._init_hidden_gems()
        self.services['price_filter'] = self._init_price_filter()
        self.services['conversation'] = self._init_conversation_handler()
        self.services['events'] = self._init_events_service()
        self.services['weather_recommendations'] = self._init_weather_recommendations()
        self.services['location_detector'] = self._init_location_detector()
        
        # Transportation services
        self.services['transport_processor'] = self._init_transport_processor()
        self.services['ml_transport_system'] = self._init_ml_transport_system()
        self.services['transportation_chat'] = self._init_transportation_chat()
        
        # Daily talk services
        self.services['daily_talks_bridge'] = self._init_daily_talks_bridge()
        self.services['enhanced_daily_talks'] = self._init_enhanced_daily_talks()
        
        # Neural processing
        self.services['neural_processor'] = self._init_neural_processor()
        
        # Museum and attractions
        self.services['museum_generator'] = self._init_museum_system()
        self.services['hours_checker'] = self._init_hours_checker()
        self.services['museum_db'] = self._init_museum_database()
        self.services['advanced_museum_system'] = self._init_advanced_museum_system()
        self.services['advanced_attractions_system'] = self._init_advanced_attractions()
        
        # Route planning
        self.services['multi_intent_handler'] = self._init_multi_intent_handler()
        self.services['museum_route_planner'] = self._init_museum_route_planner()
        self.services['gps_route_planner'] = self._init_gps_route_planner()
        self.services['advanced_route_planner'] = self._init_advanced_route_planner()
        self.services['gps_route_service'] = self._init_gps_route_service()
        
        # Weather
        self.services['weather_client'] = self._init_weather_client()
        
        # ML Context Builder (shared across handlers)
        self.services['ml_context_builder'] = self._init_ml_context_builder()
        
        # Advanced systems
        self.services['personalization'] = self._init_personalization_system()
        self.services['feedback_loop'] = self._init_feedback_loop()
        
        # GPS and LLM integration
        self.services['gps_location_service'] = self._init_gps_location_service()
        self.services['llm_service'] = self._init_llm_service()
        
        # Transportation RAG System (for advanced route finding)
        self.services['transportation_rag'] = self._init_transportation_rag()
        
        logger.info(f"✅ Service initialization complete: {self._get_success_count()}/{len(self.services)} services initialized")
        
        return self.services
    
    # ============================================================================
    # CORE SERVICES
    # ============================================================================
    
    def _init_hidden_gems(self) -> Optional[Any]:
        """Initialize hidden gems handler"""
        try:
            from backend.services.hidden_gems_handler import HiddenGemsHandler
            service = HiddenGemsHandler()
            self._record_success('hidden_gems', "💎 Hidden Gems Handler")
            return service
        except Exception as e:
            self._record_error('hidden_gems', e)
            return None
    
    def _init_price_filter(self) -> Optional[Any]:
        """Initialize price filter service"""
        try:
            from backend.services.price_filter_service import PriceFilterService
            service = PriceFilterService()
            self._record_success('price_filter', "💰 Price Filter Service")
            return service
        except Exception as e:
            self._record_error('price_filter', e)
            return None
    
    def _init_conversation_handler(self) -> Optional[Any]:
        """Initialize conversation handler"""
        try:
            from backend.services.conversation_handler import get_conversation_handler
            service = get_conversation_handler()
            self._record_success('conversation', "💬 Conversation Handler")
            return service
        except Exception as e:
            self._record_error('conversation', e)
            return None
    
    def _init_events_service(self) -> Optional[Any]:
        """Initialize events service"""
        try:
            from backend.services.events_service import get_events_service
            service = get_events_service()
            self._record_success('events', "🎭 Events Service")
            return service
        except Exception as e:
            self._record_error('events', e)
            return None
    
    def _init_weather_recommendations(self) -> Optional[Any]:
        """Initialize weather recommendations service"""
        try:
            from backend.services.weather_recommendations import get_weather_recommendations_service
            service = get_weather_recommendations_service()
            self._record_success('weather_recommendations', "🌤️ Weather Recommendations Service")
            return service
        except Exception as e:
            self._record_error('weather_recommendations', e)
            return None
    
    def _init_location_detector(self) -> Optional[Any]:
        """Initialize intelligent location detector"""
        try:
            # Use relative import for internal service
            from istanbul_ai.services.intelligent_location_detector import IntelligentLocationDetector
            service = IntelligentLocationDetector()
            self._record_success('location_detector', "📍 Intelligent Location Detector")
            return service
        except ImportError as e:
            self._record_error('location_detector', e, warning=True)
            return None
    
    # ============================================================================
    # TRANSPORTATION SERVICES
    # ============================================================================
    
    def _init_transport_processor(self) -> Optional[Any]:
        """Initialize transportation query processor"""
        try:
            from enhanced_transportation_integration import TransportationQueryProcessor
            service = TransportationQueryProcessor()
            self._record_success('transport_processor', "🚇 Transportation Query Processor")
            return service
        except Exception as e:
            self._record_error('transport_processor', e)
            return None
    
    def _init_ml_transport_system(self) -> Optional[Any]:
        """Initialize ML-enhanced transportation system"""
        try:
            from enhanced_transportation_integration import create_ml_enhanced_transportation_system
            service = create_ml_enhanced_transportation_system()
            self._record_success('ml_transport_system', "🚇 ML-Enhanced Transportation System")
            return service
        except Exception as e:
            self._record_error('ml_transport_system', e)
            return None
    
    def _init_transportation_chat(self) -> Optional[Any]:
        """Initialize transportation chat integration"""
        try:
            from transportation_chat_integration import TransportationChatIntegration
            service = TransportationChatIntegration()
            self._record_success('transportation_chat', "🗺️ Transfer Instructions & Map Visualization")
            return service
        except Exception as e:
            self._record_error('transportation_chat', e)
            return None
    
    # ============================================================================
    # DAILY TALK SERVICES
    # ============================================================================
    
    def _init_daily_talks_bridge(self) -> Optional[Any]:
        """Initialize ML-Enhanced Daily Talks Bridge (Legacy)"""
        try:
            from ml_enhanced_daily_talks_bridge import MLEnhancedDailyTalksBridge
            service = MLEnhancedDailyTalksBridge()
            self._record_success('daily_talks_bridge', "🤖 ML-Enhanced Daily Talks Bridge (legacy)")
            return service
        except Exception as e:
            self._record_error('daily_talks_bridge', e)
            return None
    
    def _init_enhanced_daily_talks(self) -> Optional[Any]:
        """Initialize Enhanced Bilingual Daily Talks System (PRIMARY)"""
        try:
            from enhanced_bilingual_daily_talks import EnhancedBilingualDailyTalks
            service = EnhancedBilingualDailyTalks()
            self._record_success('enhanced_daily_talks', "💬 Enhanced Bilingual Daily Talks System (PRIMARY)")
            return service
        except Exception as e:
            self._record_error('enhanced_daily_talks', e)
            return None
    
    # ============================================================================
    # NEURAL PROCESSING
    # ============================================================================
    
    def _init_neural_processor(self) -> Optional[Any]:
        """Initialize lightweight neural query enhancement system"""
        try:
            from backend.services.lightweight_neural_query_enhancement import get_lightweight_neural_processor
            service = get_lightweight_neural_processor()
            self._record_success('neural_processor', "🧠 Lightweight Neural Query Enhancement System")
            return service
        except Exception as e:
            self._record_error('neural_processor', e)
            return None
    
    # ============================================================================
    # MUSEUM AND ATTRACTIONS
    # ============================================================================
    
    def _init_museum_system(self) -> Optional[Any]:
        """Initialize museum response generator"""
        try:
            import sys
            import os
            # Add parent directory to path
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            from museum_response_generator import MuseumResponseGenerator
            service = MuseumResponseGenerator()
            self._record_success('museum_generator', "🏛️ Museum Response Generator")
            return service
        except ImportError as e:
            self._record_error('museum_generator', e, warning=True)
            return None
    
    def _init_hours_checker(self) -> Optional[Any]:
        """Initialize Google Maps hours checker"""
        try:
            from google_maps_hours_checker import GoogleMapsHoursChecker
            service = GoogleMapsHoursChecker()
            self._record_success('hours_checker', "🕐 Google Maps Hours Checker")
            return service
        except ImportError as e:
            self._record_error('hours_checker', e, warning=True)
            return None
    
    def _init_museum_database(self) -> Optional[Any]:
        """Initialize museum database"""
        try:
            from updated_museum_database import UpdatedIstanbulMuseumDatabase
            service = UpdatedIstanbulMuseumDatabase()
            self._record_success('museum_db', "🏛️ Museum Database")
            return service
        except ImportError as e:
            self._record_error('museum_db', e, warning=True)
            return None
    
    def _init_advanced_museum_system(self) -> Optional[Any]:
        """Initialize advanced ML-powered museum system"""
        try:
            from museum_advising_system import IstanbulMuseumSystem
            service = IstanbulMuseumSystem()
            self._record_success('advanced_museum_system', "🎨 Advanced Museum System (ML-powered)")
            return service
        except ImportError as e:
            self._record_error('advanced_museum_system', e, warning=True)
            return None
    
    def _init_advanced_attractions(self) -> Optional[Any]:
        """Initialize advanced attractions system"""
        try:
            from istanbul_attractions_system import IstanbulAttractionsSystem
            service = IstanbulAttractionsSystem()
            self._record_success('advanced_attractions_system', "🌟 Advanced Attractions System")
            return service
        except ImportError as e:
            self._record_error('advanced_attractions_system', e, warning=True)
            return None
    
    # ============================================================================
    # ROUTE PLANNING
    # ============================================================================
    
    def _init_multi_intent_handler(self) -> Optional[Any]:
        """Initialize multi-intent query handler"""
        try:
            from multi_intent_query_handler import MultiIntentQueryHandler
            service = MultiIntentQueryHandler()
            self._record_success('multi_intent_handler', "🎯 Multi-Intent Query Handler")
            return service
        except Exception as e:
            self._record_error('multi_intent_handler', e)
            return None
    
    def _init_museum_route_planner(self) -> Optional[Any]:
        """Initialize enhanced museum route planner"""
        try:
            from enhanced_museum_route_planner import EnhancedMuseumRoutePlanner
            service = EnhancedMuseumRoutePlanner()
            self._record_success('museum_route_planner', "🗺️ Enhanced Museum Route Planner")
            return service
        except ImportError as e:
            self._record_error('museum_route_planner', e, warning=True)
            return None
    
    def _init_gps_route_planner(self) -> Optional[Any]:
        """Initialize enhanced GPS route planner"""
        try:
            from enhanced_gps_route_planner import EnhancedGPSRoutePlanner
            service = EnhancedGPSRoutePlanner()
            self._record_success('gps_route_planner', "🗺️ Enhanced GPS Route Planner")
            return service
        except ImportError as e:
            self._record_error('gps_route_planner', e, warning=True)
            return None
    
    def _init_advanced_route_planner(self) -> Optional[Any]:
        """Initialize enhanced route planner V2"""
        try:
            from enhanced_route_planner_v2 import EnhancedRoutePlannerV2
            service = EnhancedRoutePlannerV2()
            self._record_success('advanced_route_planner', "🧭 Enhanced Route Planner V2")
            return service
        except ImportError as e:
            self._record_error('advanced_route_planner', e, warning=True)
            return None
    
    def _init_gps_route_service(self) -> Optional[Any]:
        """Initialize GPS Route Service for location-based routing"""
        try:
            from istanbul_ai.services.gps_route_service import GPSRouteService
            # Pass transport processor if available for enhanced routing
            transport_processor = self.services.get('transport_processor')
            service = GPSRouteService(transport_processor=transport_processor)
            self._record_success('gps_route_service', "🗺️ GPS Route Service")
            return service
        except ImportError as e:
            self._record_error('gps_route_service', e, warning=True)
            return None
    
    # ============================================================================
    # WEATHER
    # ============================================================================
    
    def _init_weather_client(self) -> Optional[Any]:
        """Initialize enhanced weather client"""
        try:
            from backend.api_clients.enhanced_weather import EnhancedWeatherClient
            service = EnhancedWeatherClient()
            self._record_success('weather_client', "🌤️ Enhanced Weather System")
            return service
        except ImportError as e:
            self._record_error('weather_client', e, warning=True)
            return None
    
    # ============================================================================
    # ML CONTEXT BUILDER
    # ============================================================================
    
    def _init_ml_context_builder(self) -> Optional[Any]:
        """Initialize ML context builder (shared across handlers)"""
        try:
            from backend.services.ml_context_builder import MLContextBuilder
            service = MLContextBuilder()
            self._record_success('ml_context_builder', "🧠 ML Context Builder (shared)")
            return service
        except ImportError as e:
            self._record_error('ml_context_builder', e, warning=True)
            return None
    
    # ============================================================================
    # ADVANCED SYSTEMS
    # ============================================================================
    
    def _init_personalization_system(self) -> Optional[Any]:
        """Initialize advanced personalization system"""
        try:
            from backend.services.advanced_personalization import AdvancedPersonalizationSystem
            service = AdvancedPersonalizationSystem()
            self._record_success('personalization', "🎯 Advanced Personalization System")
            return service
        except Exception as e:
            self._record_error('personalization', e)
            return None
    
    def _init_feedback_loop(self) -> Optional[Any]:
        """Initialize real-time feedback loop system"""
        try:
            from backend.services.feedback_loop import FeedbackLoopSystem
            service = FeedbackLoopSystem()
            self._record_success('feedback_loop', "🔄 Real-time Feedback Loop System")
            return service
        except Exception as e:
            self._record_error('feedback_loop', e)
            return None
    
    # ============================================================================
    # GPS AND LLM INTEGRATION SERVICES
    # ============================================================================
    
    def _init_gps_location_service(self) -> Optional[Any]:
        """Initialize GPS location service for district detection"""
        try:
            from istanbul_ai.services.gps_location_service import GPSLocationService
            service = GPSLocationService()
            self._record_success('gps_location_service', "📍 GPS Location Service")
            return service
        except Exception as e:
            self._record_error('gps_location_service', e, warning=True)
            return None
    
    def _init_llm_service(self) -> Optional[Any]:
        """
        Initialize LLM service (DISABLED - Using RunPod LLM Instead)
        
        ⚠️ Local LLM loading has been disabled to use RunPod remote LLM.
        All LLM requests should go through the RunPod LLM Client in backend/main.py
        """
        # Local LLM loading disabled - use RunPod LLM client instead
        logger.info("ℹ️  Local LLM service disabled - using RunPod LLM")
        logger.info("   Endpoint: RunPod (RTX 5080 GPU)")
        return None
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _record_success(self, service_name: str, display_name: str):
        """Record successful service initialization"""
        self.initialization_order.append(service_name)
        logger.info(f"{display_name} initialized successfully!")
    
    def _record_error(self, service_name: str, error: Exception, warning: bool = False):
        """Record service initialization error"""
        self.initialization_errors.append((service_name, error))
        if warning:
            logger.warning(f"{service_name} not available: {error}")
        else:
            logger.error(f"Failed to initialize {service_name}: {error}")
    
    def _get_success_count(self) -> int:
        """Get count of successfully initialized services"""
        return sum(1 for s in self.services.values() if s is not None)
    
    def get_initialization_report(self) -> Dict[str, Any]:
        """
        Generate detailed initialization status report
        
        Returns:
            Dictionary with initialization statistics and errors
        """
        initialized = self._get_success_count()
        failed = len(self.initialization_errors)
        total = len(self.services)
        
        return {
            'total_services': total,
            'initialized': initialized,
            'failed': failed,
            'success_rate': f"{(initialized/total*100):.1f}%",
            'errors': [
                {'service': name, 'error': str(err)} 
                for name, err in self.initialization_errors
            ],
            'initialization_order': self.initialization_order,
            'available_services': [
                name for name, service in self.services.items() 
                if service is not None
            ]
        }
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a specific initialized service
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service instance or None if not available
        """
        return self.services.get(service_name)
    
    def is_service_available(self, service_name: str) -> bool:
        """
        Check if a service is available
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if service is initialized and available
        """
        return self.services.get(service_name) is not None
    
    def _init_transportation_rag(self) -> Optional[Any]:
        """Initialize Transportation RAG System for advanced route finding"""
        try:
            from backend.services.transportation_rag_system import get_transportation_rag
            service = get_transportation_rag()
            self._record_success('transportation_rag', "🚇 Transportation RAG System")
            return service
        except Exception as e:
            self._record_error('transportation_rag', e, "🚇 Transportation RAG System")
            return None
