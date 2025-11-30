"""
Service Manager - Central access point for all Istanbul AI services
Provides unified interface for LLM to access all local services and data
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class ServiceManager:
    """Centralized manager for all Istanbul AI services"""
    
    def __init__(self):
        logger.info("🔧 Initializing Service Manager...")
        
        # Core data services
        self.restaurant_service = None
        self.transportation_service = None
        self.hidden_gems_service = None
        self.events_service = None
        self.attractions_service = None
        self.airport_service = None
        self.daily_life_service = None
        self.info_service = None
        self.weather_service = None  # NEW: Weather service
        
        # Intelligence services
        self.entity_extractor = None
        self.intent_classifier = None
        self.context_manager = None
        self.typo_corrector = None
        
        # Service status
        self.services_initialized = False
        self._service_errors = []
        
    def initialize_all(self):
        """Initialize all services with error handling"""
        logger.info("📦 Loading Istanbul AI services...")
        
        # Core data services
        self._init_restaurant_service()
        self._init_transportation_service()
        self._init_hidden_gems_service()
        self._init_events_service()
        self._init_attractions_service()
        self._init_airport_service()
        self._init_daily_life_service()
        self._init_info_service()
        self._init_weather_service()  # NEW: Initialize weather service
        
        # Intelligence services
        self._init_entity_extractor()
        self._init_intent_classifier()
        self._init_context_manager()
        self._init_typo_corrector()
        
        self.services_initialized = True
        
        # Report status
        status = self.get_service_status()
        active_count = sum(1 for v in status.values() if v)
        total_count = len(status)
        
        logger.info(f"🎉 Service Manager initialized: {active_count}/{total_count} services active")
        
        if self._service_errors:
            logger.warning(f"⚠️ Some services failed to load: {len(self._service_errors)} errors")
            for error in self._service_errors:
                logger.warning(f"  - {error}")
    
    def _init_restaurant_service(self):
        """Initialize restaurant service"""
        try:
            from services.restaurant_database_service import RestaurantDatabaseService
            self.restaurant_service = RestaurantDatabaseService()
            logger.info("✅ Restaurant service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Restaurant service not available: {e}")
            self._service_errors.append(f"restaurant_service: {e}")
    
    def _init_transportation_service(self):
        """Initialize transportation service"""
        try:
            from services.transportation_directions_service import TransportationDirectionsService
            self.transportation_service = TransportationDirectionsService()
            logger.info("✅ Transportation service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Transportation service not available: {e}")
            self._service_errors.append(f"transportation_service: {e}")
    
    def _init_hidden_gems_service(self):
        """Initialize hidden gems service"""
        try:
            from services.hidden_gems_service import HiddenGemsService
            self.hidden_gems_service = HiddenGemsService()
            logger.info("✅ Hidden gems service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Hidden gems service not available: {e}")
            self._service_errors.append(f"hidden_gems_service: {e}")
    
    def _init_events_service(self):
        """Initialize events service"""
        try:
            from services.events_service import EventsService
            self.events_service = EventsService()
            logger.info("✅ Events service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Events service not available: {e}")
            self._service_errors.append(f"events_service: {e}")
    
    def _init_attractions_service(self):
        """Initialize attractions service"""
        try:
            from services.enhanced_attractions_service import EnhancedAttractionsService
            self.attractions_service = EnhancedAttractionsService()
            logger.info("✅ Attractions service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Attractions service not available: {e}")
            self._service_errors.append(f"attractions_service: {e}")
    
    def _init_airport_service(self):
        """Initialize airport transport service"""
        try:
            from services.airport_transport_service import IstanbulAirportTransportService
            self.airport_service = IstanbulAirportTransportService()
            logger.info("✅ Airport transport service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Airport service not available: {e}")
            self._service_errors.append(f"airport_service: {e}")
    
    def _init_daily_life_service(self):
        """Initialize daily life suggestions service"""
        try:
            from services.daily_life_suggestions_service import DailyLifeSuggestionsService
            self.daily_life_service = DailyLifeSuggestionsService()
            logger.info("✅ Daily life service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Daily life service not available: {e}")
            self._service_errors.append(f"daily_life_service: {e}")
    
    def _init_info_service(self):
        """Initialize info retrieval service"""
        try:
            from services.info_retrieval_service import InfoRetrievalService
            self.info_service = InfoRetrievalService()
            logger.info("✅ Info retrieval service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Info service not available: {e}")
            self._service_errors.append(f"info_service: {e}")
    
    def _init_weather_service(self):
        """Initialize weather service"""  # NEW: Weather service initialization
        try:
            from services.weather_service import WeatherService
            self.weather_service = WeatherService()
            logger.info("✅ Weather service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Weather service not available: {e}")
            self._service_errors.append(f"weather_service: {e}")
    
    def _init_entity_extractor(self):
        """Initialize entity extractor"""
        try:
            from services.entity_extractor import EntityExtractor
            self.entity_extractor = EntityExtractor()
            logger.info("✅ Entity extractor loaded")
        except Exception as e:
            logger.warning(f"⚠️ Entity extractor not available: {e}")
            self._service_errors.append(f"entity_extractor: {e}")
    
    def _init_intent_classifier(self):
        """Initialize intent classifier"""
        try:
            from services.enhanced_intent_classifier import EnhancedIntentClassifier
            self.intent_classifier = EnhancedIntentClassifier()
            logger.info("✅ Intent classifier loaded")
        except Exception as e:
            logger.warning(f"⚠️ Intent classifier not available: {e}")
            self._service_errors.append(f"intent_classifier: {e}")
    
    def _init_context_manager(self):
        """Initialize conversation context manager"""
        try:
            from services.conversation_context_manager import ConversationContextManager
            self.context_manager = ConversationContextManager()
            logger.info("✅ Context manager loaded")
        except Exception as e:
            logger.warning(f"⚠️ Context manager not available: {e}")
            self._service_errors.append(f"context_manager: {e}")
    
    def _init_typo_corrector(self):
        """Initialize Turkish typo corrector"""
        try:
            from services.turkish_typo_corrector import TurkishTypoCorrector
            self.typo_corrector = TurkishTypoCorrector()
            logger.info("✅ Typo corrector loaded")
        except Exception as e:
            logger.warning(f"⚠️ Typo corrector not available: {e}")
            self._service_errors.append(f"typo_corrector: {e}")
    
    # Service accessor methods
    
    def get_restaurant_recommendations(self, query: str, filters: Dict[str, Any] = None) -> Optional[List]:
        """Get restaurant recommendations"""
        if self.restaurant_service:
            try:
                return self.restaurant_service.search_restaurants(query, filters)
            except Exception as e:
                logger.error(f"Error getting restaurant recommendations: {e}")
        return None
    
    def get_transportation_directions(self, origin: str, destination: str, mode: str = "transit") -> Optional[Dict]:
        """Get transportation directions"""
        if self.transportation_service:
            try:
                return self.transportation_service.get_directions(origin, destination, mode)
            except Exception as e:
                logger.error(f"Error getting transportation directions: {e}")
        return None
    
    def get_hidden_gems(self, district: str = None, category: str = None) -> Optional[List]:
        """Get hidden gems recommendations"""
        if self.hidden_gems_service:
            try:
                return self.hidden_gems_service.search_gems(district=district, category=category)
            except Exception as e:
                logger.error(f"Error getting hidden gems: {e}")
        return None
    
    def get_events(self, date_range: str = None, category: str = None) -> Optional[List]:
        """Get events"""
        if self.events_service:
            try:
                return self.events_service.get_events(date_range, category)
            except Exception as e:
                logger.error(f"Error getting events: {e}")
        return None
    
    def get_attractions(self, category: str = None, district: str = None) -> Optional[List]:
        """Get attractions"""
        if self.attractions_service:
            try:
                return self.attractions_service.search_attractions(category, district)
            except Exception as e:
                logger.error(f"Error getting attractions: {e}")
        return None
    
    def get_airport_transport(self, airport_code: str, destination: str = None) -> Optional[Dict]:
        """Get airport transport options"""
        if self.airport_service:
            try:
                return self.airport_service.get_transport_options(airport_code, destination)
            except Exception as e:
                logger.error(f"Error getting airport transport: {e}")
        return None
    
    def extract_entities(self, query: str) -> Dict:
        """Extract entities from query"""
        if self.entity_extractor:
            try:
                return self.entity_extractor.extract(query)
            except Exception as e:
                logger.error(f"Error extracting entities: {e}")
        return {}
    
    def classify_intent(self, query: str) -> str:
        """Classify query intent"""
        if self.intent_classifier:
            try:
                return self.intent_classifier.classify(query)
            except Exception as e:
                logger.error(f"Error classifying intent: {e}")
        return "general"
    
    def correct_typos(self, query: str) -> str:
        """Correct Turkish typos"""
        if self.typo_corrector:
            try:
                return self.typo_corrector.correct(query)
            except Exception as e:
                logger.error(f"Error correcting typos: {e}")
        return query
    
    def get_service_status(self) -> Dict[str, bool]:
        """Get status of all services"""
        return {
            "restaurant_service": self.restaurant_service is not None,
            "transportation_service": self.transportation_service is not None,
            "hidden_gems_service": self.hidden_gems_service is not None,
            "events_service": self.events_service is not None,
            "attractions_service": self.attractions_service is not None,
            "airport_service": self.airport_service is not None,
            "daily_life_service": self.daily_life_service is not None,
            "info_service": self.info_service is not None,
            "weather_service": self.weather_service is not None,  # NEW: Weather service status
            "entity_extractor": self.entity_extractor is not None,
            "intent_classifier": self.intent_classifier is not None,
            "context_manager": self.context_manager is not None,
            "typo_corrector": self.typo_corrector is not None,
        }


# Global service manager instance
service_manager = ServiceManager()
