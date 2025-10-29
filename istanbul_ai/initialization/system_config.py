"""
System Configuration Module
Manages feature flags, imports, and system-wide configuration for Istanbul AI.

This module extracts all configuration and feature availability checks from main_system.py
to improve modularity and make feature management more maintainable.
"""

import logging
import sys
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlags:
    """Feature availability flags for the Istanbul AI system"""
    
    # Core Services
    hidden_gems_available: bool = False
    price_filter_available: bool = False
    events_service_available: bool = False
    weather_recommendations_available: bool = False
    
    # Advanced Systems
    advanced_transport_available: bool = False
    transfer_map_integration_available: bool = False
    ml_daily_talks_available: bool = False
    enhanced_daily_talks_available: bool = False
    neural_query_enhancement_available: bool = False
    multi_intent_handler_available: bool = False
    advanced_personalization_available: bool = False
    feedback_loop_available: bool = False
    
    # Initializers
    service_initializer_available: bool = False
    handler_initializer_available: bool = False
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert feature flags to dictionary"""
        return {
            'hidden_gems': self.hidden_gems_available,
            'price_filter': self.price_filter_available,
            'events_service': self.events_service_available,
            'weather_recommendations': self.weather_recommendations_available,
            'advanced_transport': self.advanced_transport_available,
            'transfer_map_integration': self.transfer_map_integration_available,
            'ml_daily_talks': self.ml_daily_talks_available,
            'enhanced_daily_talks': self.enhanced_daily_talks_available,
            'neural_query_enhancement': self.neural_query_enhancement_available,
            'multi_intent_handler': self.multi_intent_handler_available,
            'advanced_personalization': self.advanced_personalization_available,
            'feedback_loop': self.feedback_loop_available,
            'service_initializer': self.service_initializer_available,
            'handler_initializer': self.handler_initializer_available,
        }
    
    def get_available_count(self) -> int:
        """Get count of available features"""
        return sum(1 for flag in self.to_dict().values() if flag)
    
    def get_total_count(self) -> int:
        """Get total count of features"""
        return len(self.to_dict())
    
    def get_availability_rate(self) -> float:
        """Get feature availability rate as percentage"""
        total = self.get_total_count()
        if total == 0:
            return 0.0
        return (self.get_available_count() / total) * 100


class SystemConfig:
    """
    System Configuration Manager
    
    Handles:
    - Feature flag detection and management
    - Dynamic imports with error handling
    - System path configuration
    - Logging configuration
    """
    
    def __init__(self):
        """Initialize system configuration"""
        self.feature_flags = FeatureFlags()
        self.imports = {}
        self._setup_system_paths()
        self._detect_all_features()
    
    def _setup_system_paths(self):
        """Configure system paths for imports"""
        try:
            # Add parent directory to path for advanced modules
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                logger.debug(f"Added parent directory to sys.path: {parent_dir}")
        except Exception as e:
            logger.warning(f"Could not setup system paths: {e}")
    
    def _detect_all_features(self):
        """Detect availability of all features"""
        logger.info("🔍 Detecting system features and capabilities...")
        
        # Detect core services
        self._detect_hidden_gems()
        self._detect_price_filter()
        self._detect_events_service()
        self._detect_weather_recommendations()
        
        # Detect advanced systems
        self._detect_advanced_transport()
        self._detect_transfer_map_integration()
        self._detect_ml_daily_talks()
        self._detect_enhanced_daily_talks()
        self._detect_neural_query_enhancement()
        self._detect_multi_intent_handler()
        self._detect_advanced_personalization()
        self._detect_feedback_loop()
        
        # Detect initializers
        self._detect_initializers()
        
        # Log summary
        available = self.feature_flags.get_available_count()
        total = self.feature_flags.get_total_count()
        rate = self.feature_flags.get_availability_rate()
        logger.info(f"✅ Feature detection complete: {available}/{total} features available ({rate:.1f}%)")
    
    def _detect_hidden_gems(self):
        """Detect Hidden Gems Handler"""
        try:
            from backend.services.hidden_gems_handler import HiddenGemsHandler
            self.imports['HiddenGemsHandler'] = HiddenGemsHandler
            self.feature_flags.hidden_gems_available = True
            logger.info("✅ Hidden Gems Handler loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Hidden Gems Handler not available: {e}")
            self.feature_flags.hidden_gems_available = False
    
    def _detect_price_filter(self):
        """Detect Price Filter Service"""
        try:
            from backend.services.price_filter_service import PriceFilterService
            self.imports['PriceFilterService'] = PriceFilterService
            self.feature_flags.price_filter_available = True
            logger.info("✅ Price Filter Service loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Price Filter Service not available: {e}")
            self.feature_flags.price_filter_available = False
    
    def _detect_events_service(self):
        """Detect Events Service"""
        try:
            from backend.services.events_service import get_events_service
            self.imports['get_events_service'] = get_events_service
            self.feature_flags.events_service_available = True
            logger.info("✅ Events Service loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Events Service not available: {e}")
            self.feature_flags.events_service_available = False
    
    def _detect_weather_recommendations(self):
        """Detect Weather Recommendations Service"""
        try:
            from backend.services.weather_recommendations import get_weather_recommendations_service
            self.imports['get_weather_recommendations_service'] = get_weather_recommendations_service
            self.feature_flags.weather_recommendations_available = True
            logger.info("✅ Weather Recommendations Service loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Weather Recommendations Service not available: {e}")
            self.feature_flags.weather_recommendations_available = False
    
    def _detect_advanced_transport(self):
        """Detect Advanced Transportation System"""
        try:
            from enhanced_transportation_integration import (
                TransportationQueryProcessor,
                create_ml_enhanced_transportation_system,
                GPSLocation
            )
            self.imports['TransportationQueryProcessor'] = TransportationQueryProcessor
            self.imports['create_ml_enhanced_transportation_system'] = create_ml_enhanced_transportation_system
            self.imports['GPSLocation'] = GPSLocation
            self.feature_flags.advanced_transport_available = True
            logger.info("✅ Advanced transportation system loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Advanced transportation system not available: {e}")
            self.feature_flags.advanced_transport_available = False
    
    def _detect_transfer_map_integration(self):
        """Detect Transfer Instructions & Map Visualization"""
        try:
            from transportation_chat_integration import TransportationChatIntegration
            self.imports['TransportationChatIntegration'] = TransportationChatIntegration
            self.feature_flags.transfer_map_integration_available = True
            logger.info("✅ Transfer Instructions & Map Visualization integration loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Transfer Instructions & Map Visualization not available: {e}")
            self.feature_flags.transfer_map_integration_available = False
    
    def _detect_ml_daily_talks(self):
        """Detect ML-Enhanced Daily Talks Bridge"""
        try:
            from ml_enhanced_daily_talks_bridge import MLEnhancedDailyTalksBridge, process_enhanced_daily_talk
            self.imports['MLEnhancedDailyTalksBridge'] = MLEnhancedDailyTalksBridge
            self.imports['process_enhanced_daily_talk'] = process_enhanced_daily_talk
            self.feature_flags.ml_daily_talks_available = True
            logger.info("✅ ML-Enhanced Daily Talks Bridge loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ ML-Enhanced Daily Talks Bridge not available: {e}")
            self.feature_flags.ml_daily_talks_available = False
    
    def _detect_enhanced_daily_talks(self):
        """Detect Enhanced Bilingual Daily Talks System"""
        try:
            from enhanced_bilingual_daily_talks import (
                EnhancedBilingualDailyTalks,
                UserContext as DailyTalkContext,
                Language
            )
            self.imports['EnhancedBilingualDailyTalks'] = EnhancedBilingualDailyTalks
            self.imports['DailyTalkContext'] = DailyTalkContext
            self.imports['Language'] = Language
            self.feature_flags.enhanced_daily_talks_available = True
            logger.info("✅ Enhanced Bilingual Daily Talks System loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Enhanced Bilingual Daily Talks System not available: {e}")
            self.feature_flags.enhanced_daily_talks_available = False
    
    def _detect_neural_query_enhancement(self):
        """Detect Neural Query Enhancement System"""
        try:
            from backend.services.lightweight_neural_query_enhancement import (
                get_lightweight_neural_processor,
                LightweightNeuralInsights
            )
            self.imports['get_lightweight_neural_processor'] = get_lightweight_neural_processor
            self.imports['LightweightNeuralInsights'] = LightweightNeuralInsights
            self.feature_flags.neural_query_enhancement_available = True
            logger.info("✅ Neural Query Enhancement System loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Neural Query Enhancement System not available: {e}")
            self.feature_flags.neural_query_enhancement_available = False
    
    def _detect_multi_intent_handler(self):
        """Detect Multi-Intent Query Handler"""
        try:
            from multi_intent_query_handler import MultiIntentQueryHandler, MultiIntentResult
            self.imports['MultiIntentQueryHandler'] = MultiIntentQueryHandler
            self.imports['MultiIntentResult'] = MultiIntentResult
            self.feature_flags.multi_intent_handler_available = True
            logger.info("✅ Multi-Intent Query Handler loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Multi-Intent Query Handler not available: {e}")
            self.feature_flags.multi_intent_handler_available = False
    
    def _detect_advanced_personalization(self):
        """Detect Advanced Personalization System"""
        try:
            from backend.services.advanced_personalization import AdvancedPersonalizationSystem
            self.imports['AdvancedPersonalizationSystem'] = AdvancedPersonalizationSystem
            self.feature_flags.advanced_personalization_available = True
            logger.info("✅ Advanced Personalization System loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Advanced Personalization System not available: {e}")
            self.feature_flags.advanced_personalization_available = False
    
    def _detect_feedback_loop(self):
        """Detect Feedback Loop System"""
        try:
            from backend.services.feedback_loop import FeedbackLoopSystem
            self.imports['FeedbackLoopSystem'] = FeedbackLoopSystem
            self.feature_flags.feedback_loop_available = True
            logger.info("✅ Real-time Feedback Loop System loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Real-time Feedback Loop System not available: {e}")
            self.feature_flags.feedback_loop_available = False
    
    def _detect_initializers(self):
        """Detect ServiceInitializer and HandlerInitializer"""
        try:
            from istanbul_ai.initialization import ServiceInitializer, HandlerInitializer
            self.imports['ServiceInitializer'] = ServiceInitializer
            self.imports['HandlerInitializer'] = HandlerInitializer
            self.feature_flags.service_initializer_available = True
            self.feature_flags.handler_initializer_available = True
            logger.info("✅ ServiceInitializer and HandlerInitializer loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Initializers not available: {e}")
            self.feature_flags.service_initializer_available = False
            self.feature_flags.handler_initializer_available = False
    
    def get_import(self, name: str) -> Optional[Any]:
        """
        Get a dynamically imported module/class
        
        Args:
            name: Name of the import
            
        Returns:
            Imported module/class or None if not available
        """
        return self.imports.get(name)
    
    def is_feature_available(self, feature_name: str) -> bool:
        """
        Check if a feature is available
        
        Args:
            feature_name: Name of the feature (e.g., 'hidden_gems', 'price_filter')
            
        Returns:
            True if feature is available, False otherwise
        """
        feature_dict = self.feature_flags.to_dict()
        return feature_dict.get(feature_name, False)
    
    def get_configuration_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive configuration report
        
        Returns:
            Dictionary with configuration details
        """
        return {
            'feature_flags': self.feature_flags.to_dict(),
            'available_features': self.feature_flags.get_available_count(),
            'total_features': self.feature_flags.get_total_count(),
            'availability_rate': f"{self.feature_flags.get_availability_rate():.1f}%",
            'loaded_imports': list(self.imports.keys())
        }


# Global configuration instance (singleton pattern)
_system_config_instance: Optional[SystemConfig] = None


def get_system_config() -> SystemConfig:
    """
    Get the global system configuration instance (singleton)
    
    Returns:
        SystemConfig instance
    """
    global _system_config_instance
    if _system_config_instance is None:
        _system_config_instance = SystemConfig()
    return _system_config_instance


def reset_system_config():
    """Reset the global system configuration (mainly for testing)"""
    global _system_config_instance
    _system_config_instance = None
