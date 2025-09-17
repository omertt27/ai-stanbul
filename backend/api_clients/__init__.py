"""
API Clients package for AI Istanbul Travel Guide

This package contains various API client modules for external services
and advanced AI capabilities.
"""

# Enhanced API Clients
from .enhanced_google_places import EnhancedGooglePlacesClient
from .google_weather import GoogleWeatherClient
from .enhanced_weather import EnhancedWeatherClient
from .enhanced_api_service import EnhancedAPIService
from .istanbul_transport import IstanbulTransportClient

# Language Processing and AI
from .language_processing import (
    AdvancedLanguageProcessor,
    IntentResult,
    EntityExtractionResult
)

from .multimodal_ai import MultimodalAIService
from .realtime_data import RealTimeDataAggregator  
from .predictive_analytics import PredictiveAnalyticsService

__all__ = [
    # Enhanced API Clients
    'EnhancedGooglePlacesClient',
    'GoogleWeatherClient',
    'EnhancedWeatherClient', 
    'EnhancedAPIService',
    'IstanbulTransportClient',
    # Language Processing and AI
    'AdvancedLanguageProcessor',
    'IntentResult', 
    'EntityExtractionResult',
    'MultimodalAIService',
    'RealTimeDataAggregator',
    'PredictiveAnalyticsService'
]
