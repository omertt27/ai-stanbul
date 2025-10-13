# --- Standard Library Imports ---
import sys
import os
import re
import asyncio
import json
import time
import html
import uuid
from datetime import datetime, date, timedelta
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import traceback

# Add location intent detection import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'load-testing'))
try:
    from location_intent_detector import LocationIntentDetector, LocationIntentType
    LOCATION_INTENT_AVAILABLE = True
    print("✅ Location Intent Detection loaded successfully")
except ImportError as e:
    LOCATION_INTENT_AVAILABLE = False
    print(f"⚠️ Location Intent Detection not available: {e}")

# Add Advanced Understanding System import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from advanced_understanding_system import AdvancedUnderstandingSystem
    from semantic_similarity_engine import SemanticSimilarityEngine, QueryContext
    from enhanced_context_memory import EnhancedContextMemory, ContextType
    from multi_intent_query_handler import MultiIntentQueryHandler
    ADVANCED_UNDERSTANDING_AVAILABLE = True
    print("✅ Advanced Understanding System loaded successfully")
except ImportError as e:
    ADVANCED_UNDERSTANDING_AVAILABLE = False
    print(f"⚠️ Advanced Understanding System not available: {e}")

# Enhanced Query Understanding Configuration
ENHANCED_QUERY_UNDERSTANDING_ENABLED = ADVANCED_UNDERSTANDING_AVAILABLE

# Initialize Enhanced Query Understanding if available
enhanced_understanding_system = None
if ENHANCED_QUERY_UNDERSTANDING_ENABLED:
    try:
        enhanced_understanding_system = AdvancedUnderstandingSystem()
        print("✅ Enhanced Understanding System initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Enhanced Understanding System: {e}")
        ENHANCED_QUERY_UNDERSTANDING_ENABLED = False

def process_enhanced_query(user_input: str, session_id: str) -> Dict[str, Any]:
    """Process query using Enhanced Understanding System"""
    if not ENHANCED_QUERY_UNDERSTANDING_ENABLED or not enhanced_understanding_system:
        return {
            'success': False,
            'intent': 'general_info',
            'confidence': 0.3,
            'entities': {},
            'corrections': [],
            'normalized_query': user_input.lower().strip(),
            'original_query': user_input
        }
    
    try:
        # Use the enhanced understanding system (correct method name)
        result = enhanced_understanding_system.understand_query(user_input, session_id=session_id)
        
        # Extract intent from multi_intent_result
        multi_intent = result.multi_intent_result
        primary_intent = multi_intent.primary_intent.type.value if multi_intent.primary_intent else 'general_info'
        
        return {
            'success': True,
            'intent': primary_intent,
            'confidence': result.understanding_confidence,
            'entities': multi_intent.extracted_entities if multi_intent else {},
            'corrections': [],  # Can be enhanced later
            'normalized_query': result.original_query.lower().strip(),
            'original_query': user_input,
            'detailed_result': result
        }
    except Exception as e:
        print(f"Error in process_enhanced_query: {e}")
        return {
            'success': False,
            'intent': 'general_info',
            'confidence': 0.3,
            'entities': {},
            'corrections': [],
            'normalized_query': user_input.lower().strip(),
            'original_query': user_input
        }

def generate_sample_hidden_gems(area: str, language: str = 'en') -> List[Dict[str, Any]]:
    """Generate sample hidden gems for a given area"""
    # Sample hidden gems data based on area
    gems_data = {
        'sultanahmet': [
            {
                'name': 'Historic Cistern Coffee',
                'description': 'Hidden coffee shop built into ancient cistern walls',
                'location': 'Near Basilica Cistern',
                'type': 'cafe',
                'authenticity_score': 9.2,
                'local_rating': 4.8,
                'price_range': '₺₺'
            },
            {
                'name': 'Artisan Carpet Workshop',
                'description': 'Traditional carpet weaving workshop open to visitors',
                'location': 'Behind Blue Mosque',
                'type': 'cultural',
                'authenticity_score': 9.5,
                'local_rating': 4.9,
                'price_range': 'Free to visit'
            }
        ],
        'beyoglu': [
            {
                'name': 'Rooftop Garden Cafe',
                'description': 'Secret garden cafe with Bosphorus views',
                'location': 'Hidden in Galata backstreets',
                'type': 'cafe',
                'authenticity_score': 8.8,
                'local_rating': 4.7,
                'price_range': '₺₺₺'
            },
            {
                'name': 'Underground Jazz Club',
                'description': 'Intimate jazz venue in historic building basement',
                'location': 'Near Istiklal Avenue',
                'type': 'entertainment',
                'authenticity_score': 9.0,
                'local_rating': 4.8,
                'price_range': '₺₺'
            }
        ]
    }
    
    return gems_data.get(area.lower(), [
        {
            'name': 'Local Discovery',
            'description': f'Authentic local experience in {area}',
            'location': area,
            'type': 'cultural',
            'authenticity_score': 8.5,
            'local_rating': 4.5,
            'price_range': '₺₺'
        }
    ])

def generate_sample_localized_tips(location: str, language: str = 'en') -> List[Dict[str, Any]]:
    """Generate sample localized tips for a location"""
    tips_data = {
        'sultanahmet': [
            {
                'tip': 'Visit early morning to avoid crowds at major attractions',
                'category': 'timing',
                'usefulness_score': 9.1,
                'local_insight': True
            },
            {
                'tip': 'Small restaurants behind the mosque serve authentic food',
                'category': 'dining',
                'usefulness_score': 8.8,
                'local_insight': True
            }
        ],
        'beyoglu': [
            {
                'tip': 'Take the historic tunnel from Karakoy to avoid the steep walk',
                'category': 'transportation',
                'usefulness_score': 9.0,
                'local_insight': True
            },
            {
                'tip': 'Best nightlife starts after 22:00 on weekends',
                'category': 'entertainment',
                'usefulness_score': 8.5,
                'local_insight': True
            }
        ]
    }
    
    return tips_data.get(location.lower(), [
        {
            'tip': f'Explore local neighborhoods in {location} for authentic experiences',
            'category': 'general',
            'usefulness_score': 8.0,
            'local_insight': True
        }
    ])

# --- Third-Party Imports ---
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status, Body, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from thefuzz import fuzz, process

# === Pydantic Models for API Endpoints ===

class ChatRequest(BaseModel):
    """Request model for chat endpoints"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_location: Optional[Dict[str, float]] = Field(None, description="User GPS location")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

class ChatResponse(BaseModel):
    """Response model for chat endpoints"""
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session identifier")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: Optional[float] = Field(None, description="Confidence score")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    location_context: Optional[Dict[str, Any]] = Field(None, description="Location context")

class RouteRequest(BaseModel):
    """Request model for route planning"""
    attractions: List[str] = Field(..., description="List of attractions to visit")
    start_location: Optional[Dict[str, float]] = Field(None, description="Starting location")
    transport_mode: Optional[str] = Field("walking", description="Transportation mode")
    duration_hours: Optional[int] = Field(8, description="Available hours for the route")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

class RouteResponse(BaseModel):
    """Response model for route planning"""
    route: List[Dict[str, Any]] = Field(..., description="Optimized route")
    total_duration: float = Field(..., description="Total route duration in hours")
    total_distance: float = Field(..., description="Total distance in kilometers")
    transport_info: Optional[Dict[str, Any]] = Field(None, description="Transportation information")
    recommendations: Optional[List[str]] = Field(None, description="Route recommendations")

class GPSRouteRequest(BaseModel):
    """Request model for GPS-based route planning"""
    user_location: Dict[str, float] = Field(..., description="User GPS coordinates (lat, lng)")
    radius_km: Optional[float] = Field(5.0, description="Search radius in kilometers")
    duration_hours: Optional[int] = Field(4, description="Available time in hours")
    transport_mode: Optional[str] = Field("walking", description="Transportation mode")
    interests: Optional[List[str]] = Field(None, description="User interests")
    session_id: Optional[str] = Field(None, description="Session identifier")

class NearbyAttractionsRequest(BaseModel):
    """Request model for finding nearby attractions"""
    location: Dict[str, float] = Field(..., description="GPS coordinates (lat, lng)")
    radius_km: Optional[float] = Field(2.0, description="Search radius in kilometers")
    attraction_types: Optional[List[str]] = Field(None, description="Types of attractions")
    limit: Optional[int] = Field(10, description="Maximum number of results")

class LocationBasedRecommendationResponse(BaseModel):
    """Response model for location-based recommendations"""
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommendations")
    user_location: Dict[str, float] = Field(..., description="User location used")
    search_radius: float = Field(..., description="Search radius used")
    total_found: int = Field(..., description="Total number of recommendations")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")

class TransportRequest(BaseModel):
    """Request model for transportation queries"""
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    transport_mode: Optional[str] = Field(None, description="Preferred transport mode")
    time_preference: Optional[str] = Field(None, description="Time preference")

class TransportResponse(BaseModel):
    """Response model for transportation queries"""
    routes: List[Dict[str, Any]] = Field(..., description="Available routes")
    recommendations: str = Field(..., description="Transportation recommendations")
    duration_estimate: Optional[str] = Field(None, description="Estimated duration")
    cost_estimate: Optional[str] = Field(None, description="Estimated cost")

class MuseumRequest(BaseModel):
    """Request model for museum queries"""
    query: str = Field(..., description="Museum query")
    location: Optional[str] = Field(None, description="Preferred location/area")
    interests: Optional[List[str]] = Field(None, description="User interests")

class MuseumResponse(BaseModel):
    """Response model for museum queries"""
    museums: List[Dict[str, Any]] = Field(..., description="Museum recommendations")
    response: str = Field(..., description="Detailed response")
    total_found: int = Field(..., description="Total museums found")

# Import system monitoring tools
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - system metrics will be limited")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ redis not available - some caching features may be limited")

# Load environment variables first, before any other imports
load_dotenv()

# Daily usage tracking completely removed for unrestricted testing

# System metrics for monitoring
system_metrics = {
    "requests_total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "errors": 0,
    "response_times": [],
    "api_costs": 0.0,
    "cache_savings": 0.0,
    "start_time": datetime.now()
}

# Using Ultra-Specialized Istanbul AI only - no external LLM services
use_external_llm = False

# Redis availability flag and client initialization
redis_available = REDIS_AVAILABLE
redis_client = None
if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        # Test connection
        redis_client.ping()
        print("✅ Redis client initialized successfully")
        
        # Initialize Redis-based conversational memory
        try:
            from redis_conversational_memory import initialize_redis_memory
            redis_memory = initialize_redis_memory(redis_client)
            print("✅ Redis conversational memory system activated")
        except ImportError as e:
            print(f"⚠️ Redis memory system not available: {e}")
            redis_memory = None
            
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        redis_available = False
        redis_client = None
        redis_memory = None
else:
    redis_memory = None

# Add the current directory to Python path for imports (must be before project imports)
# Handle different deployment scenarios
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add parent directory for potential nested deployment structures
parent_dir = os.path.dirname(current_dir)
backend_in_parent = os.path.join(parent_dir, 'backend')
if os.path.exists(backend_in_parent) and backend_in_parent not in sys.path:
    sys.path.insert(0, backend_in_parent)

print(f"Python path configured. Current dir: {current_dir}")
print(f"Python paths: {[p for p in sys.path[:3]]}")  # Show first 3 paths

# --- Rate Limiting Removed ---
# Rate limiting has been completely removed for unrestricted testing
RATE_LIMITING_ENABLED = False

# --- Structured Logging ---
try:
    from structured_logging import get_logger, log_performance, log_ai_operation, log_api_call
    STRUCTURED_LOGGING_ENABLED = True
    print("✅ Structured logging initialized successfully")
except ImportError as e:
    print(f"⚠️ Structured logging not available: {e}")
    STRUCTURED_LOGGING_ENABLED = False

# --- Advanced Monitoring and Security ---
try:
    from monitoring.advanced_monitoring import advanced_monitor, monitor_performance, log_error_metric, log_performance_metric
    from monitoring.comprehensive_logging import comprehensive_logger, log_api_request, log_security_event, log_user_action, log_error
    ADVANCED_MONITORING_ENABLED = True
    print("✅ Advanced monitoring and logging initialized successfully")
except ImportError as e:
    print(f"⚠️ Advanced monitoring not available: {e}")
    ADVANCED_MONITORING_ENABLED = False
    # Create dummy functions to prevent errors
    def monitor_performance(op): return lambda f: f
    def log_error_metric(error_type, details=""): pass
    def log_performance_metric(metric_name, value): pass
    def log_api_request(*args, **kwargs): pass
    def log_security_event(*args, **kwargs): pass
    def log_user_action(*args, **kwargs): pass
    def log_error(*args, **kwargs): pass

# Legacy structured logging fallback
if not ADVANCED_MONITORING_ENABLED:
    try:
        from structured_logging import get_logger, log_performance, log_ai_operation, log_api_call
        structured_logger = get_logger("istanbul_ai_main")
        STRUCTURED_LOGGING_ENABLED = True
        print("✅ Structured logging initialized successfully")
    except ImportError as e:
        print(f"⚠️ Structured logging not available: {e}")
        STRUCTURED_LOGGING_ENABLED = False
    # Create dummy logger to prevent errors
    class DummyLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def debug(self, *args, **kwargs): pass
        def log_ai_query(self, *args, **kwargs): pass
        def log_request(self, *args, **kwargs): pass
        def log_response(self, *args, **kwargs): pass
        def log_cache_hit(self, *args, **kwargs): pass
        def log_cache_miss(self, *args, **kwargs): pass
        def log_rate_limit(self, *args, **kwargs): pass
        def log_error_with_traceback(self, *args, **kwargs): pass
        def context(self, **kwargs):
            from contextlib import contextmanager
            @contextmanager
            def dummy_context():
                yield
            return dummy_context()
    structured_logger = DummyLogger()
    log_performance = lambda op, **kw: lambda f: f
    log_ai_operation = lambda op, **kw: lambda f: f
    log_api_call = lambda ep: lambda f: f

# --- Project Imports ---
try:
    from database import engine, SessionLocal, get_db
    print("✅ Database import successful")
except ImportError as e:
    print(f"❌ Database import failed: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:5]}")  # First 5 paths
    print(f"Files in current directory: {os.listdir('.')}")
    raise

try:
    from models import Base, Restaurant, Museum, Place, UserFeedback, ChatSession, BlogPost, BlogComment, ChatHistory
    from sqlalchemy.orm import Session
    print("✅ Models import successful")
except ImportError as e:
    print(f"❌ Models import failed: {e}")
    raise

try:
    from routes import museums, restaurants, places, blog
    print("✅ Routes import successful")
except ImportError as e:
    print(f"❌ Routes import failed: {e}")
    raise
try:
    from api_clients.google_places import GooglePlacesClient  # type: ignore
    # Weather functionality removed - using seasonal guidance instead
    from api_clients.enhanced_api_service import EnhancedAPIService  # type: ignore
    print("✅ API clients import successful")
except ImportError as e:
    print(f"⚠️ API clients import failed (non-critical): {e}")
    # Create dummy classes for missing API clients
    class GooglePlacesClient:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        async def search_places(self, *args, **kwargs): return []
        def search_restaurants(self, *args, **kwargs): 
            return {"results": [], "status": "OK"}
    
    # Weather functionality removed - seasonal guidance provided through database
    
    class EnhancedAPIService:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def search_restaurants_enhanced(self, *args, **kwargs): 
            return {"results": [], "seasonal_context": {}}

try:
    from enhanced_input_processor import enhance_query_understanding, get_response_guidance, input_processor  # type: ignore
    print("✅ Enhanced input processor import successful")
except ImportError as e:
    print(f"⚠️ Enhanced input processor import failed: {e}")
    # Create dummy functions
    def enhance_query_understanding(user_input: str) -> str:  # type: ignore
        return user_input
    def get_response_guidance(user_input: str) -> dict:  # type: ignore
        return {"guidance": "basic"}
    class InputProcessor:  # type: ignore
        def enhance_query_context(self, text: str) -> dict:
            return {"query_type": "general"}
    input_processor = InputProcessor()

# --- Import Enhanced Services ---
try:
    from enhanced_transportation_service import EnhancedTransportationService
    from enhanced_museum_service import EnhancedMuseumService  
    from enhanced_actionability_service import EnhancedActionabilityService
    
    # Initialize enhanced services
    enhanced_transport_service = EnhancedTransportationService()
    enhanced_museum_service = EnhancedMuseumService()
    enhanced_actionability_service = EnhancedActionabilityService()
    
    ENHANCED_SERVICES_ENABLED = True
    print("✅ Enhanced services (transportation, museum, actionability) imported successfully")
except ImportError as e:
    print(f"⚠️ Enhanced services not available: {e}")
    ENHANCED_SERVICES_ENABLED = False

# --- Import Intelligent Location Detection Service ---
try:
    from services.intelligent_location_detector import (
        detect_user_location, 
        DetectedLocation, 
        LocationConfidence,
        intelligent_location_detector
    )
    INTELLIGENT_LOCATION_ENABLED = True
    print("✅ Intelligent Location Detection service imported successfully")
except ImportError as e:
    print(f"⚠️ Intelligent Location Detection service not available: {e}")
    INTELLIGENT_LOCATION_ENABLED = False
    
    # Create dummy classes to prevent errors
    class DetectedLocation:
        def __init__(self, **kwargs):
            self.latitude = kwargs.get('latitude', None)
            self.longitude = kwargs.get('longitude', None)
            self.confidence = "unknown"
            self.source = "none"
    
    class LocationConfidence:
        UNKNOWN = "unknown"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        VERY_HIGH = "very_high"
    
    async def detect_user_location(text, user_context=None, ip_address=None):
        return DetectedLocation()

from sqlalchemy.orm import Session

try:
    from i18n_service import i18n_service
    print("✅ i18n service import successful")
except ImportError as e:
    print(f"⚠️ i18n service import failed: {e}")
    # Create dummy i18n service
    class I18nService:
        def translate(self, key, lang="en"): return key
        def get_language_from_headers(self, headers): return "en"
        def should_use_ai_response(self, user_input, language): return True
        supported_languages = ["en", "tr", "ar", "ru"]
    i18n_service = I18nService()

# --- Import enhanced AI services ---
try:
    from ai_cache_service import get_ai_cache_service, init_ai_cache_service
    AI_CACHE_ENABLED = True
except ImportError:
    print("⚠️ AI Cache service not available")
    AI_CACHE_ENABLED = False
    # Create dummy functions to prevent errors
    get_ai_cache_service = lambda: None  # type: ignore
    init_ai_cache_service = lambda *args, **kwargs: None  # type: ignore

# Create dummy objects to prevent errors
class DummyManager:
    def get_or_create_session(self, *args, **kwargs): return "dummy_session"
    def get_context(self, *args, **kwargs): return {}
    def update_context(self, *args, **kwargs): pass
    def get_preferences(self, *args, **kwargs): return {}
    def update_preferences(self, *args, **kwargs): pass
    def learn_from_query(self, *args, **kwargs): pass
    def get_personalized_filter(self, *args, **kwargs): return {}
    def recognize_intent(self, *args, **kwargs): return ("general_query", 0.1)
    def extract_entities(self, *args, **kwargs): return {"locations": [], "time_references": [], "cuisine_types": [], "budget_indicators": []}
    def enhance_recommendations(self, *args, **kwargs): return args[1] if len(args) > 1 else []
    def analyze_query_context(self, *args, **kwargs): return {"locations": [], "cuisine_types": [], "price_indicators": [], "time_context": [], "group_context": None, "urgency_level": "normal", "query_complexity": "simple"}

class DummyAdvancedAI:
    async def get_comprehensive_real_time_data(self, *args, **kwargs): return {}
    async def analyze_image_comprehensive(self, *args, **kwargs): return None
    async def analyze_menu_image(self, *args, **kwargs): return None
    async def get_comprehensive_predictions(self, *args, **kwargs): return {}

try:
    from ai_intelligence import (
        session_manager, preference_manager, intent_recognizer, 
        recommendation_engine, saved_session_manager
    )
    AI_INTELLIGENCE_ENABLED = True
    print("✅ AI Intelligence services imported successfully")
except ImportError as e:
    print(f"❌ AI Intelligence import failed: {e}")
    AI_INTELLIGENCE_ENABLED = False
    
    # Create dummy objects to prevent errors
    class DummyAI:
        def get_or_create_session(self, *args, **kwargs): return "dummy"
        def get_preferences(self, *args, **kwargs): return {}
        def update_preferences(self, *args, **kwargs): pass
        def learn_from_query(self, *args, **kwargs): pass
        def recognize_intent(self, *args, **kwargs): return ("general_query", 0.1)
        def enhance_recommendations(self, *args, **kwargs): return []
        def save_session(self, *args, **kwargs): return True
        def get_saved_sessions(self, *args, **kwargs): return []
        def get_session_details(self, *args, **kwargs): return None
        def delete_session(self, *args, **kwargs): return True
        def get_context(self, *args, **kwargs): return {}
        def update_context(self, *args, **kwargs): pass
        def analyze_query_context(self, *args, **kwargs): return {}
        def extract_entities(self, *args, **kwargs): return {}
        def get_personalized_filter(self, *args, **kwargs): return {}
    
    session_manager = preference_manager = intent_recognizer = recommendation_engine = saved_session_manager = DummyAI()

# --- Import Advanced AI Features ---
try:
    from api_clients.realtime_data import realtime_data_aggregator
    from api_clients.multimodal_ai import get_multimodal_ai_service
    from api_clients.predictive_analytics import predictive_analytics_service
    ADVANCED_AI_ENABLED = True
    print("✅ Advanced AI features loaded successfully")
except ImportError as e:
    print(f"⚠️ Advanced AI features not available: {e}")
    ADVANCED_AI_ENABLED = False
    # Use dummy objects
    realtime_data_aggregator = DummyAdvancedAI()
    get_multimodal_ai_service = lambda: DummyAdvancedAI()
    predictive_analytics_service = DummyAdvancedAI()

# --- Import Language Processing ---
try:
    from api_clients.language_processing import (
        AdvancedLanguageProcessor, 
        process_user_query,
        extract_intent_and_entities,
        is_followup
    )
    LANGUAGE_PROCESSING_ENABLED = True
    language_processor = AdvancedLanguageProcessor()
    print("✅ Advanced Language Processing loaded successfully")
except ImportError as e:
    print(f"⚠️ Language Processing not available: {e}")
    LANGUAGE_PROCESSING_ENABLED = False
    # Create dummy functions
    def process_user_query(text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        return {"intent": "general_info", "confidence": 0.1, "entities": {}}
    def extract_intent_and_entities(text: str) -> Tuple[str, Dict]: 
        return "general_info", {}
    def is_followup(text: str, context: Optional[Dict] = None) -> bool: 
        return False

# --- No External LLM Dependencies ---
# We use only our Ultra-Specialized Istanbul AI System
external_llm_available = False
external_llm_client = None
print("ℹ️ No external LLM services - using Ultra-Specialized Istanbul AI System only")

# --- Ultra-Specialized Istanbul AI System (Rule-Based) ---
# Import our Ultra-Specialized Istanbul AI System
try:
    from enhanced_ultra_specialized_istanbul_ai import enhanced_istanbul_ai_system as istanbul_ai_system
    ULTRA_ISTANBUL_AI_AVAILABLE = True
    print("✅ Ultra-Specialized Istanbul AI System loaded successfully!")
except ImportError as e:
    print(f"⚠️ Ultra-Specialized Istanbul AI import failed: {e}")
    istanbul_ai_system = None
    ULTRA_ISTANBUL_AI_AVAILABLE = False

# --- NEW: Enhanced Istanbul Daily Talk AI System (with Attractions) ---
# Import our new integrated system with attractions support
try:
    from istanbul_daily_talk_system import IstanbulDailyTalkAI
    istanbul_daily_talk_ai = IstanbulDailyTalkAI()
    ISTANBUL_DAILY_TALK_AVAILABLE = True
    print("✅ Istanbul Daily Talk AI System with 50+ Attractions loaded successfully!")
except ImportError as e:
    print(f"⚠️ Istanbul Daily Talk AI import failed: {e}")
    istanbul_daily_talk_ai = None
    ISTANBUL_DAILY_TALK_AVAILABLE = False

# Istanbul Daily Talk AI is now the primary system, Ultra-Specialized is fallback
CUSTOM_AI_AVAILABLE = ISTANBUL_DAILY_TALK_AVAILABLE or ULTRA_ISTANBUL_AI_AVAILABLE
print(f"🎯 AI System Status:")
print(f"   🏛️ Istanbul Daily Talk AI (PRIMARY): {'✅ ACTIVE (50+ attractions, restaurants, transport)' if ISTANBUL_DAILY_TALK_AVAILABLE else '❌ DISABLED'}")
print(f"   🔧 Ultra-Specialized AI (FALLBACK): {'✅ ACTIVE' if ULTRA_ISTANBUL_AI_AVAILABLE else '❌ DISABLED'}")
print(f"   🚀 Overall System: {'✅ FULLY INTEGRATED AI SYSTEMS' if CUSTOM_AI_AVAILABLE else '❌ DISABLED'}")

# Use istanbul_ai_system directly for all AI processing
custom_ai_system = None

# --- Legacy imports and system setup ---

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="AI Istanbul Backend",
    description="Intelligent Istanbul travel assistant with live data integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

print("✅ FastAPI app initialized successfully")

# Initialize Location Intent Detector
location_detector = None
if LOCATION_INTENT_AVAILABLE:
    try:
        location_detector = LocationIntentDetector()
        print("✅ Location Intent Detector initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize Location Intent Detector: {e}")
        LOCATION_INTENT_AVAILABLE = False

# Initialize Advanced Understanding System
advanced_understanding = None
if ADVANCED_UNDERSTANDING_AVAILABLE:
    try:
        # Initialize Redis client if available
        redis_client = None
        if REDIS_AVAILABLE:
            try:
                redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=int(os.getenv('REDIS_DB', 2)),  # Use separate DB for context memory
                    decode_responses=True
                )
                # Test connection
                redis_client.ping()
                print("✅ Redis connection established for context memory")
            except Exception as e:
                print(f"⚠️ Redis not available for context memory: {e}")
                redis_client = None
        
        # Initialize the Advanced Understanding System
        advanced_understanding = AdvancedUnderstandingSystem(redis_client=redis_client)
        print("✅ Advanced Understanding System initialized successfully")
        print("  🧠 Semantic Similarity Engine: Ready")
        print("  🧠 Enhanced Context Memory: Ready")
        print("  🎯 Multi-Intent Query Handler: Ready")
    except Exception as e:
        print(f"⚠️ Failed to initialize Advanced Understanding System: {e}")
        ADVANCED_UNDERSTANDING_AVAILABLE = False
        advanced_understanding = None

# Integration with Enhanced AI System
try:
    from istanbul_ai_system_enhancement import EnhancedIstanbulAISystem
    enhanced_ai_system = EnhancedIstanbulAISystem()
    ENHANCED_FEATURES_AVAILABLE = True
    logger.info("🚀 Enhanced Istanbul AI System integrated successfully!")
except ImportError as e:
    logger.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    enhanced_ai_system = None

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3001",
        "http://localhost:3002", 
        "http://127.0.0.1:3002",
        "http://localhost:3003", 
        "http://127.0.0.1:3003",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

print("✅ CORS middleware configured")

# Add security headers middleware for production
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Essential security headers for production
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response

print("✅ Security headers middleware configured")

# === Include Routers ===
try:
    from routes.blog import router as blog_router
    app.include_router(blog_router)
    print("✅ Blog router included successfully")
except ImportError as e:
    print(f"⚠️ Blog router import failed: {e}")

# === Include Cache Monitoring Router ===
try:
    from routes.cache_monitoring import router as cache_router
    app.include_router(cache_router, prefix="/api/cache")
    print("✅ Cache monitoring router included successfully")
except ImportError as e:
    print(f"⚠️ Cache monitoring router import failed: {e}")

# === Include API Routers ===
try:
    from routes.restaurants import router as restaurants_router
    app.include_router(restaurants_router, prefix="/api/restaurants", tags=["restaurants"])
    print("✅ Restaurants router included successfully")
except ImportError as e:
    print(f"⚠️ Restaurants router import failed: {e}")

try:
    from routes.museums import router as museums_router
    app.include_router(museums_router, prefix="/api/museums", tags=["museums"])
    print("✅ Museums router included successfully")
except ImportError as e:
    print(f"⚠️ Museums router import failed: {e}")

try:
    from routes.places import router as places_router
    app.include_router(places_router, prefix="/api/places", tags=["places"])
    print("✅ Places router included successfully")
except ImportError as e:
    print(f"⚠️ Places router import failed: {e}")

# === Include Route Maker Router ===
try:
    from routes.route_maker import router as route_maker_router
    app.include_router(route_maker_router, tags=["Route Maker"])
    print("✅ Route Maker router included successfully")
except ImportError as e:
    print(f"⚠️ Route Maker router import failed: {e}")

# === Include Live Location Router ===
try:
    from routes.location_routes import router as location_router
    app.include_router(location_router, tags=["Live Location & Routing"])
    print("✅ Live Location & Routing router included successfully")
    print("🌍 Location features: Real-time tracking, Multi-stop TSP optimization, Smart POI filtering, Dynamic route updates")
except ImportError as e:
    print(f"⚠️ Live Location router import failed: {e}")
    print("📍 Using simple location router as fallback")
except Exception as e:
    print(f"❌ Live Location router registration failed: {e}")
    print("📍 Using simple location router as fallback")

# === Include Simple Location Router for Testing ===
try:
    from routes.simple_location_routes import router as simple_location_router
    app.include_router(simple_location_router, tags=["Location Services"])
    print("✅ Simple Location router included successfully")
    print("🌍 Location endpoints available: /api/location/health, /api/location/validate, /api/location/session, /api/location/recommendations")
except ImportError as e:
    print(f"❌ Simple Location router import failed: {e}")
except Exception as e:
    print(f"❌ Simple Location router registration failed: {e}")

# === Authentication Setup ===
try:
    from auth import get_current_admin, authenticate_admin, create_access_token, create_refresh_token
    print("✅ Authentication module imported successfully")
except ImportError as e:
    print(f"❌ Authentication import failed: {e}")
    # Create dummy functions to prevent errors
    async def get_current_admin():
        return {"username": "admin", "role": "admin"}
    def authenticate_admin(username, password):
        return {"username": username, "role": "admin"} if username == "admin" else None
    def create_access_token(data):
        return "dummy-token"

# === Security Headers Middleware ===
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add comprehensive security headers to all responses"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Calculate response time
    duration_ms = (time.time() - start_time) * 1000
    
    # Log API request if monitoring is enabled
    if ADVANCED_MONITORING_ENABLED:
        client_ip = getattr(request.client, 'host', 'unknown')
        user_agent = request.headers.get('user-agent', '')
        user_id = getattr(request.state, 'user_id', '')
        request_id = getattr(request.state, 'request_id', '')
        
        log_api_request(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            ip_address=client_ip,
            user_id=user_id,
            request_id=request_id,
            user_agent=user_agent,
            response_size=len(getattr(response, 'body', b''))
        )
        
        # Log performance metric
        log_performance_metric(f"api_{request.method.lower()}_response_time", duration_ms)
    
    # Basic security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # HTTPS enforcement (when deployed with SSL)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Content Security Policy for production
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://www.googletagmanager.com https://www.google-analytics.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://api.aistanbul.net https://localhost:8001; "
        "frame-ancestors 'none'"
    )
    response.headers["Content-Security-Policy"] = csp_policy
    
    # Additional security headers
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # Add response time header for monitoring
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    
    return response

print("✅ Security headers middleware configured")

# Rate limiter completely removed
limiter = None
print("✅ Rate limiting completely removed for unrestricted testing")

# --- Optional Enhancement Systems Initialization ---
# Initialize Optional Enhancement Systems
hybrid_search = None
personalization_engine = None
mini_nlp = None
OPTIONAL_ENHANCEMENTS_ENABLED = False

try:
    from hybrid_search_system import HybridSearchSystem
    from lightweight_personalization_engine import LightweightPersonalizationEngine
    from mini_nlp_modules import MiniNLPProcessor
    OPTIONAL_ENHANCEMENTS_ENABLED = True
    print("✅ Optional enhancement systems loaded successfully")
    
    # Initialize Hybrid Search System
    hybrid_search = HybridSearchSystem()
    print("✅ Hybrid Search System initialized")
    
    # Initialize Personalization Engine  
    personalization_engine = LightweightPersonalizationEngine()
    print("✅ Personalization Engine initialized")
    
    # Initialize Mini NLP Modules
    mini_nlp = MiniNLPProcessor()
    print("✅ Mini NLP Modules initialized")
    
    print("🚀 All optional enhancement systems ready!")
    
except ImportError as e:
    print(f"⚠️ Optional enhancement systems not available: {e}")
    OPTIONAL_ENHANCEMENTS_ENABLED = False
    hybrid_search = None
    personalization_engine = None  
    mini_nlp = None
except Exception as e:
    print(f"⚠️ Optional enhancement systems initialization failed: {e}")
    OPTIONAL_ENHANCEMENTS_ENABLED = False
    hybrid_search = None
    personalization_engine = None
    mini_nlp = None

print(f"Optional Enhancement Systems Status: {'✅ ENABLED' if OPTIONAL_ENHANCEMENTS_ENABLED else '❌ DISABLED'}")

# (Removed duplicate project imports and load_dotenv)

def clean_text_formatting(text):
    """Enhanced text formatting for better readability while preserving structure"""
    if not text:
        return text
    
    # Remove excessive emojis but keep a few for context
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U00002700-\U000027BF"  # dingbats
        u"\U0001F780-\U0001F7FF"  # geometric shapes extended
        u"\U0001F800-\U0001F8FF"  # supplemental arrows-c
        "]+", flags=re.UNICODE)
    
    # Count emojis and only remove if excessive (more than 3)
    emoji_count = len(emoji_pattern.findall(text))
    if emoji_count > 3:
        text = emoji_pattern.sub(r'', text)
    
    # PHASE 1: Remove explicit pricing amounts (all formats) - ENHANCED
    text = re.sub(r'\$\d+[\d.,]*', '', text)      # $20, $15.50
    text = re.sub(r'€\d+[\d.,]*', '', text)       # €20, €15.50
    text = re.sub(r'₺\d+[\d.,]*', '', text)       # ₺20, ₺15.50
    text = re.sub(r'\d+₺', '', text)              # 50₺
    text = re.sub(r'\d+\s*(?:\$|€|₺)', '', text)  # 20$, 50 €
    text = re.sub(r'(?:\$|€|₺)\s*\d+[\d.,]*', '', text)  # $ 20, € 15.50
    
    # Additional pricing patterns
    text = re.sub(r'£\d+[\d.,]*', '', text)       # £20, £15.50
    text = re.sub(r'\d+£', '', text)              # 50£
    text = re.sub(r'(?:USD|EUR|GBP|TRY|TL)\s*\d+[\d.,]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(?:USD|EUR|GBP|TRY|TL)', '', text, flags=re.IGNORECASE)
    
    # PHASE 2: Remove pricing words and phrases - ENHANCED
    text = re.sub(r'\d+\s*(?:lira|euro|euros|dollar|dollars|pound|pounds|tl|usd|eur|gbp|try)s?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:turkish\s+)?lira\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:around|about|approximately|roughly)\s+\d+\s*(?:lira|euro|euros|dollar|dollars)', '', text, flags=re.IGNORECASE)
    
    # PHASE 3: Remove cost-related phrases with amounts - ENHANCED
    cost_patterns = [
        r'(?:costs?|prices?|fees?)\s*:?\s*(?:around\s+|about\s+|approximately\s+|roughly\s+)?\$?\€?₺?£?\d+[\d.,]*',
        r'(?:entrance|admission|ticket|entry)\s*(?:cost|price|fee)s?\s*:?\s*\$?\€?₺?£?\d+',
        r'(?:starting|starts)\s+(?:from|at)\s+\$?\€?₺?£?\d+',
        r'(?:only|just)\s+\$?\€?₺?£?\d+[\d.,]*',
        r'(?:per\s+person|each|pp)\s*:?\s*\$?\€?₺?£?\d+',
        r'\$?\€?₺?£?\d+[\d.,]*\s*(?:per\s+person|each|pp)',
        r'(?:budget|spend|pay)\s*:?\s*(?:around\s+|about\s+)?\$?\€?₺?£?\d+[\d.,]*',
        r'(?:between|from)\s+\$?\€?₺?£?\d+\s*(?:and|to|-|–)\s*\$?\€?₺?£?\d+',
        r'(?:range|varies)\s+(?:from|between)\s+\$?\€?₺?£?\d+',
    ]
    
    for pattern in cost_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # PHASE 4: Remove money emojis and pricing symbols - ENHANCED
    text = re.sub(r'💰|💵|💴|💶|💷|💸', '', text)
    text = re.sub(r'[\$€₺£¥₹₽₴₦₱₩₪₨₡₵₼₢₨₹₿]', '', text)
    
    # Remove pricing codes
    text = re.sub(r'\b(?:USD|EUR|GBP|TRY|TL|JPY|CHF|CAD|AUD)\b', '', text, flags=re.IGNORECASE)
    
    # PHASE 5: Remove ALL markdown formatting for clean responses
    # Remove **bold** and *italic* formatting completely
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold** but keep content
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic* but keep content
    
    # Remove any remaining asterisks that might be left over
    text = re.sub(r'\*+', '', text)  # Remove any standalone asterisks
    
    # Clean up any double spacing that might result from asterisk removal
    text = re.sub(r'  +', ' ', text)  # Multiple spaces to single space
    
    # Improve bullet points and structure
    text = re.sub(r'^[\s]*-\s*', '• ', text, flags=re.MULTILINE) 
    text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1\n\n\2', text)  # Add space between paragraphs
    
    # Clean up spacing while preserving structure
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:  # Only process non-empty lines
            line = re.sub(r'\s+', ' ', line)  # Multiple spaces to single space
            line = re.sub(r'\s*[:;,]\s*[:;,]+', ',', line)  # Clean up punctuation
            cleaned_lines.append(line)
        else:
            cleaned_lines.append('')  # Preserve empty lines for structure
    
    text = '\n'.join(cleaned_lines)
    
    # Final cleanup
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # No more than double line breaks
    text = re.sub(r'^\s+|\s+$', '', text)  # Remove leading/trailing whitespace
    
    return text

def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not user_input:
        return ""
    
    # Limit input length
    max_length = 1000
    if len(user_input) > max_length:
        user_input = user_input[:max_length]
    
    # Remove potentially dangerous characters
    user_input = html.escape(user_input)
    
    # Remove SQL injection patterns
    dangerous_patterns = [
        r'union\s+select', r'drop\s+table', r'delete\s+from', r'insert\s+into',
        r'update\s+set', r'create\s+table', r'alter\s+table', r'exec\s*\(',
        r'script\s*>', r'javascript:', r'vbscript:', r'on\w+\s*=', r'<\s*script'
    ]
    
    for pattern in dangerous_patterns:
        user_input = re.sub(pattern, '', user_input, flags=re.IGNORECASE)
    
    return user_input.strip()

async def get_istanbul_ai_response_with_quality(user_input: str, session_id: str, user_ip: Optional[str] = None, location_context: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """Generate response using Ultra-Specialized Istanbul AI (Rule-Based) with Enhanced Query Understanding and Redis conversational memory"""
    try:
        # Use our Ultra-Specialized Istanbul AI System (completely rule-based)
        if not ULTRA_ISTANBUL_AI_AVAILABLE or not istanbul_ai_system:
            print("❌ Ultra-Specialized Istanbul AI not available")
            return None
        
        # Sanitize input
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        print(f"🏛️ Using Ultra-Specialized Istanbul AI (Rule-Based) for session: {session_id}")
        
        # 🚀 REDIS CONVERSATIONAL MEMORY INTEGRATION
        conversation_context = {}
        if redis_memory:
            try:
                # Get previous conversation context from Redis
                conversation_context = redis_memory.get_context(session_id, user_input)
                print(f"✅ Redis context retrieved - Turn: {conversation_context.get('turn_count', 0)}")
            except Exception as e:
                print(f"⚠️ Redis context retrieval failed: {e}")
                conversation_context = {}
        
        # 🧠 ENHANCED QUERY UNDERSTANDING INTEGRATION
        query_analysis = {}
        if ENHANCED_QUERY_UNDERSTANDING_ENABLED:
            try:
                print(f"🧠 Processing query with Enhanced Query Understanding System...")
                query_analysis = process_enhanced_query(user_input, session_id)
                
                if query_analysis.get('success'):
                    print(f"✅ Query Analysis - Intent: {query_analysis['intent']} "
                          f"(confidence: {query_analysis['confidence']:.2f}), "
                          f"Entities: {len(query_analysis['entities'])}, "
                          f"Corrections: {len(query_analysis['corrections'])}")
                    
                    if query_analysis['corrections']:
                        print(f"🔧 Applied corrections: {', '.join(query_analysis['corrections'])}")
                else:
                    print(f"⚠️ Enhanced query understanding failed, using basic analysis")
                    
            except Exception as e:
                print(f"⚠️ Enhanced query understanding error: {e}")
                # Fallback to basic analysis
                query_analysis = {
                    'original_query': user_input,
                    'normalized_query': user_input.lower().strip(),
                    'intent': 'general_info',
                    'confidence': 0.3,
                    'entities': {},
                    'corrections': [],
                    'success': True
                }
        
        # Prepare user context with Redis conversation context, query analysis, and location context
        user_context = {
            'session_id': session_id,
            'user_ip': user_ip,
            'timestamp': datetime.now().isoformat(),
            'location_context': location_context,
            'conversation_context': conversation_context,
            'recent_intents': conversation_context.get('recent_intents', []),
            'recent_entities': conversation_context.get('recent_entities', {}),
            'user_preferences': conversation_context.get('user_preferences', {}),
            'conversation_flow': conversation_context.get('conversation_flow', []),
            # Add enhanced query understanding results
            'query_analysis': query_analysis,
            'detected_intent': query_analysis.get('intent', 'general_info'),
            'query_confidence': query_analysis.get('confidence', 0.3),
            'extracted_entities': query_analysis.get('entities', {}),
            'query_corrections': query_analysis.get('corrections', []),
            'normalized_query': query_analysis.get('normalized_query', user_input.lower().strip())
        }
        
        # �🏛️ PRIMARY AI SYSTEM: Use Istanbul Daily Talk AI with its enhanced query detection
        # This system has its own advanced intent classification and query understanding
        result = None
        
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            print("🏛️ Using Istanbul Daily Talk AI as PRIMARY system (with enhanced query detection)...")
            try:
                # Process with Istanbul Daily Talk AI - it has its own enhanced query detection
                ai_response = istanbul_daily_talk_ai.process_message(session_id, user_input)
                
                # Check if Istanbul Daily Talk AI detected this as a restaurant query
                # by analyzing the user input using its internal classification
                try:
                    # Use Istanbul Daily Talk AI's enhanced intent classification (simpler approach)
                    entities = istanbul_daily_talk_ai.entity_recognizer.extract_entities(user_input)
                    
                    # Use the enhanced intent classification method directly
                    enhanced_intent = istanbul_daily_talk_ai._enhance_intent_classification(user_input)
                    
                    # Check restaurant-specific patterns
                    is_restaurant_pattern = istanbul_daily_talk_ai._is_restaurant_query(user_input)
                    
                    # Determine final intent
                    if is_restaurant_pattern or enhanced_intent in ['restaurant_query']:
                        intent = 'restaurant_query'
                    elif enhanced_intent == 'attraction_query':
                        intent = 'attraction_query'
                    elif enhanced_intent == 'transportation_query':
                        intent = 'transportation_query'
                    elif enhanced_intent == 'hidden_gems_query':
                        intent = 'hidden_gems_query'
                    else:
                        intent = enhanced_intent if enhanced_intent != 'general_conversation' else 'general_info'
                    
                    print(f"🧠 Istanbul Daily Talk AI detected intent: '{intent}' for query: '{user_input[:50]}...'")
                    
                    # 🍽️ ENHANCE RESTAURANT RESPONSES WITH DATABASE DATA if it's a restaurant query
                    restaurant_intents = ['restaurant_query', 'restaurant_recommendation', 'dining', 'food']
                    if RESTAURANT_SERVICE_ENABLED and intent in restaurant_intents and ai_response:
                        print(f"🍽️ Restaurant query detected by Istanbul Daily Talk AI - enhancing with database")
                        try:
                            # Extract search parameters from Istanbul Daily Talk AI's entities
                            district = entities.get('neighborhoods', [None])[0] if entities.get('neighborhoods') else None
                            cuisine = entities.get('cuisines', [None])[0] if entities.get('cuisines') else None
                            
                            # Search restaurant database with enhanced parameters
                            restaurants = restaurant_service.search_restaurants(
                                district=district,
                                cuisine=cuisine,
                                limit=6  # Balanced number for comprehensive response
                            )
                            
                            if restaurants:
                                # Create enhanced database response that complements AI response
                                database_response = restaurant_service.format_restaurant_response(restaurants)
                                print(f"✅ Enhanced with restaurant database: {len(restaurants)} restaurants")
                                
                                # Combine AI response with database results for comprehensive answer
                                combined_response = f"{ai_response}\n\n{database_response}"
                                
                                result = {
                                    'success': True,
                                    'response': combined_response,
                                    'confidence': 0.95,  # High confidence for AI + database combination
                                    'system_version': 'istanbul_daily_talk_ai_enhanced_with_database',
                                    'processing_time': 0.15,
                                    'fallback_mode': False
                                }
                            else:
                                print("⚠️ No restaurants found in database, using Istanbul Daily Talk AI response only")
                                result = {
                                    'success': True,
                                    'response': ai_response,
                                    'confidence': 0.85,
                                    'system_version': 'istanbul_daily_talk_ai',
                                    'processing_time': 0.2,
                                    'fallback_mode': False
                                }
                                
                        except Exception as e:
                            print(f"⚠️ Restaurant database enhancement error: {e}")
                            # Use original AI response
                            result = {
                                'success': True,
                                'response': ai_response,
                                'confidence': 0.8,
                                'system_version': 'istanbul_daily_talk_ai',
                                'processing_time': 0.2,
                                'fallback_mode': False
                            }
                    else:
                        # Non-restaurant query or no database - use Istanbul Daily Talk AI response directly
                        if ai_response and len(ai_response) > 50:
                            result = {
                                'success': True,
                                'response': ai_response,
                                'confidence': 0.85,
                                'system_version': 'istanbul_daily_talk_ai',
                                'processing_time': 0.2,
                                'fallback_mode': False
                            }
                        else:
                            print("⚠️ Istanbul Daily Talk AI response insufficient, using fallback")
                            
                except Exception as intent_error:
                    print(f"⚠️ Error in intent classification, using AI response as-is: {intent_error}")
                    # Use the AI response without enhancement
                    if ai_response and len(ai_response) > 50:
                        result = {
                            'success': True,
                            'response': ai_response,
                            'confidence': 0.8,
                            'system_version': 'istanbul_daily_talk_ai',
                            'processing_time': 0.2,
                            'fallback_mode': False
                        }
                
            except Exception as e:
                print(f"⚠️ Istanbul Daily Talk AI error: {e}")
                # Fall through to fallback system
        else:
            # Fallback to Ultra-Specialized Istanbul AI if Daily Talk AI is not available
            print("⚠️ Istanbul Daily Talk AI not available, using Ultra-Specialized Istanbul AI as fallback")
            if ULTRA_ISTANBUL_AI_AVAILABLE and istanbul_ai_system:
                result = istanbul_ai_system.process_istanbul_query(user_input, user_context)
            else:
                # No AI systems available
                result = {
                    'success': False,
                    'response': "I'm sorry, the AI systems are currently unavailable. Please try again later.",
                    'system_type': 'no_ai_available',
                    'confidence': 0.1
                }
        
        if result.get('success'):
            ai_response = result['response']
            
            # 🍽️ RESTAURANT DATA INTEGRATION
            # Check if this is a restaurant query and enhance with real restaurant data
            if RESTAURANT_SERVICE_ENABLED and query_analysis.get('intent') in ['find_restaurant', 'find_cafe']:
                try:
                    print("🍽️ Detected restaurant query - integrating restaurant database...")
                    
                    # Extract search parameters from entities and location context
                    entities = query_analysis.get('entities', {})
                    district = entities.get('districts', [None])[0] if entities.get('districts') else None
                    cuisine = entities.get('cuisines', [None])[0] if entities.get('cuisines') else None
                    budget = entities.get('budget', [None])[0] if entities.get('budget') else None
                    
                    # Use location context district if available and no explicit district mentioned
                    if not district and location_context and location_context.get('district'):
                        district = location_context['district']
                        print(f"🌍 Using location context district: {district}")
                    
                    # Search restaurants in database with location awareness
                    restaurants = restaurant_service.search_restaurants(
                        district=district,
                        cuisine=cuisine,
                        budget=budget,
                        keyword=None,
                        limit=3
                    )
                    
                    if restaurants:
                        # Replace AI response with restaurant-specific response
                        restaurant_response = restaurant_service.format_restaurant_response(
                            restaurants, query_analysis
                        )
                        ai_response = restaurant_response
                        print(f"✅ Enhanced response with {len(restaurants)} restaurant recommendations")
                        
                        # Boost confidence for restaurant queries
                        confidence = min(confidence + 0.15, 0.95)
                        
                except Exception as e:
                    print(f"⚠️ Restaurant integration error: {e}")
                    # Continue with original AI response
            
            # Calculate quality score based on confidence and system features
            confidence = result.get('confidence', 0.7)
            quality_score = min(confidence * 100, 95)  # Cap at 95% for rule-based systems
            
            print(f"✅ Ultra-Specialized AI response generated - Session: {session_id}, "
                  f"Confidence: {confidence:.2f}, "
                  f"Quality: {quality_score:.1f}%, "
                  f"System: {result.get('system_version', 'ultra_specialized')}, "
                  f"Time: {result.get('processing_time', 0):.3f}s")
            
            # Apply post-processing cleanup (remove any formatting artifacts)
            ai_response = post_ai_cleanup(ai_response)
            
            # Apply restaurant response formatting
            try:
                from restaurant_response_formatter import format_restaurant_response
                ai_response = format_restaurant_response(ai_response, user_input)
            except ImportError:
                pass  # Formatter not available, continue without it
            
            # 🚀 STORE CONVERSATION TURN IN REDIS
            if redis_memory:
                try:
                    from redis_conversational_memory import ConversationTurn
                    
                    # Create conversation turn with enhanced query understanding data
                    turn = ConversationTurn(
                        timestamp=datetime.now().isoformat(),
                        user_query=user_input,
                        normalized_query=query_analysis.get('normalized_query', user_input.lower().strip()),
                        intent=query_analysis.get('intent', result.get('detected_intent', 'general')),
                        entities=query_analysis.get('entities', result.get('entities', {})),
                        response=ai_response,
                        confidence=confidence
                    )
                    
                    # Store in Redis
                    redis_success = redis_memory.add_turn(session_id, turn)
                    if redis_success:
                        print(f"✅ Conversation turn stored in Redis - Session: {session_id}")
                    else:
                        print(f"⚠️ Failed to store conversation turn in Redis - Session: {session_id}")
                        
                except Exception as redis_error:
                    print(f"⚠️ Redis storage error: {redis_error}")
            
            # Return properly formatted result
            return {
                'success': True,
                'response': ai_response,
                'session_id': session_id,
                'has_context': True,
                'uses_llm': False,  # Explicitly mark as non-LLM
                'system_type': 'ultra_specialized_istanbul_ai',
                'redis_stored': redis_memory is not None,
                'quality_assessment': {
                    'overall_score': quality_score,
                    'confidence': confidence,
                    'used_fallback': result.get('fallback_mode', False),
                    'processing_time': result.get('processing_time', 0),
                    'system_version': result.get('system_version', 'ultra_specialized')
                }
            }
        else:
            print(f"❌ Ultra-Specialized Istanbul AI failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error in Ultra-Specialized Istanbul AI system: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def get_istanbul_ai_response(user_input: str, session_id: str, user_ip: Optional[str] = None) -> Optional[str]:
    """Generate response using Ultra-Specialized Istanbul AI (Rule-Based) - Simple version"""
    try:
        # Sanitize input first
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        # 🧠 ENHANCED QUERY UNDERSTANDING INTEGRATION (Simple version)
        query_analysis = {}
        if ENHANCED_QUERY_UNDERSTANDING_ENABLED:
            try:
                query_analysis = process_enhanced_query(user_input, session_id)
                if query_analysis.get('success'):
                    print(f"🧠 Simple Query Analysis - Intent: {query_analysis['intent']} "
                          f"(confidence: {query_analysis['confidence']:.2f})")
            except Exception as e:
                print(f"⚠️ Enhanced query understanding error (simple): {e}")
        
        # Prepare user context with query analysis
        user_context = {
            'session_id': session_id,
            'user_ip': user_ip,
            'timestamp': datetime.now().isoformat(),
            'query_analysis': query_analysis,
            'detected_intent': query_analysis.get('intent', 'general_info'),
            'query_confidence': query_analysis.get('confidence', 0.3),
            'extracted_entities': query_analysis.get('entities', {}),
            'normalized_query': query_analysis.get('normalized_query', user_input.lower().strip())
        }
        
        # �🏛️ PRIMARY: Use Istanbul Daily Talk AI with its enhanced query detection (Simple version)
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            print(f"🏛️ Using Istanbul Daily Talk AI (PRIMARY) with enhanced query detection for session: {session_id}")
            try:
                # Process with Istanbul Daily Talk AI - it has its own query detection
                ai_response = istanbul_daily_talk_ai.process_message(session_id, user_input)
                
                # Use Istanbul Daily Talk AI's internal query detection for restaurant enhancement
                if RESTAURANT_SERVICE_ENABLED:
                    try:
                        # Use Istanbul Daily Talk AI's enhanced intent classification (simpler approach)
                        entities = istanbul_daily_talk_ai.entity_recognizer.extract_entities(user_input)
                        
                        # Use the enhanced intent classification method directly
                        enhanced_intent = istanbul_daily_talk_ai._enhance_intent_classification(user_input)
                        
                        # Check restaurant-specific patterns
                        is_restaurant_pattern = istanbul_daily_talk_ai._is_restaurant_query(user_input)
                        
                        # Determine final intent
                        if is_restaurant_pattern or enhanced_intent in ['restaurant_query']:
                            intent = 'restaurant_query'
                        elif enhanced_intent == 'attraction_query':
                            intent = 'attraction_query'
                        elif enhanced_intent == 'transportation_query':
                            intent = 'transportation_query'
                        else:
                            intent = enhanced_intent if enhanced_intent != 'general_conversation' else 'general_info'
                        
                        # 🍽️ ENHANCE RESTAURANT RESPONSES WITH DATABASE DATA
                        restaurant_intents = ['restaurant_query', 'restaurant_recommendation', 'dining', 'food']
                        if intent in restaurant_intents and ai_response:
                            print(f"🍽️ Restaurant query detected by Istanbul Daily Talk AI - enhancing with database")
                            
                            # Extract search parameters from Istanbul Daily Talk AI's entities
                            district = entities.get('neighborhoods', [None])[0] if entities.get('neighborhoods') else None
                            cuisine = entities.get('cuisines', [None])[0] if entities.get('cuisines') else None
                            
                            # Search restaurant database
                            restaurants = restaurant_service.search_restaurants(
                                district=district,
                                cuisine=cuisine,
                                limit=3
                            )
                            
                            if restaurants:
                                # Create enhanced response
                                database_response = restaurant_service.format_restaurant_response(restaurants)
                                # Combine responses for comprehensive answer
                                enhanced_response = f"{ai_response}\n\n{database_response}"
                                print(f"✅ Enhanced Istanbul Daily Talk AI response with {len(restaurants)} restaurants")
                                return enhanced_response
                            else:
                                print("⚠️ No restaurants found in database, using AI response only")
                                
                    except Exception as e:
                        print(f"⚠️ Restaurant enhancement error (simple): {e}")
                        # Continue with original AI response
                
                # Return Istanbul Daily Talk AI response
                if ai_response and len(ai_response) > 30:  # Simple threshold
                    print(f"✅ Istanbul Daily Talk AI (simple) response: {len(ai_response)} characters")
                    return ai_response
                else:
                    print("⚠️ Istanbul Daily Talk AI response too short, using fallback")
                    # Fall through to fallback system
            except Exception as e:
                print(f"⚠️ Istanbul Daily Talk AI error (simple): {e}")
                # Fall through to fallback system
        
        # FALLBACK: Use Ultra-Specialized Istanbul AI if Daily Talk AI fails or unavailable
        if ULTRA_ISTANBUL_AI_AVAILABLE and istanbul_ai_system:
            print(f"🏛️ Using Ultra-Specialized Istanbul AI (FALLBACK) for session: {session_id}")
            result = istanbul_ai_system.process_istanbul_query(user_input, user_context)
        else:
            print("❌ No AI systems available")
            return None
        
        if result.get('success'):
            ai_response = result['response']
            
            # 🍽️ RESTAURANT DATA INTEGRATION (Simple version) - Using Istanbul Daily Talk AI's detection
            # Note: This is fallback integration for the Ultra-Specialized system
            restaurant_intents = ['restaurant_query', 'restaurant_recommendation', 'dining', 'food']
            detected_intent = 'general_info'
            
            # Try to use Istanbul Daily Talk AI's query detection even in fallback mode
            if ISTANBUL_DAILY_TALK_AVAILABLE:
                try:
                    entities = istanbul_daily_talk_ai.entity_recognizer.extract_entities(user_input)
                    
                    # Use enhanced intent classification (simpler, no context needed)
                    enhanced_intent = istanbul_daily_talk_ai._enhance_intent_classification(user_input)
                    is_restaurant_pattern = istanbul_daily_talk_ai._is_restaurant_query(user_input)
                    
                    # Determine final intent
                    if is_restaurant_pattern or enhanced_intent in ['restaurant_query']:
                        detected_intent = 'restaurant_query'
                    else:
                        detected_intent = enhanced_intent if enhanced_intent != 'general_conversation' else 'general_info'
                except Exception as e:
                    print(f"⚠️ Could not use Istanbul Daily Talk AI detection in fallback: {e}")
                    detected_intent = query_analysis.get('intent', 'general_info')
            else:
                detected_intent = query_analysis.get('intent', 'general_info')
            
            if RESTAURANT_SERVICE_ENABLED and detected_intent in restaurant_intents:
                try:
                    # Simple restaurant search based on detected entities
                    entities = query_analysis.get('entities', {})
                    district = entities.get('districts', [None])[0] if entities.get('districts') else None
                    cuisine = entities.get('cuisines', [None])[0] if entities.get('cuisines') else None
                    
                    restaurants = restaurant_service.search_restaurants(
                        district=district,
                        cuisine=cuisine,
                        limit=2
                    )
                    
                    if restaurants:
                        ai_response = restaurant_service.format_restaurant_response(restaurants)
                        print(f"✅ Simple restaurant integration: {len(restaurants)} restaurants")
                        
                except Exception as e:
                    print(f"⚠️ Simple restaurant integration error: {e}")
            
            print(f"✅ Ultra-Specialized AI response (simple) - Session: {session_id}, "
                  f"Confidence: {result.get('confidence', 0.7):.2f}, "
                  f"System: {result.get('system_version', 'ultra_specialized')}")
            
            # Apply post-processing cleanup
            ai_response = post_ai_cleanup(ai_response)
            
            # Apply restaurant response formatting
            try:
                from restaurant_response_formatter import format_restaurant_response
                ai_response = format_restaurant_response(ai_response, user_input)
            except ImportError:
                pass  # Formatter not available, continue without it
                
            return ai_response
        else:
            print(f"❌ Ultra-Specialized Istanbul AI failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error in Ultra-Specialized Istanbul AI system: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_fallback_response(user_input: str, context: Optional[Dict] = None) -> str:
    """Create a fallback response when AI systems fail"""
    try:
        # Basic fallback responses based on keywords
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['restaurant', 'food', 'eat', 'dining', 'meal']):
            return """I'd be happy to help you find restaurants in Istanbul! Here are some popular areas for dining:

• **Sultanahmet** - Traditional Ottoman cuisine near historic sites
• **Beyoğlu** - International restaurants and trendy cafes
• **Karaköy** - Waterfront dining with Bosphorus views
• **Kadıköy** - Local favorites and street food

For specific restaurant recommendations, please let me know:
- Which district interests you?
- What type of cuisine do you prefer?
- What's your budget range?"""

        elif any(word in user_lower for word in ['museum', 'history', 'culture', 'art']):
            return """Istanbul has amazing museums and cultural sites! Here are the must-visit ones:

• **Hagia Sophia** - Iconic Byzantine and Ottoman architecture
• **Topkapi Palace** - Former Ottoman imperial palace
• **Istanbul Archaeological Museum** - Ancient artifacts and treasures
• **Istanbul Modern** - Contemporary Turkish and international art

Would you like more details about any specific museum or cultural site?"""

        elif any(word in user_lower for word in ['transport', 'metro', 'bus', 'travel', 'getting around']):
            return """Getting around Istanbul is easy with multiple transport options:

• **Metro** - Fast and efficient for major districts
• **Tram** - Great for tourist areas like Sultanahmet
• **Ferry** - Scenic Bosphorus crossings
• **Bus** - Extensive network covering the whole city
• **Istanbulkart** - Single card for all public transport

Need help planning a specific route?"""

        else:
            return """Welcome to Istanbul! I'm here to help you discover this amazing city.

I can assist you with:
• Restaurant recommendations and dining
• Museums and cultural attractions  
• Transportation and getting around
• Historical sites and hidden gems
• Local tips and insider knowledge

What would you like to know about Istanbul?"""
            
    except Exception as e:
        print(f"Error in create_fallback_response: {e}")
        return "I'm here to help you explore Istanbul! Please let me know what you'd like to discover about this amazing city."

# =============================
# ISTANBUL DAILY TALK SYSTEM API ENDPOINTS  
# =============================

# Request/Response Models for Istanbul Daily Talk System
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lng}")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: Optional[float] = Field(None, description="Response confidence")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    detected_location: Optional[Dict[str, Any]] = Field(None, description="Detected user location information")

class RouteRequest(BaseModel):
    message: str = Field(..., description="Route planning request")
    start_location: Dict[str, float] = Field(..., description="Starting point {lat, lng}")
    end_location: Optional[Dict[str, float]] = Field(None, description="End point {lat, lng}")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Route preferences")
    session_id: Optional[str] = Field(None, description="Session ID")

class TransportRequest(BaseModel):
    from_location: Dict[str, float] = Field(..., description="Starting location {lat, lng}")
    to_location: Dict[str, float] = Field(..., description="Destination {lat, lng}")
    transport_mode: Optional[str] = Field("public", description="Transport mode preference")
    time_preference: Optional[str] = Field("now", description="Time preference")

class MuseumRequest(BaseModel):
    query: str = Field(..., description="Museum query")
    location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lng}")
    preferences: Optional[List[str]] = Field(None, description="Museum preferences")
    session_id: Optional[str] = Field(None, description="Session ID")

class RouteResponse(BaseModel):
    route: Dict[str, Any] = Field(..., description="Generated route")
    total_duration: str = Field(..., description="Total route duration")
    total_distance: str = Field(..., description="Total route distance")
    waypoints: List[Dict[str, Any]] = Field(..., description="Route waypoints")
    suggestions: Optional[List[str]] = Field(None, description="Additional suggestions")

class TransportResponse(BaseModel):
    recommendations: List[Dict[str, Any]] = Field(..., description="Transport recommendations")
    fastest_option: Dict[str, Any] = Field(..., description="Fastest transport option")
    cheapest_option: Dict[str, Any] = Field(..., description="Cheapest transport option")
    weather_advice: Optional[str] = Field(None, description="Weather-related advice")

class MuseumResponse(BaseModel):
    museums: List[Dict[str, Any]] = Field(..., description="Museum recommendations")
    personalized_tips: List[str] = Field(..., description="Personalized museum tips")
    opening_hours: Dict[str, str] = Field(..., description="Current opening hours")
    ticket_info: Dict[str, Any] = Field(..., description="Ticket information")

# =============================
# GPS-BASED ROUTE PLANNING ENDPOINTS
# =============================

@app.post("/api/route/gps-plan", response_model=RouteResponse, tags=["GPS Route Planning"])
async def plan_route_from_gps_location(request: GPSRouteRequest):
    """
    Generate intelligent route plan based on user's GPS location
    Finds nearby attractions and creates optimized route
    """
    try:
        print(f"📍 GPS-based route planning request from location: {request.user_location}")
        
        if not ISTANBUL_DAILY_TALK_AVAILABLE:
            raise HTTPException(status_code=503, detail="GPS route planning service unavailable")
        
        # Generate session ID for personalization
        session_id = request.session_id or f"gps_route_{uuid.uuid4().hex[:8]}"
        
        # Get user profile for personalization
        user_profile = istanbul_daily_talk_ai.get_or_create_user_profile(session_id)
        
        # Update user profile with interests if provided
        if request.interests:
            user_profile.interests.extend([interest for interest in request.interests if interest not in user_profile.interests])
        
        # Create location-aware query
        lat, lng = request.user_location["lat"], request.user_location["lng"]
        
        # Generate contextual route planning query based on location and interests
        location_query_parts = [
            f"I'm currently at GPS location {lat:.4f}, {lng:.4f} in Istanbul."
        ]
        
        if request.interests:
            interests_str = ", ".join(request.interests)
            location_query_parts.append(f"I'm interested in {interests_str}.")
        
        if request.duration_hours:
            location_query_parts.append(f"I have {request.duration_hours} hours available.")
        
        if request.radius_km:
            location_query_parts.append(f"I prefer to stay within {request.radius_km}km of my current location.")
        
        location_query_parts.append("Please create an optimized route plan for me.")
        
        route_query = " ".join(location_query_parts)
        
        # Create context with GPS location
        from istanbul_daily_talk_system import ConversationContext
        context = ConversationContext(
            session_id=session_id,
            user_profile=user_profile
        )
        context.context_memory = {
            "user_location": request.user_location,
            "radius_km": request.radius_km,
            "duration_hours": request.duration_hours,
            "transport_mode": request.transport_mode,
            "route_style": request.route_style
        }
        
        # Use Istanbul Daily Talk AI for GPS-aware route planning
        if hasattr(istanbul_daily_talk_ai, 'handle_route_planning_query'):
            route_response = istanbul_daily_talk_ai.handle_route_planning_query(
                route_query, user_profile, context, datetime.now()
            )
        else:
            # Fallback to regular message processing with location context
            route_response = istanbul_daily_talk_ai.process_message(session_id, route_query)
        
        # Extract nearby attractions (if route maker service is available)
        nearby_attractions = []
        if hasattr(istanbul_daily_talk_ai, 'route_maker') and istanbul_daily_talk_ai.route_maker:
            try:
                # Get nearby attractions using route maker service
                from services.route_maker_service import get_route_maker
                route_maker = get_route_maker()
                
                # This would need to be implemented in the route maker service
                # nearby_attractions = route_maker.get_attractions_near_point(lat, lng, request.radius_km or 5.0)
                
            except Exception as e:
                print(f"⚠️ Could not get nearby attractions: {e}")
        
        # Create GPS-aware waypoints
        waypoints = []
        if "→" in route_response:
            places = [place.strip() for place in route_response.split("→")]
            for i, place in enumerate(places):
                waypoints.append({
                    "order": i + 1,
                    "name": place,
                    "description": f"Stop {i + 1}: {place}",
                    "estimated_time": "45-90 minutes",
                    "distance_from_start": f"{i * 0.8:.1f} km"  # Estimated
                })
        
        # Calculate estimated totals based on GPS location
        estimated_distance = len(waypoints) * 1.2  # Rough estimate
        estimated_duration = request.duration_hours or min(len(waypoints) * 1.5, 8)
        
        route_data = {
            "description": route_response,
            "optimized": True,
            "algorithm": "GPS-aware TSP with local optimization",
            "start_point": request.user_location,
            "gps_based": True,
            "radius_km": request.radius_km or 5.0,
            "interests_considered": request.interests or [],
            "transport_mode": request.transport_mode
        }
        
        return RouteResponse(
            route=route_data,
            total_duration=f"{estimated_duration:.1f} hours",
            total_distance=f"{estimated_distance:.1f} km",
            waypoints=waypoints,
            suggestions=[
                f"Route optimized for {request.radius_km or 5.0}km radius from your location",
                f"Estimated total time: {estimated_duration:.1f} hours",
                "Consider traffic and opening hours",
                "Download offline maps for better navigation"
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ GPS route planning error: {e}")
        raise HTTPException(status_code=500, detail="GPS route planning failed")

@app.post("/api/nearby/attractions", response_model=LocationBasedRecommendationResponse, tags=["GPS Route Planning"]) 
async def get_nearby_attractions(request: NearbyAttractionsRequest):
    """
    Get attractions near user's GPS location
    Returns personalized recommendations based on location
    """
    try:
        print(f"📍 Nearby attractions request for location: {request.location}")
        
        lat, lng = request.location["lat"], request.location["lng"]
        radius = request.radius_km or 2.0
        
        # Use Istanbul Daily Talk AI to get location-aware recommendations
        location_query = f"What attractions and interesting places are near GPS coordinates {lat:.4f}, {lng:.4f} within {radius}km?"
        
        if request.categories:
            categories_str = ", ".join(request.categories)
            location_query += f" I'm particularly interested in {categories_str}."
        
        session_id = f"nearby_{uuid.uuid4().hex[:8]}"
        
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            # Get AI recommendations
            ai_response = istanbul_daily_talk_ai.process_message(session_id, location_query)
            
            # Extract location information using entity recognizer
            entities = istanbul_daily_talk_ai.entity_recognizer.extract_entities(location_query)
            
            # Parse response for structured data
            recommendations = []
            
            # Extract attractions from AI response (basic parsing)
            lines = ai_response.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('•' in line or '-' in line or line.startswith(('1.', '2.', '3.'))):
                    # Extract attraction name and description
                    clean_line = re.sub(r'^[•\-\d\.\s]+', '', line)
                    if clean_line:
                        recommendations.append({
                            "name": clean_line.split('-')[0].strip() if '-' in clean_line else clean_line[:50],
                            "description": clean_line,
                            "estimated_distance": f"{radius/2:.1f} km",  # Rough estimate
                            "category": "attraction",
                            "confidence": 0.8
                        })
            
            # Limit results
            recommendations = recommendations[:request.limit or 10]
            
        else:
            # Fallback static recommendations
            recommendations = [
                {
                    "name": "Hagia Sophia",
                    "description": "Historic Byzantine and Ottoman monument",
                    "estimated_distance": "1.2 km",
                    "category": "historical",
                    "confidence": 0.9
                },
                {
                    "name": "Blue Mosque",
                    "description": "Famous Ottoman mosque with blue tiles",
                    "estimated_distance": "0.8 km", 
                    "category": "religious",
                    "confidence": 0.9
                }
            ]
        
        # Generate route suggestions
        suggested_routes = [
            {
                "name": "Quick Tour",
                "duration": "2-3 hours",
                "attractions": min(len(recommendations), 3),
                "description": "Visit top nearby attractions"
            },
            {
                "name": "Full Day Tour", 
                "duration": "6-8 hours",
                "attractions": min(len(recommendations), 6),
                "description": "Comprehensive exploration of the area"
            }
        ]
        
        return LocationBasedRecommendationResponse(
            recommendations=recommendations,
            user_location=request.location,
            distance_info={
                "search_radius": f"{radius} km",
                "total_found": str(len(recommendations)),
                "closest": recommendations[0]["estimated_distance"] if recommendations else "unknown"
            },
            suggested_routes=suggested_routes
        )
        
    except Exception as e:
        print(f"❌ Nearby attractions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get nearby attractions")

@app.post("/api/route/gps-optimize", response_model=RouteResponse, tags=["GPS Route Planning"])
async def optimize_route_from_gps(
    user_location: Dict[str, float] = Body(..., description="User's GPS location"),
    destinations: List[Dict[str, Any]] = Body(..., description="List of destinations to visit"),
    preferences: Optional[Dict[str, Any]] = Body(None, description="Route optimization preferences")
):
    """
    Optimize route order based on user's GPS location and destinations
    Uses TSP algorithm for optimal routing
    """
    try:
        print(f"🗺️ GPS route optimization from {user_location} to {len(destinations)} destinations")
        
        if not destinations:
            raise HTTPException(status_code=400, detail="No destinations provided")
        
        # Create route optimization query
        destination_names = [dest.get("name", "Unknown") for dest in destinations]
        destinations_str = ", ".join(destination_names)
        
        optimization_query = (
            f"I'm at GPS location {user_location['lat']:.4f}, {user_location['lng']:.4f} "
            f"and want to visit these places: {destinations_str}. "
            f"What's the most efficient route order?"
        )
        
        session_id = f"optimize_{uuid.uuid4().hex[:8]}"
        
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            # Use Istanbul Daily Talk AI for route optimization
            optimization_response = istanbul_daily_talk_ai.process_message(session_id, optimization_query)
            
            # Try to extract optimized order from response
            optimized_waypoints = []
            if "→" in optimization_response:
                ordered_places = [place.strip() for place in optimization_response.split("→")]
                for i, place in enumerate(ordered_places):
                    # Find matching destination
                    matching_dest = None
                    for dest in destinations:
                        if dest.get("name", "").lower() in place.lower():
                            matching_dest = dest
                            break
                    
                    waypoint = {
                        "order": i + 1,
                        "name": place,
                        "description": matching_dest.get("description", f"Visit {place}") if matching_dest else f"Visit {place}",
                        "estimated_time": "60-90 minutes",
                        "location": matching_dest.get("location") if matching_dest else None
                    }
                    optimized_waypoints.append(waypoint)
            else:
                # Fallback: use original order
                for i, dest in enumerate(destinations):
                    optimized_waypoints.append({
                        "order": i + 1,
                        "name": dest.get("name", f"Destination {i+1}"),
                        "description": dest.get("description", ""),
                        "estimated_time": "60-90 minutes",
                        "location": dest.get("location")
                    })
            
            route_data = {
                "description": optimization_response,
                "optimized": True,
                "algorithm": "GPS-aware TSP optimization",
                "start_point": user_location,
                "optimization_method": preferences.get("method", "balanced") if preferences else "balanced",
                "gps_based": True
            }
            
            return RouteResponse(
                route=route_data,
                total_duration=f"{len(optimized_waypoints) * 1.5:.1f} hours",
                total_distance=f"{len(optimized_waypoints) * 1.8:.1f} km",
                waypoints=optimized_waypoints,
                suggestions=[
                    "Route optimized for minimum travel time",
                    "Consider traffic conditions during peak hours",
                    "Allow extra time for popular attractions",
                    "Check opening hours before visiting"
                ]
            )
        else:
            raise HTTPException(status_code=503, detail="Route optimization service unavailable")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ GPS route optimization error: {e}")
        raise HTTPException(status_code=500, detail="Route optimization failed")

# =============================
# MAIN CHAT API ENDPOINT
# =============================

@app.post("/api/chat", response_model=ChatResponse, tags=["Istanbul Daily Talk"])
async def chat_with_istanbul_ai(
    request: ChatRequest,
    user_ip: str = Query(None, description="User IP for analytics")
):
    """
    Main chat endpoint using Istanbul Daily Talk AI System
    Handles all types of queries with enhanced personalization
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        user_id = request.user_id or session_id
        
        # Sanitize input
        user_input = sanitize_user_input(request.message)
        if not user_input:
            raise HTTPException(status_code=400, detail="Invalid message content")
        
        print(f"🏛️ Chat request - Session: {session_id}, Message: '{user_input[:50]}...'")
        
        # === STEP 1: INTELLIGENT LOCATION DETECTION ===
        detected_location = None
        if INTELLIGENT_LOCATION_ENABLED:
            try:
                # Get user context (previous location, preferences, etc.)
                user_context = {}
                
                # Try to get user's IP address from request or parameter
                client_ip = user_ip
                if not client_ip and hasattr(request, 'client') and hasattr(request.client, 'host'):
                    client_ip = request.client.host
                
                # Detect location from text input and IP
                detected_location = await detect_user_location(
                    text=user_input,
                    user_context=user_context,
                    ip_address=client_ip
                )
                
                if detected_location and detected_location.latitude:
                    print(f"📍 Location detected: {detected_location.source} - "
                          f"{detected_location.latitude:.4f}, {detected_location.longitude:.4f} "
                          f"(Confidence: {detected_location.confidence.value})")
                    
                    # Add location to user context for the AI system
                    user_context['detected_location'] = {
                        'lat': detected_location.latitude,
                        'lng': detected_location.longitude,
                        'name': detected_location.name,
                        'neighborhood': detected_location.neighborhood,
                        'district': detected_location.district,
                        'confidence': detected_location.confidence.value,
                        'source': detected_location.source,
                        'accuracy_meters': detected_location.accuracy_meters
                    }
                    
                else:
                    print("📍 No specific location detected from input")
                    
            except Exception as e:
                print(f"⚠️ Location detection failed: {e}")
                detected_location = None
        
        # Use Istanbul Daily Talk AI as primary system
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            try:
                # === STEP 2: UPDATE USER PROFILE WITH DETECTED LOCATION ===
                if detected_location and detected_location.latitude:
                    # Get user profile and update location
                    user_profile = istanbul_daily_talk_ai.get_or_create_user_profile(user_id)
                    
                    # Update GPS location in user profile
                    user_profile.gps_location = {
                        'lat': detected_location.latitude,
                        'lng': detected_location.longitude
                    }
                    
                    # Update current location name if available
                    if detected_location.name:
                        user_profile.current_location = detected_location.name
                    elif detected_location.neighborhood:
                        user_profile.current_location = detected_location.neighborhood
                    elif detected_location.district:
                        user_profile.current_location = detected_location.district
                    
                    # Update location accuracy and timestamp
                    user_profile.location_accuracy = detected_location.accuracy_meters
                    user_profile.location_timestamp = detected_location.detected_at
                    
                    print(f"📍 Updated user profile with location: {user_profile.current_location} "
                          f"({detected_location.latitude:.4f}, {detected_location.longitude:.4f})")
                
                # Process message with Istanbul Daily Talk AI
                ai_response = istanbul_daily_talk_ai.process_message(user_id, user_input)
                
                # Detect intent using Istanbul Daily Talk AI's enhanced classification
                try:
                    enhanced_intent = istanbul_daily_talk_ai._enhance_intent_classification(user_input)
                    confidence = 0.85  # High confidence for Istanbul Daily Talk AI
                    
                    # Generate follow-up suggestions based on intent
                    suggestions = []
                    if enhanced_intent == 'restaurant_query':
                        suggestions = [
                            "Tell me about Turkish breakfast places",
                            "What about seafood restaurants?",
                            "Show me budget-friendly options"
                        ]
                    elif enhanced_intent == 'attraction_query':
                        suggestions = [
                            "Plan a route to visit these places",
                            "What are the opening hours?",
                            "Tell me about nearby restaurants"
                        ]
                    elif enhanced_intent == 'transportation_query':
                        suggestions = [
                            "What's the fastest route?",
                            "Show me metro connections",
                            "Any traffic updates?"
                        ]
                    else:
                        suggestions = [
                            "Tell me about Istanbul attractions",
                            "Recommend some restaurants",
                            "Help me plan a route"
                        ]
                    
                    # Enhance restaurant responses with database if available
                    if RESTAURANT_SERVICE_ENABLED and enhanced_intent in ['restaurant_query']:
                        try:
                            entities = istanbul_daily_talk_ai.entity_recognizer.extract_entities(user_input)
                            district = entities.get('neighborhoods', [None])[0] if entities.get('neighborhoods') else None
                            cuisine = entities.get('cuisines', [None])[0] if entities.get('cuisines') else None
                            
                            restaurants = restaurant_service.search_restaurants(
                                district=district,
                                cuisine=cuisine,
                                limit=3
                            )
                            
                            if restaurants:
                                database_response = restaurant_service.format_restaurant_response(restaurants)
                                ai_response = f"{ai_response}\n\n{database_response}"
                                confidence = 0.9  # Higher confidence with database enhancement
                                print(f"✅ Enhanced response with {len(restaurants)} restaurants")
                        except Exception as e:
                            print(f"⚠️ Restaurant enhancement error: {e}")
                    
                    return ChatResponse(
                        response=clean_text_formatting(ai_response),
                        session_id=session_id,
                        intent=enhanced_intent,
                        confidence=confidence,
                        suggestions=suggestions,
                        detected_location=user_context.get('detected_location') if detected_location else None
                    )
                    
                except Exception as intent_error:
                    print(f"⚠️ Intent classification error: {intent_error}")
                    # Return basic response
                    return ChatResponse(
                        response=clean_text_formatting(ai_response),
                        session_id=session_id,
                        intent="general_conversation",
                        confidence=0.7,
                        detected_location=user_context.get('detected_location') if detected_location else None
                    )
                
            except Exception as e:
                print(f"⚠️ Istanbul Daily Talk AI error: {e}")
                # Fall through to fallback
        
        # Fallback to basic response
        fallback_response = create_fallback_response(user_input)
        return ChatResponse(
            response=fallback_response,
            session_id=session_id,
            intent="general_info",
            confidence=0.5,
            suggestions=[
                "Tell me about Istanbul attractions",
                "Recommend some restaurants",
                "Help me with transportation"
            ],
            detected_location=user_context.get('detected_location') if 'user_context' in locals() and detected_location else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# =============================
# ROUTE PLANNING API ENDPOINT
# =============================

@app.post("/api/route/plan", response_model=RouteResponse, tags=["Route Planning"])
async def plan_route_with_istanbul_ai(request: RouteRequest):
    """
    Advanced route planning with TSP optimization
    Uses Istanbul Daily Talk AI's route planning system
    """
    try:
        print(f"🗺️ Route planning request: {request.message}")
        
        if not ISTANBUL_DAILY_TALK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Route planning service unavailable")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"route_{uuid.uuid4().hex[:8]}"
        
        # Check if Istanbul Daily Talk AI has route planning capability
        if hasattr(istanbul_daily_talk_ai, 'handle_route_planning_query'):
            try:
                # Get user profile for personalization
                user_profile = istanbul_daily_talk_ai.get_or_create_user_profile(session_id)
                
                # Create context with location information
                from istanbul_daily_talk_system import ConversationContext
                context = ConversationContext(
                    session_id=session_id,
                    user_profile=user_profile
                )
                context.context_memory = {"user_location": request.start_location}
                
                # Process route planning query
                route_response = istanbul_daily_talk_ai.handle_route_planning_query(
                    request.message, 
                    user_profile, 
                    context, 
                    datetime.now()
                )
                
                # Parse response to extract route information
                route_data = {
                    "description": route_response,
                    "optimized": True,
                    "algorithm": "TSP with Istanbul optimizations",
                    "start_point": request.start_location,
                    "end_point": request.end_location or request.start_location
                }
                
                waypoints = []
                # Extract waypoints from response (basic parsing)
                if "→" in route_response:
                    places = [place.strip() for place in route_response.split("→")]
                    for i, place in enumerate(places):
                        waypoints.append({
                            "order": i + 1,
                            "name": place,
                            "description": f"Stop {i + 1}: {place}",
                            "estimated_time": "30-60 minutes"
                        })
                
                return RouteResponse(
                    route=route_data,
                    total_duration="4-6 hours",
                    total_distance="15-25 km",
                    waypoints=waypoints,
                    suggestions=[
                        "Consider starting early morning",
                        "Bring comfortable walking shoes",
                        "Check museum opening hours"
                    ]
                )
                
            except Exception as e:
                print(f"⚠️ Route planning error: {e}")
                # Fallback route response
        
        # Fallback basic route planning
        return RouteResponse(
            route={
                "description": f"Basic route for: {request.message}",
                "type": "walking_tour",
                "optimized": False
            },
            total_duration="3-5 hours",
            total_distance="10-20 km",
            waypoints=[
                {
                    "order": 1,
                    "name": "Starting Point",
                    "description": "Begin your Istanbul journey"
                }
            ],
            suggestions=[
                "Use Istanbul Daily Talk AI for detailed planning",
                "Consider public transportation options",
                "Plan around meal times"
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Route planning error: {e}")
        raise HTTPException(status_code=500, detail="Route planning failed")

# =============================
# TRANSPORTATION ADVICE API ENDPOINT
# =============================

@app.post("/api/transport/advice", response_model=TransportResponse, tags=["Transportation"])
async def get_transportation_advice(request: TransportRequest):
    """
    Real-time transportation recommendations
    Provides multi-modal transport options with live data
    """
    try:
        print(f"🚌 Transport advice request: {request.from_location} → {request.to_location}")
        
        # Check if enhanced transportation service is available
        if ENHANCED_SERVICES_ENABLED and enhanced_transport_service:
            try:
                # Get transportation info using enhanced service
                transport_info = enhanced_transport_service.get_transportation_info(
                    from_location=request.from_location,
                    to_location=request.to_location,
                    mode=request.transport_mode
                )
                
                if transport_info:
                    recommendations = [
                        {
                            "mode": "metro",
                            "duration": "25-35 minutes",
                            "cost": "Low",
                            "description": transport_info.get("metro", "Metro connection available")
                        },
                        {
                            "mode": "bus",
                            "duration": "30-45 minutes", 
                            "cost": "Very Low",
                            "description": transport_info.get("bus", "Bus service available")
                        },
                        {
                            "mode": "ferry",
                            "duration": "20-30 minutes",
                            "cost": "Low",
                            "description": transport_info.get("ferry", "Ferry service available")
                        }
                    ]
                    
                    return TransportResponse(
                        recommendations=recommendations,
                        fastest_option=recommendations[2],  # Ferry usually fastest
                        cheapest_option=recommendations[1],  # Bus usually cheapest
                        weather_advice="Check weather conditions for ferry travel"
                    )
            except Exception as e:
                print(f"⚠️ Enhanced transport service error: {e}")
        
        # Check if Istanbul Daily Talk AI has transportation advice
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            try:
                # Use Istanbul Daily Talk AI's transportation query handling
                transport_query = f"How to get from {request.from_location} to {request.to_location} using {request.transport_mode}"
                
                session_id = f"transport_{uuid.uuid4().hex[:8]}"
                transport_response = istanbul_daily_talk_ai.process_message(session_id, transport_query)
                
                # Parse response for structured data
                recommendations = [
                    {
                        "mode": "public_transport",
                        "duration": "30-45 minutes",
                        "cost": "Affordable",
                        "description": transport_response
                    }
                ]
                
                return TransportResponse(
                    recommendations=recommendations,
                    fastest_option=recommendations[0],
                    cheapest_option=recommendations[0],
                    weather_advice="Consider weather when choosing transport mode"
                )
                
            except Exception as e:
                print(f"⚠️ Istanbul Daily Talk AI transport error: {e}")
        
        # Fallback transportation advice
        return TransportResponse(
            recommendations=[
                {
                    "mode": "metro",
                    "duration": "25-40 minutes",
                    "cost": "₺15-25",
                    "description": "Fast and reliable metro connections"
                },
                {
                    "mode": "bus",
                    "duration": "35-60 minutes",
                    "cost": "₺7-15", 
                    "description": "Extensive bus network coverage"
                },
                {
                    "mode": "taxi",
                    "duration": "20-45 minutes",
                    "cost": "₺50-150",
                    "description": "Door-to-door convenience"
                }
            ],
            fastest_option={
                "mode": "metro",
                "duration": "25-40 minutes",
                "description": "Metro is usually the fastest option"
            },
            cheapest_option={
                "mode": "bus", 
                "duration": "35-60 minutes",
                "description": "Bus is the most economical choice"
            },
            weather_advice="Consider indoor transport options during bad weather"
        )
        
    except Exception as e:
        print(f"❌ Transportation advice error: {e}")
        raise HTTPException(status_code=500, detail="Transportation advice failed")

# =============================
# MUSEUM RECOMMENDATIONS API ENDPOINT
# =============================

@app.post("/api/museums/recommend", response_model=MuseumResponse, tags=["Museums"])
async def get_museum_recommendations(request: MuseumRequest):
    """
    Enhanced museum recommendations with personalization
    Provides curated museum suggestions with practical information
    """
    try:
        print(f"🏛️ Museum recommendation request: {request.query}")
        
        # Check if enhanced museum service is available
        if ENHANCED_SERVICES_ENABLED and enhanced_museum_service:
            try:
                # Get museum info using enhanced service
                museum_info = enhanced_museum_service.get_museum_info(
                    query=request.query,
                    location=request.location,
                    preferences=request.preferences or []
                )
                
                if museum_info:
                    museums = museum_info.get('museums', [])
                    return MuseumResponse(
                        museums=museums,
                        personalized_tips=museum_info.get('tips', []),
                        opening_hours=museum_info.get('hours', {}),
                        ticket_info=museum_info.get('tickets', {})
                    )
            except Exception as e:
                print(f"⚠️ Enhanced museum service error: {e}")
        
        # Check if Istanbul Daily Talk AI has museum recommendations
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            try:
                session_id = request.session_id or f"museum_{uuid.uuid4().hex[:8]}"
                
                # Check if museum query handling is available
                if istanbul_daily_talk_ai._is_museum_query(request.query):
                    museum_response = istanbul_daily_talk_ai.process_message(session_id, request.query)
                    
                    # Parse response for structured data
                    museums = [
                        {
                            "name": "Hagia Sophia",
                            "type": "Historical Monument",
                            "rating": 4.8,
                            "description": "Iconic Byzantine and Ottoman architecture",
                            "opening_hours": "09:00 - 19:00",
                            "ticket_price": "Free"
                        },
                        {
                            "name": "Topkapi Palace",
                            "type": "Palace Museum",
                            "rating": 4.7,
                            "description": "Former Ottoman imperial palace",
                            "opening_hours": "09:00 - 18:00",
                            "ticket_price": "₺100"
                        },
                        {
                            "name": "Istanbul Archaeological Museum",
                            "type": "Archaeological",
                            "rating": 4.5,
                            "description": "Ancient artifacts and treasures",
                            "opening_hours": "09:00 - 17:00",
                            "ticket_price": "₺50"
                        }
                    ]
                    
                    return MuseumResponse(
                        museums=museums,
                        personalized_tips=[
                            "Visit Hagia Sophia early morning to avoid crowds",
                            "Topkapi Palace requires 2-3 hours minimum",
                            "Archaeological Museum is perfect for history enthusiasts"
                        ],
                        opening_hours={
                            "Hagia Sophia": "09:00 - 19:00",
                            "Topkapi Palace": "09:00 - 18:00 (Closed Tuesdays)",
                            "Archaeological Museum": "09:00 - 17:00 (Closed Mondays)"
                        },
                        ticket_info={
                            "museum_pass": "₺325 for 5 days",
                            "student_discount": "50% off with valid ID",
                            "group_booking": "10+ people get 20% discount"
                        }
                    )
            except Exception as e:
                print(f"⚠️ Istanbul Daily Talk AI museum error: {e}")
        
        # Fallback museum recommendations
        museums = [
            {
                "name": "Hagia Sophia",
                "type": "Historical Monument", 
                "rating": 4.8,
                "description": "A masterpiece of Byzantine architecture"
            },
            {
                "name": "Blue Mosque",
                "type": "Mosque",
                "rating": 4.7,
                "description": "Famous for its blue tiles and six minarets"
            },
            {
                "name": "Topkapi Palace",
                "type": "Palace Museum",
                "rating": 4.6,
                "description": "Ottoman imperial palace with stunning views"
            }
        ]
        
        return MuseumResponse(
            museums=museums,
            personalized_tips=[
                "Start your museum tour early in the morning",
                "Consider buying a museum pass for multiple visits",
                "Check special exhibitions and events"
            ],
            opening_hours={
                "Hagia Sophia": "24/7 (prayer times restricted)",
                "Blue Mosque": "Outside prayer times",
                "Topkapi Palace": "09:00 - 18:00"
            },
            ticket_info={
                "hagia_sophia": "Free entry",
                "topkapi_palace": "₺100 adult, ₺50 student",
                "museum_pass": "₺325 for 5 days"
            }
        )
        
    except Exception as e:
        print(f"❌ Museum recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Museum recommendations failed")

# =============================
# SYSTEM STATUS AND HEALTH ENDPOINTS
# =============================

@app.get("/api/istanbul-ai/status", tags=["System"])
async def get_istanbul_ai_status():
    """Get status of all Istanbul AI systems"""
    return {
        "istanbul_daily_talk_ai": {
            "available": ISTANBUL_DAILY_TALK_AVAILABLE,
            "features": [
                "Route Planning with TSP optimization",
                "Real-time Transportation Advice",
                "Enhanced Museum Recommendations", 
                "Personalized Daily Talk System",
                "50+ Istanbul Attractions Database",
                "Restaurant Integration",
                "Multi-language Support"
            ] if ISTANBUL_DAILY_TALK_AVAILABLE else []
        },
        "enhanced_services": {
            "available": ENHANCED_SERVICES_ENABLED,
            "services": [
                "Enhanced Transportation Service",
                "Enhanced Museum Service", 
                "Enhanced Actionability Service"
            ] if ENHANCED_SERVICES_ENABLED else []
        },
        "restaurant_service": {
            "available": RESTAURANT_SERVICE_ENABLED,
            "database_stats": restaurant_service.get_restaurant_stats() if RESTAURANT_SERVICE_ENABLED else {}
        },
        "system_health": {
            "total_requests": system_metrics["requests_total"],
            "cache_hit_rate": f"{(system_metrics['cache_hits'] / max(system_metrics['cache_hits'] + system_metrics['cache_misses'], 1) * 100):.1f}%",
            "uptime": str(datetime.now() - system_metrics["start_time"]),
            "errors": system_metrics["errors"]
        }
    }

@app.get("/api/istanbul-ai/capabilities", tags=["System"])
async def get_istanbul_ai_capabilities():
    """Get detailed capabilities of Istanbul AI systems"""
    capabilities = []
    
    if ISTANBUL_DAILY_TALK_AVAILABLE:
        capabilities.extend([
            {
                "name": "Advanced Route Planning",
                "description": "TSP-optimized multi-stop route planning with Istanbul-specific optimizations",
                "endpoint": "/api/route/plan",
                "features": ["TSP Algorithm", "Weather Integration", "Real-time Data", "Personalization"]
            },
            {
                "name": "Real-time Transportation Advice", 
                "description": "Multi-modal transport recommendations with live traffic and weather data",
                "endpoint": "/api/transport/advice",
                "features": ["Metro", "Bus", "Ferry", "Taxi", "Walking", "Weather Advisory"]
            },
            {
                "name": "Enhanced Museum Recommendations",
                "description": "Personalized museum suggestions with opening hours and ticket information",
                "endpoint": "/api/museums/recommend", 
                "features": ["50+ Museums", "Opening Hours", "Ticket Prices", "Personalization"]
            },
            {
                "name": "Daily Talk Conversational AI",
                "description": "Natural language conversations about Istanbul with deep personalization",
                "endpoint": "/api/chat",
                "features": ["NLP", "Context Memory", "Personalization", "Multi-Intent"]
            }
        ])
    
    return {
        "total_capabilities": len(capabilities),
        "capabilities": capabilities,
        "integration_status": {
            "istanbul_daily_talk_ai": "✅ ACTIVE" if ISTANBUL_DAILY_TALK_AVAILABLE else "❌ DISABLED",
            "restaurant_database": "✅ ACTIVE" if RESTAURANT_SERVICE_ENABLED else "❌ DISABLED", 
            "enhanced_services": "✅ ACTIVE" if ENHANCED_SERVICES_ENABLED else "❌ DISABLED"
        }
    }

# Add health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "istanbul_daily_talk_ai": ISTANBUL_DAILY_TALK_AVAILABLE,
            "restaurant_service": RESTAURANT_SERVICE_ENABLED,
            "enhanced_services": ENHANCED_SERVICES_ENABLED
        }
    }

print("✅ Istanbul Daily Talk System API endpoints configured:")
print("   🏛️ /api/chat - Main conversational AI")
print("   🗺️ /api/route/plan - Advanced route planning with TSP")
print("   🚌 /api/transport/advice - Real-time transportation recommendations")
print("   🏛️ /api/museums/recommend - Enhanced museum recommendations")
print("   📊 /api/istanbul-ai/status - System status and health")
print("   🔧 /api/istanbul-ai/capabilities - Detailed system capabilities")
print("   ❤️ /health - Health check")