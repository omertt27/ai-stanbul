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
# defaultdict import removed - no longer needed after removing daily usage tracking

# --- Third-Party Imports ---
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status, Body, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from thefuzz import fuzz, process

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
    
    # Create dummy services to prevent errors
    class DummyEnhancedService:
        def get_transportation_info(self, *args, **kwargs): return {}
        def get_route_info(self, *args, **kwargs): return {}
        def get_museum_info(self, *args, **kwargs): return {}
        def search_museums(self, *args, **kwargs): return []
        def enhance_response_actionability(self, *args, **kwargs): return {"success": False}
        def format_structured_response(self, *args, **kwargs): return ""
        def add_cultural_context(self, *args, **kwargs): return ""
        def translate_key_phrases(self, *args, **kwargs): return ""
    
    enhanced_transport_service = DummyEnhancedService()
    enhanced_museum_service = DummyEnhancedService()
    enhanced_actionability_service = DummyEnhancedService()

# Import live data services for museums and transport
try:
    from real_museum_service import real_museum_service
    print("✅ Real museum service import successful")
    REAL_MUSEUM_SERVICE_ENABLED = True
except ImportError as e:
    print(f"⚠️ Real museum service import failed: {e}")
    real_museum_service = None
    REAL_MUSEUM_SERVICE_ENABLED = False

try:
    from real_transportation_service import real_transportation_service
    print("✅ Real transportation service import successful")
    REAL_TRANSPORT_SERVICE_ENABLED = True
except ImportError as e:
    print(f"⚠️ Real transportation service import failed: {e}")
    real_transportation_service = None
    REAL_TRANSPORT_SERVICE_ENABLED = False

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

# Use only our specialized rule-based system
CUSTOM_AI_AVAILABLE = ULTRA_ISTANBUL_AI_AVAILABLE  
print(f"🎯 AI System Status: {'✅ ULTRA-SPECIALIZED ISTANBUL AI ACTIVE' if CUSTOM_AI_AVAILABLE else '❌ DISABLED'}")

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
    app.include_router(cache_router)
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

# === Optional Enhancement Systems Initialization ===
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

async def get_istanbul_ai_response_with_quality(user_input: str, session_id: str, user_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        
        # Prepare user context with Redis conversation context and query analysis
        user_context = {
            'session_id': session_id,
            'user_ip': user_ip,
            'timestamp': datetime.now().isoformat(),
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
        
        # Generate response using rule-based Ultra-Specialized Istanbul AI
        result = istanbul_ai_system.process_istanbul_query(user_input, user_context)
        
        if result.get('success'):
            ai_response = result['response']
            
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
        # Use our Ultra-Specialized Istanbul AI System (completely rule-based)
        if not ULTRA_ISTANBUL_AI_AVAILABLE or not istanbul_ai_system:
            print("❌ Ultra-Specialized Istanbul AI not available")
            return None
        
        # Sanitize input
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        print(f"🏛️ Using Ultra-Specialized Istanbul AI (Rule-Based) for session: {session_id}")
        
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
        
        # Generate response using rule-based Ultra-Specialized Istanbul AI
        result = istanbul_ai_system.process_istanbul_query(user_input, user_context)
        
        if result.get('success'):
            ai_response = result['response']
            
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

def get_museums_from_database() -> List[Dict[str, Any]]:
    """Fetch museums from the expanded accurate museum database (40 museums)"""
    try:
        # Use our expanded museum database instead of SQL database
        from accurate_museum_database import istanbul_museums
        
        # Format the data from our comprehensive museum database
        museum_data = []
        
        for museum_key, museum_info in istanbul_museums.museums.items():
            # Extract district from location (assuming format "District, Area" or just "District")
            location_parts = museum_info.location.split(',')
            district = location_parts[-1].strip() if location_parts else "Istanbul"
            
            # Categorize museums based on their type and name
            if 'palace' in museum_info.name.lower():
                category = 'Palace Museum'
                museum_type = 'palace'
            elif 'mosque' in museum_info.name.lower():
                category = 'Historic Mosque'
                museum_type = 'mosque'
            elif 'church' in museum_info.name.lower():
                category = 'Historic Church'
                museum_type = 'church'
            elif 'fortress' in museum_info.name.lower() or 'tower' in museum_info.name.lower():
                category = 'Historical Site'
                museum_type = 'historical'
            elif 'bazaar' in museum_info.name.lower():
                category = 'Cultural Site'
                museum_type = 'cultural'
            else:
                category = 'Museum'
                museum_type = 'museum'
            
            museum_data.append({
                'name': museum_info.name,
                'category': category,
                'district': district,
                'type': museum_type,
                'location': museum_info.location,
                'historical_period': museum_info.historical_period,
                'opening_hours': museum_info.opening_hours,
                'entrance_fee': museum_info.entrance_fee,
                'visiting_duration': museum_info.visiting_duration,
                'key_features': museum_info.key_features[:3],  # Top 3 features
                'must_see_highlights': museum_info.must_see_highlights[:3]  # Top 3 highlights
            })
        
        print(f"✅ Loaded {len(museum_data)} museums from expanded database")
        return museum_data
        
    except Exception as e:
        print(f"Error fetching museums from expanded database: {e}")
        # Fallback to SQL database if expanded database fails
        try:
            from database import SessionLocal
            from models import Place
            
            db = SessionLocal()
            museums = db.query(Place).filter(Place.category == 'Museum').all()
            historical_sites = db.query(Place).filter(Place.category == 'Historical Site').all()
            palaces = db.query(Place).filter(Place.category == 'Palace').all()
            db.close()
            
            museum_data = []
            for museum in museums:
                museum_data.append({
                    'name': museum.name,
                    'category': 'Museum',
                    'district': museum.district,
                    'type': 'museum'
                })
            for site in historical_sites:
                museum_data.append({
                    'name': site.name,
                    'category': 'Historical Site',
                    'district': site.district,
                    'type': 'historical'
                })
            for palace in palaces:
                museum_data.append({
                    'name': palace.name,
                    'category': 'Palace Museum',
                    'district': palace.district,
                    'type': 'palace'
                })
            
            print(f"⚠️ Using fallback SQL database: {len(museum_data)} museums")
            return museum_data
            
        except Exception as fallback_error:
            print(f"Error with fallback database: {fallback_error}")
            return []

def format_museums_response(museums_data: List[Dict[str, Any]]) -> str:
    """Format museums data into a comprehensive response"""
    if not museums_data:
        return """**Best Museums to Visit in Istanbul**

I'd love to help you discover Istanbul's amazing museums! Unfortunately, I'm having trouble accessing the latest museum information right now. Here are some must-visit museums:

🏛️ **Topkapi Palace Museum** - Former Ottoman imperial palace
⛪ **Hagia Sophia** - Iconic Byzantine church with incredible mosaics  
🎨 **Istanbul Modern** - Contemporary Turkish and international art
🖼️ **Pera Museum** - European art and rotating exhibitions

Would you like me to help you with specific museum information or directions?"""
    
    # Group museums by type and district
    museums_by_district = {}
    museum_types = {'museum': [], 'historical': [], 'palace': []}
    
    for museum in museums_data:
        district = museum['district']
        if district not in museums_by_district:
            museums_by_district[district] = []
        museums_by_district[district].append(museum)
        
        # Group by type
        museum_type = museum.get('type', 'museum')
        if museum_type in museum_types:
            museum_types[museum_type].append(museum)
    
    # Build the response
    response = "**Best Museums to Visit in Istanbul**\n\n"
    
    # Museums section
    if museum_types['museum']:
        response += "**🏛️ Museums:**\n"
        for museum in museum_types['museum']:
            response += f"• **{museum['name']}** - {museum['district']} district\n"
        response += "\n"
    
    # Historical sites section
    if museum_types['historical']:
        response += "**🏛️ Historical Sites & Museums:**\n"
        for site in museum_types['historical']:
            response += f"• **{site['name']}** - {site['district']} district\n"
        response += "\n"
    
    # Palace museums section
    if museum_types['palace']:
        response += "**🏰 Palace Museums:**\n"
        for palace in museum_types['palace']:
            response += f"• **{palace['name']}** - {palace['district']} district\n"
        response += "\n"
    
    # Add district-based recommendations
    response += "**📍 By District:**\n"
    for district, district_museums in sorted(museums_by_district.items()):
        if len(district_museums) > 1:
            response += f"**{district}:** "
            museum_names = [m['name'] for m in district_museums]
            response += ", ".join(museum_names) + "\n"
    
    response += "\n**💡 Visitor Tips:**\n"
    response += "• **Museum Pass Istanbul** - Skip lines and save money at major museums\n"
    response += "• **Best times:** Early morning or late afternoon to avoid crowds\n"
    response += "• **Combined visits:** Sultanahmet area has several museums within walking distance\n"
    response += "• **Photography:** Check each museum's photo policy\n\n"
    
    response += "Would you like detailed information about any specific museum or area?"
    
    return response

def post_ai_cleanup(text):
    """Post-AI cleanup pass to catch any remaining pricing and location issues while preserving readable formatting"""
    if not text:
        return text
    
    # ENHANCED PRICING REMOVAL - More aggressive patterns
    post_patterns = [
        # Direct cost patterns
        r'\b(?:costs?|prices?)\s+(?:around\s+|about\s+|approximately\s+|roughly\s+)?\d+[\d.,]*',
        r'\d+\s*(?:lira|euro|euros|dollar|dollars|pound|pounds|tl|usd|eur)\s*(?:per|each|only|approximately|around)?',
        r'(?:(?:only|just|around|about|approximately)\s+)?\d+\s*(?:lira|euro|euros|dollar|dollars|pound|pounds|tl|usd|eur)',
        r'budget\s*:?\s*\d+[\d.,]*',
        r'price\s*(?:range|list)?\s*:?\s*\d+[\d.,]*',
        r'\b\d+\s*(?:-|to|–)\s*\d+\s*(?:lira|euro|euros|dollar|dollars|pound|pounds|tl|usd|eur)',
        
        # Cost indicators with numbers
        r'(?:entrance|admission|ticket|entry)\s*(?:cost|price|fee)s?\s*:?\s*\d+[\d.,]*',
        r'(?:starting|starts)\s+(?:from|at)\s+\d+[\d.,]*',
        r'(?:pay|spend|budget)\s+(?:around|about|approximately)?\s*\d+[\d.,]*',
        r'(?:between|from)\s+\d+\s*(?:and|to|-|–)\s*\d+\s*(?:lira|euro|euros|dollar|dollars)',
        
        # Currency symbols with numbers
        r'[₺$€£¥]\s*\d+[\d.,]*',
        r'\d+[\d.,]*\s*[₺$€£¥]',
        
        # Turkish Lira specific patterns
        r'\d+\s*(?:turkish\s+)?lira',
        r'(?:turkish\s+)?lira\s*\d+',
        r'\d+\s*tl\b',
        r'\btl\s*\d+',
        
        # Additional cost phrases
        r'(?:costs?|charges?|fees?)\s+(?:around|about|approximately|roughly)?\s*\d+',
        r'(?:expensive|cheap|affordable)\s*[–-]\s*\d+',
        r'per\s+(?:person|adult|child|visit)\s*:?\s*\d+',
        r'\d+\s*per\s+(?:person|adult|child|visit|entry)',
    ]
    
    for pattern in post_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove standalone currency amounts that might have been missed
    text = re.sub(r'\b\d{1,4}[\d.,]*\s*(?=\s|$|[.,!?]|\n)', lambda m: '' if any(c in text.lower() for c in ['lira', 'euro', 'dollar', 'cost', 'price', 'fee']) else m.group(), text)
    
    # LOCATION CORRECTION - Fix mentions of wrong cities in Istanbul context
    wrong_city_patterns = [
        # Replace mentions of other Turkish cities with Istanbul context
        (r'\b(?:ankara|izmir|antalya|bursa|adana)\b(?!\s+(?:vs|versus|compared?))', 'Istanbul', re.IGNORECASE),
        # Fix "Turkey" to "Istanbul, Turkey" for restaurant context
        (r'\bin\s+turkey\b(?!\s*,)', 'in Istanbul, Turkey', re.IGNORECASE),
        (r'\brestaurants?\s+in\s+turkey\b', 'restaurants in Istanbul', re.IGNORECASE),
        (r'\bfood\s+in\s+turkey\b', 'food in Istanbul', re.IGNORECASE),
        # Fix generic Turkey mentions when clearly talking about Istanbul
        (r'\bturkey\s+offers\b', 'Istanbul offers', re.IGNORECASE),
        (r'\bturkey\s+is\s+known\b', 'Istanbul is known', re.IGNORECASE),
    ]
    
    for pattern, replacement, flags in wrong_city_patterns:
        text = re.sub(pattern, replacement, text, flags=flags)
    
    # Clean up any resulting grammar issues
    text = re.sub(r'\bIstanbul\s+Istanbul\b', 'Istanbul', text)
    text = re.sub(r'\bin\s+Istanbul,?\s+Istanbul\b', 'in Istanbul', text, flags=re.IGNORECASE)
    
    # REMOVE ALL MARKDOWN FORMATTING - Most important cleanup
    # Remove **bold** and *italic* formatting completely
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold** but keep content
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic* but keep content
    
    # Remove any remaining standalone asterisks
    text = re.sub(r'\*+', '', text)  # Remove any standalone asterisks
    text = re.sub(r'^\s*\*\s*', '', text, flags=re.MULTILINE)  # Remove asterisks at line start
    
    # Clean up double spaces that result from asterisk removal
    text = re.sub(r'  +', ' ', text)  # Multiple spaces to single space
    
    # Improve readability formatting
    # Ensure proper spacing around bullet points
    text = re.sub(r'\n•\s*([^\n]+)', r'\n• \1', text)
    
    # Add spacing around sections if missing
    text = re.sub(r'([a-z])\n([A-Z][^•\n]{10,})', r'\1\n\n\2', text)
    
    # Final cleanup - remove excessive spaces but preserve structure
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n {2,}', '\n', text)  # Remove spaces at start of lines
    text = re.sub(r' \n', '\n', text)  # Remove spaces before line breaks
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # Limit to max 3 line breaks
    
    # Remove any empty lines created by pricing removal
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

# --- Import Enhanced Query Understanding System ---
try:
    from enhanced_query_understanding import process_enhanced_query, enhanced_query_processor
    ENHANCED_QUERY_UNDERSTANDING_ENABLED = True
    print("✅ Enhanced Query Understanding System imported successfully")
except ImportError as e:
    print(f"⚠️ Enhanced Query Understanding System not available: {e}")
    ENHANCED_QUERY_UNDERSTANDING_ENABLED = False
    # Create dummy function
    def process_enhanced_query(query: str, session_id: str = None) -> Dict[str, Any]:
        return {
            'original_query': query,
            'normalized_query': query.lower().strip(),
            'intent': 'general_info',
            'confidence': 0.3,
            'entities': {},
            'corrections': [],
            'success': True
        }

# === REDIS CONVERSATIONAL MEMORY ENDPOINTS ===

@app.get("/api/redis/conversation/{session_id}")
async def get_redis_conversation(session_id: str):
    """Get conversation history from Redis for a specific session"""
    try:
        if not redis_memory:
            return {
                "success": False,
                "error": "Redis conversational memory not available",
                "session_id": session_id
            }
        
        conversation = redis_memory.get_conversation(session_id)
        preferences = redis_memory.get_preferences(session_id)
        context = redis_memory.get_context(session_id, "")
        
        return {
            "success": True,
            "session_id": session_id,
            "conversation": [
                {
                    "timestamp": turn.timestamp,
                    "user_query": turn.user_query,
                    "intent": turn.intent,
                    "entities": turn.entities,
                    "response": turn.response,
                    "confidence": turn.confidence
                }
                for turn in conversation
            ],
            "preferences": preferences.__dict__ if preferences else None,
            "context": context,
            "stats": redis_memory.get_session_stats()
        }
        
    except Exception as e:
        print(f"❌ Error getting Redis conversation: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

@app.get("/api/redis/sessions/active")
async def get_active_redis_sessions():
    """Get all active Redis sessions"""
    try:
        if not redis_memory:
            return {
                "success": False,
                "error": "Redis conversational memory not available"
            }
        
        stats = redis_memory.get_session_stats()
        
        # Get a sample of active sessions
        active_sessions = []
        if redis_client:
            try:
                session_keys = redis_client.smembers("active_sessions")
                for session_id in list(session_keys)[:10]:  # Limit to first 10
                    conversation = redis_memory.get_conversation(session_id)
                    if conversation:
                        active_sessions.append({
                            "session_id": session_id,
                            "turn_count": len(conversation),
                            "last_activity": conversation[-1].timestamp if conversation else None,
                            "recent_intent": conversation[-1].intent if conversation else None
                        })
            except Exception as e:
                print(f"⚠️ Error getting session list: {e}")
        
        return {
            "success": True,
            "stats": stats,
            "active_sessions": active_sessions,
            "redis_connected": redis_available
        }
        
    except Exception as e:
        print(f"❌ Error getting active Redis sessions: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/redis/test-conversation")
async def test_redis_conversation():
    """Test Redis conversational memory with a sample conversation"""
    try:
        if not redis_memory:
            return {
                "success": False,
                "error": "Redis conversational memory not available"
            }
        
        test_session_id = f"test_redis_{int(time.time())}"
        
        # Test conversation turns
        test_turns = [
            {
                "query": "I'm looking for a romantic restaurant in Sultanahmet",
                "intent": "restaurant_search",
                "entities": {"district": ["Sultanahmet"], "vibe": ["romantic"]},
                "response": "I recommend Pandeli Restaurant in Sultanahmet for a romantic dinner..."
            },
            {
                "query": "What about something similar in Beyoğlu?",
                "intent": "restaurant_search",
                "entities": {"district": ["Beyoğlu"], "reference": ["similar"]},
                "response": "For a similar romantic experience in Beyoğlu, try Mikla Restaurant..."
            },
            {
                "query": "How do I get there from my hotel?",
                "intent": "transportation",
                "entities": {"reference": ["there"]},
                "response": "To get to Mikla Restaurant in Beyoğlu from most hotels..."
            }
        ]
        
        # Add turns to Redis
        from redis_conversational_memory import ConversationTurn
        
        results = []
        for i, turn_data in enumerate(test_turns):
            turn = ConversationTurn(
                timestamp=datetime.now().isoformat(),
                user_query=turn_data["query"],
                normalized_query=turn_data["query"].lower().strip(),
                intent=turn_data["intent"],
                entities=turn_data["entities"],
                response=turn_data["response"],
                confidence=0.85 + (i * 0.05)  # Increasing confidence
            )
            
            success = redis_memory.add_turn(test_session_id, turn)
            results.append({
                "turn": i + 1,
                "query": turn_data["query"],
                "stored": success
            })
            
            # Small delay between turns
            time.sleep(0.1)
        
        # Get the full conversation back
        conversation = redis_memory.get_conversation(test_session_id)
        context = redis_memory.get_context(test_session_id, "follow up query")
        
        return {
            "success": True,
            "test_session_id": test_session_id,
            "turns_added": len(results),
            "turns_stored": results,
            "conversation_retrieved": len(conversation),
            "context": {
                "turn_count": context.get("turn_count", 0),
                "recent_intents": context.get("recent_intents", []),
                "recent_entities": context.get("recent_entities", {}),
                "conversation_flow": context.get("conversation_flow", [])
            },
            "redis_stats": redis_memory.get_session_stats()
        }
        
    except Exception as e:
        print(f"❌ Error testing Redis conversation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/api/redis/conversation/{session_id}")
async def clear_redis_conversation(session_id: str):
    """Clear a specific conversation from Redis"""
    try:
        if not redis_memory or not redis_client:
            return {
                "success": False,
                "error": "Redis not available"
            }
        
        # Delete conversation and preferences
        conv_key = redis_memory._get_conversation_key(session_id)
        pref_key = redis_memory._get_preferences_key(session_id)
        
        deleted_conv = redis_client.delete(conv_key)
        deleted_pref = redis_client.delete(pref_key)
        
        # Remove from active sessions
        redis_client.srem("active_sessions", session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "conversation_deleted": bool(deleted_conv),
            "preferences_deleted": bool(deleted_pref)
        }
        
    except Exception as e:
        print(f"❌ Error clearing Redis conversation: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# === Enhanced Query Understanding Test Endpoint ===
@app.post("/api/test/enhanced-query-understanding")
async def test_enhanced_query_understanding(
    query: str = Body(..., description="Query to analyze"),
    session_id: str = Body("test_session", description="Session ID for context")
):
    """Test the enhanced query understanding system with comprehensive intent categories"""
    try:
        if not ENHANCED_QUERY_UNDERSTANDING_ENABLED:
            return {
                "success": False,
                "error": "Enhanced Query Understanding System not available"
            }
        
        print(f"🧠 Testing Enhanced Query Understanding with query: '{query}'")
        
        # Process the query through enhanced understanding system
        analysis_result = process_enhanced_query(query, session_id)
        
        if analysis_result.get('success'):
            # Format results for detailed analysis
            entities = analysis_result['entities']
            
            # Extract meaningful data from entities
            entity_summary = {}
            for entity_type, values in entities.items():
                if values:  # Only include non-empty entities
                    entity_summary[entity_type] = values
            
            # Format the response
            return {
                "success": True,
                "query_analysis": {
                    "original_query": analysis_result['original_query'],
                    "normalized_query": analysis_result['normalized_query'],
                    "detected_intent": analysis_result['intent'],
                    "confidence_score": analysis_result['confidence'],
                    "spell_corrections": analysis_result['corrections'],
                    "extracted_entities": entity_summary,
                    "temporal_context": analysis_result.get('temporal_context'),
                    "vibe_tags": analysis_result.get('vibe_tags', [])
                },
                "system_capabilities": {
                    "supported_intents": [
                        # Food & Dining
                        "find_restaurant", "find_cafe",
                        # Attractions & Culture  
                        "find_attraction", "find_museum", "cultural_experience",
                        # Neighborhoods & Areas
                        "explore_neighborhood", "compare_areas",
                        # Transportation & Routes
                        "get_directions", "plan_route", "transport_info",
                        # Daily Activities & Talks
                        "daily_conversation", "local_tips",
                        # Shopping & Entertainment
                        "find_shopping", "find_entertainment", 
                        # Accommodation & Wellness
                        "find_accommodation", "wellness_spa",
                        # Nature & Outdoor
                        "find_nature",
                        # Business & Work
                        "business_info",
                        # Emergency & Practical
                        "emergency_help", "practical_info",
                        # General
                        "general_info"
                    ],
                    "entity_types": [
                        "districts", "categories", "cuisines", "transport_modes",
                        "attraction_types", "vibes", "temporal", "budget", 
                        "group_size", "duration"
                    ],
                    "spell_correction": "Turkish-aware with Istanbul locations",
                    "conversation_context": "Session-based memory support"
                },
                "processing_stats": {
                    "system_version": "enhanced_v2.0",
                    "rule_based": True,
                    "uses_ml": False,
                    "istanbul_specialized": True
                }
            }
        else:
            return {
                "success": False,
                "error": "Query analysis failed",
                "fallback_used": True
            }
            
    except Exception as e:
        print(f"❌ Enhanced Query Understanding test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# === Main AI Chat Endpoint ===
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    success: bool
    system_type: str
    quality_assessment: Optional[Dict[str, Any]] = None

@app.post("/ai", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    user_request: Request
):
    """
    Main AI chat endpoint using Ultra-Specialized Istanbul AI (Rule-Based System)
    
    This endpoint processes user queries about Istanbul using our specialized,
    rule-based AI system with enhanced query understanding and Redis memory.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get user IP for context
        user_ip = user_request.client.host if user_request.client else None
        
        # Sanitize and validate input
        user_message = sanitize_user_input(request.message)
        if not user_message:
            return ChatResponse(
                response="I need a valid message to help you with Istanbul information.",
                session_id=session_id,
                success=False,
                system_type="ultra_specialized_istanbul_ai"
            )
        
        print(f"🏛️ AI Chat Request - Session: {session_id[:8]}..., Message: '{user_message[:50]}...'")
        
        # Use the full-featured AI response system with quality assessment
        ai_result = await get_istanbul_ai_response_with_quality(user_message, session_id, user_ip)
        
        if ai_result and ai_result.get('success'):
            return ChatResponse(
                response=ai_result['response'],
                session_id=session_id,
                success=True,
                system_type=ai_result.get('system_type', 'ultra_specialized_istanbul_ai'),
                quality_assessment=ai_result.get('quality_assessment')
            )
        else:
            # Fallback response
            fallback_response = (
                "I'm here to help you explore Istanbul! You can ask me about:\n\n"
                "• Restaurants and local cuisine\n"
                "• Museums and cultural attractions\n"
                "• Neighborhoods and districts\n"
                "• Transportation and getting around\n"
                "• Shopping and entertainment\n"
                "• Daily activities and local tips\n\n"
                "What would you like to know about Istanbul?"
            )
            
            return ChatResponse(
                response=fallback_response,
                session_id=session_id,
                success=True,
                system_type="ultra_specialized_istanbul_ai_fallback"
            )
            
    except Exception as e:
        print(f"❌ AI Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            response="I'm sorry, I encountered an issue processing your request. Please try again.",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            system_type="error_fallback"
        )

# === Health Check ===
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_system": "ultra_specialized_istanbul_ai",
        "redis_available": redis_available,
        "enhanced_query_understanding": ENHANCED_QUERY_UNDERSTANDING_ENABLED,
        "version": "2.0.0"
    }