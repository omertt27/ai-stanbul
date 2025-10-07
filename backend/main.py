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

# No OpenAI API needed - using Ultra-Specialized Istanbul AI only
openai_api_key = None

# Redis availability flag and client initialization
redis_available = REDIS_AVAILABLE
redis_client = None
if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        # Test connection
        redis_client.ping()
        print("✅ Redis client initialized successfully")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        redis_available = False
        redis_client = None

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

# --- No OpenAI/GPT Dependencies ---
# We use only our Ultra-Specialized Istanbul AI System
OpenAI_available = False
OpenAI = None
print("ℹ️ OpenAI/GPT not used - using Ultra-Specialized Istanbul AI System only")

# --- Ultra-Specialized Istanbul AI System (NO GPT) ---
# Import our Ultra-Specialized Istanbul AI System
try:
    from enhanced_ultra_specialized_istanbul_ai import enhanced_istanbul_ai_system as istanbul_ai_system
    ULTRA_ISTANBUL_AI_AVAILABLE = True
    print("✅ Ultra-Specialized Istanbul AI System loaded successfully!")
except ImportError as e:
    print(f"⚠️ Ultra-Specialized Istanbul AI import failed: {e}")
    istanbul_ai_system = None
    ULTRA_ISTANBUL_AI_AVAILABLE = False

# Use only our specialized system (NO GPT, NO CustomAISystemOrchestrator)
CUSTOM_AI_AVAILABLE = ULTRA_ISTANBUL_AI_AVAILABLE
print(f"🎯 AI System Status: {'✅ ULTRA-SPECIALIZED ISTANBUL AI ACTIVE' if CUSTOM_AI_AVAILABLE else '❌ DISABLED'}")

# No need for custom_ai_system - we use istanbul_ai_system directly
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

async def get_gpt_response_with_quality(user_input: str, session_id: str, user_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Generate response using unified AI system with quality assessment"""
    try:
        # Import the unified AI system
        from unified_ai_system import get_unified_ai_system
        from database import SessionLocal
        
        # Sanitize input
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        # Get database session
        db = SessionLocal()
        
        try:
            # Use unified AI system for response generation
            unified_ai = get_unified_ai_system(db)
            
            print(f"🤖 Using unified AI system with quality assessment for session: {session_id}")
            
            # Generate response with persistent context and quality assessment
            result = await unified_ai.generate_response(
                user_input=user_input,
                session_id=session_id,
                user_ip=user_ip
            )
            
            if result.get('success'):
                ai_response = result['response']
                
                print(f"✅ Unified AI response generated - Session: {result['session_id']}, "
                      f"Context: {result.get('has_context', False)}, "
                      f"Quality: {result.get('quality_assessment', {}).get('overall_score', 0):.1f}%, "
                      f"Fallback: {result.get('quality_assessment', {}).get('used_fallback', False)}")
                
                # Apply post-processing cleanup
                ai_response = post_llm_cleanup(ai_response)
                result['response'] = ai_response
                
                return result
            else:
                print(f"❌ Unified AI system failed: {result.get('error', 'Unknown error')}")
                return None
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error in unified AI system: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def get_gpt_response(user_input: str, session_id: str, user_ip: Optional[str] = None) -> Optional[str]:
    """Generate response using unified AI system with persistent context and resolved prompt conflicts"""
    try:
        # Import the unified AI system
        from unified_ai_system import get_unified_ai_system
        from database import SessionLocal
        
        # Sanitize input
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        # Get database session
        db = SessionLocal()
        
        try:
            # Use unified AI system for response generation
            unified_ai = get_unified_ai_system(db)
            
            print(f"🤖 Using unified AI system for session: {session_id}")
            
            # Generate response with persistent context
            result = await unified_ai.generate_response(
                user_input=user_input,
                session_id=session_id,
                user_ip=user_ip
            )
            
            if result.get('success'):
                ai_response = result['response']
                session_info = unified_ai.get_session_info(result['session_id'])
                
                print(f"✅ Unified AI response generated - Session: {result['session_id']}, "
                      f"Context: {result.get('has_context', False)}, "
                      f"Turns: {result.get('conversation_turns', 0)}, "
                      f"Category: {result.get('category', 'unknown')}")
                
                # Apply post-processing cleanup
                ai_response = post_llm_cleanup(ai_response)
                
                return ai_response
            else:
                print(f"❌ Unified AI system failed: {result.get('error', 'Unknown error')}")
                return None
                
        finally:
            db.close()
            
    except ImportError as e:
        print(f"⚠️ Unified AI system not available, falling back to legacy: {e}")
        # Fallback to legacy system with minimal context
        return await get_legacy_gpt_response(user_input, session_id)
    except Exception as e:
        print(f"❌ Unified AI system error: {e}")
        return await get_legacy_gpt_response(user_input, session_id)

async def get_legacy_gpt_response(user_input: str, session_id: str) -> Optional[str]:
    """Legacy GPT response system (fallback only)"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not OpenAI_available or not openai_api_key or OpenAI is None:
            return None
        
        # Get database session for minimal personalization
        from database import SessionLocal
        db = SessionLocal()
        
        # Get basic personalized context
        try:
            from personalized_memory import get_personalized_context, generate_personalized_prompt_enhancement
            personalization = get_personalized_context(session_id, user_input, db)
            print(f"🧠 Legacy personalization context loaded: {personalization.get('has_history', False)}")
        except ImportError as e:
            print(f"⚠️ Personalized memory not available: {e}")
            personalization = {"has_history": False}
        except Exception as e:
            print(f"⚠️ Error loading personalization: {e}")
            personalization = {"has_history": False}
        
        # Use enhanced category-specific prompts
        try:
            from enhanced_gpt_prompts import get_category_specific_prompt
            from query_analyzer import analyze_and_enhance_query, LocationContext
            
            # First, get basic query analysis for location context
            analysis, _ = analyze_and_enhance_query(user_input)
            location_context = analysis.location_context.value if analysis.location_context != LocationContext.NONE else None
            
            # 📍 ENHANCED LOCATION CONTEXT PROCESSING
            enhanced_location_info = ""
            if location_context:
                try:
                    from actionability_service import actionability_enhancer
                    
                    # Get detailed location information
                    location_info = actionability_enhancer.get_location_actionable_info(location_context)
                    
                    if location_info:
                        enhanced_location_info = f"""

🌍 ENHANCED LOCATION CONTEXT FOR {location_context.upper()}:
- Exact Address Area: {location_info.exact_address}
- Nearest Metro: {location_info.nearest_metro_station}
- Walking Distance from Metro: {location_info.walking_distance_from_metro}
- GPS Coordinates: {location_info.gps_coordinates or 'Available on Google Maps'}
- Landmark References: {', '.join(location_info.landmark_references) if location_info.landmark_references else 'See local landmarks'}

LOCATION-SPECIFIC INSTRUCTIONS:
- Focus all recommendations within walking distance of {location_context}
- Include exact walking directions from the nearest metro station
- Mention specific street names and local landmarks in {location_context}
- Provide neighborhood-specific cultural context and atmosphere
- Include local insider tips specific to {location_context}

"""
                        print(f"📍 Enhanced location context added for {location_context}")
                    
                except Exception as e:
                    print(f"⚠️ Enhanced location context error: {e}")
                    enhanced_location_info = ""
            
            # Get category-specific enhanced prompt
            category, enhanced_system_prompt, max_tokens, temperature, expected_features = get_category_specific_prompt(
                user_input, location_context
            )
            
            # 🗺️ INTEGRATE COMPREHENSIVE GOOGLE MAPS DATA FOR ALL PLACE TYPES
            google_maps_data = None
            
            # Determine what type of Google Maps data to fetch based on category
            maps_search_type = None
            if category.value in ["restaurant_specific", "restaurant_general"]:
                maps_search_type = "restaurant"
            elif category.value == "museum_advice":
                maps_search_type = "museum"  
            elif category.value == "transportation":
                maps_search_type = "transportation"
            elif category.value in ["district_advice", "cultural_sites"]:
                maps_search_type = "attraction"
            
            if maps_search_type:
                try:
                    from enhanced_google_maps_service import enhanced_google_maps
                    print(f"🗺️ Fetching live {maps_search_type} data from Google Maps for query: {user_input[:50]}...")
                    
                    # Fetch comprehensive Google Maps data
                    if maps_search_type == "restaurant":
                        google_maps_data = enhanced_google_maps.search_restaurants(user_input, location_context)
                    elif maps_search_type == "museum":
                        google_maps_data = enhanced_google_maps.search_museums(user_input, location_context)
                    elif maps_search_type == "attraction":
                        google_maps_data = enhanced_google_maps.search_attractions(user_input, location_context)
                    elif maps_search_type == "transportation":
                        google_maps_data = enhanced_google_maps.search_transportation(user_input, location_context)
                    
                    if google_maps_data.get('success') and google_maps_data.get('places'):
                        places_count = len(google_maps_data['places'])
                        print(f"✅ Successfully fetched {places_count} live {maps_search_type} places from Google Maps")
                        
                        # Enhance the system prompt with real Google Maps data
                        maps_data_text = f"\n\nREAL GOOGLE MAPS {maps_search_type.upper()} DATA (Use this live data in your response):\n"
                        
                        for i, place in enumerate(google_maps_data['places'][:6], 1):
                            maps_data_text += f"{i}. {place.name}\n"
                            maps_data_text += f"   - Rating: {place.rating}/5 ({place.user_ratings_total} reviews)\n"
                            maps_data_text += f"   - Address: {place.formatted_address}\n"
                            
                            if place.price_level:
                                maps_data_text += f"   - Price Level: {place.price_level}\n"
                            
                            if place.is_open_now is not None:
                                status = "Open now" if place.is_open_now else "Currently closed"
                                maps_data_text += f"   - Status: {status}\n"
                            
                            if place.phone:
                                maps_data_text += f"   - Phone: {place.phone}\n"
                            
                            if place.website:
                                maps_data_text += f"   - Website: {place.website}\n"
                            
                            maps_data_text += f"   - Google Maps: {place.google_maps_link}\n"
                            
                            if place.fact_checked:
                                maps_data_text += f"   - ✅ Fact-checked from: {place.official_source}\n"
                            
                            maps_data_text += "\n"
                        
                        maps_data_text += f"Search performed for: {google_maps_data['location_context']} - {user_input[:50]}\n"
                        maps_data_text += f"Data retrieved: {google_maps_data['timestamp']}\n\n"
                        maps_data_text += f"IMPORTANT: Use the above real Google Maps data in your response. Include exact names, addresses, ratings, and contact details from this live data.\n"
                        
                        # Append the real data to the system prompt
                        enhanced_system_prompt += maps_data_text
                        print(f"✅ Enhanced system prompt with live Google Maps {maps_search_type} data")
                        
                    else:
                        print(f"⚠️ Google Maps {maps_search_type} data not available: {google_maps_data.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"⚠️ Error fetching Google Maps {maps_search_type} data: {e}")
                    google_maps_data = None
            
            # 🚌🏛️ ENHANCED SERVICES INTEGRATION
            enhanced_service_data = ""
            if ENHANCED_SERVICES_ENABLED:
                try:
                    # Real Transportation Service Integration (with fallback to enhanced service)
                    if category.value == "transportation" or any(word in user_message.lower() for word in ['metro', 'bus', 'transport', 'how to get', 'travel to', 'route', 'ferry', 'ferries', 'boat', 'boats', 'cross', 'bosphorus', 'golden horn', 'prince islands', 'kadıköy', 'üsküdar', 'beşiktaş', 'eminönü', 'karaköy', 'kabataş', 'directions', 'way to', 'get from', 'get to']):
                        print(f"🚌 Fetching real-time transportation data...")
                        
                        # Try real transportation service first
                        transport_data = None
                        if REAL_TRANSPORT_SERVICE_ENABLED and real_transportation_service:
                            try:
                                # Extract origin and destination from user input
                                origin = location_context or "Current Location"
                                destination = user_input  # Simplified - could be improved with NLP
                                
                                # Get real-time routes
                                routes = await real_transportation_service.get_real_time_routes(origin, destination)
                                if routes:
                                    # Get service alerts
                                    service_alerts = await real_transportation_service.get_service_alerts()
                                    
                                    transport_data = {
                                        'success': True,
                                        'routes': [
                                            {
                                                'summary': f"{route.transport_type.title()} {route.line_number or route.line_name}",
                                                'duration': f"{route.duration_minutes} min",
                                                'cost': f"{route.cost_tl:.2f} TL",
                                                'instructions': route.instructions[:2],
                                                'next_departures': route.next_departures,
                                                'real_time_status': route.real_time_status
                                            } for route in routes[:5]
                                        ],
                                        'service_alerts': service_alerts
                                    }
                                    print(f"✅ Real transportation service provided {len(routes)} routes")
                            except Exception as e:
                                print(f"⚠️ Real transportation service error: {e}")
                        
                        # Fallback to enhanced service if real service fails
                        if not transport_data:
                            transport_data = enhanced_transport_service.get_transportation_info(user_input, location_context)
                        
                        if transport_data.get('success'):
                            enhanced_service_data += f"\n\nREAL-TIME ISTANBUL TRANSPORTATION DATA:\n"
                            
                            # Route information
                            if transport_data.get('routes'):
                                enhanced_service_data += f"📍 RECOMMENDED ROUTES:\n"
                                for route in transport_data['routes'][:3]:  # Top 3 routes
                                    enhanced_service_data += f"• {route['summary']}\n"
                                    enhanced_service_data += f"  Duration: {route['duration']} | Distance: {route['distance']}\n"
                                    enhanced_service_data += f"  Instructions: {route['instructions']}\n\n"
                            
                            # Live transit data
                            if transport_data.get('live_data'):
                                enhanced_service_data += f"🚇 LIVE TRANSIT STATUS:\n"
                                live_data = transport_data['live_data']
                                for line, status in live_data.items():
                                    enhanced_service_data += f"• {line}: {status}\n"
                                enhanced_service_data += f"\n"
                            
                            print(f"✅ Enhanced transportation data integrated")
                    
                    # Real Museum Service Integration (with fallback to enhanced service)
                    if category.value == "museum_advice" or any(word in user_message.lower() for word in ['museum', 'gallery', 'exhibition', 'art', 'history', 'culture']):
                        print(f"🏛️ Fetching real museum data...")
                        
                        # Try real museum service first
                        museum_data = None
                        if REAL_MUSEUM_SERVICE_ENABLED and real_museum_service:
                            try:
                                # Extract museum name or search for all museums
                                if any(museum_name in user_message.lower() for museum_name in ['hagia sophia', 'topkapi', 'blue mosque', 'basilica cistern', 'galata tower', 'dolmabahce']):
                                    # Get specific museum
                                    # Extract museum name for specific queries
                                    museum_name = None
                                    for name in ['hagia sophia', 'topkapi', 'blue mosque', 'basilica cistern']:
                                        if name in user_message.lower():
                                            museum_name = name
                                            break
                                    
                                    if museum_name:
                                        museum_info = await real_museum_service.get_museum_info(museum_name)
                                        if museum_info:
                                            museum_data = {
                                                'success': True,
                                                'museums': [{
                                                    'name': museum_info.name,
                                                    'location': museum_info.address,
                                                    'highlights': f"Rating: {museum_info.rating}/5 ({museum_info.total_ratings} reviews)",
                                                    'practical_info': f"Status: {museum_info.current_status}, Hours: {', '.join(museum_info.opening_hours_text[:2])}",
                                                    'cultural_context': f"Website: {museum_info.website or 'N/A'}"
                                                }],
                                                'real_time_data': True
                                            }
                                else:
                                    # Get all museums for general queries
                                    all_museums = await real_museum_service.get_all_museums()
                                    if all_museums:
                                        museum_data = {
                                            'success': True,
                                            'museums': [
                                                {
                                                    'name': museum_info.name,
                                                    'location': museum_info.address,
                                                    'highlights': f"Rating: {museum_info.rating}/5" if museum_info.rating else "Popular Istanbul museum",
                                                    'practical_info': f"Status: {museum_info.current_status}",
                                                    'cultural_context': f"Phone: {museum_info.phone or 'N/A'}"
                                                } for museum_info in list(all_museums.values())[:4]
                                            ],
                                            'real_time_data': True
                                        }
                                print(f"✅ Real museum service provided data")
                            except Exception as e:
                                print(f"⚠️ Real museum service error: {e}")
                        
                        # Fallback to enhanced service if real service fails
                        if not museum_data:
                            museum_data = enhanced_museum_service.get_museum_info(user_input, location_context)
                        
                        if museum_data.get('success'):
                            enhanced_service_data += f"\n\nENHANCED MUSEUM & CULTURAL SITE DATA:\n"
                            
                            # Museum recommendations
                            if museum_data.get('museums'):
                                enhanced_service_data += f"🏛️ RECOMMENDED MUSEUMS:\n"
                                for museum in museum_data['museums'][:4]:  # Top 4 museums
                                    enhanced_service_data += f"• {museum['name']}\n"
                                    enhanced_service_data += f"  Location: {museum['location']}\n"
                                    enhanced_service_data += f"  Highlights: {museum['highlights']}\n"
                                    enhanced_service_data += f"  Practical Info: {museum['practical_info']}\n"
                                    if museum.get('cultural_context'):
                                        enhanced_service_data += f"  Cultural Context: {museum['cultural_context']}\n"
                                    enhanced_service_data += f"\n"
                            
                            # Cultural insights
                            if museum_data.get('cultural_insights'):
                                enhanced_service_data += f"🎭 CULTURAL INSIGHTS:\n{museum_data['cultural_insights']}\n\n"
                            
                            print(f"✅ Enhanced museum data integrated")
                    
                    # Add enhanced service data to system prompt
                    if enhanced_service_data:
                        enhanced_service_data += f"IMPORTANT: Use the above enhanced service data to provide detailed, actionable responses with specific transit routes, museum details, and cultural context.\n"
                        enhanced_system_prompt += enhanced_service_data
                        print(f"✅ Enhanced services data added to system prompt")
                        
                except Exception as e:
                    print(f"⚠️ Error integrating enhanced services: {e}")
                    
            # 📍 ADD ENHANCED LOCATION CONTEXT TO PROMPT
            if enhanced_location_info:
                enhanced_system_prompt += enhanced_location_info
            
            # 🌤️ ADD SEASONAL CONTEXT TO PROMPT
            try:
                from datetime import datetime
                import calendar
                
                current_month = datetime.now().month
                season_info = {
                    (12, 1, 2): "Winter season - indoor attractions preferred, cozy atmosphere",
                    (3, 4, 5): "Spring season - tulip season, moderate crowds, pleasant visiting conditions", 
                    (6, 7, 8): "Summer season - peak tourist time, early morning/evening visits recommended",
                    (9, 10, 11): "Autumn season - beautiful lighting, fewer crowds, ideal visiting time"
                }
                
                seasonal_context = None
                for months, description in season_info.items():
                    if current_month in months:
                        seasonal_context = description
                        break
                
                if seasonal_context:
                    season_context = f"""
� CURRENT SEASONAL INFORMATION:
{seasonal_context}

Seasonal Recommendations:
- Consider seasonal timing for outdoor activities
- Account for crowd patterns and tourist seasons
- Suggest appropriate clothing and preparation

IMPORTANT: When providing recommendations, consider the current season. For activity suggestions, timing advice, or preparation recommendations, use this seasonal information to provide contextually appropriate advice.

"""
                    enhanced_system_prompt += season_context
                    print(f"✅ Enhanced system prompt with seasonal context: {seasonal_context}")
                
            except Exception as e:
                print(f"⚠️ Error adding seasonal context: {e}")
            
            # 🧠 ENHANCE PROMPT WITH PERSONALIZATION
            if personalization.get('has_history'):
                enhanced_system_prompt = generate_personalized_prompt_enhancement(personalization, enhanced_system_prompt)
                print(f"🧠 Personalized prompt enhancement applied")
            
            # Dynamic token optimization based on query complexity
            query_complexity_indicators = [
                len(user_input.split()) > 15,  # Long query
                any(word in user_input.lower() for word in ['detailed', 'comprehensive', 'everything', 'all about']),  # Detailed request
                any(word in user_input.lower() for word in ['walking', 'directions', 'step by step']),  # Needs detailed instructions
                user_input.count('?') > 1,  # Multiple questions
                any(word in user_input.lower() for word in ['compare', 'difference', 'versus', 'vs'])  # Comparison request
            ]
            
            complexity_score = sum(query_complexity_indicators)
            
            # Adjust tokens based on complexity and category (add extra for personalization)
            if complexity_score >= 3:  # High complexity
                max_tokens = min(max_tokens + 100, 800)  # Increase but cap at 800
                print(f"🔧 High complexity query detected, increasing tokens to {max_tokens}")
            elif complexity_score <= 1 and category.value in ['daily_talk', 'safety_practical']:  # Simple queries
                max_tokens = max(max_tokens - 100, 300)  # Decrease but minimum 300
                print(f"🔧 Simple query detected, optimizing tokens to {max_tokens}")
            
            # Add tokens for personalization if history exists
            if personalization.get('has_history'):
                max_tokens = min(max_tokens + 50, 850)  # Small boost for personalization
            
            print(f"🎯 Enhanced Prompts - Category: {category.value}, Location: {location_context}, Max Tokens: {max_tokens}, Personalized: {personalization.get('has_history', False)}")
            
            # Use the enhanced system prompt and user input directly
            system_prompt = enhanced_system_prompt
            user_message = user_input
            use_enhanced_prompts = True
            
        except ImportError as e:
            print(f"⚠️ Enhanced prompts not available: {e}, using legacy system")
            use_enhanced_prompts = False
            # Fallback to legacy system
            try:
                from query_analyzer import analyze_and_enhance_query, QueryType, LocationContext
                from location_enhancer import get_location_enhanced_gpt_prompt, enhance_response_with_location_context
                
                analysis, enhanced_user_prompt = analyze_and_enhance_query(user_input)
                
                print(f"🔍 Query Analysis - Type: {analysis.query_type.value}, Location: {analysis.location_context.value}, Confidence: {analysis.confidence_score:.2f}")
                
                # Further enhance prompt with location-specific context
                if analysis.location_context != LocationContext.NONE:
                    enhanced_user_prompt = get_location_enhanced_gpt_prompt(enhanced_user_prompt, analysis.location_context.value)
                    print(f"📍 Location-enhanced prompt generated for {analysis.location_context.value}")
                
                # Use enhanced prompt instead of basic user input
                user_message = enhanced_user_prompt
                
            except ImportError:
                print("⚠️ Query analyzer/location enhancer not available, using basic prompt")
                user_message = f"Question about Istanbul: {user_input}"
                analysis = None
        
        # Only use legacy system prompt if enhanced prompts are not available
        if not use_enhanced_prompts:
            # Legacy system prompt
            if analysis and analysis.location_context != LocationContext.NONE:
                location_focus = f"\n\nSPECIAL LOCATION FOCUS: The user is asking specifically about {analysis.location_context.value.title()}. Make sure your entire response is focused on this area with specific local details, walking distances to landmarks, and practical information for that neighborhood."
            else:
                location_focus = ""
                
            system_prompt = f"""You are an expert Istanbul travel assistant with deep knowledge of the city. Provide comprehensive, informative responses about Istanbul tourism, culture, attractions, restaurants, and travel tips.

CRITICAL RULES:
1. LOCATION FOCUS: Only provide information about ISTANBUL, Turkey. If asked about other cities (Ankara, Izmir, etc.), politely redirect to Istanbul or clarify that you specialize in Istanbul only.
2. NO PRICING: Never include specific prices, costs, fees, or monetary amounts. Instead use terms like "affordable", "moderate", "upscale", "budget-friendly", "varies".
3. NO SPECIFIC PRICING: Avoid all pricing symbols, numbers with cost terms, or specific amounts. Use "budget-friendly", "moderate", "upscale" instead.
4. DIRECT RELEVANCE: Answer exactly what the user asks. Don't provide generic information - be specific to their query.

Guidelines:
- Give DIRECT, HELPFUL answers - avoid asking for clarification unless absolutely necessary
- Include specific names of places, attractions, districts, and landmarks IN ISTANBUL
- Provide practical information: hours, locations, transportation details (but NOT costs/prices)
- Mention key topics and keywords relevant to the question
- Be enthusiastic but informative (250-500 words)
- For districts/neighborhoods: mention key attractions, character, and what makes them special
- For museums/attractions: include historical context, highlights, and practical visiting tips
- For transportation: provide specific routes and alternatives (but avoid specific costs)
- For general questions: give comprehensive overviews with specific examples
- Always include relevant keywords and topics that tourists would search for

IMPORTANT: When answering questions, make sure to include these specific terms when relevant:
- Time/planning: "days", "time needed", "planning", "itinerary suggestions"
- Cultural uniqueness: "East meets West", "diversity", "cultural bridge", "two continents"
- Transportation: "metro lines", "Golden Horn crossing", "BiTaksi app", "taxi apps", "Istanbulkart"
- Districts: "Jewish quarter" (for Balat), "Ottoman houses", "ferry terminal" (for Eminönü), "Instagram-worthy", "romantic atmosphere"
- Museums: "European style architecture", "luxury interiors", "guided tours", "skip-the-line tickets"
- Restaurants: "food court options", "snacks and treats", "service charge policy", "tipping percentage"

Key Istanbul topics to reference when relevant:
- Sultanahmet (historic district with Hagia Sophia, Blue Mosque, Topkapi Palace)
- Beyoğlu (modern area with Istiklal Street, Galata Tower, nightlife)
- Kadiköy (Asian side, authentic, local markets, Moda)
- Galata (trendy cafes, art galleries, views)
- Bosphorus (bridges, ferry rides, waterfront)
- Transportation (metro, tram, ferry, Istanbulkart, BiTaksi)
- Districts, museums, restaurants, culture, history, Byzantine, Ottoman, Asia/Europe{location_focus}"""
            
            max_tokens = 450
            temperature = 0.7
        
        # Create OpenAI client with improved reliability settings
        client = OpenAI(api_key=openai_api_key, timeout=45.0, max_retries=3)
        
        # Make the API call with enhanced prompt and retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=40
                )
                break  # Success, exit retry loop
            except Exception as e:
                print(f"⚠️ OpenAI API attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    print(f"❌ All {max_attempts} attempts failed")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
        
        gpt_response = response.choices[0].message.content
        if gpt_response:
            gpt_response = gpt_response.strip()
            
            # Apply post-LLM cleanup to remove pricing and fix location issues
            gpt_response = post_llm_cleanup(gpt_response)
            
            # 🔍 FACT-CHECKING LAYER
            fact_check_result = None
            try:
                from fact_checking_service import fact_checker
                category_name = category.value if 'category' in locals() else "general"
                
                print(f"🔍 Running fact-check for {category_name} response...")
                fact_check_result = fact_checker.fact_check_response(gpt_response, category_name)
                
                print(f"✅ Fact-check completed - Accuracy: {fact_check_result.accuracy_score:.2f}, Verified facts: {len(fact_check_result.verified_facts)}")
                
                # If accuracy is very low, let GPT handle it with a warning
                if fact_checker.should_use_gpt_fallback(fact_check_result.accuracy_score, category_name):
                    print(f"⚠️ Low accuracy detected ({fact_check_result.accuracy_score:.2f}) - Adding verification notice")
                    gpt_response += f"\n\n⚠️ VERIFICATION NOTICE: Some information in this response may need verification. Please check official sources for critical details like opening hours, prices, and schedules."
                
            except ImportError:
                print("⚠️ Fact-checking service not available")
            except Exception as e:
                print(f"⚠️ Fact-checking error: {e}")
            
            # 🎯 ENHANCED ACTIONABILITY & CULTURAL ENHANCEMENT
            try:
                if ENHANCED_SERVICES_ENABLED:
                    # Use enhanced actionability service with Turkish support and structured format
                    category_name = category.value if 'category' in locals() else "general"
                    
                    print(f"🎯 Applying enhanced actionability with cultural context for {category_name}...")
                    enhanced_result = enhanced_actionability_service.enhance_response_actionability(
                        gpt_response, user_input, category_name, location_context
                    )
                    
                    if enhanced_result.get("success"):
                        actionability_score = enhanced_result.get("actionability_score", 0.5)
                        print(f"✅ Enhanced actionability applied - Score: {actionability_score:.2f}")
                        
                        # Apply structured formatting (Address → Directions → Timing → Tips)
                        if enhanced_result.get("structured_response"):
                            gpt_response = enhanced_result["structured_response"]
                            print(f"📋 Structured format applied (Address → Directions → Timing → Tips)")
                        
                        # Add Turkish language support and cultural context
                        if enhanced_result.get("cultural_enhancement"):
                            gpt_response += f"\n\n{enhanced_result['cultural_enhancement']}"
                            print(f"🇹🇷 Cultural context and Turkish phrases added")
                        
                        # Add local insights if available
                        if enhanced_result.get("local_insights"):
                            gpt_response += f"\n\n💡 **Local Insider Tips:** {enhanced_result['local_insights']}"
                            print(f"💡 Local insider tips added")
                    else:
                        print(f"⚠️ Enhanced actionability failed: {enhanced_result.get('error', 'Unknown error')}")
                else:
                    # Fallback to original actionability service
                    from actionability_service import actionability_enhancer
                    category_name = category.value if 'category' in locals() else "general"
                    
                    print(f"🎯 Using fallback actionability enhancement for {category_name}...")
                    actionability_result = actionability_enhancer.enhance_response_actionability(
                        gpt_response, user_input, category_name
                    )
                    
                    if actionability_result.get("success"):
                        actionability_score = actionability_result["actionability_score"]
                        print(f"✅ Fallback actionability enhanced - Score: {actionability_score:.2f}")
                        
                        if actionability_score < 0.7 and actionability_result.get("enhanced_response"):
                            print(f"🔧 Using enhanced response due to low actionability ({actionability_score:.2f})")
                            gpt_response = actionability_result["enhanced_response"]
                    else:
                        print(f"⚠️ Fallback actionability enhancement failed: {actionability_result.get('error', 'Unknown error')}")
                    
            except ImportError:
                print("⚠️ Actionability enhancement services not available")
            except Exception as e:
                print(f"⚠️ Actionability enhancement error: {e}")
            
            # Apply format enforcement for enhanced prompts
            if use_enhanced_prompts:
                try:
                    from format_enforcer import enforce_response_format
                    original_response = gpt_response
                    gpt_response = enforce_response_format(gpt_response, category)
                    
                    if gpt_response != original_response:
                        print(f"📋 Format enforcement applied for category: {category.value}")
                    
                except ImportError:
                    print("⚠️ Format enforcer not available, using original response")
            
            # Enhanced response quality checking and improvement
            try:
                from advanced_response_enhancer import enhance_response_quality
                
                # Use expected features from enhanced prompts if available
                expected_feats = expected_features if 'expected_features' in locals() else []
                category_name = category.value if 'category' in locals() else "generic"
                
                # Enhance response if quality is low
                enhanced_response = enhance_response_quality(
                    gpt_response, category_name, user_input
                )
                
                if enhanced_response != gpt_response:
                    print(f"🔧 Response enhanced due to low quality detection")
                    gpt_response = enhanced_response
                    
            except ImportError:
                print("⚠️ Response quality enhancer not available")
            except Exception as e:
                print(f"❌ Enhancement error: {e}")
            
            print(f"📝 Enhanced GPT response generated successfully")
            
            # Smart response length management
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                print(f"⚠️ Response truncated due to token limit for query ID: {hash(user_input) % 10000}")
                
                # Try to create a more complete response by optimizing content
                if not gpt_response.endswith(('.', '!', '?', ':')):
                    # Response was cut off mid-sentence - need better handling
                    # Find the last complete sentence
                    sentences = gpt_response.split('.')
                    if len(sentences) > 1:
                        # Keep all complete sentences, add summary
                        complete_sentences = '.'.join(sentences[:-1]) + '.'
                        gpt_response = complete_sentences + f"\n\n📝 For more detailed information about this topic, please ask a more specific question."
                    else:
                        gpt_response += "\n\n*[Ask a more specific question for complete details.]*"
                else:
                    # Natural ending - likely complete response despite length limit
                    print(f"✅ Response appears complete despite token limit")
                    
                # Log for optimization
                word_count = len(gpt_response.split())
                print(f"📊 Response stats: {word_count} words, Max tokens: {max_tokens}, Category: {category.value if 'category' in locals() else 'unknown'}")
            
            # Enhanced feature analysis for debugging
            try:
                from enhanced_feature_detection import analyze_response_features
                
                # Use category from enhanced prompts if available
                category_name = category.value if 'category' in locals() else "generic"
                expected_feats = expected_features if 'expected_features' in locals() else []
                
                feature_analysis = analyze_response_features(gpt_response, category_name, expected_feats)
                
                print(f"📊 Feature Analysis - Completeness: {feature_analysis['completeness_score']:.1f}/5.0, Coverage: {feature_analysis['coverage_percentage']:.1f}%, Features: {feature_analysis['total_features_detected']}")
                
                if feature_analysis['missing_features']:
                    print(f"⚠️ Missing features: {', '.join(feature_analysis['missing_features'][:3])}")
                    
            except ImportError:
                print("📊 Feature analysis not available")
            
            print(f"✅ GPT response generated successfully for query ID: {hash(user_input) % 10000}")
            return gpt_response
        else:
            print(f"⚠️ GPT returned empty content for query ID: {hash(user_input) % 10000}")
            return None
        
    except Exception as e:
        print(f"❌ GPT response generation failed: {e}")
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

def post_llm_cleanup(text):
    """Post-LLM cleanup pass to catch any remaining pricing and location issues while preserving readable formatting"""
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

# --- Import Content Quality Enhancement ---
try:
    from content_quality_enhancer import enhance_user_content_quality, content_quality_enhancer
    from realtime_content_adapter import adapt_content_realtime, realtime_content_adapter
    CONTENT_QUALITY_ENHANCEMENT_ENABLED = True
    print("✅ Content Quality Enhancement and Real-time Adaptation loaded successfully")
except ImportError as e:
    print(f"⚠️ Content Quality Enhancement not available: {e}")
    CONTENT_QUALITY_ENHANCEMENT_ENABLED = False
    # Create dummy functions
    def enhance_user_content_quality(response, user_id, query, category, context=None):
        return {"enhanced_response": response, "enhancements_applied": 0}
    async def adapt_content_realtime(response, user_id, session_id, query, interaction_data=None, context=None):
        return {"adapted_response": response, "adaptations_applied": 0}
    
    # Create dummy objects
    class DummyContentEnhancer:
        def update_user_profile(self, *args, **kwargs): pass
    content_quality_enhancer = DummyContentEnhancer()

# ===== MAIN AI CHAT ENDPOINTS =====
# These endpoints were missing due to accidental deletion

@app.post("/ai/chat")
async def ai_chat_endpoint(request: Request):
    """Main AI chat endpoint for handling user queries"""
    start_time = time.time()
    session_id = None
    user_ip = None
    
    try:
        # Get user IP
        user_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        
        # Parse request
        data = await request.json()
        user_input = data.get("user_input", "").strip()
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        # Input validation
        if not user_input:
            raise HTTPException(status_code=400, detail="Empty input provided")
        
        if len(user_input) > 2000:
            raise HTTPException(status_code=400, detail="Input too long")
        
        # Rate limiting and daily usage tracking completely removed for unrestricted testing
        
        # Daily usage limits completely removed
        
        print(f"🤖 Chat request from {user_ip} (session: {session_id}): {user_input[:50]}...")
        
        # Track request metrics
        system_metrics["requests_total"] += 1
        
        # Generate AI response using the high-quality function
        response_data = await get_gpt_response_with_quality(user_input, session_id, user_ip)
        
        if response_data and response_data.get('success'):
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            system_metrics["response_times"].append(response_time)
            
            # Keep only last 100 response times
            if len(system_metrics["response_times"]) > 100:
                system_metrics["response_times"] = system_metrics["response_times"][-100:]
            
            # 🎯 CONTENT QUALITY ENHANCEMENT
            enhanced_response = response_data['response']
            content_enhancements = {}
            adaptation_results = {}
            
            if CONTENT_QUALITY_ENHANCEMENT_ENABLED:
                try:
                    # Determine content category from query
                    content_category = "general"
                    if any(word in user_input.lower() for word in ['restaurant', 'food', 'eat', 'dining']):
                        content_category = "restaurant"
                    elif any(word in user_input.lower() for word in ['museum', 'gallery', 'art', 'history']):
                        content_category = "museum"
                    elif any(word in user_input.lower() for word in ['district', 'neighborhood', 'area']):
                        content_category = "district"
                    elif any(word in user_input.lower() for word in ['transport', 'metro', 'bus', 'taxi']):
                        content_category = "transportation"
                    
                    # 1. Apply content quality enhancement
                    quality_context = {
                        "location": "Istanbul",
                        "session_id": session_id,
                        "response_time_ms": response_time,
                        "turn_number": 1  # Could be tracked more accurately
                    }
                    
                    content_enhancements = enhance_user_content_quality(
                        enhanced_response, 
                        user_ip,  # Using IP as user ID for now
                        user_input, 
                        content_category, 
                        quality_context
                    )
                    
                    if content_enhancements.get("enhanced_response"):
                        enhanced_response = content_enhancements["enhanced_response"]
                        print(f"✅ Content quality enhanced - Level: {content_enhancements.get('quality_level', 'unknown')}, "
                              f"Enhancements: {content_enhancements.get('enhancements_applied', 0)}")
                    
                    # 2. Apply real-time content adaptation
                    interaction_data = {
                        "query": user_input,
                        "category": content_category,
                        "response_time": 30.0,  # Default assumption
                        "query_complexity": len(user_input.split()) / 20.0,  # Simple complexity estimate
                        "satisfaction_signals": [],  # Would be populated by frontend feedback
                        "user_rating": response_data.get('quality_assessment', {}).get('overall_score', 0.7),
                        "content_type": content_category,
                        "turn_number": 1
                    }
                    
                    adaptation_results = await adapt_content_realtime(
                        enhanced_response,
                        user_ip,  # Using IP as user ID
                        session_id,
                        user_input,
                        interaction_data,
                        quality_context
                    )
                    
                    if adaptation_results.get("adapted_response") and adaptation_results.get("adaptations_applied", 0) > 0:
                        enhanced_response = adaptation_results["adapted_response"]
                        print(f"✅ Real-time adaptation applied - Adaptations: {adaptation_results.get('adaptations_applied', 0)}, "
                              f"Engagement: {adaptation_results.get('engagement_level', 'unknown')}")
                    
                    # Update user profile for future enhancements
                    content_quality_enhancer.update_user_profile(user_ip, {
                        "query": user_input,
                        "category": content_category,
                        "response_quality": response_data.get('quality_assessment', {}),
                        "session_id": session_id
                    })
                    
                except Exception as enhancement_error:
                    print(f"⚠️ Content enhancement error: {enhancement_error}")
                    # Continue with original response if enhancement fails
                    enhanced_response = response_data['response']
            
            return {
                "response": enhanced_response,
                "session_id": response_data.get('session_id', session_id),
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": response_time,
                "source": "ai_system_enhanced",
                "quality_score": response_data.get('quality_assessment', {}).get('overall_score', 0),
                "has_context": response_data.get('has_context', False),
                "content_enhancements": {
                    "quality_level": content_enhancements.get('quality_level', 'enhanced'),
                    "enhancements_applied": content_enhancements.get('enhancements_applied', 0),
                    "personalization_score": content_enhancements.get('personalization_score', 0.0),
                    "adaptations_applied": adaptation_results.get('adaptations_applied', 0),
                    "engagement_level": adaptation_results.get('engagement_level', 'medium_engagement'),
                    "recommended_followups": content_enhancements.get('recommended_followups', [])
                }
            }
        else:
            # Fallback response
            fallback_response = {
                "response": "I'm here to help you explore Istanbul! Ask me about restaurants, museums, attractions, or anything else about the city.",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": (time.time() - start_time) * 1000,
                "source": "fallback"
            }
            return fallback_response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        system_metrics["errors"] += 1
        
        # Return error response
        return {
            "response": "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
            "session_id": session_id or f"session_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": (time.time() - start_time) * 1000,
            "source": "error",
            "error": "Internal processing error"
        }

@app.post("/ai/stream")
async def ai_stream_endpoint(request: Request):
    """Streaming AI endpoint for real-time chat responses"""
    try:
        # Get user IP
        user_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        
        # Parse request
        data = await request.json()
        user_input = data.get("user_input", "").strip()
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        # Input validation
        if not user_input:
            raise HTTPException(status_code=400, detail="Empty input provided")
        
        if len(user_input) > 2000:
            raise HTTPException(status_code=400, detail="Input too long")
        
        print(f"🌊 Streaming request from {user_ip} (session: {session_id}): {user_input[:50]}...")
        
        async def generate_streaming_response():
            """Generate streaming response chunks"""
            try:
                # Get AI response
                response_data = await get_gpt_response_with_quality(user_input, session_id, user_ip)
                
                if response_data and response_data.get('success'):
                    response_text = response_data['response']
                    
                    # Stream the response word by word with realistic timing
                    words = response_text.split()
                    current_text = ""
                    
                    for i, word in enumerate(words):
                        current_text += word + " "
                        
                        # Create streaming chunk
                        chunk = {
                            "content": current_text.strip(),
                            "is_complete": i == len(words) - 1,
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Send chunk
                        yield f"data: {json.dumps(chunk)}\n\n"
                        
                        # Add small delay between words for realistic streaming
                        await asyncio.sleep(0.05)  # 50ms delay
                    
                    # Send final completion chunk
                    final_chunk = {
                        "content": current_text.strip(),
                        "is_complete": True,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "quality_score": response_data.get('quality_assessment', {}).get('overall_score', 0)
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                
                else:
                    # Fallback streaming response
                    fallback_text = "I'm here to help you explore Istanbul! Ask me about restaurants, museums, attractions, or anything else about the city."
                    
                    chunk = {
                        "content": fallback_text,
                        "is_complete": True,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "source": "fallback"
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
            except Exception as e:
                print(f"❌ Error in streaming response: {e}")
                
                # Send error chunk
                error_chunk = {
                    "content": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                    "is_complete": True,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": "Processing error"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        # Return streaming response
        return StreamingResponse(
            generate_streaming_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in streaming endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/debug/redis-status")
async def get_redis_status():
    """Enhanced Redis status endpoint for debugging and monitoring"""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "redis_available": redis_available,
            "redis_client_exists": redis_client is not None,
            "session_management": {
                "enabled": redis_available,
                "type": "redis" if redis_available else "memory"
            },
            "environment": {
                "redis_host": os.getenv("REDIS_HOST", "localhost"),
                "redis_port": os.getenv("REDIS_PORT", "6379"),
                "redis_db": os.getenv("REDIS_DB", "0")
            }
        }
        
        # Test Redis connection if available
        if redis_client:
            try:
                # Test basic operations
                test_key = f"health_check_{int(time.time())}"
                redis_client.set(test_key, "test_value", ex=10)  # Expire in 10 seconds
                retrieved_value = redis_client.get(test_key)
                redis_client.delete(test_key)
                
                status["redis_test"] = {
                    "connection": "success",
                    "write_read": "success" if retrieved_value == "test_value" else "failed",
                    "ping": redis_client.ping()
                }
                
                # Get Redis info
                redis_info = redis_client.info()
                status["redis_info"] = {
                    "version": redis_info.get("redis_version", "unknown"),
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0)
                }
                
            except Exception as e:
                status["redis_test"] = {
                    "connection": "failed",
                    "error": str(e)
                }
        else:
            status["redis_test"] = {
                "connection": "not_available",
                "reason": "Redis client not initialized"
            }
        
        # System status
        if PSUTIL_AVAILABLE:
            import psutil
            status["system"] = {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "disk_percent": psutil.disk_usage('/').percent
            }
        
        return status
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "redis_available": False
        }

@app.post("/api/admin/login")
async def admin_login(request: Request):
    """Admin authentication endpoint"""
    try:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")
        
        # Authenticate admin
        user = authenticate_admin(username, password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create access token
        access_token = create_access_token(data={"sub": username, "role": "admin"})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Admin login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

print("✅ Essential endpoints configured successfully")

# === ADMIN DASHBOARD ENDPOINTS ===
# Chat Sessions Management with Feedback Tracking

@app.get("/api/chat-sessions")
async def get_all_chat_sessions():
    """Get all saved chat sessions with feedback data for admin dashboard"""
    try:
        from database import SessionLocal
        
        # Try to get sessions from saved_session_manager if available
        if AI_INTELLIGENCE_ENABLED and saved_session_manager:
            try:
                sessions = saved_session_manager.get_all_sessions()
                
                # Transform sessions for admin dashboard
                session_data = []
                for session in sessions:
                    # Calculate feedback statistics
                    feedback_stats = {"likes": 0, "dislikes": 0, "mixed": 0}
                    conversation_history = session.get('conversation_history', [])
                    
                    for entry in conversation_history:
                        if entry.get('feedback') == 'like':
                            feedback_stats["likes"] += 1
                        elif entry.get('feedback') == 'dislike':
                            feedback_stats["dislikes"] += 1
                    
                    if feedback_stats["likes"] > 0 and feedback_stats["dislikes"] > 0:
                        feedback_stats["mixed"] = 1
                    
                    session_data.append({
                        "id": session.get('id', session.get('session_id', 'unknown')),
                        "title": session.get('title', session.get('conversation_title', 'Untitled Session')),
                        "user_ip": session.get('user_ip', 'unknown'),
                        "saved_at": session.get('saved_at', session.get('created_at', datetime.now().isoformat())),
                        "message_count": len(conversation_history),
                        "conversation_history": conversation_history,
                        "feedback_stats": feedback_stats,
                        "has_feedback": feedback_stats["likes"] > 0 or feedback_stats["dislikes"] > 0
                    })
                
                # Filter to only include sessions with feedback for admin observation
                sessions_with_feedback = [s for s in session_data if s["has_feedback"]]
                
                return {
                    "success": True,
                    "sessions": sessions_with_feedback,
                    "total_count": len(sessions_with_feedback),
                    "feedback_summary": {
                        "total_likes": sum(s["feedback_stats"]["likes"] for s in sessions_with_feedback),
                        "total_dislikes": sum(s["feedback_stats"]["dislikes"] for s in sessions_with_feedback),
                        "mixed_feedback_sessions": sum(s["feedback_stats"]["mixed"] for s in sessions_with_feedback)
                    }
                }
                
            except Exception as e:
                print(f"⚠️ Error getting sessions from saved_session_manager: {e}")
        
        # Fallback: Try to get sessions from database
        try:
            db = SessionLocal()
            
            # Try to get sessions from SavedChatSession model if available
            try:
                from models import SavedChatSession
                saved_sessions = db.query(SavedChatSession).all()
                
                session_data = []
                for session in saved_sessions:
                    # Parse conversation history to extract feedback
                    conversation_history = []
                    feedback_stats = {"likes": 0, "dislikes": 0, "mixed": 0}
                    
                    try:
                        if hasattr(session, 'conversation_history') and session.conversation_history:
                            if isinstance(session.conversation_history, str):
                                import json
                                conversation_history = json.loads(session.conversation_history)
                            else:
                                conversation_history = session.conversation_history
                            
                            # Count feedback
                            for entry in conversation_history:
                                if entry.get('feedback') == 'like':
                                    feedback_stats["likes"] += 1
                                elif entry.get('feedback') == 'dislike':
                                    feedback_stats["dislikes"] += 1
                            
                            if feedback_stats["likes"] > 0 and feedback_stats["dislikes"] > 0:
                                feedback_stats["mixed"] = 1
                    
                    except Exception as parse_error:
                        print(f"⚠️ Error parsing conversation history: {parse_error}")
                    
                    session_data.append({
                        "id": str(session.id),
                        "title": session.title or "Untitled Session",
                        "user_ip": getattr(session, 'user_ip', 'unknown'),
                        "saved_at": session.created_at.isoformat() if hasattr(session, 'created_at') else datetime.now().isoformat(),
                        "message_count": len(conversation_history),
                        "conversation_history": conversation_history,
                        "feedback_stats": feedback_stats,
                        "has_feedback": feedback_stats["likes"] > 0 or feedback_stats["dislikes"] > 0
                    })
                
                db.close()
                
                # Filter to sessions with feedback only
                sessions_with_feedback = [s for s in session_data if s["has_feedback"]]
                
                return {
                    "success": True,
                    "sessions": sessions_with_feedback,
                    "total_count": len(sessions_with_feedback),
                    "feedback_summary": {
                        "total_likes": sum(s["feedback_stats"]["likes"] for s in sessions_with_feedback),
                        "total_dislikes": sum(s["feedback_stats"]["dislikes"] for s in sessions_with_feedback),
                        "mixed_feedback_sessions": sum(s["feedback_stats"]["mixed"] for s in sessions_with_feedback)
                    }
                }
                
            except ImportError:
                print("⚠️ SavedChatSession model not available")
                db.close()
        
        except Exception as db_error:
            print(f"⚠️ Database error: {db_error}")
        
        # Return empty result if no sessions found
        return {
            "success": True,
            "sessions": [],
            "total_count": 0,
            "message": "No chat sessions with feedback found",
            "feedback_summary": {
                "total_likes": 0,
                "total_dislikes": 0,
                "mixed_feedback_sessions": 0
            }
        }
        
    except Exception as e:
        print(f"❌ Error getting chat sessions: {e}")
        return {
            "success": False,
            "error": str(e),
            "sessions": [],
            "total_count": 0
        }

@app.get("/api/chat-sessions/{session_id}")
async def get_chat_session_details(session_id: str):
    """Get detailed information about a specific chat session"""
    try:
        from database import SessionLocal
        
        # Try saved_session_manager first
        if AI_INTELLIGENCE_ENABLED and saved_session_manager:
            try:
                session = saved_session_manager.get_session(session_id)
                if session:
                    # Calculate detailed feedback statistics
                    conversation_history = session.get('conversation_history', [])
                    feedback_details = []
                    
                    for i, entry in enumerate(conversation_history):
                        feedback_info = {
                            "message_index": i,
                            "query": entry.get('query', ''),
                            "response": entry.get('response', ''),
                            "feedback": entry.get('feedback'),
                            "timestamp": entry.get('timestamp', ''),
                            "response_time_ms": entry.get('response_time_ms', 0)
                        }
                        feedback_details.append(feedback_info)
                    
                    return {
                        "success": True,
                        "session": {
                            "id": session_id,
                            "title": session.get('title', 'Untitled Session'),
                            "user_ip": session.get('user_ip', 'unknown'),
                            "saved_at": session.get('saved_at', session.get('created_at', datetime.now().isoformat())),
                            "message_count": len(conversation_history),
                            "conversation_history": conversation_history,
                            "feedback_details": feedback_details
                        }
                    }
            except Exception as e:
                print(f"⚠️ Error getting session from saved_session_manager: {e}")
        
        # Fallback to database
        try:
            db = SessionLocal()
            from models import SavedChatSession
            
            session = db.query(SavedChatSession).filter(SavedChatSession.id == session_id).first()
            db.close()
            
            if session:
                # Parse conversation history
                conversation_history = []
                try:
                    if hasattr(session, 'conversation_history') and session.conversation_history:
                        if isinstance(session.conversation_history, str):
                            import json
                            conversation_history = json.loads(session.conversation_history)
                        else:
                            conversation_history = session.conversation_history
                except Exception as parse_error:
                    print(f"⚠️ Error parsing conversation history: {parse_error}")
                
                return {
                    "success": True,
                    "session": {
                        "id": str(session.id),
                        "title": session.title or "Untitled Session",
                        "user_ip": getattr(session, 'user_ip', 'unknown'),
                        "saved_at": session.created_at.isoformat() if hasattr(session, 'created_at') else datetime.now().isoformat(),
                        "message_count": len(conversation_history),
                        "conversation_history": conversation_history
                    }
                }
        
        except ImportError:
            print("⚠️ SavedChatSession model not available")
        except Exception as db_error:
            print(f"⚠️ Database error: {db_error}")
        
        return {
            "success": False,
            "error": "Session not found",
            "session": None
        }
        
    except Exception as e:
        print(f"❌ Error getting session details: {e}")
        return {
            "success": False,
            "error": str(e),
            "session": None
        }

@app.delete("/api/chat-sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a specific chat session"""
    try:
        from database import SessionLocal
        
        # Try saved_session_manager first
        if AI_INTELLIGENCE_ENABLED and saved_session_manager:
            try:
                result = saved_session_manager.delete_session(session_id)
                if result:
                    return {
                        "success": True,
                        "message": "Session deleted successfully"
                    }
            except Exception as e:
                print(f"⚠️ Error deleting session from saved_session_manager: {e}")
        
        # Fallback to database
        try:
            db = SessionLocal()
            from models import SavedChatSession
            
            session = db.query(SavedChatSession).filter(SavedChatSession.id == session_id).first()
            
            if session:
                db.delete(session)
                db.commit()
                db.close()
                
                return {
                    "success": True,
                    "message": "Session deleted successfully"
                }
            else:
                db.close()
                return {
                    "success": False,
                    "error": "Session not found"
                }
        
        except ImportError:
            print("⚠️ SavedChatSession model not available")
        except Exception as db_error:
            print(f"⚠️ Database error: {db_error}")
        
        return {
            "success": False,
            "error": "Session not found or could not be deleted"
        }
        
    except Exception as e:
        print(f"❌ Error deleting session: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/admin/feedback-analytics")
async def get_feedback_analytics():
    """Get analytics about user feedback (likes/dislikes) for admin dashboard"""
    try:
        from database import SessionLocal
        
        analytics = {
            "total_feedback_sessions": 0,
            "total_likes": 0,
            "total_dislikes": 0,
            "like_rate": 0.0,
            "dislike_rate": 0.0,
            "mixed_feedback_sessions": 0,
            "feedback_by_category": {},
            "recent_feedback": []
        }
        
        # Try to get feedback data from saved sessions
        if AI_INTELLIGENCE_ENABLED and saved_session_manager:
            try:
                sessions = saved_session_manager.get_all_sessions()
                
                for session in sessions:
                    conversation_history = session.get('conversation_history', [])
                    session_has_feedback = False
                    
                    for entry in conversation_history:
                        feedback = entry.get('feedback')
                        if feedback:
                            session_has_feedback = True
                            
                            if feedback == 'like':
                                analytics["total_likes"] += 1
                            elif feedback == 'dislike':
                                analytics["total_dislikes"] += 1
                            
                            # Add to recent feedback (last 10)
                            if len(analytics["recent_feedback"]) < 10:
                                analytics["recent_feedback"].append({
                                    "session_id": session.get('id', 'unknown'),
                                    "feedback": feedback,
                                    "query": entry.get('query', '')[:100] + "..." if len(entry.get('query', '')) > 100 else entry.get('query', ''),
                                    "timestamp": entry.get('timestamp', ''),
                                    "user_ip": session.get('user_ip', 'unknown')
                                })
                    
                    if session_has_feedback:
                        analytics["total_feedback_sessions"] += 1
                
            except Exception as e:
                print(f"⚠️ Error getting analytics from saved_session_manager: {e}")
        
        # Calculate rates
        total_feedback = analytics["total_likes"] + analytics["total_dislikes"]
        if total_feedback > 0:
            analytics["like_rate"] = round((analytics["total_likes"] / total_feedback) * 100, 1)
            analytics["dislike_rate"] = round((analytics["total_dislikes"] / total_feedback) * 100, 1)
        
        # Sort recent feedback by timestamp (most recent first)
        analytics["recent_feedback"].sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {
            "success": True,
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting feedback analytics: {e}")
        return {
            "success": False,
            "error": str(e),
            "analytics": None
        }

# Blog Management Endpoints
@app.get("/api/admin/blog-posts")
async def get_all_blog_posts():
    """Get all blog posts for admin management"""
    try:
        from database import SessionLocal
        from models import BlogPost as BlogPostModel
        
        db = SessionLocal()
        blog_posts = db.query(BlogPostModel).order_by(BlogPostModel.created_at.desc()).all()
        
        posts_data = []
        for post in blog_posts:
            posts_data.append({
                "id": post.id,
                "title": post.title,
                "slug": post.slug,
                "content": post.content[:200] + "..." if len(post.content) > 200 else post.content,
                "full_content": post.content,
                "author": post.author,
                "created_at": post.created_at.isoformat(),
                "updated_at": post.updated_at.isoformat() if post.updated_at else None,
                "is_published": getattr(post, 'is_published', True),
                "view_count": getattr(post, 'view_count', 0),
                "like_count": getattr(post, 'like_count', 0),
                "category": getattr(post, 'category', 'general'),
                "tags": getattr(post, 'tags', [])
            })
        
        db.close()
        
        return {
            "success": True,
            "posts": posts_data,
            "total_count": len(posts_data)
        }
        
    except Exception as e:
        print(f"❌ Error getting blog posts: {e}")
        return {
            "success": False,
            "error": str(e),
            "posts": []
        }

@app.get("/api/admin/blog-comments")
async def get_all_blog_comments():
    """Get all blog comments for admin management"""
    try:
        from database import SessionLocal
        
        # Try to get comments from database
        try:
            from models import BlogComment
            
            db = SessionLocal()
            comments = db.query(BlogComment).order_by(BlogComment.created_at.desc()).all()
            
            comments_data = []
            for comment in comments:
                comments_data.append({
                    "id": comment.id,
                    "blog_post_id": comment.blog_post_id,
                    "author_name": comment.author_name,
                    "author_email": getattr(comment, 'author_email', ''),
                    "content": comment.content,
                    "created_at": comment.created_at.isoformat(),
                    "is_approved": getattr(comment, 'is_approved', True),
                    "is_spam": getattr(comment, 'is_spam', False)
                })
            
            db.close()
            
            return {
                "success": True,
                "comments": comments_data,
                "total_count": len(comments_data)
            }
            
        except ImportError:
            print("⚠️ BlogComment model not available")
            return {
                "success": True,
                "comments": [],
                "total_count": 0,
                "message": "Blog comments feature not available"
            }
        
    except Exception as e:
        print(f"❌ Error getting blog comments: {e}")
        return {
            "success": False,
            "error": str(e),
            "comments": []
        }

@app.delete("/api/admin/blog-posts/{post_id}")
async def delete_blog_post(post_id: int):
    """Delete a blog post"""
    try:
        from database import SessionLocal
        from models import BlogPost as BlogPostModel
        
        db = SessionLocal()
        post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        
        if post:
            db.delete(post)
            db.commit()
            db.close()
            
            return {
                "success": True,
                "message": "Blog post deleted successfully"
            }
        else:
            db.close()
            return {
                "success": False,
                "error": "Blog post not found"
            }
        
    except Exception as e:
        print(f"❌ Error deleting blog post: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/api/admin/blog-comments/{comment_id}")
async def delete_blog_comment(comment_id: int):
    """Delete a blog comment"""
    try:
        from database import SessionLocal
        from models import BlogComment
        
        db = SessionLocal()
        comment = db.query(BlogComment).filter(BlogComment.id == comment_id).first()
        
        if comment:
            db.delete(comment)
            db.commit()
            db.close()
            
            return {
                "success": True,
                "message": "Comment deleted successfully"
            }
        else:
            db.close()
            return {
                "success": False,
                "error": "Comment not found"
            }
        
    except ImportError:
        return {
            "success": False,
            "error": "Blog comments feature not available"
        }
    except Exception as e:
        print(f"❌ Error deleting comment: {e}")
        return {
            "success": False,
            "error": str(e)
        }

print("✅ Admin dashboard endpoints configured successfully")

# === END ESSENTIAL ENDPOINTS ===