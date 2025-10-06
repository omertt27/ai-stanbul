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
from collections import defaultdict

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

# Daily usage tracking
daily_usage = defaultdict(int)  # IP -> count
last_reset_date = date.today()
DAILY_LIMIT = 200  # 200 requests per IP per day (increased for testing)

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

# --- Rate Limiting and Security ---
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_ENABLED = True
    print("✅ Rate limiting (slowapi) loaded successfully")
except ImportError as e:
    print(f"⚠️ Rate limiting not available - install slowapi: {e}")
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

# Rate limiter initialization
limiter = None
if RATE_LIMITING_ENABLED:
    try:
        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        print("✅ Rate limiting enabled successfully")
    except Exception as e:
        print(f"⚠️ Rate limiting initialization failed: {e}")
        RATE_LIMITING_ENABLED = False
        limiter = None
else:
    print("⚠️ Rate limiting disabled")

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
                                    museum_key = extract_museum_key_from_input(user_message)
                                    if museum_key:
                                        museum_info = await real_museum_service.get_museum_info(museum_key)
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

# === Redis Session Manager Integration ===
try:
    from redis_session_manager import RedisSessionManager, get_redis_session_manager
    from multi_turn_query_handler import MultiTurnQueryHandler, ConversationStack, ConversationTurn
    from context_aware_filtering import ContextAwareFilteringSystem
    from database import SessionLocal
    
    # Initialize Redis session manager
    redis_session_manager = None
    def init_redis_session_manager():
        global redis_session_manager
        try:
            db = SessionLocal()
            redis_session_manager = RedisSessionManager(
                redis_url="redis://localhost:6379/0",
                db_session=db
            )
            print("✅ Redis Session Manager initialized successfully")
            return redis_session_manager
        except Exception as e:
            print(f"⚠️ Redis Session Manager initialization failed: {e}")
            return None
    
    # Try to initialize
    redis_session_manager = init_redis_session_manager()
    REDIS_SESSION_ENABLED = redis_session_manager is not None
    
    # Initialize Multi-Turn Query Handler
    multi_turn_handler = MultiTurnQueryHandler(max_history_turns=10)
    print("✅ Multi-Turn Query Handler initialized successfully")
    
    # Initialize Context-Aware Filtering System
    context_filtering = ContextAwareFilteringSystem()
    print("✅ Context-Aware Filtering System initialized successfully")
    
except ImportError as e:
    print(f"⚠️ Redis session manager not available: {e}")
    REDIS_SESSION_ENABLED = False
    redis_session_manager = None
    multi_turn_handler = None
    context_filtering = None

print(f"Redis Session Management Status: {'✅ ENABLED' if REDIS_SESSION_ENABLED else '❌ DISABLED'}")

# === SESSION & CONTEXT MANAGEMENT ENDPOINTS ===

@app.post("/api/session/create")
async def create_session_endpoint(request: Request):
    """Create or get session with context management"""
    try:
        client_ip = getattr(request.client, 'host', 'unknown')
        user_agent = request.headers.get('user-agent', '')
        
        body = await request.json()
        session_id = body.get('session_id')
        
        if REDIS_SESSION_ENABLED and redis_session_manager:
            # Use Redis-based session management
            session_id, context = redis_session_manager.get_or_create_session(
                session_id=session_id,
                user_ip=client_ip,
                user_agent=user_agent
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "context": {
                    "conversation_stage": context.conversation_stage,
                    "entities_count": {k: len(v) for k, v in context.entities.items()},
                    "places_mentioned": context.places_mentioned[-5:],  # Last 5
                    "last_queries_count": len(context.last_queries),
                    "user_preferences": context.user_preferences
                },
                "session_type": "redis_managed",
                "created_at": context.created_at
            }
        else:
            # Fallback to basic session creation
            if not session_id:
                session_id = str(uuid.uuid4())
            
            return {
                "success": True,
                "session_id": session_id,
                "context": {
                    "conversation_stage": "initial",
                    "entities_count": {},
                    "places_mentioned": [],
                    "last_queries_count": 0,
                    "user_preferences": {"budget": "medium", "distance_limit_km": 2}
                },
                "session_type": "basic",
                "created_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": str(uuid.uuid4()),  # Fallback session
            "session_type": "fallback"
        }

@app.get("/api/session/{session_id}/context")
async def get_session_context_endpoint(session_id: str):
    """Get session context and conversation history"""
    try:
        if REDIS_SESSION_ENABLED and redis_session_manager:
            context = redis_session_manager.get_session_context(session_id)
            
            if context:
                # Get cross-query reference resolution
                resolution = redis_session_manager.resolve_cross_query_references(
                    session_id, ""  # Empty query for general context
                )
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "context": {
                        "last_queries": context.last_queries[-5:],  # Last 5 queries
                        "entities": context.entities,
                        "user_preferences": context.user_preferences,
                        "current_intent": context.current_intent,
                        "conversation_stage": context.conversation_stage,
                        "places_mentioned": context.places_mentioned,
                        "topics_discussed": context.topics_discussed,
                        "last_activity": context.last_activity
                    },
                    "reference_resolution": resolution,
                    "session_summary": redis_session_manager.get_session_summary(session_id)
                }
            else:
                return {
                    "success": False,
                    "error": "Session context not found",
                    "session_id": session_id
                }
        else:
            return {
                "success": False,
                "error": "Redis session management not available",
                "fallback_context": {
                    "conversation_stage": "initial",
                    "entities": {},
                    "user_preferences": {"budget": "medium", "distance_limit_km": 2}
                }
            }
            
    except Exception as e:
        print(f"❌ Get context error: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

@app.post("/api/session/{session_id}/update-context")
async def update_session_context_endpoint(session_id: str, request: Request):
    """Update session context with new query and extracted entities"""
    try:
        body = await request.json()
        query = body.get('query', '').strip()
        intent = body.get('intent', 'general')
        entities = body.get('entities', {})
        ai_response = body.get('ai_response', '')
        
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        
        if REDIS_SESSION_ENABLED and redis_session_manager:
            success = redis_session_manager.update_session_context(
                session_id=session_id,
                query=query,
                intent=intent,
                entities=entities,
                ai_response=ai_response
            )
            
            if success:
                # Get updated context
                updated_context = redis_session_manager.get_session_context(session_id)
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": "Context updated successfully",
                    "updated_context": {
                        "current_intent": updated_context.current_intent,
                        "conversation_stage": updated_context.conversation_stage,
                        "entities_count": {k: len(v) for k, v in updated_context.entities.items()},
                        "places_mentioned_count": len(updated_context.places_mentioned),
                        "queries_count": len(updated_context.last_queries)
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to update context",
                    "session_id": session_id
                }
        else:
            return {
                "success": False,
                "error": "Redis session management not available",
                "session_id": session_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Update context error: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

@app.post("/api/session/{session_id}/resolve-references")
async def resolve_references_endpoint(session_id: str, request: Request):
    """Resolve cross-query references like 'which is closest?', 'that restaurant'"""
    try:
        body = await request.json()
        current_query = body.get('query', '').strip()
        
        if not current_query:
            raise HTTPException(status_code=400, detail="query is required")
        
        if REDIS_SESSION_ENABLED and redis_session_manager:
            resolution = redis_session_manager.resolve_cross_query_references(
                session_id, current_query
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "query": current_query,
                "resolution": resolution,
                "resolved": resolution["resolved"],
                "suggested_context": resolution.get("context", {}),
                "suggested_entities": resolution.get("suggested_entities", [])
            }
        else:
            return {
                "success": False,
                "error": "Redis session management not available",
                "session_id": session_id,
                "query": current_query
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Resolve references error: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

@app.get("/api/session/stats")
async def session_stats_endpoint():
    """Get session management statistics"""
    try:
        if REDIS_SESSION_ENABLED and redis_session_manager:
            # Clean up expired sessions and get stats
            cleaned = redis_session_manager.cleanup_expired_sessions()
            
            return {
                "success": True,
                "redis_enabled": True,
                "cleaned_sessions": cleaned,
                "session_ttl_hours": redis_session_manager.session_ttl / 3600,
                "context_ttl_hours": redis_session_manager.context_ttl / 3600,
                "redis_connected": True
            }
        else:
            return {
                "success": True,
                "redis_enabled": False,
                "message": "Using fallback session management"
            }
            
    except Exception as e:
        print(f"❌ Session stats error: {e}")
        return {
            "success": False,
            "error": str(e),
            "redis_enabled": REDIS_SESSION_ENABLED
        }

print("✅ Session & Context Management endpoints configured")

# === MAIN AI ENDPOINTS ===

@app.post("/ai")
async def ai_istanbul_router(request: Request):
    """Main AI endpoint for chatbot queries with Redis session management"""
    session_id = "unknown"
    try:
        data = await request.json()
        user_input = data.get("user_input", "")
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        # Enhanced input validation
        if not user_input or len(user_input.strip()) < 1:
            return {"response": "I'm here to help you explore Istanbul! What would you like to know?", "session_id": session_id}
        
        if len(user_input.strip()) < 2:
            return {"response": "I'm here to help you explore Istanbul! What would you like to know?", "session_id": session_id}
        
        # Check for spam-like input
        if re.search(r'(.)\1{4,}', user_input):
            return {"response": "Sorry, I couldn't understand that. Please ask me about Istanbul attractions, restaurants, or travel information.", "session_id": session_id}
        
        # Check for only special characters or numbers
        if not re.search(r'[a-zA-Z\u0600-\u06FF\u00C0-\u017F\u0400-\u04FF]', user_input):
            return {"response": "Sorry, I couldn't understand that. Please ask me about Istanbul attractions, restaurants, or travel information.", "session_id": session_id}
        
        # Sanitize input - basic cleanup
        user_input = clean_text_formatting(user_input)
        
        # === REDIS SESSION MANAGEMENT INTEGRATION ===
        # Get or create session with context
        context = {}
        if REDIS_SESSION_ENABLED and redis_session_manager:
            try:
                session_context = redis_session_manager.get_session_context(session_id)
                if session_context:
                    if hasattr(session_context, 'to_dict'):
                        context = session_context.to_dict()
                    else:
                        context = session_context
                    print(f"✅ Retrieved session context for {session_id}: {len(context.get('last_queries', []))} previous queries")
                else:
                    # Create new session
                    new_session_id, new_context = redis_session_manager.get_or_create_session(
                        session_id=session_id,
                        user_ip=client_ip
                    )
                    session_id = new_session_id
                    context = new_context.to_dict()
                    print(f"✅ Created new session: {session_id}")
            except Exception as e:
                print(f"⚠️ Redis session error: {e}")
                # Continue without Redis session management
        
        # Get database session
        db = SessionLocal()
        try:
            client_ip = request.client.host if request.client else "unknown"
            
            # === MULTI-TURN QUERY PROCESSING ===
            # Process query for follow-up references and context
            processed_query = user_input
            conversation_context = {}
            
            if REDIS_SESSION_ENABLED and redis_session_manager and multi_turn_handler:
                try:
                    # Get or load conversation stack from Redis
                    conversation_stack = None
                    redis_key = f"conversation_stack:{session_id}"
                    
                    try:
                        stack_data = redis_session_manager.redis_client.get(redis_key)
                        if stack_data:
                            stack_dict = json.loads(stack_data)
                            conversation_stack = ConversationStack.from_dict(stack_dict)
                            print(f"✅ Loaded conversation stack with {len(conversation_stack.turns)} turns")
                        else:
                            # Create new conversation stack
                            conversation_stack = ConversationStack(
                                session_id=session_id,
                                turns=[],
                                current_context={},
                                conversation_topic="",
                                last_results={},
                                reference_cache={}
                            )
                            print(f"✅ Created new conversation stack for session {session_id}")
                    except Exception as e:
                        print(f"⚠️ Error loading conversation stack: {e}")
                        conversation_stack = ConversationStack(
                            session_id=session_id,
                            turns=[],
                            current_context={},
                            conversation_topic="",
                            last_results={},
                            reference_cache={}
                        )
                    
                    # Check if this is a follow-up query
                    if conversation_stack.turns:
                        is_followup, followup_type = multi_turn_handler.is_follow_up_query(user_input)
                        
                        if is_followup:
                            print(f"🔄 Detected follow-up query: '{user_input}' (type: {followup_type})")
                            
                            # Resolve follow-up query using conversation context
                            followup_result = multi_turn_handler.resolve_follow_up_query(
                                user_input, conversation_stack
                            )
                            
                            if followup_result.get('enhanced_query') and followup_result['enhanced_query'] != user_input:
                                processed_query = followup_result['enhanced_query']
                                print(f"✅ Enhanced query: '{processed_query}'")
                            
                            # Store conversation context for potential use
                            conversation_context = {
                                'followup_type': followup_type,
                                'context_provided': followup_result.get('context_provided', False),
                                'last_results': conversation_stack.last_results,
                                'current_topic': conversation_stack.conversation_topic
                            }
                            print(f"✅ Retrieved conversation context: topic='{conversation_stack.conversation_topic}'")
                    
                except Exception as e:
                    print(f"⚠️ Multi-turn processing error: {e}")
                    # Continue with original query
                    processed_query = user_input
            
            # Generate AI response using the unified system (with processed query and context-aware filtering)
            if context_filtering and 'conversation_stack' in locals() and conversation_stack:
                ai_response = await get_context_aware_gpt_response(processed_query, session_id, client_ip, conversation_stack)
            else:
                ai_response = await get_gpt_response(processed_query, session_id, client_ip)
            
            if not ai_response:
                # Fallback response
                ai_response = "I can help you explore Istanbul! Ask me about restaurants, museums, districts, or transportation."
            
            # Clean the response
            clean_response = clean_text_formatting(ai_response)
            
            # === UPDATE REDIS SESSION CONTEXT & CONVERSATION STACK ===
            if REDIS_SESSION_ENABLED and redis_session_manager:
                try:
                    # Extract basic entities and intent (simple implementation)
                    entities = {}
                    intent = 'general_travel_info'
                    
                    # Simple entity extraction
                    istanbul_districts = ['sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'taksim', 'eminonu', 'fatih']
                    for district in istanbul_districts:
                        if district in user_input.lower():
                            if 'locations' not in entities:
                                entities['locations'] = []
                            entities['locations'].append(district.title())
                    
                    # Simple intent detection
                    if any(word in user_input.lower() for word in ['restaurant', 'food', 'eat', 'dining']):
                        intent = 'restaurant_search'
                    elif any(word in user_input.lower() for word in ['museum', 'gallery', 'history']):
                        intent = 'museum_inquiry'
                    elif any(word in user_input.lower() for word in ['transport', 'metro', 'bus', 'taxi']):
                        intent = 'transportation_info'
                    elif any(word in user_input.lower() for word in ['place', 'attraction', 'visit', 'see']):
                        intent = 'place_recommendation'
                    
                    # Update session context (original method)
                    redis_session_manager.update_session_context(
                        session_id=session_id,
                        query=user_input,
                        intent=intent,
                        entities=entities,
                        ai_response=clean_response
                    )
                    
                    # === UPDATE CONVERSATION STACK FOR MULTI-TURN ===
                    if multi_turn_handler and 'conversation_stack' in locals():
                        try:
                            # Create conversation turn
                            conversation_turn = multi_turn_handler.create_conversation_turn(
                                user_query=user_input,
                                ai_response=clean_response,
                                intent=intent,
                                entities=entities
                            )
                            
                            # Add turn to conversation stack
                            conversation_stack = multi_turn_handler.update_conversation_stack(conversation_stack, conversation_turn)
                            
                            # Save updated conversation stack to Redis
                            redis_key = f"conversation_stack:{session_id}"
                            stack_json = json.dumps(conversation_stack.to_dict())
                            redis_session_manager.redis_client.setex(
                                redis_key, 
                                3600,  # 1 hour expiry
                                stack_json
                            )
                            
                            print(f"✅ Updated conversation stack for {session_id}: {len(conversation_stack.turns)} turns, topic='{conversation_stack.conversation_topic}'")
                            
                        except Exception as e:
                            print(f"⚠️ Failed to update conversation stack: {e}")
                    
                    print(f"✅ Updated session context for {session_id}: intent={intent}, entities={entities}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to update Redis session context: {e}")
            
            # Store in database for legacy support
            try:
                chat_record = ChatHistory(
                    user_message=user_input,
                    ai_response=clean_response,
                    session_id=session_id,
                    user_ip=client_ip
                )
                db.add(chat_record)
                db.commit()
                print(f"✅ Stored chat in database for session {session_id}")
            except Exception as e:
                print(f"⚠️ Database storage failed: {e}")
            
            return {"response": clean_response, "session_id": session_id}
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Critical error in AI endpoint: {e}")
        return {"response": "Sorry, I encountered an error. Please try again.", "session_id": session_id}

@app.post("/ai/stream")
async def ai_istanbul_streaming(request: Request):
    """Streaming AI endpoint for real-time responses with session management"""
    try:
        data = await request.json()
        user_input = data.get("user_input", "")
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        if not user_input:
            return {"response": "I'm here to help you explore Istanbul! What would you like to know?", "session_id": session_id}
        
        # Get or update session context
        context = {}
        if REDIS_SESSION_ENABLED and redis_session_manager:
            try:
                session_context = redis_session_manager.get_session_context(session_id)
                if session_context:
                    if hasattr(session_context, 'to_dict'):
                        context = session_context.to_dict()
                    else:
                        context = session_context
                else:
                    # Create new session for streaming
                    session_id, context = redis_session_manager.get_or_create_session(
                        session_id=session_id,
                        user_ip=client_ip
                    )
                    context = context.to_dict()
            except Exception as e:
                print(f"⚠️ Redis session error in streaming: {e}")
        
        client_ip = request.client.host if request.client else "unknown"
        
        # === MULTI-TURN QUERY PROCESSING FOR STREAMING ===
        processed_query = user_input
        if REDIS_SESSION_ENABLED and redis_session_manager and multi_turn_handler:
            try:
                # Load conversation stack for streaming
                redis_key = f"conversation_stack:{session_id}"
                stack_data = redis_session_manager.redis_client.get(redis_key)
                if stack_data:
                    stack_dict = json.loads(stack_data)
                    conversation_stack = ConversationStack.from_dict(stack_dict)
                    
                    # Check for follow-up query
                    if conversation_stack.turns:
                        is_followup, followup_type = multi_turn_handler.is_follow_up_query(user_input)
                        if is_followup:
                            followup_result = multi_turn_handler.resolve_follow_up_query(user_input, conversation_stack)
                            if followup_result.get('enhanced_query'):
                                processed_query = followup_result['enhanced_query']
                                print(f"🔄 Streaming follow-up enhanced: '{processed_query}'")
            except Exception as e:
                print(f"⚠️ Streaming multi-turn error: {e}")
        
        # Generate streaming response with context-aware filtering if available
        conversation_stack_for_streaming = None
        if REDIS_SESSION_ENABLED and redis_session_manager and multi_turn_handler:
            try:
                redis_key = f"conversation_stack:{session_id}"
                stack_data = redis_session_manager.redis_client.get(redis_key)
                if stack_data:
                    stack_dict = json.loads(stack_data)
                    conversation_stack_for_streaming = ConversationStack.from_dict(stack_dict)
            except Exception as e:
                print(f"⚠️ Error loading conversation stack for streaming context filtering: {e}")
        
        if context_filtering and conversation_stack_for_streaming:
            ai_response = await get_context_aware_gpt_response(processed_query, session_id, client_ip, conversation_stack_for_streaming)
        else:
            ai_response = await get_gpt_response(processed_query, session_id, client_ip)
        
        if not ai_response:
            ai_response = "I can help you explore Istanbul! Ask me about restaurants, museums, districts, or transportation."
        
        # Update session context after streaming response
        if REDIS_SESSION_ENABLED and redis_session_manager:
            try:
                # Simple intent and entity extraction
                intent = 'general_travel_info'
                entities = {}
                
                if any(word in user_input.lower() for word in ['restaurant', 'food', 'eat']):
                    intent = 'restaurant_search'
                elif any(word in user_input.lower() for word in ['museum', 'gallery']):
                    intent = 'museum_inquiry'
                
                redis_session_manager.update_session_context(
                    session_id=session_id,
                    query=user_input,
                    intent=intent,
                    entities=entities,
                    ai_response=ai_response
                )
                
                # Update conversation stack for streaming too
                if multi_turn_handler:
                    try:
                        redis_key = f"conversation_stack:{session_id}"
                        stack_data = redis_session_manager.redis_client.get(redis_key)
                        if stack_data:
                            stack_dict = json.loads(stack_data)
                            conversation_stack = ConversationStack.from_dict(stack_dict)
                        else:
                            conversation_stack = ConversationStack(
                                session_id=session_id, turns=[], current_context={},
                                conversation_topic="", last_results={}, reference_cache={}
                            )
                        
                        # Create and add turn
                        conversation_turn = multi_turn_handler.create_conversation_turn(
                            user_query=user_input, ai_response=ai_response, intent=intent, entities=entities
                        )
                        conversation_stack = multi_turn_handler.update_conversation_stack(conversation_stack, conversation_turn)
                        
                        # Save back to Redis
                        stack_json = json.dumps(conversation_stack.to_dict())
                        redis_session_manager.redis_client.setex(redis_key, 3600, stack_json)
                        
                    except Exception as e:
                        print(f"⚠️ Failed to update streaming conversation stack: {e}")
                        
            except Exception as e:
                print(f"⚠️ Failed to update streaming session context: {e}")
        
        return {"response": ai_response, "session_id": session_id}
        
    except Exception as e:
        print(f"❌ Error in streaming endpoint: {e}")
        return {"response": "Sorry, I encountered an error. Please try again.", "session_id": session_id}

async def get_context_aware_gpt_response(user_input: str, session_id: str, user_ip: Optional[str] = None, 
                                       conversation_stack=None) -> Optional[str]:
    """Generate AI response with context-aware filtering for better personalization"""
    
    # Get the base AI response
    base_response = await get_gpt_response(user_input, session_id, user_ip)
    
    if not base_response or not context_filtering or not conversation_stack:
        return base_response
    
    try:
        # Determine if this is a query that would benefit from context filtering
        intent = 'general_travel_info'
        if any(word in user_input.lower() for word in ['restaurant', 'food', 'eat', 'dining', 'cafe']):
            intent = 'restaurant_search'
        elif any(word in user_input.lower() for word in ['museum', 'gallery', 'history']):
            intent = 'museum_inquiry'
        elif any(word in user_input.lower() for word in ['place', 'attraction', 'visit', 'see']):
            intent = 'place_recommendation'
        
        # Only apply context filtering for relevant queries
        if intent in ['restaurant_search', 'place_recommendation', 'museum_inquiry']:
            
            # Get relevant data from database
            from database import SessionLocal
            db = SessionLocal()
            
            try:
                if intent == 'restaurant_search':
                    # Get restaurants from database
                    restaurants = db.query(Restaurant).limit(50).all()
                    restaurant_data = []
                    for r in restaurants:
                        restaurant_data.append({
                            'name': r.name,
                            'description': r.description or '',
                            'district': r.district or '',
                            'cuisine': r.cuisine or '',
                            'rating': getattr(r, 'rating', 4.0),
                            'address': getattr(r, 'address', ''),
                        })
                    
                    # Apply context-aware filtering
                    filtered_restaurants = context_filtering.apply_context_filtering(
                        restaurant_data, conversation_stack, user_input, intent
                    )
                    
                    # If we have filtered results, enhance the response
                    if filtered_restaurants and len(filtered_restaurants) < len(restaurant_data):
                        print(f"✅ Context filtering applied: {len(restaurant_data)} → {len(filtered_restaurants)} restaurants")
                        
                        # Create enhanced response with top filtered results
                        enhanced_response = create_enhanced_restaurant_response(
                            base_response, filtered_restaurants[:10], conversation_stack, user_input
                        )
                        
                        if enhanced_response:
                            return enhanced_response
                
                elif intent == 'place_recommendation':
                    # Get places from database
                    places = db.query(Place).limit(50).all()
                    place_data = []
                    for p in places:
                        place_data.append({
                            'name': p.name,
                            'description': p.description or '',
                            'district': p.district or '',
                            'category': p.category or '',
                            'rating': getattr(p, 'rating', 4.0),
                        })
                    
                    # Apply context-aware filtering (reuse restaurant logic for now)
                    filtered_places = context_filtering.apply_context_filtering(
                        place_data, conversation_stack, user_input, intent
                    )
                    
                    if filtered_places and len(filtered_places) < len(place_data):
                        print(f"✅ Context filtering applied: {len(place_data)} → {len(filtered_places)} places")
                        
                        enhanced_response = create_enhanced_place_response(
                            base_response, filtered_places[:10], conversation_stack, user_input
                        )
                        
                        if enhanced_response:
                            return enhanced_response
                            
            finally:
                db.close()
                
    except Exception as e:
        print(f"⚠️ Context-aware filtering error: {e}")
        # Return base response if filtering fails
    
    return base_response

def create_enhanced_restaurant_response(base_response: str, filtered_restaurants: List[Dict[str, Any]], 
                                      conversation_stack, user_input: str) -> Optional[str]:
    """Create enhanced response with context-aware restaurant recommendations"""
    
    if not filtered_restaurants:
        return None
    
    # Check if this looks like a restaurant recommendation response
    if not any(word in base_response.lower() for word in ['restaurant', 'cafe', 'food', 'dining', 'eat']):
        return None
    
    # Extract location context for personalized intro
    location_intro = ""
    if conversation_stack and conversation_stack.turns:
        # Look for location mentions in recent conversation
        for turn in conversation_stack.turns[-3:]:
            query_lower = turn.user_query.lower()
            for district in ['sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'taksim']:
                if district in query_lower:
                    location_intro = f"Based on your interest in {district.title()}, "
                    break
            if location_intro:
                break
    
    # Create enhanced response
    enhanced_lines = []
    
    # Add personalized intro if we have context
    if location_intro:
        enhanced_lines.append(f"{location_intro}here are the most suitable restaurants for you:")
    else:
        enhanced_lines.append("Here are restaurants that match your preferences:")
    
    enhanced_lines.append("")
    
    # Add top filtered restaurants with context-aware descriptions
    for i, restaurant in enumerate(filtered_restaurants[:8], 1):
        name = restaurant.get('name', 'Unknown Restaurant')
        district = restaurant.get('district', '')
        cuisine = restaurant.get('cuisine', '')
        rating = restaurant.get('rating', 0)
        
        # Format rating
        rating_str = f"{rating:.1f}/5.0" if isinstance(rating, (int, float)) and rating > 0 else "Great reviews"
        
        # Create description with context
        description_parts = []
        if district:
            description_parts.append(f"in {district}")
        if cuisine:
            description_parts.append(f"serving {cuisine} cuisine")
        
        description = f"({', '.join(description_parts)})" if description_parts else ""
        
        enhanced_lines.append(f"{i}. **{name}** - {rating_str} {description}")
        
        # Add brief context if available
        if restaurant.get('description'):
            brief_desc = restaurant['description'][:100]
            if len(restaurant['description']) > 100:
                brief_desc += "..."
            enhanced_lines.append(f"   {brief_desc}")
        
        enhanced_lines.append("")
    
    # Add context-aware tips
    enhanced_lines.append("💡 **Personalized Tips:**")
    
    # Add location-specific tip
    if location_intro:
        location_name = location_intro.split("in ")[-1].split(",")[0].strip()
        enhanced_lines.append(f"• These restaurants are selected based on your interest in {location_name}")
    
    # Add preference-based tips
    user_query_lower = user_input.lower()
    if 'cheap' in user_query_lower or 'budget' in user_query_lower:
        enhanced_lines.append("• Filtered for budget-friendly options")
    elif 'expensive' in user_query_lower or 'fine dining' in user_query_lower:
        enhanced_lines.append("• Selected upscale dining experiences")
    
    if 'authentic' in user_query_lower or 'traditional' in user_query_lower:
        enhanced_lines.append("• Prioritized authentic and traditional restaurants")
    
    enhanced_lines.append("")
    enhanced_lines.append("Would you like more details about any of these restaurants or help with directions?")
    
    return "\n".join(enhanced_lines)

def create_enhanced_place_response(base_response: str, filtered_places: List[Dict[str, Any]], 
                                 conversation_stack, user_input: str) -> Optional[str]:
    """Create enhanced response with context-aware place recommendations"""
    
    if not filtered_places:
        return None
    
    # Check if this looks like a place recommendation response
    if not any(word in base_response.lower() for word in ['place', 'attraction', 'visit', 'see', 'museum']):
        return None
    
    # Create enhanced response similar to restaurant response
    enhanced_lines = []
    enhanced_lines.append("Here are places that match your interests:")
    enhanced_lines.append("")
    
    for i, place in enumerate(filtered_places[:8], 1):
        name = place.get('name', 'Unknown Place')
        district = place.get('district', '')
        category = place.get('category', '')
        rating = place.get('rating', 0)
        
        rating_str = f"{rating:.1f}/5.0" if isinstance(rating, (int, float)) and rating > 0 else "Highly rated"
        
        description_parts = []
        if district:
            description_parts.append(f"in {district}")
        if category:
            description_parts.append(category.lower())
        
        description = f"({', '.join(description_parts)})" if description_parts else ""
        
        enhanced_lines.append(f"{i}. **{name}** - {rating_str} {description}")
        
        if place.get('description'):
            brief_desc = place['description'][:100]
            if len(place['description']) > 100:
                brief_desc += "..."
            enhanced_lines.append(f"   {brief_desc}")
        
        enhanced_lines.append("")
    
    enhanced_lines.append("Would you like more information about any of these places?")
    
    return "\n".join(enhanced_lines)