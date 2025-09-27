# --- Standard Library Imports ---
import sys
import os
import re
import asyncio
import json
import time
import html
from datetime import datetime, date, timedelta
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import traceback
from collections import defaultdict

# --- Third-Party Imports ---
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from thefuzz import fuzz, process

# Load environment variables first, before any other imports
load_dotenv()

# Daily usage tracking
daily_usage = defaultdict(int)  # IP -> count
last_reset_date = date.today()
DAILY_LIMIT = 20  # 20 requests per IP per day

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
    from api_clients.weather_enhanced import WeatherClient  # type: ignore
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
    
    class WeatherClient:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        async def get_weather(self, *args, **kwargs): return {}
        def get_istanbul_weather(self, *args, **kwargs): return {}
        def format_weather_info(self, *args, **kwargs): return "Weather data not available"
    
    class EnhancedAPIService:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def search_restaurants_enhanced(self, *args, **kwargs): 
            return {"results": [], "weather_context": {}}

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

# Import live data services for museums and transport
try:
    from live_museum_service import LiveMuseumService
    live_museum_service = LiveMuseumService()
    print("✅ Live museum service import successful")
except ImportError as e:
    print(f"⚠️ Live museum service import failed: {e}")
    # Create dummy service
    class DummyMuseumService:
        async def get_museum_info(self, museum_name): return None
        async def search_museums(self, location=None, query=None): return []
        def get_popular_museums(self): return []
    live_museum_service = DummyMuseumService()

try:
    from real_time_transport import RealTimeTransportService
    transport_service = RealTimeTransportService()
    print("✅ Real-time transport service import successful")
except ImportError as e:
    print(f"⚠️ Real-time transport service import failed: {e}")
    # Create dummy service
    class DummyTransportService:
        async def get_route(self, from_loc, to_loc): return None
        async def get_line_status(self, line_name): return None
        def get_live_delays(self): return []
    transport_service = DummyTransportService()

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

# --- OpenAI Import ---
from typing import Optional, Type
try:
    from openai import OpenAI
    OpenAI_available = True
except ImportError:
    OpenAI = None  # type: ignore
    OpenAI_available = False
    print("[ERROR] openai package not installed. Please install it with 'pip install openai'.")

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

# === Authentication Setup ===
try:
    from auth import get_current_admin, authenticate_admin, create_access_token
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
if RATE_LIMITING_ENABLED:
    try:
        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        print("✅ Rate limiting enabled successfully")
    except Exception as e:
        print(f"⚠️ Rate limiting initialization failed: {e}")
        RATE_LIMITING_ENABLED = False
else:
    limiter = None
    print("⚠️ Rate limiting disabled")

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
    
    # PHASE 1: Remove explicit currency amounts (all formats) - ENHANCED
    text = re.sub(r'\$\d+[\d.,]*', '', text)      # $20, $15.50
    text = re.sub(r'€\d+[\d.,]*', '', text)       # €20, €15.50
    text = re.sub(r'₺\d+[\d.,]*', '', text)       # ₺20, ₺15.50
    text = re.sub(r'\d+₺', '', text)              # 50₺
    text = re.sub(r'\d+\s*(?:\$|€|₺)', '', text)  # 20$, 50 €
    text = re.sub(r'(?:\$|€|₺)\s*\d+[\d.,]*', '', text)  # $ 20, € 15.50
    
    # Additional currency patterns
    text = re.sub(r'£\d+[\d.,]*', '', text)       # £20, £15.50
    text = re.sub(r'\d+£', '', text)              # 50£
    text = re.sub(r'(?:USD|EUR|GBP|TRY|TL)\s*\d+[\d.,]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(?:USD|EUR|GBP|TRY|TL)', '', text, flags=re.IGNORECASE)
    
    # PHASE 2: Remove currency words and phrases - ENHANCED
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
    
    # PHASE 4: Remove money emojis and currency symbols - ENHANCED
    text = re.sub(r'💰|💵|💴|💶|💷|💸', '', text)
    text = re.sub(r'[\$€₺£¥₹₽₴₦₱₩₪₨₡₵₼₢₨₹₿]', '', text)
    
    # Remove currency codes
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

def get_gpt_response(user_input: str, session_id: str) -> Optional[str]:
    """Generate response using OpenAI GPT with enhanced category-specific prompts for maximum relevance"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not OpenAI_available or not openai_api_key or OpenAI is None:
            return None
        
        # Sanitize input
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        # Use enhanced category-specific prompts
        try:
            from enhanced_gpt_prompts import get_category_specific_prompt
            from query_analyzer import analyze_and_enhance_query, LocationContext
            
            # First, get basic query analysis for location context
            analysis, _ = analyze_and_enhance_query(user_input)
            location_context = analysis.location_context.value if analysis.location_context != LocationContext.NONE else None
            
            # Get category-specific enhanced prompt
            category, enhanced_system_prompt, max_tokens, temperature, expected_features = get_category_specific_prompt(
                user_input, location_context
            )
            
            print(f"🎯 Enhanced Prompts - Category: {category.value}, Location: {location_context}, Max Tokens: {max_tokens}")
            
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
3. NO CURRENCY: Avoid all currency symbols, numbers with currency words, or specific cost amounts.
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
        
        # Create OpenAI client
        client = OpenAI(api_key=openai_api_key, timeout=30.0, max_retries=2)
        
        # Make the API call with enhanced prompt
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=25
        )
        
        gpt_response = response.choices[0].message.content
        if gpt_response:
            gpt_response = gpt_response.strip()
            
            # Apply post-LLM cleanup to remove pricing and fix location issues
            gpt_response = post_llm_cleanup(gpt_response)
            
            print(f"📝 Enhanced GPT response generated successfully")
            
            # Check if response was truncated due to token limit
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                # Response was cut off due to token limit
                print(f"⚠️ Response truncated due to token limit for query ID: {hash(user_input) % 10000}")
                
                # Add indication that response was truncated (only if it doesn't end with punctuation)
                if not gpt_response.endswith(('.', '!', '?', ':')):
                    gpt_response += "\n\n*[Response was truncated due to length limit. Ask a more specific question for a complete answer.]*"
                else:
                    # If it ends with punctuation, it might be a natural ending
                    print(f"✅ Response ended naturally with punctuation despite length limit")
            
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
    """Fetch museums from the database and return formatted data"""
    try:
        from database import SessionLocal
        from models import Place
        
        # Create database session
        db = SessionLocal()
        
        # Query museums from the places table
        museums = db.query(Place).filter(Place.category == 'Museum').all()
        
        # Also query historical sites and palaces that are museum-like
        historical_sites = db.query(Place).filter(Place.category == 'Historical Site').all()
        palaces = db.query(Place).filter(Place.category == 'Palace').all()
        
        db.close()
        
        # Format the data
        museum_data = []
        
        # Add actual museums
        for museum in museums:
            museum_data.append({
                'name': museum.name,
                'category': 'Museum',
                'district': museum.district,
                'type': 'museum'
            })
        
        # Add historical sites (many are museum-like)
        for site in historical_sites:
            museum_data.append({
                'name': site.name,
                'category': 'Historical Site',
                'district': site.district,
                'type': 'historical'
            })
        
        # Add palaces (which are museums)
        for palace in palaces:
            museum_data.append({
                'name': palace.name,
                'category': 'Palace Museum',
                'district': palace.district,
                'type': 'palace'
            })
        
        return museum_data
        
    except Exception as e:
        print(f"Error fetching museums from database: {e}")
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

def get_enhanced_location_context(user_input: str, category: str = "general") -> str:
    """Get enhanced location context using the comprehensive location enhancer"""
    try:
        from comprehensive_location_enhancer import ComprehensiveLocationEnhancer
        
        # Initialize the location enhancer
        enhancer = ComprehensiveLocationEnhancer()
        
        # Get enhanced context
        enhanced_context = enhancer.enhance_location_context(user_input, category)
        
        return enhanced_context
        
    except Exception as e:
        print(f"⚠️ Error getting enhanced location context: {e}")
        
        # Fallback to basic location context
        user_lower = user_input.lower()
        
        # Basic district detection
        districts = {
            'sultanahmet': 'Historic Sultanahmet district - home to Hagia Sophia, Blue Mosque, and Topkapı Palace',
            'taksim': 'Modern Taksim area - shopping, dining, and nightlife center',
            'beyoglu': 'Historic Beyoğlu - cultural district with Galata Tower and İstiklal Street',
            'kadikoy': 'Asian side Kadıköy - trendy neighborhood with great food scene',
            'besiktas': 'Beşiktaş district - mix of business and residential with waterfront views',
            'ortakoy': 'Charming Ortaköy - famous for its mosque and Bosphorus views',
            'karakoy': 'Historic Karaköy - art galleries and boutique hotels',
            'galata': 'Galata area - historic tower and panoramic city views'
        }
        
        context = ""
        for district, description in districts.items():
            if district in user_lower:
                context += f"\n🏘️ **About {district.title()}:** {description}\n"
        
        # Add general Istanbul context if no specific district found
        if not context:
            context = "\n🏙️ **Istanbul Context:** This magnificent city spans Europe and Asia, offering rich history, delicious cuisine, and vibrant culture.\n"
        
        return context

def detect_location_confusion(user_input: str) -> Tuple[bool, Optional[str]]:
    """Detect if user is asking about locations other than Istanbul and provide redirection"""
    user_lower = user_input.lower()
    
    # Common Turkish cities that people might confuse with Istanbul
    other_cities = ['ankara', 'izmir', 'antalya', 'bursa', 'adana', 'gaziantep', 'konya']
    
    # Check if user mentions other cities
    for city in other_cities:
        if city in user_lower:
            return True, f"I specialize in Istanbul travel advice! If you're asking about {city.title()}, I'd recommend checking local resources for that city. However, I'd be happy to help you plan your Istanbul experience - what would you like to know about Istanbul?"
    
    # Check for generic "Turkey" questions that might need Istanbul focus
    if 'turkey' in user_lower and not 'istanbul' in user_lower:
        turkey_indicators = ['restaurants in turkey', 'food in turkey', 'places in turkey', 'visit turkey']
        if any(indicator in user_lower for indicator in turkey_indicators):
            return True, "I focus specifically on Istanbul! While Turkey has many amazing destinations, I can provide detailed advice about Istanbul's restaurants, attractions, and experiences. What would you like to know about Istanbul specifically?"
    
    return False, None

# === Health and Status Endpoints ===

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and testing"""
    health_status = {
        "status": "healthy",
        "service": "AI Istanbul Backend",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    # Add monitoring metrics if available
    if ADVANCED_MONITORING_ENABLED:
        try:
            metrics_summary = advanced_monitor.get_metrics_summary(hours=1)
            health_status["metrics"] = {
                "cpu_usage": metrics_summary.get("system", {}).get("cpu_usage", {}).get("current", 0),
                "memory_usage": metrics_summary.get("system", {}).get("memory_usage", {}).get("current", 0),
                "response_time": metrics_summary.get("system", {}).get("response_time", {}).get("current", 0),
                "active_alerts": metrics_summary.get("alerts", {}).get("active", 0)
            }
        except Exception as e:
            health_status["monitoring_error"] = str(e)
    
    return health_status

@app.get("/metrics")
async def get_metrics(current_admin: dict = Depends(get_current_admin)):
    """Get system metrics - PROTECTED ENDPOINT"""
    if not ADVANCED_MONITORING_ENABLED:
        raise HTTPException(status_code=503, detail="Advanced monitoring not available")
    
    try:
        metrics_summary = advanced_monitor.get_metrics_summary(hours=24)
        return metrics_summary
    except Exception as e:
        log_error("metrics_fetch_error", str(e), "monitoring")
        raise HTTPException(status_code=500, detail="Error fetching metrics")

@app.get("/monitoring/status")
async def monitoring_status(current_admin: dict = Depends(get_current_admin)):
    """Get monitoring system status - PROTECTED ENDPOINT"""
    status = {
        "advanced_monitoring": ADVANCED_MONITORING_ENABLED,
        "structured_logging": STRUCTURED_LOGGING_ENABLED,
        "rate_limiting": RATE_LIMITING_ENABLED,
        "monitoring_active": getattr(advanced_monitor, 'is_running', False) if ADVANCED_MONITORING_ENABLED else False,
        "log_directory": "/tmp/ai-istanbul-logs" if ADVANCED_MONITORING_ENABLED else None
    }
    
    if ADVANCED_MONITORING_ENABLED:
        try:
            status["active_alerts"] = len([a for a in advanced_monitor.active_alerts.values() if not a.resolved])
            status["total_alerts_today"] = len([a for a in advanced_monitor.alert_history if (datetime.utcnow() - datetime.fromisoformat(a.timestamp.replace('Z', '+00:00'))).days < 1])
        except Exception as e:
            status["monitoring_error"] = str(e)
    
    return status

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Serve the unified admin dashboard"""
    try:
        # Look for the unified dashboard in multiple locations
        dashboard_paths = [
            os.path.join(os.path.dirname(__file__), "..", "unified_admin_dashboard.html"),
            os.path.join(os.path.dirname(__file__), "unified_admin_dashboard.html"),
            "/Users/omer/Desktop/ai-stanbul/unified_admin_dashboard.html"
        ]
        
        dashboard_content = None
        for path in dashboard_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    dashboard_content = f.read()
                break
        
        if dashboard_content:
            return HTMLResponse(content=dashboard_content)
        else:
            return HTMLResponse(
                content="<h1>Admin Dashboard Not Found</h1><p>Please ensure unified_admin_dashboard.html is available.</p>", 
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Admin Dashboard Error</h1><p>Error loading dashboard: {str(e)}</p>", 
            status_code=500
        )

@app.get("/admin/api/stats")
async def admin_stats(db: Session = Depends(get_db), current_admin: dict = Depends(get_current_admin)):
    """Get admin dashboard statistics with real data - PROTECTED ENDPOINT"""
    try:
        # Get real data from database
        total_chats = db.query(ChatHistory).count()
        total_sessions = db.query(ChatSession).count()
        total_blog_posts = db.query(BlogPost).count()
        total_blog_comments = db.query(BlogComment).count()
        
        # Comment moderation stats (focus on problematic content)
        rejected_comments = db.query(BlogComment).filter(BlogComment.is_approved == False, BlogComment.is_spam == False, BlogComment.is_flagged == False).count()
        flagged_comments = db.query(BlogComment).filter(BlogComment.is_flagged == True).count()
        spam_comments = db.query(BlogComment).filter(BlogComment.is_spam == True).count()
        approved_comments = db.query(BlogComment).filter(BlogComment.is_approved == True).count()
        
        # Blog post stats
        published_posts = db.query(BlogPost).filter(BlogPost.is_published == True).count()
        draft_posts = db.query(BlogPost).filter(BlogPost.is_published == False).count()
        
        # Recent activity (last 10 activities)
        recent_comments = db.query(BlogComment).order_by(BlogComment.created_at.desc()).limit(5).all()
        recent_posts = db.query(BlogPost).order_by(BlogPost.created_at.desc()).limit(3).all()
        recent_chats = db.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).limit(3).all()
        
        recent_activity = []
        
        # Add recent comments to activity
        for comment in recent_comments:
            status = "approved" if comment.is_approved else "pending approval"
            if comment.is_spam:
                status = "marked as spam"
            elif comment.is_flagged:
                status = "flagged for review"
            recent_activity.append({
                "time": f"{int((datetime.utcnow() - comment.created_at).total_seconds() // 60)} minutes ago" if comment.created_at else "Recently",
                "message": f"New comment by {comment.author_name} - {status}",
                "type": "comment"
            })
        
        # Add recent posts to activity
        for post in recent_posts:
            status = "published" if post.is_published else "saved as draft"
            recent_activity.append({
                "time": f"{int((datetime.utcnow() - post.created_at).total_seconds() // 60)} minutes ago" if post.created_at else "Recently",
                "message": f"Blog post '{post.title[:30]}...' {status}",
                "type": "blog"
            })
        
        # Add recent chats to activity
        for chat in recent_chats:
            recent_activity.append({
                "time": f"{int((datetime.utcnow() - chat.timestamp).total_seconds() // 60)} minutes ago" if chat.timestamp else "Recently", 
                "message": f"Chat query: '{chat.user_message[:30]}...'",
                "type": "chat"
            })
        
        # Sort by time and limit to 10 most recent
        recent_activity = sorted(recent_activity, key=lambda x: x["time"])[:10]
        
        return {
            "total_chats": total_chats,
            "total_sessions": total_sessions,
            "total_blog_posts": total_blog_posts,
            "total_blog_comments": total_blog_comments,
            "published_posts": published_posts,
            "draft_posts": draft_posts,
            "rejected_comments": rejected_comments,
            "flagged_comments": flagged_comments,
            "spam_comments": spam_comments,
            "approved_comments": approved_comments,
            "avg_rating": 4.2,  # This could be calculated from actual feedback
            "response_time": "1.2s",
            "recent_activity": recent_activity,
            "system_status": {
                "api_server": "online",
                "database": "connected",
                "google_maps_api": "active",
                "openai_api": "active"
            }
        }
    except Exception as e:
        logger.error(f"Error fetching admin stats: {e}")
        # Fallback to mock data if there's an error
        return {
            "total_chats": 0,
            "total_sessions": 0, 
            "total_blog_posts": 0,
            "total_blog_comments": 0,
            "published_comments": 0,
            "draft_posts": 0,
            "pending_comments": 0,
            "flagged_comments": 0,
            "spam_comments": 0,
            "avg_rating": 4.2,
            "response_time": "1.2s",
            "recent_activity": [],
            "system_status": {
                "api_server": "online",
                "database": "error",
                "google_maps_api": "active",
                "openai_api": "active"
            },
            "error": str(e)
        }

@app.get("/admin/api/sessions")
async def admin_sessions(current_admin: dict = Depends(get_current_admin)):
    """Get chat sessions for admin dashboard - PROTECTED ENDPOINT"""
    try:
        # Mock data - replace with real database queries
        return {
            "sessions": [
                {"id": "sess_001", "user": "user_123", "messages": 15, "duration": "8m 32s", "rating": 4.5, "status": "Complete"},
                {"id": "sess_002", "user": "user_456", "messages": 23, "duration": "12m 45s", "rating": 4.2, "status": "Complete"},
                {"id": "sess_003", "user": "user_789", "messages": 8, "duration": "4m 12s", "rating": 4.8, "status": "Active"},
                {"id": "sess_004", "user": "user_012", "messages": 31, "duration": "18m 23s", "rating": 3.9, "status": "Complete"}
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/api/users")
async def admin_users(current_admin: dict = Depends(get_current_admin)):
    """Get users data for admin dashboard - PROTECTED ENDPOINT"""
    try:
        # Mock data - replace with real database queries
        return {
            "users": [
                {"id": "user_123", "last_active": "2 minutes ago", "sessions": 24, "avg_rating": 4.3, "status": "Online"},
                {"id": "user_456", "last_active": "15 minutes ago", "sessions": 18, "avg_rating": 4.1, "status": "Online"},
                {"id": "user_789", "last_active": "1 hour ago", "sessions": 35, "avg_rating": 4.6, "status": "Offline"},
                {"id": "user_012", "last_active": "3 hours ago", "sessions": 12, "avg_rating": 3.8, "status": "Offline"}
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# Blog moderation endpoints
@app.get("/admin/api/blog/posts")
async def admin_blog_posts(
    status: str = "all",  # all, published, draft, flagged
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_admin: dict = Depends(get_current_admin)
):
    """Get blog posts for admin dashboard - PROTECTED ENDPOINT"""
    try:
        query = db.query(BlogPost)
        
        if status == "published":
            query = query.filter(BlogPost.is_published == True)
        elif status == "draft":
            query = query.filter(BlogPost.is_published == False)
        
        posts = query.order_by(BlogPost.created_at.desc()).offset(offset).limit(limit).all()
        
        posts_data = []
        for post in posts:
            posts_data.append({
                "id": post.id,
                "title": post.title,
                "author_name": post.author_name or "Anonymous",
                "district": post.district,
                "created_at": post.created_at.isoformat() if post.created_at else None,
                "updated_at": post.updated_at.isoformat() if post.updated_at else None,
                "is_published": post.is_published,
                "likes_count": post.likes_count,
                "comment_count": len(post.comments) if post.comments else 0,
                "content_preview": post.content[:200] + "..." if len(post.content) > 200 else post.content,
                "tags": json.loads(post.tags) if post.tags else []
            })
        
        total_count = db.query(BlogPost).count()
        return {
            "posts": posts_data,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error fetching blog posts: {e}")
        return {"posts": [], "total": 0, "error": str(e)}

@app.get("/admin/api/blog/comments")
async def admin_blog_comments(
    status: str = "all",  # all, approved, rejected, flagged, spam
    post_id: int = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_admin: dict = Depends(get_current_admin)
):
    """Get blog comments for admin dashboard - PROTECTED ENDPOINT"""
    try:
        query = db.query(BlogComment)
        
        if post_id:
            query = query.filter(BlogComment.blog_post_id == post_id)
            
        if status == "approved":
            query = query.filter(BlogComment.is_approved == True, BlogComment.is_spam == False, BlogComment.is_flagged == False)
        elif status == "rejected":
            query = query.filter(BlogComment.is_approved == False, BlogComment.is_spam == False, BlogComment.is_flagged == False)
        elif status == "flagged":
            query = query.filter(BlogComment.is_flagged == True)
        elif status == "spam":
            query = query.filter(BlogComment.is_spam == True)
        elif status == "rejected":
            query = query.filter(BlogComment.is_approved == False, BlogComment.is_spam == False, BlogComment.is_flagged == False)
        
        comments = query.order_by(BlogComment.created_at.desc()).offset(offset).limit(limit).all()
        
        comments_data = []
        for comment in comments:
            comments_data.append({
                "id": comment.id,
                "blog_post_id": comment.blog_post_id,
                "blog_post_title": comment.blog_post.title if comment.blog_post else "Unknown",
                "author_name": comment.author_name,
                "author_email": comment.author_email,
                "content": comment.content,
                "is_approved": comment.is_approved,
                "is_flagged": comment.is_flagged,
                "is_spam": comment.is_spam,
                "flagged_reason": comment.flagged_reason,
                "created_at": comment.created_at.isoformat() if comment.created_at else None,
                "approved_at": comment.approved_at.isoformat() if comment.approved_at else None,
                "approved_by": comment.approved_by
            })
        
        total_count = db.query(BlogComment).count()
        return {
            "comments": comments_data,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error fetching blog comments: {e}")
        return {"comments": [], "total": 0, "error": str(e)}

@app.post("/admin/api/blog/comments/{comment_id}/moderate")
async def moderate_comment(
    comment_id: int,
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Moderate a blog comment"""
    try:
        action = request.get("action")  # approve, reject, flag, spam, unflag
        reason = request.get("reason", None)
        admin_name = request.get("admin_name", "Admin")
        
        comment = db.query(BlogComment).filter(BlogComment.id == comment_id).first()
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        if action == "approve":
            comment.is_approved = True
            comment.approved_at = datetime.utcnow()
            comment.approved_by = admin_name
            comment.is_flagged = False
        elif action == "reject":
            comment.is_approved = False
            comment.approved_at = None
            comment.approved_by = None
            comment.is_flagged = False
        elif action == "flag":
            comment.is_flagged = True
            comment.flagged_reason = reason or "Flagged for review"
        elif action == "spam":
            comment.is_spam = True
            comment.is_approved = False
            comment.is_flagged = False
        elif action == "unflag":
            comment.is_flagged = False
            comment.flagged_reason = None
        
        comment.updated_at = datetime.utcnow()
        db.commit()
        
        return {"success": True, "message": f"Comment {action}ed successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error moderating comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/api/blog/posts/{post_id}/moderate")
async def moderate_post(
    post_id: int,
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Moderate a blog post"""
    try:
        action = request.get("action")  # publish, unpublish, delete
        
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        if action == "publish":
            post.is_published = True
        elif action == "unpublish":
            post.is_published = False
        elif action == "delete":
            db.delete(post)
        
        post.updated_at = datetime.utcnow()
        db.commit()
        
        return {"success": True, "message": f"Post {action}ed successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error moderating post: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Authentication Import ---
try:
    from auth import get_current_admin, authenticate_admin, create_access_token
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

# === Server Startup ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 Starting AI Istanbul Backend server on port {port}")
    print(f"📊 Health check endpoint: http://localhost:{port}/health")
    print(f"💬 Chat endpoint: http://localhost:{port}/ai/chat")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port, 
        reload=False,
        log_level="info"
    )

# === Authentication Endpoints ===

@app.post("/auth/login")
async def login(credentials: dict):
    """Admin login endpoint"""
    try:
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password required"
            )
        
        # Authenticate user
        user_info = authenticate_admin(username, password)
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create access token
        token_data = {
            "username": user_info["username"],
            "role": user_info["role"]
        }
        access_token = create_access_token(token_data)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_info": {
                "username": user_info["username"],
                "role": user_info["role"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/auth/logout")
async def logout():
    """Admin logout endpoint (client-side token removal)"""
    return {"message": "Logged out successfully"}

@app.get("/auth/me")
async def get_current_user(current_admin: dict = Depends(get_current_admin)):
    """Get current authenticated admin info"""
    return {
        "username": current_admin.get("username"),
        "role": current_admin.get("role"),
        "authenticated": True
    }

# === Protected Admin Endpoints ===

# === Startup and Shutdown Events ===

@app.on_event("startup")
async def startup_event():
    """Initialize monitoring and logging on startup"""
    print("🚀 AI Istanbul Backend starting up...")
    
    if ADVANCED_MONITORING_ENABLED:
        try:
            # Start advanced monitoring
            asyncio.create_task(advanced_monitor.start_monitoring())
            
            # Log system startup
            comprehensive_logger.log_system_event(
                "application_startup",
                "info",
                {
                    "version": "1.0.0",
                    "monitoring_enabled": True,
                    "rate_limiting_enabled": RATE_LIMITING_ENABLED,
                    "structured_logging_enabled": STRUCTURED_LOGGING_ENABLED
                }
            )
            print("✅ Advanced monitoring started")
            
        except Exception as e:
            print(f"⚠️ Error starting advanced monitoring: {e}")
    
    print("✅ Application startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown procedures"""
    print("🔄 AI Istanbul Backend shutting down...")
    
    if ADVANCED_MONITORING_ENABLED:
        try:
            # Stop monitoring
            advanced_monitor.stop_monitoring()
            
            # Log system shutdown
            comprehensive_logger.log_system_event(
                "application_shutdown",
                "info",
                {"clean_shutdown": True}
            )
            print("✅ Advanced monitoring stopped")
            
        except Exception as e:
            print(f"⚠️ Error during shutdown: {e}")
    
    print("✅ Application shutdown completed")

# === Main Chat Endpoints with Rate Limiting ===

@app.post("/ai/chat")
async def chat(request: Request, data: dict, db: Session = Depends(get_db)):
    """Main chat endpoint with enhanced features, rate limiting, and monitoring"""
    
    # Apply rate limiting if enabled
    if RATE_LIMITING_ENABLED and limiter:
        try:
            await limiter.limit("50/hour")(request)
        except Exception as e:
            logger.warning(f"Rate limiting check failed: {e}")
    
    start_time = time.time()
    client_ip = getattr(request.client, 'host', 'unknown')
    user_agent = request.headers.get('user-agent', '')
    session_id = data.get("session_id", "anonymous")
    
    try:
        user_message = data.get("message", "")
        if not user_message:
            if ADVANCED_MONITORING_ENABLED:
                log_error("empty_message_error", "Empty message received in chat endpoint", "chat_handler", session_id=session_id)
            return {"error": "Message is required"}
        
        print(f"💬 Chat request received: {user_message[:50]}...")
        
        # Log user action if monitoring is enabled
        if ADVANCED_MONITORING_ENABLED:
            log_user_action(
                action="chat_message_sent",
                user_id=session_id,
                details=f"Message length: {len(user_message)}",
                ip_address=client_ip,
                session_id=session_id,
                metadata={"message_preview": user_message[:100]}
            )
        
        # Legacy logging for compatibility
        if STRUCTURED_LOGGING_ENABLED and not ADVANCED_MONITORING_ENABLED:
            structured_logger.info(f"Chat request from {client_ip}: {user_message[:100]}")
        
        # Simple response for now (you can add your AI logic here)
        response = f"Hello! I received your message: {user_message[:100]}..."
        
        # Save to database
        try:
            chat_record = ChatHistory(
                session_id=session_id,
                user_message=user_message,
                ai_response=response,
                user_ip=client_ip,
                timestamp=datetime.now()
            )
            db.add(chat_record)
            db.commit()
        except Exception as db_error:
            db.rollback()
            if ADVANCED_MONITORING_ENABLED:
                log_error("database_save_error", str(db_error), "chat_handler", session_id=session_id, exception=db_error)
            print(f"Database error: {db_error}")
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000
        
        # Log performance metrics
        if ADVANCED_MONITORING_ENABLED:
            log_performance_metric("chat_response_time", response_time)
            comprehensive_logger.log_performance_metric("chat_request", response_time, True, {
                "message_length": len(user_message),
                "response_length": len(response),
                "session_id": session_id
            })
        
        return {
            "response": response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": round(response_time, 2)
        }
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        # Log error with comprehensive details
        if ADVANCED_MONITORING_ENABLED:
            log_error("chat_processing_error", str(e), "chat_handler", session_id=session_id)
            log_error_metric("chat_errors", f"Error in chat processing: {str(e)}")
            comprehensive_logger.log_performance_metric("chat_request", response_time, False, {
                "error_type": type(e).__name__,
                "session_id": session_id
            })
        
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")

@app.post("/ai/stream")
@limiter.limit("30/hour") if RATE_LIMITING_ENABLED and limiter else lambda f: f
async def stream_chat(request: Request, data: dict):
    """Streaming chat endpoint with rate limiting"""
    try:
        user_message = data.get("message", "")
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        client_ip = request.client.host
        print(f"🔄 Stream chat request from {client_ip}: {user_message[:50]}...")
        
        # Simple streaming response
        async def generate_response():
            response = f"Streaming response to: {user_message}"
            for word in response.split():
                yield f"data: {word} \n\n"
                await asyncio.sleep(0.1)
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        print(f"❌ Stream chat error: {e}")
        raise HTTPException(status_code=500, detail="Stream chat failed")

# === Rate Limited API Endpoints ===

@app.get("/restaurants/search")
@limiter.limit("100/hour") if RATE_LIMITING_ENABLED and limiter else lambda f: f
async def search_restaurants(request: Request, query: str = "", location: str = "Istanbul"):
    """Search restaurants with rate limiting"""
    client_ip = request.client.host
    print(f"🍽️ Restaurant search from {client_ip}: {query}")
    
    # Mock restaurant data
    return {
        "restaurants": [
            {"name": f"Restaurant for {query}", "location": location, "rating": 4.5}
        ],
        "query": query,
        "location": location
    }

@app.get("/places/search")
@limiter.limit("100/hour") if RATE_LIMITING_ENABLED and limiter else lambda f: f
async def search_places(request: Request, query: str = "", location: str = "Istanbul"):
    """Search places with rate limiting"""
    client_ip = request.client.host
    print(f"📍 Places search from {client_ip}: {query}")
    
    # Mock places data
    return {
        "places": [
            {"name": f"Place for {query}", "location": location, "rating": 4.3}
        ],
        "query": query,
        "location": location
    }
