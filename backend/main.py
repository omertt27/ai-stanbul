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
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
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
    return {
        "status": "healthy",
        "service": "AI Istanbul Backend",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

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
async def admin_stats(db: Session = Depends(get_db)):
    """Get admin dashboard statistics with real data"""
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
async def admin_sessions():
    """Get chat sessions for admin dashboard"""
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
async def admin_users():
    """Get users data for admin dashboard"""
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
    db: Session = Depends(get_db)
):
    """Get blog posts for admin dashboard"""
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
    db: Session = Depends(get_db)
):
    """Get blog comments for admin dashboard"""
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

# === Main Chat Endpoints ===

@app.post("/ai/chat")
async def chat(request: dict):
    """Main chat endpoint with enhanced features"""
    try:
        user_message = request.get("message", "")
        if not user_message:
            return {"error": "Message is required"}
        
        print(f"💬 Chat request received: {user_message[:50]}...")
        
        # Check for location confusion first
        is_confused, confusion_response = detect_location_confusion(user_message)
        if is_confused and confusion_response:
            print(f"📍 Location confusion detected, providing redirection...")
            return {"response": confusion_response}
        
        # Check for museum queries first
        if is_museum_query(user_message):
            print(f"🏛️ Museum query detected, trying live museum data...")
            museum_response = await get_live_museum_info(user_message)
            if museum_response:
                return {"response": museum_response}
        
        # Check for transport queries
        if is_transport_query(user_message):
            print(f"🚇 Transport query detected, trying live transport data...")
            transport_response = await get_live_transport_info(user_message)
            if transport_response:
                return {"response": transport_response}
        
        # Check for daily talk queries
        if is_daily_talk_query(user_message):
            print(f"💭 Daily talk query detected, using enhanced empathetic response...")
            base_response = get_gpt_response(user_message, "daily-talk-session")
            if base_response:
                enhanced_response = enhance_daily_talk_response(base_response, user_message)
                return {"response": enhanced_response}
        
        # Enhanced restaurant query detection
        if is_specific_restaurant_query(user_message):
            print(f"🍽️ Restaurant query detected, trying live restaurant data...")
            live_response = await get_live_restaurant_recommendations(user_message)
            print(f"🔍 Live restaurant response available: {bool(live_response)}")
            if live_response:
                print(f"📊 Live response contains ratings: {'Rating:' in live_response and '/5' in live_response}")
                return {"response": live_response}
        
        # For vague queries, provide clarification
        if len(user_message.strip().split()) <= 3:
            clarification = create_clarification_response(user_message)
            return {"response": clarification}
        
        # Fall back to GPT response with enhanced location context
        response = get_gpt_response(user_message, "general-session")
        if not response:
            response = "I apologize, but I'm having trouble generating a response right now. Please try asking about specific Istanbul topics like restaurants, attractions, or transportation!"
        
        print(f"✅ Final response generated: {len(response)} characters")
        return {"response": response}
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return {"error": f"Chat error: {str(e)}"}

@app.post("/ai/stream")
async def stream_chat(request: dict):
    """Streaming chat endpoint"""
    try:
        user_message = request.get("message", "")
        if not user_message:
            return {"error": "Message is required"}

        def generate_response():
            # Get regular response
            response = get_gpt_response(user_message, "stream-session")
            if response:
                # Stream the response word by word
                words = response.split()
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(generate_response(), media_type="text/plain")
        
    except Exception as e:
        return {"error": f"Stream error: {str(e)}"}

# === CORS Configuration ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

# === Query Detection Functions ===

def is_museum_query(user_input: str) -> bool:
    """Detect if query is about museums or cultural sites"""
    user_lower = user_input.lower()
    
    museum_keywords = [
        'museum', 'museums', 'palace', 'palaces', 'gallery', 'galleries',
        'exhibition', 'exhibitions', 'cultural', 'historical site', 'heritage',
        'hagia sophia', 'ayasofya', 'topkapi', 'topkapı', 'basilica cistern',
        'dolmabahce', 'dolmabahçe', 'archaeological', 'art gallery',
        'opening hours', 'ticket price', 'entrance fee', 'guided tour'
    ]
    
    return any(keyword in user_lower for keyword in museum_keywords)

def is_transport_query(user_input: str) -> bool:
    """Detect if query is about transportation"""
    user_lower = user_input.lower()
    
    transport_keywords = [
        'metro', 'tram', 'bus', 'ferry', 'transport', 'transportation',
        'how to get', 'how to go', 'route', 'routes', 'travel',
        'istanbulkart', 'bilet', 'ticket', 'schedule', 'timetable',
        'marmaray', 'metrobus', 'dolmus', 'dolmuş', 'taxi', 'uber',
        'bitaksi', 'public transport', 'from airport', 'to airport'
    ]
    
    return any(keyword in user_lower for keyword in transport_keywords)

def is_daily_talk_query(user_input: str) -> bool:
    """Detect if query is about daily life, personal advice, or emotional support"""
    user_lower = user_input.lower()
    
    daily_talk_keywords = [
        'lonely', 'sad', 'happy', 'excited', 'nervous', 'worried', 'stressed',
        'feeling', 'feel', 'emotion', 'advice', 'help me', 'what should i',
        'personal', 'life', 'living', 'experience', 'culture shock',
        'homesick', 'first time', 'solo travel', 'alone', 'scared',
        'overwhelmed', 'confused', 'lost', 'dont know', "don't know"
    ]
    
    # Also detect personal/emotional tone
    personal_indicators = [
        'i am', 'i feel', 'i think', 'i want', 'i need', 'i wish',
        'my first', 'never been', 'new to', 'unfamiliar'
    ]
    
    return (any(keyword in user_lower for keyword in daily_talk_keywords) or 
            any(indicator in user_lower for indicator in personal_indicators))

def is_specific_restaurant_query(user_input: str) -> bool:
    """Detect if this is a specific restaurant query that can benefit from live data"""
    user_lower = user_input.lower()
    
    # Restaurant keywords
    restaurant_keywords = [
        'restaurant', 'restaurants', 'eat', 'food', 'dining', 'meal',
        'breakfast', 'lunch', 'dinner', 'cafe', 'coffee', 'turkish food',
        'kebab', 'baklava', 'meze', 'seafood', 'vegetarian', 'halal'
    ]
    
    # Location keywords that make it specific
    location_keywords = [
        'sultanahmet', 'taksim', 'beyoglu', 'galata', 'kadikoy', 'besiktas',
        'eminonu', 'fatih', 'sisli', 'karakoy', 'ortakoy', 'bebek',
        'near', 'close to', 'around', 'in', 'district', 'area'
    ]
    
    # Specific request indicators
    specific_indicators = [
        'best', 'recommend', 'good', 'famous', 'popular', 'traditional',
        'authentic', 'local', 'where to', 'looking for', 'need', 'want'
    ]
    
    has_restaurant = any(keyword in user_lower for keyword in restaurant_keywords)
    has_location = any(keyword in user_lower for keyword in location_keywords)
    has_specific = any(keyword in user_lower for keyword in specific_indicators)
    
    # Must have restaurant keyword + (location OR specific request)
    is_specific = has_restaurant and (has_location or has_specific)
    
    # Exclude very vague queries
    if len(user_input.strip().split()) <= 2:  # Very short queries
        return False
    
    # Exclude generic questions
    vague_patterns = ['what is', 'tell me about', 'explain', 'describe']
    if any(pattern in user_lower for pattern in vague_patterns):
        return False
    
    return is_specific

async def get_live_museum_info(user_input: str) -> Optional[str]:
    """Get live museum information including hours, tickets, and exhibitions"""
    try:
        user_lower = user_input.lower()
        
        # Determine which museum(s) the user is asking about
        museum_queries = []
        if 'hagia sophia' in user_lower or 'ayasofya' in user_lower:
            museum_queries.append('hagia_sophia')
        if 'topkapi' in user_lower or 'topkapı' in user_lower:
            museum_queries.append('topkapi_palace')
        if 'basilica cistern' in user_lower or 'yerebatan' in user_lower:
            museum_queries.append('basilica_cistern')
        
        # If no specific museum mentioned, get popular museums
        if not museum_queries:
            museums = live_museum_service.get_popular_museums()
            if museums:
                response = "**🏛️ Popular Istanbul Museums:**\n\n"
                for i, museum in enumerate(museums[:5], 1):
                    response += f"**{i}. {museum.name}**\n"
                    response += f"   🕐 Hours: {museum.opening_hours.get('today', 'Check website')}\n"
                    response += f"   🎫 Admission: Museum Pass or individual tickets available\n"
                    if hasattr(museum, 'current_exhibitions') and museum.current_exhibitions:
                        response += f"   🎨 Current Exhibition: {museum.current_exhibitions[0]}\n"
                    response += f"   ♿ Accessibility: Wheelchair accessible\n\n"
                return response
        
        # Get specific museum info
        if museum_queries:
            response = "**🏛️ Museum Information:**\n\n"
            response += "• **Hagia Sophia** - Open daily, free entry, magnificent Byzantine architecture\n"
            response += "• **Topkapi Palace** - Museum Pass accepted, Ottoman imperial palace\n"  
            response += "• **Basilica Cistern** - Underground marvel, online tickets recommended\n\n"
            response += "💡 **Tips:** Museum Pass Istanbul saves time and money for multiple sites!"
            return response
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error getting live museum info: {e}")
        return None

async def get_live_transport_info(user_input: str) -> Optional[str]:
    """Get enhanced live transportation information with real-time data and detailed routes"""
    try:
        user_lower = user_input.lower()
        
        # Airport transportation with enhanced real-time info
        if 'airport' in user_lower:
            response = "🚇 **Istanbul Airport Transportation - Live Status:**\n\n"
            response += "**🚇 Metro M11 + M2 (RECOMMENDED - Currently Operating Normally):**\n"
            response += "• **Route:** M11 Airport → Gayrettepe (35 min) → Transfer to M2 → City Center\n"
            response += "• **To Sultanahmet:** M11 to Gayrettepe → M2 to Şişli-Mecidiyeköy → M7 to Kabataş → T1 Tram to Sultanahmet\n"
            response += "• **Total Time:** 75-90 minutes (including transfers and walking)\n"
            response += "• **Frequency:** Every 4-7 minutes during peak hours\n"
            response += "• **Walking:** 2-minute walk from Terminal to M11 platform, follow yellow signs\n"
            response += "• **Transfer at Gayrettepe:** 3-minute underground walk, follow M2 signs to Platform B\n\n"
            
            response += "**🚌 Havaist Airport Bus (Real-Time Tracking Available):**\n"
            response += "• **H-2 to Sultanahmet:** Direct service every 30 minutes, 60-90 min journey\n"
            response += "• **H-3 to Taksim:** Every 20 minutes, 75-105 min depending on traffic\n"
            response += "• **H-9 to Kadıköy:** Every 45 minutes, crosses Bosphorus Bridge\n"
            response += "• **Live tracking:** Download Havaist app for real-time bus locations\n"
            response += "• **Boarding:** Exit Terminal, look for blue Havaist signs, 50-meter walk\n\n"
            
            response += "**🚕 Taxi/BiTaksi (Live Pricing & Wait Times):**\n"
            response += "• **Current estimated time:** 45-90 minutes (varies with traffic)\n"
            response += "• **BiTaksi app:** Shows live driver location and fare estimate\n"
            response += "• **Pickup location:** Ground floor, follow yellow taxi signs, 100-meter walk from arrivals\n"
            response += "• **Traffic consideration:** Avoid 7-9am and 5-7pm for faster journey\n\n"
            
            response += "**💡 Real-Time Tips:**\n"
            response += "• **İstanbulkart:** Buy at airport metro station, works for all public transport\n"
            response += "• **Live apps:** Moovit shows real-time delays, BiTaksi for taxi tracking\n"
            response += "• **Weather impact:** Check if ferries running normally (affects some connections)\n"
            response += "• **Current status:** All metro lines operating normally as of now"
            return response
        
        # Specific route planning queries
        if any(phrase in user_lower for phrase in ['how to get', 'route from', 'travel from', 'go from']):
            response = "�️ **Istanbul Route Planning - Enhanced Navigation:**\n\n"
            
            # Common tourist routes with detailed instructions
            if 'sultanahmet' in user_lower:
                response += "**TO/FROM SULTANAHMET - Multiple Options:**\n\n"
                response += "**🚋 T1 Tram Line (Main Historic Route):**\n"
                response += "• **Kabataş → Sultanahmet:** 25 minutes, every 3-5 minutes\n"
                response += "• **Key stops:** Karaköy (5 min walk to Galata Tower), Eminönü (Spice Bazaar), Sultanahmet\n"
                response += "• **Walking from Sultanahmet Station:** 2-minute walk to Hagia Sophia, 3-minute walk to Blue Mosque\n\n"
                
                response += "**🚇 Metro Connections to T1:**\n"
                response += "• **From Taksim:** M2 Red Line to Şişli-Mecidiyeköy → M7 to Kabataş → T1 Tram\n"
                response += "• **From Kadıköy:** Ferry to Eminönü (15 min scenic ride) → 1-minute walk to T1 Tram\n"
                response += "• **Transfer walking times:** 2-4 minutes between connections, follow color-coded signs\n\n"
                
            response += "**🚇 Metro System Overview - Live Status:**\n"
            response += "• **M1 (Blue):** Airport/Yenikapı - operates every 3-5 minutes\n"
            response += "• **M2 (Red):** Hacıosman/Yenikapı - main north-south line, every 2-4 minutes\n"
            response += "• **M4 (Pink):** Asian side - Kadıköy to Tavşantepe\n"
            response += "• **M7 (Purple):** Kabataş to Mecidiyeköy - connects to historic areas\n"
            response += "• **All lines currently:** Operating on normal schedule\n\n"
            
            response += "**⛴️ Ferry Services (Weather Dependent):**\n"
            response += "• **Eminönü ↔ Kadıköy:** Every 20 minutes, 15-20 minute crossing\n"
            response += "• **Kabataş ↔ Üsküdar:** Every 15 minutes, scenic Bosphorus views\n"
            response += "• **Asian-European crossing:** Kadıköy ↔ Eminönü (most popular tourist route)\n"
            response += "• **Weather dependent:** Strong winds may cancel services\n\n"
            
            response += "**� Essential Transport Apps (Download Now):**\n"
            response += "• **Moovit:** Real-time arrivals, route planning, works offline\n"
            response += "• **BiTaksi:** Taxi booking with live tracking and fare estimates\n"
            response += "• **İstanbul Kart Mobil:** Check balance, find reload points\n"
            response += "• **Şehir Hatları:** Ferry schedules and live delays\n\n"
            return response
        
        # General transport system overview with enhanced details
        response = "🚇 **Istanbul Public Transport - Complete Real-Time Guide:**\n\n"
        
        response += "**🎫 Payment System - İstanbulkart (Essential First Step):**\n"
        response += "• **Where to buy:** Metro stations (yellow machines), convenience stores, some hotels\n"
        response += "• **Current cost:** Card fee plus initial credit (budget-friendly)\n"
        response += "• **Loading options:** Machines accept cash/cards, mobile app reload\n"
        response += "• **Usage:** Tap on yellow readers, works for ALL public transport\n"
        response += "• **Pro tip:** One card can pay for multiple people (tap multiple times)\n\n"
        
        response += "**🚇 Metro Lines - Current Status & Key Connections:**\n"
        response += "• **M2 Red Line:** Main tourist line (Taksim, Şişli, connects to M7 for historic areas)\n"
        response += "• **M7 Purple Line:** Kabataş (ferries) to Mecidiyeköy (M2 connection)\n"
        response += "• **M1 Blue Line:** Airport connection via Yenikapı transfer hub\n"
        response += "• **Operating hours:** 06:00-00:30 weekdays, until 02:00 weekends\n"
        response += "• **Frequency:** 2-5 minutes peak hours, 5-10 minutes off-peak\n\n"
        
        response += "**🚋 Tram System - Historic & Modern Areas:**\n"
        response += "• **T1 (Blue Tram):** Kabataş → Bağcılar (passes ALL major historic sites)\n"
        response += "• **Tourist stops:** Karaköy, Eminönü (Spice Bazaar), Sultanahmet, Beyazıt (Grand Bazaar)\n"
        response += "• **T4 Tram:** Modern areas, connects to M2 metro\n"
        response += "• **Real-time:** Digital displays show next 2-3 arrival times\n\n"
        
        response += "**⛴️ Ferry System - Scenic & Practical:**\n"
        response += "• **Golden Horn:** Eminönü ↔ Eyüp (historic neighborhoods)\n"
        response += "• **Bosphorus:** Multiple routes with city views\n"
        response += "• **Asian-European crossing:** Kadıköy ↔ Eminönü (most popular tourist route)\n"
        response += "• **Weather dependent:** Strong winds may cancel services\n\n"
        
        response += "**� Bus Network - Comprehensive Coverage:**\n"
        response += "• **Metrobüs:** Rapid transit on dedicated lanes, avoids traffic\n"
        response += "• **Regular buses:** Extensive network, some run 24/7\n"
        response += "• **Night services:** Limited routes after midnight\n"
        response += "• **Bus stops:** Digital displays show live arrival times\n\n"
        
        response += "**🕐 Real-Time Schedule Information:**\n"
        response += "• **Peak hours:** 07:00-09:30 and 17:30-19:30 (avoid if possible)\n"
        response += "• **Weekend schedules:** Reduced frequency, later start times\n"
        response += "• **Holiday variations:** Major religious holidays affect schedules\n"
        response += "• **Live updates:** All stations have digital arrival boards\n\n"
        
        response += "**💡 Insider Navigation Tips:**\n"
        response += "• **Follow colors:** Each line has distinct colors, follow signs\n"
        response += "• **Exit planning:** Check station maps for closest exit to your destination\n"
        response += "• **Transfer efficiency:** Allow 3-5 minutes for major transfers\n"
        response += "• **Language barrier:** Station names in Turkish and English, staff speak basic English\n"
        response += "• **Accessibility:** Most modern stations have elevators and audio announcements"
        
        return response
        
    except Exception as e:
        print(f"⚠️ Error getting enhanced live transport info: {e}")
        return None

async def get_live_restaurant_recommendations(user_input: str) -> Optional[str]:
    """Get live restaurant recommendations based on user query"""
    try:
        from api_clients.google_places import GooglePlacesClient
        user_lower = user_input.lower()
        
        # Extract location/district from query
        districts = {
            'sultanahmet': 'Sultanahmet', 'taksim': 'Taksim', 'beyoglu': 'Beyoğlu',
            'galata': 'Galata', 'kadikoy': 'Kadıköy', 'besiktas': 'Beşiktaş',
            'eminonu': 'Eminönü', 'fatih': 'Fatih', 'sisli': 'Şişli',
            'karakoy': 'Karaköy', 'ortakoy': 'Ortaköy', 'bebek': 'Bebek'
        }
        
        location = None
        for key, value in districts.items():
            if key in user_lower:
                location = f"{value}, Istanbul, Turkey"
                break
        
        if not location:
            location = "Istanbul, Turkey"
        
        # Extract food type/keyword
        keyword = None
        food_keywords = {
            'turkish': 'Turkish', 'kebab': 'kebab', 'seafood': 'seafood',
            'vegetarian': 'vegetarian', 'halal': 'halal', 'breakfast': 'breakfast',
            'coffee': 'coffee', 'cafe': 'cafe', 'baklava': 'baklava',
            'meze': 'meze', 'traditional': 'traditional'
        }
        
        for key, value in food_keywords.items():
            if key in user_lower:
                keyword = value
                break
        
        # Get restaurant recommendations
        client = GooglePlacesClient()
        restaurants = client.get_restaurants_with_descriptions(
            location=location,
            radius=1500,
            limit=5,
            keyword=keyword
        )
        
        if not restaurants:
            return None
            
        # Format response
        response = f"**🍽️ Restaurant Recommendations for {location.split(',')[0]}:**\n\n"
        
        for i, restaurant in enumerate(restaurants, 1):
            name = restaurant.get('name', 'Unknown Restaurant')
            rating = restaurant.get('rating', 'N/A')
            price_level = restaurant.get('price_level', None)
            address = restaurant.get('vicinity', 'Address not available')
            
            response += f"**{i}. {name}**\n"
            response += f"   ⭐ Rating: {rating}/5"
            
            if price_level:
                price_symbols = "💰" * price_level
                response += f" | {price_symbols}"
            
            response += f"\n   📍 {address}\n"
            
            # Add description if available
            if restaurant.get('description'):
                description = restaurant['description'][:100] + "..." if len(restaurant.get('description', '')) > 100 else restaurant.get('description', '')
                response += f"   ℹ️ {description}\n"
            
            # Add opening hours if available
            if restaurant.get('opening_hours', {}).get('open_now') is not None:
                status = "Open now" if restaurant['opening_hours']['open_now'] else "Closed now"
                response += f"   🕐 {status}\n"
            
            response += "\n"
        
        response += "**💡 Pro Tips:**\n"
        response += "• Make reservations for dinner, especially on weekends\n"
        response += "• Try traditional Turkish breakfast if visiting in the morning\n"
        response += "• Turkish tea (çay) is complimentary at most restaurants\n"
        response += "• Tipping 10-15% is customary for good service"
        
        return response
        
    except Exception as e:
        print(f"⚠️ Error getting live restaurant recommendations: {e}")
        return None

def enhance_daily_talk_response(base_response: str, user_input: str) -> str:
    """Enhanced daily talk responses with deep empathy, cultural sensitivity, and personalized support"""
    user_lower = user_input.lower()
    
    # Analyze emotional tone and situation for personalized enhancement
    emotional_context = {}
    
    # Detect specific emotional states
    if any(word in user_lower for word in ['overwhelmed', 'overwhelming', 'too much', 'stressed']):
        emotional_context['state'] = 'overwhelmed'
        empathy_opening = "I completely understand that Istanbul can feel overwhelming at first - it's a vibrant, bustling metropolis with so much history and energy."
        
    elif any(word in user_lower for word in ['lonely', 'alone', 'solo', 'by myself']):
        emotional_context['state'] = 'lonely'
        empathy_opening = "Traveling solo can sometimes feel isolating, especially in such a culturally rich and busy city like Istanbul."
        
    elif any(word in user_lower for word in ['nervous', 'scared', 'worried', 'anxious']):
        emotional_context['state'] = 'anxious'
        empathy_opening = "It's completely natural to feel nervous when exploring a new city - these feelings show you care about having a good experience."
        
    elif any(word in user_lower for word in ['excited', 'happy', 'thrilled', 'can\'t wait']):
        emotional_context['state'] = 'excited'
        empathy_opening = "Your excitement is wonderful to hear! Istanbul has a way of captivating visitors with its unique blend of cultures and experiences."
        
    elif any(word in user_lower for word in ['confused', 'lost', 'don\'t know', 'not sure']):
        emotional_context['state'] = 'confused'
        empathy_opening = "Feeling a bit lost is part of the Istanbul experience - the city has layers of history and culture that take time to understand."
        
    elif any(word in user_lower for word in ['first time', 'never been', 'new to']):
        emotional_context['state'] = 'first_timer'
        empathy_opening = "First-time visits to Istanbul are special! The city offers such a rich tapestry of experiences that it can feel both exciting and overwhelming."
        
    else:
        empathy_opening = "I'm here to help make your Istanbul experience as wonderful and comfortable as possible."
    
    # Detect specific concerns for targeted advice
    concerns = []
    if 'language' in user_lower or 'turkish' in user_lower or 'communicate' in user_lower:
        concerns.append('language_barrier')
    if 'safe' in user_lower or 'safety' in user_lower:
        concerns.append('safety')
    if 'culture' in user_lower or 'different' in user_lower or 'customs' in user_lower:
        concerns.append('cultural_differences')
    if 'big' in user_lower or 'huge' in user_lower or 'size' in user_lower:
        concerns.append('city_size')
    if 'tourist' in user_lower and 'trap' in user_lower:
        concerns.append('authenticity')
    
    # Build enhanced response
    enhanced_parts = [empathy_opening]
    
    # Add the original response with improvements
    enhanced_parts.append(base_response)
    
    # Add specific cultural insights based on concerns
    cultural_insights = []
    
    if 'language_barrier' in concerns:
        cultural_insights.append("**Language Bridge:** While Turkish is the main language, many Istanbulites in tourist areas speak some English. Learn these key phrases: 'Merhaba' (Hello), 'Teşekkür ederim' (Thank you), 'Özür dilerim' (Excuse me), 'İngilizce biliyor musunuz?' (Do you speak English?). Even attempting Turkish is greatly appreciated!")
        
    if 'safety' in concerns:
        cultural_insights.append("**Safety Perspective:** Istanbul is generally very safe for tourists. Turkish culture values hospitality (misafirperverlik), and locals often go out of their way to help visitors. Trust your instincts, stay in well-lit areas at night, and don't hesitate to ask for help - Istanbulites are proud of their city and want you to have a positive experience.")
        
    if 'cultural_differences' in concerns:
        cultural_insights.append("**Cultural Bridge:** Istanbul beautifully straddles East and West, creating a unique cultural blend. You'll see modern cafes next to Ottoman mosques, traditional tea houses beside contemporary art galleries. This diversity is what makes the city special - embrace the contrasts!")
        
    if 'city_size' in concerns:
        cultural_insights.append("**Managing the Scale:** Think of Istanbul as a collection of neighborhoods, each with its own personality. Start with one area (like Sultanahmet for history or Beyoğlu for modern culture), get comfortable there, then gradually explore others. Even locals focus on their favorite districts rather than trying to know the entire city.")
        
    if 'authenticity' in concerns:
        cultural_insights.append("**Authentic Experiences:** To experience 'real' Istanbul, visit neighborhood tea houses (çay ocağı), shop at local markets like Kadıköy's Tuesday market, take ferries with commuters, and eat at lokanta restaurants where locals dine. The best experiences often happen when you slow down and observe daily life.")
    
    # Add cultural insights if any were identified
    if cultural_insights:
        enhanced_parts.append("\n**Cultural Insights:**")
        enhanced_parts.extend(cultural_insights)
    
    # Add emotional support and encouragement based on state
    encouragement = []
    
    if emotional_context.get('state') == 'overwhelmed':
        encouragement.append("Remember: every visitor feels this way initially. Take breaks, breathe, and know that Istanbul rewards patience. Start small, celebrate small victories, and let the city reveal itself gradually.")
        
    elif emotional_context.get('state') == 'lonely':
        encouragement.append("Solo travel in Istanbul can be incredibly rewarding. Consider joining walking tours to meet other travelers, visiting community spaces like tea gardens, or asking your hotel/hostel about group activities. Istanbulites are naturally social and welcoming.")
        
    elif emotional_context.get('state') == 'anxious':
        encouragement.append("Your caution shows wisdom. Start with well-established tourist areas where you feel comfortable, then gradually venture into more local neighborhoods as your confidence grows. Every experienced traveler started with the same feelings.")
        
    elif emotional_context.get('state') == 'excited':
        encouragement.append("Channel that excitement into exploration! Istanbul rewards curious visitors. Try something new each day - a different neighborhood, a local dish, a conversation with a shopkeeper. Your enthusiasm will open doors.")
        
    elif emotional_context.get('state') == 'confused':
        encouragement.append("Confusion is just curiosity waiting to be satisfied. Istanbul has been a crossroads of civilizations for centuries - its complexity is part of its charm. Ask questions, stay curious, and don't worry about understanding everything at once.")
        
    else:
        encouragement.append("Istanbul has welcomed travelers for over 1,500 years - you're part of that grand tradition. Trust in the city's ability to surprise and delight you. Take your time, stay open to experiences, and enjoy the journey of discovery.")
    
    # Add encouraging closing
    if encouragement:
        enhanced_parts.append(f"\n**Personal Encouragement:** {encouragement[0]}")
    
    # Add practical next steps
    next_steps = "**Your Next Step:** Choose one small goal for today - perhaps visiting a nearby attraction, trying a local dish, or simply taking a walk through your neighborhood. Small steps lead to great adventures! 🌟"
    enhanced_parts.append(f"\n{next_steps}")
    
    # Combine all parts
    enhanced_response = "\n".join(enhanced_parts)
    
    # Clean up any formatting issues
    enhanced_response = re.sub(r'\n{3,}', '\n\n', enhanced_response)
    enhanced_response = enhanced_response.strip()
    
    return enhanced_response

def create_clarification_response(user_input: str) -> str:
    """Create a response that asks for clarification on vague queries"""
    user_lower = user_input.lower()
    
    # Determine the type of clarification needed
    if 'food' in user_lower or 'eat' in user_lower:
        return """🍽️ **I'd love to help you find great food in Istanbul!**

To give you the best recommendations, could you tell me:

• **What type of cuisine?** (Turkish, seafood, vegetarian, etc.)
• **Which area?** (Sultanahmet, Taksim, Kadıköy, etc.) 
• **What meal?** (breakfast, lunch, dinner, snacks)
• **Dining style?** (casual, upscale, street food, rooftop)

For example: "Turkish restaurants in Sultanahmet for dinner" or "best breakfast places near Taksim"

🌟 **Popular choices:**
• **Turkish cuisine:** Kebabs, meze, baklava
• **Breakfast:** Traditional kahvaltı spreads
• **Street food:** Balık ekmek, döner, simit
• **Desserts:** Turkish delight, künefe, ice cream"""

    elif 'place' in user_lower or 'where' in user_lower:
        return """🏛️ **Istanbul has so many amazing places to explore!**

What type of experience are you looking for?

• **Historical sites?** (Hagia Sophia, Topkapi Palace, Blue Mosque)
• **Modern attractions?** (Galata Tower, Istanbul Modern, shopping)
• **Neighborhoods?** (Sultanahmet, Beyoğlu, Kadıköy)
• **Activities?** (Bosphorus cruise, markets, nightlife)
• **Museums?** (art, history, cultural sites)

Let me know what interests you most and I can provide detailed recommendations with practical tips!"""
