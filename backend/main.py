# --- Standard Library Imports ---
import sys
import os
import re
import asyncio
import json
import time
import html
from datetime import datetime, date
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import traceback
from collections import defaultdict

# --- Third-Party Imports ---
from fastapi import FastAPI, Request, UploadFile, File, Form
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
    from database import engine, SessionLocal
    print("✅ Database import successful")
except ImportError as e:
    print(f"❌ Database import failed: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:5]}")  # First 5 paths
    print(f"Files in current directory: {os.listdir('.')}")
    raise

try:
    from models import Base, Restaurant, Museum, Place, UserFeedback, ChatSession
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
    
    # PHASE 5: Enhance formatting for readability
    # Convert markdown to readable format
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold** but keep content
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic* but keep content
    
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
    """Generate response using OpenAI GPT with advanced query analysis for better relevance"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not OpenAI_available or not openai_api_key or OpenAI is None:
            return None
        
        # Sanitize input
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        # Import and use advanced query analysis and location enhancement
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
        
        # Create OpenAI client
        client = OpenAI(api_key=openai_api_key, timeout=30.0, max_retries=2)
        
        # Enhanced system prompt with query analysis insights
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
- Kadıköy (Asian side, authentic, local markets, Moda)
- Galata (trendy cafes, art galleries, views)
- Bosphorus (bridges, ferry rides, waterfront)
- Transportation (metro, tram, ferry, Istanbulkart, BiTaksi)
- Districts, museums, restaurants, culture, history, Byzantine, Ottoman, Asia/Europe{location_focus}"""

        # Make the API call with enhanced prompt
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=450,
            temperature=0.7,
            timeout=25
        )
        
        gpt_response = response.choices[0].message.content
        if gpt_response:
            gpt_response = gpt_response.strip()
            
            # Apply post-LLM cleanup to remove pricing and fix location issues
            gpt_response = post_llm_cleanup(gpt_response)
            
            # Temporarily disable ALL location enhancement to see core GPT responses
            print(f"📝 Using pure GPT response without enhancement for debugging")
            
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
        r'\b(?:costs?|prices?)\s+(?:around\s+|about\s+|approximately\s+)?\d+[\d.,]*',
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

def generate_restaurant_info(restaurant_name, location="Istanbul"):
    """Generate a brief, plain-text description for a restaurant based on its name"""
    name_lower = restaurant_name.lower()
    
    # Common Turkish restaurant types and food indicators
    if any(word in name_lower for word in ['kebap', 'kebab', 'çiğ köfte', 'döner', 'dürüm']):
        return "A popular kebab restaurant serving traditional Turkish grilled meats and specialties."
    elif any(word in name_lower for word in ['pizza', 'pizzeria', 'italian']):
        return "An Italian restaurant specializing in authentic pizzas and Mediterranean cuisine."
    elif any(word in name_lower for word in ['sushi', 'japanese', 'asian']):
        return "A Japanese restaurant offering fresh sushi and traditional Asian dishes."
    elif any(word in name_lower for word in ['burger', 'american', 'grill']):
        return "A casual dining spot known for burgers, grilled foods, and American-style cuisine."
    elif any(word in name_lower for word in ['cafe', 'kahve', 'coffee']):
        return "A cozy cafe perfect for coffee, light meals, and a relaxed atmosphere."
    elif any(word in name_lower for word in ['balık', 'fish', 'seafood', 'deniz']):
        return "A seafood restaurant featuring fresh fish and Mediterranean coastal cuisine."
    elif any(word in name_lower for word in ['ev yemeği', 'lokanta', 'traditional']):
        return "A traditional Turkish restaurant serving home-style cooking and local specialties."
    elif any(word in name_lower for word in ['meze', 'rakı', 'taverna']):
        return "A traditional meze restaurant offering small plates and Turkish appetizers."
    elif any(word in name_lower for word in ['pastane', 'tatlı', 'dessert', 'bakery']):
        return "A bakery and dessert shop known for Turkish sweets and fresh pastries."
    elif any(word in name_lower for word in ['steakhouse', 'et', 'meat']):
        return "A steakhouse specializing in premium cuts of meat and grilled dishes."
    else:
        return "A well-regarded restaurant offering quality dining and local cuisine."

def enhance_ai_response_formatting(text):
    """Enhance AI responses for better readability and structure"""
    if not text:
        return text
    
    # Ensure proper paragraph separation
    # Add line breaks before new topics (sentences starting with capital after period)
    text = re.sub(r'([.!?])\s+([A-Z][^.!?]{20,})', r'\1\n\n\2', text)
    
    # Improve list formatting - detect natural lists
    # Convert sequences like "First, ... Second, ... Third, ..." into bullet points
    text = re.sub(r'\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)[,:]?\s+', r'• ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)[.)]\s+', r'• ', text)  # Convert "1. " to "• "
    
    # Improve section detection - capitalize important keywords
    important_keywords = [
        'highlights', 'location', 'hours', 'features', 'attractions', 'neighborhoods',
        'districts', 'museums', 'restaurants', 'cuisine', 'atmosphere', 'specialties',
        'recommended', 'popular', 'famous', 'historic', 'cultural', 'traditional'
    ]
    
    for keyword in important_keywords:
        # Make keywords at start of sentences stand out
        pattern = r'\b' + keyword + r'\b(?=\s*:|\s+include|\s+are)'
        text = re.sub(pattern, keyword.upper(), text, flags=re.IGNORECASE)
    
    # Improve readability of long sentences by adding breathing room
    text = re.sub(r'([,;])\s*([A-Z][^,;.]{30,})', r'\1\n\2', text)
    
    # Ensure bullet points are properly formatted
    text = re.sub(r'\n\s*[•·-]\s*', '\n• ', text)
    text = re.sub(r'^[•·-]\s*', '• ', text, flags=re.MULTILINE)
    
    # Add proper spacing around sections
    text = re.sub(r'\n([A-Z][^•\n]{10,})\n([•])', r'\n\1\n\n\2', text)
    
    # Clean up excessive spacing
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    return text.strip()

def should_use_gpt_for_query(user_input: str) -> bool:
    """Determine if a query should be handled by GPT instead of database/hardcoded responses"""
    
    user_lower = user_input.lower()
    
    # Keywords that suggest the query needs more nuanced/detailed answers
    gpt_suitable_keywords = [
        # Basic Istanbul attractions and places (most important for test coverage)
        'sultanahmet', 'beyoglu', 'beyoğlu', 'kadikoy', 'kadıköy', 'galata', 'taksim',
        'besiktas', 'beşiktaş', 'fatih', 'eminonu', 'eminönü', 'ortakoy', 'ortaköy',
        'karakoy', 'karaköy', 'cihangir', 'balat', 'uskudar', 'üsküdar', 'arnavutkoy',
        'arnavutköy', 'moda', 'istiklal', 'spice bazaar', 'grand bazaar',
        'hagia sophia', 'blue mosque', 'topkapi palace', 'dolmabahce', 'dolmabahçe',
        'galata tower', 'maiden tower', 'bosphorus', 'golden horn', 'archaeological museum',
        'istanbul modern', 'pera museum', 'basilica cistern', 'turkish and islamic',
        
        # District and area questions
        'district', 'neighborhood', 'area', 'region', 'side', 'part of istanbul',
        'tell me about', 'what about', 'should i visit', 'worth visiting',
        'whats special', "what's special", 'known for', 'famous for',
        
        # Transportation basics
        'airport', 'metro', 'tram', 'ferry', 'bus', 'taxi', 'transport',
        'istanbulkart', 'istanbul card', 'how to get', 'get from', 'get to',
        'bitaksi', 'uber', 'public transport', 'metrobus',
        
        # Basic travel questions
        'museum', 'palace', 'mosque', 'church', 'tower', 'bridge', 'market',
        'restaurant', 'food', 'dining', 'eat', 'shopping', 'nightlife',
        'attractions', 'sightseeing', 'tourist', 'visit', 'see', 'do',
        
        # Art and culture specific queries
        'art museums', 'art galleries', 'contemporary art', 'modern art', 'art scene',
        'cultural events', 'art exhibitions', 'gallery openings', 'art districts',
        'street art', 'local artists', 'art history', 'byzantine art', 'ottoman art',
        
        # Detailed cultural questions  
        'culture', 'cultural heritage', 'traditions', 'customs', 'local customs',
        'turkish culture', 'byzantine culture', 'ottoman culture', 'cultural sites',
        'cultural experiences', 'authentic experiences', 'local life', 'local lifestyle',
        
        # Complex travel questions
        'itinerary', 'travel plan', 'how many days', 'best time to visit',
        'what should i do', 'things to avoid', 'travel tips', 'first time visiting',
        'budget travel', 'luxury travel', 'solo travel', 'family travel',
        
        # Transportation questions that need detailed responses
        'transportation options', 'transport system', 'public transport guide', 
        'transportation tips', 'transport planning', 'transport costs', 'transport passes',
        'airport transfer', 'airport transport', 'airport connection',
        'late night transport', 'early morning transport', 'transport schedule',
        'accessibility transport', 'disabled transport', 'wheelchair accessible',
        'transport for families', 'transport with luggage', 'transport safety',
        'cheapest transport', 'fastest transport', 'most comfortable transport',
        'transport during rush hour', 'avoiding traffic', 'transport strikes',
        
        # Food and dining questions
        'vegetarian', 'vegan', 'halal', 'kosher', 'dietary', 'allergies', 'gluten free',
        'turkish pizza', 'turkish coffee', 'turkish breakfast', 'street food',
        'tipping culture', 'restaurant etiquette', 'dining customs', 'alcohol policy',
        'ramadan dining', 'iftar', 'local food', 'authentic cuisine',
        
        # Cultural and religious questions
        'hagia sophia', 'mosque etiquette', 'prayer times', 'dress code',
        'religious customs', 'islamic culture', 'christian heritage',
        'religious tolerance', 'interfaith', 'secular turkey',
        'church or mosque', 'mosque or church', 'religious history',
        'byzantine church', 'ottoman mosque', 'conversion', 'converted',
        'museum to mosque', 'church to mosque', 'ayasofya', 'aya sofia',
        'blue mosque', 'sultan ahmed', 'suleymaniye', 'religious site',
        'basilica', 'cathedral', 'minarets', 'dome', 'orthodox', 'islamic',
        'christian', 'muslim', 'history of hagia sophia', 'hagia sophia history',
        
        # Geographic and location questions
        'europe or asia', 'continents', 'geography', 'bosphorus',
        'asian side', 'european side', 'golden horn', 'marmara sea',
        'neighborhoods', 'districts', 'areas to avoid', 'safe areas',
        'local vs tourist areas', 'authentic neighborhoods',
        
        # Language and communication
        'language', 'turkish language', 'arabic', 'kurdish', 'english',
        'communication', 'greeting', 'phrases', 'translation', 'speak',
        
        # Currency and economics
        'currency', 'lira', 'dollars', 'euros', 'exchange', 'money',
        'bargaining', 'negotiation', 'prices', 'costs', 'budget',
        
        # Safety and practical concerns
        'safety', 'security', 'scams', 'pickpockets', 'areas to avoid',
        'emergency', 'police', 'hospital', 'embassy', 'consulate',
        
        # Weather and timing
        'weather', 'climate', 'winter', 'summer', 'spring', 'autumn',
        'rain', 'snow', 'temperature', 'seasonal', 'best season',
        'ramadan', 'holidays', 'festivals', 'events', 'celebrations',
        
        # Complex logistical questions
        'uber', 'taxi apps', 'bitaksi', 'ride sharing', 'transportation apps',
        'night life', 'subway vs metro', 'train schedule', 'ferry schedule',
        
        # Specific expertise questions
        'history of', 'story behind', 'why is', 'how did', 'when was',
        'what makes', 'tell me about', 'explain', 'describe', 'background',
        
        # Comparative questions
        'compare', 'versus', 'vs', 'better than', 'difference between',
        'similar to', 'like', 'alternative to', 'instead of',
        
        # Opinion-based questions
        'recommend', 'suggest', 'opinion', 'thoughts', 'what do you think',
        'best', 'favorite', 'top', 'must see', 'must do', 'essential',
        
        # Complex logistics
        'how to plan', 'planning', 'organize', 'schedule', 'timing'
    ]
    
    # Check if query contains GPT-suitable keywords
    has_gpt_keywords = any(keyword in user_lower for keyword in gpt_suitable_keywords)
    
    # Question words that indicate need for detailed responses
    question_words = ['what', 'why', 'how', 'when', 'where', 'which', 'who', 'can', 'should', 'is', 'are', 'do', 'does', 'will', 'would', 'could', 'tell me']
    has_question_words = any(word in user_lower for word in question_words)
    
    # Use GPT for most questions that aren't just simple location queries
    is_question = has_question_words and len(user_input.split()) > 2  # Lowered from 3 to 2
    
    # Exceptions: Simple hardcoded responses that work well
    simple_fallback_patterns = [
        r'^\w+$',  # Single word queries
        r'^(hello|hi|hey)\b',  # Greetings with word boundaries
    ]
    
    is_simple_pattern = any(re.search(pattern, user_lower) for pattern in simple_fallback_patterns)
    
    # Use GPT for most queries except very simple ones
    return (has_gpt_keywords or is_question) and not is_simple_pattern

def is_complex_transportation_query(user_input: str) -> bool:
    """Determine if a transportation query is complex and should use GPT fallback"""
    
    user_lower = user_input.lower()
    
    # Complex transportation scenarios that benefit from GPT
    complex_transport_indicators = [
        # General transportation guidance (not specific routes)
        'transportation options', 'transport system', 'public transport guide', 
        'transportation in istanbul', 'getting around istanbul', 'transport overview',
        'how does transport work', 'transport explanation', 'transport help',
        
        # Cost and planning questions
        'transport costs', 'transport prices', 'cheap transport', 'expensive transport',
        'transport budget', 'transport planning', 'transport passes', 'transport cards',
        'best way to travel', 'fastest way to travel', 'cheapest way to travel',
        
        # Time-sensitive questions
        'late night transport', 'early morning transport', 'transport schedule',
        'transport during rush hour', 'avoiding traffic', 'rush hour transport',
        'transport at night', 'night buses', 'night transport',
        
        # Special needs transportation
        'accessibility transport', 'disabled transport', 'wheelchair accessible',
        'transport for families', 'transport with children', 'transport with luggage',
        'transport with baby', 'stroller friendly transport',
        
        # Airport and long-distance connections
        'airport transfer', 'airport transport', 'airport connection', 'airport shuttle',
        'from airport to', 'to airport from', 'airport to city', 'city to airport',
        
        # Multi-modal or complex routing
        'multiple stops', 'several places', 'tour multiple', 'visit multiple',
        'day trip transport', 'transport for sightseeing', 'tourist transport',
        
        # Comparative transportation questions
        'best transport option', 'transport comparison', 'metro vs bus', 'taxi vs metro',
        'which transport is better', 'transport recommendations',
        
        # Transport system questions
        'how to buy tickets', 'where to buy tickets', 'transport tickets',
        'istanbulkart', 'transport card', 'oyster card equivalent',
        
        # Safety and comfort
        'transport safety', 'safe transport', 'comfortable transport',
        'transport tips', 'transport advice', 'transport warnings'
    ]
    
    # Check for complex indicators
    has_complex_indicators = any(indicator in user_lower for indicator in complex_transport_indicators)
    
    # Check for question words combined with transportation
    transport_keywords = ['transport', 'metro', 'bus', 'ferry', 'taxi', 'travel', 'getting']
    has_transport_keywords = any(keyword in user_lower for keyword in transport_keywords)
    question_words = ['what', 'why', 'how', 'when', 'where', 'which', 'who']
    has_question_with_transport = (any(word in user_lower for word in question_words) and has_transport_keywords)
    
    # Simple informational queries that should NOT be considered complex
    simple_info_patterns = [
        r'what.*metro.*lines?',           # "what are the metro lines"
        r'metro.*lines?',                 # "metro lines in istanbul"
        r'what.*bus.*routes?',            # "what bus routes"
        r'what.*transport.*options?',     # "what transport options" (but this can be complex)
        r'tell me about.*metro',          # "tell me about metro"
        r'tell me about.*bus',            # "tell me about bus"
    ]
    
    # If it's a simple informational query, prefer hardcoded response
    is_simple_info = any(re.search(pattern, user_lower) for pattern in simple_info_patterns)
    
    # Simple route patterns that should use hardcoded responses
    simple_route_patterns = [
        r'from\s+\w+.*to\s+\w+',          # "from beyoglu to kadikoy"
        r'\w+\s+to\s+\w+',                # "beyoglu to kadikoy"  
        r'go.*\w+.*to\s+\w+',             # "go from beyoglu to kadikoy"
        r'get.*\w+.*to\s+\w+',            # "get from beyoglu to kadikoy"
    ]
    
    # If it's a simple route query, prefer hardcoded response
    is_simple_route = any(re.search(pattern, user_lower) for pattern in simple_route_patterns)
    
    return (has_complex_indicators or has_question_with_transport) and not is_simple_route and not is_simple_info

app = FastAPI(title="AIstanbul API")

# Daily usage tracking functions
import hashlib
import sqlite3
from datetime import datetime, timedelta

# Initialize daily usage database
def init_daily_usage_db():
    """Initialize the daily usage tracking database"""
    try:
        conn = sqlite3.connect('daily_usage.db')
        cursor = conn.cursor()
        
        # Create daily_usage table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_hash TEXT NOT NULL,
                date TEXT NOT NULL,
                usage_count INTEGER DEFAULT 1,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ip_hash, date)
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ip_date ON daily_usage (ip_hash, date)
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Daily usage database initialized")
        
    except Exception as e:
        print(f"❌ Error initializing daily usage database: {str(e)}")

def get_usage_db():
    """Get database connection for usage tracking"""
    try:
        conn = sqlite3.connect('daily_usage.db')
        conn.row_factory = sqlite3.Row  # Enable named access to columns
        return conn
    except Exception as e:
        print(f"❌ Error connecting to usage database: {str(e)}")
        raise

def get_client_ip(request):
    """Extract client IP address from request headers"""
    # Check for forwarded IP headers (common in reverse proxy setups)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in case of multiple forwarded IPs
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to client host
    return request.client.host if request.client else "unknown"

def hash_ip(ip_address):
    """Create a hash of IP address for privacy"""
    return hashlib.sha256(ip_address.encode()).hexdigest()[:16]

def check_and_update_daily_usage(client_ip, daily_limit=1000):
    """Check and update daily usage for a client IP"""
    try:
        conn = get_usage_db()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        ip_hash = hash_ip(client_ip)
        
        # Check current usage
        cursor.execute("""
            SELECT usage_count FROM daily_usage 
            WHERE ip_hash = ? AND date = ?
        """, (ip_hash, today))
        
        result = cursor.fetchone()
        current_usage = result[0] if result else 0
        
        # Check if limit exceeded
        if current_usage >= daily_limit:
            conn.close()
            return {
                'allowed': False,
                'current_usage': current_usage,
                'daily_limit': daily_limit,
                'remaining': 0,
                'message': f'Daily limit of {daily_limit} requests exceeded',
                'reset_time': (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
            }
        
        # Update usage count
        cursor.execute("""
            INSERT OR REPLACE INTO daily_usage (ip_hash, date, usage_count)
            VALUES (?, ?, ?)
        """, (ip_hash, today, current_usage + 1))
        
        conn.commit()
        conn.close()
        
        new_usage = current_usage + 1
        remaining = daily_limit - new_usage
        
        return {
            'allowed': True,
            'current_usage': new_usage,
            'daily_limit': daily_limit,
            'remaining': remaining,
            'message': 'Request allowed',
            'reset_time': (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error in daily usage check: {str(e)}")
        # On error, allow the request to proceed
        return {
            'allowed': True,
            'current_usage': 0,
            'daily_limit': daily_limit,
            'remaining': daily_limit,
            'message': 'Usage check failed, request allowed',
            'reset_time': (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
        }

def get_daily_usage_stats():
    """Get daily usage statistics for monitoring"""
    try:
        conn = get_usage_db()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get total usage today
        cursor.execute("""
            SELECT COUNT(*) as unique_ips, SUM(usage_count) as total_requests
            FROM daily_usage WHERE date = ?
        """, (today,))
        
        result = cursor.fetchone()
        unique_ips = result[0] if result else 0
        total_requests = result[1] if result else 0
        
        # Get top usage IPs today
        cursor.execute("""
            SELECT ip_hash, usage_count FROM daily_usage 
            WHERE date = ? ORDER BY usage_count DESC LIMIT 10
        """, (today,))
        
        top_usage = cursor.fetchall()
        
        # Get usage over last 7 days
        cursor.execute("""
            SELECT date, COUNT(*) as unique_ips, SUM(usage_count) as total_requests
            FROM daily_usage 
            WHERE date >= date('now', '-7 days')
            GROUP BY date ORDER BY date DESC
        """)
        
        weekly_stats = cursor.fetchall()
        conn.close()
        
        return {
            'today': {
                'date': today,
                'unique_ips': unique_ips,
                'total_requests': total_requests
            },
            'top_usage_today': [
                {'ip_hash': row[0], 'requests': row[1]} 
                for row in top_usage
            ],
            'weekly_stats': [
                {'date': row[0], 'unique_ips': row[1], 'total_requests': row[2]}
                for row in weekly_stats
            ]
        }
        
    except Exception as e:
        print(f"❌ Error getting usage stats: {str(e)}")
        return {
            'error': str(e),
            'today': {'date': datetime.now().strftime("%Y-%m-%d"), 'unique_ips': 0, 'total_requests': 0},
            'top_usage_today': [],
            'weekly_stats': []
        }

def reset_daily_usage(force=False):
    """Reset daily usage counters"""
    try:
        conn = get_usage_db()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        
        if force:
            # Reset all usage data
            cursor.execute("DELETE FROM daily_usage")
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            return {
                'message': 'All usage data cleared',
                'rows_affected': rows_affected,
                'force': True
            }
        else:
            # Only reset today's data
            cursor.execute("DELETE FROM daily_usage WHERE date = ?", (today,))
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            return {
                'message': f'Usage data cleared for {today}',
                'rows_affected': rows_affected,
                'date': today,
                'force': False
            }
            
    except Exception as e:
        print(f"❌ Error resetting usage: {str(e)}")
        return {
            'error': str(e),
            'message': 'Failed to reset usage data'
        }

def get_ip_usage_info(client_ip):
    """Get usage information for a specific IP address"""
    try:
        conn = get_usage_db()
        cursor = conn.cursor()
        ip_hash = hash_ip(client_ip)
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get today's usage
        cursor.execute("""
            SELECT usage_count FROM daily_usage 
            WHERE ip_hash = ? AND date = ?
        """, (ip_hash, today))
        
        result = cursor.fetchone()
        today_usage = result[0] if result else 0
        
        # Get usage over last 7 days
        cursor.execute("""
            SELECT date, usage_count FROM daily_usage 
            WHERE ip_hash = ? AND date >= date('now', '-7 days')
            ORDER BY date DESC
        """, (ip_hash,))
        
        weekly_usage = cursor.fetchall()
        conn.close()
        
        return {
            'ip_hash': ip_hash,
            'today_usage': today_usage,
            'today_date': today,
            'weekly_usage': [
                {'date': row[0], 'requests': row[1]} 
                for row in weekly_usage
            ],
            'total_weekly': sum(row[1] for row in weekly_usage)
        }
        
    except Exception as e:
        print(f"❌ Error getting IP usage info: {str(e)}")
        return {
            'error': str(e),
            'ip_hash': hash_ip(client_ip),
            'today_usage': 0,
            'weekly_usage': [],
            'total_weekly': 0
        }

# --- Daily Usage Tracking Middleware ---
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware
import json

class DailyUsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Skip usage tracking for admin endpoints and health checks
        if request.url.path.startswith(("/admin", "/health", "/static")):
            return await call_next(request)
        
        # Get client IP
        client_ip = get_client_ip(request)
        
        # Check and update daily usage
        usage_check = check_and_update_daily_usage(client_ip)
        
        # If usage limit exceeded, return error
        if not usage_check['allowed']:
            return Response(
                content=json.dumps({
                    "error": "Daily request limit exceeded",
                    "message": usage_check['message'],
                    "remaining": usage_check['remaining'],
                    "reset_time": usage_check['reset_time']
                }),
                status_code=429,
                media_type="application/json"
            )
        
        # Add usage info to response headers
        response = await call_next(request)
        response.headers["X-Daily-Usage"] = str(usage_check['current_usage'])
        response.headers["X-Daily-Limit"] = str(usage_check['daily_limit'])
        response.headers["X-Daily-Remaining"] = str(usage_check['remaining'])
        
        return response

# Add daily usage middleware
app.add_middleware(DailyUsageMiddleware)

# Admin endpoint for monitoring daily usage
@app.get("/admin/usage-stats")
async def get_usage_stats():
    """Get daily usage statistics for monitoring"""
    try:
        stats = get_daily_usage_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        print(f"❌ Error getting usage stats: {str(e)}")
        return {"error": f"Failed to get usage stats: {str(e)}"}

@app.post("/admin/reset-usage")
async def reset_usage_admin(request: dict = None):
    """Reset daily usage counters (admin only)"""
    try:
        force = request.get("force", False) if request else False
        result = reset_daily_usage(force=force)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        print(f"❌ Error resetting usage: {str(e)}")
        return {"error": f"Failed to reset usage: {str(e)}"}

@app.get("/admin/ip-usage/{ip_hash}")
async def get_ip_usage_admin(ip_hash: str):
    """Get usage info for a specific IP (using hash for privacy)"""
    try:
        # This is a simplified version - in real implementation you'd need to reverse the hash
        # or store IP mappings securely
        return {
            "status": "success", 
            "message": "Use get_ip_usage_info() function directly with actual IP address"
        }
    except Exception as e:
        print(f"❌ Error getting IP usage: {str(e)}")
        return {"error": f"Failed to get IP usage: {str(e)}"}

# Basic AI chat endpoints - add these BEFORE including other routers
@app.post("/ai/chat")
async def chat(request: dict):
    try:
        user_message = request.get("message", "")
        if not user_message:
            return {"error": "Message is required"}
        
        print(f"💬 Chat request received: {user_message[:50]}...")
        
        # Use advanced query analysis for better routing
        try:
            from query_analyzer import analyze_and_enhance_query, QueryType, LocationContext
            from location_enhancer import location_enhancer
            
            analysis, enhanced_prompt = analyze_and_enhance_query(user_message)
            
            print(f"🔍 Query Analysis: {analysis.query_type.value} | {analysis.location_context.value} | Confidence: {analysis.confidence_score:.2f}")
            
            # Log analysis for debugging
            if analysis.dietary_restrictions:
                print(f"🥗 Dietary restrictions detected: {analysis.dietary_restrictions}")
            if analysis.time_context:
                print(f"⏰ Time context: {analysis.time_context}")
            if analysis.location_context != LocationContext.NONE:
                print(f"📍 Location focus: {analysis.location_context.value}")
            
            # Detect additional locations from query if analysis missed it
            detected_location = location_enhancer.detect_location_from_query(user_message)
            if detected_location and analysis.location_context == LocationContext.NONE:
                print(f"📍 Additional location detected by location enhancer: {detected_location}")
                
        except ImportError:
            print("⚠️ Query analyzer/location enhancer not available, using legacy detection")
            analysis = None
            detected_location = None
        
        # Check for location confusion first
        is_confused, confusion_response = detect_location_confusion(user_message)
        if is_confused and confusion_response:
            print(f"📍 Location confusion detected, providing redirection...")
            return {"response": confusion_response}
        
        # Enhanced restaurant query detection using analysis
        user_lower = user_message.lower()
        response = None  # Initialize response variable
        
        # Use query analysis for better restaurant detection
        if analysis and analysis.query_type in [QueryType.RESTAURANT_SPECIFIC, QueryType.RESTAURANT_GENERAL]:
            if analysis.location_context != LocationContext.NONE or analysis.confidence_score >= 0.5:
                print(f"🍽️ Restaurant query with location/high confidence detected, trying Google Maps integration...")
                response = get_live_restaurant_recommendations(user_message)
                if response:
                    print(f"✅ Google Maps restaurant response generated: {len(response)} characters")
                else:
                    print(f"⚠️ Google Maps integration failed, falling back to GPT...")
        elif is_specific_restaurant_query(user_message):
            # Fallback to legacy detection
            print(f"🍽️ Legacy restaurant query detection, trying Google Maps integration...")
            response = get_live_restaurant_recommendations(user_message)
            if response:
                print(f"✅ Google Maps restaurant response generated: {len(response)} characters")
            else:
                print(f"⚠️ Google Maps integration failed, falling back to GPT...")
        
        # Enhanced vague query detection using analysis
        if not response:
            if analysis and analysis.query_type == QueryType.RESTAURANT_GENERAL and analysis.confidence_score < 0.3:
                print(f"🤔 Vague food query detected via analysis, asking for clarification...")
                response = create_clarification_response(user_message)
                print(f"✅ Clarification response generated: {len(response)} characters")
                return {"response": response}
            else:
                # Legacy vague query detection
                food_keywords = ['food', 'eat', 'hungry', 'where eat', 'best place']
                if any(keyword in user_lower for keyword in food_keywords):
                    if len(user_message.strip().split()) <= 3:  # Short vague queries
                        print(f"🤔 Legacy vague food query detected, asking for clarification...")
                        response = create_clarification_response(user_message)
                        print(f"✅ Clarification response generated: {len(response)} characters")
                        return {"response": response}
        
        # Fall back to GPT if no Google Maps response or not a restaurant query
        if not response:
            response = get_gpt_response(user_message, "test-session")
            if not response:
                response = "I'd be happy to help you with information about Istanbul! Could you please provide more details about what you're looking for?"
        
        print(f"✅ Final response generated: {len(response)} characters")
        return {"response": response}
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return {"error": f"Chat error: {str(e)}"}

@app.post("/ai/stream")
async def stream_chat(request: dict):
    try:
        user_message = request.get("message", "")
        if not user_message:
            return {"error": "Message is required"}
        
        print(f"🔄 Stream chat request received: {user_message[:50]}...")
        
        # Check for location confusion first
        is_confused, confusion_response = detect_location_confusion(user_message)
        if is_confused and confusion_response:
            print(f"📍 Location confusion detected, providing redirection...")
            return {"response": confusion_response}
        
        # Check if this is a restaurant query 
        user_lower = user_message.lower()
        response = None  # Initialize response variable
        
        # Check for specific restaurant queries (not vague ones)
        if is_specific_restaurant_query(user_message):
            print(f"🍽️ Specific restaurant query detected, trying Google Maps integration...")
            response = get_live_restaurant_recommendations(user_message)
            if response:
                print(f"✅ Google Maps restaurant response generated: {len(response)} characters")
            else:
                print(f"⚠️ Google Maps integration failed, falling back to GPT...")
        else:
            # Check if it's a vague food-related query that needs clarification
            food_keywords = ['food', 'eat', 'hungry', 'where eat', 'best place']
            if any(keyword in user_lower for keyword in food_keywords):
                if len(user_message.strip().split()) <= 3:  # Short vague queries
                    print(f"🤔 Vague food query detected, asking for clarification...")
                    response = create_clarification_response(user_message)
                    print(f"✅ Clarification response generated: {len(response)} characters")
                    return {"response": response}
        
        # Fall back to GPT if no response generated yet
        if not response:
            response = get_gpt_response(user_message, "test-session")
            if not response:
                response = "I'd be happy to help you with information about Istanbul! Could you please provide more details about what you're looking for?"
        
        print(f"✅ Stream chat response generated: {len(response)} characters")
        return {"response": response}
    except Exception as e:
        print(f"❌ Stream chat error: {str(e)}")
        return {"error": f"Stream error: {str(e)}"}

# Missing API endpoints that frontend expects
@app.get("/ai/api/chat-sessions")
async def get_chat_sessions():
    """Get user's chat sessions"""
    try:
        # For now, return empty array since we don't have session persistence yet
        return {"sessions": []}
    except Exception as e:
        print(f"❌ Error getting chat sessions: {str(e)}")
        return {"error": f"Failed to get chat sessions: {str(e)}"}

@app.post("/ai/api/chat-sessions")
async def create_chat_session(request: dict):
    """Create a new chat session"""
    try:
        # For now, just return a dummy session ID
        return {"sessionId": "temp_session_" + str(hash(str(datetime.now())) % 10000)}
    except Exception as e:
        print(f"❌ Error creating chat session: {str(e)}")
        return {"error": f"Failed to create chat session: {str(e)}"}

@app.delete("/ai/api/chat-sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    try:
        # For now, just return success
        return {"success": True, "message": f"Session {session_id} deleted"}
    except Exception as e:
        print(f"❌ Error deleting chat session: {str(e)}")
        return {"error": f"Failed to delete chat session: {str(e)}"}

# Include blog router
app.include_router(blog.router)

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "OK", "message": "AI Istanbul Backend is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": str(datetime.now())}

# Add CORS middleware with secure origins
CORS_ORIGINS = [
    # Development ports (only for development)
    "http://localhost:3000",
    "http://localhost:3001", 
    "http://localhost:3002",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3002",
    "http://127.0.0.1:5173",
    # Production frontend URLs
    "https://aistanbul.vercel.app",
    "https://aistanbul.net",
    # Remove file:// protocol support for security
]

# Add environment variable for additional origins
additional_origins = os.getenv("CORS_ORIGINS", "")
if additional_origins:
    CORS_ORIGINS.extend([origin.strip() for origin in additional_origins.split(",")])

print(f"🌐 CORS enabled for origins: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Headers Middleware
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Security headers for GDPR compliance and general security
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
       
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        
        return response

# Add security and usage tracking middleware
app.add_middleware(SecurityHeadersMiddleware)
# Daily usage middleware will be added after function definitions

# Mount static files for admin dashboard
app.mount("/static", StaticFiles(directory="."), name="static")

# Create tables if needed
Base.metadata.create_all(bind=engine)

# Initialize AI Cache Service
if AI_CACHE_ENABLED:
    try:
        # Try to initialize with Redis, fallback to memory-only if Redis unavailable
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        # Use higher rate limits for development/testing
        ai_cache_service = init_ai_cache_service(
            redis_url=redis_url,
            rate_limit_per_user=100,  # 100 requests per user per hour
            rate_limit_per_ip=500     # 500 requests per IP per hour
        )  # type: ignore
        print(f"✅ AI Cache Service initialized with Redis: {redis_url}")
        print(f"📊 Rate limits: 100 per user/hour, 500 per IP/hour")
    except Exception as e:
        print(f"⚠️ AI Cache Service initialized without Redis: {e}")
        ai_cache_service = get_ai_cache_service()  # type: ignore
else:
    ai_cache_service = None

# --- Import GDPR service ---
try:
    from gdpr_service import gdpr_service
    GDPR_SERVICE_ENABLED = True
    print("✅ GDPR service loaded successfully")
except ImportError as e:
    print(f"⚠️ GDPR service not available: {e}")
    GDPR_SERVICE_ENABLED = False
    gdpr_service = None

def is_specific_restaurant_query(user_input: str) -> bool:
    """Determine if query is specific enough to warrant restaurant recommendations"""
    user_lower = user_input.strip().lower()
    
    # Filter out very vague single-word or short queries
    if len(user_input.strip().split()) <= 1:
        vague_words = ['food', 'eat', 'hungry', 'restaurant', 'dinner', 'lunch', 'breakfast']
        if user_lower in vague_words:
            return False
    
    # Filter out very short vague phrases
    vague_phrases = [
        'where eat', 'where eat?', 'eat something', 'eat something good', 
        'best place', 'good food', 'what eat', 'what to eat'
    ]
    if user_lower in vague_phrases:
        return False
    
    # Filter out cooking/recipe queries (not restaurant location queries)
    cooking_indicators = [
        'how to cook', 'recipe', 'cooking', 'ingredients', 'make at home',
        'prepare', 'homemade', 'cooking class', 'learn to cook'
    ]
    if any(indicator in user_lower for indicator in cooking_indicators):
        return False
    
    # Filter out queries about other cities
    other_cities = ['ankara', 'izmir', 'antalya', 'bursa', 'adana']
    if any(city in user_lower for city in other_cities) and 'istanbul' not in user_lower:
        return False
    
    # Filter out general Turkey queries (should be Istanbul-specific)
    if 'turkey' in user_lower and 'istanbul' not in user_lower:
        if len(user_input.split()) < 4:  # Short general queries like "restaurants in Turkey"
            return False
    
    # Now check for legitimate restaurant query indicators
    specific_restaurant_indicators = [
        'restaurant', 'restaurants', 'dining', 'where to eat',
        'best places to eat', 'recommendations', 'suggest', 'recommend',
        'seafood', 'vegetarian', 'vegan', 'fine dining', 'rooftop',
        'bosphorus view', 'waterfront', 'turkish cuisine', 'ottoman cuisine',
        'meze', 'kebab', 'breakfast place', 'lunch spot', 'dinner',
        'near', 'in sultanahmet', 'in beyoglu', 'in kadikoy', 'in galata'
    ]
    
    has_specific_indicators = any(indicator in user_lower for indicator in specific_restaurant_indicators)
    
    # Must have specific indicators AND be reasonably detailed (more than 2 words for most cases)
    if has_specific_indicators and len(user_input.split()) >= 2:
        return True
    
    # Allow some specific single-word cuisine queries
    cuisine_words = ['seafood', 'vegetarian', 'vegan', 'italian', 'japanese', 'chinese']
    if user_lower in cuisine_words:
        return True
    
    return False

def create_clarification_response(user_input: str) -> str:
    """Create a response asking for clarification on vague queries"""
    user_lower = user_input.lower()
    
    if user_lower in ['food', 'eat', 'hungry']:
        return """I'd love to help you find great food in Istanbul! To give you the best recommendations, could you tell me:

• What type of cuisine are you interested in? (Turkish, Italian, seafood, vegetarian, etc.)
• Which area of Istanbul are you in or planning to visit?
• What's your preferred dining style? (Fine dining, casual, street food, etc.)
• Any specific dietary requirements?

For example, you could ask: "vegetarian restaurants in Sultanahmet" or "seafood places with Bosphorus view" """

    elif 'where eat' in user_lower:
        return """I can help you find excellent restaurants in Istanbul! To provide better suggestions, please let me know:

• What area of Istanbul? (Sultanahmet, Beyoğlu, Kadıköy, etc.)
• What type of food do you prefer?
• Any budget preferences?

Try asking something like: "best Turkish restaurants in Galata" or "budget-friendly places in Kadıköy" """

    else:
        return """To help you find the perfect restaurant in Istanbul, I'd need a bit more information:

• What type of cuisine interests you?
• Which district or area are you looking at?
• What's the occasion? (casual meal, special dinner, quick bite, etc.)

Feel free to ask something specific like: "romantic restaurants with Bosphorus view" or "traditional Turkish breakfast places in Sultanahmet" """

# Enhanced restaurant advice with Google Maps integration
def get_live_restaurant_recommendations(user_input: str) -> Optional[str]:
    """Get live restaurant recommendations from Google Maps API"""
    try:
        # Try to get live data from Google Places API
        places_client = GooglePlacesClient()
        
        # Parse query to determine search parameters
        user_lower = user_input.lower()
        
        search_params = {
            "location": "Istanbul, Turkey",
            "radius": 5000,  # 5km radius
            "min_rating": 3.5
        }
        
        # Customize search based on query type
        if "vegetarian" in user_lower or "vegan" in user_lower:
            search_params["keyword"] = "vegetarian restaurant Istanbul"
        elif "seafood" in user_lower or "fish" in user_lower:
            search_params["keyword"] = "seafood restaurant Istanbul"
        elif "fine dining" in user_lower or "luxury" in user_lower:
            search_params["keyword"] = "fine dining restaurant Istanbul"
        elif "traditional" in user_lower or "turkish" in user_lower:
            search_params["keyword"] = "traditional Turkish restaurant Istanbul"
        elif "bosphorus" in user_lower or "view" in user_lower:
            search_params["keyword"] = "restaurant Bosphorus view Istanbul"
        elif "rooftop" in user_lower:
            search_params["keyword"] = "rooftop restaurant Istanbul"
        elif "breakfast" in user_lower or "kahvaltı" in user_lower:
            search_params["keyword"] = "Turkish breakfast Istanbul"
        elif any(district in user_lower for district in ['sultanahmet', 'beyoğlu', 'beyoglu', 'galata', 'kadıköy', 'kadikoy']):
            district = next(d for d in ['sultanahmet', 'beyoğlu', 'beyoglu', 'galata', 'kadıköy', 'kadikoy'] if d in user_lower)
            search_params["keyword"] = f"restaurant {district} Istanbul"
        else:
            search_params["keyword"] = "restaurant Istanbul"
        
        # Get results from Google Maps
        results = places_client.search_restaurants(**search_params)
        
        if results.get("results"):
            return format_google_restaurants_response(results["results"], user_input)
        else:
            return None  # Fall back to static response
            
    except Exception as e:
        print(f"⚠️ Error getting live restaurant data: {e}")
        return None  # Fall back to static response

def format_google_restaurants_response(restaurants: list, query_type: str) -> str:
    """Format Google Maps restaurant results into a clean response with comprehensive location enhancement"""
    if not restaurants:
        return None
    
    # Try to enhance with comprehensive location context
    try:
        from comprehensive_location_enhancer import enhance_response_comprehensively
        
        # Create basic restaurant response first
        basic_response = _create_basic_restaurant_response(restaurants, query_type)
        
        # Enhance with comprehensive location information
        enhanced_response = enhance_response_comprehensively(query_type, basic_response)
        
        # ALWAYS use the basic Google Maps response for restaurants - don't override with templates
        print("� Using live Google Maps restaurant data with location context")
        return basic_response
            
    except ImportError:
        print("⚠️ Comprehensive location enhancer not available, using basic response")
        return _create_basic_restaurant_response(restaurants, query_type)

def _create_basic_restaurant_response(restaurants: list, query_type: str) -> str:
    """Create comprehensive restaurant response with live Google Maps data and location context"""
    # Take top 5 restaurants for more focused results
    top_restaurants = restaurants[:5]
    
    # Import location enhancer for neighborhood context
    try:
        from location_enhancer import location_enhancer
    except ImportError:
        location_enhancer = None
    
    query_lower = query_type.lower()
    
    # Detect location from query
    detected_location = None
    if location_enhancer:
        detected_location = location_enhancer.detect_location_from_query(query_type)
    
    # Create location-specific header
    location_context = ""
    if detected_location and location_enhancer:
        location_data = location_enhancer.location_database.get(detected_location.lower())
        if location_data:
            location_context = f"🏛️ **{location_data.name} Dining Scene**\n"
            location_context += f"{location_data.description}\n"
            location_context += f"Known for: {', '.join(location_data.dining_specialties[:3])}\n\n"
    
    # Build response with context-specific intro
    response = location_context
    
    if "vegetarian" in query_lower:
        response += "🌱 **Top Vegetarian Restaurants:**\n\n"
    elif "seafood" in query_lower or "fish" in query_lower:
        response += "🐟 **Fresh Seafood Restaurants:**\n\n"
    elif "kebab" in query_lower or "kebap" in query_lower:
        response += "🥙 **Authentic Kebab Houses:**\n\n"
    elif "breakfast" in query_lower or "kahvaltı" in query_lower:
        response += "☕ **Traditional Turkish Breakfast:**\n\n"
    elif "view" in query_lower or "bosphorus" in query_lower:
        response += "🌅 **Restaurants with Stunning Views:**\n\n"
    elif "romantic" in query_lower:
        response += "❤️ **Romantic Dining Experiences:**\n\n"
    elif "fine dining" in query_lower or "upscale" in query_lower:
        response += "🍾 **Fine Dining Establishments:**\n\n"
    else:
        response += "🍽️ **Top Restaurant Recommendations:**\n\n"
    
    for i, restaurant in enumerate(top_restaurants, 1):
        name = restaurant.get("name", "Restaurant")
        rating = restaurant.get("rating", 0)
        user_ratings_total = restaurant.get("user_ratings_total", 0)
        vicinity = restaurant.get("vicinity", "Istanbul")
        price_level = restaurant.get("price_level", 0)
        types = restaurant.get("types", [])
        
        # Get more detailed information from place details
        place_id = restaurant.get("place_id")
        
        # Format price level with more detail
        if price_level == 1:
            price_indicator = "💰 Budget-friendly (₺₺)"
        elif price_level == 2:
            price_indicator = "💰💰 Moderate (₺₺₺)"
        elif price_level == 3:
            price_indicator = "💰💰💰 Upscale (₺₺₺₺)"
        elif price_level == 4:
            price_indicator = "💰💰💰💰 Fine dining (₺₺₺₺₺)"
        else:
            price_indicator = "💰 Price varies"
        
        # Format rating with more precision
        if rating >= 4.7:
            stars = "⭐⭐⭐⭐⭐ Exceptional"
        elif rating >= 4.4:
            stars = "⭐⭐⭐⭐⭐ Excellent"
        elif rating >= 4.0:
            stars = "⭐⭐⭐⭐ Very Good"
        elif rating >= 3.5:
            stars = "⭐⭐⭐⭐ Good"
        else:
            stars = "⭐⭐⭐ Decent"
        
        # Add cuisine type context
        cuisine_type = ""
        if "restaurant" in types:
            # Try to determine cuisine from name or location
            name_lower = name.lower()
            if any(word in name_lower for word in ['kebap', 'kebab', 'döner', 'çiğ köfte']):
                cuisine_type = "🥙 Turkish Kebab"
            elif any(word in name_lower for word in ['balık', 'fish', 'seafood']):
                cuisine_type = "🐟 Seafood"
            elif any(word in name_lower for word in ['meze', 'rakı', 'taverna']):
                cuisine_type = "🍽️ Turkish Meze"
            elif any(word in name_lower for word in ['kahvaltı', 'breakfast']):
                cuisine_type = "☕ Turkish Breakfast"
            elif any(word in name_lower for word in ['pizza', 'italian']):
                cuisine_type = "🍕 Italian"
            elif any(word in name_lower for word in ['sushi', 'japanese']):
                cuisine_type = "🍣 Japanese"
            else:
                cuisine_type = "🍽️ Turkish Cuisine"
        
        response += f"**{i}. {name}**\n"
        response += f"   {stars} ({rating}/5.0 • {user_ratings_total:,} reviews)\n"
        response += f"   {price_indicator}\n"
        if cuisine_type:
            response += f"   {cuisine_type}\n"
        response += f"   📍 {vicinity}\n"
        
        # Add location-specific context if available
        if detected_location and location_enhancer:
            location_data = location_enhancer.location_database.get(detected_location.lower())
            if location_data and location_data.walking_distances:
                # Find relevant walking distance
                for route, distance in location_data.walking_distances.items():
                    if any(landmark in route.lower() for landmark in ['square', 'station', 'mosque', 'palace']):
                        response += f"   🚶 {distance} from major attractions\n"
                        break
        
        response += "\n"
    
    # Add location-specific practical tips
    response += "💡 **Local Dining Tips:**\n"
    
    if detected_location and location_enhancer:
        location_data = location_enhancer.location_database.get(detected_location.lower())
        if location_data:
            # Add location-specific tips
            for tip in location_data.practical_tips[:2]:
                if any(word in tip.lower() for word in ['restaurant', 'dining', 'book', 'reservation']):
                    response += f"• {tip}\n"
            
            # Add neighborhood-specific dining advice
            if location_data.dining_specialties:
                response += f"• This area is known for: {', '.join(location_data.dining_specialties[:2])}\n"
    
    # Add general Istanbul dining tips
    response += "• Reservations recommended for dinner (7-9 PM peak hours)\n"
    response += "• Tipping 10-15% is standard for good service\n"
    response += "• Try 'meze' (appetizers) for authentic Turkish dining experience\n"
    response += "• Many restaurants offer excellent lunch specials\n\n"
    
    # Add transportation info if location detected
    if detected_location and location_enhancer:
        location_data = location_enhancer.location_database.get(detected_location.lower())
        if location_data:
            response += "🚇 **Getting There:**\n"
            if location_data.metro_stations:
                response += f"• Metro: {', '.join(location_data.metro_stations)}\n"
            if location_data.tram_stops:
                response += f"• Tram: {', '.join(location_data.tram_stops)}\n"
            if location_data.ferry_terminals:
                response += f"• Ferry: {', '.join(location_data.ferry_terminals)}\n"
            response += "\n"
    
    return clean_response_formatting(response)

def clean_response_formatting(response: str) -> str:
    """Clean up response formatting to remove double asterisks and improve aesthetics"""
    if not response:
        return response
    
    # Fix the ugly double asterisks issue
    response = response.replace("** **", "")
    response = response.replace("**  **", "")
    response = response.replace("**   **", "")
    response = response.replace("****", "")
    
    # Remove extra spacing around markdown formatting
    response = re.sub(r'\*\*\s+\*\*', '', response)
    response = re.sub(r'\*\*\s*\*\*', '', response)
    
    # Fix bullet point formatting  
    response = response.replace("•  **", "• **")
    response = response.replace("- **", "• **")
    
    # Ensure consistent spacing after headers
    response = re.sub(r'\*\*:\s*\n', '**:\n', response)
    
    # Clean up multiple newlines but preserve structure
    response = re.sub(r'\n\n\n\n+', '\n\n', response)
    response = re.sub(r'\n\n\n', '\n\n', response)
    
    # Ensure proper emoji spacing
    response = re.sub(r'🍽️\s*\*\*', '🍽️ **', response)
    response = re.sub(r'💡\s*\*\*', '💡 **', response)
    response = re.sub(r'🗺️\s*\*', '🗺️ *', response)
    
    return response.strip()

def create_enhanced_transportation_fallback(user_input: str) -> str:
    """Enhanced transportation fallback responses with better formatting"""
    user_lower = user_input.lower()
    
    # Metro system specific
    if 'metro' in user_lower and not any(word in user_lower for word in ['bus', 'ferry']):
        return """🚇 Istanbul Metro System Guide

🗺️ **Metro Lines Overview:**

• **M1A (Red Line):** Yenikapı ↔ Atatürk Airport
• **M1B (Red Line):** Yenikapı ↔ Kirazlı  
• **M2 (Green Line):** Yenikapı ↔ Hacıosman
• **M3 (Blue Line):** Kirazlı ↔ Başakşehir/Kayaşehir
• **M4 (Pink Line):** Kadıköy ↔ Sabiha Gökçen Airport
• **M5 (Purple Line):** Üsküdar ↔ Çekmeköy
• **M6 (Brown Line):** Levent ↔ Boğaziçi University
• **M7 (Light Blue):** Mecidiyeköy ↔ Mahmutbey

🎫 **Using the Metro:**
• Get **Istanbulkart** from machines or kiosks
• Valid for metro, bus, tram, ferry, metrobus
• Tap card at entry and exit turnstiles
• Keep ticket/card until you exit

⏰ **Operating Hours:**
• **Daily:** 6:00 AM - 12:30 AM (00:30)
• **Frequency:** Every 3-5 minutes during peak hours
• **Peak hours:** 7-9 AM, 5-7 PM (avoid if possible)

🚇 **Key Connections:**
• **Yenikapı:** M1, M2 interchange + ferry terminal
• **Vezneciler:** M2 line for Sultanahmet area
• **Şişli-Mecidiyeköy:** M2, M7 interchange
• **Kadıköy:** M4 line terminus + ferry connections

💡 **Metro Tips:**
• Download "Metro Istanbul" app for real-time info
• Stand right on escalators, walk left
• Priority seating for elderly, disabled, pregnant
• Clean and air-conditioned stations"""

    # Ferry system specific  
    elif 'ferry' in user_lower:
        return """⛴️ Istanbul Ferry System Guide

🌊 **Ferry Routes & Terminals:**

• **Bosphorus Line:** Eminönü ↔ Üsküdar ↔ Beşiktaş ↔ Sarıyer
• **Golden Horn:** Eminönü ↔ Eyüp ↔ Sütlüce
• **Kadıköy Lines:** Kadıköy ↔ Karaköy ↔ Eminönü ↔ Beşiktaş
• **Princess Islands:** Kabataş/Bostancı ↔ Adalar

🎫 **Ferry Tickets:**
• Use **Istanbulkart** - same card as metro/bus
• Cash tickets available at terminals
• Discounts for students, seniors

⏰ **Schedule & Frequency:**
• **Peak hours:** Every 15-20 minutes
• **Off-peak:** Every 30-45 minutes  
• **First ferry:** Around 6:30 AM
• **Last ferry:** Around 9-10 PM (varies by route)

🏛️ **Major Ferry Terminals:**
• **Eminönü:** Historic peninsula, near Spice Bazaar
• **Karaköy:** Galata area, near Galata Tower
• **Kadıköy:** Asian side, vibrant local area
• **Beşiktaş:** European side, Dolmabahçe Palace area
• **Üsküdar:** Asian side, historic mosques

🌅 **Scenic Routes:**
• **Bosphorus tour ferries** - 2-hour round trips
• **Sunset ferries** - Best views 5-7 PM
• **Weekend long cruises** - Full Bosphorus to Black Sea

💡 **Ferry Tips:**
• Arrive 10 minutes early during peak times
• Upper deck offers best views (weather permitting)  
• Ferries are slower but more scenic than bridges
• Great way to avoid traffic between continents"""

    # Transportation costs and cards
    elif any(word in user_lower for word in ['istanbulkart', 'card', 'cost', 'price']):
        return """🎫 Istanbul Transport Cards & Costs

💳 **Istanbulkart - Your Essential Transport Card:**
• **Card cost:** ₺13 (one-time purchase)
• **Where to buy:** Metro stations, ferry terminals, kiosks, some shops
• **Works on:** Metro, bus, tram, ferry, metrobus, funicular
• **Top-up:** Minimum ₺5, maximum ₺300

💰 **Transportation Costs (2024):**
• **Single ride:** ₺5-7 for most transport
• **Transfers:** Discounted within 2 hours
• **Daily maximum:** Around ₺30 per day
• **Student discount:** 50% off with student card

🚌 **Transport Options & Costs:**
• **Metro/Tram:** ₺5 per ride
• **Bus:** ₺5 per ride
• **Ferry:** ₺7-10 depending on route
• **Metrobus:** ₺6 per ride
• **Taxi:** ₺15 starting fare + ₺3 per km

💡 **Money-Saving Tips:**
• **Transfer discounts:** Use within 2 hours for reduced fare
• **Multiple rides:** Card shared among family (each person taps)
• **Monthly pass:** Available for regular commuters
• **Student rates:** Bring student ID for discounted card
• **Tourist pass:** Special tourist cards available at airports

⚠️ **Important Notes:**
• Keep minimum ₺5 balance on card
• Card can be used by multiple people (each taps separately)
• Lost cards can be replaced with remaining balance
• Refunds available at main stations"""

    # General transportation
    else:
        return """🚌 Getting Around Istanbul

🎫 **Istanbulkart - Your Transport Key:**
• **Essential** for all public transport
• Available at: Metro stations, ferry terminals, kiosks
• Works on: Metro, bus, tram, ferry, metrobus, funicular
• **Cost:** ₺13 card + credit you add

🚇 **Transport Options:**

• **Metro:** Fast, clean, air-conditioned underground system
• **Metrobus:** Dedicated bus lanes, connects Europe-Asia quickly  
• **Regular Buses:** Extensive network, can be crowded
• **Trams:** Historic trams in Beyoğlu, modern trams elsewhere
• **Ferries:** Scenic water transport, slower but beautiful views
• **Funicular:** Tünel (historic), cable cars to high areas

🗺️ **Key Routes for Tourists:**
• **Airport to city:** M1 metro line or Havaist bus
• **Sultanahmet:** Tram T1 or metro M2 to Vezneciler
• **Galata Tower:** M2 to Şişhane or funicular from Karaköy
• **Asian side:** Ferry from Eminönü/Karaköy to Kadıköy/Üsküdar

⏰ **Operating Hours:**
• **Metro/Bus:** 6:00 AM - 12:30 AM
• **Ferries:** 6:30 AM - 9:00 PM (varies by route)  
• **Night buses:** Limited service after midnight

💰 **Costs (with Istanbulkart):**
• **Single ride:** ₺5-7 for most transport
• **Transfers:** Discounted within 2 hours
• **Daily maximum:** Around ₺30

🚖 **Taxis & Ride Apps:**
• **Yellow taxis:** Use meter or negotiate fare
• **BiTaksi:** Popular local ride app
• **Uber:** Limited availability
• **Typical costs:** ₺20-50 for cross-city trips

💡 **Transport Tips:**
• Download "Moovit" app for real-time directions
• Rush hours: 7-9 AM, 5-7 PM - expect crowds
• Keep Istanbulkart topped up - minimum ₺5 balance
• Ferry rides offer best city views - highly recommended"""

def detect_ambiguity(user_input: str) -> bool:
    """
    Detect if a user query contains ambiguous terms that could refer to multiple places or concepts.
    """
    user_lower = user_input.lower()
    
    # Ambiguous location terms
    ambiguous_terms = [
        'bridge',      # Could be Golden Horn, Bosphorus, Galata Bridge
        'palace',      # Could be Topkapi, Dolmabahce, Beylerbeyi
        'mosque',      # Many mosques without specific area
        'tower',       # Galata Tower, Maiden's Tower, etc.
        'market',      # Grand Bazaar, Spice Bazaar, local markets
        'museum',      # Many museums without location
        'university',  # Multiple universities
        'airport',     # IST vs SAW
        'old city',    # Could refer to different historic areas
        'downtown',    # No clear downtown in Istanbul
        'the square',  # Taksim, Sultanahmet, etc.
        'waterfront',  # Multiple waterfront areas
        'restaurant',  # When no cuisine/location specified
        'hotel'        # When no area specified
    ]
    
    # Check for ambiguous terms without specific qualifiers
    for term in ambiguous_terms:
        if term in user_lower:
            # Check if there are qualifying terms that make it less ambiguous
            if term == 'bridge' and any(qualifier in user_lower for qualifier in ['golden horn', 'bosphorus', 'galata', '15 july', 'fatih sultan mehmet']):
                continue
            elif term == 'palace' and any(qualifier in user_lower for qualifier in ['topkapi', 'dolmabahce', 'beylerbeyi']):
                continue
            elif term == 'mosque' and any(qualifier in user_lower for qualifier in ['blue mosque', 'hagia sophia', 'sultan ahmed', 'suleymaniye', 'ortakoy']):
                continue
            elif term == 'tower' and any(qualifier in user_lower for qualifier in ['galata', 'maiden', 'beyazit']):
                continue
            elif term == 'market' and any(qualifier in user_lower for qualifier in ['grand bazaar', 'spice bazaar', 'egyptian bazaar', 'kapali carsi']):
                continue
            elif term == 'airport' and any(qualifier in user_lower for qualifier in ['ist', 'saw', 'istanbul', 'sabiha']):
                continue
            else:
                return True
    
    # Check for location queries without specific area
    location_patterns = [
        r'near me',
        r'closest to me',
        r'around here',
        r'in the area',
        r'nearby'
    ]
    
    for pattern in location_patterns:
        if re.search(pattern, user_lower):
            return True
    
    return False

def detect_context_dependency(user_input: str) -> bool:
    """
    Detect if a query depends on context that the user hasn't provided.
    """
    user_lower = user_input.lower()
    
    # Context-dependent phrases
    context_dependent = [
        'how do i get there',
        'how far is it',
        'what time does it close',
        'is it open',
        'how much does it cost',
        'what\'s the best way',
        'which one is better',
        'what should i choose',
        'is it worth it',
        'how long does it take',
        'when should i go',
        'which is closest'
    ]
    
    for phrase in context_dependent:
        if phrase in user_lower:
            return True
    
    # Questions with pronouns but no clear antecedent
    pronoun_patterns = [
        r'\bit\b',      # "it" without clear reference
        r'\bthey\b',    # "they" without clear reference  
        r'\bthat\b',    # "that" without clear reference
        r'\bthis\b',    # "this" without clear reference
        r'\bthere\b'    # "there" without clear location
    ]
    
    for pattern in pronoun_patterns:
        if re.search(pattern, user_lower):
            # Check if it's at the start of a question (more likely to be ambiguous)
            if re.search(r'^(how|what|where|when|why|is|can|should|will).*' + pattern, user_lower):
                return True
    
    return False

def generate_clarification_prompt(user_input: str) -> str:
    """
    Generate an appropriate clarification prompt based on the type of ambiguity detected.
    """
    user_lower = user_input.lower()
    
    # Bridge clarification
    if 'bridge' in user_lower:
        return """I'd be happy to help with information about bridges in Istanbul! Could you clarify which bridge you're interested in?

**Major Bridges:**
- **Galata Bridge** - Historic bridge over Golden Horn
- **Bosphorus Bridge (15 July Bridge)** - Connects Europe & Asia
- **Fatih Sultan Mehmet Bridge** - Second Bosphorus bridge
- **Golden Horn Bridge** - Modern cable-stayed bridge

Which bridge would you like to know about?"""

    # Palace clarification  
    elif 'palace' in user_lower:
        return """Istanbul has several magnificent palaces! Which one interests you?

**Major Palaces:**
- **Topkapi Palace** - Ottoman imperial palace, Sultanahmet
- **Dolmabahce Palace** - 19th century palace, European side
- **Beylerbeyi Palace** - Summer palace, Asian side

Could you specify which palace you'd like information about?"""

    # Mosque clarification
    elif 'mosque' in user_lower and not any(qualifier in user_lower for qualifier in ['blue', 'hagia sophia', 'sultan ahmed']):
        return """Istanbul has thousands of beautiful mosques! To give you the most helpful information, could you specify:

- **Which area** are you visiting? (Sultanahmet, Beyoglu, etc.)
- **Any particular mosque** you've heard about?
- **What you're looking for** - historic significance, architecture, etc.

Some famous options: Blue Mosque, Suleymaniye Mosque, Ortakoy Mosque"""

    # Tower clarification
    elif 'tower' in user_lower:
        return """Istanbul has several notable towers! Which one are you asking about?

**Popular Towers:**
- **Galata Tower** - Medieval stone tower, panoramic views
- **Maiden's Tower** - Historic tower on Bosphorus
- **Beyazit Tower** - Historic fire tower

Which tower interests you?"""

    # Restaurant clarification
    elif 'restaurant' in user_lower and not any(qualifier in user_lower for qualifier in ['in', 'near', 'turkish', 'seafood']):
        return """I'd love to recommend restaurants! To give you the best suggestions, could you tell me:

- **Which area** of Istanbul? (Sultanahmet, Galata, Kadıköy, etc.)
- **What type of cuisine** are you interested in?
- **Your budget range** - budget-friendly, mid-range, or fine dining?
- **Any dietary preferences** or restrictions?

This will help me suggest the perfect places for you!"""

    # Market clarification
    elif 'market' in user_lower:
        return """Istanbul has several amazing markets! Which type of market are you looking for?

**Major Markets:**
- **Grand Bazaar (Kapalı Çarşı)** - Historic covered market with 4,000 shops
- **Spice Bazaar (Egyptian Bazaar)** - Traditional spices, teas, Turkish delight
- **Galata Bridge Fish Market** - Fresh seafood and fish restaurants
- **Kadıköy Tuesday Market** - Local produce and street food
- **Fatih Wednesday Market** - Traditional neighborhood market

Are you looking for:
- **Spices and traditional products**?
- **Souvenirs and handicrafts**?
- **Fresh food and local produce**?
- **A specific neighborhood market**?

Let me know what interests you most!"""

    # Generic location clarification
    elif any(term in user_lower for term in ['near me', 'closest', 'around here']):
        return """I'd be happy to help you find places nearby! However, I don't have access to your current location. Could you tell me:

- **Which area of Istanbul** are you in or planning to visit?
- **Which district or landmark** are you near?

For example: Sultanahmet, Taksim Square, Galata Tower, Kadıköy, etc.

This will help me give you accurate directions and recommendations!"""

    # Airport clarification
    elif 'airport' in user_lower and not any(qualifier in user_lower for qualifier in ['ist', 'saw', 'istanbul', 'sabiha']):
        return """Istanbul has two main airports. Which one are you referring to?

**Istanbul Airport (IST):**
- New main international airport
- European side, about 35km from city center
- Most international flights arrive here

**Sabiha Gökçen Airport (SAW):**
- Asian side, about 45km from city center  
- Some international and domestic flights
- Often more budget-friendly

Which airport are you asking about?"""

    # Generic clarification
    else:
        return """I'd be happy to help! To give you the most accurate information, could you provide a bit more detail about what specifically you're looking for?

For example:
- Which area of Istanbul interests you?
- What type of experience are you seeking?
- Any specific preferences or requirements?

The more details you share, the better I can assist you!"""
        
def detect_location_confusion(user_input: str) -> tuple[bool, Optional[str]]:
    """Detect if query mentions other cities and needs Istanbul-specific redirection"""
    user_lower = user_input.lower()
    
    # Check for other Turkish cities
    other_cities = ['ankara', 'izmir', 'antalya', 'bursa', 'adana', 'trabzon', 'konya', 'gaziantep']
    mentioned_city = None
    
    for city in other_cities:
        if city in user_lower:
            mentioned_city = city
            break
    
    # If another city is mentioned
    if mentioned_city:
        # But Istanbul is also mentioned - this is a comparison, not confusion
        if 'istanbul' in user_lower:
            return False, None
        
        # Check if it's a general Turkey query that should be Istanbul-focused
        if 'turkey' in user_lower and not 'istanbul' in user_lower:
            if any(word in user_lower for word in ['restaurant', 'food', 'eat', 'museum', 'attraction', 'visit', 'travel']):
                return True, f"I specialize in Istanbul travel advice. While {mentioned_city.title()} is a great city, I can provide detailed information about Istanbul's {', '.join([w for w in ['restaurants', 'attractions', 'museums', 'districts', 'transportation'] if any(k in user_lower for k in [w[:-1], w])])}. Would you like to know about Istanbul instead?"
        
        # Direct query about another city
        elif any(word in user_lower for word in ['restaurant', 'food', 'museum', 'attraction', 'hotel', 'transport', 'visit']):
            return True, f"I'm specialized in Istanbul tourism and can't provide specific information about {mentioned_city.title()}. However, I can offer comprehensive advice about Istanbul's attractions, restaurants, transportation, and districts. Would you like to explore what Istanbul has to offer?"
    
    return False, None
