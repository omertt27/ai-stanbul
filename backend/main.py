# --- Standard Library Imports ---
import sys
import os
import re
import asyncio
import json
import time
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import traceback

# --- Third-Party Imports ---
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process

# Load environment variables first, before any other imports
load_dotenv()

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
    
    # PHASE 1: Remove explicit currency amounts (all formats)
    text = re.sub(r'\$\d+[\d.,]*', '', text)      # $20, $15.50
    text = re.sub(r'€\d+[\d.,]*', '', text)       # €20, €15.50
    text = re.sub(r'₺\d+[\d.,]*', '', text)       # ₺20, ₺15.50
    text = re.sub(r'\d+₺', '', text)              # 50₺
    text = re.sub(r'\d+\s*(?:\$|€|₺)', '', text)  # 20$, 50 €
    text = re.sub(r'(?:\$|€|₺)\s*\d+[\d.,]*', '', text)  # $ 20, € 15.50
    
    # PHASE 2: Remove currency words and phrases
    text = re.sub(r'\d+\s*(?:lira|euro|euros|dollar|dollars|pound|pounds)s?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:turkish\s+)?lira\s*\d+', '', text, flags=re.IGNORECASE)
    
    # PHASE 3: Remove cost-related phrases with amounts
    cost_patterns = [
        r'(?:costs?|prices?|fees?)\s*:?\s*(?:around\s+|about\s+|approximately\s+)?\$?\€?₺?\d+[\d.,]*',
        r'(?:entrance|admission|ticket)\s*(?:cost|price|fee)s?\s*:?\s*\$?\€?₺?\d+',
        r'(?:starting|starts)\s+(?:from|at)\s+\$?\€?₺?\d+',
        r'(?:only|just)\s+\$?\€?₺?\d+[\d.,]*',
        r'(?:per\s+person|each|pp)\s*:?\s*\$?\€?₺?\d+',
        r'\$?\€?₺?\d+[\d.,]*\s*(?:per\s+person|each|pp)',
    ]
    
    for pattern in cost_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # PHASE 4: Remove money emojis and currency symbols
    text = re.sub(r'💰|💵|💴|💶|💷', '', text)
    text = re.sub(r'[\$€₺£¥₹₽₴₦₱₩₪₨]', '', text)
    
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

def get_gpt_response(user_input: str, session_id: str) -> Optional[str]:
    """Generate response using OpenAI GPT for queries we can't handle with database/hardcoded responses"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not OpenAI_available or not openai_api_key or OpenAI is None:
            return None
        
        # Create OpenAI client
        client = OpenAI(api_key=openai_api_key, timeout=30.0, max_retries=2)
        
        # Create a specialized prompt for Istanbul tourism
        system_prompt = """You are an expert Istanbul travel assistant. Provide helpful, accurate information about Istanbul tourism, culture, attractions, restaurants, and travel tips. Keep responses conversational but informative. Focus specifically on Istanbul, Turkey.

Guidelines:
- Be enthusiastic and helpful about Istanbul
- Provide specific, actionable advice when possible  
- Include practical details like locations, timing, or tips
- Keep responses concise but comprehensive (200-400 words)
- Use emojis sparingly for better readability
- Always relate answers back to Istanbul specifically
- If asked about something not related to Istanbul tourism, politely redirect to Istanbul topics

Current specialties: Museums, art galleries, cultural sites, historical information, local experiences, food culture, shopping, nightlife, transportation, and general travel advice for Istanbul."""

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question about Istanbul: {user_input}"}
            ],
            max_tokens=500,
            temperature=0.7,
            timeout=25
        )
        
        gpt_response = response.choices[0].message.content
        if gpt_response:
            gpt_response = gpt_response.strip()
            print(f"✅ GPT response generated successfully for: {user_input[:50]}...")
            return gpt_response
        else:
            print(f"⚠️ GPT returned empty content for: {user_input[:50]}...")
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
    """Post-LLM cleanup pass to catch any remaining pricing while preserving readable formatting"""
    if not text:
        return text
    
    # Catch any remaining pricing patterns that might have been generated
    post_patterns = [
        r'\b(?:costs?|prices?)\s+(?:around\s+|about\s+)?\d+',
        r'\d+\s*(?:lira|euro|dollar)s?\s*(?:per|each|only)',
        r'(?:(?:only|just|around)\s+)?\d+\s*(?:lira|euro|dollar)',
        r'budget\s*:\s*\d+',
        r'price\s+range\s*:\s*\d+',
        r'\b\d+\s*(?:-|to)\s*\d+\s*(?:lira|euro|dollar)',
    ]
    
    for pattern in post_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove any remaining standalone numbers that might be pricing (but be more conservative)
    text = re.sub(r'\b\d{2,3}\s*(?=\s|$|[.,!?])', '', text)
    
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
    
    # Keywords that suggest the query needs more nuanced/detailed answers
    gpt_suitable_keywords = [
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
        
        # Complex transportation questions
        'transportation options', 'transport system', 'public transport guide', 
        'transportation tips', 'transport planning', 'transport costs', 'transport passes',
        'airport transfer', 'airport transport', 'airport connection',
        'late night transport', 'early morning transport', 'transport schedule',
        'accessibility transport', 'disabled transport', 'wheelchair accessible',
        'transport for families', 'transport with luggage', 'transport safety',
        'cheapest transport', 'fastest transport', 'most comfortable transport',
        'transport during rush hour', 'avoiding traffic', 'transport strikes',
        
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
    
    user_lower = user_input.lower()
    
    # Check if query contains GPT-suitable keywords
    has_gpt_keywords = any(keyword in user_lower for keyword in gpt_suitable_keywords)
    
    # Also use GPT for questions (queries with question words)
    question_words = ['what', 'why', 'how', 'when', 'where', 'which', 'who']
    has_question_words = any(word in user_lower for word in question_words)
    
    # Use GPT for longer, more complex queries (likely to need detailed answers)
    is_complex_query = len(user_input.split()) > 5
    
    return has_gpt_keywords or (has_question_words and is_complex_query)

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

# Add CORS middleware with dynamic origins
CORS_ORIGINS = [
    # Development ports
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:3003",
    "http://localhost:3004",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
    "http://127.0.0.1:3003",
    "http://127.0.0.1:3004",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:5176",
    # Production frontend URLs
    "https://aistanbul.vercel.app",
    "https://aistanbul-fdsqdpks5-omers-projects-3eea52d8.vercel.app",
    "https://aistanbul-dz2rju4mf-omers-projects-3eea52d8.vercel.app",
    "https://aistanbul-e5rcj5qm3-omers-projects-3eea52d8.vercel.app",
    # Allow file:// protocol (null origin) for admin dashboard
    "null",
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

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

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

def create_fallback_response(user_input, places):
    """Create intelligent fallback responses when OpenAI API is unavailable"""
    user_input_lower = user_input.lower()
    print(f"DEBUG: create_fallback_response called with input: '{user_input_lower}'")
    
    # District-specific responses
    if 'kadıköy' in user_input_lower or 'kadikoy' in user_input_lower:
        response = """Kadıköy District Guide

Kadıköy is Istanbul's vibrant Asian side cultural hub!

Top Attractions in Kadıköy:
- Moda Seaside - Waterfront promenade with cafes
- Kadıköy Market - Bustling local market 
- Fenerbahçe Park - Green space by the sea
- Barlar Sokağı - Famous bar street for nightlife
- Yeldeğirmeni - Artsy neighborhood with murals
- Haydarpaşa Train Station - Historic Ottoman architecture

What to Do:
- Stroll along Moda seafront
- Browse vintage shops and bookstores
- Try local street food at the market
- Experience authentic Turkish neighborhood life
- Take ferry to European side for Bosphorus views

Getting There:
- Ferry from Eminönü or Beşiktaş (most scenic)
- Metro from Ayrılık Çeşmesi station
- Bus connections from major areas

Kadıköy offers an authentic Istanbul experience away from tourist crowds!"""
        return enhance_ai_response_formatting(clean_text_formatting(response))
    
    elif 'sultanahmet' in user_input_lower:
        response = """Sultanahmet Historic District

Istanbul's historic heart with UNESCO World Heritage sites!

Major Attractions:
- Hagia Sophia - Former Byzantine church and Ottoman mosque
- Blue Mosque - Stunning Ottoman architecture  
- Topkapi Palace - Ottoman imperial palace
- Grand Bazaar - Historic covered market
- Basilica Cistern - Underground Byzantine cistern
- Hippodrome - Ancient Roman chariot racing arena

Walking Distance Sites:
- Gülhane Park - Historic park
- Turkish and Islamic Arts Museum
- Soğukçeşme Street - Ottoman houses
- Archaeological Museums

Tips:
- Start early to avoid crowds
- Comfortable walking shoes essential
- Many sites within 10-minute walk
- Free WiFi in most cafes and museums"""
        return enhance_ai_response_formatting(clean_text_formatting(response))
    
    # Check for restaurant queries in Turkish before falling back to district info
    elif 'beyoğlu' in user_input_lower or 'beyoglu' in user_input_lower:
        # Check if this is a restaurant query specifically
        turkish_restaurant_keywords = ['restoran', 'restoranlar', 'en iyi', 'iyi', 'yemek', 'lokanta', 'tavsiye']
        print(f"DEBUG: Checking Turkish restaurant keywords in '{user_input_lower}'")
        print(f"DEBUG: Keywords: {turkish_restaurant_keywords}")
        found_keywords = [keyword for keyword in turkish_restaurant_keywords if keyword in user_input_lower]
        print(f"DEBUG: Found keywords: {found_keywords}")
        if any(keyword in user_input_lower for keyword in turkish_restaurant_keywords):
            response = """**Best Restaurants in Beyoğlu:**

🍽️ **Pandeli** - Historic Ottoman restaurant in Eminönü serving traditional Turkish cuisine
🥘 **Hamdi Restaurant** - Famous for lamb tandoor and kebabs
🍷 **Mikla** - Modern Turkish cuisine with rooftop dining in Pera
🧀 **Karaköy Lokantası** - Nostalgic ambiance in Karaköy, excellent meze
🍕 **360 Istanbul** - Panoramic views in Galata, international cuisine
🥙 **Datli Maya** - Organic and natural ingredients, healthy options

**Area Recommendations:**
• **Galata Tower area** - Romantic dinner spots
• **Istiklal Street** - Quick dining options  
• **Karaköy** - Trendy cafes and restaurants
• **Cihangir** - Bohemian atmosphere, art cafes

**Tip:** Don't forget to make reservations, especially for weekends!"""
        else:
            response = """Beyoğlu Cultural District

Modern Istanbul's cultural and nightlife center!

Key Areas:
- Istiklal Avenue - Main pedestrian street
- Galata Tower - Iconic medieval tower
- Taksim Square - Central meeting point
- Karaköy - Trendy waterfront area
- Cihangir - Bohemian neighborhood

Attractions:
- Galata Tower views
- Pera Museum
- Istanbul Modern Art Museum
- Historic trams on Istiklal
- Rooftop bars and restaurants

Activities:
- Walk the length of Istiklal Avenue
- Take nostalgic tram ride
- Explore art galleries in Karaköy
- Experience Istanbul's nightlife
- Browse vintage shops and bookstores"""
        return enhance_ai_response_formatting(clean_text_formatting(response))
    
    # Food and cuisine questions - check BEFORE history/culture to avoid conflicts
    elif any(word in user_input_lower for word in ['food', 'eat', 'cuisine', 'dish', 'meal', 'breakfast', 'lunch', 'dinner', 'restaurant', 'restaurants', 'tipping', 'tip']):
        # Check for Bosphorus view restaurant queries first
        if any(word in user_input_lower for word in ['bosphorus', 'view', 'waterfront', 'water view', 'sea view', 'scenic']):
            response = """Restaurants with Bosphorus Views

**Ortaköy Area:**
- House Cafe Ortaköy - Waterfront dining with mosque views
- Feriye Palace Restaurant - Ottoman palace setting
- Reina - Upscale nightclub with Bosphorus terrace

**Bebek & Arnavutköy:**
- Bebek Restaurant - Classic seafood with water views
- Arnavutköy Balıkçısı - Traditional fish restaurant
- Lucca Style - Italian cuisine with Bosphorus panorama

**Asian Side:**
- Maiden's Tower Restaurant - Unique tower location
- Çubuklu 29 - Elegant dining with European side views
- Körfez Restaurant - Local favorite in Kanlıca

**Rooftop Options:**
- 360 Istanbul (Beyoğlu) - Panoramic city and water views
- Mikla - Modern Turkish cuisine with skyline views
- Leb-i Derya - Trendy rooftop in Beyoğlu

**Tips:**
- Make reservations for sunset timing
- Dress smart casual for upscale places
- Consider ferry ride to restaurant for full Bosphorus experience"""

        # Check for Grand Bazaar food court queries
        elif any(word in user_input_lower for word in ['grand bazaar', 'kapalıçarşı', 'food court', 'bazaar food']):
            response = """Grand Bazaar Food Experience

**Food Court Options:**
- Traditional Turkish köfte (meatballs)
- Fresh pide (Turkish pizza) 
- Döner kebab stalls
- Turkish delight and baklava
- Turkish tea and coffee

**Nearby Traditional Eateries:**
- Havuzlu Restaurant - Historic restaurant inside the bazaar
- Pandeli Restaurant - Ottoman-era restaurant near Spice Bazaar
- Local çay (tea) houses for breaks

**What to Try:**
- Fresh simit (Turkish bagel)
- Lokum (Turkish delight) - try before buying
- Turkish coffee preparation demonstrations
- Seasonal fruit juices
- Traditional pastries

**Tips:**
- Food court gets busy during lunch hours
- Try samples before purchasing sweets
- Bargaining expected for food items too
- Combine with Spice Bazaar visit for more food options"""

        # Check for Ottoman cuisine queries
        elif any(word in user_input_lower for word in ['ottoman', 'historic', 'traditional', 'authentic', 'palace']):
            response = """Authentic Ottoman Cuisine in Istanbul

**Historic Restaurants:**
- Pandeli Restaurant - Established 1901, serves Ottoman palace recipes
- Hamdi Restaurant - Traditional tandoor cooking methods
- Deraliye Ottoman Palace Cuisine - Recreates historical palace dishes
- Tugra Restaurant - Fine Ottoman dining in Four Seasons hotel

**Signature Ottoman Dishes:**
- Hünkar Beğendi - Lamb stew with eggplant puree
- İmam Bayıldı - Stuffed eggplant (vegetarian)
- Kuzu Tandır - Slow-roasted lamb
- Ottoman-style rice pilaf
- Traditional meze selections
- Turkish coffee service (UNESCO recognized)

**Palace-Style Dining:**
- Çırağan Palace Restaurant - Actual Ottoman palace
- Esma Sultan Mansion - Restored Ottoman waterfront mansion
- Les Ottomans Hotel Restaurant - Ottoman-themed fine dining

**Cultural Experience:**
- Some restaurants offer traditional music
- Authentic Ottoman table setting and service
- Historical recipes dating back centuries"""

        # Check for tipping culture queries  
        elif any(word in user_input_lower for word in ['tipping', 'tip', 'service', 'gratuity', 'how much']):
            response = """Tipping Culture in Istanbul Restaurants

**Standard Tipping:**
- 10-15% is standard for good service
- Round up to nearest 5-10 TL for casual places
- 15-20% for upscale restaurants

**How to Tip:**
- Cash is preferred over adding to card
- Leave tip on table or give directly to server
- Say "Üstü kalsın" (keep the change)

**Service Charge:**
- Some upscale places add 10% service charge
- Check your bill - additional tip not required if service charge included
- Ask "Servis ücreti dahil mi?" (Is service charge included?)

**When NOT to Tip:**
- Fast food places
- Self-service cafes
- If service was genuinely poor

**Cultural Notes:**
- Tipping shows appreciation for good service
- Turkish hospitality culture appreciates recognition
- Don't feel obligated if service was poor"""

        # Check for Turkish breakfast queries
        elif any(word in user_input_lower for word in ['breakfast', 'kahvaltı']):
            response = """Turkish Breakfast Culture (Kahvaltı)

Traditional Turkish breakfast is a feast! Key components:

**Essential Items:**
- Fresh Turkish bread (ekmek)
- Various cheeses (beyaz peynir, kaşar)
- Olives (black and green varieties)
- Turkish tea (çay) - served in small glasses
- Honey and jam varieties
- Tomatoes, cucumbers, peppers
- Börek (flaky pastry)
- Sucuklu yumurta (eggs with Turkish sausage)

**Popular Breakfast Places:**
- Van Kahvaltı Evi (traditional Van breakfast)
- Çiya Sofrası in Kadıköy
- Pandeli Restaurant
- Local breakfast houses in Beyoğlu

**Culture:**
- Breakfast is a leisurely social meal
- Weekends often feature extended family breakfasts
- Tea is essential - coffee is not traditional for breakfast"""
        
        elif any(word in user_input_lower for word in ['bosphorus', 'view', 'waterfront']):
            response = """Restaurants with Bosphorus Views

**Ortaköy Area:**
- House Cafe Ortaköy - Waterfront dining with mosque views
- Feriye Palace Restaurant - Ottoman palace setting
- Reina - Upscale nightclub with Bosphorus terrace

**Bebek & Arnavutköy:**
- Bebek Restaurant - Classic seafood with water views
- Arnavutköy Balıkçısı - Traditional fish restaurant
- Lucca Style - Italian cuisine with Bosphorus panorama

**Asian Side:**
- Maiden's Tower Restaurant - Unique tower location
- Çubuklu 29 - Elegant dining with European side views
- Körfez Restaurant - Local favorite in Kanlıca

**Rooftop Options:**
- 360 Istanbul (Beyoğlu) - Panoramic city and water views
- Mikla - Modern Turkish cuisine with skyline views
- Leb-i Derya - Trendy rooftop in Beyoğlu"""

        elif any(word in user_input_lower for word in ['tipping', 'tip', 'service']):
            response = """Tipping Culture in Istanbul Restaurants

**Standard Tipping:**
- 10-15% is standard for good service
- Round up to nearest 5-10 TL for casual places
- 15-20% for upscale restaurants

**How to Tip:**
- Cash is preferred over adding to card
- Leave tip on table or give directly to server
- Say "Üstü kalsın" (keep the change)

**Service Charge:**
- Some upscale places add 10% service charge
- Check your bill - additional tip not required if service charge included
- Ask "Servis ücreti dahil mi?" (Is service charge included?)

**When NOT to Tip:**
- Fast food places
- Self-service cafes
- If service was genuinely poor

**Cultural Notes:**
- Tipping shows appreciation for good service
- Turkish hospitality culture appreciates recognition
- Don't feel obligated if service was poor"""

        else:
            response = """Turkish Cuisine in Istanbul

I can provide live restaurant recommendations using Google Maps. Please ask for a specific type of restaurant or cuisine, and I'll fetch the latest options for you!

Must-try Turkish dishes include döner kebab, simit, balık ekmek, midye dolma, iskender kebab, manti, lahmacun, börek, baklava, Turkish delight, künefe, Turkish tea, ayran, and raki.

For restaurant recommendations, please specify your preference (e.g., 'seafood in Kadıköy')."""
        
        return enhance_ai_response_formatting(clean_text_formatting(response))
    
    # History and culture questions
    elif any(word in user_input_lower for word in ['history', 'historical', 'culture', 'byzantine', 'ottoman']):
        response = """Istanbul's Rich History

Istanbul has over 2,500 years of history! Here are key highlights:

Byzantine Era (330-1453 CE):
- Originally called Constantinople
- Hagia Sophia built in 537 CE
- Capital of Byzantine Empire

Ottoman Era (1453-1922):
- Conquered by Mehmed II in 1453
- Became capital of Ottoman Empire
- Blue Mosque, Topkapi Palace built

Modern Istanbul:
- Turkey's largest city with 15+ million people
- Spans Europe and Asia across the Bosphorus
- UNESCO World Heritage sites in historic areas

Would you like to know about specific historical sites or districts?"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # Transportation questions
    elif any(word in user_input_lower for word in ['transport', 'metro', 'bus', 'ferry', 'taxi', 'getting around']):
        # Check for metro vs metrobus comparison first
        if any(word in user_input_lower for word in ['metrobus', 'difference', 'vs', 'versus', 'compare']):
            response = """Metro vs Metrobus in Istanbul

**Metro (M Lines):**
- Underground rapid transit system
- M1: Airport to city center
- M2: North-south European side
- M3, M4: Various city connections
- Clean, air-conditioned, frequent
- Stations have elevators and escalators

**Metrobus (BRT):**
- Bus Rapid Transit system
- Dedicated bus lanes on highways
- Connects European and Asian sides
- Faster than regular buses
- More crowded, especially rush hours
- Runs above ground

**Key Differences:**
- Metro: Underground, cleaner, more comfortable
- Metrobus: Faster for long distances, crosses continents
- Metro: Better for central areas
- Metrobus: Good for cross-city travel
- Both use same Istanbul Card (Istanbulkart)"""
        
        # Check for app recommendations
        elif any(word in user_input_lower for word in ['app', 'mobile', 'smartphone', 'best app']):
            response = """Best Transportation Apps for Istanbul

**Essential Apps:**

**Moovit:**
- Real-time public transport info
- Route planning with multiple options
- Offline maps available
- Works in English

**Google Maps:**
- Excellent for walking directions
- Public transport integration
- Real-time traffic updates
- Works offline with downloaded maps

**BiTaksi:**
- Official taxi app
- Track your ride and driver
- Cashless payment options
- Available in English

**İstanbul Kart Mobile:**
- Check card balance
- Add money to your Istanbul Card
- Find nearest top-up points

**Uber:**
- Alternative to taxis
- Fixed pricing
- English interface
- Available in most areas"""
        
        # Check for taxi cost/meter queries
        elif any(word in user_input_lower for word in ['taxi', 'cost', 'price', 'meter', 'how much']):
            response = """Taxi Costs and Tips in Istanbul

**Pricing:**
- Base fare: 4.10 TL (day) / 4.50 TL (night)
- Per km: 2.70 TL (day) / 3.24 TL (night)
- Waiting time: 32.40 TL per hour
- Night rate: 00:00-06:00

**Distance Examples:**
- Airport to Sultanahmet: 100-150 TL
- Taksim to Kadıköy: 60-80 TL
- Sultanahmet to Galata: 25-35 TL

**Essential Phrases:**
- "Taksimetre lütfen" (Please use the meter)
- "Üstü kalsın" (Keep the change)
- "Fatura lütfen" (Receipt please)

**Tips:**
- Always insist on meter usage
- Avoid taxis near major tourist sites (may overcharge)
- Use BiTaksi or Uber for transparency
- Have destination written in Turkish
- 10-15% tip is customary"""

        # Check for specific transportation queries
        elif any(word in user_input_lower for word in ['sultanahmet', 'galata', 'tower']):
            response = """Getting from Sultanahmet to Galata Tower

**Best Routes:**

**Option 1: Tram + Metro (25 mins)**
- Take T1 tram from Sultanahmet to Karaköy
- Walk 8 minutes uphill to Galata Tower
- Most convenient and scenic

**Option 2: Walking (35-40 mins)**
- Walk across Golden Horn via Galata Bridge
- Enjoy street life and Bosphorus views
- Uphill climb to tower but very scenic

**Option 3: Metro (30 mins)**
- M2 metro from Vezneciler to Şişhane
- 5-minute walk to Galata Tower
- Good if coming from other metro stations

**Tips:**
- Galata Bridge walk offers great photo opportunities
- Tower area has many cafes for rest stops
- Consider sunset timing for best tower views"""
        
        elif any(word in user_input_lower for word in ['night', 'late', 'evening', 'safety']):
            response = """Night Transportation Safety in Istanbul

**Metro System:**
- Operates until 00:30 (12:30 AM) on weekdays
- Until 02:00 (2:00 AM) on Fridays/Saturdays
- Well-lit stations with security cameras
- Generally very safe

**Night Buses:**
- Special night bus routes (marked with 'N')
- Run from 00:30 to 06:00
- Less frequent but cover main routes
- Safe but more crowded

**Taxis:**
- Available 24/7
- Use BiTaksi or Uber for tracking
- Always ask for meter ("taksimetre lütfen")
- Avoid unofficial taxis

**Safety Tips:**
- Stay in well-lit, busy areas
- Keep belongings secure
- Travel in groups when possible
- Know your destination name in Turkish
- Have your hotel address written down"""
        
        elif any(word in user_input_lower for word in ['metrobus', 'difference']):
            response = """Metro vs Metrobus in Istanbul

**Metro (M Lines):**
- Underground rapid transit system
- M1: Airport to city center
- M2: North-south European side
- M3, M4: Various city connections
- Clean, air-conditioned, frequent
- Stations have elevators and escalators

**Metrobus (BRT):**
- Bus Rapid Transit system
- Dedicated bus lanes on highways
- Connects European and Asian sides
- Faster than regular buses
- More crowded, especially rush hours
- Runs above ground

**Key Differences:**
- Metro: Underground, cleaner, more comfortable
- Metrobus: Faster for long distances, crosses continents
- Metro: Better for central areas
- Metrobus: Good for cross-city travel
- Both use same Istanbul Card (Istanbulkart)"""

        elif any(word in user_input_lower for word in ['app', 'mobile', 'smartphone']):
            response = """Best Transportation Apps for Istanbul

**Essential Apps:**

**Moovit:**
- Real-time public transport info
- Route planning with multiple options
- Offline maps available
- Works in English

**Google Maps:**
- Excellent for walking directions
- Public transport integration
- Real-time traffic updates
- Works offline with downloaded maps

**BiTaksi:**
- Official taxi app
- Track your ride and driver
- Cashless payment options
- Available in English

**İstanbul Kart Mobile:**
- Check card balance
- Add money to your Istanbul Card
- Find nearest top-up points

**Uber:**
- Alternative to taxis
- Fixed pricing
- English interface
- Available in most areas"""

        else:
            response = """Getting Around Istanbul

Istanbul Card (Istanbulkart):
- Essential for all public transport
- Buy at metro stations or kiosks
- Works on metro, bus, tram, ferry

Metro & Tram:
- Clean, efficient, connects major areas
- M1: Airport to city center
- M2: European side north-south
- Tram: Historic peninsula (Sultanahmet)

Ferries:
- Cross between European & Asian sides
- Scenic Bosphorus tours
- Kadıköy to Eminönü popular route

Taxis & Apps:
- BiTaksi and Uber available
- Always ask for meter ("taksimetre")

Tips:
- Rush hours: 8-10 AM, 5-7 PM
- Download offline maps
- Learn basic Turkish transport terms"""
        
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # Weather and timing questions
    elif any(word in user_input_lower for word in ['weather', 'climate', 'season', 'when to visit', 'best time']):
        response = """Istanbul Weather & Best Times to Visit

Seasons:

Spring (April-May): BEST
- Perfect weather (15-22°C)
- Blooming tulips in parks
- Fewer crowds

Summer (June-August):
- Hot (25-30°C), humid
- Peak tourist season
- Great for Bosphorus activities

Fall (September-November): EXCELLENT
- Mild weather (18-25°C)
- Beautiful autumn colors
- Ideal for walking tours

Winter (December-March):
- Cool, rainy (8-15°C)
- Fewer tourists
- Cozy indoor experiences

What to Pack:
- Comfortable walking shoes
- Layers for temperature changes
- Light rain jacket
- Modest clothing for mosques"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # Shopping questions (but exclude food court queries)
    elif any(word in user_input_lower for word in ['shop', 'shopping', 'bazaar', 'market', 'buy']) and not any(word in user_input_lower for word in ['food court', 'food', 'eat', 'restaurant']):
        response = """Shopping in Istanbul

Traditional Markets:
- Grand Bazaar (Kapalıçarşı) - 4,000 shops, carpets, jewelry
- Spice Bazaar - Turkish delight, spices, teas
- Arasta Bazaar - Near Blue Mosque, smaller crowds

Modern Shopping:
- Istinye Park - Luxury brands, European side
- Kanyon - Unique architecture in Levent
- Zorlu Center - High-end shopping in Beşiktaş

What to Buy:
- Turkish carpets & kilims
- Ceramic tiles and pottery
- Turkish delight & spices
- Leather goods
- Gold jewelry

Bargaining Tips:
- Expected in bazaars, not in modern stores
- Start at 30-50% of asking price
- Be polite and patient
- Compare prices at multiple shops"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # General recommendations
    elif any(word in user_input_lower for word in ['recommend', 'suggest', 'what to do', 'attractions', 'sights']):
        response = """Top Istanbul Recommendations

Must-See Historic Sites:
- Hagia Sophia - Byzantine masterpiece
- Blue Mosque - Ottoman architecture
- Topkapi Palace - Ottoman sultans' palace
- Basilica Cistern - Underground marvel

Neighborhoods to Explore:
- Sultanahmet - Historic peninsula
- Beyoğlu - Modern culture, nightlife
- Galata - Trendy area, great views
- Kadıköy - Asian side, local vibe

Unique Experiences:
- Bosphorus ferry cruise at sunset
- Turkish bath (hamam) experience
- Rooftop dining with city views
- Local food tour in Kadıköy

Day Trip Ideas:
- Princes' Islands (Büyükada)
- Büyükçekmece Lake
- Belgrade Forest hiking

Ask me about specific areas or activities for more detailed information!"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # Family and special interest queries
    elif any(word in user_input_lower for word in ['family', 'families', 'kids', 'children', 'child', 'family friendly', 'baby', 'stroller']):
        response = """Family-Friendly Istanbul

Best Districts for Families:
- Sultanahmet - Historic sites, easy walking
- Beyoğlu - Museums, cultural activities  
- Büyükçekmece - Lake activities, parks
- Florya - Beach area, parks

Family Attractions:
- Miniaturk - Miniature park with Turkish landmarks
- Istanbul Aquarium - Large aquarium with sea life
- Vialand (İsfanbul) - Theme park and shopping
- Princes' Islands - Car-free islands, horse carriages
- Emirgan Park - Beautiful gardens, playgrounds
- Gülhane Park - Historic park near Sultanahmet

Family-Friendly Activities:
- Bosphorus ferry rides (not too long)
- Galata Tower visit
- Turkish bath experience (family sections available)
- Street food tasting in safe areas
- Park picnics with city views

Tips for Families:
- Many museums have family discounts
- Strollers work well in most tourist areas
- Public transport is stroller-friendly
- Many restaurants welcome children"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # Romantic and couples queries
    elif any(word in user_input_lower for word in ['romantic', 'couple', 'couples', 'honeymoon', 'anniversary', 'date', 'romance']):
        response = """Romantic Istanbul

Romantic Neighborhoods:
- Ortaköy - Waterfront dining, Bosphorus views
- Bebek - Upscale cafes, scenic walks
- Galata - Historic charm, rooftop bars
- Sultanahmet - Historic atmosphere, sunset views

Romantic Experiences:
- Private Bosphorus sunset cruise
- Rooftop dinner with city skyline views
- Traditional Turkish bath for couples
- Walk through Gülhane Park at sunset
- Evening stroll across Galata Bridge
- Private guided tour of historic sites

Romantic Restaurants:
- Waterfront restaurants in Ortaköy
- Rooftop dining in Beyoğlu
- Traditional Ottoman cuisine in Sultanahmet
- Seafood restaurants in Kumkapı

Romantic Views:
- Galata Tower at sunset
- Çamlıca Hill panoramic views
- Pierre Loti Hill overlooking Golden Horn
- Maiden's Tower (Kız Kulesi) boat trip

Perfect for couples seeking memorable experiences in this enchanting city!"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # Rainy day and indoor activities
    elif any(word in user_input_lower for word in ['rainy', 'rain', 'indoor', 'indoors', 'bad weather', 'cold day', 'winter activities']):
        response = """Rainy Day Istanbul

Indoor Attractions:
- Hagia Sophia - Historic marvel to explore
- Topkapi Palace - Ottoman history and treasures  
- Basilica Cistern - Underground architectural wonder
- Istanbul Archaeological Museums
- Turkish and Islamic Arts Museum
- Pera Museum - Art and cultural exhibitions

Shopping Centers:
- Grand Bazaar - Historic covered market
- Spice Bazaar - Aromatic indoor market
- Istinye Park - Modern luxury mall
- Kanyon - Unique architecture, many shops
- Zorlu Center - Shopping and entertainment

Indoor Experiences:
- Traditional Turkish bath (hamam)
- Turkish cooking classes
- Tea and coffee house visits
- Indoor restaurant tours
- Art gallery visits in Beyoğlu
- Covered passages (pasaj) exploration

Cozy Cafes:
- Historic neighborhoods have many warm cafes
- Traditional tea houses in Sultanahmet
- Modern coffee shops in Galata
- Hookah lounges for cultural experience

Perfect activities to enjoy Istanbul regardless of weather!"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # Budget and free activities queries
    elif any(word in user_input_lower for word in ['budget', 'cheap', 'free', 'affordable', 'low cost', 'student', 'backpacker']):
        response = """Budget-Friendly Istanbul

Free Activities:
- Walk through historic Sultanahmet district
- Visit many mosques (free entry)
- Explore Balat and Fener neighborhoods
- Walk across Galata Bridge
- Ferry rides (very affordable public transport)
- Parks: Gülhane, Emirgan, Yıldız
- Street markets and bazaars (free to explore)

Budget Accommodations:
- Hostels in Sultanahmet and Beyoğlu
- Guesthouses in Kadıköy
- Budget hotels in Fatih district

Affordable Food:
- Street food: döner, simit, balık ekmek
- Local eateries (lokanta) for home-style meals
- Lunch menus at many restaurants
- Turkish breakfast (kahvaltı) offers great value
- Supermarkets for self-catering

Budget Transportation:
- Istanbul Card for public transport discounts
- Walking between nearby attractions
- Ferry rides for sightseeing
- Shared dolmuş (minibus) rides

Money-Saving Tips:
- Many museums have student discounts
- Free WiFi widely available
- Happy hour at many bars and restaurants
- Local markets for fresh, affordable produce"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

    # For all other queries (including casual conversation), return None to pass to OpenAI
    else:
        # Only return fallback for extremely short or non-alphabetic inputs
        if len(user_input.strip()) < 2 or not any(char.isalpha() for char in user_input):
            return "Sorry, I couldn't understand. Can you type again?"
        
        # Return None to indicate this should go to OpenAI for natural conversation
        return None

def create_fuzzy_keywords():
    """Create a comprehensive list of keywords for fuzzy matching"""
    keywords = {
        # Location names and variations
        'locations': [
            'kadikoy', 'kadıköy', 'sultanahmet', 'beyoglu', 'beyoğlu', 'galata', 
            'taksim', 'besiktas', 'beşiktaş', 'uskudar', 'üsküdar', 'fatih', 
            'sisli', 'şişli', 'karakoy', 'karaköy', 'ortakoy', 'ortaköy', 
            'bebek', 'arnavutkoy', 'arnavutköy', 'balat', 'fener', 'eminonu', 
            'eminönü', 'bakirkoy', 'bakırköy', 'maltepe', 'istanbul', 'instanbul'
        ],
        # Query types and variations
        'places': [
            'places', 'place', 'plases', 'plases', 'plase', 'spots', 'locations', 'areas'
        ],
        'restaurants': [
            'restaurants', 'restaurant', 'restourant', 'resturant', 'restarnts', 'restrant', 'food', 
            'eat', 'dining', 'eatery', 'cafe', 'cafes'
        ],
        'attractions': [
            'attractions', 'attraction', 'atraction', 'sights', 'sites', 
            'tourist', 'visit', 'see', 'things to do', 'activities'
        ],
        'museums': [
            'museums', 'museum', 'musem', 'gallery', 'galleries', 'art', 
            'culture', 'cultural', 'history', 'historical'
        ],
        'nightlife': [
            'nightlife', 'night', 'bars', 'bar', 'clubs', 'club', 'party', 
            'drinks', 'entertainment'
        ],
        'shopping': [
            'shopping', 'shop', 'shops', 'market', 'markets', 'bazaar', 
            'bazaars', 'mall', 'malls', 'store', 'stores'
        ],
        'transport': [
            'transport', 'transportation', 'metro', 'bus', 'taxi', 'travel', 
            'getting around', 'how to get'
        ],
        # Common misspellings
        'common_words': [
            'where', 'whre', 'wher', 'find', 'fnd', 'good', 'gud', 'great', 'grate',
            'can', 'cna', 'you', 'me', 'i', 'help', 'hlp'
        ]
    }
    return keywords

def correct_typos(text, threshold=75):
    """Enhanced typo correction with improved fuzzy matching"""
    try:
        keywords = create_fuzzy_keywords()
        words = text.lower().split()
        corrected_words = []
        
        # Extended list of common words that should not be corrected
        stop_words = {'in', 'to', 'at', 'on', 'for', 'with', 'by', 'from', 'up', 
                     'about', 'into', 'through', 'during', 'before', 'after', 
                     'above', 'below', 'between', 'among', 'a', 'an', 'the', 
                     'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'what', 'where', 'when', 'how', 'why', 'which', 'who', 'whom',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
                     'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
                     'can', 'could', 'should', 'would', 'will', 'may', 'might', 'must',
                     'good', 'great', 'best', 'nice', 'fine', 'well', 'bad', 'better',
                     'find', 'get', 'go', 'come', 'see', 'know', 'think', 'want',
                     'one', 'two', 'three', 'some', 'many', 'few', 'all', 'any',
                     'looking', 'staying', 'romantic', 'families', 'options', 'near',
                     'there', 'here', 'help', 'plan', 'give', 'show', 'tell'}
        
        # Expanded common word corrections for frequent typos
        common_corrections = {
            'whre': 'where', 'wher': 'where', 'were': 'where', 'wheere': 'where',
            'fnd': 'find', 'findd': 'find', 'finde': 'find', 'fimd': 'find',
            'gud': 'good', 'goood': 'good', 'goo': 'good', 'goof': 'good',
            'resturant': 'restaurant', 'restrant': 'restaurant', 'restarant': 'restaurant',
            'restarnts': 'restaurants', 'resturants': 'restaurants', 'restaurnt': 'restaurant',
            'musuem': 'museum', 'musem': 'museum', 'musuems': 'museums', 'musium': 'museum',
            'tourst': 'tourist', 'turist': 'tourist', 'touryst': 'tourist', 'turrist': 'tourist',
            'istambul': 'istanbul', 'instanbul': 'istanbul', 'istanbuul': 'istanbul', 'istambul': 'istanbul',
            'atraction': 'attraction', 'atractions': 'attractions', 'atracttion': 'attraction',
            'recomend': 'recommend', 'recomendation': 'recommendation', 'recomendations': 'recommendations',
            'grate': 'great', 'graet': 'great', 'greate': 'great',
            'familys': 'families', 'familey': 'family', 'famly': 'family',
            'childern': 'children', 'childs': 'children', 'childrens': 'children',
            'romantik': 'romantic', 'romamtic': 'romantic', 'romanctic': 'romantic',
            'beutiful': 'beautiful', 'beautifull': 'beautiful', 'beatiful': 'beautiful',
            'intresting': 'interesting', 'intersting': 'interesting', 'interessting': 'interesting',
            'expencive': 'expensive', 'expensiv': 'expensive', 'expesive': 'expensive',
            'cheep': 'cheap', 'chep': 'cheap', 'chip': 'cheap',
            'freindly': 'friendly', 'frendly': 'friendly', 'frienly': 'friendly',
            'awsome': 'awesome', 'awsom': 'awesome', 'awesme': 'awesome'
        }
        
        for word in words:
            # Remove punctuation for comparison
            clean_word = word.strip('.,!?;:')
            original_word = word
            
            # Skip common stop words
            if clean_word in stop_words:
                corrected_words.append(word)
                continue
            
            # Skip very short words (likely not typos)
            if len(clean_word) <= 2:
                corrected_words.append(word)
                continue
            
            # Check common corrections first
            if clean_word in common_corrections:
                corrected_word = common_corrections[clean_word]
                # Preserve original punctuation
                if original_word != clean_word:
                    corrected_word += original_word[len(clean_word):]
                corrected_words.append(corrected_word)
                print(f"Common typo correction: '{clean_word}' -> '{common_corrections[clean_word]}'")
                continue
                
            best_match = None
            best_score = 0
            best_category = None
            
            # Check all categories including common words for better coverage
            categories_to_check = ['locations', 'restaurants', 'museums', 'attractions', 'places', 'common_words']
            
            # Check each category of keywords
            for category in categories_to_check:
                if category in keywords:
                    match = process.extractOne(clean_word, keywords[category])
                    if match and match[1] > best_score and match[1] >= threshold:
                        # Don't correct if it's already very similar or too short
                        if match[1] < 98 or clean_word != match[0]:
                            # Only correct if the word is reasonably long and the correction makes sense
                            if len(clean_word) >= 4 and abs(len(clean_word) - len(match[0])) <= 3:
                                best_match = match[0]
                                best_score = match[1]
                                best_category = category
            
            if best_match and best_score >= threshold and clean_word != best_match:
                # Additional check: don't replace common English words with single letters
                if len(best_match) >= 3 and len(clean_word) >= 3:
                    # Preserve original punctuation
                    corrected_word = best_match
                    if original_word != clean_word:
                        corrected_word += original_word[len(clean_word):]
                    corrected_words.append(corrected_word)
                    print(f"Fuzzy typo correction: '{clean_word}' -> '{best_match}' (score: {best_score}, category: {best_category})")
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        return corrected_text
    except Exception as e:
        print(f"Error in typo correction: {e}")
        return text

# Note: enhance_query_understanding is now imported from enhanced_input_processor.py

# Routers
app.include_router(museums.router)
app.include_router(restaurants.router)
app.include_router(places.router)
app.include_router(blog.router)

# ===== MAIN AI ENDPOINTS =====

@app.post("/ai")
@log_ai_operation("chatbot_query")
async def ai_istanbul_router(request: Request):
    """Main AI endpoint for chatbot queries"""
    session_id = "unknown"
    try:
        data = await request.json()
        user_input = data.get("user_input", "")
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        # Get language from headers - fix the header access
        accept_language = request.headers.get("accept-language")
        language = i18n_service.get_language_from_headers(accept_language)
        
        # Get conversation history for context
        conversation_history = []
        if hasattr(data, 'messages'):
            conversation_history = data.get('messages', [])
        
        # Debug logging
        structured_logger.info(
            f"🌐 Detected language: {language}",
            session_id=session_id,
            user_input=user_input[:100]
        )
        
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
        
        # Sanitize input
        user_input = clean_text_formatting(user_input)
        
        # Enhanced query understanding
        enhanced_user_input = enhance_query_understanding(user_input)
        user_input = enhanced_user_input
        
        # Create database session
        db = SessionLocal()
        try:
            # Handle simple greetings with template responses
            if not i18n_service.should_use_ai_response(user_input, language):
                welcome_message = i18n_service.translate("welcome", language)
                return {"response": welcome_message, "session_id": session_id}
            
            # Check for specific query types
            places = db.query(Place).all()
            
            # Try fallback response first (for structured queries)
            fallback_response = create_fallback_response(user_input, places)
            if fallback_response and fallback_response.strip():
                enhanced_response = enhance_ai_response_formatting(fallback_response)
                clean_response = clean_text_formatting(enhanced_response)
                return {"response": clean_response, "session_id": session_id}
            
            # For complex queries or when fallback returns None, use GPT
            gpt_response = get_gpt_response(user_input, session_id)
            if gpt_response and gpt_response.strip():
                enhanced_response = enhance_ai_response_formatting(gpt_response)
                clean_response = clean_text_formatting(enhanced_response)
                return {"response": clean_response, "session_id": session_id}
            
            # Ultimate fallback
            return {"response": "I can help you explore Istanbul! Ask me about restaurants, museums, districts, or transportation.", "session_id": session_id}
            
        finally:
            db.close()
            
    except Exception as e:
        structured_logger.log_error_with_traceback(
            "Critical error in AI endpoint",
            e,
            session_id=session_id,
            component="ai_endpoint"
        )
        print(f"[ERROR] Exception in /ai endpoint: {e}")
        traceback.print_exc()
        return {"response": "Sorry, I encountered an error. Please try again.", "session_id": session_id}

@app.post("/ai/stream")
@log_ai_operation("chatbot_streaming")
async def ai_istanbul_streaming(request: Request):
    """Streaming AI endpoint for real-time responses"""
    try:
        data = await request.json()
        user_input = data.get("user_input", "")
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        # Get language from headers - fix the header access
        accept_language = request.headers.get("accept-language")
        language = i18n_service.get_language_from_headers(accept_language)
        
        async def generate_response():
            try:
                # Create database session
                db = SessionLocal()
                try:
                    # Enhanced input validation
                    if not user_input or len(user_input.strip()) < 2:
                        response_data = json.dumps({'chunk': "I'm here to help you explore Istanbul! What would you like to know?"})
                        yield f"data: {response_data}\n\n"
                        return
                    
                    # Sanitize input
                    clean_input = clean_text_formatting(user_input)
                    enhanced_input = enhance_query_understanding(clean_input)
                    
                    # Check for simple greetings
                    if not i18n_service.should_use_ai_response(enhanced_input, language):
                        welcome_message = i18n_service.translate("welcome", language)
                        response_data = json.dumps({'chunk': welcome_message})
                        yield f"data: {response_data}\n\n"
                        return
                    
                    # Try fallback response first
                    places = db.query(Place).all()
                    fallback_response = create_fallback_response(enhanced_input, places)
                    
                    if fallback_response and fallback_response.strip():
                        # Stream the fallback response
                        enhanced_response = enhance_ai_response_formatting(fallback_response)
                        clean_response = clean_text_formatting(enhanced_response)
                        
                        # Stream in chunks for better UX
                        words = clean_response.split()
                        chunk_size = 3
                        for i in range(0, len(words), chunk_size):
                            chunk = " ".join(words[i:i+chunk_size])
                            if chunk.strip():  # Only send non-empty chunks
                                # Add space after chunk (except for last chunk)
                                if i + chunk_size < len(words):
                                    chunk += " "
                                response_data = json.dumps({'chunk': chunk})
                                yield f"data: {response_data}\n\n"
                                await asyncio.sleep(0.05)  # Small delay for streaming effect
                    else:
                        # Use GPT for complex queries
                        gpt_response = get_gpt_response(enhanced_input, session_id)
                        if gpt_response:
                            enhanced_response = enhance_ai_response_formatting(gpt_response)
                            clean_response = clean_text_formatting(enhanced_response)
                            
                            # Stream the GPT response
                            words = clean_response.split()
                            chunk_size = 3
                            for i in range(0, len(words), chunk_size):
                                chunk = " ".join(words[i:i+chunk_size])
                                if chunk.strip():  # Only send non-empty chunks
                                    # Add space after chunk (except for last chunk)
                                    if i + chunk_size < len(words):
                                        chunk += " "
                                    response_data = json.dumps({'chunk': chunk})
                                    yield f"data: {response_data}\n\n"
                                    await asyncio.sleep(0.05)
                        else:
                            response_data = json.dumps({'chunk': 'I can help you explore Istanbul! Ask me about restaurants, museums, districts, or transportation.'})
                            yield f"data: {response_data}\n\n"
                    
                    # Send completion signal
                    completion_data = json.dumps({'done': True})
                    yield f"data: {completion_data}\n\n"
                    
                except Exception as e:
                    structured_logger.log_error_with_traceback(
                        "Error in streaming endpoint",
                        e,
                        session_id=session_id,
                        component="ai_streaming"
                    )
                    error_data = json.dumps({'error': 'Sorry, I encountered an error. Please try again.'})
                    yield f"data: {error_data}\n\n"
                finally:
                    db.close()
            except Exception as e:
                # Outer exception handler for any uncaught errors
                error_data = json.dumps({'error': 'Critical streaming error occurred.'})
                yield f"data: {error_data}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
        
    except Exception as e:
        structured_logger.log_error_with_traceback(
            "Critical error in streaming endpoint",
            e,
            component="ai_streaming"
        )
        print(f"[ERROR] Exception in /ai/stream endpoint: {e}")
        return {"error": "Streaming not available"}

@app.get("/")
def root():
    return {"message": "Welcome to AIstanbul API"}

@app.get("/admin_dashboard.html", response_class=HTMLResponse)
async def admin_dashboard():
    """Serve the admin dashboard HTML page"""
    try:
        dashboard_path = os.path.join(os.path.dirname(__file__), "admin_dashboard.html")
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Admin Dashboard Not Found</h1><p>Please ensure admin_dashboard.html is in the backend directory.</p>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error Loading Dashboard</h1><p>{str(e)}</p>", status_code=500)

@app.post("/feedback")
async def receive_feedback(request: Request):
    """Endpoint to receive user feedback on AI responses"""
    try:
        feedback_data = await request.json()
        client_ip = request.client.host if request.client else "unknown"
        
        # Store feedback in database
        db = SessionLocal()
        try:
            # Store feedback
            new_feedback = UserFeedback(
                session_id=feedback_data.get('sessionId', 'N/A'),
                feedback_type=feedback_data.get('feedbackType', 'unknown'),
                user_query=feedback_data.get('userQuery', 'N/A')[:500],
                response_preview=feedback_data.get('messageText', '')[:200],
                message_content=feedback_data.get('messageText', ''),
                user_ip=client_ip,
                timestamp=datetime.now()
            )
            db.add(new_feedback)
            
            # Create or update chat session for both like and dislike feedback
            feedback_type = feedback_data.get('feedbackType', 'unknown')
            if feedback_type in ['like', 'dislike']:
                session_id = feedback_data.get('sessionId', 'N/A')
                session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                
                if not session:
                    # Create new session for both liked and disliked responses
                    title = feedback_data.get('userQuery', 'Chat Session')[:100] + '...' if len(feedback_data.get('userQuery', '')) > 100 else feedback_data.get('userQuery', 'Chat Session')
                    conversation_data = [{
                        'query': feedback_data.get('userQuery', 'N/A'),
                        'response': feedback_data.get('messageText', ''),
                        'timestamp': datetime.now().isoformat(),
                        'feedback': feedback_type
                    }]
                    session = ChatSession(
                        id=session_id,
                        title=title,
                        user_ip=client_ip,
                        message_count=1,
                        is_saved=True,
                        saved_at=datetime.now(),
                        conversation_history=conversation_data
                    )
                    db.add(session)
                    print(f"📝 Creating new {feedback_type} session: {session_id}")
                else:
                    # Update existing session
                    db.refresh(session)  # Ensure we have latest data
                    current_history = session.conversation_history if hasattr(session, 'conversation_history') and session.conversation_history is not None else []
                    if isinstance(current_history, list):
                        history_list = current_history.copy()
                    else:
                        history_list = []
                    
                    history_list.append({
                        'query': feedback_data.get('userQuery', 'N/A'),
                        'response': feedback_data.get('messageText', ''),
                        'timestamp': datetime.now().isoformat(),
                        'feedback': feedback_type
                    })
                    
                    # Use update statement for proper SQLAlchemy assignment
                    db.query(ChatSession).filter(ChatSession.id == session_id).update({
                        ChatSession.is_saved: True,
                        ChatSession.saved_at: datetime.now(),
                        ChatSession.last_activity_at: datetime.now(),
                        ChatSession.conversation_history: history_list,
                        ChatSession.message_count: len(history_list)
                    })
                    print(f"📝 Updating existing {feedback_type} session: {session_id}")
            
            # Single commit for all database operations
            db.commit()
            
        except Exception as db_error:
            db.rollback()
            print(f"Database error storing feedback: {db_error}")
        finally:
            db.close()
        
        # Log feedback using structured logging
        structured_logger.info(
            "User feedback received and stored",
            feedback_type=feedback_data.get('feedbackType', 'unknown'),
            user_query=feedback_data.get('userQuery', 'N/A')[:200],
            response_preview=feedback_data.get('messageText', '')[:100],
            session_id=feedback_data.get('sessionId', 'N/A'),
            timestamp=datetime.now().isoformat()
        )
        
        # Log feedback to console for observation
        print(f"\n📊 FEEDBACK RECEIVED & STORED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Type: {feedback_data.get('feedbackType', 'unknown')}")
        print(f"Query: {feedback_data.get('userQuery', 'N/A')}")
        print(f"Response: {feedback_data.get('messageText', '')[:100]}...")
        print(f"Session: {feedback_data.get('sessionId', 'N/A')}")
        print("-" * 50)
        
        return {
            "status": "success",
            "message": "Feedback received and stored",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        structured_logger.log_error_with_traceback(
            "Error processing feedback",
            e,
            component="feedback_endpoint"
        )
        print(f"Error processing feedback: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/chat-sessions")
async def get_chat_sessions():
    """API endpoint to retrieve saved chat sessions for the admin dashboard"""
    try:
        db = SessionLocal()
        try:
            # Get all saved sessions (sessions with liked messages)
            sessions = db.query(ChatSession).filter(ChatSession.is_saved == True).order_by(ChatSession.saved_at.desc()).all()
            
            session_list = []
            for session in sessions:
                session_data = {
                    'id': session.id,
                    'title': session.title or 'Untitled Session',
                    'user_ip': session.user_ip or 'unknown',
                    'message_count': session.message_count or 0,
                    'saved_at': session.saved_at.isoformat() if hasattr(session, 'saved_at') and session.saved_at is not None else datetime.now().isoformat(),
                    'first_message_at': session.first_message_at.isoformat() if hasattr(session, 'first_message_at') and session.first_message_at is not None else None,
                    'last_activity_at': session.last_activity_at.isoformat() if hasattr(session, 'last_activity_at') and session.last_activity_at is not None else None,
                    'conversation_history': session.conversation_history or []
                }
                session_list.append(session_data)
            
            return {
                "status": "success",
                "sessions": session_list,
                "total_count": len(session_list)
            }
            
        except Exception as db_error:
            print(f"Database error retrieving sessions: {db_error}")
            return {"status": "error", "error": "Database error", "sessions": []}
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error retrieving chat sessions: {e}")
        return {"status": "error", "error": str(e), "sessions": []}

@app.get("/api/chat-sessions/{session_id}")
async def get_chat_session_detail(session_id: str):
    """API endpoint to retrieve detailed information for a specific chat session"""
    try:
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            
            if not session:
                return {"status": "error", "error": "Session not found"}
            
            # Also get feedback for this session
            feedback_entries = db.query(UserFeedback).filter(UserFeedback.session_id == session_id).all()
            
            feedback_list = []
            for feedback in feedback_entries:
                feedback_list.append({
                    'feedback_type': feedback.feedback_type,
                    'user_query': feedback.user_query,
                    'response_preview': feedback.response_preview,
                    'timestamp': feedback.timestamp.isoformat() if hasattr(feedback, 'timestamp') and feedback.timestamp is not None else None,
                    'message_index': getattr(feedback, 'message_index', None)
                })
            
            session_detail = {
                'id': session.id,
                'title': session.title or 'Untitled Session',
                'user_ip': session.user_ip or 'unknown',
                'message_count': session.message_count or 0,
                'saved_at': session.saved_at.isoformat() if hasattr(session, 'saved_at') and session.saved_at is not None else None,
                'first_message_at': session.first_message_at.isoformat() if hasattr(session, 'first_message_at') and session.first_message_at is not None else None,
                'last_activity_at': session.last_activity_at.isoformat() if hasattr(session, 'last_activity_at') and session.last_activity_at is not None else None,
                'conversation_history': session.conversation_history or [],
                'feedback_entries': feedback_list
            }
            
            return {
                "status": "success",
                "session": session_detail
            }
            
        except Exception as db_error:
            print(f"Database error retrieving session detail: {db_error}")
            return {"status": "error", "error": "Database error"}
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error retrieving session detail: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/api/feedback")
async def api_receive_feedback(request: Request):
    """API endpoint to receive user feedback on AI responses with updated format"""
    try:
        feedback_data = await request.json()
        client_ip = request.client.host if request.client else "unknown"
        
        # Store feedback in database
        db = SessionLocal()
        try:
            # Store feedback with updated format
            new_feedback = UserFeedback(
                session_id=feedback_data.get('session_id', 'N/A'),
                feedback_type=feedback_data.get('feedback_type', 'unknown'),
                user_query=feedback_data.get('user_query', 'N/A')[:500],
                response_preview=feedback_data.get('message_content', '')[:200],
                message_content=feedback_data.get('message_content', ''),
                message_index=feedback_data.get('message_index', 0),
                user_ip=client_ip,
                timestamp=datetime.now()
            )
            db.add(new_feedback)
            
            # Create or update chat session for both like and dislike feedback
            feedback_type = feedback_data.get('feedback_type', 'unknown')
            if feedback_type in ['like', 'dislike']:
                session_id = feedback_data.get('session_id', 'N/A')
                session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                
                if not session:
                    # Create new session for both liked and disliked responses
                    title = feedback_data.get('user_query', 'Chat Session')[:100] + '...' if len(feedback_data.get('user_query', '')) > 100 else feedback_data.get('user_query', 'Chat Session')
                    conversation_data = [{
                        'query': feedback_data.get('user_query', 'N/A'),
                        'response': feedback_data.get('message_content', ''),
                        'timestamp': datetime.now().isoformat(),
                        'feedback': feedback_type,
                        'message_index': feedback_data.get('message_index', 0)
                    }]
                    session = ChatSession(
                        id=session_id,
                        title=title,
                        user_ip=client_ip,
                        message_count=1,
                        is_saved=True,
                        saved_at=datetime.now(),
                        conversation_history=conversation_data
                    )
                    db.add(session)
                else:
                    # Update existing session
                    db.refresh(session)
                    current_history = session.conversation_history if hasattr(session, 'conversation_history') and session.conversation_history is not None else []
                    if isinstance(current_history, list):
                        history_list = current_history.copy()
                    else:
                        history_list = []
                    
                    history_list.append({
                        'query': feedback_data.get('user_query', 'N/A'),
                        'response': feedback_data.get('message_content', ''),
                        'timestamp': datetime.now().isoformat(),
                        'feedback': feedback_type,
                        'message_index': feedback_data.get('message_index', 0)
                    })
                    
                    # Use update statement for proper SQLAlchemy assignment
                    db.query(ChatSession).filter(ChatSession.id == session_id).update({
                        ChatSession.is_saved: True,
                        ChatSession.saved_at: datetime.now(),
                        ChatSession.last_activity_at: datetime.now(),
                        ChatSession.conversation_history: history_list,
                        ChatSession.message_count: len(history_list)
                    })
                    db.commit()
            
            db.commit()
            
        except Exception as db_error:
            db.rollback()
            print(f"Database error storing API feedback: {db_error}")
        finally:
            db.close()
        
        # Log feedback using structured logging with new format
        structured_logger.info(
            "User feedback received and stored via API",
            feedback_type=feedback_data.get('feedback_type', 'unknown'),
            session_id=feedback_data.get('session_id', 'N/A'),
            message_index=feedback_data.get('message_index', 'N/A'),
            message_preview=feedback_data.get('message_content', '')[:100],
            timestamp=feedback_data.get('timestamp', datetime.now().isoformat())
        )
        
        # Log feedback to console for observation
        print(f"\n📊 FEEDBACK RECEIVED & STORED (API) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Type: {feedback_data.get('feedback_type', 'unknown')}")
        print(f"Session: {feedback_data.get('session_id', 'N/A')}")
        print(f"Message Index: {feedback_data.get('message_index', 'N/A')}")
        print(f"Message: {feedback_data.get('message_content', '')[:100]}...")
        print("-" * 50)
        
        return {
            "status": "success",
            "message": "Feedback received and stored successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        structured_logger.log_error_with_traceback(
            "Error processing API feedback",
            e,
            component="api_feedback_endpoint"
        )
        print(f"Error processing API feedback: {e}")
        return {
            "status": "error", 
            "message": "Failed to process feedback",
            "timestamp": datetime.now().isoformat()
        }

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Istanbul Backend Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📊 API documentation at: http://localhost:8000/docs")
    print("🔧 Health check at: http://localhost:8000/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )




