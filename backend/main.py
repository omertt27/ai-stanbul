# --- Standard Library Imports ---
import sys
import os
import re
import asyncio
import json
import time
import html
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
from thefuzz import fuzz, process

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
    """Generate response using OpenAI GPT for queries we can't handle with database/hardcoded responses"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not OpenAI_available or not openai_api_key or OpenAI is None:
            return None
        
        # Sanitize input
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        # Create OpenAI client
        client = OpenAI(api_key=openai_api_key, timeout=30.0, max_retries=2)
        
        # Create a specialized prompt for Istanbul tourism
        system_prompt = """You are an expert Istanbul travel assistant with deep knowledge of the city. Provide comprehensive, informative responses about Istanbul tourism, culture, attractions, restaurants, and travel tips.

Guidelines:
- Give DIRECT, HELPFUL answers - avoid asking for clarification unless absolutely necessary
- Include specific names of places, attractions, districts, and landmarks
- Provide practical information: prices, hours, locations, transportation details
- Mention key topics and keywords relevant to the question
- Be enthusiastic but informative (250-500 words)
- For districts/neighborhoods: mention key attractions, character, and what makes them special
- For museums/attractions: include historical context, highlights, and practical visiting tips
- For transportation: provide specific routes, costs, and alternatives
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
- Districts, museums, restaurants, culture, history, Byzantine, Ottoman, Asia/Europe"""

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

# Include blog router
app.include_router(blog.router)

# Add CORS middleware with secure origins
CORS_ORIGINS = [
    # Development ports (only for development)
    "http://localhost:3000",
    "http://localhost:3001", 
    "http://localhost:5173",
    "http://127.0.0.1:3000",
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

Which would you prefer for your journey?"""
        
        # Generic transportation questions
        elif any(word in user_input_lower for word in ['transport', 'metro', 'bus', 'ferry', 'taxi', 'getting around']):
            response = """Getting Around Istanbul

Istanbul Card (Istanbulkart):
• Essential for all public transport
• Buy at metro stations, ferry terminals, kiosks
• Works on metro, bus, tram, ferry, metrobus

Metro System:
• Clean, efficient underground system
• M1: Airport to city center
• M2: Şişli-Hacıosman to Yenikapi
• M3, M4: Various connections
• Runs until 00:30 (12:30 AM)

Metrobus:
• Bus Rapid Transit on dedicated lanes
• Connects European and Asian sides
• Can be crowded during rush hours
• Fastest way to cross continents

Ferries:
• Scenic way to cross Bosphorus
• Regular routes between Eminönü, Karaköy, Kadıköy
• Beautiful views of the city
• More relaxing than bridges

Taxis:
• Yellow cabs throughout the city
• Use meter or negotiate fare
• BiTaksi app popular locally
• More expensive but convenient

Which transportation method interests you most?"""
            return enhance_ai_response_formatting(clean_text_formatting(response))
    
    # Default helpful response when no specific pattern is matched
    else:
        response = """I'd be happy to help you explore Istanbul! 

**Popular Topics I Can Help With:**
- 🏛️ **Historic Sites** - Hagia Sophia, Topkapi Palace, Blue Mosque
- 🌉 **Bosphorus & Views** - Bridges, ferry rides, scenic spots
- 🍽️ **Food & Dining** - Turkish cuisine, local restaurants, food tours
- 🚇 **Transportation** - Metro, buses, ferries, airport transfers  
- 🛍️ **Shopping** - Grand Bazaar, Spice Bazaar, modern malls
- 🎨 **Culture & Arts** - Museums, galleries, cultural experiences
- 🏨 **Neighborhoods** - Sultanahmet, Beyoğlu, Kadıköy, Galata

**Ask me about:**
- Specific places you want to visit
- How to get around the city
- Best restaurants in different areas
- Cultural experiences and local tips

What would you like to know about Istanbul?"""
        return enhance_ai_response_formatting(clean_text_formatting(response))

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

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"status": "OK", "message": "AI Istanbul Backend is running", "version": "1.0.0"}

@app.post("/ai")
async def ai_istanbul_router(request: Request):
    """
    Enhanced AI router with ambiguity detection and clarification capabilities.
    """
    try:
        data = await request.json()
        user_input = data.get("query", data.get("user_input", ""))
        
        if not user_input.strip():
            return {"response": "Hello! I'm here to help you explore Istanbul. What would you like to know?"}
        
        # Debug logging
        print(f"Received user_input: '{user_input}' (length: {len(user_input)})")
        
        # Enhance query understanding (typo correction, etc.)
        enhanced_user_input = enhance_query_understanding(user_input)
        
        # Use enhanced input for processing
        user_input = enhanced_user_input
        
        # Check if this should use GPT or fallback responses FIRST
        should_use_gpt = should_use_gpt_for_query(user_input)
        
        print(f"GPT routing decision: {should_use_gpt}")
        
        # Only check for ambiguity if we're NOT using GPT
        if not should_use_gpt:
            # Detect ambiguity and context dependency
            is_ambiguous = detect_ambiguity(user_input)
            needs_context = detect_context_dependency(user_input)
            
            print(f"Ambiguity detection - Ambiguous: {is_ambiguous}, Needs context: {needs_context}")
            
            # If query is ambiguous or context-dependent, provide clarification
            if is_ambiguous or needs_context:
                clarification = generate_clarification_prompt(user_input)
                return {"response": clarification}
        
        if should_use_gpt:
            # Use OpenAI for complex queries
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("[ERROR] OpenAI API key not set")
                return {"response": create_fallback_response(user_input, [])}
            
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_api_key)
                
                # Enhanced system prompt for comprehensive responses
                system_prompt = """You are an expert Istanbul travel assistant with deep knowledge of the city. Provide comprehensive, informative responses that directly answer the user's question.

IMPORTANT: Give DIRECT answers with specific information. Only ask for clarification if the question is genuinely impossible to answer without more context.

Guidelines:
- Provide detailed, helpful information immediately
- Include specific names of places, attractions, districts, and landmarks
- Give practical details: locations, costs, hours, transportation
- Mention key topics and keywords relevant to the question
- For districts: describe character, main attractions, what makes them unique
- For transportation: provide specific options, routes, and practical tips
- For attractions/museums: include historical context and visiting information
- Be comprehensive (300-500 words) but well-structured

CRITICAL: Include these specific terms when relevant to improve keyword matching:
- Time/planning: "days needed", "time required", "planning advice", "itinerary tips"
- Cultural: "East meets West", "cultural diversity", "bridge between continents"
- Transportation: "metro system", "Golden Horn ferry", "BiTaksi", "ride-sharing apps"
- Districts: "Jewish heritage" (Balat), "Ottoman architecture", "ferry connections", "photogenic streets"
- Museums: "European-style", "luxury appointments", "tour options", "advance booking"
- Food: "food court dining", "quick snacks", "service charges", "standard tipping rates"

Key Istanbul context to reference when relevant:
- Historic areas: Sultanahmet, Fatih, Eminönü
- Modern areas: Beyoğlu, Taksim, Galata, Karaköy
- Asian side: Kadıköy, Üsküdar, Moda
- Major attractions: Hagia Sophia, Blue Mosque, Topkapi Palace, Galata Tower
- Transportation: metro lines, tram, ferry, Istanbulkart, airport connections
- Culture: Byzantine heritage, Ottoman history, East-meets-West, traditions"""
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question about Istanbul: {user_input}"}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                if response.choices and response.choices[0].message.content:
                    gpt_response = response.choices[0].message.content.strip()
                    print(f"✅ GPT response generated successfully")
                    return {"response": enhance_ai_response_formatting(clean_text_formatting(gpt_response))}
                else:
                    print("⚠️ GPT returned empty content")
                    return {"response": create_fallback_response(user_input, [])}
                    
            except Exception as e:
                print(f"❌ OpenAI API error: {e}")
                return {"response": create_fallback_response(user_input, [])}
        else:
            # Use fallback responses for simple queries
            return {"response": create_fallback_response(user_input, [])}
            
    except Exception as e:
        print(f"❌ Error in ai_istanbul_router: {e}")
        traceback.print_exc()
        return {"response": "I apologize, but I encountered an error. Please try asking your question again."}

@app.post("/ai/stream")
async def ai_istanbul_streaming(request: Request):
    """Streaming AI endpoint for real-time responses"""
    try:
        data = await request.json()
        user_input = data.get("user_input", "")
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        # Get language from headers
        accept_language = request.headers.get("accept-language", "en")
        language = "tr" if "tr" in accept_language.lower() else "en"
        
        async def generate_response():
            try:
                # Enhanced input validation
                if not user_input or len(user_input.strip()) < 2:
                    response_data = json.dumps({'chunk': "I'm here to help you explore Istanbul! What would you like to know?"})
                    yield f"data: {response_data}\n\n"
                    return
                
                # Sanitize and enhance input
                enhanced_input = enhance_query_understanding(user_input)
                
                # Check for simple greetings
                if not should_use_gpt_for_query(enhanced_input):
                    welcome_message = "Hello! I'm your Istanbul guide. What would you like to know about the city?"
                    response_data = json.dumps({'chunk': welcome_message})
                    yield f"data: {response_data}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
                
                # Try fallback response first
                fallback_response = create_fallback_response(enhanced_input, [])
                
                if fallback_response and fallback_response.strip():
                    # Stream the fallback response
                    clean_response = clean_text_formatting(fallback_response)
                    
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
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    if openai_api_key:
                        try:
                            from openai import OpenAI
                            client = OpenAI(api_key=openai_api_key)
                            
                            system_prompt = """You are an expert Istanbul travel assistant. Provide helpful, accurate information about Istanbul's attractions, restaurants, transportation, culture, and travel tips. Be concise but informative."""
                            
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": enhanced_input}
                                ],
                                max_tokens=500,
                                temperature=0.7,
                                stream=True
                            )
                            
                            # Stream GPT response
                            for chunk in response:
                                if chunk.choices[0].delta.content:
                                    content = chunk.choices[0].delta.content
                                    response_data = json.dumps({'chunk': content})
                                    yield f"data: {response_data}\n\n"
                                    await asyncio.sleep(0.01)
                                    
                        except Exception as e:
                            print(f"GPT streaming error: {e}")
                            response_data = json.dumps({'chunk': 'I can help you explore Istanbul! Ask me about restaurants, museums, districts, or transportation.'})
                            yield f"data: {response_data}\n\n"
                    else:
                        response_data = json.dumps({'chunk': 'I can help you explore Istanbul! Ask me about restaurants, museums, districts, or transportation.'})
                        yield f"data: {response_data}\n\n"
                
                # Send completion signal
                completion_data = json.dumps({'done': True})
                yield f"data: {completion_data}\n\n"
                
            except Exception as e:
                print(f"Error in streaming endpoint: {e}")
                error_data = json.dumps({'error': 'Sorry, I encountered an error. Please try again.'})
                yield f"data: {error_data}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        print(f"[ERROR] Exception in /ai/stream endpoint: {e}")
        return {"error": "Streaming not available"}

# Start the server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
