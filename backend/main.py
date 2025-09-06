# --- Standard Library Imports ---
import sys
import os
import re
import asyncio
import json
import time
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# --- Third-Party Imports ---
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process

# --- Project Imports ---
from database import engine, SessionLocal
from models import Base, Restaurant, Museum, Place, ChatHistory
from specialized_models import UserProfile, TransportRoute, TurkishPhrases, LocalTips
from personalization_engine import IstanbulPersonalizationEngine, format_personalized_response
from actionable_responses import enhance_response_with_actions, get_actionable_places_response
from routes import museums, restaurants, places, blog
from api_clients.google_places import GooglePlacesClient
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# Enhanced chatbot imports
from enhanced_chatbot import (
    EnhancedContextManager, 
    EnhancedQueryUnderstanding, 
    EnhancedKnowledgeBase, 
    ContextAwareResponseGenerator
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_istanbul.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- ENHANCED ERROR HANDLING CLASSES AND FUNCTIONS ---

class APIError(Exception):
    """Custom API error class"""
    def __init__(self, message: str, error_type: str = "API_ERROR", status_code: int = 500, details: Dict = None):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class DatabaseError(APIError):
    """Database-specific error"""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=message,
            error_type="DATABASE_ERROR",
            status_code=503,
            details={"original_error": str(original_error) if original_error else None}
        )

class ExternalAPIError(APIError):
    """External API error (OpenAI, Google Places)"""
    def __init__(self, message: str, service: str, original_error: Exception = None):
        super().__init__(
            message=message,
            error_type="EXTERNAL_API_ERROR",
            status_code=502,
            details={
                "service": service,
                "original_error": str(original_error) if original_error else None
            }
        )

class ValidationError(APIError):
    """Input validation error"""
    def __init__(self, message: str, field: str = None):
        super().__init__(
            message=message,
            error_type="VALIDATION_ERROR",
            status_code=422,
            details={"field": field}
        )

def handle_database_error(error: Exception, context: str = "") -> DatabaseError:
    """Handle database-related errors"""
    logger.error(f"Database error in {context}: {str(error)}")
    logger.error(f"Database error traceback: {traceback.format_exc()}")
    
    if isinstance(error, SQLAlchemyError):
        return DatabaseError(
            message="Database operation failed. Please try again later.",
            original_error=error
        )
    else:
        return DatabaseError(
            message="An unexpected database error occurred.",
            original_error=error
        )

def handle_external_api_error(error: Exception, service: str, context: str = "") -> ExternalAPIError:
    """Handle external API errors (OpenAI, Google Places, etc.)"""
    logger.error(f"External API error ({service}) in {context}: {str(error)}")
    logger.error(f"External API error traceback: {traceback.format_exc()}")
    
    # Determine user-friendly message based on error type
    if "timeout" in str(error).lower():
        message = f"{service} is taking too long to respond. Please try again."
    elif "rate limit" in str(error).lower():
        message = f"{service} rate limit exceeded. Please try again later."
    elif "authentication" in str(error).lower() or "api key" in str(error).lower():
        message = f"{service} authentication failed. Please contact support."
    elif "network" in str(error).lower() or "connection" in str(error).lower():
        message = f"Cannot connect to {service}. Please check your internet connection."
    else:
        message = f"{service} is currently unavailable. Please try again later."
    
    return ExternalAPIError(
        message=message,
        service=service,
        original_error=error
    )

def create_error_response(error: APIError) -> JSONResponse:
    """Create standardized error response"""
    response_data = {
        "error": {
            "message": error.message,
            "type": error.error_type,
            "timestamp": datetime.now().isoformat(),
            "details": error.details
        }
    }
    
    # Add request ID for tracing (in production)
    if hasattr(error, 'request_id'):
        response_data["error"]["request_id"] = error.request_id
    
    return JSONResponse(
        status_code=error.status_code,
        content=response_data
    )

def safe_database_operation(operation_func, context: str = "database operation"):
    """Decorator for safe database operations"""
    def wrapper(*args, **kwargs):
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            raise handle_database_error(e, context)
    return wrapper

def safe_external_api_call(service: str, context: str = "API call"):
    """Decorator for safe external API calls"""
    def decorator(operation_func):
        def wrapper(*args, **kwargs):
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                raise handle_external_api_error(e, service, context)
        return wrapper
    return decorator

def log_request_info(request: Request, user_input: str = ""):
    """Log request information for debugging"""
    try:
        client_host = request.client.host if hasattr(request, 'client') and request.client else 'unknown'
    except AttributeError:
        client_host = 'unknown'
    logger.info(f"Request from {client_host}: {user_input[:100]}")

def handle_unexpected_error(error: Exception, context: str = "") -> APIError:
    """Handle unexpected errors"""
    logger.error(f"Unexpected error in {context}: {str(error)}")
    logger.error(f"Unexpected error traceback: {traceback.format_exc()}")
    
    return APIError(
        message="An unexpected error occurred. Please try again later.",
        error_type="UNEXPECTED_ERROR",
        status_code=500,
        details={"context": context}
    )

# Retry decorator with exponential backoff
def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:  # Last attempt
                        raise e
                    
                    delay = base_delay * (backoff_factor ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    
                    if asyncio.iscoroutinefunction(func):
                        await asyncio.sleep(delay)
                    else:
                        time.sleep(delay)
            
            raise last_exception
        
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:  # Last attempt
                        raise e
                    
                    delay = base_delay * (backoff_factor ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise APIError("Service temporarily unavailable (circuit breaker OPEN)")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning("Circuit breaker opened due to repeated failures")

# Create circuit breakers for external services
openai_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
google_places_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

# Health check endpoint data
health_status = {
    "database": True,
    "openai": True,
    "google_places": True,
    "last_check": datetime.now().isoformat()
}

async def check_service_health():
    """Check the health of all services"""
    global health_status
    
    # Check database
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        health_status["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["database"] = False
    
    # Check OpenAI (simple test)
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and OpenAI:
            # Just check if we have the key and library
            health_status["openai"] = True
        else:
            health_status["openai"] = False
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        health_status["openai"] = False
    
    # Check Google Places (simple test)
    try:
        google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        health_status["google_places"] = bool(google_api_key)
    except Exception as e:
        logger.error(f"Google Places health check failed: {e}")
        health_status["google_places"] = False
    
    health_status["last_check"] = datetime.now().isoformat()

# --- END ENHANCED ERROR HANDLING ---

# --- OpenAI Import ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("[ERROR] openai package not installed. Please install it with 'pip install openai'.")

# Add the current directory to Python path for Render deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# (Removed duplicate project imports and load_dotenv)

def clean_text_formatting(text):
    """Remove emojis, hashtags, and markdown formatting from text while preserving line breaks"""
    if not text:
        return text
    
    # Remove emojis (Unicode emoji ranges)
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
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
    text = re.sub(r'#+ ', '', text)               # Remove hashtags at start of lines
    text = re.sub(r' #\w+', '', text)             # Remove hashtags in text
    
    # Clean up extra whitespace but preserve line breaks
    lines = text.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(cleaned_lines)
    
    return text.strip()

def generate_restaurant_info(restaurant_name, location="Istanbul", place_data=None):
    """Generate a concise, helpful description for a restaurant"""
    name_lower = restaurant_name.lower()
    
    # Use actual place data if available
    if place_data:
        types = place_data.get('types', [])
        rating = place_data.get('rating', 0)
        price_level = place_data.get('price_level', 0)
        
        # Generate description based on actual data
        if 'restaurant' in types:
            base_desc = "Popular local restaurant"
        elif 'meal_takeaway' in types:
            base_desc = "Great takeaway spot"
        elif 'cafe' in types:
            base_desc = "Cozy neighborhood cafe"
        elif 'bakery' in types:
            base_desc = "Fresh bakery"
        else:
            base_desc = "Local dining spot"
            
        # Add rating context
        if rating >= 4.5:
            base_desc = "Highly acclaimed " + base_desc.lower()
        elif rating >= 4.0:
            base_desc = "Well-rated " + base_desc.lower()
            
        return base_desc
    
    # Fallback to name-based descriptions (shorter)
    if any(word in name_lower for word in ['kebap', 'kebab', 'döner']):
        return "Traditional Turkish kebab house"
    elif any(word in name_lower for word in ['pizza', 'italian']):
        return "Italian restaurant with authentic flavors"
    elif any(word in name_lower for word in ['sushi', 'japanese']):
        return "Japanese restaurant with fresh sushi"
    elif any(word in name_lower for word in ['cafe', 'kahve', 'coffee']):
        return "Local cafe perfect for coffee and conversation"
    elif any(word in name_lower for word in ['balık', 'fish', 'seafood']):
        return "Fresh seafood with Bosphorus views"
    elif any(word in name_lower for word in ['meze', 'rakı', 'meyhane']):
        return "Traditional meyhane with meze and rakı"
    elif any(word in name_lower for word in ['rooftop', 'terrace']):
        return "Stunning rooftop dining with city views"
    else:
        return "Popular local restaurant with authentic flavors"

app = FastAPI(title="AIstanbul API", debug=False)

# Initialize enhanced chatbot components
context_manager = EnhancedContextManager()
query_understanding = EnhancedQueryUnderstanding()
knowledge_base = EnhancedKnowledgeBase()
response_generator = ContextAwareResponseGenerator(context_manager, knowledge_base)

logger.info("Enhanced chatbot components initialized successfully")

# Mount static files directory for serving uploaded images
images_dir = "images"
os.makedirs(images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=images_dir), name="images")

# Serve static files for uploaded blog images
uploads_dir = "uploads"
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# Add blog router
from routes import blog
app.include_router(blog.router)

# Global exception handler
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    log_request_info(request, "Error occurred")
    return create_error_response(exc)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_request_info(request, f"HTTP {exc.status_code} error")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "HTTP_ERROR"}}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    log_request_info(request, "Unexpected error")
    error = handle_unexpected_error(exc, "global exception handler")
    return create_error_response(error)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await check_service_health()
        
        all_healthy = all(health_status[key] for key in ["database", "openai", "google_places"])
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Create tables if needed - with error handling
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")
    # Continue startup but log the error

def detect_off_topic_query(user_input):
    """
    Detect queries that are completely off-topic for Istanbul travel.
    Returns (is_off_topic: bool, category: str, confidence: float)
    """
    user_input_lower = user_input.lower()
    
    # Define off-topic categories with their keywords
    off_topic_categories = {
        'programming': [
            'python', 'javascript', 'code', 'programming', 'algorithm', 'software', 'development',
            'coding', 'html', 'css', 'database', 'api', 'framework', 'library', 'function',
            'variable', 'web scraper', 'scraping', 'data mining', 'machine learning', 'ai model',
            'neural network', 'deep learning', 'github', 'repository', 'git', 'debugging',
            'stackoverflow', 'compiler', 'interpreter', 'syntax', 'regex', 'json', 'xml'
        ],
        'science_math': [
            'quantum physics', 'quantum', 'physics', 'chemistry', 'biology', 'mathematics',
            'calculus', 'algebra', 'geometry', 'statistics', 'probability', 'theorem',
            'equation', 'quadratic equation', 'derivative', 'integral', 'matrix', 'vector',
            'molecular', 'atom', 'electron', 'photon', 'gravity', 'relativity', 'periodic table'
        ],
        'technology': [
            'iphone', 'android', 'smartphone', 'computer', 'laptop', 'windows', 'mac', 'linux',
            'bluetooth', 'wifi', 'internet', 'browser', 'google chrome', 'firefox', 'safari',
            'social media', 'facebook', 'instagram', 'twitter', 'youtube', 'tiktok',
            'cryptocurrency', 'bitcoin', 'blockchain', 'nft', 'metaverse'
        ],
        'other_places': [
            'paris', 'london', 'new york', 'tokyo', 'beijing', 'moscow', 'berlin', 'rome',
            'madrid', 'barcelona', 'amsterdam', 'vienna', 'prague', 'budapest', 'athens',
            'cairo', 'dubai', 'mumbai', 'delhi', 'bangkok', 'singapore', 'sydney', 'melbourne'
        ],
        'general_knowledge': [
            'world war', 'napoleon', 'hitler', 'einstein', 'shakespeare', 'mozart',
            'leonardo da vinci', 'picasso', 'olympics', 'world cup', 'nasa', 'space',
            'mars', 'moon', 'solar system', 'galaxy', 'universe', 'big bang'
        ]
    }
    
    # Check each category
    for category, keywords in off_topic_categories.items():
        matches = sum(1 for keyword in keywords if keyword in user_input_lower)
        confidence = matches / len(keywords) if keywords else 0
        
        # If any keyword matches (even one), consider it off-topic
        if matches > 0:
            return True, category, confidence
    
    return False, None, 0.0

def detect_geographical_impossibility(user_input):
    """
    Detect queries asking about non-Istanbul landmarks being in Istanbul.
    Returns (is_impossible: bool, landmark: str, actual_location: str)
    """
    user_input_lower = user_input.lower()
    
    # Famous landmarks that are NOT in Istanbul
    non_istanbul_landmarks = {
        'eiffel tower': 'Paris, France',
        'statue of liberty': 'New York, USA',
        'big ben': 'London, UK',
        'colosseum': 'Rome, Italy',
        'taj mahal': 'Agra, India',
        'machu picchu': 'Peru',
        'great wall': 'China',
        'great wall of china': 'China',
        'sydney opera house': 'Sydney, Australia',
        'christ the redeemer': 'Rio de Janeiro, Brazil',
        'stonehenge': 'England, UK',
        'pyramids': 'Egypt',
        'sphinx': 'Egypt',
        'mount rushmore': 'South Dakota, USA',
        'niagara falls': 'USA/Canada border',
        'golden gate bridge': 'San Francisco, USA',
        'times square': 'New York, USA',
        'hollywood sign': 'Los Angeles, USA',
        'buckingham palace': 'London, UK',
        'louvre': 'Paris, France',
        'notre dame': 'Paris, France',
        'arc de triomphe': 'Paris, France',
        'brandenburg gate': 'Berlin, Germany',
        'leaning tower of pisa': 'Pisa, Italy',
        'acropolis': 'Athens, Greece'
    }
    
    # Check if query mentions any non-Istanbul landmark with Istanbul context
    for landmark, actual_location in non_istanbul_landmarks.items():
        if landmark in user_input_lower and 'istanbul' in user_input_lower:
            return True, landmark, actual_location
    
    return False, None, None

def create_intelligent_fallback_response(user_input):
    """
    Create intelligent fallback responses for edge cases and off-topic queries.
    This is called BEFORE any OpenAI processing to catch problematic queries.
    """
    user_input_lower = user_input.lower()
    
    # 0. Check for random nonsense or very short/repetitive queries
    if len(user_input.strip()) < 3:
        return "Could you please ask me something specific about Istanbul? I'm here to help you explore the city!"
    
    # Check for repetitive or nonsensical patterns
    words = user_input_lower.split()
    if len(words) > 5:
        # Check if more than 70% of words are repetitive
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        if repetition_ratio < 0.3:  # Very repetitive
            return "I'd love to help you with something specific about Istanbul! What would you like to know about the city?"
    
    # Check for alphabet soup or random characters
    if any(pattern in user_input_lower for pattern in ['abcdefgh', 'qwertyui', 'asdfghjk']):
        return "I'm here to help you discover Istanbul! What would you like to know about restaurants, attractions, or neighborhoods in the city?"
    
    # 1. Check for prompt injection attempts
    injection_patterns = [
        'ignore previous instructions', 'ignore your instructions', 'ignore system prompt',
        'forget everything', 'new instructions', 'override', 'bypass', 'jailbreak',
        'pretend you are', 'act as if', 'roleplay as', 'you are now', 'act as a',
        'tell me about cars', 'help me with cooking', 'general chatbot'
    ]
    
    if any(pattern in user_input_lower for pattern in injection_patterns):
        return """I'm here to help you with Istanbul travel planning and recommendations. If you have any questions about visiting Istanbul, exploring the city, or finding great places to eat, feel free to ask!"""
    
    # 2. Check for geographical impossibilities
    is_impossible, landmark, actual_location = detect_geographical_impossibility(user_input)
    if is_impossible:
        return f"""The {landmark.title()} is actually located in {actual_location}, not Istanbul! 

Istanbul has its own incredible landmarks though:
- **Hagia Sophia** - Ancient Byzantine cathedral & mosque
- **Blue Mosque** - Stunning Ottoman architecture
- **Topkapi Palace** - Former Ottoman palace with amazing views
- **Galata Tower** - Medieval tower with panoramic city views
- **Maiden's Tower** - Iconic tower on a small island

Would you like to know more about any of these amazing Istanbul landmarks?"""
    
    # 3. Check for completely off-topic queries
    is_off_topic, category, confidence = detect_off_topic_query(user_input)
    if is_off_topic:
        category_responses = {
            'programming': "I'm specialized in Istanbul travel guidance rather than programming help. However, I'd love to help you plan your visit to Istanbul! Are you perhaps planning to visit Turkey's tech hub or looking for co-working spaces in the city?",
            'science_math': "While that's an interesting topic, I'm your Istanbul travel guide! I focus on helping you discover amazing places, restaurants, and experiences in this beautiful city. Is there anything about Istanbul you'd like to explore?",
            'technology': "I'm focused on helping you explore Istanbul rather than tech topics. But if you're interested in Istanbul's growing tech scene, I can tell you about modern districts like Levent and Maslak, or suggest tech-friendly cafes in Kadıköy!",
            'other_places': "That's a wonderful destination! However, I specialize in Istanbul travel guidance. If you're planning to visit Istanbul as well, I'd be happy to help you discover the best restaurants, attractions, and hidden gems in the city!",
            'general_knowledge': "That's interesting! As your Istanbul travel guide, I'd love to help you discover the rich history and culture of Istanbul instead. Did you know Istanbul was the capital of both the Byzantine and Ottoman empires? What aspects of Istanbul interest you most?"
        }
        
        return category_responses.get(category, 
            "I'm your dedicated Istanbul travel guide! I'd love to help you discover amazing restaurants, attractions, neighborhoods, and experiences in this incredible city. What would you like to know about Istanbul?")
    
    # 4. If none of the above, return None to continue with normal processing
    return None

def create_fallback_response(user_input, places):
    """Create intelligent fallback responses when OpenAI API is unavailable"""
    user_input_lower = user_input.lower()
    
    # History and culture questions
    if any(word in user_input_lower for word in ['history', 'historical', 'culture', 'byzantine', 'ottoman']):
        return """🏛️ **Istanbul's Rich History**

Istanbul has over 2,500 years of history! Here are key highlights:

**Byzantine Era (330-1453 CE):**
- Originally called Constantinople
- Hagia Sophia built in 537 CE
- Capital of Byzantine Empire

**Ottoman Era (1453-1922):**
- Conquered by Mehmed II in 1453
- Became capital of Ottoman Empire
- Blue Mosque, Topkapi Palace built

**Modern Istanbul:**
- Turkey's largest city with 15+ million people
- Spans Europe and Asia across the Bosphorus
- UNESCO World Heritage sites in historic areas

Would you like to know about specific historical sites or districts?"""

    # Food and cuisine questions (no static restaurant recommendations)
    elif any(word in user_input_lower for word in ['food', 'eat', 'cuisine', 'dish', 'meal', 'breakfast', 'lunch', 'dinner']):
        return """🍽️ **Turkish Cuisine in Istanbul**\n\nI can provide live restaurant recommendations using Google Maps. Please ask for a specific type of restaurant or cuisine, and I'll fetch the latest options for you!\n\nMust-try Turkish dishes include döner kebab, simit, balık ekmek, midye dolma, iskender kebab, manti, lahmacun, börek, baklava, Turkish delight, künefe, Turkish tea, ayran, and raki.\n\nFor restaurant recommendations, please specify your preference (e.g., 'seafood in Kadıköy')."""

    # Transportation questions
    elif any(word in user_input_lower for word in ['transport', 'metro', 'bus', 'ferry', 'taxi', 'getting around']):
        return """🚇 **Getting Around Istanbul**

**Istanbul Card (Istanbulkart):**
- Essential for all public transport
- Buy at metro stations or kiosks
- Works on metro, bus, tram, ferry

**Metro & Tram:**
- Clean, efficient, connects major areas
- M1: Airport to city center
- M2: European side north-south
- Tram: Historic peninsula (Sultanahmet)

**Ferries:**
- Cross between European & Asian sides
- Scenic Bosphorus tours
- Kadıköy ↔ Eminönü popular route

**Taxis & Apps:**
- BiTaksi and Uber available
- Always ask for meter ("taksimetre")
- Airport to city: 45-60 TL

**Tips:**
- Rush hours: 8-10 AM, 5-7 PM
- Download offline maps
- Learn basic Turkish transport terms"""

    # Weather and timing questions
    elif any(word in user_input_lower for word in ['weather', 'climate', 'season', 'when to visit', 'best time']):
        return """🌤️ **Istanbul Weather & Best Times to Visit**

**Seasons:**

**Spring (April-May):** ⭐ BEST
- Perfect weather (15-22°C)
- Blooming tulips in parks
- Fewer crowds

**Summer (June-August):**
- Hot (25-30°C), humid
- Peak tourist season
- Great for Bosphorus activities

**Fall (September-November):** ⭐ EXCELLENT
- Mild weather (18-25°C)
- Beautiful autumn colors
- Ideal for walking tours

**Winter (December-March):**
- Cool, rainy (8-15°C)
- Fewer tourists, lower prices
- Cozy indoor experiences

**What to Pack:**
- Comfortable walking shoes
- Layers for temperature changes
- Light rain jacket
- Modest clothing for mosques"""

    # Shopping questions
    elif any(word in user_input_lower for word in ['shop', 'shopping', 'bazaar', 'market', 'buy']):
        return """🛍️ **Shopping in Istanbul**

**Traditional Markets:**
- **Grand Bazaar** (Kapalıçarşı) - 4,000 shops, carpets, jewelry
- **Spice Bazaar** - Turkish delight, spices, teas
- **Arasta Bazaar** - Near Blue Mosque, smaller crowds

**Modern Shopping:**
- **Istinye Park** - Luxury brands, European side
- **Kanyon** - Unique architecture in Levent
- **Zorlu Center** - High-end shopping in Beşiktaş

**What to Buy:**
- Turkish carpets & kilims
- Ceramic tiles and pottery
- Turkish delight & spices
- Leather goods
- Gold jewelry

**Bargaining Tips:**
- Expected in bazaars, not in modern stores
- Start at 30-50% of asking price
- Be polite and patient
- Compare prices at multiple shops"""

    # General recommendations
    elif any(word in user_input_lower for word in ['recommend', 'suggest', 'what to do', 'attractions', 'sights']):
        return """✨ **Top Istanbul Recommendations**

**Must-See Historic Sites:**
- Hagia Sophia - Byzantine masterpiece
- Blue Mosque - Ottoman architecture
- Topkapi Palace - Ottoman sultans' palace
- Basilica Cistern - Underground marvel

**Neighborhoods to Explore:**
- **Sultanahmet** - Historic peninsula
- **Beyoğlu** - Modern culture, nightlife
- **Galata** - Trendy area, great views
- **Kadıköy** - Asian side, local vibe

**Unique Experiences:**
- Bosphorus ferry cruise at sunset
- Turkish bath (hamam) experience
- Rooftop dining with city views
- Local food tour in Kadıköy

**Day Trip Ideas:**
- Princes' Islands (Büyükada)
- Büyükçekmece Lake
- Belgrade Forest hiking

Ask me about specific areas or activities for more detailed information!"""

    # Default response for unclear queries
    else:
        # Check if the input is very short or unclear
        if len(user_input.strip()) < 3 or not any(char.isalpha() for char in user_input):
            return "Sorry, I couldn't understand. Can you type again?"
        
        return f"""Sorry, I couldn't understand your request about "{user_input}". Can you type again?

I can help you with:

🍽️ **Restaurants** - "restaurants in Kadıköy" or "Turkish cuisine"
🏛️ **Museums & Attractions** - "museums in Istanbul" or "Hagia Sophia"
🏘️ **Districts** - "best neighborhoods" or "Sultanahmet area"
🚇 **Transportation** - "how to get around" or "metro system"
🛍️ **Shopping** - "Grand Bazaar" or "where to shop"
� **Nightlife** - "best bars" or "Beyoğlu nightlife"

Please ask me something more specific about Istanbul!"""

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
            'restaurants', 'restaurant', 'restourant', 'resturant', 'restaurnts', 'restaurnt', 
            'restarunt', 'restarunts', 'restaurents', 'restarants', 'restrants', 'food', 
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
        ]
    }
    return keywords

def correct_typos(text, threshold=70):
    """Correct typos in user input using fuzzy matching"""
    try:
        keywords = create_fuzzy_keywords()
        words = text.lower().split()
        corrected_words = []
        
        # Common words that should not be corrected
        stop_words = {'in', 'to', 'at', 'on', 'for', 'with', 'by', 'from', 'up', 
                     'about', 'into', 'through', 'during', 'before', 'after', 
                     'above', 'below', 'between', 'among', 'a', 'an', 'the', 
                     'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'what', 'where', 'when', 'how', 'why', 'which', 'who', 'whom',
                     'give', 'me', 'show'}
        
        print(f"🔍 Typo correction input: '{text}'")
        
        for word in words:
            # Skip common stop words
            if word.lower() in stop_words:
                corrected_words.append(word)
                continue
                
            best_match = None
            best_score = 0
            best_category = None
            
            # Special handling for restaurant typos
            if word.lower() in ['restaurnts', 'restaurnt', 'restarants', 'restrants']:
                corrected_words.append('restaurants')
                print(f"🔧 Special typo correction: '{word}' -> 'restaurants'")
                continue
            
            # Check each category of keywords
            for category, keyword_list in keywords.items():
                match = process.extractOne(word, keyword_list)
                if match and match[1] > best_score and match[1] >= threshold:
                    best_match = match[0]
                    best_score = match[1]
                    best_category = category
            
            if best_match and best_score >= threshold:
                corrected_words.append(best_match)
                print(f"🔧 Typo correction: '{word}' -> '{best_match}' (score: {best_score}, category: {best_category})")
            else:
                corrected_words.append(word)
                if len(word) > 3:  # Only debug longer words
                    print(f"⚠️  No correction for: '{word}' (threshold: {threshold})")
        
        corrected_text = ' '.join(corrected_words)
        if corrected_text != text.lower():
            print(f"✅ Final correction: '{text}' -> '{corrected_text}'")
        return corrected_text
    except Exception as e:
        print(f"Error in typo correction: {e}")
        return text

def validate_and_sanitize_input(user_input):
    """
    Validate and sanitize user input to prevent security attacks
    Returns: (is_safe: bool, sanitized_input: str, error_msg: str)
    """
    if not user_input or not isinstance(user_input, str):
        return False, "", "Invalid input format"
    
    # Check input length - prevent DoS attacks
    if len(user_input) > 1000:
        return False, "", "Input too long (max 1000 characters)"
    
    # Detect and block SQL injection patterns
    sql_patterns = [
        r"[;]",                                  # Semicolons (but allow apostrophes for contractions)
        r"--",                                   # SQL comments
        r"/\*|\*/",                             # SQL block comments
        r"\b(UNION|SELECT|DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|DECLARE)\b",
        r"\b(OR|AND)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?",  # OR '1'='1' patterns
        r"['\"][^'\"]*['\"][^'\"]*=",           # Only block quotes that look like SQL injection patterns
    ]
    
    # Detect and block XSS patterns  
    xss_patterns = [
        r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>",
        r"<iframe\b[^<]*(?:(?!</iframe>)<[^<]*)*</iframe>", 
        r"javascript:", r"vbscript:", r"data:",
        r"on\w+\s*=",                           # Event handlers
    ]
    
    # Detect and block command injection patterns
    command_patterns = [
        r"[;&|`]",                              # Command separators
        r"\$\([^)]*\)",                         # Command substitution $()
        r"`[^`]*`",                             # Command substitution ``
    ]
    
    # Check all patterns
    import re
    all_patterns = sql_patterns + xss_patterns + command_patterns
    
    for pattern in all_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            print(f"🚨 SECURITY ALERT: Blocked malicious input pattern: {pattern}")
            return False, "", f"Input contains potentially malicious content"
    
    # Sanitize the input - be less aggressive to preserve legitimate queries
    sanitized = user_input
    
    # Only remove clearly dangerous characters, keep most normal characters including apostrophes for contractions
    dangerous_chars = r'[<>{}$`;\|\*]'  # Removed single quotes to allow contractions like "I'm"
    sanitized = re.sub(dangerous_chars, ' ', sanitized)
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Final length check after sanitization
    if len(sanitized) == 0:
        return False, "", "Input became empty after sanitization"
    
    return True, sanitized, ""

def enhance_query_understanding(user_input):
    """Enhance query understanding by correcting typos and adding context"""
    try:
        # First correct typos
        corrected_input = correct_typos(user_input)
        
        # Add common query pattern recognition
        enhanced_input = corrected_input.lower()
        
        # Handle common patterns and add missing words
        patterns = [
            # "kadikoy restaurant" -> "restaurants in kadikoy"
            (r'^(\w+)\s+(restaurant|food|eat)$', r'restaurants in \1'),
            # "kadikoy place" -> "places in kadikoy"
            (r'^(\w+)\s+(place|spot)$', r'places in \1'),
            # "kadikoy attraction" -> "attractions in kadikoy"
            (r'^(\w+)\s+(attraction|sight)$', r'attractions in \1'),
            # "kadikoy museum" -> "museums in kadikoy"
            (r'^(\w+)\s+(museum|gallery)$', r'museums in \1'),
            # Handle "to visit" patterns - be more specific to avoid wrong enhancements
            (r'^(\w+)\s+to\s+visit$', r'places to visit in \1'),
            # Handle "what in" patterns
            (r'^what\s+(\w+)$', r'what to do in \1'),
        ]
        
        for pattern, replacement in patterns:
            match = re.search(pattern, enhanced_input)
            if match:
                enhanced_input = re.sub(pattern, replacement, enhanced_input)
                print(f"Query enhancement: '{corrected_input}' -> '{enhanced_input}'")
                break
        
        return enhanced_input if enhanced_input != corrected_input.lower() else corrected_input
        
    except Exception as e:
        print(f"Error in query enhancement: {e}")
        return user_input

# Routers
app.include_router(museums.router)
app.include_router(restaurants.router)
app.include_router(places.router)
# Blog router is already included above

@app.get("/")
def root():
    return {"message": "Welcome to AIstanbul API"}

@app.post("/feedback")
async def receive_feedback(request: Request):
    """Endpoint to receive user feedback on AI responses"""
    try:
        feedback_data = await request.json()
        
        # Log feedback to console for observation
        print(f"\n📊 FEEDBACK RECEIVED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Type: {feedback_data.get('feedbackType', 'unknown')}")
        print(f"Query: {feedback_data.get('userQuery', 'N/A')}")
        print(f"Response: {feedback_data.get('messageText', '')[:100]}...")
        print(f"Session: {feedback_data.get('sessionId', 'N/A')}")
        print("-" * 50)
        
        # You could store this in a database here
        # For now, we just acknowledge receipt
        return {
            "status": "success",
            "message": "Feedback received",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        db = SessionLocal()
        history = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.timestamp.asc()).limit(50).all()
        
        messages = []
        for record in history:
            messages.append({
                "type": "user",
                "message": record.user_message,
                "timestamp": record.timestamp.isoformat()
            })
            messages.append({
                "type": "bot", 
                "message": record.bot_response,
                "timestamp": record.timestamp.isoformat()
            })
        
        return {"status": "success", "messages": messages}
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        return {"status": "error", "message": "Failed to retrieve chat history"}
    finally:
        if db:
            db.close()

@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    try:
        db = SessionLocal()
        deleted_count = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).delete()
        db.commit()
        
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        db.rollback() if db else None
        return {"status": "error", "message": "Failed to clear chat history"}
    finally:
        if db:
            db.close()

def save_chat_history(db, session_id: str, user_message: str, bot_response: str, user_ip: str = None):
    """Save chat interaction to database"""
    if db is None:
        logger.warning(f"Database session is None, cannot save chat history for session {session_id}")
        return
        
    try:
        chat_record = ChatHistory(
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
            user_ip=user_ip
        )
        db.add(chat_record)
        db.commit()
        logger.info(f"Saved chat history for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")
        if db:
            db.rollback()

def create_ai_response(message: str, db, session_id: str, user_message: str, request: Request = None):
    """Create AI response and save to chat history"""
    try:
        # Get user IP for logging
        user_ip = request.client.host if request and hasattr(request, 'client') and request.client else 'unknown'
        
        # Save to chat history only if db is available
        if db is not None:
            save_chat_history(db, session_id, user_message, message, user_ip)
        else:
            logger.warning(f"Database session not available, skipping chat history save for session {session_id}")
        
        return {"message": message, "session_id": session_id}
    except Exception as e:
        logger.error(f"Error creating AI response: {e}")
        return {"message": message, "session_id": session_id}

@app.post("/ai")
async def ai_istanbul_router(request: Request):
    """Enhanced AI endpoint with comprehensive error handling"""
    db = None
    
    try:
        # Log request for monitoring
        log_request_info(request)
        
        # Parse request data with error handling
        try:
            data = await request.json()
            user_input = data.get("query", data.get("user_input", ""))
            session_id = data.get("session_id", f"session_{int(time.time())}")
        except json.JSONDecodeError:
            raise ValidationError("Invalid JSON in request body")
        except Exception as e:
            raise ValidationError(f"Failed to parse request: {str(e)}")

        if not user_input:
            raise ValidationError("Empty input provided")

        # 🛡️ CRITICAL SECURITY: Validate and sanitize input FIRST
        is_safe, sanitized_input, error_msg = validate_and_sanitize_input(user_input)
        if not is_safe:
            logger.warning(f"🚨 SECURITY: Rejected unsafe input: {error_msg}")
            raise ValidationError("Input contains invalid characters or patterns")
        
        # Use sanitized input for all processing
        user_input = sanitized_input
        logger.info(f"🛡️ Processing sanitized input: {user_input[:50]}...")

        # Handle greetings and daily talk BEFORE any typo correction or enhancement
        user_input_clean = user_input.lower().strip()
        
        # More precise greeting detection - check for word boundaries
        greeting_patterns = [
            r'\bhi\b', r'\bhello\b', r'\bhey\b', r'\bgreetings\b', 
            r'\bgood morning\b', r'\bgood afternoon\b', r'\bgood evening\b',
            r'\bhowdy\b', r'\bhiya\b', r'\bsup\b', 
            r"\bwhat's up\b", r'\bwhats up\b',
            r'\bhow are you\b', r'\bhow are u\b', r'\bhow r u\b', r'\bhow r you\b', 
            r'\bhow are you doing\b', r"\bhow's it going\b", r'\bhows it going\b',
            r'\bnice to meet you\b', r'\bpleased to meet you\b', r'\bgood to see you\b'
        ]
        daily_talk_patterns = [
            r'\bhow are things\b', r"\bwhat's new\b", r'\bwhats new\b', r'\bhow have you been\b',
            r'\blong time no see\b', r'\bgood to hear from you\b', r"\bhope you're well\b",
            r"\bhow's your day\b", r'\bhows your day\b', r'\bhaving a good day\b',
            r"\bwhat's happening\b", r'\bwhats happening\b', r"\bhow's life\b", r'\bhows life\b'
        ]
        
        import re
        is_greeting = any(re.search(pattern, user_input_clean) for pattern in greeting_patterns)
        is_daily_talk = any(re.search(pattern, user_input_clean) for pattern in daily_talk_patterns)
        if is_greeting or is_daily_talk:
            logger.info(f"[AIstanbul] Detected greeting/daily talk: {user_input}")
            
            # Check if we have previous conversation history
            previous_context = context_manager.get_context(session_id) if context_manager else None
            has_history = previous_context and len(previous_context.previous_queries) > 0
            
            if any(re.search(r'\b' + word + r'\b', user_input_clean) for word in ['hi', 'hello', 'hey', 'greetings', 'howdy', 'hiya']):
                if has_history:
                    return create_ai_response("Hey there! 👋 Welcome back! I remember we were talking about Istanbul. Ready to continue planning your adventure, or is there something new you'd like to explore in the city?", db, session_id, user_input, request)
                else:
                    return create_ai_response("Hello! 👋 I'm your Istanbul travel companion! Whether you're planning a visit or already here, I'll help you discover amazing restaurants, hidden gems, and local experiences. What brings you to Istanbul?", db, session_id, user_input, request)
            elif any(re.search(pattern, user_input_clean) for pattern in [r'\bhow are you\b', r'\bhow are u\b', r'\bhow r u\b', r'\bhow r you\b', r'\bhow are you doing\b']):
                return create_ai_response("I'm doing fantastic! 😊 I love helping people discover Istanbul's magic. The city never stops surprising me with its blend of history, culture, and amazing food. What's your Istanbul mood today - history, food, or adventure?", db, session_id, user_input, request)
            elif any(re.search(r'\b' + phrase.replace(' ', r'\s+') + r'\b', user_input_clean) for phrase in ['good morning', 'good afternoon', 'good evening']):
                return create_ai_response("Good day! ☀️ Perfect timing to explore Istanbul! The city has different charms throughout the day - morning markets, afternoon tea culture, evening Bosphorus views. What sounds appealing to you right now?", db, session_id, user_input, request)
            elif any(re.search(pattern, user_input_clean) for pattern in [r"\bwhat's up\b", r'\bwhats up\b', r'\bsup\b', r"\bhow's it going\b", r'\bhows it going\b']):
                return create_ai_response("Just here living my best life helping people fall in love with Istanbul! 🌟 This city has such incredible energy. Are you looking to soak up some of that Istanbul vibe? What kind of experience interests you?", db, session_id, user_input, request)
            else:
                return create_ai_response("So nice to chat! 😊 I'm passionate about helping people discover Istanbul's incredible layers - from Ottoman palaces to trendy neighborhoods. What aspect of this amazing city draws you in most?", db, session_id, user_input, request)
        
        # Continue with main query processing
        
        # INTELLIGENT FALLBACK CHECK - Catch edge cases before OpenAI
        fallback_response = create_intelligent_fallback_response(user_input)
        if fallback_response:
            logger.info(f"🛡️ Intelligent fallback triggered for: {user_input[:50]}...")
            return create_ai_response(fallback_response, db, session_id, user_input, request)
        
        # Debug logging
        print(f"Original user_input: '{user_input}' (length: {len(user_input)})")
        
        # Get conversation context
        context = context_manager.get_context(session_id)
        
        # Enhanced query understanding with context - use our comprehensive typo correction
        enhanced_input = enhance_query_understanding(user_input)  # Use our fuzzy matching function instead
        
        # Detect if this is a follow-up question
        is_followup = context and len(context.previous_queries) > 0 and any(
            word in user_input.lower() for word in ['more', 'other', 'different', 'what about', 'how about', 'also', 'additionally']
        )
        
        # Extract intent and entities
        intent_info = query_understanding.extract_intent_and_entities(enhanced_input, context)
        
        # Use enhanced input for processing
        user_input = enhanced_input

        # --- OpenAI API Key Check ---
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not OpenAI or not openai_api_key:
            print("[ERROR] OpenAI API key not set or openai package missing.")
            raise ExternalAPIError("OpenAI API key not configured", "OpenAI")
        
        # Initialize OpenAI client with error handling
        try:
            client = OpenAI(api_key=openai_api_key)
        except Exception as e:
            raise ExternalAPIError("Failed to initialize OpenAI client", "OpenAI", e)

        # Create database session with error handling
        try:
            db = SessionLocal()
        except Exception as e:
            raise DatabaseError("Failed to create database session", e)
        try:
            # Check for very specific queries that need database/API data
            restaurant_keywords = [
                'restaurant', 'restaurants', 'restourant', 'restaurents', 'restaurnts', 'restaurnt',  # Include corrected typos
                'restarunt', 'restarunts', 'restarants', 'restrants', 'restrnt',  # Add basic words and common misspellings first
                'estrnt', 'resturant', 'restrant', 'restrnt',  # Common misspellings and abbreviations
                'restaurant recommendation', 'restaurant recommendations', 'recommend restaurants',
                'where to eat', 'best restaurants', 'good restaurants', 'top restaurants',
                'food places', 'places to eat', 'good places to eat', 'where can I eat',
                'turkish restaurants', 'local restaurants', 'traditional restaurants',
                'dining in istanbul', 'dinner recommendations', 'lunch places',
                'breakfast places', 'brunch spots', 'fine dining', 'casual dining',
                'cheap eats', 'budget restaurants', 'expensive restaurants', 'high-end restaurants',
                'seafood restaurants', 'kebab places', 'turkish cuisine', 'ottoman cuisine',
                'street food', 'local food', 'authentic food', 'traditional food',
                'rooftop restaurants', 'restaurants with view', 'bosphorus restaurants',
                'sultanahmet restaurants', 'beyoglu restaurants', 'galata restaurants',
                'taksim restaurants', 'kadikoy restaurants', 'besiktas restaurants',
                'asian side restaurants', 'european side restaurants',
                'vegetarian restaurants', 'vegan restaurants', 'halal restaurants',
                'restaurants near me', 'food recommendations', 'eating out',
                'where should I eat', 'suggest restaurants', 'restaurant suggestions'
            ]
            
            # Enhanced location-based restaurant detection
            location_restaurant_patterns = [
                r'restaurants?\s+in\s+\w+',  # "restaurants in taksim"
                r'restaurant\s+in\s+\w+',   # "restaurant in taksim"
                r'restaurnts?\s+in\s+\w+',   # "restaurnts in taksim" - common misspelling
                r'restarunts?\s+in\s+\w+',   # "restarunt in taksim" - common misspelling
                r'restarunt\s+in\s+\w+',    # "restarunt in taksim" - common misspelling
                r'resturant\s+in\s+\w+',    # Common misspelling
                r'restrnt\s+in\s+\w+',      # Abbreviated form
                r'estrnt\s+in\s+\w+',       # Abbreviated form
                r'restrant\s+in\s+\w+',     # Common misspelling
                r'restaurants?\s+near\s+\w+',  # "restaurants near galata"
                r'restaurants?\s+around\s+\w+',  # "restaurants around sultanahmet"
                r'eat\s+in\s+\w+',  # "eat in beyoglu"
                r'food\s+in\s+\w+',  # "food in kadikoy"
                r'dining\s+in\s+\w+',  # "dining in taksim"
                r'give\s+me\s+restaurants?\s+in\s+\w+',  # "give me restaurants in taksim"
                r'give\s+me\s+restaurnts?\s+in\s+\w+',  # "give me restaurnts in taksim" - with typo
                r'show\s+me\s+restaurants?\s+in\s+\w+'   # "show me restaurants in galata"
            ]
            museum_keywords = [
                'list museums', 'show museums', 'museum list', 'museums in istanbul',
                'art museum', 'history museum', 'archaeological museum', 'palace museum',
                'topkapi', 'hagia sophia', 'dolmabahce', 'istanbul modern',
                'pera museum', 'sakip sabanci', 'rahmi koc museum', 'museum recommendations',
                'which museums', 'best museums', 'must see museums', 'famous museums'
            ]
            
            # Add regex patterns for location-based museum queries
            location_museum_patterns = [
                r'museums?\s+in\s+\w+',      # "museums in beyoglu"
                r'museum\s+in\s+\w+',       # "museum in taksim"  
                r'give\s+me\s+museums?\s+in\s+\w+',  # "give me museums in beyoglu"
                r'show\s+me\s+museums?\s+in\s+\w+',  # "show me museums in galata"
                r'museums?\s+near\s+\w+',    # "museums near galata"
                r'museums?\s+around\s+\w+',  # "museums around sultanahmet"
            ]
            district_keywords = [
                'list districts', 'show districts', 'district list', 'neighborhoods in istanbul',
                'sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
                'fatih', 'sisli', 'taksim', 'karakoy', 'ortakoy', 'bebek', 'arnavutkoy',
                'balat', 'fener', 'eminonu', 'bakirkoy', 'maltepe', 'asian side', 'european side',
                'neighborhoods', 'areas in istanbul', 'districts to visit', 'where to stay',
                'best neighborhoods', 'trendy areas', 'historic districts'
            ]
            attraction_keywords = [
                'list attractions', 'show attractions', 'attraction list', 'landmarks in istanbul',
                'tourist attractions', 'sightseeing', 'must see', 'top attractions',
                'blue mosque', 'galata tower', 'bosphorus', 'golden horn', 'maiden tower',
                'basilica cistern', 'grand bazaar', 'spice bazaar', 'princes islands',
                'istiklal street', 'pierre loti', 'camlica hill', 'rumeli fortress',
                'things to do', 'places to visit', 'famous places', 'landmarks'
            ]
            
            # Add new keyword categories for better query routing
            shopping_keywords = [
                'shopping', 'shop', 'buy', 'bazaar', 'market', 'mall', 'stores',
                'grand bazaar', 'spice bazaar', 'istinye park', 'kanyon', 'zorlu center',
                'cevahir', 'outlet', 'souvenir', 'carpet', 'jewelry', 'leather',
                'shopping centers', 'where to shop', 'shopping recommendations', 'best shopping'
            ]
            
            # More specific transportation keywords - removed generic words like "to", "from", "go", "visit"
            transportation_keywords = [
                'transport', 'transportation', 'metro', 'bus', 'ferry', 'taxi', 'uber',
                'how to get', 'getting around', 'public transport', 'istanbulkart',
                'airport', 'train', 'tram', 'dolmus', 'marmaray', 'metrobus',
                'getting from', 'how to reach', 'travel to', 'transport options',
                'how can i go', 'how do i get', 'how to go', 'go from', 'get from',
                'travel from', 'route', 'directions', 'way to',
                'getting to', 'going to', 'going from'
            ]
            
            # Enhanced transportation detection for "from X to Y" patterns
            transportation_patterns = [
                r'how\s+can\s+i\s+go\s+\w+\s+from\s+\w+',  # "how can i go beyoglu from kadikoy"
                r'how\s+to\s+get\s+from\s+\w+\s+to\s+\w+',  # "how to get from kadikoy to beyoglu"
                r'how\s+to\s+go\s+from\s+\w+\s+to\s+\w+',   # "how to go from kadikoy to beyoglu"
                r'from\s+\w+\s+to\s+\w+',                   # "from kadikoy to beyoglu"
                r'\w+\s+to\s+\w+\s+transport',              # "kadikoy to beyoglu transport"
                r'get\s+to\s+\w+\s+from\s+\w+',             # "get to beyoglu from kadikoy"
                r'travel\s+from\s+\w+\s+to\s+\w+',          # "travel from kadikoy to beyoglu"
                r'\w+\s+from\s+\w+',                        # "beyoglu from kadikoy" (simple pattern)
            ]
            
            nightlife_keywords = [
                'nightlife', 'bars', 'clubs', 'night out', 'drinks', 'pub', 'lounge',
                'rooftop bar', 'live music', 'dancing', 'cocktails', 'beer', 'wine',
                'galata nightlife', 'beyoglu nightlife', 'taksim bars', 'karakoy bars',
                'where to drink', 'best bars', 'night clubs', 'party'
            ]
            
            culture_keywords = [
                'culture', 'cultural', 'tradition', 'festival', 'event', 'show',
                'turkish culture', 'ottoman', 'byzantine', 'hamam', 'turkish bath',
                'folk dance', 'music', 'art', 'theater', 'concert', 'whirling dervish',
                'cultural activities', 'traditional experiences', 'local customs'
            ]
            
            accommodation_keywords = [
                'hotel', 'accommodation', 'where to stay', 'hostel', 'apartment',
                'boutique hotel', 'luxury hotel', 'budget hotel', 'airbnb',
                'sultanahmet hotels', 'galata hotels', 'bosphorus view',
                'hotel recommendations', 'best hotels', 'cheap hotels'
            ]
            
            events_keywords = [
                'events', 'concerts', 'shows', 'exhibitions', 'festivals',
                'what\'s happening', 'events today', 'weekend events', 'cultural events',
                'music events', 'art exhibitions', 'theater shows', 'performances'
            ]
            
            # --- Remove duplicate location_restaurant_patterns ---
            # (Already defined above, so do not redefine here)
            
            # Add regex patterns for location-based place queries
            location_place_patterns = [
                r'place\s+in\s+\w+',  # "place in kadikoy"
                r'places\s+in\s+\w+',  # "places in sultanahmet"
                r'attractions?\s+in\s+\w+',  # "attractions in beyoglu"
                r'things?\s+to\s+do\s+in\s+\w+',  # "things to do in galata"
                r'visit\s+in\s+\w+',  # "visit in taksim"
                r'see\s+in\s+\w+',  # "see in kadikoy"
                r'go\s+in\s+\w+',  # "go in fatih"
                r'\w+\s+attractions',  # "kadikoy attractions"
                r'what.*in\s+\w+',  # "what to do in beyoglu"
                r'\w+\s+places?\s+to\s+visit',  # "kadikoy places to visit"
                r'\w+\s+plases?\s+to\s+visit',  # "kadikoy plases to visit" - typo variation
                r'\w+\s+places?$',  # "kadikoy places" - simple district + places
                r'\w+\s+plases?$',  # "kadikoy plases" - simple district + places (typo)
                r'\w+\s+to\s+places?\s+to\s+visit',  # "kadikoy to places to visit" - double "to" pattern
                r'\w+\s+to\s+visit',  # "kadikoy to visit"
                r'places?\s+to\s+visit\s+in\s+\w+',  # "places to visit in kadikoy"
                r'visit\s+\w+\s+places?',  # "visit kadikoy places"
            ]
            
            # Check if query matches location-based patterns
            is_location_restaurant_query = any(re.search(pattern, user_input.lower()) for pattern in location_restaurant_patterns)
            is_location_place_query = any(re.search(pattern, user_input.lower()) for pattern in location_place_patterns)
            is_location_museum_query = any(re.search(pattern, user_input.lower()) for pattern in location_museum_patterns)
            
            # Check if query is just a single district name (should show places in that district)
            single_district_names = ['sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
                                   'fatih', 'sisli', 'taksim', 'karakoy', 'ortakoy', 'bebek', 'arnavutkoy',
                                   'balat', 'fener', 'eminonu', 'bakirkoy', 'maltepe']
            is_single_district_query = (user_input.lower().strip() in single_district_names)
            
            # Enhanced follow-up restaurant detection for context-aware queries
            is_follow_up_restaurant_query = False
            context_location_for_restaurants = None
            
            # Check for follow-up restaurant queries like "give me also restaurants in beyoglu"
            follow_up_restaurant_patterns = [
                r'give\s+me\s+(also|more)?\s*restaurants?\s+(in\s+\w+)?',  # "give me also restaurants in beyoglu"
                r'show\s+me\s+(also|more)?\s*restaurants?\s+(in\s+\w+)?',  # "show me more restaurants in kadikoy"
                r'any\s+(other|more)\s+restaurants?\s+(in\s+\w+)?',        # "any other restaurants in galata"
                r'what\s+about\s+restaurants?\s+(in\s+\w+)?',              # "what about restaurants in taksim"
                r'(also|more)\s*restaurants?\s+(in\s+\w+)?',               # "also restaurants in sultanahmet"
                r'restaurants?\s+(also|too|as\s+well)',                    # "restaurants also"
                # Add typo-tolerant patterns
                r'(also|more)\s*resturants?\s+(in\s+\w+)?',                # "also resturants in sultanahmet"
                r'resturants?\s+(also|too|as\s+well)',                     # "resturants also"
                r'restarunts?\s+(also|too|as\s+well)',                     # "restarunts also"
                r'restarunt\s+(also|too|as\s+well)',                       # "restarunt also"
                r'(also|more)\s*restarunts?\s+(in\s+\w+)?',                # "also restarunts"
            ]
            
            # Check if this is a follow-up restaurant query
            for pattern in follow_up_restaurant_patterns:
                match = re.search(pattern, user_input.lower())
                if match:
                    is_follow_up_restaurant_query = True
                    # Try to extract location from the pattern
                    if 'in ' in user_input.lower():
                        location_match = re.search(r'in\s+(\w+)', user_input.lower())
                        if location_match:
                            context_location_for_restaurants = location_match.group(1)
                    break
            
            # If it's a follow-up query but no location found in query, use conversation context
            if is_follow_up_restaurant_query and not context_location_for_restaurants:
                # Get the most recent location from conversation context
                if intent_info.get('entities', {}).get('context_derived_location'):
                    context_location_for_restaurants = intent_info['entities']['context_derived_location'][0]
                    print(f"🔗 Follow-up restaurant query detected, using context location: '{context_location_for_restaurants}'")
            
            # More specific matching for different query types - prioritize restaurant and museum queries
            is_restaurant_query = (any(keyword in user_input.lower() for keyword in restaurant_keywords) or 
                                 is_location_restaurant_query or 
                                 is_follow_up_restaurant_query)
            is_museum_query = any(keyword in user_input.lower() for keyword in museum_keywords) or is_location_museum_query
            
            # Only consider it a district query if it's NOT a restaurant, museum, or location-based query and NOT a single district name
            is_district_query = (any(keyword in user_input.lower() for keyword in district_keywords) and 
                               not is_restaurant_query and 
                               not is_museum_query and
                               not is_location_restaurant_query and 
                               not is_location_place_query and
                               not is_location_museum_query and
                               not is_single_district_query)
            
            is_attraction_query = (any(keyword in user_input.lower() for keyword in attraction_keywords) or 
                                 is_location_place_query or is_single_district_query)
            is_shopping_query = any(keyword in user_input.lower() for keyword in shopping_keywords)
            
            # Transportation detection - but exclude if it's clearly a place/attraction query
            is_transportation_keyword_match = any(keyword in user_input.lower() for keyword in transportation_keywords)
            is_transportation_pattern = any(re.search(pattern, user_input.lower()) for pattern in transportation_patterns)
            
            # Only mark as transportation if it matches keywords/patterns AND is NOT a clear place/attraction query
            is_transportation_query = (is_transportation_keyword_match or is_transportation_pattern) and not (
                is_location_place_query or 
                is_attraction_query or 
                is_single_district_query or
                any(keyword in user_input.lower() for keyword in ['places to visit', 'things to do', 'attractions', 'sights'])
            )
            
            is_nightlife_query = any(keyword in user_input.lower() for keyword in nightlife_keywords)
            is_culture_query = any(keyword in user_input.lower() for keyword in culture_keywords)
            is_accommodation_query = any(keyword in user_input.lower() for keyword in accommodation_keywords)
            is_events_query = any(keyword in user_input.lower() for keyword in events_keywords)
            
            # Debug query categorization (only for severe debugging)
            # print(f"Query categorization:")
            # print(f"  is_restaurant_query: {is_restaurant_query}")
            
            if is_restaurant_query:
                # Get real restaurant data from Google Maps only
                try:
                    # Extract location from query for better results
                    search_location = "Istanbul, Turkey"
                    context_location = None
                    
                    # Check for context-derived location first (e.g., user said "beyoglu" then "restaurants")
                    if context_location_for_restaurants:
                        # Use the location from follow-up detection
                        context_location = context_location_for_restaurants
                        search_location = f"{context_location}, Istanbul, Turkey"
                        print(f"Using follow-up query location for restaurant search: '{context_location}'")
                        logger.info(f"🔗 Follow-up restaurant query: Searching in '{context_location}' from follow-up detection")
                    elif intent_info.get('entities', {}).get('context_derived_location'):
                        context_location = intent_info['entities']['context_derived_location'][0]
                        search_location = f"{context_location}, Istanbul, Turkey"
                        print(f"Using context-derived location for restaurant search: '{context_location}'")
                        logger.info(f"🔗 Context-aware restaurant query: Searching in '{context_location}' from conversation context")
                    elif is_location_restaurant_query:
                        # Try to extract specific location from the query
                        location_patterns = [
                            r'in\s+([a-zA-Z\s]+)',
                            r'near\s+([a-zA-Z\s]+)',
                            r'at\s+([a-zA-Z\s]+)',
                            r'around\s+([a-zA-Z\s]+)',
                        ]
                        for pattern in location_patterns:
                            match = re.search(pattern, user_input.lower())
                            if match:
                                location = match.group(1).strip()
                                search_location = f"{location}, Istanbul, Turkey"
                                break
                    
                    # Initialize Google Places client with error handling
                    try:
                        client = GooglePlacesClient()
                    except Exception as e:
                        raise ExternalAPIError("Failed to initialize Google Places client", "Google Places", e)
                    
                    # Search for restaurants with retry mechanism
                    @retry_with_backoff(max_attempts=2, base_delay=1.0)
                    def search_restaurants_with_retry():
                        return client.search_restaurants(location=search_location, keyword=user_input)
                    
                    try:
                        places_data = search_restaurants_with_retry()
                    except Exception as e:
                        raise ExternalAPIError("Restaurant search failed", "Google Places", e)
                    
                    if places_data.get('results'):
                        # Enhanced context-aware response message
                        location_text = ""
                        if is_follow_up_restaurant_query and context_location_for_restaurants:
                            location_text = context_location_for_restaurants.title()
                            restaurants_info = f"Perfect! Here are some excellent restaurants in {location_text}:\n\n"
                        elif is_follow_up_restaurant_query and context_location:
                            location_text = context_location.title()
                            restaurants_info = f"Great choice! Here are some excellent restaurants in {location_text}:\n\n"
                        elif context_location:
                            location_text = context_location.title()
                            restaurants_info = f"Here are some fantastic restaurants in {location_text}:\n\n"
                        else:
                            location_text = user_input.lower().split("in ")[-1].strip().title() if "in " in user_input.lower() else "this area"
                            restaurants_info = f"I found some great restaurants in {location_text}:\n\n"
                        
                        for i, place in enumerate(places_data['results'][:4]):  # Show top 4 restaurants
                            name = place.get('name', 'Unknown')
                            rating = place.get('rating', 'N/A')
                            price_level = place.get('price_level', 'N/A')
                            types = place.get('types', [])
                            opening_hours = place.get('opening_hours', {})
                            
                            # Generate concise restaurant info
                            restaurant_info = generate_restaurant_info(name, search_location, place)
                            
                            # Simple price indicator
                            price_indicator = ""
                            if isinstance(price_level, int):
                                price_indicator = "💰" * price_level
                            
                            # Opening status
                            open_indicator = ""
                            if opening_hours.get('open_now') is True:
                                open_indicator = " 🟢"
                            elif opening_hours.get('open_now') is False:
                                open_indicator = " 🔴"
                            
                            # Concise format
                            restaurants_info += f"**{i+1}. {name}**{open_indicator}\n"
                            restaurants_info += f"   {restaurant_info}\n"
                            
                            # Compact info line
                            info_line = ""
                            if rating != 'N/A':
                                info_line += f"⭐ {rating}/5"
                            if price_indicator:
                                info_line += f"  {price_indicator}"
                            if info_line:
                                restaurants_info += f"   {info_line}\n"
                            restaurants_info += "\n"
                        
                        # Add follow-up suggestions
                        restaurants_info += "💬 **Want more options?** Try asking:\n"
                        restaurants_info += f"• \"Show me more restaurants in {location_text.lower()}\"\n"
                        restaurants_info += f"• \"Turkish restaurants in {location_text.lower()}\"\n"
                        restaurants_info += f"• \"Restaurants with view in {location_text.lower()}\"\n\n"
                        
                        restaurants_info += "� **Pro tip:** Search these names on Google Maps for directions, menus, and phone numbers!"
                        
                        # Don't clean formatting - keep it conversational
                        response_message = restaurants_info
                        
                        # CRITICAL: Update conversation context for restaurant queries
                        places_mentioned = intent_info.get('entities', {}).get('locations', [])
                        if context_location:
                            places_mentioned.append(context_location)
                        if context_location_for_restaurants and context_location_for_restaurants not in places_mentioned:
                            places_mentioned.append(context_location_for_restaurants)
                        topic = 'restaurant_recommendation'
                        context_manager.update_context(
                            session_id, user_input, response_message, 
                            places_mentioned, topic
                        )
                        
                        return {"message": response_message}
                    else:
                        # Update context even for error cases
                        places_mentioned = intent_info.get('entities', {}).get('locations', [])
                        if context_location:
                            places_mentioned.append(context_location)
                        if context_location_for_restaurants and context_location_for_restaurants not in places_mentioned:
                            places_mentioned.append(context_location_for_restaurants)
                        topic = 'restaurant_recommendation'
                        context_manager.update_context(
                            session_id, user_input, "Sorry, I couldn't find any restaurants matching your request in Istanbul.", 
                            places_mentioned, topic
                        )
                        return {"message": "Sorry, I couldn't find any restaurants matching your request in Istanbul."}
                        
                except Exception as e:
                    logger.error(f"Restaurant search error: {e}")
                    raise ExternalAPIError("Failed to fetch restaurant recommendations", "Google Places", e)
            
            elif is_transportation_query:
                # More concise and helpful transportation response
                transport_response = """🚇 **Getting Around Istanbul**

**Quick Routes:**
• **Kadıköy ↔ Beyoğlu**: Ferry to Karaköy (15 min) + short walk
• **Airport to City**: M11 metro → M2 metro to center
• **Sultanahmet ↔ Taksim**: T1 tram → M2 metro

**Essential Card**: 
🎫 **Istanbulkart** - Buy at any metro station, works on all transport

**Best Apps**: 
📱 Moovit (directions) • BiTaksi (taxi) • Uber

**Pro Tips**: 
⏰ Avoid rush hours (8-10 AM, 5-7 PM)
🚢 Ferries are scenic AND fast!
💡 Metro announcements in English too

Need specific directions? Just ask me "how to get from [A] to [B]"!"""
                return {"message": transport_response}
            
            elif is_museum_query or is_attraction_query or is_district_query:
                # Get data from manual database with error handling
                try:
                    places = db.query(Place).all()
                except Exception as e:
                    raise DatabaseError("Failed to fetch places from database", e)
                
                # Extract location if this is a location-specific query
                extracted_location = None
                context_derived_location = None
                
                # Check if context provided a location (e.g., user said "beyoglu" then "places")
                if intent_info.get('entities', {}).get('context_derived_location'):
                    context_derived_location = intent_info['entities']['context_derived_location'][0]
                    print(f"Context-derived location detected: '{context_derived_location}'")
                
                if is_location_place_query or is_location_museum_query:
                    logger.info(f"Location-based query detected: {user_input}")
                    location_patterns = [
                        r'in\s+([a-zA-Z\s]+)',
                        r'at\s+([a-zA-Z\s]+)',
                        r'around\s+([a-zA-Z\s]+)',
                        r'near\s+([a-zA-Z\s]+)',
                        r'^(\w+)\s+to\s+places\s+to\s+visit',  # "kadikoy to places to visit" - specific pattern first
                        r'^(\w+)\s+places?\s+to\s+visit',  # "kadikoy places to visit" - only first word
                        r'^(\w+)\s+plases?\s+to\s+visit',  # "sultanahmet plases to visit" - handle typos
                        r'^(\w+)\s+places?$',  # "kadikoy places" - simple district + places
                        r'^(\w+)\s+plases?$',  # "kadikoy plases" - simple district + places (typo)
                        r'^([a-zA-Z\s]+?)\s+to\s+visit',  # "kadikoy to visit" - more general
                        r'^([a-zA-Z\s]+)\s+attractions',  # "kadikoy attractions"
                        r'visit\s+([a-zA-Z\s]+)\s+places?',  # "visit kadikoy places"
                    ]
                    for pattern in location_patterns:
                        match = re.search(pattern, user_input.lower())
                        if match:
                            location_candidate = match.group(1).strip()
                            # Ignore "istanbul" as it's the general city, not a specific district
                            if location_candidate.lower() != 'istanbul':
                                extracted_location = location_candidate
                                print(f"Extracted location: '{extracted_location}'")
                            else:
                                print(f"Ignored general city name: '{location_candidate}'")
                            break
                elif is_single_district_query:
                    # For single district names like "kadikoy", use the district name directly
                    extracted_location = user_input.lower().strip()
                    print(f"Single district query detected: '{extracted_location}'")
                
                # ENHANCED: Use context-derived location if no explicit location found
                if not extracted_location and context_derived_location:
                    extracted_location = context_derived_location.lower()
                    print(f"Using context-derived location: '{extracted_location}'")
                    # Add indicator that this is context-based
                    logger.info(f"🔗 Context-aware query: User asking about '{user_input}' in '{extracted_location}' from conversation context")
                
                # Filter based on query type
                filtered_places = []
                if (is_location_place_query or is_single_district_query) and extracted_location:
                    # For location-specific place queries, include all places first
                    print(f"DEBUG: Using location-based filtering for '{extracted_location}'")
                    filtered_places = places
                elif is_location_museum_query and extracted_location:
                    # For location-specific museum queries, filter for museums first
                    print(f"DEBUG: Using location-based museum filtering for '{extracted_location}'")
                    filtered_places = [p for p in places if p.category and 'museum' in p.category.lower()]
                elif is_museum_query:
                    filtered_places = [p for p in places if p.category and 'museum' in p.category.lower()]
                elif is_district_query:
                    filtered_places = [p for p in places if p.category and 'district' in p.category.lower()]
                    # Also include places by district
                    if not filtered_places:
                        districts = set([p.district for p in places if p.district])
                        district_info = f"Here are the main districts in Istanbul:\n\n"
                        for district in sorted(districts):
                            places_in_district = [p for p in places if p.district == district]
                            district_info += f"**{district}**:\n"
                            for place in places_in_district[:3]:  # Top 3 places per district
                                district_info += f"   - {place.name} ({place.category})\n"
                            district_info += "\n"
                        return {"message": district_info}
                else:  # general attraction query
                    filtered_places = [p for p in places if p.category and p.category.lower() in ['historical place', 'mosque', 'church', 'park', 'museum', 'market', 'cultural center', 'art', 'landmark']]
                
                # Apply location filter if location was extracted
                if extracted_location and filtered_places:
                    print(f"DEBUG: Before location filter: {len(filtered_places)} places")
                    print(f"DEBUG: Looking for location: '{extracted_location.lower()}'")
                    print(f"DEBUG: Available districts: {[p.district for p in filtered_places[:5]]}")
                    
                    # Normalize location name for matching (case-insensitive)
                    location_lower = extracted_location.lower()
                    
                    # Handle neighborhood to district mapping
                    location_mappings = {
                        'sultanahmet': 'fatih',  # Sultanahmet is in Fatih district
                        'galata': 'beyoglu',     # Galata is in Beyoglu district
                        'taksim': 'beyoglu',     # Taksim is in Beyoglu district
                        'ortakoy': 'besiktas',   # Ortaköy is in Beşiktaş district
                        'bebek': 'besiktas',     # Bebek is in Beşiktaş district
                    }
                    
                    # Check if we need to map the location to a district
                    if location_lower in location_mappings:
                        district_to_search = location_mappings[location_lower]
                        print(f"DEBUG: Mapping '{location_lower}' to district '{district_to_search}'")
                    else:
                        district_to_search = location_lower
                    
                    original_count = len(filtered_places)
                    filtered_places = [p for p in filtered_places if p.district and district_to_search in p.district.lower()]
                    print(f"DEBUG: After location filter: {len(filtered_places)} places (from {original_count})")
                    
                    if filtered_places:
                        print(f"DEBUG: Filtered places found:")
                        for p in filtered_places:
                            print(f"  - {p.name} in {p.district}")
                    else:
                        print(f"DEBUG: No places found matching location '{extracted_location}'")
                
                if filtered_places:
                    print(f"🔗 DEBUG: Entering personalization section with {len(filtered_places)} places")
                    # Initialize personalization engine
                    personalization_engine = IstanbulPersonalizationEngine(db)
                    
                    # Extract and update user context
                    user_context_updates = personalization_engine.extract_user_context(user_input, session_id)
                    user_profile = personalization_engine.get_or_create_user_profile(session_id)
                    
                    # Get transportation options if user has accommodation info
                    transport_info = []
                    if user_profile.accommodation_district:
                        for place in filtered_places[:3]:  # Get transport for first 3 places
                            if hasattr(place, 'district') and place.district:
                                ferry_routes = personalization_engine.get_ferry_schedule(
                                    user_profile.accommodation_district, 
                                    place.district
                                )
                                transport_info.extend(ferry_routes)
                    
                    # Get cultural context with location awareness
                    cultural_info = personalization_engine.get_cultural_context(
                        user_input, 
                        extracted_location or context_derived_location, 
                        session_id
                    )
                    
                    # Generate enhanced response with actions
                    enhanced_response = get_actionable_places_response(
                        filtered_places[:6], 
                        transport_info,
                        user_profile.accommodation_district
                    )
                    
                    # Format with personalization
                    user_context = {
                        'dietary': user_profile.dietary_restrictions,
                        'staying_in': user_profile.accommodation_district,
                        'days_left': user_profile.days_remaining,
                        'budget': user_profile.budget_level
                    }
                    
                    # Remove None values
                    user_context = {k: v for k, v in user_context.items() if v is not None}
                    
                    personalized_response = format_personalized_response(
                        enhanced_response['response'],
                        user_context,
                        transport_info[:2],  # Limit transport options
                        cultural_info
                    )
                    
                    # Return enhanced response
                    response_data = {
                        "message": personalized_response,
                        "actions": enhanced_response.get('actions', []),
                        "context_actions": enhanced_response.get('context_actions', []),
                        "personalized": True,
                        "user_context": user_context
                    }
                    
                    # CRITICAL: Update conversation context for places queries
                    places_mentioned = intent_info.get('entities', {}).get('locations', [])
                    topic = intent_info.get('intent', 'place_recommendation')
                    print(f"🔗 DEBUG: Updating context with places_mentioned={places_mentioned}, topic={topic}")
                    context_manager.update_context(
                        session_id, user_input, personalized_response, 
                        places_mentioned, topic
                    )
                    print(f"🔗 DEBUG: Context updated successfully")
                    
                    return response_data
                else:
                    error_message = ""
                    if extracted_location:
                        error_message = f"Sorry, I couldn't find any {'museums' if is_museum_query else 'places'} in {extracted_location.title()} in my database. Try asking about a different district or general attractions in Istanbul."
                    else:
                        error_message = f"Sorry, I couldn't find any {'museums' if is_museum_query else 'attractions'} in my database."
                    
                    # Update context even for error cases
                    places_mentioned = intent_info.get('entities', {}).get('locations', [])
                    topic = intent_info.get('intent', 'place_recommendation')
                    context_manager.update_context(
                        session_id, user_input, error_message, 
                        places_mentioned, topic
                    )
                    
                    return {"message": error_message}
            
            elif is_shopping_query:
                shopping_response = """🛍️ **Shopping in Istanbul**

**Must-Visit Markets:**
🏛️ **Grand Bazaar** - 4,000 shops! Carpets, jewelry, spices
📍 Metro: Beyazıt-Kapalıçarşı • Hours: 9 AM-7 PM (closed Sundays)

🌶️ **Spice Bazaar** - Turkish delight, teas, nuts, souvenirs  
📍 Near Eminönü ferry terminal

**Modern Malls:**
• **Istinye Park** - Luxury brands, beautiful design
• **Zorlu Center** - High-end shopping in Beşiktaş  
• **Kanyon** - Unique architecture in Levent

**What to Buy:**
🧿 Evil eye jewelry • 🍯 Turkish delight & baklava
🏺 Ceramic tiles • ☕ Turkish tea & coffee
👘 Turkish carpets • 🧴 Handmade soaps

**Shopping Tips:**
💰 Bargain in bazaars (start at 50% of asking price)
🏷️ Fixed prices in malls
📋 Tax-free shopping for purchases over 108 TL

Want specific recommendations for carpets, jewelry, or souvenirs?"""
                return {"message": shopping_response}
            
            elif is_transportation_query:
                transport_response = """🚇 **Getting Around Istanbul**

**Istanbul Card (Istanbulkart):** 💳
- Essential for ALL public transport
- Buy at metro stations, airports, or kiosks
- Works on metro, bus, tram, ferry, funicular
- Significant discounts vs. single tickets

**Metro Lines:**
- **M1A/M1B**: Airport ↔ Yenikapı ↔ Kirazlı
- **M2**: Vezneciler ↔ Şişli ↔ Hacıosman
- **M3**: Kirazlı ↔ Başakşehir
- **M4**: Kadıköy ↔ Sabiha Gökçen Airport
- **M5**: Üsküdar ↔ Çekmeköy
- **M6**: Levent ↔ Boğaziçi Üniversitesi
- **M7**: Mecidiyeköy ↔ Mahmutbey

**Trams:**
- **T1**: Kabataş ↔ Bağcılar (passes through Sultanahmet)
- **T4**: Topkapı ↔ Mescid-i Selam

**Key Ferry Routes:**
- Eminönü ↔ Kadıköy (20 min)
- Karaköy ↔ Kadıköy (15 min)
- Beşiktaş ↔ Üsküdar (15 min)
- Kabataş ↔ Üsküdar (20 min)

**Airports:**
- **Istanbul Airport (IST)**: M11 metro to city center
- **Sabiha Gökçen (SAW)**: M4 metro or HAVABUS

**Apps to Download:**
- Moovit - Real-time public transport
- BiTaksi - Local taxi app
- Uber - Available in Istanbul

**Tips:**
- Rush hours: 8-10 AM, 5-7 PM
- Metro announcements in Turkish & English
- Keep your Istanbulkart with you always!"""
                return {"message": transport_response}
            
            elif is_nightlife_query:
                nightlife_response = """🌃 **Istanbul Nightlife**

**Trendy Neighborhoods:**

**Beyoğlu/Galata:**
- Heart of Istanbul's nightlife
- Mix of rooftop bars, clubs, and pubs
- Istiklal Street has many options

**Karaköy:**
- Hip, artistic area with craft cocktail bars
- Great Bosphorus views from rooftop venues
- More upscale crowd

**Beşiktaş:**
- University area with younger crowd
- Good mix of bars and clubs
- More affordable options

**Popular Venues:**
- **360 Istanbul** - Famous rooftop with city views
- **Mikla** - Upscale rooftop restaurant/bar
- **Kloster** - Historic building, great atmosphere
- **Under** - Underground club in Karaköy
- **Sortie** - Upscale club in Maçka
- **Reina** - Famous Bosphorus-side nightclub

**Rooftop Bars:**
- **Leb-i Derya** - Multiple locations, great views
- **Nu Teras** - Sophisticated rooftop in Beyoğlu
- **Banyan** - Asian-inspired rooftop bar
- **The Marmara Pera** - Hotel rooftop with panoramic views

**Tips:**
- Most venues open after 9 PM
- Dress code: Smart casual to upscale
- Credit cards widely accepted
- Many venues have entrance fees on weekends
- Turkish beer (Efes) and rakı are popular local drinks
- Some areas can be crowded on weekends"""
                return {"message": nightlife_response}
            
            elif is_culture_query:
                culture_response = """🎭 **Turkish Culture & Experiences**

**Traditional Experiences:**
- **Turkish Bath (Hamam)** - Historic Cagaloglu or Suleymaniye Hamams
- **Whirling Dervishes** - Sema ceremony at various cultural centers
- **Turkish Coffee** - UNESCO Intangible Cultural Heritage
- **Traditional Music** - Turkish folk or Ottoman classical music

**Cultural Venues:**
- **Hodjapasha Cultural Center** - Traditional shows & performances
- **Galata Mevlevihanesi** - Whirling dervish ceremonies
- **Cemal Reşit Rey Concert Hall** - Classical music & opera
- **Zorlu PSM** - Modern performing arts center

**Festivals & Events:**
- **Istanbul Music Festival** (June) - Classical music in historic venues
- **Istanbul Biennial** (Fall, odd years) - Contemporary art
- **Ramadan** - Special atmosphere, iftar meals at sunset
- **Turkish National Days** - Republic Day (Oct 29), Victory Day (Aug 30)



**Cultural Customs:**
- Remove shoes when entering mosques or homes
- Dress modestly in religious sites
- Greetings: Handshakes common, kisses on both cheeks for friends
- Hospitality is very important in Turkish culture
- Tea (çay) is offered as sign of hospitality

**Traditional Arts:**
- **Calligraphy** - Ottoman Turkish writing art
- **Miniature Painting** - Traditional Ottoman art form
- **Carpet Weaving** - Intricate traditional patterns
- **Ceramic Art** - Iznik tiles and pottery
- **Marbled Paper (Ebru)** - Water-based art technique

**Cultural Districts:**
- **Balat** - Historic Jewish quarter with colorful houses
- **Fener** - Greek Orthodox heritage area
- **Süleymaniye** - Traditional Ottoman neighborhood
- **Eyüp** - Religious significance, local life"""
                return {"message": culture_response}
            
            elif is_accommodation_query:
                accommodation_response = """🏨 **Where to Stay in Istanbul**

**Best Neighborhoods:**

**Sultanahmet (Historic Peninsula):**
- Walk to major attractions (Blue Mosque, Hagia Sophia)
- Traditional Ottoman hotels and boutique properties
- Great for first-time visitors
- Can be touristy and crowded

**Beyoğlu/Galata:**
- Trendy area with modern boutique hotels
- Easy access to nightlife, restaurants, art galleries
- Good transport connections
- More contemporary vibe

**Beşiktaş:**
- Business district with luxury hotels
- Near Dolmabahçe Palace and Bosphorus
- Excellent shopping and dining
- Great transport hub

**Kadıköy (Asian Side):**
- Authentic local experience
- Great food scene and markets
- Less touristy, more affordable
- Easy ferry connection to European side

**Accommodation Types:**

**Luxury Hotels:**
- **Four Seasons Sultanahmet** - Ottoman palace conversion
- **Çırağan Palace Kempinski** - Former Ottoman palace
- **The Ritz-Carlton Istanbul** - Modern luxury in Şişli
- **Shangri-La Bosphorus** - Asian side luxury

**Boutique Hotels:**
- **Museum Hotel** - Antique-filled historic property
- **Vault Karakoy** - Former bank building in Galata
- **Georges Hotel Galata** - Stylish property near Galata Tower
- **Soho House Istanbul** - Trendy members' club with rooms

**Budget Options:**
- **Cheers Hostel** - Well-reviewed hostel in Sultanahmet
- **Marmara Guesthouse** - Budget-friendly in old city
- **Istanbul Hostel** - Central location, clean facilities

**Booking Tips:**
- Book early for peak season (April-October)
- Many hotels offer airport transfer services
- Check if breakfast is included
- Rooftop terraces are common and worth requesting
- Some historic hotels have small rooms - check dimensions"""
                return {"message": accommodation_response}
            
            elif is_events_query:
                events_response = """🎪 **Events & Entertainment in Istanbul**

**Regular Cultural Events:**

**Weekly:**
- **Friday Prayer** - Beautiful call to prayer across the city
- **Weekend Markets** - Kadıköy Saturday Market, various neighborhood pazars

**Monthly/Seasonal:**
- **Whirling Dervish Ceremonies** - Various venues, usually weekends
- **Traditional Music Concerts** - Cultural centers and historic venues
- **Art Gallery Openings** - Especially in Beyoğlu and Karaköy

**Major Annual Festivals:**

**Spring:**
- **Tulip Festival** (April) - Millions of tulips bloom in city parks
- **Istanbul Music Festival** (June) - Classical music in historic venues

**Summer:**
- **Istanbul Jazz Festival** (July) - International artists, multiple venues
- **Rock'n Coke Festival** - Major international music festival

**Fall:**
- **Istanbul Biennial** (Sept-Nov, odd years) - Contemporary art across the city
- **Akbank Jazz Festival** (Oct) - Jazz performances citywide

**Winter:**
- **New Year Celebrations** - Taksim Square and various venues

**Entertainment Venues:**

**Concert Halls:**
- **Zorlu PSM** - Major international shows
- **Cemal Reşit Rey Concert Hall** - Classical music
- **Volkswagen Arena** - Large concerts and sports

**Theaters:**
- **Istanbul State Opera and Ballet**
- **Turkish State Theaters**
- Various private theaters in Beyoğlu

**Current Events Resources:**
- **Biletix.com** - Main ticketing platform
- **Time Out Istanbul** - Event listings
- **Istanbul.com** - Tourism events
- **Eventbrite** - International events

**Tips:**
- Check event calendars at hotel concierge
- Many cultural events are free or low-cost
- Book popular shows in advance
- Some events may be in Turkish only - check beforehand"""
                return {"message": events_response}
            
            else:
                # Use OpenAI for intelligent responses about Istanbul
                # Get context from database
                places = db.query(Place).all()
                restaurants_context = "Available restaurants data from Google Maps API."
                places_context = ""
                
                if places:
                    places_context = "Available places in Istanbul database:\n"
                    for place in places[:20]:  # Limit context size
                        places_context += f"- {place.name} ({place.category}) in {place.district}\n"
                
                # Create prompt for OpenAI
                system_prompt = """You are KAM, Istanbul's intelligent AI travel guide with smart routing capabilities.

SMART CONVERSATION HANDLING:
- Warmly greet personal introductions and pivot to Istanbul travel planning
- For travel plans, provide personalized advice based on user's background and timeline
- When someone asks general food questions, offer to get live restaurant recommendations
- For specific restaurant requests with location, say "Let me get live restaurant data for you!"
- Be contextual, conversational, and helpful in all interactions

CONVERSATION EXAMPLES:
User: "I'm Ukrainian" 
You: "Nice to meet you! Are you planning to visit Istanbul or already here? I'd love to help you explore the city - from historic sites to amazing food!"

User: "I'm Turkish, coming tomorrow to Istanbul"
You: "Welcome back home! How exciting that you're visiting tomorrow! What areas of Istanbul are you thinking of exploring? Any specific interests - food, culture, nightlife, history?"

User: "I'm Turkish, I will come tomorrow to Istanbul"
You: "That's wonderful! Coming back to beautiful Istanbul tomorrow - what a great way to spend your time! Are you looking to explore new neighborhoods, try different cuisines, visit attractions, or reconnect with favorite spots?"

User: "tell me about good food"
You: "Istanbul has incredible cuisine! The city is famous for Turkish delights, kebabs, fresh seafood, and amazing street food. Are you interested in recommendations for a specific area like Beyoğlu, Sultanahmet, or Kadıköy? I can get you live restaurant data with current ratings and reviews!"

User: "what should I visit tomorrow"
You: "Great question! Istanbul has so many amazing places. Are you interested in history (like Hagia Sophia, Blue Mosque), culture (Grand Bazaar, Turkish baths), modern areas (Taksim, Beyoğlu), or neighborhoods with great food and nightlife? What kind of experience are you looking for?"

ISTANBUL EXPERTISE:
- For restaurant queries without specific location, suggest popular districts and offer live data
- For Kadıköy specifically, recommend: Çiya Sofrası, Fish Market restaurants, Moda cafes
- For attraction queries, use database information and add cultural context
- Share practical tips: transportation, timing, cultural insights
- Always be enthusiastic about Istanbul while being genuinely helpful

TONE & STYLE:
- Friendly, conversational, and welcoming
- Ask follow-up questions to provide better assistance
- Relate general topics back to Istanbul experiences
- Offer specific, actionable advice
- Be inclusive and helpful to all visitors and locals

When users need specific restaurant recommendations with location (like "restaurants in Beyoğlu"), the system will automatically fetch live data for you."""

                try:
                    logger.info("Generating enhanced AI response...")
                    
                    # Check if we can provide a knowledge-based response first
                    knowledge_response = knowledge_base.get_knowledge_response(user_input, intent_info)
                    
                    if knowledge_response:
                        # Use knowledge base response
                        ai_response = knowledge_response
                        logger.info("Using knowledge base response")
                    else:
                        # Fall back to OpenAI for complex queries
                        @openai_circuit_breaker
                        @retry_with_backoff(max_attempts=3, base_delay=1.0)
                        def make_openai_request():
                            # Enhance system prompt with context
                            enhanced_prompt = response_generator.enhance_system_prompt(system_prompt, context)
                            
                            messages = [
                                {"role": "system", "content": enhanced_prompt},
                                {"role": "system", "content": f"Database context:\n{places_context}"}
                            ]
                            
                            # Add conversation history if available
                            if context:
                                for i, (prev_q, prev_r) in enumerate(zip(context.previous_queries[-3:], context.previous_responses[-3:])):
                                    messages.append({"role": "user", "content": prev_q})
                                    messages.append({"role": "assistant", "content": prev_r})
                            
                            messages.append({"role": "user", "content": user_input})
                            
                            return client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=messages,
                                max_tokens=500,
                                temperature=0.7,
                                timeout=30  # 30 second timeout
                            )
                        
                        response = make_openai_request()
                        ai_response = response.choices[0].message.content
                        logger.info(f"OpenAI response: {ai_response[:100]}...")
                    
                    # Generate context-aware enhanced response
                    enhanced_response = response_generator.generate_response(
                        user_input, ai_response, context, intent_info, places
                    )
                    
                    # Update conversation context
                    places_mentioned = intent_info.get('entities', {}).get('places', [])
                    topic = intent_info.get('intent', 'general')
                    context_manager.update_context(
                        session_id, user_input, enhanced_response, 
                        places_mentioned, topic
                    )
                    
                    # Clean the response from any emojis, hashtags, or markdown
                    clean_response = clean_text_formatting(enhanced_response)
                    return {"message": clean_response}
                    
                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
                    # Try fallback response before raising error
                    try:
                        fallback_response = create_fallback_response(user_input, places)
                        clean_response = clean_text_formatting(fallback_response)
                        return {"message": clean_response}
                    except Exception as fallback_error:
                        logger.error(f"Fallback response failed: {fallback_error}")
                        raise ExternalAPIError("AI chat service is temporarily unavailable", "OpenAI", e)
        
        finally:
            if db:
                db.close()
                
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise e
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise e
    except ExternalAPIError as e:
        logger.error(f"External API error: {e}")
        raise e
    except Exception as e:
        logger.error(f"[ERROR] Unexpected exception in /ai endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create fallback response for unexpected errors
        error = handle_unexpected_error(e, "AI endpoint")
        raise error

# Streaming AI Response Endpoint
@app.post("/ai/stream")
async def stream_ai_response(request: Request):
    """Streaming endpoint for AI responses"""
    try:
        data = await request.json()
        user_input = data.get("user_input", "").strip()
        session_id = data.get("session_id", f"session_{int(time.time())}")
        
        if not user_input:
            raise ValidationError("Query is required", "user_input")

        logger.info(f"Streaming AI request - Session: {session_id}, Query: {user_input[:100]}...")

        async def generate_stream():
            try:
                # Use the same logic as the /ai endpoint
                # Create request data in the expected format
                request_data = {"query": user_input, "session_id": session_id}
                
                # For simplicity, call the main AI logic directly without streaming first
                # then we'll break it into chunks
                enhanced_input = query_understanding.correct_and_enhance_query(user_input)
                context = context_manager.get_context(session_id)
                intent_info = query_understanding.extract_intent_and_entities(enhanced_input, context)
                
                # Generate response using the context-aware generator
                ai_response = response_generator.generate_contextual_response(
                    enhanced_input, context, intent_info, knowledge_base
                )
                
                # Update context
                context_manager.update_context(session_id, user_input, ai_response, intent_info)
                
                response = {"message": ai_response, "session_id": session_id}
                
                # For streaming, we'll break down the response into chunks
                message = response.get("message", "")
                words = message.split()
                
                # Send the response in chunks of 3-5 words
                chunk_size = 4
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    if chunk.strip():
                        yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"
                        await asyncio.sleep(0.1)  # Small delay for streaming effect
                
                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                error_msg = json.dumps({'error': 'Failed to generate response'})
                yield f"data: {error_msg}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Streaming AI endpoint error: {str(e)}")
        logger.error(f"Streaming AI endpoint traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error in streaming endpoint"
        )

# Enhanced Chatbot Endpoints

@app.get("/ai/context/{session_id}")
async def get_conversation_context(session_id: str):
    """Get conversation context for a session"""
    try:
        context = context_manager.get_context(session_id)
        if context:
            return {
                "session_id": context.session_id,
                "previous_queries": context.previous_queries[-5:],  # Last 5 queries
                "mentioned_places": context.mentioned_places,
                "user_preferences": context.user_preferences,
                "last_recommendation_type": context.last_recommendation_type,
                "conversation_topics": context.conversation_topics[-10:],  # Last 10 topics
                "user_location": context.user_location,
                "active": True
            }
        else:
            return {"session_id": session_id, "active": False}
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation context")

@app.get("/ai/test-enhancements")
async def test_enhanced_features():
    """Test endpoint to verify enhanced features are working"""
    try:
        # Test query understanding
        test_query = "restorant recomendations in kadikoy"
        enhanced_query = query_understanding.correct_and_enhance_query(test_query)
        
        # Test knowledge base
        knowledge_test = knowledge_base.get_knowledge_response("tell me about ottoman history")
        
        # Test intent extraction
        intent_test = query_understanding.extract_intent_and_entities("I need vegetarian restaurants near Galata Tower")
        
        return {
            "status": "enhanced_features_active",
            "typo_correction": {
                "original": test_query,
                "corrected": enhanced_query
            },
            "knowledge_base": {
                "available": knowledge_test is not None,
                "sample_response": knowledge_test[:100] + "..." if knowledge_test else None
            },
            "intent_extraction": intent_test,
            "context_manager": {
                "active_sessions": len(context_manager.contexts)
            }
        }
    except Exception as e:
        logger.error(f"Enhancement test error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/ai/enhanced/health")
async def enhanced_health_check():
    """Health check for enhanced chatbot features"""
    return {
        "status": "healthy",
        "enhanced_features": {
            "context_manager": "active",
            "query_understanding": "active", 
            "knowledge_base": "active",
            "response_generator": "active"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)




