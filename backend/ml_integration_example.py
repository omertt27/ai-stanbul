"""
Example: How to Integrate ML Service into Main Backend

Add this code to your backend/main.py or app.py
"""

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Add imports at the top of your file
# ═══════════════════════════════════════════════════════════════════

from backend.ml_service_client import get_ml_answer, get_ml_status, check_ml_health


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Create Request/Response models (if you don't have them)
# ═══════════════════════════════════════════════════════════════════

from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lon}")
    use_llm: bool = Field(False, description="Use LLM for detailed response (slower)")
    language: str = Field("en", description="Response language (en/tr)")
    user_id: Optional[str] = Field(None, description="User ID for personalization")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Bot response text")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Confidence score")
    method: str = Field(..., description="Response generation method")
    context: List[Dict] = Field(default=[], description="Context items used")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    response_time: float = Field(..., description="Response time in seconds")
    ml_service_used: bool = Field(..., description="Whether ML service was used")


# ═══════════════════════════════════════════════════════════════════
# STEP 3: Create the chat endpoint
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with ML integration
    
    Workflow:
    1. Try ML service first (semantic search + templates or LLM)
    2. If ML unavailable or fails, use existing fallback logic
    3. Return unified response format
    """
    start_time = time.time()
    
    try:
        # Optional: Use your existing intent classifier
        # detected_intent = await your_intent_classifier(request.message)
        detected_intent = "general"  # Default
        
        # Try ML service first
        logger.info(f"💬 Chat query: '{request.message}' (intent: {detected_intent})")
        
        ml_response = await get_ml_answer(
            query=request.message,
            intent=detected_intent,
            user_location=request.user_location,
            use_llm=request.use_llm,
            language=request.language
        )
        
        if ml_response and ml_response.get('success'):
            # ML service succeeded! ✅
            logger.info(f"✅ Using ML service ({ml_response.get('generation_method')})")
            
            return ChatResponse(
                response=ml_response['answer'],
                intent=ml_response.get('intent', detected_intent),
                confidence=ml_response.get('confidence', 0.8),
                method=f"ml_{ml_response.get('generation_method', 'unknown')}",
                context=ml_response.get('context', []),
                suggestions=ml_response.get('suggestions', []),
                response_time=time.time() - start_time,
                ml_service_used=True
            )
        
        # Fallback to existing logic
        logger.info("⚠️ ML service unavailable - using fallback")
        
        fallback_response = await generate_fallback_response(
            message=request.message,
            intent=detected_intent,
            user_location=request.user_location
        )
        
        return ChatResponse(
            response=fallback_response['answer'],
            intent=detected_intent,
            confidence=0.6,
            method="fallback",
            context=fallback_response.get('context', []),
            suggestions=generate_traditional_suggestions(detected_intent),
            response_time=time.time() - start_time,
            ml_service_used=False
        )
    
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════
# STEP 4: Add fallback response function
# ═══════════════════════════════════════════════════════════════════

async def generate_fallback_response(
    message: str,
    intent: str,
    user_location: Optional[Dict] = None
) -> Dict:
    """
    Generate response using existing rule-based logic
    Replace this with your actual implementation
    """
    
    # Example fallback responses
    fallback_responses = {
        "restaurant_recommendation": {
            "answer": "Istanbul has amazing restaurants! Popular areas include Beyoğlu, Kadıköy, and Beşiktaş. What type of cuisine are you interested in?",
            "context": []
        },
        "attraction_query": {
            "answer": "Istanbul is full of incredible attractions! The must-sees include Hagia Sophia, Blue Mosque, Topkapı Palace, and the Grand Bazaar. Which area would you like to explore?",
            "context": []
        },
        "transportation_help": {
            "answer": "Istanbul has excellent public transportation including metro, tram, ferry, and buses. You can use an Istanbulkart for all of them. Where do you need to go?",
            "context": []
        },
        "general": {
            "answer": "I'm here to help you explore Istanbul! I can recommend restaurants, attractions, help with transportation, and suggest local experiences. What would you like to know?",
            "context": []
        }
    }
    
    # Return appropriate fallback
    return fallback_responses.get(intent, fallback_responses["general"])


def generate_traditional_suggestions(intent: str) -> List[str]:
    """Generate follow-up suggestions based on intent"""
    
    suggestions = {
        "restaurant_recommendation": [
            "Show me vegetarian restaurants",
            "What about seafood options?",
            "Budget-friendly restaurants near me"
        ],
        "attraction_query": [
            "Tell me about museums",
            "Historical sites in Sultanahmet",
            "Best views in Istanbul"
        ],
        "transportation_help": [
            "How to use the metro?",
            "Ferry schedules",
            "Taxi vs Uber in Istanbul"
        ],
        "general": [
            "Best restaurants in Istanbul",
            "Top attractions to visit",
            "How to get around the city"
        ]
    }
    
    return suggestions.get(intent, suggestions["general"])


# ═══════════════════════════════════════════════════════════════════
# STEP 5: Add health/status endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/v1/ml/status", tags=["System Health"])
async def ml_service_status():
    """Get ML service status and health"""
    status = await get_ml_status()
    return status


@app.get("/api/v1/ml/health", tags=["System Health"])
async def ml_service_health():
    """Quick health check for ML service"""
    health = await check_ml_health()
    return health


# ═══════════════════════════════════════════════════════════════════
# STEP 6: Add to your main health endpoint
# ═══════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System Health"])
async def health_check():
    """
    Overall system health check
    Update your existing health endpoint to include ML service
    """
    ml_health = await check_ml_health()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "database": "healthy",  # Your DB check
            "ml_service": "healthy" if ml_health["healthy"] else "degraded"
        },
        "ml_service_details": ml_health
    }


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE USAGE IN EXISTING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

# If you have existing endpoints that could benefit from ML:

@app.get("/api/recommendations/{category}")
async def get_recommendations(category: str, query: Optional[str] = None):
    """
    Example: Enhance existing recommendation endpoint with ML
    """
    if query:
        # Try ML service for semantic search
        ml_response = await get_ml_answer(
            query=query,
            intent=f"{category}_recommendation",
            use_llm=False  # Fast template response
        )
        
        if ml_response and ml_response.get('success'):
            return {
                "category": category,
                "query": query,
                "recommendations": ml_response.get('context', []),
                "method": "ml_semantic_search"
            }
    
    # Fallback to your existing logic
    recommendations = await get_traditional_recommendations(category)
    return {
        "category": category,
        "recommendations": recommendations,
        "method": "traditional"
    }


# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

"""
Add to your .env file:

# ML Service Configuration
ML_SERVICE_ENABLED=true
ML_SERVICE_URL=http://localhost:8000
ML_SERVICE_TIMEOUT=30.0
ML_CACHE_TTL=300

# Production (T4 GPU)
# ML_SERVICE_URL=http://YOUR_T4_INSTANCE_IP:8000
"""


# ═══════════════════════════════════════════════════════════════════
# STARTUP LOGGING
# ═══════════════════════════════════════════════════════════════════

"""
Add to your startup function:

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting AI Istanbul Backend")
    
    # Check ML service
    ml_status = await get_ml_status()
    if ml_status['ml_service']['healthy']:
        logger.info("✅ ML service connected and healthy")
    else:
        logger.warning("⚠️ ML service not available - fallback mode active")
    
    # Your other startup code...
"""
