"""
PASTE THIS INTO YOUR backend/main.py or app.py

Add these sections to integrate ML service.
Modify as needed for your existing code structure.
"""

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: ADD THESE IMPORTS (at top of file)
# ═══════════════════════════════════════════════════════════════════

import os
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Import ML service client
try:
    from backend.ml_service_client import (
        get_ml_answer, 
        get_ml_status, 
        check_ml_health
    )
    ML_CLIENT_AVAILABLE = True
    print("✅ ML Service Client loaded")
except ImportError as e:
    ML_CLIENT_AVAILABLE = False
    print(f"⚠️ ML Service Client not available: {e}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: ADD THESE MODELS
# ═══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=1000)
    user_location: Optional[Dict[str, float]] = None
    use_llm: Optional[bool] = Field(None, description="Override: Use LLM (None=use default from config)")
    language: str = Field(default="en")
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    intent: str
    confidence: float
    method: str
    context: List[Dict] = []
    suggestions: List[str] = []
    response_time: float
    ml_service_used: bool


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: ADD THESE HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

async def generate_fallback_response(
    message: str,
    intent: str = "general",
    user_location: Optional[Dict] = None
) -> Dict:
    """
    Fallback response when ML service unavailable
    Replace with your existing logic
    """
    responses = {
        "restaurant_recommendation": {
            "answer": "Istanbul has amazing restaurants! Popular areas include Beyoğlu, Kadıköy, and Beşiktaş. What type of cuisine interests you?",
            "context": []
        },
        "attraction_query": {
            "answer": "Istanbul is full of incredible attractions! Must-sees include Hagia Sophia, Blue Mosque, Topkapı Palace, and the Grand Bazaar. Which area would you like to explore?",
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
    
    return responses.get(intent, responses["general"])


def generate_suggestions(intent: str) -> List[str]:
    """Generate follow-up suggestions"""
    suggestions = {
        "restaurant_recommendation": [
            "Show me vegetarian options",
            "What about seafood restaurants?",
            "Budget-friendly places near me"
        ],
        "attraction_query": [
            "Tell me about museums",
            "Historical sites in Sultanahmet",
            "Best views in Istanbul"
        ],
        "general": [
            "Best restaurants in Istanbul",
            "Top attractions to visit",
            "How to get around the city"
        ]
    }
    return suggestions.get(intent, suggestions["general"])


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: ADD THIS ENDPOINT
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with ML integration (LLM by default)
    
    Tries ML service first, falls back to rule-based if unavailable
    """
    start_time = time.time()
    
    # ML Configuration
    ML_USE_LLM_DEFAULT = os.getenv("ML_USE_LLM_DEFAULT", "true").lower() == "true"
    
    try:
        # Detect intent (use your existing classifier if you have one)
        intent = "general"  # TODO: Replace with your intent detection
        
        # Determine if should use LLM (default to config, allow request override)
        use_llm = request.use_llm if request.use_llm is not None else ML_USE_LLM_DEFAULT
        
        logger.info(f"💬 Query: '{request.message}' (intent: {intent}, llm: {use_llm})")
        
        # Try ML service if available
        if ML_CLIENT_AVAILABLE:
            ml_response = await get_ml_answer(
                query=request.message,
                intent=intent,
                user_location=request.user_location,
                use_llm=use_llm,  # Use LLM by default
                language=request.language
            )
            
            if ml_response and ml_response.get('success'):
                # ML service succeeded ✅
                logger.info(f"✅ ML response: {ml_response.get('generation_method')}")
                
                return ChatResponse(
                    response=ml_response['answer'],
                    intent=ml_response.get('intent', intent),
                    confidence=ml_response.get('confidence', 0.8),
                    method=f"ml_{ml_response.get('generation_method', 'unknown')}",
                    context=ml_response.get('context', []),
                    suggestions=ml_response.get('suggestions', []),
                    response_time=time.time() - start_time,
                    ml_service_used=True
                )
        
        # Fallback to rule-based
        logger.info("⚠️ Using fallback response")
        
        fallback = await generate_fallback_response(
            request.message,
            intent,
            request.user_location
        )
        
        return ChatResponse(
            response=fallback['answer'],
            intent=intent,
            confidence=0.6,
            method="fallback",
            context=fallback.get('context', []),
            suggestions=generate_suggestions(intent),
            response_time=time.time() - start_time,
            ml_service_used=False
        )
    
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: ADD THESE STATUS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/v1/ml/status", tags=["System"])
async def ml_status_endpoint():
    """Get ML service status"""
    if not ML_CLIENT_AVAILABLE:
        return {
            "available": False,
            "reason": "ML client not loaded"
        }
    
    status = await get_ml_status()
    return status


@app.get("/api/v1/ml/health", tags=["System"])
async def ml_health_endpoint():
    """Quick ML service health check"""
    if not ML_CLIENT_AVAILABLE:
        return {
            "healthy": False,
            "reason": "ML client not loaded"
        }
    
    health = await check_ml_health()
    return health


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: UPDATE YOUR EXISTING HEALTH ENDPOINT (if you have one)
# ═══════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    """Overall system health"""
    
    # Check ML service
    ml_healthy = False
    if ML_CLIENT_AVAILABLE:
        ml_health = await check_ml_health()
        ml_healthy = ml_health.get('healthy', False)
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "ml_service": "healthy" if ml_healthy else "degraded"
        }
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: ADD TO STARTUP EVENT (if you have one)
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Starting AI Istanbul Backend")
    
    # Check ML service
    if ML_CLIENT_AVAILABLE:
        ml_status = await get_ml_status()
        if ml_status['ml_service']['healthy']:
            logger.info("✅ ML service connected and healthy")
            logger.info(f"   URL: {ml_status['ml_service']['url']}")
        else:
            logger.warning("⚠️ ML service unavailable - fallback mode active")
    else:
        logger.warning("⚠️ ML client not available - fallback mode active")
    
    # Your other startup code...


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: ADD TO .env FILE
# ═══════════════════════════════════════════════════════════════════

"""
Add these to your .env file:

# ML Service Configuration (LLM BY DEFAULT)
ML_SERVICE_ENABLED=true
ML_SERVICE_URL=http://localhost:8000
ML_SERVICE_TIMEOUT=60.0  # Increased for LLM generation
ML_CACHE_TTL=300
ML_USE_LLM_DEFAULT=true  # Use LLM by default for quality ⭐

# For production with T4 GPU (faster LLM):
# ML_SERVICE_URL=http://YOUR_T4_INSTANCE_IP:8000
# ML_SERVICE_TIMEOUT=30.0  # Can be lower on T4
"""


# ═══════════════════════════════════════════════════════════════════
# THAT'S IT! Now test:
# ═══════════════════════════════════════════════════════════════════

"""
Terminal 1:
    source venv_ml/bin/activate
    ./start_ml_service.sh

Terminal 2:
    cd backend
    python main.py

Terminal 3:
    curl -X POST http://localhost:YOUR_PORT/api/v1/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "Best restaurants in Beyoğlu?"}'
"""
