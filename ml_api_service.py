"""
FastAPI ML Service for Istanbul AI Guide
RESTful API for ML-powered answering
"""

import logging
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ml_systems.ml_answering_service import create_ml_service, MLAnsweringService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global ML service instance
ml_service: Optional[MLAnsweringService] = None


# Pydantic models for API
class UserLocation(BaseModel):
    """GPS coordinates"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lng: float = Field(..., ge=-180, le=180, description="Longitude")


class ChatMessage(BaseModel):
    """Message in conversation history"""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request payload"""
    query: str = Field(..., min_length=1, description="User's question")
    user_location: Optional[UserLocation] = Field(None, description="User's GPS location")
    conversation_history: Optional[list[ChatMessage]] = Field(None, description="Previous messages")


class ChatResponse(BaseModel):
    """Chat response payload"""
    answer: str = Field(..., description="Generated answer")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., ge=0, le=1, description="Intent confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    sources: list[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Create FastAPI app
app = FastAPI(
    title="Istanbul AI - ML Service",
    description="Machine Learning powered answering service for Istanbul tourism",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize ML service on startup"""
    global ml_service
    
    logger.info("🚀 Starting Istanbul AI ML Service...")
    
    # Initialize ML service with LLM enabled for high-quality responses
    # Set to False for faster startup if LLM is not needed
    ml_service = await create_ml_service(enable_llm=True)
    
    logger.info("✅ ML Service ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down ML Service")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Istanbul AI ML Service",
        "version": "1.0.0",
        "ml_loaded": ml_service is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if not ml_service:
        raise HTTPException(status_code=503, detail="ML service not initialized")
    
    return {
        "status": "healthy",
        "components": {
            "intent_classifier": ml_service.intent_classifier is not None,
            "semantic_search": ml_service.semantic_search is not None,
            "llm_generator": ml_service.llm_generator is not None
        }
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint - answers user queries using ML pipeline
    
    This endpoint:
    1. Classifies the intent of the user's query
    2. Retrieves relevant context (semantic search or structured data)
    3. Generates a natural language response
    
    Example request:
    ```json
    {
        "query": "Best restaurants in Beyoğlu with sea view",
        "user_location": {"lat": 41.0345, "lng": 28.9784}
    }
    ```
    """
    if not ml_service:
        raise HTTPException(status_code=503, detail="ML service not initialized")
    
    start_time = time.time()
    
    try:
        # Convert Pydantic models to dicts
        user_location = request.user_location.dict() if request.user_location else None
        conversation_history = [msg.dict() for msg in request.conversation_history] if request.conversation_history else None
        
        # Process query through ML pipeline
        result = await ml_service.answer_query(
            query=request.query,
            user_location=user_location,
            conversation_history=conversation_history
        )
        
        # Add API processing time
        result["processing_time"] = time.time() - start_time
        
        logger.info(f"✅ Query processed: '{request.query[:50]}...' -> {result['intent']} ({result['processing_time']:.2f}s)")
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search")
async def semantic_search_endpoint(
    query: str,
    collection: str = "restaurants",
    top_k: int = 5
):
    """
    Direct semantic search endpoint
    
    Returns relevant items from the specified collection without LLM generation.
    Useful for quick lookups and autocomplete features.
    
    Args:
        query: Search query
        collection: Collection to search ('restaurants', 'attractions', 'tips')
        top_k: Number of results to return
    """
    if not ml_service or not ml_service.semantic_search:
        raise HTTPException(status_code=503, detail="Semantic search not available")
    
    try:
        results = ml_service.semantic_search.search(
            query=query,
            top_k=top_k,
            collection=collection
        )
        
        return {
            "query": query,
            "collection": collection,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/intents")
async def list_intents():
    """
    List all supported intents
    
    Returns a list of all intent categories the system can handle.
    """
    from ml_systems.ml_answering_service import Intent
    
    return {
        "intents": [
            {
                "name": intent.value,
                "description": _get_intent_description(intent)
            }
            for intent in Intent
        ]
    }


def _get_intent_description(intent) -> str:
    """Get human-readable description for an intent"""
    descriptions = {
        "restaurant_recommendation": "Find and recommend restaurants based on cuisine, location, or preferences",
        "attraction_query": "Provide information about tourist attractions, museums, monuments",
        "neighborhood_info": "Describe neighborhoods, their character, and what to do there",
        "transportation_help": "Help with public transport, routes, and navigation",
        "daily_talk": "General conversation and casual questions about Istanbul",
        "local_tips": "Share insider tips and local knowledge",
        "weather_info": "Provide weather information and forecasts",
        "events_query": "Information about events, concerts, festivals",
        "route_planning": "Plan routes and itineraries"
    }
    return descriptions.get(intent.value, "No description available")


# Development server
if __name__ == "__main__":
    logger.info("🚀 Starting ML Service in development mode...")
    logger.info("📍 API will be available at: http://localhost:8001")
    logger.info("📖 API docs at: http://localhost:8001/docs")
    
    uvicorn.run(
        "ml_api_service:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
