"""
AI Istanbul Backend - Pure LLM Architecture
============================================

Minimal backend with ONLY Pure LLM components:
- RunPod LLM (Llama 3.1 8B 4-bit)
- Pure LLM Handler
- Database context injection
- Redis caching
- Essential services only

NO rule-based systems, NO template systems, NO local ML models.

Author: Istanbul AI Team
Date: November 12, 2025
"""

# === Standard Library Imports ===
import sys
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# === FastAPI and Pydantic ===
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# === Load Environment Variables ===
# Load from parent directory's .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize FastAPI ===
app = FastAPI(
    title="Istanbul AI Guide API - Pure LLM",
    description="Pure LLM architecture with RunPod Llama 3.1 8B",
    version="3.0.0"
)

# === CORS Middleware ===
# Explicitly allow frontend on port 3001 (and other common development ports)
allowed_origins = [
    "http://localhost:3001",  # Frontend Vite server
    "http://localhost:5173",  # Default Vite port
    "http://localhost:3000",  # Alternative frontend port
    "http://127.0.0.1:3001",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Database Imports ===
try:
    from database import engine, SessionLocal, get_db
    from models import Base, Restaurant, Museum, Place, UserFeedback, ChatSession
    from sqlalchemy.orm import Session
    logger.info("✅ Database loaded")
except ImportError as e:
    logger.error(f"❌ Database import failed: {e}")
    raise

# === Redis Setup ===
REDIS_AVAILABLE = False
redis_client = None
try:
    import redis
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    redis_client = redis.from_url(redis_url, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info(f"✅ Redis connected: {redis_url}")
except Exception as e:
    logger.warning(f"⚠️ Redis unavailable: {e}")

# === RunPod LLM Client ===
RUNPOD_LLM_AVAILABLE = False
try:
    from services.runpod_llm_client import RunPodLLMClient, get_llm_client
    RUNPOD_LLM_AVAILABLE = True
    logger.info("✅ RunPod LLM Client loaded")
    logger.info(f"   Endpoint: {os.getenv('LLM_API_URL')}")
except ImportError as e:
    logger.error(f"❌ RunPod LLM Client failed: {e}")

# === Pure LLM Handler (New Modular API) ===
PURE_LLM_HANDLER_AVAILABLE = False
pure_llm_core = None
pure_llm_enabled = os.getenv("PURE_LLM_MODE", "false").lower() == "true"

try:
    from services.llm import create_pure_llm_core, PureLLMCore
    PURE_LLM_HANDLER_AVAILABLE = True
    logger.info("✅ Pure LLM Handler (Modular API) loaded")
except ImportError as e:
    logger.error(f"❌ Pure LLM Handler failed: {e}")

# === RAG Service ===
RAG_SERVICE_AVAILABLE = False
try:
    from services.rag_service import get_rag_instance
    RAG_SERVICE_AVAILABLE = True
    logger.info("✅ RAG Service loaded")
except ImportError as e:
    logger.error(f"❌ RAG Service failed: {e}")

# === Weather Service ===
WEATHER_SERVICE_AVAILABLE = False
try:
    from api_clients.enhanced_weather import EnhancedWeatherClient
    WEATHER_SERVICE_AVAILABLE = True
    logger.info("✅ Weather Service loaded")
except ImportError as e:
    logger.error(f"❌ Weather Service failed: {e}")

# === Map & Routing Services ===
MAP_SERVICE_AVAILABLE = False
ROUTING_SERVICE_AVAILABLE = False
try:
    from services.map_visualization_engine import MapVisualizationEngine
    from services.osrm_routing_service import OSRMRoutingService
    MAP_SERVICE_AVAILABLE = True
    ROUTING_SERVICE_AVAILABLE = True
    logger.info("✅ Map & Routing Services loaded")
except ImportError as e:
    logger.error(f"❌ Map & Routing Services failed: {e}")

# =============================
# STARTUP EVENT
# =============================

@app.on_event("startup")
async def startup_event():
    """Initialize Pure LLM Core (New Modular API) with RAG"""
    global pure_llm_core
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Pure LLM Backend with RAG (Modular Architecture)")
    logger.info("=" * 60)
    
    if PURE_LLM_HANDLER_AVAILABLE and RUNPOD_LLM_AVAILABLE and pure_llm_enabled:
        try:
            # Get database session
            db = next(get_db())
            
            # Initialize RAG service (if available)
            rag_service = None
            if RAG_SERVICE_AVAILABLE:
                try:
                    logger.info("📚 Initializing RAG Service...")
                    rag_service = get_rag_instance()
                    logger.info("✅ RAG Service initialized")
                except Exception as rag_error:
                    logger.warning(f"⚠️ RAG Service initialization failed: {rag_error}")
            
            # Initialize Weather service (if available)
            weather_service = None
            if WEATHER_SERVICE_AVAILABLE:
                try:
                    logger.info("🌤️ Initializing Weather Service...")
                    weather_service = EnhancedWeatherClient()
                    logger.info("✅ Weather Service initialized")
                except Exception as weather_error:
                    logger.warning(f"⚠️ Weather Service initialization failed: {weather_error}")
            
            # Initialize Map & Routing services (if available)
            map_service = None
            routing_service = None
            if MAP_SERVICE_AVAILABLE and ROUTING_SERVICE_AVAILABLE:
                try:
                    logger.info("🗺️ Initializing Map & Routing Services...")
                    routing_service = OSRMRoutingService(profile='foot', use_cache=REDIS_AVAILABLE)
                    map_service = MapVisualizationEngine(use_osrm=True)
                    logger.info("✅ Map & Routing Services initialized")
                except Exception as map_error:
                    logger.warning(f"⚠️ Map & Routing Services initialization failed: {map_error}")
            
            # Initialize Pure LLM Core using new modular API
            pure_llm_core = create_pure_llm_core(
                db=db,
                rag_service=rag_service,
                redis_client=redis_client if REDIS_AVAILABLE else None,
                weather_service=weather_service,
                map_service=map_service,
                routing_service=routing_service,
                enable_cache=REDIS_AVAILABLE,
                enable_analytics=True,
                enable_experimentation=False,
                enable_conversation=True,
                enable_query_enhancement=True
            )
            
            logger.info("✅ Pure LLM Core initialized (Modular Architecture)")
            logger.info(f"   LLM Model: Llama 3.1 8B (4-bit)")
            logger.info(f"   Endpoint: {os.getenv('LLM_API_URL')}")
            logger.info(f"   Redis Cache: {'Enabled' if REDIS_AVAILABLE else 'Disabled'}")
            logger.info(f"   RAG Service: {'Enabled' if rag_service else 'Disabled'}")
            logger.info(f"   Weather Service: {'Enabled' if weather_service else 'Disabled'}")
            logger.info(f"   Map & Routing: {'Enabled' if map_service and routing_service else 'Disabled'}")
            logger.info(f"   Modular System: ✅ Core + 9 specialized modules")
        except Exception as e:
            logger.error(f"❌ Pure LLM Core initialization failed: {e}")
            pure_llm_core = None
    else:
        logger.warning("⚠️ Pure LLM mode disabled or components unavailable")
    
    logger.info("=" * 60)
    logger.info("✅ Startup complete")
    logger.info("=" * 60)


# =============================
# HEALTH CHECK
# =============================

@app.get("/health", tags=["System Health"])
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "architecture": "Pure LLM (Modular)",
        "services": {
            "api": "healthy",
            "runpod_llm": "healthy" if RUNPOD_LLM_AVAILABLE else "unavailable",
            "pure_llm_core": "healthy" if pure_llm_core else "unavailable",
            "redis": "healthy" if REDIS_AVAILABLE else "unavailable",
            "database": "healthy"
        }
    }


# =============================
# PYDANTIC MODELS
# =============================

class PureLLMChatRequest(BaseModel):
    """Request model for Pure LLM chat"""
    message: str = Field(..., min_length=1, max_length=1000, description="User message/query")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session ID")
    user_location: Optional[Dict[str, float]] = Field(None, description="GPS location {lat, lon}")
    language: str = Field(default="en", description="Response language (en/tr)")
    intent: Optional[str] = Field(None, description="Pre-detected intent")


class PureLLMChatResponse(BaseModel):
    """Response model for Pure LLM chat"""
    response: str = Field(..., description="LLM-generated response")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: float = Field(..., description="Confidence (0-1)")
    method: str = Field(..., description="Generation method")
    context_used: List[str] = Field(default=[], description="Context sources")
    response_time: float = Field(..., description="Response time (seconds)")
    cached: bool = Field(default=False, description="Was response cached")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


# =============================
# PURE LLM CHAT ENDPOINT
# =============================

@app.post("/api/chat", response_model=PureLLMChatResponse, tags=["Pure LLM Chat"])
async def pure_llm_chat(
    request: PureLLMChatRequest,
    db: Session = Depends(get_db)
):
    """
    🎯 Pure LLM Chat Endpoint
    
    Routes ALL queries through RunPod LLM with intelligent context injection.
    
    Architecture:
    - RunPod: Llama 3.1 8B (4-bit quantized)
    - Database: PostgreSQL (context from restaurants, museums, places)
    - Cache: Redis (performance optimization)
    - No rule-based fallback - pure LLM only
    """
    start_time = time.time()
    
    # Check if Pure LLM Core is available
    if not pure_llm_core:
        logger.error("❌ Pure LLM Core not initialized")
        return PureLLMChatResponse(
            response="Pure LLM mode is not currently enabled. Please check system configuration.",
            intent=request.intent or "error",
            confidence=0.0,
            method="error",
            context_used=[],
            response_time=time.time() - start_time,
            cached=False,
            suggestions=["Contact support", "Check system status"],
            metadata={"error": "Pure LLM Core not initialized"}
        )
    
    try:
        logger.info(f"🎯 Pure LLM Query: '{request.message[:100]}...'")
        
        # Process query through Pure LLM Core (new modular API)
        result = await pure_llm_core.process_query(
            query=request.message,
            user_id=request.user_id or "anonymous",
            language=request.language or "en",
            max_tokens=250
        )
        
        # Build response
        response_time = time.time() - start_time
        
        return PureLLMChatResponse(
            response=result['response'],
            intent=result.get('intent', request.intent),
            confidence=result.get('confidence', 0.8),
            method=result.get('method', 'pure_llm'),
            context_used=result.get('context_used', []),
            response_time=response_time,
            cached=result.get('cached', False),
            suggestions=result.get('suggestions', []),
            metadata={
                'llm_model': 'Llama 3.1 8B (4-bit)',
                'context_count': len(result.get('context_used', [])),
                'cache_key': result.get('cache_key', None)
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Pure LLM Chat error: {e}", exc_info=True)
        response_time = time.time() - start_time
        
        # Emergency fallback
        return PureLLMChatResponse(
            response="I apologize, but I'm having trouble processing your request. Please try again.",
            intent="error",
            confidence=0.0,
            method="error_fallback",
            context_used=[],
            response_time=response_time,
            cached=False,
            suggestions=[
                "Try rephrasing your question",
                "Ask about specific attractions",
                "Check system status"
            ],
            metadata={"error": str(e)}
        )


@app.get("/api/chat/status", tags=["Pure LLM Chat"])
async def pure_llm_status():
    """Get Pure LLM Core system status"""
    if not pure_llm_core:
        return {
            "enabled": False,
            "available": PURE_LLM_HANDLER_AVAILABLE,
            "reason": "Pure LLM Core not initialized",
            "config": {
                "pure_llm_mode": pure_llm_enabled,
                "runpod_available": RUNPOD_LLM_AVAILABLE,
                "redis_available": REDIS_AVAILABLE
            }
        }
    
    try:
        # Get analytics from modular system
        stats = {}
        if pure_llm_core.analytics:
            stats = pure_llm_core.analytics.get_analytics()
        
        return {
            "enabled": True,
            "available": True,
            "architecture": "Modular (10 specialized modules)",
            "llm_model": "Llama 3.1 8B (4-bit)",
            "endpoint": "/api/chat",
            "statistics": stats,
            "config": {
                "redis_cache": REDIS_AVAILABLE,
                "database_context": True
            }
        }
    except Exception as e:
        logger.error(f"Error getting Pure LLM status: {e}")
        return {
            "enabled": True,
            "available": False,
            "error": str(e)
        }


# =============================
# RAG SERVICE TEST ENDPOINTS
# =============================

class RAGTestRequest(BaseModel):
    """Request model for RAG testing"""
    query: str = Field(..., description="Query for RAG retrieval")
    top_k: Optional[int] = Field(3, description="Number of results")
    category_filter: Optional[str] = Field(None, description="Category filter")


class RAGTestResponse(BaseModel):
    """Response model for RAG testing"""
    success: bool
    query: str
    results: List[Dict[str, Any]] = []
    enhanced_prompt: Optional[str] = None
    error: Optional[str] = None


@app.get("/api/rag/health", tags=["RAG Service"])
async def rag_health_check():
    """Check RAG service health and statistics"""
    if not RAG_SERVICE_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "RAG Service not loaded"
        }
    
    try:
        rag = get_rag_instance()
        stats = rag.get_category_stats()
        
        return {
            "status": "healthy",
            "statistics": stats,
            "total_documents": sum(stats.values()),
            "categories": list(stats.keys())
        }
    except Exception as e:
        logger.error(f"RAG health check error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/api/rag/retrieve", response_model=RAGTestResponse, tags=["RAG Service"])
async def rag_retrieve_test(request: RAGTestRequest):
    """Test RAG retrieval system"""
    if not RAG_SERVICE_AVAILABLE:
        return RAGTestResponse(
            success=False,
            query=request.query,
            error="RAG Service not available"
        )
    
    try:
        rag = get_rag_instance()
        
        # Retrieve relevant documents
        results = rag.retrieve(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter
        )
        
        # Get enhanced prompt
        enhanced_prompt = rag.enhance_prompt(request.query, top_k=request.top_k)
        
        return RAGTestResponse(
            success=True,
            query=request.query,
            results=results,
            enhanced_prompt=enhanced_prompt
        )
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return RAGTestResponse(
            success=False,
            query=request.query,
            error=str(e)
        )


# =============================
# RUNPOD LLM TEST ENDPOINTS
# =============================

class LLMTestRequest(BaseModel):
    """Request model for LLM testing"""
    prompt: str = Field(..., description="Prompt for LLM generation")
    max_tokens: Optional[int] = Field(250, description="Maximum tokens")


class LLMTestResponse(BaseModel):
    """Response model for LLM testing"""
    success: bool
    generated_text: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = None


@app.get("/api/llm/health", tags=["RunPod LLM"])
async def llm_health_check():
    """Check RunPod LLM service health"""
    if not RUNPOD_LLM_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "RunPod LLM client not loaded",
            "endpoint": os.getenv("LLM_API_URL", "Not configured")
        }
    
    try:
        llm_client = get_llm_client()
        health = await llm_client.health_check()
        return health
    except Exception as e:
        logger.error(f"LLM health check error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/api/llm/generate", response_model=LLMTestResponse, tags=["RunPod LLM"])
async def llm_generate_test(request: LLMTestRequest):
    """Test RunPod LLM generation"""
    if not RUNPOD_LLM_AVAILABLE:
        return LLMTestResponse(
            success=False,
            error="RunPod LLM client not available"
        )
    
    try:
        llm_client = get_llm_client()
        result = await llm_client.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )
        
        if result and 'generated_text' in result:
            return LLMTestResponse(
                success=True,
                generated_text=result['generated_text'],
                model="Llama 3.1 8B (4-bit)"
            )
        else:
            return LLMTestResponse(
                success=False,
                error="No response from LLM"
            )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return LLMTestResponse(
            success=False,
            error=str(e)
        )


# =============================
# APPLICATION STARTUP
# =============================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8002))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("=" * 70)
    print("🚀 AI Istanbul Backend - Pure LLM Architecture")
    print("=" * 70)
    print(f"📍 Host: {host}")
    print(f"📍 Port: {port}")
    print(f"🌐 Health: http://localhost:{port}/health")
    print(f"🎯 Chat: http://localhost:{port}/api/chat")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"🔧 LLM Model: Llama 3.1 8B (4-bit)")
    print(f"🚀 Architecture: Pure LLM (No Rule-Based Systems)")
    print("=" * 70)
    
    uvicorn.run(
        "main_pure_llm:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
