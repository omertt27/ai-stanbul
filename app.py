#!/usr/bin/env python3
"""
FastAPI Web Application for Advanced Istanbul AI
Production-ready web service with authentication, monitoring, and scalability
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Import our advanced Istanbul AI system
from advanced_istanbul_ai import AdvancedIstanbulAI, UserProfile, ConversationTone, UserType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('istanbul_ai_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('istanbul_ai_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('istanbul_ai_active_connections', 'Active WebSocket connections')
USER_SATISFACTION = Histogram('istanbul_ai_user_satisfaction', 'User satisfaction ratings')
RESPONSE_GENERATION_TIME = Histogram('istanbul_ai_response_time_seconds', 'AI response generation time')

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global AI instance
ai_system: Optional[AdvancedIstanbulAI] = None
redis_client: Optional[redis.Redis] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.user_sessions[user_id] = session_id
        ACTIVE_CONNECTIONS.inc()
        logger.info(f"WebSocket connected: user={user_id}, session={session_id}")
    
    def disconnect(self, session_id: str, user_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        ACTIVE_CONNECTIONS.dec()
        logger.info(f"WebSocket disconnected: user={user_id}, session={session_id}")
    
    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(message)

manager = ConnectionManager()

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    global ai_system, redis_client
    
    # Startup
    logger.info("🚀 Starting Advanced Istanbul AI Web Service...")
    
    try:
        # Initialize AI system
        ai_system = AdvancedIstanbulAI()
        logger.info("✅ AI System initialized")
        
        # Initialize Redis
        redis_client = redis.from_url("redis://localhost:6379/0", decode_responses=True)
        await redis_client.ping()
        logger.info("✅ Redis connected")
        
        logger.info("🎯 Service startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Istanbul AI Web Service...")
    if redis_client:
        await redis_client.close()
    logger.info("✅ Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Advanced Istanbul AI API",
    description="Production-ready conversational AI for Istanbul visitors and locals",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://istanbul-ai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Session identifier for multi-turn conversations")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI generated response")
    session_id: str = Field(..., description="Session identifier")
    processing_time: float = Field(..., description="Response generation time in seconds")
    confidence: Optional[float] = Field(None, description="Response confidence score")
    intents: Optional[List[Dict[str, Any]]] = Field(None, description="Detected user intents")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted entities")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")

class UserProfileResponse(BaseModel):
    user_id: str
    user_type: str
    preferred_tone: str
    favorite_neighborhoods: List[str]
    interests: List[str]
    interaction_count: int
    satisfaction_score: Optional[float]

class FeedbackRequest(BaseModel):
    session_id: str
    rating: float = Field(..., ge=1.0, le=5.0, description="User satisfaction rating (1-5)")
    comment: Optional[str] = Field(None, max_length=500, description="Optional feedback comment")

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "2.0.0"
    services: Dict[str, str] = Field(default_factory=dict)

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract user from JWT token (simplified for demo)"""
    if not credentials:
        return "anonymous_user"
    
    # In production: validate JWT token
    return "authenticated_user"

async def get_redis() -> redis.Redis:
    """Get Redis client"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    return redis_client

# Utility functions
async def cache_get(key: str, redis_conn: redis.Redis) -> Optional[str]:
    """Get value from cache"""
    try:
        return await redis_conn.get(key)
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        return None

async def cache_set(key: str, value: str, ttl: int, redis_conn: redis.Redis):
    """Set value in cache"""
    try:
        await redis_conn.setex(key, ttl, value)
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")

def create_cache_key(user_id: str, message: str) -> str:
    """Create cache key for similar messages"""
    import hashlib
    message_hash = hashlib.md5(message.lower().encode()).hexdigest()[:8]
    return f"response:{user_id}:{message_hash}"

# API Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check AI system
    if ai_system:
        services["ai_system"] = "healthy"
    else:
        services["ai_system"] = "unhealthy"
    
    # Check Redis
    try:
        if redis_client:
            await redis_client.ping()
            services["redis"] = "healthy"
        else:
            services["redis"] = "unavailable"
    except Exception:
        services["redis"] = "unhealthy"
    
    overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    return HealthResponse(status=overall_status, services=services)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/chat", response_model=ChatResponse)
@limiter.limit("60/minute")
async def chat(
    request: Request,
    message: ChatMessage,
    background_tasks: BackgroundTasks,
    redis_conn: redis.Redis = Depends(get_redis),
    current_user: str = Depends(get_current_user)
):
    """Main chat endpoint"""
    start_time = time.time()
    
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/chat", status="processing").inc()
        
        # Check cache for similar responses
        cache_key = create_cache_key(message.user_id, message.message)
        cached_response = await cache_get(cache_key, redis_conn)
        
        if cached_response:
            logger.info(f"Cache hit for user {message.user_id}")
            response_data = json.loads(cached_response)
            REQUEST_COUNT.labels(method="POST", endpoint="/chat", status="200").inc()
            return ChatResponse(**response_data)
        
        # Generate AI response
        if not ai_system:
            raise HTTPException(status_code=503, detail="AI system not available")
        
        ai_response = await ai_system.process_message(
            message.message,
            message.user_id,
            message.session_id
        )
        
        processing_time = time.time() - start_time
        RESPONSE_GENERATION_TIME.observe(processing_time)
        
        # Create response
        response_data = {
            "response": ai_response,
            "session_id": message.session_id or f"{message.user_id}_{int(time.time())}",
            "processing_time": processing_time,
            "confidence": 0.85,  # Mock confidence score
            "suggestions": [
                "Tell me more about Turkish cuisine",
                "How do I get around Istanbul?",
                "What are the best neighborhoods to visit?"
            ]
        }
        
        # Cache the response
        background_tasks.add_task(
            cache_set, 
            cache_key, 
            json.dumps(response_data), 
            300,  # 5 minutes TTL
            redis_conn
        )
        
        REQUEST_COUNT.labels(method="POST", endpoint="/chat", status="200").inc()
        REQUEST_DURATION.observe(processing_time)
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        processing_time = time.time() - start_time
        REQUEST_COUNT.labels(method="POST", endpoint="/chat", status="500").inc()
        REQUEST_DURATION.observe(processing_time)
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/user/{user_id}/profile", response_model=UserProfileResponse)
@limiter.limit("30/minute")
async def get_user_profile(
    user_id: str,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Get user profile and analytics"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="AI system not available")
        
        analytics = ai_system.get_user_analytics(user_id)
        
        return UserProfileResponse(
            user_id=user_id,
            user_type=analytics.get("user_type", "first_time_visitor"),
            preferred_tone="friendly",
            favorite_neighborhoods=analytics.get("favorite_neighborhoods", []),
            interests=analytics.get("interests", []),
            interaction_count=analytics.get("interaction_count", 0),
            satisfaction_score=analytics.get("average_satisfaction")
        )
        
    except Exception as e:
        logger.error(f"Profile endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/feedback")
@limiter.limit("10/minute")
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Submit user feedback"""
    try:
        USER_SATISFACTION.observe(feedback.rating)
        logger.info(f"Feedback received: session={feedback.session_id}, rating={feedback.rating}")
        
        # In production: store feedback in database
        return {"status": "success", "message": "Feedback received"}
        
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, user_id, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if not ai_system:
                await websocket.send_text(json.dumps({
                    "error": "AI system not available",
                    "timestamp": datetime.now().isoformat()
                }))
                continue
            
            # Process message
            start_time = time.time()
            ai_response = await ai_system.process_message(
                message_data["message"],
                user_id,
                session_id
            )
            processing_time = time.time() - start_time
            
            # Send response
            response_data = {
                "response": ai_response,
                "session_id": session_id,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(response_data))
            RESPONSE_GENERATION_TIME.observe(processing_time)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id, user_id)

@app.get("/api/v1/status")
async def get_system_status():
    """Get system status and statistics"""
    return {
        "status": "operational",
        "version": "2.0.0",
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Advanced Intent Classification",
            "Context-Aware Dialogue",
            "Real-time Data Integration",
            "Personalized Responses",
            "Cultural Intelligence",
            "Multi-language Support"
        ]
    }

@app.get("/api/v1/neighborhoods")
async def get_neighborhoods():
    """Get available Istanbul neighborhoods"""
    if not ai_system:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    neighborhoods = ai_system.entity_recognizer.knowledge_graph.neighborhoods
    return {
        "neighborhoods": [
            {
                "name": name,
                "type": info.get("type"),
                "atmosphere": info.get("atmosphere"),
                "best_for": info.get("best_for", []),
                "cultural_significance": info.get("cultural_significance", 0.5)
            }
            for name, info in neighborhoods.items()
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Development server
if __name__ == "__main__":
    print("🚀 Starting Advanced Istanbul AI Web Service...")
    print("📚 API Documentation: http://localhost:8080/docs")
    print("📊 Metrics: http://localhost:8080/metrics")
    print("💬 WebSocket: ws://localhost:8080/ws/{user_id}/{session_id}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
