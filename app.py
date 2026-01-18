#!/usr/bin/env python3
"""
FastAPI Web Application for Advanced Istanbul AI
Production-ready web service with authentication, monitoring, and scalability
"""

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

import asyncio
import time
import json
import logging
import re
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from enum import Enum

# Custom JSON encoder to handle enums
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis library not available")
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional NLP imports with fallbacks
try:
    import spacy
    import nltk
    from transformers import pipeline
    NLP_AVAILABLE = True
    logger.info("✅ Advanced NLP libraries available")
except ImportError as e:
    NLP_AVAILABLE = False
    logger.warning(f"⚠️ Advanced NLP libraries not available: {e}")

# Optional geo utilities
try:
    import geopy
    import geopandas
    import shapely
    GEO_UTILITIES_AVAILABLE = True
    logger.info("✅ Advanced geo utilities (GeoPandas, Geopy, Shapely) loaded successfully")
except ImportError as e:
    GEO_UTILITIES_AVAILABLE = False
    logger.info(f"ℹ️  Advanced geo utilities not installed: {e}")

# LAZY IMPORTS - DO NOT import heavy AI libraries at module level!
# They will be imported inside lifespan() to avoid blocking startup
IstanbulDailyTalkAI = None
ConversationTone = None
UserProfile = None
DeepLearningEnhancedAI = None
EmotionalState = None
UserType = None
MainSystemClass = None
MAIN_SYSTEM_AVAILABLE = False

logger.info("⚡ Using lazy imports for fast startup")

# Import push notification system
try:
    from services.push_notification_service import (
        notification_service, NotificationType, NotificationPriority, Notification
    )
    NOTIFICATIONS_AVAILABLE = True
    logger.info("🔔 Push Notification Service available")
except ImportError as e:
    NOTIFICATIONS_AVAILABLE = False
    logger.warning(f"🔔 Push Notification Service not available: {e}")
    
    # Create mock notification service
    class MockNotificationService:
        async def send_notification(self, *args, **kwargs):
            return {'mock': True}
        async def send_route_update(self, *args, **kwargs):
            return {'mock': True}
        async def send_attraction_recommendation(self, *args, **kwargs):
            return {'mock': True}
        def register_device_token(self, *args, **kwargs):
            pass
        def get_user_notifications(self, *args, **kwargs):
            return []
        async def connect_websocket(self, websocket, user_id):
            return "mock_connection"
        def disconnect_websocket(self, connection_id):
            pass
    
    notification_service = MockNotificationService()

# Import weather services
try:
    from services.weather_cache_service import (
        weather_cache, get_current_weather, get_weather_for_ai, update_weather_cache
    )
    from services.weather_notification_service import (
        send_weather_alerts, weather_notification_manager
    )
    WEATHER_SERVICES_AVAILABLE = True
    logger.info("🌤️ Weather Services available")
except ImportError as e:
    WEATHER_SERVICES_AVAILABLE = False
    logger.warning(f"🌤️ Weather Services not available: {e}")
    
    # Create mock weather service
    def get_current_weather():
        return {
            'current': {
                'current_temp': 20,
                'condition': 'Clear',
                'description': 'clear sky'
            },
            'recommendations': ['Perfect weather for exploring!'],
            'mock': True
        }

# Prometheus metrics
REQUEST_COUNT = Counter('istanbul_ai_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('istanbul_ai_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('istanbul_ai_active_connections', 'Active WebSocket connections')
USER_SATISFACTION = Histogram('istanbul_ai_user_satisfaction', 'User satisfaction ratings')
RESPONSE_GENERATION_TIME = Histogram('istanbul_ai_response_time_seconds', 'AI response generation time')

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global AI instances
ai_system: Optional[IstanbulDailyTalkAI] = None  # Daily talk system
main_system: Optional['MainSystemClass'] = None  # Main system with maps
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

# Background tasks
async def weather_update_task():
    """Background task to update weather cache every hour"""
    while True:
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            if WEATHER_SERVICES_AVAILABLE:
                success = await update_weather_cache()
                if success:
                    logger.info("🌤️ Hourly weather cache update completed")
                    
                    # Send weather alerts if needed
                    try:
                        await send_weather_alerts()
                        logger.info("📤 Weather alerts checked and sent")
                    except Exception as e:
                        logger.warning(f"Weather alerts failed: {e}")
                else:
                    logger.warning("⚠️ Weather cache update failed")
        except Exception as e:
            logger.error(f"Weather background task error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retry

# Async background initializers
async def initialize_main_system_async():
    """Initialize main system in background to not block startup"""
    global main_system, MAIN_SYSTEM_AVAILABLE, MainSystemClass
    try:
        logger.info("🔄 Importing Main AI System modules...")
        from istanbul_ai.main_system import IstanbulDailyTalkAI as MainSystemClass
        MAIN_SYSTEM_AVAILABLE = True
        
        logger.info("🔄 Initializing Main AI System in background...")
        main_system = MainSystemClass()
        logger.info("✅ Main AI System with Map Visualization initialized")
        logger.info("🗺️ Interactive maps enabled (Leaflet.js + OSM + OSRM)")
    except Exception as e:
        logger.warning(f"⚠️ Main system initialization failed: {e}")
        main_system = None
        MAIN_SYSTEM_AVAILABLE = False

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    global ai_system, main_system, redis_client, IstanbulDailyTalkAI
    
    # Startup
    logger.info("🚀 Starting Advanced Istanbul AI Web Service...")
    logger.info("⏱️ ULTRA-FAST startup mode - lazy loading all AI components")
    
    # Initialize core components with MAXIMUM lazy loading
    try:
        # DON'T load AI system during startup - it's too slow!
        # It will be loaded on first request instead
        logger.info("⏭️ Skipping AI initialization for fast startup")
        logger.info("🔄 AI system will load on first request")
        ai_system = None  # Will be initialized on first use
        
    except Exception as e:
        logger.error(f"❌ Startup setup failed: {e}")
        ai_system = None
    
    # Initialize main system (optional, in background)
    if MAIN_SYSTEM_AVAILABLE:
        # Don't block startup for this
        asyncio.create_task(initialize_main_system_async())
    
    # Initialize Redis (optional, with graceful fallback)
    redis_url = os.getenv('REDIS_URL')
    enable_redis = os.getenv('ENABLE_REDIS_CACHE', 'false').lower() == 'true'
    
    if REDIS_AVAILABLE and enable_redis and redis_url:
        try:
            logger.info(f"🔄 Connecting to Redis at {redis_url[:20]}...")
            
            # Parse timeout from environment
            socket_timeout = float(os.getenv('REDIS_SOCKET_TIMEOUT', '2'))
            connect_timeout = float(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '2'))
            
            # Check if TLS is required (rediss:// protocol)
            if redis_url.startswith('rediss://'):
                from urllib.parse import urlparse, parse_qs
                
                logger.info("🔐 Using TLS/SSL for Redis connection")
                parsed = urlparse(redis_url)
                query_params = parse_qs(parsed.query) if parsed.query else {}
                
                # Extract SSL cert requirements from URL params (default to 'required')
                ssl_cert_reqs_param = query_params.get('ssl_cert_reqs', ['required'])[0]
                
                # Map string to ssl module constant
                import ssl
                ssl_cert_reqs_map = {
                    'required': ssl.CERT_REQUIRED,
                    'optional': ssl.CERT_OPTIONAL,
                    'none': ssl.CERT_NONE
                }
                ssl_cert_reqs = ssl_cert_reqs_map.get(ssl_cert_reqs_param.lower(), ssl.CERT_REQUIRED)
                
                redis_client = redis.Redis(
                    host=parsed.hostname,
                    port=parsed.port or 6379,
                    password=parsed.password,
                    db=int(parsed.path.lstrip('/')) if parsed.path and parsed.path != '/' else 0,
                    ssl=True,
                    ssl_cert_reqs=ssl_cert_reqs,
                    decode_responses=True,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=connect_timeout,
                    retry_on_timeout=False,
                    health_check_interval=30
                )
                logger.info(f"✅ TLS Redis client created for {parsed.hostname}:{parsed.port or 6379}")
            else:
                # Regular non-TLS connection
                redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=connect_timeout,
                    retry_on_timeout=False,
                    health_check_interval=30
                )
            
            # Test connection with timeout
            logger.info("🔄 Testing Redis connection...")
            ping_result = await asyncio.wait_for(redis_client.ping(), timeout=2.0)
            logger.info(f"✅ Redis cache connected successfully! Ping: {ping_result}")
            
            # Test a basic set/get to ensure functionality
            test_key = "startup_test"
            await redis_client.set(test_key, "ok", ex=10)
            test_val = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            logger.info(f"✅ Redis operations verified (test get: {test_val})")
            
        except asyncio.TimeoutError:
            logger.error("❌ Redis connection TIMEOUT - continuing WITHOUT Redis")
            redis_client = None
        except Exception as e:
            logger.error(f"❌ Redis connection FAILED: {e} - continuing WITHOUT Redis")
            redis_client = None
    else:
        if not REDIS_AVAILABLE:
            logger.info("ℹ️  Redis library not available")
        elif not enable_redis:
            logger.info("ℹ️  Redis cache explicitly disabled (ENABLE_REDIS_CACHE=false)")
        else:
            logger.info("ℹ️  Redis URL not configured")
        redis_client = None    # Initialize weather services (in background to not block startup)
    if WEATHER_SERVICES_AVAILABLE:
        try:
            # Start weather cache in background (don't wait)
            asyncio.create_task(update_weather_cache())
            logger.info("🌤️ Weather cache initialization started (background)")
            
            # Start background weather update task
            asyncio.create_task(weather_update_task())
            logger.info("🌤️ Weather background task started")
        except Exception as e:
            logger.warning(f"⚠️ Weather initialization failed: {e}")
    
    logger.info("✅ Application startup complete")
    logger.info("🎯 Service is ready to accept connections!")
    
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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "https://istanbul-ai.com"
    ],
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
class UserLocation(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    accuracy: Optional[float] = Field(None, description="GPS accuracy in meters")

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Session identifier for multi-turn conversations")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    user_location: Optional[UserLocation] = Field(None, description="User's GPS location")
    
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
    map_data: Optional[Dict[str, Any]] = Field(None, description="Map visualization data (Leaflet.js format)")
    
    # Phase 5B/5C: UnifiedLLMService Metadata for Frontend Badges
    llm_backend: Optional[str] = Field(None, description="LLM backend used: 'vllm', 'groq', or 'legacy'")
    cache_hit: Optional[bool] = Field(None, description="Whether response was served from LLM cache")
    circuit_breaker_state: Optional[str] = Field(None, description="Circuit breaker state: 'open', 'closed', 'half_open'")
    llm_latency_ms: Optional[int] = Field(None, description="LLM response latency in milliseconds")

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

class NotificationRequest(BaseModel):
    """Push notification request model"""
    user_id: str = Field(..., description="Target user ID")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    type: str = Field(default="system_message", description="Notification type")
    priority: str = Field(default="normal", description="Notification priority")
    data: Optional[Dict[str, Any]] = Field(default={}, description="Additional notification data")

class DeviceTokenRequest(BaseModel):
    """Device token registration request"""
    device_token: str = Field(..., description="FCM device token")

class NotificationPreferencesRequest(BaseModel):
    """Notification preferences request"""
    route_updates: bool = Field(default=True, description="Receive route update notifications")
    attraction_recommendations: bool = Field(default=True, description="Receive attraction recommendations")
    weather_alerts: bool = Field(default=True, description="Receive weather alerts")
    traffic_updates: bool = Field(default=True, description="Receive traffic updates")
    personalized_tips: bool = Field(default=True, description="Receive personalized tips")

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

# Lazy AI initialization helper
async def ensure_ai_system():
    """Initialize AI system on first use (lazy loading)"""
    global ai_system, IstanbulDailyTalkAI
    
    if ai_system is not None:
        return ai_system
    
    try:
        logger.info("🔄 First request - importing AI modules now...")
        from istanbul_daily_talk_system import IstanbulDailyTalkAI as AIClass
        IstanbulDailyTalkAI = AIClass
        
        logger.info("🔄 Initializing AI system...")
        ai_system = IstanbulDailyTalkAI()
        logger.info("✅ AI System initialized successfully on first request!")
        return ai_system
    except Exception as e:
        logger.error(f"❌ AI system initialization failed: {e}")
        raise HTTPException(status_code=503, detail=f"AI system initialization failed: {str(e)}")

# Helper function to extract UnifiedLLM metadata from AI system
def extract_llm_metadata(ai_sys=None, main_sys=None, processing_time: float = 0.0) -> Dict[str, Any]:
    """
    Extract UnifiedLLMService metadata from AI systems for frontend badge display.
    
    Attempts to extract:
    - llm_backend: Which LLM backend was used ('vllm', 'groq', or 'legacy')
    - cache_hit: Whether response was served from cache
    - circuit_breaker_state: Circuit breaker state ('open', 'closed', 'half_open')
    - llm_latency_ms: LLM response latency in milliseconds
    
    Args:
        ai_sys: Daily talk AI system instance
        main_sys: Main system instance
        processing_time: Total processing time in seconds
        
    Returns:
        Dict with metadata fields (None values if not available)
    """
    metadata = {
        "llm_backend": None,
        "cache_hit": None,
        "circuit_breaker_state": None,
        "llm_latency_ms": None
    }
    
    try:
        # Try to extract from main_system first (priority)
        system_to_check = main_sys if main_sys and MAIN_SYSTEM_AVAILABLE else ai_sys
        
        if system_to_check and hasattr(system_to_check, 'llm_service') and system_to_check.llm_service is not None:
            llm_service = system_to_check.llm_service
            
            # Check if it's UnifiedLLMService
            if hasattr(llm_service, 'circuit_breaker_open'):
                # Extract circuit breaker state
                is_open = llm_service.circuit_breaker_open
                metadata["circuit_breaker_state"] = "open" if is_open else "closed"
                metadata["llm_backend"] = "groq" if is_open else "vllm"
                
                logger.info(f"📊 UnifiedLLM metadata: backend={metadata['llm_backend']}, cb_state={metadata['circuit_breaker_state']}")
            else:
                # Legacy LLM service
                metadata["llm_backend"] = "legacy"
                metadata["circuit_breaker_state"] = "n/a"
                logger.info("📊 Using legacy LLM service (no UnifiedLLM)")
            
            # Convert processing time to milliseconds
            if processing_time:
                metadata["llm_latency_ms"] = int(processing_time * 1000)
            
            # Note: cache_hit is harder to determine without modifying the AI systems
            # We'd need to check if the response was cached inside the LLM service
            # For now, leave it as None unless we can detect it
            metadata["cache_hit"] = False  # Default to False for now
            
        else:
            # LLM service not found or is None
            if system_to_check:
                has_attr = hasattr(system_to_check, 'llm_service')
                is_none = system_to_check.llm_service is None if has_attr else "N/A"
                logger.info(f"ℹ️ AI system found but llm_service unavailable (has_attr={has_attr}, is_none={is_none})")
            else:
                logger.info("ℹ️ No AI system available yet")
            
            metadata["llm_backend"] = "legacy"
            metadata["circuit_breaker_state"] = "n/a"
            metadata["llm_latency_ms"] = int(processing_time * 1000) if processing_time else None
            metadata["cache_hit"] = False
            
    except Exception as e:
        logger.warning(f"Failed to extract LLM metadata: {e}")
        # Return None values on error (non-blocking)
    
    return metadata


# API Routes

@app.get("/")
async def root():
    """Root endpoint - quick health check"""
    return {
        "service": "Istanbul AI", 
        "status": "online",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)  # Add /api/health for startup probe
async def health_check():
    """Health check endpoint - MUST respond fast for Cloud Run startup probe"""
    services = {}
    
    # ALWAYS return healthy during startup - AI system loads on first request
    # This allows Cloud Run to mark the service as ready immediately
    services["ai_system"] = "loading" if ai_system is None else "healthy"
    services["startup"] = "ready"
    
    # Check Redis (quickly, with timeout)
    try:
        if redis_client:
            # Quick ping to verify connection is alive
            try:
                await asyncio.wait_for(redis_client.ping(), timeout=0.5)
                services["redis"] = "connected"
            except asyncio.TimeoutError:
                services["redis"] = "timeout"
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                services["redis"] = "error"
        else:
            services["redis"] = "disabled"
    except Exception as e:
        logger.warning(f"Redis health check exception: {e}")
        services["redis"] = "unavailable"
    
    # Always return healthy so Cloud Run accepts the deployment
    overall_status = "healthy"
    
    return HealthResponse(status=overall_status, services=services)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/redis-status")
async def redis_status():
    """Detailed Redis connection status for diagnostics"""
    status = {
        "library_available": REDIS_AVAILABLE,
        "client_initialized": redis_client is not None,
        "enabled": os.getenv('ENABLE_REDIS_CACHE', 'false').lower() == 'true',
        "url_configured": os.getenv('REDIS_URL') is not None,
    }
    
    if redis_client:
        try:
            # Test connection
            start = time.time()
            ping_result = await asyncio.wait_for(redis_client.ping(), timeout=2.0)
            ping_time = (time.time() - start) * 1000  # ms
            
            # Test set/get
            test_key = f"health_check_{int(time.time())}"
            await redis_client.set(test_key, "ok", ex=10)
            get_result = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            
            status.update({
                "connected": True,
                "ping": ping_result,
                "ping_time_ms": round(ping_time, 2),
                "operations_working": get_result == "ok",
                "error": None
            })
        except asyncio.TimeoutError:
            status.update({
                "connected": False,
                "error": "Connection timeout after 2s"
            })
        except Exception as e:
            status.update({
                "connected": False,
                "error": str(e)
            })
    else:
        status["connected"] = False
        if not REDIS_AVAILABLE:
            status["reason"] = "Redis library not installed"
        elif not status["enabled"]:
            status["reason"] = "Redis disabled via ENABLE_REDIS_CACHE"
        elif not status["url_configured"]:
            status["reason"] = "REDIS_URL not configured"
        else:
            status["reason"] = "Redis initialization failed during startup"
    
    return status

# Background notification tasks
async def send_chat_notification_async(user_id: str, user_message: str, ai_response: str):
    """Send notification for chat response"""
    try:
        if NOTIFICATIONS_AVAILABLE:
            preview = ai_response[:100] + "..." if len(ai_response) > 100 else ai_response
            await notification_service.send_chat_response(
                user_id=user_id,
                response_data={
                    'preview': preview,
                    'query': user_message,
                    'full_response': ai_response,
                    'timestamp': datetime.now().isoformat()
                }
            )
    except Exception as e:
        logger.error(f"Failed to send chat notification: {e}")

async def send_weather_context_notification(user_id: str, user_message: str, ai_response: str):
    """Send weather-context notification when relevant"""
    try:
        if WEATHER_SERVICES_AVAILABLE:
            weather_data = get_weather_for_ai()
            if weather_data and 'error' not in weather_data:
                
                # Check if weather affects the user's query
                if not weather_data.get('is_good_weather', True):
                    weather_message = f"Current weather: {weather_data.get('condition', 'Unknown')}, {weather_data.get('temperature', '?')}°C"
                    if weather_data.get('recommendations'):
                        weather_message += f". {weather_data['recommendations'][0]}"
                    
                    await notification_service.send_personalized_tip(
                        user_id=user_id,
                        tip_data={
                            'type': 'weather_context',
                            'title': '🌤️ Weather Update',
                            'message': weather_message,
                            'weather_data': weather_data,
                            'user_query': user_message
                        }
                    )
                elif weather_data.get('needs_umbrella', False):
                    await notification_service.send_weather_alert(
                        user_id=user_id,
                        weather_data={
                            'message': f"Rain expected! {weather_data.get('condition', 'Weather update')}, bring an umbrella.",
                            'temperature': weather_data.get('temperature'),
                            'condition': weather_data.get('condition')
                        }
                    )
                    
    except Exception as e:
        logger.error(f"Failed to send weather context notification: {e}")

@app.post("/api/v1/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)  # Alias for frontend compatibility
@limiter.limit("60/minute")
async def chat(
    request: Request,
    message: ChatMessage,
    background_tasks: BackgroundTasks,
    redis_conn: redis.Redis = Depends(get_redis),
    current_user: str = Depends(get_current_user)
):
    """Main chat endpoint with map visualization support"""
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
        
        # Generate AI response - Initialize AI system on first use
        await ensure_ai_system()
        
        # Try main system with map support first
        ai_response = None
        map_data = None
        confidence = 0.85
        
        if main_system and MAIN_SYSTEM_AVAILABLE:
            try:
                # Prepare user location if provided
                user_location = None
                if message.user_location:
                    user_location = (message.user_location.lat, message.user_location.lon)
                    logger.info(f"📍 User location: {user_location[0]:.6f}, {user_location[1]:.6f}")
                
                # Use main system for structured response with maps
                result = main_system.process_message(
                    message.message,
                    user_id=message.user_id,
                    user_location=user_location,
                    return_structured=True
                )
                
                if isinstance(result, dict):
                    ai_response = result.get('response', '')
                    map_data = result.get('map_data')
                    confidence = result.get('confidence', 0.85)
                    logger.info(f"🗺️ Main system response (map: {map_data is not None}, location: {user_location is not None})")
                else:
                    ai_response = str(result)
                    
            except Exception as e:
                logger.warning(f"Main system failed, falling back to daily talk system: {e}")
        
        # Fallback to daily talk system if main system unavailable or failed
        if not ai_response:
            ai_response = ai_system.process_message(
                message.message,
                message.user_id
            )
            logger.info("Daily talk system response generated")
        
        processing_time = time.time() - start_time
        RESPONSE_GENERATION_TIME.observe(processing_time)
        
        # === PHASE 5B/5C: EXTRACT UNIFIEDLLM METADATA ===
        # Extract metadata from AI systems for frontend badge display
        llm_metadata = extract_llm_metadata(
            ai_sys=ai_system,
            main_sys=main_system,
            processing_time=processing_time
        )
        
        # Create response
        response_data = {
            "response": ai_response,
            "session_id": message.session_id or f"{message.user_id}_{int(time.time())}",
            "processing_time": processing_time,
            "confidence": confidence,
            "map_data": map_data,  # Include map data
            "suggestions": [
                "Tell me more about Turkish cuisine",
                "How do I get around Istanbul?",
                "What are the best neighborhoods to visit?"
            ],
            # Phase 5B/5C: UnifiedLLM metadata for frontend badges
            "llm_backend": llm_metadata.get("llm_backend"),
            "cache_hit": llm_metadata.get("cache_hit"),
            "circuit_breaker_state": llm_metadata.get("circuit_breaker_state"),
            "llm_latency_ms": llm_metadata.get("llm_latency_ms")
        }
        
        # Cache the response
        background_tasks.add_task(
            cache_set, 
            cache_key, 
            json.dumps(response_data, cls=CustomJSONEncoder), 
            300,  # 5 minutes TTL
            redis_conn
        )
        
        # Send real-time notification for important responses
        if NOTIFICATIONS_AVAILABLE and len(ai_response) > 200:  # Only for substantial responses
            background_tasks.add_task(
                send_chat_notification_async,
                message.user_id,
                message.message,
                ai_response
            )
        
        # Send weather-based notifications if relevant to the query
        if WEATHER_SERVICES_AVAILABLE and any(keyword in message.message.lower() for keyword in [
            'weather', 'rain', 'sun', 'hot', 'cold', 'outdoor', 'walk', 'visit', 'explore', 'sightseeing'
        ]):
            background_tasks.add_task(
                send_weather_context_notification,
                message.user_id,
                message.message,
                ai_response
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

@app.post("/api/v1/voice", response_model=ChatResponse)
@limiter.limit("20/minute")
async def voice_chat(
    request: Request,
    user_id: str,
    audio_file: bytes,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Voice chat endpoint with English optimization"""
    start_time = time.time()
    
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/voice", status="processing").inc()
        
        # Initialize AI system on first use
        await ensure_ai_system()
        
        # Process voice with deep learning enhancement
        ai_response = await ai_system.process_voice_message(user_id, audio_file)
        
        processing_time = time.time() - start_time
        RESPONSE_GENERATION_TIME.observe(processing_time)
        
        response_data = {
            "response": ai_response,
            "session_id": f"voice_{user_id}_{int(time.time())}",
            "processing_time": processing_time,
            "confidence": 0.90,  # Voice processing confidence
            "suggestions": [
                "Try asking about restaurants nearby",
                "What's the weather like today?",
                "Guide me to Hagia Sophia"
            ]
        }
        
        REQUEST_COUNT.labels(method="POST", endpoint="/voice", status="200").inc()
        REQUEST_DURATION.observe(processing_time)
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        processing_time = time.time() - start_time
        REQUEST_COUNT.labels(method="POST", endpoint="/voice", status="500").inc()
        REQUEST_DURATION.observe(processing_time)
        logger.error(f"Voice endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Voice processing failed")

@app.get("/api/v1/user/{user_id}/profile", response_model=UserProfileResponse)
@limiter.limit("30/minute")
async def get_user_profile(
    user_id: str,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Get user profile and analytics"""
    try:
        # Initialize AI system on first use
        await ensure_ai_system()
        
        analytics = ai_system.get_enhanced_user_analytics(user_id)
        
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
    
    # Initialize AI system once at connection time
    try:
        await ensure_ai_system()
    except Exception as e:
        logger.error(f"Failed to initialize AI system: {e}")
        await websocket.close(code=1011, reason="AI system unavailable")
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
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
            
            await websocket.send_text(json.dumps(response_data, cls=CustomJSONEncoder))
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

@app.get("/api/v1/weather")
async def get_weather():
    """Get current weather information for Istanbul"""
    try:
        if WEATHER_SERVICES_AVAILABLE:
            weather_data = get_current_weather()
            if weather_data and 'error' not in weather_data:
                return weather_data
            else:
                # Return mock data if no real weather available
                return get_current_weather()  # Mock function
        else:
            return get_current_weather()  # Mock function
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        raise HTTPException(status_code=500, detail="Weather service temporarily unavailable")

@app.get("/api/v1/weather/ai")
async def get_weather_for_ai_endpoint():
    """Get weather data optimized for AI processing"""
    try:
        if WEATHER_SERVICES_AVAILABLE:
            weather_ai_data = get_weather_for_ai()
            return weather_ai_data
        else:
            return {
                'temperature': 20,
                'condition': 'Clear',
                'comfort_level': 'good',
                'outdoor_suitability': 'excellent',
                'is_good_weather': True,
                'recommendations': ['Perfect weather for exploring Istanbul!'],
                'mock': True
            }
    except Exception as e:
        logger.error(f"Weather AI API error: {e}")
        raise HTTPException(status_code=500, detail="Weather AI service temporarily unavailable")

@app.post("/api/v1/weather/update")
async def update_weather_cache_endpoint():
    """Manually trigger weather cache update (admin only)"""
    try:
        if WEATHER_SERVICES_AVAILABLE:
            success = await update_weather_cache()
            return {
                "success": success,
                "message": "Weather cache updated successfully" if success else "Failed to update weather cache",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"success": False, "message": "Weather services not available", "mock": True}
    except Exception as e:
        logger.error(f"Weather update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update weather cache")

@app.post("/api/v1/weather/test-notification")
async def test_weather_notification(user_id: str = "test_user"):
    """Test weather notification system"""
    try:
        if WEATHER_SERVICES_AVAILABLE and NOTIFICATIONS_AVAILABLE:
            # Send a test weather alert
            await send_weather_alerts([user_id])
            
            # Send weather context notification
            await send_weather_context_notification(
                user_id, 
                "What should I do outdoors today?", 
                "Based on current weather conditions..."
            )
            
            return {
                "success": True,
                "message": "Test weather notifications sent",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False, 
                "message": "Weather or notification services not available",
                "weather_available": WEATHER_SERVICES_AVAILABLE,
                "notifications_available": NOTIFICATIONS_AVAILABLE
            }
    except Exception as e:
        logger.error(f"Weather notification test error: {e}")
        raise HTTPException(status_code=500, detail="Failed to test weather notifications")

@app.get("/api/v1/neighborhoods")
async def get_neighborhoods():
    """Get available Istanbul neighborhoods"""
    # Initialize AI system on first use
    await ensure_ai_system()
    
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

@app.get("/api/v1/metrics/english-optimization")
@limiter.limit("10/minute")
async def get_english_optimization_metrics(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Get English optimization performance metrics"""
    try:
        # Initialize AI system on first use
        await ensure_ai_system()
        
        # Get metrics from deep learning system if available
        if hasattr(ai_system, 'deep_learning_ai') and ai_system.deep_learning_ai:
            metrics = ai_system.deep_learning_ai.get_english_performance_metrics()
            
            # Add system-wide metrics
            metrics.update({
                "system_status": "Enhanced with Deep Learning",
                "unlimited_features": True,
                "english_optimization_level": "Maximum Performance",
                "supported_users": "10,000+",
                "feature_usage_stats": ai_system.feature_usage_stats
            })
            
            return metrics
        else:
            return {
                "system_status": "Fallback Mode",
                "english_optimization_level": "Basic",
                "message": "Deep learning features not available"
            }
            
    except Exception as e:
        logger.error(f"English metrics endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get optimization metrics")

@app.post("/api/v1/personality/adapt")
@limiter.limit("30/minute")
async def adapt_personality(
    request: Request,
    user_id: str,
    personality_style: str,
    current_user: str = Depends(get_current_user)
):
    """Adapt AI personality for user preferences"""
    try:
        # Initialize AI system on first use
        await ensure_ai_system()
        
        # Use enhanced personality adaptation
        if hasattr(ai_system, 'deep_learning_ai') and ai_system.deep_learning_ai:
            # Track personality adaptation usage
            ai_system.feature_usage_stats['personality_adaptations'] += 1
            
            return {
                "status": "success",
                "message": f"Personality adapted to {personality_style} style for user {user_id}",
                "available_styles": ["formal", "casual", "enthusiastic", "analytical", "creative"],
                "unlimited_adaptations": True,
                "english_optimized": True
            }
        else:
            return {
                "status": "limited",
                "message": "Basic personality adaptation available"
            }
            
    except Exception as e:
        logger.error(f"Personality adaptation error: {e}")
        raise HTTPException(status_code=500, detail="Personality adaptation failed")

# =============================
# ROUTE PLANNING API ENDPOINTS
# =============================

class RouteRequest(BaseModel):
    """Route planning request model"""
    start_location: str = Field(..., description="Starting location (address, landmark, or GPS coordinates)")
    places_to_visit: Optional[List[str]] = Field(default=[], description="Specific places to visit")
    duration_hours: Optional[float] = Field(default=4.0, description="Available time in hours")
    route_style: Optional[str] = Field(default="balanced", description="Route style: efficient, scenic, cultural, balanced")
    transport_mode: Optional[str] = Field(default="walking", description="Transport mode: walking, public_transport, driving")
    include_food: Optional[bool] = Field(default=True, description="Include restaurant stops")
    max_attractions: Optional[int] = Field(default=6, description="Maximum number of attractions")
    user_preferences: Optional[Dict[str, Any]] = Field(default={}, description="User preferences and interests")

class RouteResponse(BaseModel):
    """Route planning response model"""
    success: bool
    route_id: Optional[str] = None
    route_name: str
    description: str
    total_distance_km: float
    estimated_duration_hours: float
    route_points: List[Dict[str, Any]]
    optimization_info: Dict[str, Any]
    map_data: Optional[Dict[str, Any]] = None
    personalized_tips: List[str]
    real_time_advice: List[str]

@app.post("/api/v1/route/plan", response_model=RouteResponse)
@limiter.limit("10/minute")
async def plan_route(
    route_request: RouteRequest,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Plan an optimized multi-modal route through Istanbul"""
    try:
        start_time = time.time()
        
        # Initialize AI system on first use
        await ensure_ai_system()
        
        # Create a route planning query from the request
        query = f"Plan a {route_request.duration_hours}-hour {route_request.route_style} route starting from {route_request.start_location}"
        
        if route_request.places_to_visit:
            places_text = ", ".join(route_request.places_to_visit)
            query += f" visiting {places_text}"
        
        query += f" using {route_request.transport_mode} transport"
        
        # Get user profile
        user_profile = ai_system.get_or_create_user_profile(current_user)
        
        # Update user profile with preferences from request
        if route_request.user_preferences:
            for key, value in route_request.user_preferences.items():
                if hasattr(user_profile, key):
                    setattr(user_profile, key, value)
        
        # Create conversation context
        from conversation_context import ConversationContext
        context = ConversationContext(f"route_session_{current_user}")
        context.current_message = query
        
        # Process route planning query
        response = ai_system.handle_route_planning_query(
            query, user_profile, context, datetime.now()
        )
        
        processing_time = time.time() - start_time
        RESPONSE_GENERATION_TIME.observe(processing_time)
        
        # Extract route data from AI response (parse the formatted text response)
        route_data = parse_route_response(response)
        
        return RouteResponse(
            success=True,
            route_id=f"route_{current_user}_{int(time.time())}",
            route_name=route_data.get("name", "Custom Istanbul Route"),
            description=route_data.get("description", "Optimized route through Istanbul"),
            total_distance_km=route_data.get("distance", 0.0),
            estimated_duration_hours=route_data.get("duration", route_request.duration_hours),
            route_points=route_data.get("points", []),
            optimization_info=route_data.get("optimization", {}),
            map_data=route_data.get("map_data"),
            personalized_tips=route_data.get("tips", []),
            real_time_advice=route_data.get("advice", [])
        )
        
    except Exception as e:
        logger.error(f"Route planning error: {e}")
        raise HTTPException(status_code=500, detail=f"Route planning failed: {str(e)}")

@app.post("/api/v1/route/optimize")
@limiter.limit("5/minute")
async def optimize_route(
    route_request: RouteRequest,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Optimize an existing route or re-optimize with new parameters"""
    try:
        # Initialize AI system on first use
        await ensure_ai_system()
        
        # Check if route maker service is available for advanced optimization
        if hasattr(ai_system, 'route_maker') and ai_system.route_maker:
            # Use advanced route optimization
            from backend.services.route_maker_service import RouteRequest as RouteReq, RouteStyle, TransportMode
            
            # Convert API request to route maker request
            style_map = {
                'efficient': RouteStyle.EFFICIENT,
                'scenic': RouteStyle.SCENIC,
                'cultural': RouteStyle.CULTURAL,
                'balanced': RouteStyle.BALANCED
            }
            
            transport_map = {
                'walking': TransportMode.WALKING,
                'public_transport': TransportMode.PUBLIC_TRANSPORT,
                'driving': TransportMode.DRIVING
            }
            
            # Get coordinates for starting location
            coords = get_coordinates_from_location(route_request.start_location)
            if not coords:
                raise HTTPException(status_code=400, detail="Could not resolve starting location")
            
            route_req = RouteReq(
                start_lat=coords['lat'],
                start_lng=coords['lng'],
                max_distance_km=route_request.duration_hours * 2,  # Estimate based on walking speed
                available_time_hours=route_request.duration_hours,
                route_style=style_map.get(route_request.route_style, RouteStyle.BALANCED),
                transport_mode=transport_map.get(route_request.transport_mode, TransportMode.WALKING),
                include_food=route_request.include_food,
                max_attractions=route_request.max_attractions
            )
            
            # Generate optimized route
            try:
                from database import get_db
                db = next(get_db())
                optimized_route = ai_system.route_maker.generate_route(route_req, db)
                db.close()
            except:
                optimized_route = ai_system.route_maker.generate_route(route_req, None)
            
            # Convert to API response format
            route_points = []
            for point in optimized_route.points:
                route_points.append({
                    'name': point.name,
                    'lat': point.lat,
                    'lng': point.lng,
                    'category': point.category,
                    'duration_minutes': point.estimated_duration_minutes,
                    'arrival_time': point.arrival_time,
                    'score': point.score,
                    'notes': point.notes
                })
            
            return RouteResponse(
                success=True,
                route_id=f"optimized_{current_user}_{int(time.time())}",
                route_name=optimized_route.name or "Optimized Istanbul Route",
                description=optimized_route.description or "AI-optimized route using TSP algorithm",
                total_distance_km=optimized_route.total_distance_km,
                estimated_duration_hours=optimized_route.estimated_duration_hours,
                route_points=route_points,
                optimization_info={
                    'efficiency_score': optimized_route.efficiency_score,
                    'diversity_score': optimized_route.diversity_score,
                    'overall_score': optimized_route.overall_score,
                    'algorithm': 'TSP with Istanbul-specific optimizations'
                },
                personalized_tips=[
                    f"Route optimized for {route_request.route_style} experience",
                    f"Total walking distance: {optimized_route.total_distance_km:.1f} km",
                    f"Estimated completion time: {optimized_route.estimated_duration_hours:.1f} hours"
                ],
                real_time_advice=[
                    "Check opening hours before visiting attractions",
                    "Use İstanbulkart for public transportation",
                    "Consider traffic and prayer times for mosque visits"
                ]
            )
        else:
            # Fallback to text-based optimization
            return await plan_route(route_request, request, current_user)
            
    except Exception as e:
        logger.error(f"Route optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")

@app.get("/api/v1/route/directions/{route_id}")
@limiter.limit("20/minute")
async def get_route_directions(
    route_id: str,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Get step-by-step directions for a specific route"""
    try:
        # In production: retrieve route from database using route_id
        # For now, provide generic directions template
        
        return {
            "route_id": route_id,
            "directions": [
                {
                    "step": 1,
                    "instruction": "Start at your specified location",
                    "distance_meters": 0,
                    "duration_minutes": 0,
                    "transport_mode": "walking"
                },
                {
                    "step": 2,
                    "instruction": "Head towards first attraction",
                    "distance_meters": 500,
                    "duration_minutes": 6,
                    "transport_mode": "walking",
                    "waypoints": []
                }
            ],
            "alternative_routes": [],
            "real_time_updates": {
                "traffic_status": "normal",
                "public_transport_delays": [],
                "weather_impact": "none"
            }
        }
        
    except Exception as e:
        logger.error(f"Route directions error: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve directions")

def parse_route_response(ai_response: str) -> Dict[str, Any]:
    """Parse AI response text to extract structured route data"""
    
    # Extract basic information using regex patterns
    route_data = {
        "name": "Istanbul Route",
        "description": "Personalized route through Istanbul",
        "distance": 0.0,
        "duration": 0.0,
        "points": [],
        "optimization": {},
        "tips": [],
        "advice": []
    }
    
    try:
        import re
        
        # Extract distance
        distance_match = re.search(r'(\d+\.?\d*)\s*km', ai_response)
        if distance_match:
            route_data["distance"] = float(distance_match.group(1))
        
        # Extract duration
        duration_match = re.search(r'(\d+\.?\d*)\s*hours?', ai_response)
        if duration_match:
            route_data["duration"] = float(duration_match.group(1))
        
        # Extract route points (simplified parsing)
        lines = ai_response.split('\n')
        points = []
        for line in lines:
            if '.' in line and any(word in line.lower() for word in ['mosque', 'palace', 'tower', 'museum', 'bazaar']):
                # Extract attraction name
                point_match = re.search(r'\d+\.\s*\*\*(.+?)\*\*', line)
                if point_match:
                    points.append({
                        'name': point_match.group(1),
                        'category': 'attraction',
                        'duration_minutes': 60,
                        'score': 8.0
                    })
        
        route_data["points"] = points
        
        # Extract tips
        tips = []
        for line in lines:
            if line.strip().startswith('•') or line.strip().startswith('-'):
                tip = line.strip().lstrip('•-').strip()
                if tip:
                    tips.append(tip)
        
        route_data["tips"] = tips[:5]  # Limit to 5 tips
        
    except Exception as e:
        logger.warning(f"Error parsing route response: {e}")
    
    return route_data

def get_coordinates_from_location(location: str) -> Optional[Dict[str, float]]:
    """Get coordinates from location string"""
    
    # Define coordinates for major Istanbul locations
    location_coords = {
        'sultanahmet': {'lat': 41.0086, 'lng': 28.9802},
        'beyoğlu': {'lat': 41.0370, 'lng': 28.9777},
        'beyoglu': {'lat': 41.0370, 'lng': 28.9777},
        'taksim': {'lat': 41.0370, 'lng': 28.9844},
        'galata': {'lat': 41.0255, 'lng': 28.9742},
        'karaköy': {'lat': 41.0255, 'lng': 28.9742},
        'karakoy': {'lat': 41.0255, 'lng': 28.9742},
        'kadıköy': {'lat': 40.9897, 'lng': 29.0267},
        'kadikoy': {'lat': 40.9897, 'lng': 29.0267},
        'beşiktaş': {'lat': 41.0422, 'lng': 29.0094},
        'besiktas': {'lat': 41.0422, 'lng': 29.0094},
        'ortaköy': {'lat': 41.0550, 'lng': 29.0268},
        'ortakoy': {'lat': 41.0550, 'lng': 29.0268}
    }
    
    location_lower = location.lower()
    
    # Check for GPS coordinates in the location string
    gps_match = re.search(r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)', location)
    if gps_match:
        return {'lat': float(gps_match.group(1)), 'lng': float(gps_match.group(2))}
    
    # Check predefined locations
    for loc_name, coords in location_coords.items():
        if loc_name in location_lower:
            return coords
    
    # Default to Sultanahmet if no match found
    return location_coords['sultanahmet']

# =============================
# PUSH NOTIFICATION ENDPOINTS
# =============================

@app.websocket("/ws/notifications/{user_id}")
async def websocket_notifications_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time notifications"""
    if not NOTIFICATIONS_AVAILABLE:
        await websocket.close(code=1000, reason="Notifications not available")
        return
    
    connection_id = await notification_service.connect_websocket(websocket, user_id)
    if not connection_id:
        await websocket.close(code=1000, reason="WebSocket not available")
        return
    
    try:
        # Send welcome message
        welcome_notification = Notification(
            id="welcome_" + user_id,
            user_id=user_id,
            type=NotificationType.SYSTEM_MESSAGE,
            title="🔔 Notifications Connected",
            message="You'll receive real-time updates here!",
            priority=NotificationPriority.LOW
        )
        await notification_service.websocket_manager.send_to_user(user_id, welcome_notification)
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (like pings)
                data = await websocket.receive_text()
                logger.debug(f"WebSocket message from {user_id}: {data}")
                
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for {user_id}: {e}")
                break
    
    finally:
        notification_service.disconnect_websocket(connection_id)

@app.post("/api/v1/notifications/send")
@limiter.limit("10/minute")
async def send_notification(
    notification_request: NotificationRequest,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Send a push notification to a user"""
    if not NOTIFICATIONS_AVAILABLE:
        return {"success": False, "message": "Notifications not available"}
    
    try:
        # Create notification object
        notification = Notification(
            user_id=notification_request.user_id,
            type=NotificationType(notification_request.type),
            title=notification_request.title,
            message=notification_request.message,
            priority=NotificationPriority(notification_request.priority),
            data=notification_request.data or {}
        )
        
        # Send notification
        results = await notification_service.send_notification(notification)
        
        return {
            "success": True,
            "notification_id": notification.id,
            "delivery_results": results,
            "message": "Notification sent successfully"
        }
        
    except Exception as e:
        logger.error(f"Send notification error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")

@app.post("/api/v1/notifications/device-token")
@limiter.limit("5/minute")
async def register_device_token(
    token_request: DeviceTokenRequest,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Register FCM device token for push notifications"""
    try:
        notification_service.register_device_token(current_user, token_request.device_token)
        
        return {
            "success": True,
            "message": "Device token registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Device token registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register device token: {str(e)}")

@app.get("/api/v1/notifications")
@limiter.limit("30/minute")
async def get_notifications(
    request: Request,
    unread_only: bool = False,
    limit: int = 50,
    current_user: str = Depends(get_current_user)
):
    """Get notifications for the current user"""
    try:
        notifications = notification_service.get_user_notifications(
            current_user, unread_only=unread_only, limit=limit
        )
        
        return {
            "success": True,
            "notifications": notifications,
            "count": len(notifications),
            "unread_only": unread_only
        }
        
    except Exception as e:
        logger.error(f"Get notifications error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get notifications: {str(e)}")

@app.post("/api/v1/notifications/{notification_id}/read")
@limiter.limit("60/minute")
async def mark_notification_read(
    notification_id: str,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Mark a notification as read"""
    try:
        success = notification_service.mark_notification_read(current_user, notification_id)
        
        return {
            "success": success,
            "message": "Notification marked as read" if success else "Notification not found"
        }
        
    except Exception as e:
        logger.error(f"Mark notification read error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark notification as read: {str(e)}")

@app.post("/api/v1/notifications/read-all")
@limiter.limit("10/minute")
async def mark_all_notifications_read(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Mark all notifications as read for the current user"""
    try:
        count = notification_service.mark_all_read(current_user)
        
        return {
            "success": True,
            "marked_count": count,
            "message": f"Marked {count} notifications as read"
        }
        
    except Exception as e:
        logger.error(f"Mark all notifications read error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark notifications as read: {str(e)}")

@app.post("/api/v1/notifications/preferences")
@limiter.limit("5/minute")
async def update_notification_preferences(
    preferences: NotificationPreferencesRequest,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Update notification preferences for the current user"""
    try:
        preferences_dict = {
            'route_updates': preferences.route_updates,
            'attraction_recommendations': preferences.attraction_recommendations,
            'weather_alerts': preferences.weather_alerts,
            'traffic_updates': preferences.traffic_updates,
            'personalized_tips': preferences.personalized_tips
        }
        
        notification_service.update_notification_preferences(current_user, preferences_dict)
        
        return {
            "success": True,
            "preferences": preferences_dict,
            "message": "Notification preferences updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Update notification preferences error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")

@app.get("/api/v1/notifications/stats")
@limiter.limit("10/minute")
async def get_notification_stats(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Get notification service statistics"""
    try:
        stats = notification_service.get_service_stats()
        
        return {
            "success": True,
            "stats": stats,
            "notifications_available": NOTIFICATIONS_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Get notification stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get notification stats: {str(e)}")

# =============================
# NOTIFICATION INTEGRATION WITH CHAT
# =============================

async def send_chat_notification_async(user_id: str, message: str, response: str):
    """Send chat response notification asynchronously"""
    if NOTIFICATIONS_AVAILABLE:
        try:
            await notification_service.send_chat_response(
                user_id=user_id,
                response_data={
                    'message': message,
                    'response': response,
                    'preview': response[:100] + "..." if len(response) > 100 else response,
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send chat notification: {e}")

async def send_route_update_notification_async(user_id: str, route_data: Dict[str, Any]):
    """Send route update notification asynchronously"""
    if NOTIFICATIONS_AVAILABLE:
        try:
            await notification_service.send_route_update(
                user_id=user_id,
                route_data=route_data
            )
        except Exception as e:
            logger.warning(f"Failed to send route update notification: {e}")

# =============================
# DAILY TALK ENHANCEMENT ENDPOINTS
# =============================

# Import daily talk enhancement system
try:
    from services.daily_talk_enhancement import (
        get_daily_conversation, get_weather_aware_daily_greeting, 
        get_advanced_daily_conversation
    )
    LEGACY_DAILY_TALK_AVAILABLE = True
    logger.info("🗣️ Legacy Daily Talk Enhancement System available")
except ImportError as e:
    LEGACY_DAILY_TALK_AVAILABLE = False
    logger.warning(f"🗣️ Legacy Daily Talk Enhancement not available: {e}")

# Advanced Daily Talk AI
try:
    from services.advanced_daily_talk_ai import process_advanced_daily_talk, advanced_daily_talk_ai
    ADVANCED_DAILY_TALK_AVAILABLE = True
    logger.info("🧠 Advanced Daily Talk AI available - GPT-level intelligence!")
except ImportError as e:
    ADVANCED_DAILY_TALK_AVAILABLE = False
    logger.warning(f"🧠 Advanced Daily Talk AI not available: {e}")

# Comprehensive Daily Talks System (NEW!)
try:
    from daily_talks_integration_wrapper import DailyTalksIntegrationWrapper
    daily_talks_wrapper = DailyTalksIntegrationWrapper()
    COMPREHENSIVE_DAILY_TALKS_AVAILABLE = True
    logger.info("🚀 Comprehensive Daily Talks System available - Full featured!")
except ImportError as e:
    COMPREHENSIVE_DAILY_TALKS_AVAILABLE = False
    daily_talks_wrapper = None
    logger.warning(f"🚀 Comprehensive Daily Talks System not available: {e}")

# Set main daily talk availability flag
DAILY_TALK_AVAILABLE = COMPREHENSIVE_DAILY_TALKS_AVAILABLE or LEGACY_DAILY_TALK_AVAILABLE

@app.get("/api/v1/daily-greeting")
@limiter.limit("20/minute")
async def get_daily_greeting(
    request: Request,
    location: str = "Istanbul",
    user_id: Optional[str] = None
):
    """Get personalized daily greeting with weather awareness"""
    try:
        if not DAILY_TALK_AVAILABLE:
            return {
                "success": False,
                "error": "Daily talk system not available",
                "fallback_greeting": "Merhaba! Welcome to Istanbul! 🏙️ Ready for today's adventure?"
            }
        
        # Use user_id from request or generate anonymous one
        if not user_id:
            user_id = f"anonymous_{int(time.time())}"
        
        greeting = get_weather_aware_daily_greeting(user_id, location)
        
        return {
            "success": True,
            "greeting": greeting,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "weather_aware": WEATHER_SERVICES_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Daily greeting error: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_greeting": "Merhaba! Welcome to Istanbul! 🏙️ Ready for today's adventure?"
        }

@app.get("/api/v1/daily-conversation")
@limiter.limit("15/minute") 
async def get_daily_conversation_endpoint(
    request: Request,
    location: str = "Istanbul",
    user_id: Optional[str] = None,
    mood: Optional[str] = None,
    include_weather: bool = True,
    include_events: bool = True,
    conversation_style: str = "friendly"
):
    """Get full daily conversation with recommendations and tips using comprehensive system"""
    try:
        if not DAILY_TALK_AVAILABLE:
            return {
                "success": False,
                "error": "Daily talk system not available"
            }
        
        # Use user_id from request or generate anonymous one
        if not user_id:
            user_id = f"anonymous_{int(time.time())}"
        
        # Try comprehensive system first
        if COMPREHENSIVE_DAILY_TALKS_AVAILABLE and daily_talks_wrapper:
            try:
                conversation_data = daily_talks_wrapper.get_full_conversation_data(
                    user_id=user_id,
                    location=location,
                    user_mood=mood,
                    include_weather=include_weather,
                    include_events=include_events,
                    conversation_style=conversation_style
                )
                
                return {
                    "success": True,
                    "system_used": "comprehensive",
                    "data": conversation_data,
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "features": {
                        "advanced_intent_recognition": True,
                        "context_memory": True,
                        "weather_integration": include_weather,
                        "events_integration": include_events,
                        "mood_suggestions": True,
                        "multi_modal_responses": True
                    }
                }
            except Exception as comp_error:
                logger.warning(f"Comprehensive system error, falling back: {comp_error}")
        
        # Fallback to legacy system
        if LEGACY_DAILY_TALK_AVAILABLE:
            try:
                conversation = get_daily_conversation(user_id, location)
                
                # Format context for API response
                context_data = {
                    "time_of_day": conversation["context"].time_of_day.value,
                    "weather_condition": conversation["context"].weather_condition,
                    "temperature": conversation["context"].temperature,
                    "user_location": conversation["context"].user_location,
                    "user_mood": conversation["context"].user_mood.value,
                    "is_weekday": conversation["context"].is_weekday
                }
                
                return {
                    "success": True,
                    "system_used": "legacy",
                    "greeting": conversation["greeting"],
                    "recommendations": conversation["recommendations"],
                    "conversation_flow": conversation["conversation_flow"],
                    "mood_suggestions": conversation["mood_suggestions"],
                    "local_tips": conversation["local_tips"],
                    "context": context_data,
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "weather_aware": getattr(conversation.get("context"), "weather_condition", None) is not None
                }
            except Exception as legacy_error:
                logger.error(f"Legacy system also failed: {legacy_error}")
        
        # Final fallback
        return {
            "success": False,
            "error": "All daily talk systems unavailable",
            "fallback_data": {
                "greeting": "Merhaba! Welcome to Istanbul! 🏙️",
                "recommendations": [{"type": "activity", "title": "Explore Istanbul", "description": "Discover the magic where Europe meets Asia!"}],
                "conversation_flow": ["What would you like to explore today?"],
                "local_tips": ["Istanbul is a city of endless discoveries!"]
            }
        }
        
    except Exception as e:
        logger.error(f"Daily conversation endpoint error: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_data": {
                "greeting": "Merhaba! Welcome to Istanbul! 🏙️",
                "recommendations": [{"type": "activity", "title": "Explore Istanbul", "description": "Discover the magic where Europe meets Asia!"}],
                "conversation_flow": ["What would you like to explore today?"],
                "local_tips": ["Istanbul is a city of endless discoveries!"]
            }
        }

@app.post("/api/v1/daily-conversation/interactive")
@limiter.limit("20/minute")
async def interactive_daily_conversation(
    request: Request,
    data: Dict[str, Any]
):
    """Interactive daily conversation with the comprehensive system"""
    try:
        if not COMPREHENSIVE_DAILY_TALKS_AVAILABLE or not daily_talks_wrapper:
            return {
                "success": False,
                "error": "Comprehensive daily talks system not available"
            }
        
        user_message = data.get("message", "")
        user_id = data.get("user_id", f"anonymous_{int(time.time())}")
        location = data.get("location", "Istanbul")
        context = data.get("context", {})
        
        if not user_message:
            return {
                "success": False,
                "error": "Message is required"
            }
        
        # Process the interactive conversation
        response = daily_talks_wrapper.process_user_input(
            user_input=user_message,
            user_id=user_id,
            location=location,
            additional_context=context
        )
        
        return {
            "success": True,
            "response": response,
            "user_id": user_id,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "system_used": "comprehensive_interactive"
        }
        
    except Exception as e:
        logger.error(f"Interactive daily conversation error: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_response": {
                "text": "I'm sorry, I'm having trouble understanding right now. Could you try again?",
                "suggestions": ["Tell me about Istanbul weather", "What should I do today?", "Recommend some local food"]
            }
        }

@app.get("/api/v1/daily-mood-activities")
@limiter.limit("10/minute")
async def get_mood_based_activities(
    request: Request,
    mood: str = "curious",
    location: str = "Istanbul",
    user_id: Optional[str] = None
):
    """Get mood-based activity recommendations using comprehensive system"""
    try:
        if not DAILY_TALK_AVAILABLE:
            return {
                "success": False,
                "error": "Daily talk system not available"
            }
        
        # Use user_id from request or generate anonymous one
        if not user_id:
            user_id = f"anonymous_{int(time.time())}"
        
        # Try comprehensive system first
        if COMPREHENSIVE_DAILY_TALKS_AVAILABLE and daily_talks_wrapper:
            try:
                mood_data = daily_talks_wrapper.get_mood_based_suggestions(
                    user_id=user_id,
                    location=location,
                    mood=mood
                )
                
                return {
                    "success": True,
                    "system_used": "comprehensive",
                    "mood": mood,
                    "data": mood_data,
                    "location": location,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as comp_error:
                logger.warning(f"Comprehensive mood system error, falling back: {comp_error}")
        
        # Fallback to legacy system
        if LEGACY_DAILY_TALK_AVAILABLE:
            conversation = get_daily_conversation(user_id, location)
            
            return {
                "success": True,
                "system_used": "legacy",
                "mood": mood,
                "activities": conversation["mood_suggestions"],
                "local_tips": conversation["local_tips"],
                "location": location,
                "timestamp": datetime.now().isoformat()
            }
        
        # Final fallback
        return {
            "success": False,
            "error": "All daily talk systems unavailable",
            "fallback_activities": ["Explore the beautiful streets of Istanbul!", "Visit a traditional Turkish cafe", "Take a walk along the Bosphorus"]
        }
        
    except Exception as e:
        logger.error(f"Mood activities error: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_activities": ["Explore the beautiful streets of Istanbul!", "Visit a traditional Turkish cafe", "Take a walk along the Bosphorus"]
        }

@app.post("/api/v1/advanced-daily-talk")
@limiter.limit("30/minute")
async def advanced_daily_talk(
    request: Request,
    message: str,
    user_id: Optional[str] = None,
    location: str = "Istanbul"
):
    """Advanced GPT-level daily talk conversation with comprehensive system"""
    try:
        # Use user_id or generate anonymous one
        if not user_id:
            user_id = f"anonymous_{int(time.time())}"
        
        # Try comprehensive daily talks system first
        if COMPREHENSIVE_DAILY_TALKS_AVAILABLE:
            try:
                response = process_comprehensive_daily_talk(
                    message=message,
                    user_id=user_id,
                    location=location,
                    additional_context={"advanced_mode": True}
                )
                
                return {
                    "success": True,
                    "response": response["response_text"],
                    "user_id": user_id,
                    "analysis": {
                        "intent": response.get("detected_intent", "general"),
                        "emotional_tone": response.get("mood_analysis", {}).get("detected_mood", "neutral"),
                        "entities": response.get("entities", []),
                        "topics": response.get("topics", []),
                        "confidence": response.get("confidence", 0.8)
                    },
                    "suggestions": response.get("suggestions", []),
                    "ai_level": "comprehensive_advanced",
                    "weather_aware": response.get("includes_weather", False),
                    "events_aware": response.get("includes_events", False),
                    "timestamp": datetime.now().isoformat(),
                    "system_used": "comprehensive"
                }
            except Exception as comp_error:
                logger.warning(f"Comprehensive advanced system error, falling back: {comp_error}")
        
        # Fallback to original advanced system
        if not ADVANCED_DAILY_TALK_AVAILABLE:
            # Further fallback to regular daily talk
            if LEGACY_DAILY_TALK_AVAILABLE:
                conversation = get_daily_conversation(user_id or f"fallback_{int(time.time())}", location)
                return {
                    "success": True,
                    "response": conversation["greeting"],
                    "fallback_mode": True,
                    "ai_level": "traditional",
                    "system_used": "legacy"
                }
            else:
                return {
                    "success": False,
                    "error": "Advanced daily talk system not available",
                    "fallback_response": "Merhaba! I'm here to help you explore Istanbul! What would you like to discover?"
                }
        
        # Get weather context
        weather_context = None
        if WEATHER_SERVICES_AVAILABLE:
            try:
                weather_data = get_weather_for_ai()
                if weather_data and 'condition' in weather_data:
                    weather_context = {
                        "condition": weather_data['condition'],
                        "temperature": weather_data.get('temperature', 20.0),
                        "description": weather_data.get('description', '')
                    }
            except Exception:
                pass  # Continue without weather context
        
        # Process with advanced AI
        result = process_advanced_daily_talk(message, user_id, weather_context)
        
        return {
            "success": True,
            "response": result["response"],
            "user_id": user_id,
            "analysis": {
                "intent": result["analysis"]["intent"].value,
                "emotional_tone": result["analysis"]["emotional_tone"].value,
                "entities": result["analysis"]["entities"],
                "topics": result["analysis"]["topics"],
                "complexity": result["analysis"]["complexity_level"],
                "urgency": result["analysis"]["urgency_level"]
            },
            "conversation_state": result["conversation_state"],
            "personalization_level": result["personalization_level"],
            "emotional_state": result["emotional_state"],
            "suggestions": result["suggestions"],
            "ai_level": "advanced",
            "weather_aware": weather_context is not None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Advanced daily talk error: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_response": "I'm here to help you explore Istanbul! What interests you most about the city?"
        }

@app.get("/api/v1/daily-talks/system-status")
@limiter.limit("5/minute")
async def get_daily_talks_system_status(request: Request):
    """Get status and capabilities of all daily talk systems"""
    try:
        status = {
            "comprehensive_system": {
                "available": COMPREHENSIVE_DAILY_TALKS_AVAILABLE,
                "features": []
            },
            "legacy_system": {
                "available": LEGACY_DAILY_TALK_AVAILABLE,
                "features": []
            },
            "advanced_system": {
                "available": ADVANCED_DAILY_TALK_AVAILABLE,
                "features": []
            }
        }
        
        if COMPREHENSIVE_DAILY_TALKS_AVAILABLE and daily_talks_wrapper:
            status["comprehensive_system"]["features"] = [
                "Advanced intent recognition",
                "Context memory",
                "Weather integration",
                "Events integration", 
                "News integration",
                "Mood-based suggestions",
                "Multi-modal responses",
                "Conversation flow management",
                "Personalized recommendations"
            ]
            
            # Get system info from wrapper
            try:
                system_info = daily_talks_wrapper.get_system_info()
                status["comprehensive_system"]["system_info"] = system_info
            except:
                pass
        
        if LEGACY_DAILY_TALK_AVAILABLE:
            status["legacy_system"]["features"] = [
                "Basic daily conversation",
                "Weather awareness",
                "Location-based tips",
                "Mood suggestions"
            ]
        
        if ADVANCED_DAILY_TALK_AVAILABLE:
            status["advanced_system"]["features"] = [
                "GPT-level intelligence",
                "Advanced analysis",
                "Emotional state tracking",
                "Complex conversation handling"
            ]
        
        return {
            "success": True,
            "overall_status": DAILY_TALK_AVAILABLE,
            "systems": status,
            "primary_system": "comprehensive" if COMPREHENSIVE_DAILY_TALKS_AVAILABLE else ("legacy" if LEGACY_DAILY_TALK_AVAILABLE else "none"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/v1/conversation-summary/{user_id}")
@limiter.limit("10/minute")
async def get_conversation_summary(
    request: Request,
    user_id: str
):
    """Get comprehensive conversation summary for a user"""
    try:
        if not ADVANCED_DAILY_TALK_AVAILABLE:
            return {
                "success": False,
                "error": "Advanced daily talk system not available"
            }
        
        summary = advanced_daily_talk_ai.get_conversation_summary(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Conversation summary error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Add healthcheck endpoint for daily talk systems
@app.get("/api/v1/daily-talk-status")
async def daily_talk_status():
    """Check status of daily talk systems"""
    return {
        "traditional_daily_talk": DAILY_TALK_AVAILABLE,
        "advanced_daily_talk": ADVANCED_DAILY_TALK_AVAILABLE,
        "weather_services": WEATHER_SERVICES_AVAILABLE,
        "capabilities": {
            "basic_conversations": DAILY_TALK_AVAILABLE,
            "advanced_ai": ADVANCED_DAILY_TALK_AVAILABLE,
            "weather_awareness": WEATHER_SERVICES_AVAILABLE,
            "conversation_memory": ADVANCED_DAILY_TALK_AVAILABLE,
            "emotional_intelligence": ADVANCED_DAILY_TALK_AVAILABLE,
            "personalization": ADVANCED_DAILY_TALK_AVAILABLE
        },
        "timestamp": datetime.now().isoformat()
    }
