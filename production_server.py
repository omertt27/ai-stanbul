"""
Istanbul AI - Production Server with Scalability Features
FastAPI server with async processing, caching, rate limiting
Designed to handle 10,000 monthly users (50+ concurrent peak)
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
from typing import Optional, Dict, Any
import asyncio

# Import core system (Updated to use unified root-level main_system.py)
from istanbul_ai.main_system import IstanbulDailyTalkAI

# Import scalability components
from istanbul_ai.core.async_orchestrator import get_orchestrator, AsyncOrchestrator
from istanbul_ai.core.cache_manager import get_cache_manager, CacheManager
from istanbul_ai.core.rate_limiter import get_rate_limiter, RateLimiter
from istanbul_ai.monitoring.performance_monitor import get_performance_monitor, PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Istanbul AI API",
    description="Scalable Istanbul Travel Assistant with GPU Acceleration",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
ai_system: Optional[IstanbulDailyTalkAI] = None
orchestrator: Optional[AsyncOrchestrator] = None
cache_manager: Optional[CacheManager] = None
rate_limiter: Optional[RateLimiter] = None
performance_monitor: Optional[PerformanceMonitor] = None


# Request/Response models
class ChatRequest(BaseModel):
    query: str
    user_id: str
    language: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    language: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    processing_time_ms: float
    from_cache: bool = False


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: int
    version: str
    gpu_available: bool
    cache_available: bool


class StatsResponse(BaseModel):
    requests_total: int
    requests_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    cache_hit_rate: float
    cpu_percent: float
    memory_percent: float
    gpu_percent: float


# Dependency injection
async def get_ai_system() -> IstanbulDailyTalkAI:
    """Get AI system instance"""
    global ai_system
    if ai_system is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    return ai_system


async def check_rate_limit(request: Request):
    """Check rate limiting (DISABLED FOR TESTING)"""
    # Rate limiting disabled for comprehensive testing
    return
    
    # Original rate limiting code (commented out for testing)
    # if rate_limiter is None:
    #     return  # Rate limiting disabled
    # 
    # # Get user ID from request (could be from auth, IP, etc.)
    # user_id = request.client.host if request.client else "unknown"
    # user_tier = "free"  # Get from auth/database
    # 
    # allowed, info = await rate_limiter.check_rate_limit(user_id, user_tier)
    # 
    # if not allowed:
    #     raise HTTPException(
    #         status_code=429,
    #         detail=f"Rate limit exceeded. Retry after {info['retry_after']} seconds",
    #         headers={"Retry-After": str(info['retry_after'])}
    #     )


# Middleware for performance monitoring
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Monitor all requests"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        success = True
    except Exception as e:
        logger.error(f"Request failed: {e}")
        success = False
        raise
    finally:
        latency = time.time() - start_time
        
        if performance_monitor:
            performance_monitor.record_request(latency, success)
    
    return response


# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize all systems on startup"""
    global ai_system, orchestrator, cache_manager, rate_limiter, performance_monitor
    
    logger.info("🚀 Starting Istanbul AI Production Server...")
    
    try:
        # Initialize AI system
        logger.info("Loading AI system...")
        ai_system = IstanbulDailyTalkAI()
        
        # Initialize async orchestrator
        logger.info("Initializing async orchestrator...")
        orchestrator = await get_orchestrator()
        
        # Initialize cache manager
        logger.info("Connecting to cache...")
        try:
            cache_manager = await get_cache_manager()
        except Exception as e:
            logger.warning(f"Cache not available: {e}")
            cache_manager = None
        
        # Initialize rate limiter
        logger.info("Starting rate limiter...")
        rate_limiter = await get_rate_limiter()
        
        # Initialize performance monitor
        logger.info("Starting performance monitor...")
        performance_monitor = await get_performance_monitor()
        
        logger.info("✅ Istanbul AI Production Server started successfully!")
        logger.info("📊 Ready to handle concurrent requests")
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Istanbul AI Production Server...")
    
    if orchestrator:
        await orchestrator.shutdown()
    
    if cache_manager:
        await cache_manager.disconnect()
    
    if rate_limiter:
        await rate_limiter.stop_cleanup()
    
    if performance_monitor:
        await performance_monitor.stop_monitoring()
    
    logger.info("✅ Shutdown complete")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Istanbul AI",
        "version": "2.0.0",
        "status": "online",
        "message": "Welcome to Istanbul AI - Your intelligent Istanbul travel guide"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import torch
    
    uptime = int(time.time() - performance_monitor.start_time) if performance_monitor else 0
    
    return HealthResponse(
        status="healthy" if ai_system else "degraded",
        uptime_seconds=uptime,
        version="2.0.0",
        gpu_available=torch.cuda.is_available(),
        cache_available=cache_manager is not None and cache_manager.redis is not None
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitor not available")
    
    stats = performance_monitor.get_stats()
    
    # Add cache stats if available
    if cache_manager:
        cache_stats = await cache_manager.get_stats()
        stats["cache_hit_rate"] = cache_stats.get("hit_rate", 0)
    else:
        stats["cache_hit_rate"] = 0
    
    return StatsResponse(**stats)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ai: IstanbulDailyTalkAI = Depends(get_ai_system),
    _: None = Depends(check_rate_limit)
):
    """
    Main chat endpoint
    Handles user queries with caching and async processing
    """
    start_time = time.time()
    from_cache = False
    
    try:
        # CACHE DISABLED FOR COMPREHENSIVE TESTING
        # Check cache first
        if False and cache_manager:  # Temporarily disabled
            cache_key = f"query:{request.query}:{request.user_id}"
            cached_response = await cache_manager.get(cache_key)
            
            if cached_response:
                from_cache = True
                processing_time = (time.time() - start_time) * 1000
                
                logger.info(f"✅ Cache HIT for user {request.user_id}")
                
                # Update processing time and from_cache flag
                cached_response["processing_time_ms"] = processing_time
                cached_response["from_cache"] = True
                
                return ChatResponse(**cached_response)
        
        # Process query
        logger.info(f"Processing query for user {request.user_id}: {request.query[:50]}...")
        
        # Use direct processing (orchestrator disabled for testing to get proper AI responses)
        result = await _process_query_direct(ai, request)
        
        # Original orchestrator code (temporarily disabled)
        # if orchestrator:
        #     context = request.context or {}
        #     result = await orchestrator.process_user_query(
        #         request.user_id,
        #         request.query,
        #         context
        #     )
        # else:
        #     # Fallback to direct processing
        #     result = await _process_query_direct(ai, request)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare response
        response_data = {
            "response": result.get("response", "I apologize, but I'm having trouble processing your request."),
            "language": result.get("language", "en"),
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "processing_time_ms": processing_time,
            "from_cache": False
        }
        
        # Cache the response (DISABLED FOR COMPREHENSIVE TESTING)
        if False and cache_manager:  # Temporarily disabled
            await cache_manager.set(
                f"query:{request.query}:{request.user_id}",
                response_data,
                ttl=1800  # 30 minutes
            )
        
        logger.info(f"✅ Query processed in {processing_time:.0f}ms")
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Error processing chat: {e}")
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response="I apologize, but I encountered an error processing your request. Please try again.",
            language=request.language or "en",
            processing_time_ms=processing_time,
            from_cache=False
        )


async def _process_query_direct(ai: IstanbulDailyTalkAI, 
                                request: ChatRequest) -> Dict[str, Any]:
    """Direct query processing - calls the AI system's main process_message method"""
    try:
        # Call the main AI system's process_message method
        # It handles all the profile, context, and processing internally
        response_text = ai.process_message(request.query, request.user_id)
        
        # Return the result in a standardized format
        return {
            "response": response_text if response_text else "I apologize, but I couldn't generate a response.",
            "language": request.language or "en",
            "intent": "processed"
        }
    except Exception as e:
        logger.error(f"Error in direct processing: {e}", exc_info=True)
        # Fallback response
        return {
            "response": "I apologize, but I'm having trouble processing your request right now.",
            "language": "en",
            "intent": "error"
        }


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    stats = await cache_manager.get_stats()
    return stats


@app.post("/api/cache/clear")
async def clear_cache(pattern: str = "*"):
    """Clear cache (admin only)"""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    await cache_manager.invalidate_pattern(pattern)
    return {"status": "success", "message": f"Cache cleared for pattern: {pattern}"}


@app.get("/api/rate-limit/{user_id}")
async def get_rate_limit_status(user_id: str):
    """Get rate limit status for user"""
    if not rate_limiter:
        raise HTTPException(status_code=503, detail="Rate limiter not available")
    
    stats = await rate_limiter.get_user_stats(user_id)
    return stats


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker with async for GPU
        log_level="info",
        access_log=True
    )
