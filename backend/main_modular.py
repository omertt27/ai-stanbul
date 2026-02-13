"""
Main Application Entry Point (Modular)

This is the new modular main.py with better organization:
- Configuration in config/settings.py
- Middleware in core/middleware.py
- Dependencies in core/dependencies.py
- Startup logic in core/startup.py
- Health endpoints in api/health.py
- Auth endpoints in api/auth.py
- Other routes in their respective modules

Author: AI Istanbul Team
Date: January 2025
"""

# =============================================================================
# CRITICAL: Set environment variables BEFORE any imports (in case run directly)
# =============================================================================
import os
import sys
import warnings

# Add parent directory to Python path for unified_system imports
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also add the current directory (for Docker deployments where unified_system is at same level)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Prevent tokenizer fork warnings (if not already set by main.py)
if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm progress bars
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
    warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# Regular imports
# =============================================================================
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Configure logging - PRODUCTION LEVEL (WARNING for optional systems)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# STARTUP GUARD - Prevent duplicate initialization
# =============================================================================
from core.startup_guard import (
    ensure_single_init, 
    is_initialized, 
    check_redis_once,
    is_redis_ready,
    log_init_status
)

# Reduce noise from verbose libraries
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('tqdm').setLevel(logging.CRITICAL)  # Silence tqdm completely

# Disable tqdm progress bars programmatically (backup for env var)
try:
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
except ImportError:
    pass

# Import configuration
from config.settings import settings

# Import core components
from core.middleware import setup_middleware
# Use fixed fast startup manager for Cloud Run
from core.startup_fixed import fast_startup_manager as startup_manager

# Import API routers directly to avoid circular imports
from api.health import router as health_router
from api.auth import router as auth_router
from api.chat import router as chat_router
# Legacy LLM testing router (uses old RunPod client, not needed for production)
# from api.llm import router as llm_router
from api.aws_diagnostics import router as aws_diagnostics_router
from api.monitoring_routes import router as monitoring_router
from api.admin import experiments as admin_experiments
from api.admin import routes as admin_routes
from api.startup_status import router as startup_status_router

# Import production recommendations API (NCF + A/B testing)
try:
    from api.production_recommendations import router as production_recs_router
    PRODUCTION_RECS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Production Recommendations API not available: {e}")
    PRODUCTION_RECS_AVAILABLE = False
    production_recs_router = None

# Import streaming API for real-time responses
try:
    from api.streaming import router as streaming_router
    STREAMING_API_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Streaming API not available: {e}")
    STREAMING_API_AVAILABLE = False
    streaming_router = None

# Import feedback API (for thumbs up/down on chat responses)
try:
    from api.feedback import router as feedback_router
    FEEDBACK_API_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Feedback API not available: {e}")
    FEEDBACK_API_AVAILABLE = False
    feedback_router = None

# Import blog API
try:
    from blog_api import router as blog_api_router
    BLOG_API_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Blog API not available: {e}")
    BLOG_API_AVAILABLE = False
    blog_api_router = None

# Import additional routes
try:
    from routes.cache_monitoring import router as cache_router
    from routes.intent_feedback import router as intent_router
    ADDITIONAL_ROUTES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Additional routes not available: {e}")
    ADDITIONAL_ROUTES_AVAILABLE = False
    cache_router = None
    intent_router = None

# Import legacy routes
try:
    from routes import museums, restaurants, places, blog
    LEGACY_ROUTES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Legacy routes not available: {e}")
    LEGACY_ROUTES_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan event handler.
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown")
    """
    # Startup - Check if already initialized (prevents duplicate init on reload)
    if not ensure_single_init("lifespan_startup"):
        logger.info("🔁 Lifespan startup already executed, skipping duplicate init")
        yield
        return
    
    logger.info("🚀 Starting application (non-blocking startup)")
    
    # Check Redis once at startup (cached for all components)
    redis_available = check_redis_once()
    logger.info(f"📡 Redis status: {'CONNECTED' if redis_available else 'NOT AVAILABLE'}")
    
    # Start initialization in background - don't wait
    asyncio.create_task(_background_initialization())
    
    logger.info("✅ Application ready to accept connections (background init in progress)")
    
    yield
    
    # Shutdown
    try:
        from services.runpod_warmer import stop_runpod_warming
        await stop_runpod_warming()
        logger.info("🔥 RunPod warmer stopped")
    except Exception as e:
        logger.warning(f"⚠️ Error stopping RunPod warmer: {e}")
        
    await startup_manager.shutdown()
    logger.info("👋 AI Istanbul Backend shut down complete")


async def _background_initialization():
    """Background initialization - runs after server starts"""
    # Guard against duplicate background init
    if not ensure_single_init("background_init"):
        logger.info("🔁 Background initialization already running/completed, skipping")
        return
    
    try:
        logger.info("🔄 Starting background initialization...")
        
        # Initialize startup manager components
        await startup_manager.initialize()
        
        # ===== UNIFIED LLM SERVICE INITIALIZATION =====
        # Initialize UnifiedLLMService singleton for API layer
        try:
            from unified_system.services.unified_llm_service import get_unified_llm
            
            unified_llm = get_unified_llm()
            app.state.unified_llm = unified_llm
            logger.info("✅ UnifiedLLMService initialized and ready for API layer")
            
            # Log configuration for visibility
            logger.info(f"📡 RunPod LLM endpoint: {os.getenv('LLM_API_URL', 'not configured')}")
            logger.info(f"🤖 Model: {os.getenv('LLM_MODEL_NAME', 'default')}")
            logger.info(f"💾 Cache: {'enabled' if unified_llm.cache_enabled else 'disabled'} ({len(unified_llm.cache)}/{unified_llm.cache_max_size} entries)")
            logger.info(f"🛡️  Circuit breaker: {'enabled' if unified_llm.circuit_breaker_enabled else 'disabled'} (threshold={unified_llm.circuit_breaker_threshold})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize UnifiedLLMService: {e}")
            app.state.unified_llm = None
            logger.warning("⚠️  API will fall back to legacy LLM clients")
        # ===== END UNIFIED LLM SERVICE INITIALIZATION =====
        
        # ===== RUNPOD WARMER INITIALIZATION =====
        # Start RunPod warmer to prevent cold starts
        try:
            from services.runpod_warmer import start_runpod_warming
            
            await start_runpod_warming()
            logger.info("🔥 RunPod warmer started - preventing cold starts")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to start RunPod warmer: {e}")
        # ===== END RUNPOD WARMER INITIALIZATION =====
        
        # Initialize admin experiments managers
        try:
            from api.admin.experiments import initialize_managers
            initialize_managers()
            logger.info("✅ Admin experiment managers initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize admin managers: {e}")
        
        logger.info("✅ Background initialization complete!")
        
    except Exception as e:
        logger.error(f"❌ Background initialization failed: {e}")


# Create FastAPI app with lifespan handler
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Setup middleware (CORS, logging, error handling)
setup_middleware(app)

# Mount static files for restaurant photos
STATIC_DIR = Path(__file__).parent / 'static'
RESTAURANT_PHOTOS_DIR = STATIC_DIR / 'restaurant_photos'

# Create directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
RESTAURANT_PHOTOS_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
logger.info(f"✅ Static files mounted at /static -> {STATIC_DIR}")

# =============================================================================
# ROOT HEALTH ENDPOINT (for simple health checks at /health)
# =============================================================================
@app.get("/health")
async def root_health():
    """Simple root health check for load balancers and monitoring."""
    from datetime import datetime
    return {
        "status": "ok",
        "service": "istanbul-ai-backend",
        "timestamp": datetime.now().isoformat()
    }

# Register API routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
# Legacy LLM testing router disabled (uses old RunPod client, not needed for production)
# app.include_router(llm_router)
app.include_router(aws_diagnostics_router)  # AWS S3 and Redis diagnostic endpoints
app.include_router(startup_status_router)  # Startup diagnostics
app.include_router(admin_experiments.router)

# Register RunPod warmer API for monitoring cold start prevention
try:
    from api.runpod_warmer_api import router as runpod_warmer_router
    app.include_router(runpod_warmer_router)
    logger.info("🔥 RunPod warmer API registered at /api/runpod")
except ImportError as e:
    logger.warning(f"⚠️ RunPod warmer API not available: {e}")

# Register streaming API for real-time chat responses
if STREAMING_API_AVAILABLE and streaming_router:
    app.include_router(streaming_router)
    logger.info("✅ Streaming API registered at /api/stream")

# Register production recommendations API (Phase 2: NCF + A/B testing)
if PRODUCTION_RECS_AVAILABLE and production_recs_router:
    app.include_router(production_recs_router)
    logger.info("✅ Production Recommendations API registered at /api/recommendations")

# Register feedback API for thumbs up/down on chat responses
if FEEDBACK_API_AVAILABLE and feedback_router:
    app.include_router(feedback_router)
    logger.info("✅ Feedback API registered at /api/feedback")

# Register direct routing API (fast, deterministic transportation routing)
try:
    from api.direct_routing import router as direct_routing_router
    app.include_router(direct_routing_router)
    logger.info("✅ Direct Routing API registered at /api/routes")
except Exception as e:
    logger.warning(f"⚠️ Direct Routing API not available: {e}")

logger.info(f"📋 Admin routes router has {len(admin_routes.router.routes)} routes")
app.include_router(admin_routes.router, prefix="/api/admin")
logger.info("✅ Admin routes registered at /api/admin")
app.include_router(monitoring_router)

# Register Blog API if available
if BLOG_API_AVAILABLE:
    app.include_router(blog_api_router)
    logger.info("✅ Blog API registered at /api/blog")

# Register additional routes if available
if ADDITIONAL_ROUTES_AVAILABLE:
    if cache_router:
        app.include_router(cache_router)
        logger.info("✅ Cache monitoring routes registered at /api/cache")
    if intent_router:
        app.include_router(intent_router)
        logger.info("✅ Intent feedback routes registered at /api/feedback")

# Register legacy routes if available
if LEGACY_ROUTES_AVAILABLE:
    app.include_router(museums.router)
    app.include_router(restaurants.router)
    app.include_router(places.router)
    app.include_router(blog.router)
    logger.info("✅ Legacy routes registered")

# =============================================================================
# ROOT HEALTH ENDPOINT (MUST BE AFTER ALL ROUTERS to ensure priority)
# =============================================================================
# Re-register the /health endpoint to ensure it's not overridden by other routers
@app.get("/health", tags=["Health"])
async def root_health_final():
    """Simple root health check for load balancers and monitoring."""
    from datetime import datetime
    return {
        "status": "ok",
        "service": "istanbul-ai-backend",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Istanbul AI Guide API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/health"
    }


# For backward compatibility, expose startup manager instances
def get_pure_llm_core():
    """Get Pure LLM Core instance"""
    return startup_manager.get_pure_llm_core()


def get_recommendation_engine():
    """Get recommendation engine instance"""
    return startup_manager.get_recommendation_engine()


# ====================================================================
# DO NOT RUN THIS FILE DIRECTLY!
# Use: uvicorn main_modular:app --host 0.0.0.0 --port 8001
# Or:  python run.py
# Running `python main_modular.py` causes DOUBLE app construction!
# ====================================================================
