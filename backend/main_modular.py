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

import logging
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
from api.llm import router as llm_router
from api.aws_diagnostics import router as aws_diagnostics_router
from api.monitoring_routes import router as monitoring_router
from api.admin import experiments as admin_experiments
from api.admin import routes as admin_routes
from api.startup_status import router as startup_status_router

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


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
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

# Register API routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(llm_router)
app.include_router(aws_diagnostics_router)  # AWS S3 and Redis diagnostic endpoints
app.include_router(startup_status_router)  # Startup diagnostics
app.include_router(admin_experiments.router)

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


@app.on_event("startup")
async def startup_event():
    """Application startup - NON-BLOCKING for Cloud Run"""
    logger.info("🚀 Starting application (non-blocking startup)")
    
    # Start initialization in background - don't wait
    asyncio.create_task(_background_initialization())
    
    logger.info("✅ Application ready to accept connections (background init in progress)")


async def _background_initialization():
    """Background initialization - runs after server starts"""
    try:
        logger.info("🔄 Starting background initialization...")
        
        # Initialize startup manager components
        await startup_manager.initialize()
        
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


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    await startup_manager.shutdown()
    logger.info("👋 AI Istanbul Backend shut down complete")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_modular:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=not settings.is_production()
    )
