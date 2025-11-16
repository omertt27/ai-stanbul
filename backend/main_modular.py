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
from fastapi import FastAPI

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
from core.startup import startup_manager

# Import API routers
from api import health, auth, chat, llm

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

# Register API routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(llm.router)

# Register legacy routes if available
if LEGACY_ROUTES_AVAILABLE:
    app.include_router(museums.router)
    app.include_router(restaurants.router)
    app.include_router(places.router)
    app.include_router(blog.router)
    logger.info("✅ Legacy routes registered")


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    await startup_manager.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("👋 Shutting down AI Istanbul Backend")


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
