"""
Istanbul AI Guide - Main Application Entry Point

This is the production entry point that uses the modular architecture.
For the legacy monolithic version, see main_legacy.py.

The modular architecture provides:
- Better code organization and maintainability
- Easier testing and debugging
- Circuit breakers and resilience patterns
- Production-ready health checks
- Clean separation of concerns

Architecture:
- config/settings.py: Configuration management
- core/: Middleware, dependencies, startup logic
- api/: Modular API endpoints (health, auth, chat, llm)
- routes/: Legacy routes (museums, restaurants, places, blog)

Author: AI Istanbul Team
Date: January 2025
"""

# Import the modular app
from main_modular import app

# Export for uvicorn (main:app)
__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    from config.settings import settings
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=not settings.is_production()
    )
