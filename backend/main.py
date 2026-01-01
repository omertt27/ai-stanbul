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

# =============================================================================
# CRITICAL: Set environment variables BEFORE any imports
# This prevents fork warnings and reduces noise from HuggingFace/Torch
# =============================================================================
import os
import warnings

# Prevent tokenizer fork warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Reduce TensorFlow/Torch noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Only show errors from transformers

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning)

# Reduce sentence-transformers progress bar noise
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.cache/sentence_transformers')

# =============================================================================
# Now import the modular app
# =============================================================================
from backend.main_modular import app

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
