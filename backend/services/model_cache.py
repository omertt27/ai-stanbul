"""
Centralized ML Model Cache

This module provides singleton access to expensive ML models.
All services should use this instead of creating their own model instances.

Benefits:
- Single model load per process (not per request)
- Shared memory for embeddings
- Consistent configuration
- Lazy initialization

Usage:
    from services.model_cache import get_sentence_transformer, get_model_cache
    
    # Get a specific model
    model = get_sentence_transformer('all-MiniLM-L6-v2')
    
    # Or use the cache directly
    cache = get_model_cache()
    model = cache.get_sentence_transformer('all-MiniLM-L6-v2')

Author: AI Istanbul Team
Date: December 2025
"""

import logging
import os
from typing import Dict, Optional, Any
from threading import Lock

# Suppress verbose HuggingFace/sentence-transformers logs
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

logger = logging.getLogger(__name__)

# Reduce noise from sentence-transformers
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


class ModelCache:
    """
    Centralized cache for expensive ML models.
    Implements singleton pattern with lazy initialization.
    """
    
    _instance = None
    _lock = Lock()
    
    # Supported models and their aliases
    MODEL_ALIASES = {
        'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
        'fast': 'all-MiniLM-L6-v2',
        'semantic': 'all-MiniLM-L6-v2',
        'default': 'all-MiniLM-L6-v2',
    }
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._sentence_transformers: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._models_lock = Lock()
        self._initialized = True
        
        logger.info("📦 ModelCache initialized")
    
    def get_sentence_transformer(self, model_name: str = 'default') -> Optional[Any]:
        """
        Get a SentenceTransformer model (cached).
        
        Args:
            model_name: Model name or alias ('multilingual', 'fast', 'default')
            
        Returns:
            SentenceTransformer model or None if loading failed
            
        NOTE: In production (Cloud Run), returns None to avoid HuggingFace downloads.
        The LLM on RunPod handles semantic understanding directly.
        """
        # In production, skip SentenceTransformer (requires HuggingFace downloads)
        is_production = os.environ.get('ENVIRONMENT', 'development').lower() == 'production'
        if is_production:
            logger.info(f"⏭️ Skipping SentenceTransformer in production (HuggingFace unavailable)")
            return None
        
        # Resolve alias
        actual_name = self.MODEL_ALIASES.get(model_name, model_name)
        
        # Check cache first
        if actual_name in self._sentence_transformers:
            return self._sentence_transformers[actual_name]
        
        # Load model (thread-safe)
        with self._models_lock:
            # Double-check after acquiring lock
            if actual_name in self._sentence_transformers:
                return self._sentence_transformers[actual_name]
            
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"🔄 Loading SentenceTransformer: {actual_name}")
                model = SentenceTransformer(actual_name)
                self._sentence_transformers[actual_name] = model
                logger.info(f"✅ SentenceTransformer loaded: {actual_name}")
                
                return model
                
            except ImportError:
                logger.warning("⚠️ sentence-transformers not installed")
                return None
            except Exception as e:
                logger.error(f"❌ Failed to load SentenceTransformer {actual_name}: {e}")
                return None
    
    def encode(self, text: str, model_name: str = 'default') -> Optional[Any]:
        """
        Encode text using a cached model.
        
        Args:
            text: Text to encode
            model_name: Model to use (default: 'default')
            
        Returns:
            Embedding vector or None
        """
        model = self.get_sentence_transformer(model_name)
        if model is None:
            return None
        
        try:
            return model.encode(text, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Encoding failed: {e}")
            return None
    
    def encode_batch(self, texts: list, model_name: str = 'default') -> Optional[Any]:
        """
        Encode multiple texts at once (more efficient).
        
        Args:
            texts: List of texts to encode
            model_name: Model to use
            
        Returns:
            List of embedding vectors or None
        """
        model = self.get_sentence_transformer(model_name)
        if model is None:
            return None
        
        try:
            return model.encode(texts, show_progress_bar=False, batch_size=32)
        except Exception as e:
            logger.warning(f"Batch encoding failed: {e}")
            return None
    
    def preload_models(self, model_names: list = None):
        """
        Preload models during startup for deterministic latency.
        
        Args:
            model_names: List of model names to preload (default: common ones)
        """
        if model_names is None:
            model_names = ['default', 'multilingual']
        
        logger.info(f"🔥 Preloading {len(model_names)} models...")
        
        for name in model_names:
            self.get_sentence_transformer(name)
        
        logger.info(f"✅ Preloaded {len(self._sentence_transformers)} models")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'loaded_sentence_transformers': list(self._sentence_transformers.keys()),
            'loaded_tokenizers': list(self._tokenizers.keys()),
            'total_models': len(self._sentence_transformers) + len(self._tokenizers),
        }
    
    def clear(self):
        """Clear all cached models (for testing)."""
        with self._models_lock:
            self._sentence_transformers.clear()
            self._tokenizers.clear()
        logger.info("🗑️ ModelCache cleared")


# Singleton accessor functions
_model_cache: Optional[ModelCache] = None
_cache_lock = Lock()

def get_model_cache() -> ModelCache:
    """Get the singleton ModelCache instance."""
    global _model_cache
    if _model_cache is None:
        with _cache_lock:
            if _model_cache is None:
                _model_cache = ModelCache()
    return _model_cache


def get_sentence_transformer(model_name: str = 'default') -> Optional[Any]:
    """
    Convenience function to get a SentenceTransformer model.
    
    Args:
        model_name: Model name or alias
        
    Returns:
        SentenceTransformer model or None
    """
    return get_model_cache().get_sentence_transformer(model_name)


def encode_text(text: str, model_name: str = 'default') -> Optional[Any]:
    """
    Convenience function to encode text.
    
    Args:
        text: Text to encode
        model_name: Model to use
        
    Returns:
        Embedding vector or None
    """
    return get_model_cache().encode(text, model_name)


def preload_models(model_names: list = None):
    """Preload models during startup."""
    get_model_cache().preload_models(model_names)
