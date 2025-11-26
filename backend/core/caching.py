"""
Caching decorator for FastAPI endpoints
Provides easy-to-use caching with Redis fallback
"""

import functools
import hashlib
import json
import logging
from typing import Optional, Callable, Any
import inspect

logger = logging.getLogger(__name__)


def cache_response(
    ttl: int = 3600,
    key_prefix: str = "api",
    include_args: bool = True
):
    """
    Decorator to cache API responses
    
    Args:
        ttl: Time to live in seconds (default: 1 hour)
        key_prefix: Prefix for cache keys
        include_args: Whether to include function arguments in cache key
    
    Usage:
        @cache_response(ttl=300, key_prefix="chat")
        async def my_endpoint(request: ChatRequest):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Import here to avoid circular dependency
            from services.redis_cache import get_cache_service
            
            cache = get_cache_service()
            
            # Generate cache key
            if include_args:
                # Extract relevant data from request
                cache_key_parts = [key_prefix, func.__name__]
                
                # Handle Pydantic models
                for arg in args:
                    if hasattr(arg, 'model_dump'):
                        # Pydantic V2
                        cache_key_parts.append(json.dumps(arg.model_dump(), sort_keys=True))
                    elif hasattr(arg, 'dict'):
                        # Pydantic V1
                        cache_key_parts.append(json.dumps(arg.dict(), sort_keys=True))
                        
                for key, value in kwargs.items():
                    if hasattr(value, 'model_dump'):
                        cache_key_parts.append(f"{key}:{json.dumps(value.model_dump(), sort_keys=True)}")
                    elif hasattr(value, 'dict'):
                        cache_key_parts.append(f"{key}:{json.dumps(value.dict(), sort_keys=True)}")
                    else:
                        cache_key_parts.append(f"{key}:{value}")
                
                # Hash the key if it's too long
                key_string = ":".join(str(part) for part in cache_key_parts)
                if len(key_string) > 200:
                    key_hash = hashlib.md5(key_string.encode()).hexdigest()
                    cache_key = f"{key_prefix}:{func.__name__}:{key_hash}"
                else:
                    cache_key = key_string
            else:
                cache_key = f"{key_prefix}:{func.__name__}"
            
            # Try to get from cache
            try:
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"✅ Cache HIT: {cache_key}")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
            
            # Cache miss - execute function
            logger.debug(f"❌ Cache MISS: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            try:
                # Convert Pydantic models to dict for caching
                if hasattr(result, 'model_dump'):
                    cache_value = result.model_dump()
                elif hasattr(result, 'dict'):
                    cache_value = result.dict()
                else:
                    cache_value = result
                    
                await cache.set(cache_key, cache_value, ttl=ttl)
                logger.debug(f"✅ Cached result: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(key_pattern: str):
    """
    Invalidate cache entries matching a pattern
    
    Usage:
        await invalidate_cache("chat:*")
    """
    async def _invalidate():
        from services.redis_cache import get_cache_service
        cache = get_cache_service()
        await cache.clear_pattern(key_pattern)
        logger.info(f"Invalidated cache: {key_pattern}")
    
    return _invalidate()
